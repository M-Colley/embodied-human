import math

import numpy as np
import pytest

import embodied_human as eh
from embodied_human.attention import ROAD, SEEV, AttenD, UncertaintySampler
from embodied_human.estimation import LaneKalman
from embodied_human.memory import ActRMemory
from embodied_human.motor import FFittsMinJerk
from embodied_human.vision.eyes import MainSequenceEyes
from embodied_human.vision.foveation import FoveatedVision
from embodied_human.vision.objects import EccentricityNoise, Item


def run_saccade(eyes, target, time=0.0):
    eyes.command(target, time)
    while eyes.busy:
        time += 0.001
        eyes.step(time)
    return time


def test_registry_and_config():
    available = eh.available()
    assert available["eyes"] == ["main_sequence"]
    assert "seev" in available["attention"]
    config = eh.HumanConfig().replace("attention", "seev", value=2.0)
    human = eh.build_human(config, seed=1)
    assert isinstance(human.attention, SEEV) and human.attention.params.value == 2.0
    assert config.as_dict()["attention"] == {"name": "seev", "params": {"value": 2.0}}
    with pytest.raises(ValueError):
        eh.create("eyes", "nope")
    with pytest.raises(ValueError):
        eh.HumanConfig().replace("hands", "x")
    kinds = {row["kind"] for row in human.provenance()}
    assert {"measured", "fitted", "assumed"} <= kinds


def test_registering_a_new_module():
    @eh.register("memory", "perfect")
    class Perfect(ActRMemory):
        def retrieve(self, key, time):
            return super().retrieve(key, time)

    human = eh.build_human(eh.HumanConfig().replace("memory", "perfect"))
    assert isinstance(human.memory, Perfect)
    with pytest.raises(ValueError):
        eh.register("memory", "perfect")(Perfect)
    del eh.REGISTRY["memory"]["perfect"]


def test_main_sequence_durations_and_minimum_jerk_profile():
    eyes = MainSequenceEyes()
    eyes.reset(np.random.default_rng(0))
    run_saccade(eyes, (10.0, 0.0))
    saccade = eyes.saccades[-1]
    assert saccade.duration == pytest.approx(0.021 + 0.0022 * 10)
    assert saccade.peak_velocity() == pytest.approx(1.875 * 10 / saccade.duration)
    halfway = saccade.position(saccade.onset + saccade.duration / 2)
    assert halfway[0] == pytest.approx(5.0)
    assert eyes.gaze == (10.0, 0.0)


def test_saccades_suppress_vision_and_queue():
    eyes = MainSequenceEyes(latency_spread=0.0)
    eyes.reset(np.random.default_rng(0))
    eyes.command((20.0, 0.0), 0.0)
    assert not eyes.step(0.1).in_saccade  # still in latency
    state = eyes.step(0.2 + 0.01)
    assert state.in_saccade and 0 < state.gaze[0] < 20
    eyes.command((0.0, 0.0), 0.21)  # during flight: starts after landing
    end = eyes.saccades[-1].end
    eyes.step(end + 0.001)
    assert eyes.gaze == (20.0, 0.0)
    while eyes.busy:
        end += 0.001
        eyes.step(end)
    assert eyes.saccades[-1].onset >= eyes.saccades[-2].end
    eyes.command((0.2, 0.0), end)  # below the minimum amplitude
    assert not eyes.busy
    with pytest.raises(ValueError):
        eyes.step(0.0)


def test_foveation_blurs_the_periphery_only():
    vision = FoveatedVision()
    rng = np.random.default_rng(0)
    image = (rng.random((120, 200, 3)) * 255).astype(np.uint8)
    percept = vision.perceive(image, (40.0, 60.0), pixels_per_degree=10.0)
    original = image / 255.0
    fovea = np.abs(percept.image[55:65, 35:45] - original[55:65, 35:45]).mean()
    periphery = np.abs(percept.image[:, 180:] - original[:, 180:]).mean()
    assert fovea < 0.02 < periphery
    assert percept.cutoff[60, 40] == pytest.approx(math.log(64) / 0.106)
    assert vision.perceive(image, (40.0, 60.0), 10.0, suppressed=True).suppressed
    gray = vision.perceive(image[:, :, 0], (40.0, 60.0), 10.0)
    assert gray.image.shape == (120, 200)


def test_object_noise_scales_with_eccentricity():
    objects = EccentricityNoise(e2=2.0)
    objects.reset(np.random.default_rng(0))
    assert objects.sd(1.0, 4.0) == pytest.approx(3.0)
    items = {"near": Item((0.0, 0.0), 1.0, 0.1), "far": Item((100.0, 0.0), 1.0, 0.1)}
    observed = objects.observe(items, (0.0, 0.0))
    assert set(observed) == {"near"}  # beyond the field of view
    assert objects.observe(items, (0.0, 0.0), suppressed=True) == {}


def test_lane_uncertainty_grows_faster_at_higher_speed():
    def sd_after(speed, seconds=1.2):
        kalman = LaneKalman()
        for _ in range(int(seconds / 0.02)):
            kalman.predict(0.02, speed=speed)
        return kalman.lateral_sd

    assert sd_after(30.0) > sd_after(20.0) > sd_after(10.0)
    kalman = LaneKalman()
    for _ in range(60):
        kalman.predict(0.02, speed=20.0)
    before = kalman.lateral_sd
    kalman.update(np.array([0.3, 0.0]), np.array([0.05, 0.01]))
    assert kalman.lateral_sd < before and kalman.mean[0] == pytest.approx(0.3, abs=0.05)


def test_uncertainty_sampler_switches_on_thresholds():
    sampler = UncertaintySampler(return_sd=0.4, threshold_spread=0.0)
    demands = {"display": 1.0}
    assert sampler.select(0.0, ROAD, demands, 0.1) == ROAD  # minimum road dwell
    assert sampler.select(0.4, ROAD, demands, 0.1) == "display"
    assert sampler.reason == "certain" and sampler.threshold == 0.4
    assert sampler.select(0.5, "display", demands, 0.2) == "display"
    assert sampler.select(0.9, "display", demands, 0.45) == ROAD
    assert sampler.reason == "uncertainty"
    assert sampler.select(3.0, "display", demands, 0.1) == "display"
    assert sampler.select(4.6, "display", demands, 0.1) == ROAD
    assert sampler.reason == "maximum_glance"
    assert sampler.select(5.0, ROAD, demands, 0.1) == ROAD
    assert sampler.select(5.4, ROAD, demands, 0.1) == "display"
    assert sampler.select(5.5, "display", {}, 0.1) == ROAD
    assert sampler.reason == "task_done"
    assert sampler.select(6.0, ROAD, {}, 0.0) == ROAD


def test_threshold_spread_varies_glances_within_a_person():
    sampler = UncertaintySampler(return_sd=0.3, threshold_spread=0.5)
    sampler.reset(np.random.default_rng(0))
    thresholds = []
    for k in range(200):
        sampler.select(2.0 * k, ROAD, {"display": 1.0}, 0.0)
        sampler.select(2.0 * k + 0.5, ROAD, {"display": 1.0}, 0.0)
        thresholds.append(sampler.threshold)
        sampler.select(2.0 * k + 1.0, "display", {"display": 1.0}, 10.0)
    logs = np.log(thresholds)
    assert np.median(thresholds) == pytest.approx(0.3, rel=0.15)
    assert np.std(logs) == pytest.approx(0.5, rel=0.15)


def test_seev_probabilities_follow_value_and_effort():
    seev = SEEV({"road": (0, 0, 1, 1), "mirror": (0, 1, 1, 0.2)})
    probabilities = seev.probabilities()
    assert probabilities["road"] > probabilities["mirror"]
    assert sum(probabilities.values()) == pytest.approx(1.0)


def test_attend_buffer_matches_kircher_and_ahlstrom():
    attend = AttenD()
    for _ in range(100):
        attend.step(0.01, "display")  # 1 s away
    assert attend.buffer == pytest.approx(1.0)
    for _ in range(100):
        attend.step(0.01, "mirror")  # mirror: 1 s grace, then drains
    assert attend.buffer == pytest.approx(1.0, abs=0.011)
    for _ in range(30):
        attend.step(0.01, ROAD)  # refills after 0.1 s
    assert attend.buffer == pytest.approx(1.2, abs=0.011)
    for _ in range(300):
        attend.step(0.01, "display")
    assert attend.buffer == 0.0 and attend.distracted_time > 0


def test_act_r_memory_focus_decay_and_latency():
    memory = ActRMemory(noise=0.0)
    for i in range(6):
        memory.encode(f"k{i}", i, float(i))
    assert list(memory.focus) == ["k2", "k3", "k4", "k5"]
    assert memory.retrieve("k5", 6.0).latency == 0.0
    # Displaced: base level ln(age^-0.5) is below the threshold of 0.
    retrieval = memory.retrieve("k0", 6.0)
    assert not retrieval.success and retrieval.latency == pytest.approx(1.0)
    assert memory.base_level("k0", 6.0) == pytest.approx(-0.5 * math.log(6.0))
    for t in (6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7):
        memory.encode("k0", 0, t)
    for i in range(6, 10):
        memory.encode(f"k{i}", i, 6.8)
    retrieval = memory.retrieve("k0", 7.0)
    assert retrieval.success and retrieval.latency == pytest.approx(
        math.exp(-retrieval.activation)
    )


def test_fitts_timing_and_endpoint_spread():
    motor = FFittsMinJerk()
    motor.reset(np.random.default_rng(0))
    assert motor.movement_time(0.08, 0.01) == pytest.approx(
        0.1477 + 0.105 * math.log2(9)
    )
    ends = np.array(
        [
            motor.plan(np.zeros(3), np.array([0.08, 0, 0]), 0.01, 0.0).end
            for _ in range(4000)
        ]
    )
    expected = math.sqrt((0.01 / 4.133) ** 2 + 0.0015**2 / 2)
    assert np.std(ends[:, 0]) == pytest.approx(expected, rel=0.05)
    movement = motor.plan(np.zeros(3), np.array([0.1, 0, 0]), 0.02, 1.0)
    np.testing.assert_allclose(movement.position(1.0), np.zeros(3))
    np.testing.assert_allclose(movement.position(movement.finish), movement.end)


def test_spindle_rates_follow_prochazka_and_gorassini():
    mujoco = pytest.importorskip("mujoco")
    from embodied_human.proprioception import SpindleProprioception

    xml = """<mujoco><worldbody><body><joint name="j" type="hinge" axis="0 1 0"/>
    <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="1"/>
    <site name="a" pos="0.1 0 0.03"/></body><site name="b" pos="0 0 0.2"/></worldbody>
    <tendon><spatial name="t"><site site="a"/><site site="b"/></spatial></tendon>
    <actuator><muscle tendon="t" name="m"/></actuator></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    sense = SpindleProprioception(baseline=10.0, delay=0.02)
    sense.bind(model, data)
    stretched = sense.ia_rates(sense.reference + 0.001, np.array([0.002]))
    assert stretched[0] == pytest.approx(4.3 * 2.0**0.6 + 2.0 * 1.0 + 10.0)
    assert sense.ia_rates(sense.reference, np.array([-1.0]))[0] == 0.0
    samples = []
    for _ in range(20):
        mujoco.mj_step(model, data)
        samples.append(sense.sense(data))
    arrived = [s for s in samples if s is not None]
    assert arrived and all(s.time <= s.sensed_time + 0.02 + 1e-9 for s in arrived)
    assert samples[0] is None  # the first sample is still in transit
