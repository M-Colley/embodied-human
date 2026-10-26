"""Hearing, touch, multisensory responses, motion sickness, fatigue, risk,
trust and emotion."""

import importlib.util
import math

import numpy as np
import pytest

import embodied_human as eh
from embodied_human.affect import (
    AppraisalAffect,
    appraise,
    emotion_study_tasks,
)
from embodied_human.fatigue import ThreeCompartment, consumed_endurance
from embodied_human.hearing import (
    THIRD_OCTAVES,
    PsychoacousticHearing,
    a_weighting,
    cabin_noise_level,
    cabin_noise_spectrum,
    iso7731_audible,
)
from embodied_human.response import Signal, TwinResponse
from embodied_human.risk import DriverRiskField
from embodied_human.touch import VibrotactileTouch
from embodied_human.trust import DecisionFieldTrust
from embodied_human.vestibular import IsoDose, SubjectiveVertical, wf_weighting

AXES = ("fore_aft", "lateral", "vertical")


def test_hearing_thresholds_masking_and_latency():
    hearing = PsychoacousticHearing()
    assert hearing.quiet_threshold(1000) == pytest.approx(2.4)
    assert hearing.quiet_threshold(20) == pytest.approx(78.1)
    quiet = hearing.sensation_level(70, 1000)
    loud_cabin = hearing.sensation_level(70, 1000, noise_dba=70)
    assert loud_cabin < hearing.sensation_level(70, 1000, noise_dba=50) < quiet
    assert hearing.latency(-1) == math.inf
    assert hearing.latency(60) == pytest.approx(0.0439)
    assert hearing.latency(10) > hearing.latency(30) > hearing.latency(90)
    # Chocholle (1940) via Kohfeld et al. (1981): 69 ms from 10 to 60 dB.
    assert 1000 * (hearing.latency(10) - hearing.latency(60)) == pytest.approx(
        70, abs=3
    )


def test_cabin_noise_and_iso7731():
    level = cabin_noise_level(100)
    assert level == pytest.approx(64.6)
    bands = cabin_noise_spectrum(level)
    total = 10 * np.log10(np.sum(10 ** ((bands + a_weighting(THIRD_OCTAVES)) / 10)))
    assert total == pytest.approx(level, abs=1e-6)
    assert bands[THIRD_OCTAVES == 125][0] > bands[THIRD_OCTAVES == 2000][0] + 15
    assert a_weighting(1000) == pytest.approx(0.0, abs=0.01)
    assert iso7731_audible(80, 60) and not iso7731_audible(70, 60)
    assert not iso7731_audible(60, 30)


def test_loudness_with_mosqito():
    # Skip only without MoSQITo; a broken install must fail, not skip.
    if importlib.util.find_spec("mosqito") is None:
        pytest.skip("needs the experience extra (MoSQITo)")
    from embodied_human.hearing import loudness

    rate = 48000
    t = np.arange(0, 1, 1 / rate)
    tone = math.sqrt(2) * 2e-5 * 10 ** (60 / 20) * np.sin(2 * math.pi * 1000 * t)
    assert loudness(tone, rate) == pytest.approx(4.0, rel=0.05)  # 60 phon


def test_vibration_needs_contact_and_follows_thresholds():
    touch = VibrotactileTouch()
    assert touch.threshold(10, "seat_vertical") == pytest.approx(0.022)
    assert touch.threshold(250, "seat_lateral") > touch.threshold(10, "seat_lateral")
    assert touch.sensation_level(1.0, 60, "seat_vertical", contact=False) == -math.inf
    level = touch.sensation_level(0.022 * 10, 10, "seat_vertical")
    assert level == pytest.approx(20.0)
    # Morioka & Griffin (2008): the seat is most sensitive vertically at
    # 8-80 Hz, the hand vertically above 100 Hz.
    axes = [f"{s}_{a}" for s in ("hand", "seat", "foot") for a in AXES]
    assert min(axes, key=lambda s: touch.threshold(20, s)) == "seat_vertical"
    assert min(axes, key=lambda s: touch.threshold(125, s)) == "hand_vertical"
    # Morioka & Griffin (2005): the small fingertip contact is the most
    # sensitive at 8 Hz but the least at 125 Hz (no Pacinian summation).
    sites = ("fingertip", "palm", "grip")
    assert min(sites, key=lambda s: touch.threshold(8, s)) == "fingertip"
    assert max(sites, key=lambda s: touch.threshold(125, s)) == "fingertip"
    # A gripped steering wheel is most sensitive around 125 Hz (Pacinian).
    wheel = [touch.threshold(f, "steering_wheel") for f in (4, 16, 31.5, 125, 250)]
    assert min(wheel) == touch.threshold(125, "steering_wheel")
    assert touch.latency(30) == pytest.approx(0.1166)
    with pytest.raises(ValueError):
        touch.threshold(10, "elbow")


def test_twin_redundant_signals_and_eccentricity():
    response = TwinResponse()
    response.reset(np.random.default_rng(0))
    assert response.visual_latency(0) == pytest.approx(0.1026)
    assert response.visual_latency(10) == pytest.approx(0.1026 + 0.0225)
    assert response.visual_latency(120) == math.inf
    auditory = Signal("auditory", 0.0, 0.05)
    visual = Signal("visual", 0.0, 0.10)
    alone = response.mean_reaction([auditory], samples=20000)
    assert alone == pytest.approx(0.05 + 0.2171, abs=0.003)
    both = response.mean_reaction([auditory, visual], samples=20000)
    assert both < alone
    reaction = response.respond([Signal("visual", 0.0, math.inf)])
    assert reaction.time == math.inf and reaction.first is None


def test_vestibular_rest_wf_and_iso_dose():
    svc = SubjectiveVertical()
    rest = svc.simulate(0.1, np.zeros((600, 3)))
    assert np.max(np.abs(rest)) < 1e-9
    assert wf_weighting(0.167)[()] == pytest.approx(1.012, abs=0.001)
    assert wf_weighting(0.5)[()] == pytest.approx(0.2239, abs=0.001)
    dt, f, peak = 0.05, 0.167, 2.0
    t = np.arange(0, 3600, dt)
    motion = np.zeros((len(t), 3))
    motion[:, 2] = peak * np.sin(2 * math.pi * f * t)
    dose = IsoDose().simulate(dt, motion)[-1]
    expected = wf_weighting(f) * peak / math.sqrt(2) * math.sqrt(3600) / 3
    assert dose == pytest.approx(float(expected), rel=0.01)


def test_vertical_conflict_matches_kamiji_band_pass():
    # Kamiji et al. (2007), Eq. 11: without rotation or an internal model of
    # acceleration, the conflict is s / (tau s^2 + s + K_vc) times a.
    svc = SubjectiveVertical(accel_gain=0.0, accel_feedback=0.0)
    p, dt = svc.params, 0.02
    t = np.arange(0, 100, dt)
    frequencies = np.array([0.05, 0.2, 0.5])
    motion = np.zeros((len(frequencies), len(t), 3))
    motion[..., 2] = np.sin(2 * math.pi * frequencies[:, None] * t)
    amplitude = svc.conflict(dt, motion)[:, t > 80].max(axis=1)
    s = 2j * math.pi * frequencies
    expected = np.abs(s / (p.vertical_lag * s**2 + s + p.vertical_feedback))
    np.testing.assert_allclose(amplitude, expected, rtol=0.005)


def test_motion_sickness_grows_with_acceleration():
    t = np.arange(0, 1800, 0.1)
    motion = np.zeros((2, len(t), 3))
    motion[0, :, 2] = 1.0 * np.sin(2 * math.pi * 0.2 * t)
    motion[1, :, 2] = 3.0 * np.sin(2 * math.pi * 0.2 * t)
    msi = SubjectiveVertical().simulate(0.1, motion)
    assert 0 < msi[0, -1] < msi[1, -1] < 85


def test_three_compartment_endurance_and_recovery():
    fatigue = ThreeCompartment()
    times = [
        fatigue.endurance_time("shoulder", load, dt=0.01) for load in (0.2, 0.5, 0.8)
    ]
    # Reproduces the independent check of Frey-Law et al.'s rates.
    assert times == pytest.approx([275, 58, 14], rel=0.03)
    for _ in range(3000):
        fatigue.step(0.02, {"shoulder": 0.5})
    tired = fatigue.fatigued("shoulder")
    for _ in range(3000):
        fatigue.step(0.02, {"shoulder": 0.0})
    assert fatigue.fatigued("shoulder") < tired
    assert consumed_endurance(0.1, 60) == 0.0
    assert consumed_endurance(0.5, 60) > consumed_endurance(0.3, 60) > 0
    with pytest.raises(KeyError):
        fatigue.step(0.1, {"neck": 0.1})


def test_trust_inertia_and_hysteresis():
    trust = DecisionFieldTrust(noise_sd=0.0)
    trust.reset(np.random.default_rng(0))
    assert trust.preference == pytest.approx(0.3)
    modes = [trust.update(0.2, None) for _ in range(6)]
    assert modes[0] == "automation" and modes[-1] == "manual"
    low = trust.trust
    # Recovery needs a margin above zero preference: hysteresis.
    trust.update(0.75, 0.7)
    assert trust.mode == "manual" and trust.trust > low
    for _ in range(20):
        trust.update(1.0, 0.6)
    assert trust.mode == "automation"
    # Without information, beliefs drift back towards their initial values.
    trust.belief_automation = 0.2
    trust.update(None, None)
    assert trust.belief_automation == pytest.approx(0.2 + 0.8 / 50)


def test_risk_field_closed_form_and_lane_width():
    risk = DriverRiskField()
    width = 3.5

    def cost(x, y):
        return np.where(np.abs(y) > width / 2, 500.0, 0.0)

    assert risk.lane_risk(15.0, width) == pytest.approx(
        risk.perceived(cost, 15.0, spacing=0.05), rel=0.02
    )
    assert (
        risk.lane_risk(20.0, 3.0)
        > risk.lane_risk(20.0, 3.5)
        > risk.lane_risk(15.0, 3.5)
    )
    tuned = DriverRiskField(threshold=risk.lane_risk(20.0, 3.5))
    assert tuned.preferred_speed(3.5) == pytest.approx(20.0, abs=0.01)
    assert tuned.preferred_speed(3.0) < 20.0 < tuned.preferred_speed(4.0)
    # Turning left: the field peaks on the arc and spreads further outward
    # (k2 > k1 = 0), so the outer side carries more risk at equal distance.
    radius = risk.params.wheelbase / math.tan(0.05)
    arc = radius - math.sqrt(radius**2 - 25.0)

    def at(y):
        return risk.field(np.array([5.0]), np.array([y]), 10.0, 0.05)[0]

    assert at(arc) > max(at(arc + 0.5), at(arc - 0.5))
    assert at(arc - 1.0) > at(arc + 1.0)


def test_appraisal_formulas():
    a = appraise(0.4, [2.0, -1.0], {"G": 8, "E": 2}, "G")
    assert a.suddenness == pytest.approx(0.2)
    assert a.goal_relevance == pytest.approx(0.4)
    assert a.conduciveness == pytest.approx(0.7)
    assert a.power == pytest.approx(abs(2.0 - 0.5) / 2.0)
    b = appraise(-3.0, [0.1, -1.9], {"F": 5}, "F")
    assert (b.goal_relevance, b.conduciveness, b.suddenness) == (1.0, 0.0, 0.0)
    assert b.power == pytest.approx(1.0 / 1.9)


def test_boredom_task_reproduces_the_published_appraisal():
    learner = AppraisalAffect()
    learner.reset(np.random.default_rng(1))
    learner.train(emotion_study_tasks()["boredom"], episodes=3000)
    story = emotion_study_tasks(appraisal=True)["boredom"]
    (appraisal,) = learner.run_episode(
        story, forced={"S1": "a1"}, appraise_at={("S1", "F"): 0.1}
    )
    assert appraisal.vector() == pytest.approx([0.0, 0.0, 0.5, 0.526], abs=0.01)


def test_emotions_are_probabilities():
    pytest.importorskip("sklearn")
    from embodied_human.affect import Appraisal

    affect = AppraisalAffect()
    affect.reset(np.random.default_rng(0))
    happy = affect.emotions(Appraisal(0.1, 0.5, 0.95, 0.5))
    assert sum(happy.values()) == pytest.approx(1.0)
    assert max(happy, key=happy.get) == "happiness"
    irritated = affect.emotions(Appraisal(0.1, 0.5, 0.0, 0.5))
    assert max(irritated, key=irritated.get) == "irritation"


def test_takeover_request_is_perceived_by_the_senses():
    from embodied_human.agents.road import simulate_takeover

    log = simulate_takeover(eh.build_human(seed=3), signals=("tactile",), seed=3)
    assert log.reaction["first"] == "tactile"
    assert log.reaction["tactile"]["sensation_db"] > 20
    parts = log.takeover_components()
    assert 0.1 < parts["response_s"] < 1.0 < parts["engaged_s"]
    before, after = log.trust[0]
    assert after < before
    acceleration, angular, gravity = log.head_motion(0.02)
    assert acceleration.shape == angular.shape == gravity.shape == (len(log.times), 3)
    assert np.allclose(gravity[:, 2], 9.81)
    msi = log.motion_sickness(SubjectiveVertical(), 0.02)
    assert msi.shape == (len(log.times),) and msi.max() < 1.0
    silent = simulate_takeover(eh.build_human(seed=3), signals=(), seed=3)
    assert silent.reaction["first"] is None


def test_central_stage_waits_for_concurrent_tasks():
    from embodied_human.response import CentralTask, central_wait

    rng = np.random.default_rng(0)
    task = [CentralTask("chunk", 1.0, 0.2)]
    waits = [central_wait(task, rng) for _ in range(20000)]
    # A random arrival waits d^2 / (2 T) on average.
    assert np.mean(waits) == pytest.approx(0.02, abs=0.002)
    assert central_wait([], rng) == 0.0
    with pytest.raises(ValueError):
        CentralTask("bad", 1.0, 2.0)
    response = TwinResponse()
    response.reset(np.random.default_rng(1))
    signal = [Signal("auditory", 0.0, 0.05)]
    loaded = response.mean_reaction(signal, samples=8000, load=task)
    assert loaded == pytest.approx(0.05 + 0.2171 + 0.02, abs=0.004)


def test_drt_slows_with_driving_and_memory_load():
    from embodied_human.bench.load import drt_rt
    from embodied_human.tasks.load import DRIVING, n_back

    human = eh.build_human(seed=0)
    alone = drt_rt(human, ["head_light"], [], samples=4000)
    driving = drt_rt(human, ["head_light"], [DRIVING], samples=4000)
    two_back = drt_rt(human, ["head_light"], [DRIVING, n_back(2)], samples=4000)
    # Threaded cognition's driver: 0.2 s of every 0.7 s, about 29 ms.
    assert driving - alone == pytest.approx(28.6, abs=8)
    assert two_back - driving == pytest.approx(292, abs=25)


def test_drt_benchmark_measures_while_driving():
    from embodied_human.bench.senses import drt

    details = drt(eh.build_human(seed=0)).details
    # Both reference studies measured while driving.
    assert details["visual_ms"] - details["visual_alone_ms"] == pytest.approx(
        28.6, abs=8
    )
    assert details["on_road_older_ms"] > details["on_road_ms"]


def test_hands_return_in_parallel_and_driving_waits_for_assessment():
    from dataclasses import replace

    from embodied_human.agents.driver import DriverAgentParameters
    from embodied_human.agents.road import simulate_takeover

    knob = replace(DriverAgentParameters(), hand_release=0.0)
    laptop = replace(DriverAgentParameters(), hand_release=1.0)
    fast_log = simulate_takeover(eh.build_human(seed=2), seed=2, params=knob)
    slow_log = simulate_takeover(eh.build_human(seed=2), seed=2, params=laptop)
    fast, late = fast_log.takeover_components(), slow_log.takeover_components()
    assert late["hand_s"] - fast["hand_s"] == pytest.approx(1.0, abs=0.05)
    # The hand starts at the reaction, whether or not the eyes are on the
    # road yet; it then takes one Fitts-timed reach.
    reach = eh.build_human(seed=2).motor.movement_time(0.45, 0.1)  # type: ignore[attr-defined]
    assert fast["hand_s"] - fast["response_s"] == pytest.approx(reach, abs=0.03)
    # Driving waits for the hand and for the assessment of the road scene.
    for parts in (fast, late):
        assert parts["engaged_s"] >= max(parts["hand_s"], parts["eyes_s"])


def test_takeover_works_with_any_attention_module():
    from embodied_human.agents.road import simulate_takeover

    config = eh.HumanConfig().replace("attention", "seev")
    parts = simulate_takeover(eh.build_human(config, seed=0)).takeover_components()
    assert math.isfinite(parts["engaged_s"])


def test_takeover_waits_for_surt_and_logs_the_gaze_reaction():
    from embodied_human.agents.road import simulate_takeover

    logs = [
        simulate_takeover(eh.build_human(seed=s), speed=120 / 3.6, seed=s)
        for s in range(8)
    ]
    waits = [log.reaction["central_wait_s"] for log in logs]
    # SuRT holds the central stage 1.77 s of every 4.31 s: 0.36 s on average.
    assert 0.0 < np.mean(waits) < 1.0
    for log in logs:
        parts = log.takeover_components()
        if math.isfinite(parts["gaze_s"]):
            # The eyes leave the task before they reach the road.
            assert 0.0 <= parts["gaze_s"] < parts["eyes_s"]
    assert any(math.isfinite(log.takeover_components()["gaze_s"]) for log in logs)


def test_peripheral_increase_carries_over_to_saccades():
    from embodied_human.bench.senses import eccentricity

    result = eccentricity(eh.build_human(seed=0))
    assert result.passed
    observed = result.details["observed_increase_ms"]
    assert observed == {50.0: 39.0, 70.0: 58.0}


def test_lane_width_refit_predicts_the_other_widths():
    from embodied_human.bench.motivation import LANE_WIDTH_SPEED, fit_lane_widths
    from embodied_human.core import ModuleChoice

    tuned = fit_lane_widths(ModuleChoice("risk_field"))
    for width in (3.0, 3.5):
        assert 3.6 * tuned.preferred_speed(width) == pytest.approx(
            LANE_WIDTH_SPEED[width], abs=0.1
        )
    assert 3.6 * tuned.preferred_speed(2.85) < 3.6 * tuned.preferred_speed(3.75)


def test_trust_course_follows_events():
    from embodied_human.bench.motivation import kraus_events, trust_course

    events = kraus_events(1, "MF+")
    assert len(events) == 11 and events[5] == ["malfunction"] and events[3] == ["tor"]
    capability = {"ok": 0.9, "tor": 0.2, "malfunction": 0.1}
    course = trust_course(4.6, events, capability)
    assert len(course) == 12 and course[0] == 4.6
    assert course[4] < course[3] and course[6] < course[5]  # request, malfunction
    assert kraus_events(2, "MF-/LT") == [["ok"]] * 9
