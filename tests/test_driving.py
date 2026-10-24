"""The attentive driver in myosuite-carla's full-body environment (offline)."""

import math

import numpy as np
import pytest

pytest.importorskip("myosuite_carla")
pytest.importorskip("myosuite.core.model_builder")

import embodied_human as eh  # noqa: E402
from embodied_human.agents.driver import AttentiveDriver, DrivingLog  # noqa: E402
from embodied_human.tasks.surt import SuRTTask  # noqa: E402


@pytest.fixture(scope="module")
def highway():
    from myosuite_carla.driving_env import make_driving_env
    from myosuite_carla.scenario import Scenario

    return make_driving_env(
        Scenario(name="highway", start_speed=20.0, target_speed=20.0, duration=15.0)
    )


def test_attentive_driver_shares_gaze_with_surt(highway):
    agent = AttentiveDriver(
        eh.build_human(seed=0), highway, SuRTTask(rng=np.random.default_rng(0))
    )
    highway.reset(seed=0)
    agent.reset(seed=0)
    while True:
        _, _, terminated, truncated, info = highway.step(agent.act())
        if terminated or truncated:
            break
    assert not terminated
    metrics = agent.log.glance_metrics()
    assert metrics["glances"] >= 4
    assert 0.5 < metrics["mean_offroad_glance_s"] < 2.0
    assert agent.task.result.summary()["screens"] >= 2
    assert {"road", "display"} <= set(agent.log.areas)


def test_driving_without_a_task_keeps_eyes_on_the_road(highway):
    agent = AttentiveDriver(eh.build_human(seed=1), highway, None)
    highway.reset(seed=1)
    agent.reset(seed=1)
    offsets = []
    for _ in range(300):
        _, _, terminated, truncated, info = highway.step(agent.act())
        offsets.append(info["scenario"]["lane_offset_m"])
    assert set(agent.log.areas) <= {"road", "transition"}
    assert max(abs(o) for o in offsets) < 0.5


def test_takeover_time_emerges_from_the_modules():
    from myosuite_carla.driving_env import make_driving_env
    from myosuite_carla.scenario import takeover_request

    env = make_driving_env(takeover_request(duration=14.0))
    agent = AttentiveDriver(
        eh.build_human(seed=2), env, SuRTTask(rng=np.random.default_rng(2))
    )
    env.reset(seed=2)
    agent.reset(seed=2)
    events = []
    while True:
        _, _, terminated, truncated, info = env.step(agent.act())
        events += info["events"]
        if terminated or truncated:
            break
    takeover = [e for e in events if e["type"] == "driver_takeover"]
    assert takeover and 0.8 < takeover[0]["reaction_time"] < 4.0
    assert agent.log.engaged_at is not None and agent.log.request_at is not None
    parts = agent.log.takeover_components()
    assert 0 < parts["response_s"] < parts["hand_s"]
    assert parts["eyes_s"] <= parts["engaged_s"]


def test_glance_log_metrics():
    log = DrivingLog()
    for i in range(300):
        t = i * 0.01
        log.times.append(t)
        log.areas.append("display" if 0.5 <= t < 1.5 or 2.0 <= t < 2.4 else "road")
    log.screen_times = [(0.0, 3.0)]
    metrics = log.glance_metrics()
    assert metrics["glances"] == 2
    assert metrics["mean_offroad_glance_s"] == pytest.approx(0.7, abs=0.02)
    assert metrics["mean_eyes_off_road_per_screen_s"] == pytest.approx(1.4, abs=0.03)
    assert metrics["long_glance_share"] == 0.0 and math.isnan(
        metrics["attend_minimum_s"]
    )
