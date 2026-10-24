"""Integrated benchmarks: the whole human driving the MyoSuite full body.

These need myosuite-carla (and MyoSuite with ``myofullbody``) and run on
its fitted vehicle surrogate, so they are reproducible offline. They are
skipped when the packages are missing.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..core import Human
from .benchmarks import Result, benchmark


def _driving_env(scenario: Any) -> Any:
    from myosuite_carla.driving_env import make_driving_env

    return make_driving_env(scenario)


def _available(result: Result) -> bool:
    try:
        import myosuite_carla  # noqa: F401
        from myosuite.core.model_builder import ModelBuilder  # noqa: F401
    except ImportError as error:
        result.skipped = f"needs myosuite-carla and MyoSuite: {error}"
        return False
    return True


@benchmark("driving.surt_glances")
def surt_glances(human: Human, seconds: float = 60.0, speed: float = 20.0) -> Result:
    """Drive a straight road at 20 m/s while doing SuRT.

    Off-road glances are scored against the NHTSA (2013) design criteria
    and against human glance data from simulator studies of in-vehicle
    visual search (Kujala and Saariluoma, 2011; Jokinen, Kujala and
    Oulasvirta, 2021). Their tasks differ from SuRT and ran at 40-120 km/h;
    Jokinen et al. found no significant effect of speed on glance duration.
    """
    result = Result(
        "driving.surt_glances",
        ("eyes", "objects", "attention", "estimator", "memory", "motor"),
    )
    if not _available(result):
        return result
    from myosuite_carla.scenario import Scenario

    from ..agents.driver import AttentiveDriver
    from ..tasks.surt import SuRTTask

    env = _driving_env(
        Scenario(name="surt", start_speed=speed, target_speed=speed, duration=seconds)
    )
    agent = AttentiveDriver(human, env, SuRTTask(rng=np.random.default_rng(0)))
    env.reset(seed=0)
    agent.reset(seed=0)
    offsets = []
    while True:
        _, _, terminated, truncated, info = env.step(agent.act())
        offsets.append(info["scenario"]["lane_offset_m"])
        if terminated or truncated:
            break
    metrics = agent.log.glance_metrics()
    result.score_against(
        "driving.mean_offroad_glance", metrics["mean_offroad_glance_s"]
    )
    result.score_against("driving.long_glance_share", metrics["long_glance_share"])
    result.score_against("driving.human_mean_glance", metrics["mean_offroad_glance_s"])
    result.score_against(
        "driving.human_long_glance_share", metrics["long_glance_share"]
    )
    assert agent.task is not None
    result.details = {
        **metrics,
        "lane_offset_sd_m": float(np.std(offsets)),
        "lane_offset_max_m": float(np.max(np.abs(offsets))),
        "terminated": bool(terminated),
        "surt": agent.task.result.summary(),
        "simulated_seconds": len(offsets) * env.dt,
    }
    return result


@benchmark("driving.takeover")
def takeover(human: Human, runs: int = 8) -> Result:
    """Take-over requests while doing SuRT during automation: the time to
    the first steering or pedal input, compared with Zhang et al. (2019).

    The details split each take-over into the mechanisms that produced it:
    the response latency, the hand's return to the wheel, the eyes' return
    to the road and the situation assessment before driving."""
    result = Result("driving.takeover", ("eyes", "attention", "estimator", "motor"))
    if not _available(result):
        return result
    from myosuite_carla.scenario import takeover_request

    from ..agents.driver import AttentiveDriver
    from ..tasks.surt import SuRTTask

    env = _driving_env(takeover_request())
    times = []
    components: list[dict[str, float]] = []
    for seed in range(runs):
        agent = AttentiveDriver(human, env, SuRTTask(rng=np.random.default_rng(seed)))
        env.reset(seed=seed)
        agent.reset(seed=seed)
        events: list[dict[str, Any]] = []
        while True:
            _, _, terminated, truncated, info = env.step(agent.act())
            events += info["events"]
            if terminated or truncated:
                break
        components.append(agent.log.takeover_components())
        takeovers = [e for e in events if e["type"] == "driver_takeover"]
        times.append(float(takeovers[0]["reaction_time"]) if takeovers else math.nan)
    valid = np.array([t for t in times if math.isfinite(t)])
    result.score_against(
        "driving.takeover_time", float(valid.mean()) if len(valid) else math.nan
    )
    result.details = {
        "takeover_times_s": times,
        "sd_s": float(valid.std(ddof=1)) if len(valid) > 1 else math.nan,
        "missed": int(len(times) - len(valid)),
        "median_components_s": {
            key: float(np.nanmedian([c.get(key, math.nan) for c in components]))
            for key in ("response_s", "hand_s", "eyes_s", "engaged_s")
        },
        "scenario": "myosuite-carla takeover_request: automation "
        "at 12 m/s, request at 10 s, 7 s budget",
    }
    return result
