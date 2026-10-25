"""A constant-speed straight road for fast attention and glance simulations.

When the eyes return to the road depends on the belief's uncertainty, and
the Kalman covariance depends only on speed and on when the road was seen,
not on where the vehicle actually is. Glance behaviour can therefore be
simulated without vehicle dynamics or the MyoSuite body, about two orders
of magnitude faster, which makes parameter inference practical. The same
:class:`~embodied_human.agents.driver.AttentiveDriver` then drives the full
body for validation.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np

from ..core import Human
from ..tasks.surt import SuRTTask
from .driver import AttentiveDriver, DriverAgentParameters, DrivingLog


class StraightRoad:
    """A vehicle cruising in its lane centre at a constant speed.

    Exposes the parts of myosuite-carla's ``FullBodyDrivingEnv`` that the
    attentive driver reads: ``dt``, ``state`` and ``task``. With
    ``request_at``, automation drives and requests a take-over at that
    time.
    """

    def __init__(
        self,
        speed: float = 20.0,
        dt: float = 0.02,
        duration: float = 60.0,
        request_at: float | None = None,
    ) -> None:
        self.speed = float(speed)
        self.dt = float(dt)
        self.duration = float(duration)
        self.request_at = request_at
        self.task = SimpleNamespace(near_distance=8.0, far_preview=1.0)
        self.time = 0.0

    @property
    def state(self) -> dict[str, float]:
        requested = self.request_at is not None and self.time >= self.request_at
        since = self.time - self.request_at if requested else 0.0  # type: ignore[operator]
        return {
            "speed_mps": self.speed,
            "target_speed_mps": self.speed,
            "lane_offset_m": 0.0,
            "heading_error_rad": 0.0,
            "near_angle_rad": 0.0,
            "far_angle_rad": 0.0,
            "yaw_rate_rad_s": 0.0,
            "automation": float(self.request_at is not None),
            "takeover_requested": float(requested),
            "time_since_request_s": since,
            "lead_present": 0.0,
            "gap_m": 100.0,
            "closing_speed_mps": 0.0,
        }

    def reset(self, seed: int | None = None) -> tuple[None, dict[str, Any]]:
        self.time = 0.0
        return None, {}

    def step(self, intent: Any) -> tuple[None, float, bool, bool, dict[str, Any]]:
        self.time += self.dt
        info = {
            "scenario": {"lane_offset_m": 0.0, "speed_mps": self.speed},
            "events": [],
        }
        return None, 0.0, False, self.time >= self.duration - 1e-9, info


class HoldLane:
    """A controller stand-in: the vehicle stays in its lane."""

    def reset(self) -> None:
        pass

    def act(self, state: Any) -> None:
        return None


def simulate_takeover(
    human: Human,
    *,
    signals: tuple[str, ...] = ("auditory", "visual"),
    speed: float = 12.0,
    request_at: float = 10.0,
    seconds: float = 14.0,
    seed: int = 0,
    params: DriverAgentParameters = DriverAgentParameters(),
) -> DrivingLog:
    """Run SuRT during automation with a take-over request; the log holds
    the reaction and the take-over components."""
    road = StraightRoad(speed=speed, duration=seconds, request_at=request_at)
    agent = AttentiveDriver(
        human,
        road,
        SuRTTask(rng=np.random.default_rng(seed)),
        replace(params, takeover_signals=signals),
        reflex=HoldLane(),
    )
    road.reset(seed)
    agent.reset(seed)
    while True:
        *_, truncated, _ = road.step(agent.act())
        if truncated:
            break
    return agent.log


def simulate_glances(
    human: Human,
    *,
    speed: float = 20.0,
    seconds: float = 60.0,
    seed: int = 0,
    params: DriverAgentParameters = DriverAgentParameters(),
) -> DrivingLog:
    """Run SuRT while cruising and return the driver's log."""
    road = StraightRoad(speed=speed, duration=seconds)
    agent = AttentiveDriver(
        human,
        road,
        SuRTTask(rng=np.random.default_rng(seed)),
        params,
        reflex=HoldLane(),
    )
    road.reset(seed)
    agent.reset(seed)
    while True:
        *_, truncated, _ = road.step(agent.act())
        if truncated:
            break
    return agent.log
