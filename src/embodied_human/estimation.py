"""State estimation: what the driver believes about the car on the road.

The baseline is a Kalman filter on lateral offset ``y`` (m) and heading
error ``psi`` (rad) relative to the lane, signed so that ``y`` grows at
``speed * psi`` (myosuite-carla: both to the right). The driver knows its
own speed and yaw rate (vestibular and proprioceptive cues); unseen road
curvature and disturbances enter as heading process noise. Without visual
updates the lateral variance grows roughly as ``q v^2 t^3 / 3``, so
off-road glances shorten with speed, as Senders et al. (1967) observed in
occlusion studies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from .core import Module, Parameters, Source, register


@dataclass(frozen=True)
class LaneParameters(Parameters):
    """Process noise and initial uncertainty of the lane estimate.

    Args:
        heading_noise: Heading process noise density, rad^2/s.
        lateral_noise: Lateral process noise density, m^2/s.
        initial_sd: Initial lateral and heading SD.
    """

    heading_noise: float = 7e-4
    lateral_noise: float = 1e-3
    initial_sd: tuple[float, float] = (0.1, 0.02)

    SOURCES: ClassVar = {
        "heading_noise": Source(
            "model choice",
            None,
            "assumed",
            "puts lateral SD at 0.4 m after 1.2 s without vision at "
            "20 m/s; fit with the glance benchmark",
        ),
        "lateral_noise": Source("model choice", None, "assumed"),
        "initial_sd": Source("model choice", None, "assumed"),
    }


class Estimator(Module):
    """Category ``estimator``: maintain a belief from noisy observations."""

    category = "estimator"

    mean: NDArray[np.float64]
    covariance: NDArray[np.float64]

    def predict(
        self,
        dt: float,
        speed: float = 0.0,
        yaw_rate: float = 0.0,
        lane_yaw_rate: float = 0.0,
    ) -> None:
        """Advance the belief by ``dt`` seconds with known speed and yaw rates."""
        raise NotImplementedError

    def update(self, values: NDArray[np.float64], sds: NDArray[np.float64]) -> None:
        """Fuse a direct, independent observation of the state."""
        raise NotImplementedError


@register("estimator", "lane_kalman")
class LaneKalman(Estimator):
    """Kalman filter on (lateral offset, heading error)."""

    def __init__(self, **params: object) -> None:
        super().__init__()
        self.params = LaneParameters(**params)  # type: ignore[arg-type]
        self.reset()

    def reset(
        self,
        rng: np.random.Generator | None = None,
        state: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        super().reset(rng)
        self.mean = np.asarray(state, dtype=float)
        sd = np.asarray(self.params.initial_sd, dtype=float)
        self.covariance = np.diag(sd**2)

    def predict(
        self,
        dt: float,
        speed: float = 0.0,
        yaw_rate: float = 0.0,
        lane_yaw_rate: float = 0.0,
    ) -> None:
        """Propagate with known speed and yaw rate (rad/s, left positive).

        ``lane_yaw_rate`` is the yaw rate the lane itself requires (speed
        times curvature) if the driver knows it, e.g. from preview.
        """
        f = np.array([[1.0, speed * dt], [0.0, 1.0]])
        self.mean = f @ self.mean + np.array([0.0, (yaw_rate - lane_yaw_rate) * dt])
        q = np.diag([self.params.lateral_noise * dt, self.params.heading_noise * dt])
        self.covariance = f @ self.covariance @ f.T + q

    def update(self, values: NDArray[np.float64], sds: NDArray[np.float64]) -> None:
        values = np.asarray(values, dtype=float)
        r = np.diag(np.asarray(sds, dtype=float) ** 2)
        s = self.covariance + r
        gain = self.covariance @ np.linalg.inv(s)
        self.mean = self.mean + gain @ (values - self.mean)
        self.covariance = (np.eye(2) - gain) @ self.covariance

    @property
    def lateral_sd(self) -> float:
        """Current lateral SD, m."""
        return float(np.sqrt(self.covariance[0, 0]))
