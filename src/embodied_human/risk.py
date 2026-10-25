"""Perceived risk: why drivers choose the speed and position they do.

Baseline :class:`DriverRiskField` is the driver's risk field of Kolekar,
de Winter and Abbink (2020). The driver predicts the car's path as an arc
at the current steering angle and spreads a subjective probability of
being somewhere around it:

    z = a(s) exp(-(d - R)^2 / (2 sigma(s)^2)),
    a(s) = p (s - v t_la)^2,   sigma_i(s) = (m + k_i |delta|) s + c,

with ``s`` the distance along the predicted arc of radius ``R``, ``d`` a
point's distance from the arc's centre, ``k_1`` inside and ``k_2``
outside the turn. Perceived risk is the field weighted by the
consequence of each place (zero in the lane, high outside it); the driver
keeps it below a threshold, which sets speed and lane keeping.

The paper sums risk over grid cells but does not report the grid spacing,
so its threshold (3000) cannot be used directly. Here risk is the
integral over area (units of cost x m^2) and the threshold is fitted to
one human condition and checked on others (``design.lane_width_speed``).
Steering angles are in radians, the only reading for which the published
width growth (k_2 = 1.3823) stays plausible.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from .core import Module, Parameters, Source, register

_KOLEKAR = (
    "Kolekar, de Winter & Abbink (2020), Table 1",
    "https://doi.org/10.1038/s41467-020-18353-4",
)
Cost = Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]


@dataclass(frozen=True)
class RiskFieldParameters(Parameters):
    """Driver's risk field and controller threshold.

    Args:
        height_gain: p, steepness of the field height parabola.
        preview: t_la, look-ahead time, s.
        width_rate: m, width growth per metre ahead.
        inner_steer_rate: k_1, extra inner width growth per rad of steering.
        outer_steer_rate: k_2, extra outer width growth per rad of steering.
        width_at_car: c, field SD at the car, m (a quarter of its width).
        outside_cost: Consequence of leaving the lane.
        threshold: Highest acceptable perceived risk, cost x m^2.
        wheelbase: L, m.
    """

    height_gain: float = 0.0064
    preview: float = 3.5
    width_rate: float = 0.001
    inner_steer_rate: float = 0.0
    outer_steer_rate: float = 1.3823
    width_at_car: float = 0.5
    outside_cost: float = 500.0
    threshold: float = 1.0e5
    wheelbase: float = 2.7

    SOURCES: ClassVar = {
        "height_gain": Source(*_KOLEKAR, "fitted", "grid search on one driver"),
        "preview": Source(*_KOLEKAR, "fitted"),
        "width_rate": Source(*_KOLEKAR, "fitted"),
        "inner_steer_rate": Source(*_KOLEKAR, "fitted"),
        "outer_steer_rate": Source(
            *_KOLEKAR, "fitted", "unit not reported; radians assumed"
        ),
        "width_at_car": Source(*_KOLEKAR, "derived", "car width 2 m / 4"),
        "outside_cost": Source(
            "Kolekar, de Winter & Abbink (2020), Table 3",
            "https://doi.org/10.1038/s41467-020-18353-4",
            "assumed",
            "C_env, assumed by the authors",
        ),
        "threshold": Source(
            "model choice",
            None,
            "assumed",
            "placeholder; fitted per study (the published 3000 is a grid sum)",
        ),
        "wheelbase": Source(
            "model choice", None, "assumed", "not reported in the paper"
        ),
    }


class Risk(Module):
    """Category ``risk``: perceived risk and the speed it allows."""

    category = "risk"

    def lane_risk(self, speed: float, lane_width: float, offset: float = 0.0) -> float:
        """Perceived risk driving straight at ``speed`` in a lane."""
        raise NotImplementedError

    def preferred_speed(self, lane_width: float, offset: float = 0.0) -> float:
        """Highest speed whose perceived risk stays below the threshold."""
        raise NotImplementedError


@register("risk", "risk_field")
class DriverRiskField(Risk):
    """Kolekar et al.'s (2020) driver's risk field."""

    def __init__(self, **params: float) -> None:
        super().__init__()
        self.params = RiskFieldParameters(**params)

    # Field ------------------------------------------------------------------

    def height(self, s: NDArray[np.float64], speed: float) -> NDArray[np.float64]:
        """Field height along the predicted path; zero beyond the preview."""
        reach = speed * self.params.preview
        a = self.params.height_gain * (np.asarray(s) - reach) ** 2
        return np.where((np.asarray(s) >= 0) & (np.asarray(s) <= reach), a, 0.0)

    def width(
        self, s: NDArray[np.float64], steering: float, outer: bool
    ) -> NDArray[np.float64]:
        """Field SD at distance ``s`` on the inner or outer side, m."""
        p = self.params
        k = p.outer_steer_rate if outer else p.inner_steer_rate
        return (p.width_rate + k * abs(steering)) * np.asarray(s) + p.width_at_car

    def field(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        speed: float,
        steering: float = 0.0,
    ) -> NDArray[np.float64]:
        """Field height at points (x ahead, y left) in the car's frame."""
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        if abs(steering) < 1e-9:
            # Straight ahead: both sides widen at the same rate.
            sd = self.width(x, 0.0, True)
            return self.height(x, speed) * np.exp(-(y**2) / (2 * sd**2))
        radius = self.params.wheelbase / math.tan(abs(steering))
        side = math.copysign(1.0, steering)  # left turns have their centre at +y
        cx, cy = 0.0, side * radius
        distance = np.hypot(x - cx, y - cy)
        angle = np.arctan2(x - cx, side * (cy - y))
        s = radius * np.mod(angle, 2 * math.pi)
        outer = distance > radius
        sd = np.where(
            outer, self.width(s, steering, True), self.width(s, steering, False)
        )
        return self.height(s, speed) * np.exp(-((distance - radius) ** 2) / (2 * sd**2))

    def perceived(
        self,
        cost: Cost,
        speed: float,
        steering: float = 0.0,
        spacing: float = 0.05,
        lateral_extent: float = 15.0,
    ) -> float:
        """Integral of field x cost over a grid ahead of the car."""
        reach = speed * self.params.preview
        xs = np.arange(0.0, reach + spacing, spacing)
        ys = np.arange(-lateral_extent, lateral_extent + spacing, spacing)
        x, y = np.meshgrid(xs, ys, indexing="ij")
        return float(
            np.sum(self.field(x, y, speed, steering) * cost(x, y)) * spacing**2
        )

    # Straight lanes (closed form across the lane) ------------------------------

    def lane_risk(self, speed: float, lane_width: float, offset: float = 0.0) -> float:
        from scipy.integrate import quad
        from scipy.special import erfc

        p = self.params
        reach = speed * p.preview
        left = lane_width / 2 - offset
        right = lane_width / 2 + offset

        def across(s: float) -> float:
            sd = p.width_rate * s + p.width_at_car
            # Gaussian mass beyond each lane edge, times the field's peak.
            tails = erfc(left / (math.sqrt(2) * sd)) + erfc(right / (math.sqrt(2) * sd))
            return float(sd * math.sqrt(math.pi / 2) * tails)

        value, _ = quad(
            lambda s: float(self.height(np.array(s), speed)) * across(s), 0.0, reach
        )
        return p.outside_cost * value

    def preferred_speed(self, lane_width: float, offset: float = 0.0) -> float:
        from scipy.optimize import brentq

        threshold = self.params.threshold

        def excess(v: float) -> float:
            return self.lane_risk(v, lane_width, offset) - threshold

        low, high = 0.5, 80.0
        if excess(low) > 0:
            return low
        if excess(high) < 0:
            return high
        return float(brentq(excess, low, high, xtol=1e-4))
