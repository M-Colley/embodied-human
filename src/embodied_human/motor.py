"""Motor control of an idealized fingertip: Fitts' law timing, minimum-jerk
paths and finger-touch endpoint noise.

For a target of width ``W`` at distance ``A`` the movement takes

    MT = a + b log2(A / W + 1),

with the 2-D finger-touch fit of Bi, Li and Zhai (2013). Endpoints scatter
with variance ``sigma_r^2 + sigma_a^2``: the relative part ``W / 4.133`` per
axis is what makes the target's effective width equal its nominal width
(Soukoreff and MacKenzie, 2004), and ``sigma_a`` is the finger's absolute
precision. The path is minimum jerk (Flash and Hogan, 1985). This is the
baseline for touch input; a musculoskeletal arm can replace it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from .core import Module, Parameters, Source, register

Vector = NDArray[np.float64]


@dataclass(frozen=True)
class PointingParameters(Parameters):
    """Finger pointing.

    Args:
        intercept: Fitts intercept a, s.
        slope: Fitts slope b, s/bit.
        absolute_sd: Finger absolute precision, m, as a bivariate SD (each
            surface axis gets ``absolute_sd / sqrt(2)``).
        relative_width_sd: Endpoint SD per unit target width per axis.
    """

    intercept: float = 0.1477
    slope: float = 0.1050
    absolute_sd: float = 0.0015
    relative_width_sd: float = 1.0 / 4.133

    SOURCES: ClassVar = {
        "intercept": Source(
            "Bi, Li & Zhai (2013), FFitts law 2-D",
            "https://doi.org/10.1145/2470654.2466180",
            "fitted",
        ),
        "slope": Source(
            "Bi, Li & Zhai (2013), FFitts law 2-D",
            "https://doi.org/10.1145/2470654.2466180",
            "fitted",
        ),
        "absolute_sd": Source(
            "Bi, Li & Zhai (2013)",
            "https://doi.org/10.1145/2470654.2466180",
            "measured",
            "1.5 mm in 2-D tasks, 0.94 mm in 1-D",
        ),
        "relative_width_sd": Source(
            "Soukoreff & MacKenzie (2004)",
            "https://doi.org/10.1016/j.ijhcs.2004.09.001",
            "derived",
            "effective width We = 4.133 SD",
        ),
    }


@dataclass(frozen=True)
class Movement:
    """A planned point-to-point movement."""

    start_time: float
    duration: float
    start: Vector
    end: Vector
    aim: Vector

    def position(self, time: float) -> Vector:
        """Minimum-jerk position at ``time``."""
        tau = min(1.0, max(0.0, (time - self.start_time) / self.duration))
        s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
        return self.start + s * (self.end - self.start)

    @property
    def finish(self) -> float:
        """End time, s."""
        return self.start_time + self.duration


class Motor(Module):
    """Category ``motor``: plan and execute movements to targets."""

    category = "motor"

    def plan(
        self,
        start: Vector,
        target: Vector,
        width: float,
        time: float,
        axes: tuple[Vector, Vector] | None = None,
    ) -> Movement:
        """Plan a movement from ``start`` to a target of ``width`` metres."""
        raise NotImplementedError


@register("motor", "ffitts_min_jerk")
class FFittsMinJerk(Motor):
    """Fitts-timed minimum-jerk reaches with finger-touch endpoint noise."""

    def __init__(self, **params: float) -> None:
        super().__init__()
        self.params = PointingParameters(**params)

    def movement_time(self, distance: float, width: float) -> float:
        """Fitts movement time for a nominal index of difficulty, s."""
        if width <= 0:
            raise ValueError("target width must be positive")
        index = math.log2(distance / width + 1.0)
        return self.params.intercept + self.params.slope * index

    def plan(
        self,
        start: Vector,
        target: Vector,
        width: float,
        time: float,
        axes: tuple[Vector, Vector] | None = None,
    ) -> Movement:
        start = np.asarray(start, dtype=float)
        target = np.asarray(target, dtype=float)
        distance = float(np.linalg.norm(target - start))
        p = self.params
        sd = math.sqrt((p.relative_width_sd * width) ** 2 + p.absolute_sd**2 / 2.0)
        if axes is None:
            axes = (np.eye(3)[0], np.eye(3)[1])
        u, v = (np.asarray(a, dtype=float) for a in axes)
        offset = sd * (self.rng.standard_normal() * u + self.rng.standard_normal() * v)
        return Movement(
            time, self.movement_time(distance, width), start, target + offset, target
        )
