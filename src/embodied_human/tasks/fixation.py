"""A continuously available visual target for self paced gaze experiments.

The target requires fixation and has no search, touch, or response stage.
Its location and size must come from the experimental apparatus. Angles are
degrees relative to looking straight along the road.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class FixationTask:
    """A fixed visual target with an explicitly supplied angular geometry."""

    target: tuple[float, float]
    half_extent: tuple[float, float] = (0.5, 0.5)

    def __post_init__(self) -> None:
        if (
            len(self.target) != 2
            or len(self.half_extent) != 2
            or not all(math.isfinite(v) for v in (*self.target, *self.half_extent))
            or any(v <= 0 for v in self.half_extent)
            or abs(self.target[0]) > 180
            or abs(self.target[1]) > 90
        ):
            raise ValueError("target and extent require finite valid gaze angles")
        if self.contains((0.0, 0.0)):
            raise ValueError("the fixation target must be separate from road gaze")

    def contains(self, gaze: tuple[float, float]) -> bool:
        """Whether a gaze lies within the measured target region."""
        return all(
            abs(value - centre) <= extent
            for value, centre, extent in zip(gaze, self.target, self.half_extent)
        )

    def reset(self, time: float = 0.0) -> None:
        """The visual target remains available throughout the episode."""
