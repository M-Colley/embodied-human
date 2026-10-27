"""Multi-directional tapping (ISO 9241-411) and Fitts' throughput.

Targets of width ``W`` lie on a circle of diameter ``A`` and are selected
in the alternating ISO order. Throughput follows Soukoreff and MacKenzie
(2004): endpoints are projected on each movement's task axis, the
effective width is ``We = 4.133 SD``, the effective amplitude ``Ae`` is the
mean projected movement distance, ``IDe = log2(Ae / We + 1)`` and
``TP = IDe / MT``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class TappingCondition:
    """One amplitude-width condition (metres) with ``targets`` circles."""

    amplitude: float
    width: float
    targets: int = 13

    @property
    def nominal_id(self) -> float:
        """Nominal index of difficulty, bits."""
        return math.log2(self.amplitude / self.width + 1.0)


def iso_order(count: int) -> list[int]:
    """Alternating selection order around the circle: 0, k, 1, k+1, ..."""
    if count < 3 or count % 2 == 0:
        raise ValueError("use an odd number of at least three targets")
    step = (count + 1) // 2
    return [(i * step) % count for i in range(count)]


def target_positions(condition: TappingCondition) -> NDArray[np.float64]:
    """Target centres (x right, y up), metres, around the origin."""
    angles = 2 * np.pi * np.arange(condition.targets) / condition.targets
    radius = condition.amplitude / 2
    return np.column_stack([radius * np.sin(angles), radius * np.cos(angles)])


@dataclass(frozen=True)
class Throughput:
    """Effective Fitts measures of one condition."""

    movement_time: float
    effective_width: float
    effective_amplitude: float
    effective_id: float
    throughput: float
    error_rate: float

    def as_dict(self) -> dict[str, float]:
        """Plain values for reports."""
        return dict(vars(self))


def throughput(
    starts: NDArray[np.float64],
    targets: NDArray[np.float64],
    endpoints: NDArray[np.float64],
    times: NDArray[np.float64],
    width: float,
) -> Throughput:
    """Compute effective measures from movements in surface coordinates.

    Args:
        starts: Where each movement started (the previous endpoint), N x 2.
        targets: Target centres, N x 2.
        endpoints: Selection points, N x 2.
        times: Movement times, s.
        width: Nominal target width, for the error rate.

    At least two finite movements with positive times, width and task-axis
    lengths are required. Zero endpoint spread gives no finite effective-width
    estimate; in that case ``effective_id`` and ``throughput`` are NaN. They
    are also NaN when the mean projected amplitude is non-positive. The other
    descriptive measures remain available for those degenerate observations.
    """
    starts, targets, endpoints = (
        np.asarray(a, dtype=float) for a in (starts, targets, endpoints)
    )
    times = np.asarray(times, dtype=float)
    if starts.ndim != 2 or starts.shape[1] != 2 or starts.shape[0] < 2:
        raise ValueError("starts must have shape (N, 2) with at least two movements")
    if targets.shape != starts.shape or endpoints.shape != starts.shape:
        raise ValueError(
            "starts, targets and endpoints must have the same (N, 2) shape"
        )
    if times.shape != (starts.shape[0],):
        raise ValueError("times must have shape (N,), one per movement")
    if not all(np.isfinite(a).all() for a in (starts, targets, endpoints, times)):
        raise ValueError("movement coordinates and times must be finite")
    if np.any(times <= 0) or not math.isfinite(width) or width <= 0:
        raise ValueError("movement times and target width must be positive and finite")
    with np.errstate(over="ignore", invalid="ignore"):
        axis = targets - starts
        length = np.linalg.norm(axis, axis=1)
    if not np.isfinite(length).all() or np.any(length <= 0):
        raise ValueError("each task axis must have a positive finite length")
    unit = axis / length[:, None]
    # Signed overshoot along the task axis and projected movement distance.
    with np.errstate(over="ignore", invalid="ignore"):
        dx = np.einsum("ij,ij->i", endpoints - targets, unit)
        distance = length + dx
        we = 4.133 * float(np.std(dx, ddof=1))
        ae = float(np.mean(distance))
        mt = float(np.mean(times))
        misses = np.linalg.norm(endpoints - targets, axis=1) > width / 2
    if not all(math.isfinite(value) for value in (we, ae, mt)):
        raise ValueError("measurements exceed the finite range of effective estimates")
    ide = math.log2(ae / we + 1.0) if we > 0 and ae > 0 else math.nan
    return Throughput(mt, we, ae, ide, ide / mt, float(np.mean(misses)))
