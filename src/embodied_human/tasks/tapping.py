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
    """
    starts, targets, endpoints = (
        np.asarray(a, dtype=float) for a in (starts, targets, endpoints)
    )
    axis = targets - starts
    length = np.linalg.norm(axis, axis=1)
    unit = axis / length[:, None]
    # Signed overshoot along the task axis and projected movement distance.
    dx = np.einsum("ij,ij->i", endpoints - targets, unit)
    distance = length + dx
    we = 4.133 * float(np.std(dx, ddof=1))
    ae = float(np.mean(distance))
    ide = math.log2(ae / we + 1.0)
    mt = float(np.mean(times))
    misses = np.linalg.norm(endpoints - targets, axis=1) > width / 2
    return Throughput(mt, we, ae, ide, ide / mt, float(np.mean(misses)))
