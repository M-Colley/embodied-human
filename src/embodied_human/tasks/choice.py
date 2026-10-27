"""Assumed two choice experiments driven by an explicit evidence drift.

This task compares seeded samples with their native PyDDM distributions.
It is a solver consistency demonstration, not a benchmark against human data.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..decision import Choice, PyDDMResponse


@dataclass(frozen=True)
class ChoiceCondition:
    drift: float
    deadline: float
    correct: Choice

    def __post_init__(self) -> None:
        if not math.isfinite(self.drift):
            raise ValueError("drift must be finite")
        if not math.isfinite(self.deadline) or self.deadline <= 0:
            raise ValueError("deadline must be positive and finite")
        if self.correct not in ("upper", "lower"):
            raise ValueError("correct must explicitly name an upper or lower boundary")


def run_choice_task(
    response: PyDDMResponse,
    conditions: Sequence[ChoiceCondition],
    *,
    trials: int = 1000,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Run the same number of independent trials at each explicit condition."""
    if isinstance(trials, bool) or not isinstance(trials, int) or trials <= 0:
        raise ValueError("trials must be a positive integer")
    if not conditions:
        raise ValueError("at least one choice condition is required")
    response.reset(np.random.default_rng(seed))
    rows = []
    for condition in conditions:
        distribution = response.solve_choice(condition.drift, condition.deadline)
        draws = response.sample_choices(distribution, trials)
        replied = [trial for trial in draws if not trial.timeout]
        rows.append(
            {
                "drift": condition.drift,
                "deadline_s": condition.deadline,
                "correct_boundary": condition.correct,
                "trials": trials,
                "accuracy": sum(t.choice == condition.correct for t in draws) / trials,
                "conditional_accuracy": (
                    sum(t.choice == condition.correct for t in replied) / len(replied)
                    if replied
                    else None
                ),
                "timeout_rate": 1.0 - len(replied) / trials,
                "mean_rt_s": (
                    float(np.mean([trial.time for trial in replied]))
                    if replied
                    else None
                ),
                "predicted_accuracy": distribution.accuracy(condition.correct),
                "predicted_timeout_rate": distribution.timeout_probability,
                "predicted_mean_rt_s": distribution.mean_rt,
                "solver_version": distribution.solver_version,
                "interpretation": "solver consistency, not human validation",
            }
        )
    return rows
