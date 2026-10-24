"""Statistics for comparing simulated participants with human data.

Thin wrappers over SciPy, so benchmark reports state uncertainty
(bootstrap intervals over simulated participants), agreement in ranking
(Kendall's tau) and how an outcome scales with a condition (elasticity).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np


def bootstrap_ci(
    values: Sequence[float] | np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    *,
    level: float = 0.95,
    resamples: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    """BCa bootstrap interval of ``statistic`` over independent values.

    Returns NaNs with fewer than three finite values or no spread.
    """
    from scipy.stats import bootstrap

    data = np.asarray([v for v in values if math.isfinite(v)], dtype=float)
    if len(data) < 3 or np.ptp(data) == 0:
        return (math.nan, math.nan)
    result = bootstrap(
        (data,),
        lambda x, axis: np.apply_along_axis(statistic, axis, x),
        confidence_level=level,
        n_resamples=resamples,
        method="BCa",
        rng=np.random.default_rng(seed),
    )
    interval = result.confidence_interval
    return (float(interval.low), float(interval.high))


def kendall_tau(predicted: Sequence[float], observed: Sequence[float]) -> float:
    """Kendall's tau-b between two orderings of the same designs."""
    from scipy.stats import kendalltau

    return float(kendalltau(predicted, observed).statistic)


def elasticity(conditions: Sequence[float], outcomes: Sequence[float]) -> float:
    """Slope of log outcome on log condition: the percent change of the
    outcome per percent change of the condition."""
    x = np.log(np.asarray(conditions, dtype=float))
    y = np.log(np.asarray(outcomes, dtype=float))
    return float(np.polyfit(x, y, 1)[0])
