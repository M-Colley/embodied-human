"""Inverting the model: infer parameters from human summary data.

Approximate Bayesian computation by rejection (Pritchard et al., 1999;
Beaumont, Zhang and Balding, 2002): draw parameters from the prior, here
with a scrambled Sobol sequence (SciPy's quasi-Monte Carlo), simulate,
compare the simulated summary statistics with the observed ones, and keep
the closest fraction as an approximate posterior. It needs nothing but a
forward simulator, so it applies to any module combination, and it reports
the uncertainty that remains, not only a point estimate.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Prior:
    """A uniform (or log-uniform) prior on one parameter."""

    name: str
    low: float
    high: float
    log: bool = False

    def __post_init__(self) -> None:
        if not self.low < self.high or (self.log and self.low <= 0):
            raise ValueError(
                f"prior {self.name}: need low < high, and low > 0 for log priors"
            )

    def scale(self, unit: np.ndarray) -> np.ndarray:
        """Map [0, 1) samples to the prior's range."""
        if self.log:
            return np.exp(
                np.log(self.low) + unit * (np.log(self.high) - np.log(self.low))
            )
        return self.low + unit * (self.high - self.low)


@dataclass(frozen=True)
class Posterior:
    """Accepted parameter sets and their distances.

    Attributes:
        names: Parameter names.
        samples: Accepted parameters, one row per sample.
        distances: Their distances to the observed summaries.
        all_samples, all_distances: Everything simulated.
        summaries: Simulated summaries of the accepted samples.
    """

    names: tuple[str, ...]
    samples: np.ndarray
    distances: np.ndarray
    all_samples: np.ndarray
    all_distances: np.ndarray
    summaries: tuple[Mapping[str, float], ...]

    def median(self) -> dict[str, float]:
        """Posterior median per parameter."""
        return {
            n: float(np.median(self.samples[:, i])) for i, n in enumerate(self.names)
        }

    def interval(self, level: float = 0.9) -> dict[str, tuple[float, float]]:
        """Central credible interval per parameter."""
        tail = (1 - level) / 2 * 100
        return {
            n: (
                float(np.percentile(self.samples[:, i], tail)),
                float(np.percentile(self.samples[:, i], 100 - tail)),
            )
            for i, n in enumerate(self.names)
        }

    def best(self) -> dict[str, float]:
        """The single closest parameter set."""
        return dict(
            zip(self.names, self.samples[int(np.argmin(self.distances))].tolist())
        )

    def as_dict(self) -> dict[str, object]:
        """JSON-serializable summary."""
        return {
            "parameters": list(self.names),
            "median": self.median(),
            "interval_90": {k: list(v) for k, v in self.interval().items()},
            "best": self.best(),
            "accepted": int(len(self.samples)),
            "simulated": int(len(self.all_samples)),
            "distance_threshold": float(self.distances.max()),
        }


def distance(
    simulated: Mapping[str, float],
    observed: Mapping[str, float],
    scales: Mapping[str, float],
) -> float:
    """Euclidean distance of scaled summary differences (NaN counts as far)."""
    total = 0.0
    for key, value in observed.items():
        x = float(simulated.get(key, math.nan))
        if not math.isfinite(x):
            return math.inf
        total += ((x - value) / scales[key]) ** 2
    return math.sqrt(total)


def abc_rejection(
    simulate: Callable[[Mapping[str, float]], Mapping[str, float]],
    observed: Mapping[str, float],
    scales: Mapping[str, float],
    priors: Sequence[Prior],
    *,
    samples: int = 256,
    accept: float = 0.1,
    seed: int = 0,
    mapper: Callable[..., Iterable[Mapping[str, float]]] = map,
) -> Posterior:
    """Rejection ABC with quasi-random prior draws.

    Args:
        simulate: Maps parameters to summary statistics.
        observed: Human summary statistics.
        scales: Scale of each statistic (e.g. its between-study spread).
        priors: One prior per parameter.
        samples: Simulations to run (rounded up to a power of two).
        accept: Fraction of the closest simulations to keep.
        seed: Seed of the Sobol scrambling.
        mapper: Applies ``simulate`` to every parameter set; pass an
            executor's ``map`` to simulate in parallel.
    """
    from scipy.stats import qmc

    if not 0 < accept <= 1:
        raise ValueError("accept must be in (0, 1]")
    names = tuple(p.name for p in priors)
    exponent = int(math.ceil(math.log2(max(samples, 2))))
    unit = qmc.Sobol(len(priors), scramble=True, rng=seed).random_base2(exponent)
    draws = np.column_stack([p.scale(unit[:, i]) for i, p in enumerate(priors)])
    settings = [dict(zip(names, values.tolist())) for values in draws]
    summaries = [dict(s) for s in mapper(simulate, settings)]
    distances = np.array([distance(s, observed, scales) for s in summaries])
    keep = max(1, int(round(accept * len(draws))))
    order = np.argsort(distances)[:keep]
    return Posterior(
        names,
        draws[order],
        distances[order],
        draws,
        distances,
        tuple(summaries[i] for i in order),
    )
