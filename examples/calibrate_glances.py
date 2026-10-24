"""Infer the glance sampler's parameters from human data, then check them.

Approximate Bayesian computation (embodied_human.inference) fits the
uncertainty threshold for returning to the road (``return_sd``) and its
glance-to-glance variability (``threshold_spread``) to two summaries of
Kujala and Saariluoma (2011, Table 2): the mean in-vehicle glance (1.06 to
1.20 s over four interfaces) and the mean within-person SD (0.48 to
0.66 s), at 50 km/h within their 40-60 km/h. Each statistic's scale is half
its range over interfaces. Each parameter set is simulated twice for
300 s and the glances pooled; posterior predictive checks on two fresh
random streams then cover what was not fitted: the share of glances over
2 s and the longest glance.

python examples/calibrate_glances.py --out evidence/glance-calibration.json
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from embodied_human.agents.road import simulate_glances
from embodied_human.bench.references import REFERENCES
from embodied_human.core import HumanConfig, build_human
from embodied_human.inference import Prior, abc_rejection

SPEED = 50 / 3.6
SECONDS = 300.0
OBSERVED = {
    "mean_s": (1.06 + 1.08 + 1.15 + 1.20) / 4,
    "sd_s": (0.66 + 0.48 + 0.63 + 0.52) / 4,
}
SCALES = {"mean_s": (1.20 - 1.06) / 2, "sd_s": (0.66 - 0.48) / 2}
PRIORS = (
    Prior("return_sd", 0.03, 0.6, log=True),
    Prior("threshold_spread", 0.0, 2.0),
)


def summarize(
    params: Mapping[str, float], seeds: tuple[int, ...] = (0, 1)
) -> dict[str, float]:
    """Glance summaries of one simulated driver with these parameters."""
    config = HumanConfig().replace("attention", "uncertainty_sampler", **params)
    durations = np.concatenate(
        [
            simulate_glances(
                build_human(config, seed=seed), speed=SPEED, seconds=SECONDS, seed=seed
            ).glance_durations()
            for seed in seeds
        ]
    )
    if len(durations) < 2:
        return {"mean_s": float("nan"), "sd_s": float("nan")}
    return {
        "mean_s": float(durations.mean()),
        "sd_s": float(durations.std(ddof=1)),
        "long_share": float(np.mean(durations > 2.0)),
        "max_s": float(durations.max()),
        "glances": float(len(durations)),
    }


def predictive(params: Mapping[str, float]) -> dict[str, float]:
    """Summaries on a fresh random stream, for posterior predictive checks."""
    return summarize(params, seeds=(2, 3))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--accept", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    started = time.perf_counter()
    with ProcessPoolExecutor(args.workers) as pool:
        posterior = abc_rejection(
            summarize,
            OBSERVED,
            SCALES,
            PRIORS,
            samples=args.samples,
            accept=args.accept,
            mapper=pool.map,
        )
        accepted = [
            dict(zip(posterior.names, row.tolist())) for row in posterior.samples
        ]
        checks = list(pool.map(predictive, accepted))

    def interval(key: str) -> list[float]:
        values = np.array([c[key] for c in checks])
        return [
            float(np.percentile(values, 5)),
            float(np.median(values)),
            float(np.percentile(values, 95)),
        ]

    long_reference = REFERENCES["driving.human_long_glance_share"]
    report = {
        "method": "rejection ABC, scrambled Sobol prior draws (SciPy), Euclidean "
        "distance of scaled summaries",
        "python": platform.python_version(),
        "wall_seconds": round(time.perf_counter() - started, 1),
        "simulation": {
            "speed_kmh": 50.0,
            "seconds": SECONDS,
            "runs_pooled": 2,
            "task": "SuRT on StraightRoad",
        },
        "observed": OBSERVED,
        "scales": SCALES,
        "source": "Kujala & Saariluoma (2011), Table 2, "
        "https://doi.org/10.1080/00140139.2011.592601",
        "priors": [vars(p) for p in PRIORS],
        "posterior": posterior.as_dict(),
        "accepted_summaries": [dict(s) for s in posterior.summaries],
        "posterior_predictive_5_50_95": {
            "mean_s": interval("mean_s"),
            "sd_s": interval("sd_s"),
            "long_share": interval("long_share"),
            "max_s": interval("max_s"),
        },
        "held_out": {
            "long_share_reference": [long_reference.low, long_reference.high],
            "long_share_citation": long_reference.citation,
            "max_glance_reference_s": [2.42, 3.70],
            "max_glance_citation": "Kujala & Saariluoma (2011), Table 2, "
            "per interface",
        },
    }
    print(json.dumps(report["posterior"], indent=2))
    print(json.dumps(report["posterior_predictive_5_50_95"], indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
