"""Fit how often a driver checks on automation to Hergeth et al. (2016).

While automation drives and SuRT is shown, the simulated driver leaves the
task to check the road at a rate of ``monitoring_gain`` times one minus its
trust. Hergeth, Lorenz, Vilimek and Krems (2016) measured a median of 0.13
monitoring glances per second during the first 80 s SuRT block, at a mean
self-reported trust of 74.43%, and 0.097 per second during the last, at
82.08%. ``monitoring_gain`` is chosen on a grid to minimise the squared log
error of the model's median rate at these two trust levels (trust's 0-1
scale read as the rating's percentage). Zeeb, Buchner and Schrauf's (2015)
monitoring data are left for ``driving.automation_monitoring`` to test.

python examples/fit_monitoring.py --out evidence/monitoring-fit.json
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np

import embodied_human as eh
from embodied_human.agents.driver import DriverAgentParameters
from embodied_human.agents.road import simulate_takeover

#: Hergeth et al. (2016): (mean trust 0-1, median monitoring glances per s).
HERGETH = ((0.7443, 0.13), (0.8208, 0.097))
RUNS = 32
START, END = 10.0, 90.0


def median_rate(gain: float, trust: float) -> float:
    """Median road checks per second during an 80 s SuRT block."""
    params = replace(DriverAgentParameters(), monitoring_gain=gain)
    config = eh.HumanConfig().replace(
        "trust", "decision_field", automation_belief=trust
    )
    rates = []
    for seed in range(RUNS):
        log = simulate_takeover(
            eh.build_human(config, seed=seed),
            speed=120 / 3.6,
            request_at=END,
            seconds=END,
            seed=seed,
            params=params,
        )
        checks = sum(START <= time < END for time, _ in log.returns)
        rates.append(checks / (END - START))
    return float(np.median(rates))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    grid = np.round(np.arange(0.40, 0.91, 0.02), 2)
    errors = {}
    for gain in grid:
        rates = [median_rate(float(gain), trust) for trust, _ in HERGETH]
        errors[float(gain)] = (
            sum(math.log(r / h) ** 2 for r, (_, h) in zip(rates, HERGETH)),
            rates,
        )
    best = min(errors, key=lambda g: errors[g][0])
    report = {
        "hergeth": [{"trust": t, "median_per_s": f} for t, f in HERGETH],
        "runs": RUNS,
        "monitoring_gain": best,
        "model_median_per_s": errors[best][1],
    }
    print(json.dumps(report, indent=2))
    if args.out:
        args.out.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
