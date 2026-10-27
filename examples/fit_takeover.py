"""Fit the hand's release from the task to Gold et al. (2013).

After a take-over request the hand leaves the task and returns to the
wheel in parallel with the gaze shift (Zeeb, Buchner & Schrauf, 2015); how
long it takes to let go grows with the task's manual load. Gold et al.'s
drivers did SuRT on a touchscreen, as the simulated driver does, and had
their hands on the wheel after 1.45 s (5 s budget) and 1.79 s (7 s). The
model has no time budget, so ``hand_release`` is chosen to match the
midpoint over the runs of ``driving.takeover_components`` (tone with an
icon, 120 km/h). The hand starts one release time after the reaction, so
the mean time to the wheel grows with it. The fit takes the release at
zero and estimates the remaining gap. It then checks a local 10 ms grid
to account for simulation time steps and selects the closest mean.
The reaction waits for the central stage
held by SuRT, whose chunk is fitted to Conti et al. (2014), so refit after
changing that chunk. Ko et al.'s (2022) 1.87 s with a laptop game is left
for ``driving.takeover_modality`` to test.

python examples/fit_takeover.py --out evidence/takeover-fit.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

import embodied_human as eh
from embodied_human.agents.driver import DriverAgentParameters
from embodied_human.agents.road import simulate_takeover
from embodied_human.bench.references import REFERENCES

RUNS = 64
SPEED = 120 / 3.6


def mean_hands(release: float) -> float:
    """Mean time to the wheel after a tone with an icon, s."""
    params = replace(DriverAgentParameters(), hand_release=release)
    times = [
        simulate_takeover(
            eh.build_human(seed=seed), speed=SPEED, seed=seed, params=params
        ).takeover_components()["hand_s"]
        for seed in range(RUNS)
    ]
    return float(np.mean(times))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    reference = REFERENCES["driving.takeover_hands"]
    target = (reference.low + reference.high) / 2
    estimate = round(max(0.0, target - mean_hands(0.0)), 2)
    candidates = sorted({round(max(0.0, estimate + delta), 2)
                         for delta in (-0.02, -0.01, 0.0, 0.01, 0.02)})
    evaluated = {release: mean_hands(release) for release in candidates}
    release = min(candidates, key=lambda value: abs(evaluated[value] - target))
    report = {
        "target_s": target,
        "runs": RUNS,
        "speed_kmh": 120,
        "hand_release_s": release,
        "mean_hands_s": evaluated[release],
        "residual_s": evaluated[release] - target,
        "local_grid_mean_hands_s": evaluated,
    }
    print(json.dumps(report, indent=2))
    if args.out:
        args.out.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
