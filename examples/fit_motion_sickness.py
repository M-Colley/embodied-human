"""Fit the motion sickness model's conflict scale to vertical-motion data,
then test it on conditions it was not fitted to.

The subjective vertical conflict model's parameters (Wada, Fujisawa & Doi,
2018) were chosen for car passengers. As Bos and Bles (1998) did for the
1-D model, one parameter is fitted here: the conflict scale ``b`` of the
Hill function, to the percentage of people who vomited within 2 hours at
14 vertical oscillation conditions (O'Hanlon & McCauley, 1974). The held-out
test is the time course of sickness at nine other conditions (McCauley et
al., 1976), compared with the authors' own descriptive model.

python examples/fit_motion_sickness.py --out evidence/motion-sickness-fit.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

from embodied_human.bench.senses import VERTICAL_SICKNESS, sickness_time_course
from embodied_human.vestibular import SubjectiveVertical


def endpoints(scale: float, dt: float) -> np.ndarray:
    """MSI after 2 hours at the 14 fitted conditions."""
    t = np.arange(0.0, 7200.0, dt)
    motion = np.zeros((len(VERTICAL_SICKNESS), len(t), 3))
    for i, (frequency, mean_g, _) in enumerate(VERTICAL_SICKNESS):
        motion[i, :, 2] = mean_g / 0.637 * 9.81 * np.sin(2 * math.pi * frequency * t)
    return SubjectiveVertical(conflict_scale=scale).simulate(dt, motion)[:, -1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    observed = np.array([row[2] for row in VERTICAL_SICKNESS], dtype=float)
    started = time.perf_counter()

    def error(scale: float) -> float:
        # A coarser step for the search; the result is rechecked at 0.1 s.
        return float(np.sqrt(np.mean((endpoints(scale, 0.25) - observed) ** 2)))

    fit = minimize_scalar(
        error, bounds=(0.05, 1.0), method="bounded", options={"xatol": 0.002}
    )
    scale = float(fit.x)
    report: dict[str, object] = {
        "fitted_conflict_scale_m_s2": scale,
        "evaluations": int(fit.nfev),
    }
    for name, value in (("published", 0.5), ("fitted", scale)):
        fitted_set = endpoints(value, 0.1)
        predicted, seen, descriptive = sickness_time_course(
            SubjectiveVertical(conflict_scale=value), 0.1
        )
        report[name] = {
            "conflict_scale": value,
            "fit_set_rmse": float(np.sqrt(np.mean((fitted_set - observed) ** 2))),
            "fit_set_r": float(np.corrcoef(fitted_set, observed)[0, 1]),
            "held_out_rmse": float(np.sqrt(np.mean((predicted - seen) ** 2))),
            "held_out_r": float(np.corrcoef(predicted, seen)[0, 1]),
        }
    report["held_out_mccauley_model_rmse"] = float(
        np.sqrt(np.mean((descriptive - seen) ** 2))
    )
    report["wall_seconds"] = round(time.perf_counter() - started, 1)
    print(json.dumps(report, indent=2))
    if args.out:
        args.out.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
