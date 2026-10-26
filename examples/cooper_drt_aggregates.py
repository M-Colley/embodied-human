"""Detection-response times of Cooper et al. (2020) by stimulus type.

Cooper et al. pooled responses to a vibration under the collarbone and a
light reflected in the windshield. Their raw data (``drt.csv`` in
``data.zip`` at https://github.com/utahcdst/Aging-Report-Frontiers, which
has no licence) allow the two to be separated. Only aggregates are
written; the raw data are not part of this package.

The filters follow the authors' ``rt.R``: hits only, and SuRT stimuli only
while SuRT was in use. As in ``rt.R``, reaction times are first averaged
per session (participant, vehicle and task); the session means of the
pooled stimuli reproduce their Table 4. Participants drove several
vehicles, so for intervals the sessions are then averaged per participant:

- means with t-based 95% confidence intervals;
- costs of the 2-back and SuRT over driving alone, paired within
  participants;
- the ratio of older to younger drivers, with a bootstrap interval.

Needs pandas. python examples/cooper_drt_aggregates.py drt.csv
--out evidence/cooper-drt-aggregates.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

#: Stimulus codes in drt.csv: 1 is the windshield light (``hr.R`` scores
#: hits for it alone), 2 the vibration (hit in 97-98% of single-task trials).
STIMULI = {"tactile": [2], "light": [1], "pooled": [1, 2]}
COHORTS = {"Age: 21-36": "younger", "Age: 55-75": "older"}
TASKS = ("Single", "nBack", "SuRT")


def interval(values: pd.Series) -> dict[str, float]:
    """Mean with a t-based 95% confidence interval."""
    x = values.dropna().to_numpy(dtype=float)
    half = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return {
        "mean": float(x.mean()),
        "low": float(x.mean() - half),
        "high": float(x.mean() + half),
        "sd": float(x.std(ddof=1)),
        "n": int(len(x)),
    }


def hits(path: Path) -> pd.DataFrame:
    """Valid responses, filtered as in rt.R."""
    data = pd.read_csv(path, low_memory=False)
    data = data[data["task"].notna()]
    in_use = data["bttn_prior"].astype(str).str.strip() == "T"
    data = data[in_use | (data["task"] != "SuRT")]
    data = data[data["hit"].astype(str).str.upper() == "TRUE"]
    return data[data["task"].isin(TASKS)]


def aggregates(data: pd.DataFrame, seed: int = 0) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    report: dict[str, object] = {}
    for stimulus, codes in STIMULI.items():
        subset = data[data["stim"].isin(codes)]
        sessions = (
            subset.groupby(["sub", "Vehicle", "task", "Cohort"])["rt"]
            .mean()
            .reset_index()
        )
        people = (
            sessions.groupby(["sub", "task", "Cohort"])["rt"].mean().unstack("task")
        )
        block: dict[str, object] = {}
        for cohort, label in COHORTS.items():
            rows = people.xs(cohort, level="Cohort")
            mine = sessions[sessions["Cohort"] == cohort]
            entry: dict[str, object] = {}
            for task in TASKS:
                entry[task] = {
                    "session_mean": float(mine[mine["task"] == task]["rt"].mean()),
                    "participants": interval(rows[task]),
                }
            for task in ("nBack", "SuRT"):
                entry[f"{task}_cost"] = interval(rows[task] - rows["Single"])
            block[label] = entry
        young = people.xs("Age: 21-36", level="Cohort")["Single"].dropna().to_numpy()
        old = people.xs("Age: 55-75", level="Cohort")["Single"].dropna().to_numpy()
        ratios = [
            rng.choice(old, len(old)).mean() / rng.choice(young, len(young)).mean()
            for _ in range(5000)
        ]
        block["older_ratio_single"] = {
            "ratio": float(old.mean() / young.mean()),
            "low": float(np.percentile(ratios, 2.5)),
            "high": float(np.percentile(ratios, 97.5)),
        }
        report[stimulus] = block
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("drt_csv", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = {
        "source": "Cooper et al. (2020), https://doi.org/10.3389/fpsyg.2020.01154; "
        "raw data github.com/utahcdst/Aging-Report-Frontiers (data.zip, drt.csv)",
        "unit": "ms",
        **aggregates(hits(args.drt_csv)),
    }
    print(json.dumps(report, indent=2))
    if args.out:
        args.out.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
