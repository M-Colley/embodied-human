"""Fit the take-over decision time to Ko et al. (2022).

Ko et al.'s drivers grabbed the wheel 1.87 s after an auditory warning with
a visual text while they played a game on the centre console. The driver's
``decision_dwell`` (time looking at the road before the hand starts back)
is chosen on a 10 ms grid so that the median time to the wheel over the
runs of ``driving.takeover_modality`` (chime and icon, SuRT during
automation) matches it. The take-over reaction waits for the central stage
held by SuRT, whose chunk is fitted to Conti et al. (2014), so refit after
changing that chunk. Gold et al.'s (2013) take-over components are not
used here; ``driving.takeover_components`` tests them.

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

RUNS = 12


def median_hands(dwell: float) -> float:
    """Median time to the wheel after a chime with an icon, s."""
    params = replace(DriverAgentParameters(), decision_dwell=dwell)
    times = [
        simulate_takeover(
            eh.build_human(seed=seed),
            signals=("auditory", "visual"),
            seed=seed,
            params=params,
        ).takeover_components()["hand_s"]
        for seed in range(RUNS)
    ]
    return float(np.median(times))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    target = REFERENCES["driving.takeover_hands_auditory"].value
    grid = np.round(np.arange(0.3, 1.31, 0.01), 2)
    hands = {float(d): median_hands(float(d)) for d in grid}
    best = min(hands, key=lambda d: abs(hands[d] - target))
    report = {
        "target_s": target,
        "runs": RUNS,
        "decision_dwell_s": best,
        "median_hands_s": hands[best],
    }
    print(json.dumps(report, indent=2))
    if args.out:
        args.out.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
