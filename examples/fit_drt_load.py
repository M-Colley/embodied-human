"""Fit the central chunks of the n-back and SuRT to Conti et al. (2014).

The visual head-mounted DRT slowed by 135 ms with a 0-back, 292 ms with a
2-back and 385 ms with easy SuRT, over driving alone (pooled over four
instruction groups, N = 71). Each chunk length is fitted (SciPy's brentq)
so that the model's increase matches; driving itself is not fitted. The
fitted values belong in ``embodied_human.tasks.load``.

python examples/fit_drt_load.py --out evidence/drt-load-fit.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

import embodied_human as eh
from embodied_human.bench.load import CONTI, display_share, drt_rt
from embodied_human.tasks.load import DRIVING, SURT_SCREEN, n_back, surt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    human = eh.build_human(seed=0)
    share = display_share(human)

    def cost(load: list, on_display: float = 0.0) -> float:
        # Common random numbers: every evaluation draws the same samples.
        human.response.reset(np.random.default_rng(0))
        driving = drt_rt(human, ["head_light"], [DRIVING], 0.0)
        human.response.reset(np.random.default_rng(0))
        return drt_rt(human, ["head_light"], load, on_display) - driving

    targets = {
        "0-back": CONTI["0-back"] - CONTI["driving"],
        "2-back": CONTI["2-back"] - CONTI["driving"],
        "surt": CONTI["surt"] - CONTI["driving"],
    }
    fitted = {
        "0-back": brentq(
            lambda d: cost([DRIVING, n_back(0, d)]) - targets["0-back"],
            0.05,
            2.2,
            xtol=1e-3,
        ),
        "2-back": brentq(
            lambda d: cost([DRIVING, n_back(2, d)]) - targets["2-back"],
            0.05,
            2.2,
            xtol=1e-3,
        ),
        "surt": brentq(
            lambda d: cost([DRIVING, surt(d)], share) - targets["surt"],
            0.05,
            SURT_SCREEN - 0.01,
            xtol=1e-3,
        ),
    }
    report = {"display_share": share, "targets_ms": targets, "chunks_s": fitted}
    print(json.dumps(report, indent=2))
    if args.out:
        args.out.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
