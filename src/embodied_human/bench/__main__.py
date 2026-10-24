"""Run the benchmark and write a report.

python -m embodied_human.bench --out results.json
python -m embodied_human.bench --only eyes.main_sequence memory.span
python -m embodied_human.bench --replace attention=seev --out seev.json
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from collections import Counter
from pathlib import Path
from typing import Any

from ..core import CATEGORIES, HumanConfig, build_human


def _config(replacements: list[str]) -> HumanConfig:
    config = HumanConfig()
    for item in replacements:
        category, _, name = item.partition("=")
        if category not in CATEGORIES or not name:
            raise SystemExit(
                f"--replace needs category=name with a category in {CATEGORIES}"
            )
        config = config.replace(category, name)
    return config


def main(argv: list[str] | None = None) -> int:
    from . import integrated  # noqa: F401
    from .benchmarks import BENCHMARKS, run

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--only", nargs="*", help=f"benchmarks to run; known: {', '.join(BENCHMARKS)}"
    )
    parser.add_argument(
        "--replace", nargs="*", default=[], help="category=name module swaps"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    config = _config(args.replace)
    human = build_human(config, seed=args.seed)
    started = time.perf_counter()
    results = run(human, args.only)
    kinds = Counter(row["kind"] for row in human.provenance())
    report: dict[str, Any] = {
        "package": "embodied-human 0.1.0",
        "python": platform.python_version(),
        "seed": args.seed,
        "config": config.as_dict(),
        "parameter_sources": dict(kinds),
        "wall_seconds": round(time.perf_counter() - started, 2),
        "results": [r.as_dict() for r in results],
    }
    width = max(len(r.name) for r in results)
    for r in results:
        status = "skip" if r.skipped else ("pass" if r.passed else "fail")
        score = "" if math.isnan(r.score) else f"{r.score:.2f}"
        values = ", ".join(
            f"{k.split('.', 1)[1]}={v:.3g}" for k, v in r.metrics.items()
        )
        print(f"{r.name:<{width}}  {status:4}  {score:>4}  {values or r.skipped or ''}")
    print(f"parameters by source: {dict(kinds)}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
