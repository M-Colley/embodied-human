"""Run the benchmark and write a report.

python -m embodied_human.bench --out results.json
python -m embodied_human.bench --only eyes.main_sequence memory.span
python -m embodied_human.bench --replace attention=seev --out seev.json
python -m embodied_human.bench --replace attention=uncertainty_sampler:return_sd=0.8
python -m embodied_human.bench --import my_modules --replace attention=my_sampler

A replacement is ``category=name`` with optional ``:key=value,...``
parameters (numbers, ``true``/``false`` or text). ``--import`` loads Python
modules first, so implementations they register can be named.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import platform
import time
from collections import Counter
from pathlib import Path
from typing import Any

from ..core import CATEGORIES, HumanConfig, build_human


def _value(text: str) -> Any:
    if text.lower() in ("true", "false"):
        return text.lower() == "true"
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _config(replacements: list[str]) -> HumanConfig:
    config = HumanConfig()
    for item in replacements:
        category, _, choice = item.partition("=")
        name, _, settings = choice.partition(":")
        if category not in CATEGORIES or not name:
            raise SystemExit(
                f"--replace needs category=name with a category in {CATEGORIES}"
            )
        params = {}
        for pair in filter(None, settings.split(",")):
            key, sep, value = pair.partition("=")
            if not sep:
                raise SystemExit(f"parameter {pair!r} needs key=value")
            params[key] = _value(value)
        config = config.replace(category, name, **params)
    return config


def main(argv: list[str] | None = None) -> int:
    from . import integrated, load, motivation, senses, studies  # noqa: F401
    from .benchmarks import BENCHMARKS, json_safe, run

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--only", nargs="*", help=f"benchmarks to run; known: {', '.join(BENCHMARKS)}"
    )
    parser.add_argument(
        "--replace", nargs="*", default=[], help="category=name module swaps"
    )
    parser.add_argument(
        "--import",
        dest="imports",
        nargs="*",
        default=[],
        help="Python modules to import first (they may register modules)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    for name in args.imports:
        importlib.import_module(name)
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
    width = max((len(r.name) for r in results), default=0)
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
        args.out.write_text(
            json.dumps(json_safe(report), indent=2, allow_nan=False), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
