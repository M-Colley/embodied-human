"""Run the benchmark and write a report.

python -m embodied_human.bench --out results.json
python -m embodied_human.bench --only eyes.main_sequence memory.span
python -m embodied_human.bench --replace attention=seev --out seev.json
python -m embodied_human.bench --replace attention=uncertainty_sampler:return_sd=0.8
python -m embodied_human.bench --import my_modules --replace attention=my_sampler
python -m embodied_human.bench --profile core --list
python -m embodied_human.bench --compare baseline.json --fail-on-regression

A replacement is ``category=name`` with optional ``:key=value,...``
parameters (numbers, ``true``/``false`` or text). ``--import`` loads Python
modules first, so implementations they register can be named.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import time
from pathlib import Path
from typing import Any

from ..core import CATEGORIES, HumanConfig, build_human
from ..reporting import load_replay, make_report, write_bundle, write_json


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
    from .benchmarks import BENCHMARKS, run

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--only", nargs="*", help=f"benchmarks to run; known: {', '.join(BENCHMARKS)}"
    )
    selection.add_argument(
        "--profile", choices=("core", "full"),
        help="declared packaged benchmark profile",
    )
    parser.add_argument(
        "--list", action="store_true", help="list selected protocol names"
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
    parser.add_argument("--bundle", type=Path, help="write a replayable run directory")
    parser.add_argument("--config", type=Path, help="builder JSON project or config")
    parser.add_argument("--replay", type=Path, help="replay a checked run manifest")
    parser.add_argument("--allow-source-mismatch", action="store_true")
    parser.add_argument("--allow-environment-mismatch", action="store_true")
    parser.add_argument(
        "--compare", type=Path, help="compare with a baseline JSON report"
    )
    parser.add_argument("--summary", type=Path, help="write a Markdown result summary")
    parser.add_argument("--fail-on-regression", action="store_true")
    parser.add_argument("--fail-on-skip", action="store_true")
    parser.add_argument(
        "--regression-tolerance", type=float, default=1e-9,
        help="nonnegative tolerance for normalized reference error increases",
    )
    args = parser.parse_args(argv)

    if not math.isfinite(args.regression_tolerance) or args.regression_tolerance < 0:
        parser.error("--regression-tolerance must be finite and nonnegative")
    if args.fail_on_regression and args.compare is None:
        parser.error("--fail-on-regression requires --compare")
    if args.compare and args.only == []:
        parser.error("benchmark comparisons require a nonempty selection")
    if args.replay and (args.only is not None or args.profile is not None):
        parser.error("the replay manifest selects its benchmark protocols")

    if args.bundle and args.imports:
        parser.error(
            "bundles require packaged implementations; "
            "external --import sources are not captured"
        )

    for name in args.imports:
        importlib.import_module(name)
    selected = args.only
    if args.profile:
        from .catalog import benchmark_names

        selected = benchmark_names(args.profile)
    if args.list:
        if any((args.compare, args.summary, args.out, args.bundle, args.replay)):
            parser.error("--list cannot write or compare benchmark runs")
        unknown = set(selected or []) - set(BENCHMARKS)
        if unknown:
            parser.error(f"unknown benchmarks: {', '.join(sorted(unknown))}")
        for name in list(BENCHMARKS) if selected is None else selected:
            print(name)
        return 0
    config = _config(args.replace)
    if args.config and args.replay:
        parser.error("choose either --config or --replay")
    if args.config or args.replay:
        from ..builder import config_from_dict

        if args.replay:
            replay = load_replay(
                args.replay,
                allow_source_mismatch=args.allow_source_mismatch,
                allow_environment_mismatch=args.allow_environment_mismatch,
            )
            if replay.get("kind") != "benchmark":
                parser.error("manifest does not contain a benchmark run")
            config = config_from_dict(replay["config"])
            selected, args.seed = replay["benchmarks"], replay["seed"]
        else:
            data = json.loads(args.config.read_text(encoding="utf-8"))
            config = config_from_dict(data.get("config", data))
            args.seed = data.get("seed", args.seed)
    human = build_human(config, seed=args.seed)
    started = time.perf_counter()
    results = run(human, selected, seed=args.seed)
    report = make_report(
        human, results, seed=args.seed, wall_seconds=time.perf_counter() - started
    )
    width = max((len(r.name) for r in results), default=0)
    for r in results:
        status = "skip" if r.skipped else ("pass" if r.passed else "fail")
        score = "" if math.isnan(r.score) else f"{r.score:.2f}"
        values = ", ".join(
            f"{k.split('.', 1)[1]}={v:.3g}" for k, v in r.metrics.items()
        )
        print(f"{r.name:<{width}}  {status:4}  {score:>4}  {values or r.skipped or ''}")
    print(f"parameters by source: {report['parameter_sources']}")
    print(f"validation by metric: {report['validation_summary']}")
    comparison = None
    comparison_error = None
    baseline_text = None
    if args.compare:
        from .comparison import compare_reports

        try:
            baseline_text = args.compare.read_bytes().decode("utf-8")
            baseline = json.loads(baseline_text)
            comparison = compare_reports(
                report, baseline, names=args.only, tolerance=args.regression_tolerance
            )
        except (OSError, ValueError) as exc:
            comparison_error = str(exc)
            report["comparison_error"] = comparison_error
        if comparison is not None:
            report["comparison"] = comparison
            print(
                f"comparison: {len(comparison['regressions'])} regressions; "
                f"{len(comparison['incomparable'])} incomparable results"
            )
    summary_text = None
    if args.summary or args.compare:
        from .comparison import markdown_summary

        summary_text = markdown_summary(report, comparison)
    extra_files = {}
    if summary_text is not None:
        extra_files["summary.md"] = summary_text
    if baseline_text is not None:
        extra_files["comparison-baseline.json"] = baseline_text
    if args.out:
        write_json(args.out, report)
    if args.bundle:
        write_bundle(
            args.bundle,
            report,
            replay={
                "kind": "benchmark",
                "config": report["resolved_config"],
                "benchmarks": [r.name for r in results],
                "seed": args.seed,
            },
            extra_files=extra_files,
        )
    if args.summary and summary_text is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(summary_text, encoding="utf-8", newline="")
    if comparison_error is not None:
        parser.error(f"cannot compare benchmark reports: {comparison_error}")
    regression = args.fail_on_regression and comparison is not None and (
        comparison["regressions"] or comparison["incomparable"]
    )
    skipped = args.fail_on_skip and any(result.skipped for result in results)
    return 1 if regression or skipped else 0


if __name__ == "__main__":
    raise SystemExit(main())
