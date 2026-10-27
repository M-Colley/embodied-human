"""Seeded comparisons, parameter sweeps and replayable experiment execution."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import time
from collections import defaultdict
from collections.abc import Callable
from html import escape
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np

from .bench.benchmarks import BENCHMARKS, json_safe, run
from .bench.metrics import bootstrap_ci
from .bench.validation import evaluation_for, validation_summary
from .builder import config_from_dict
from .core import HumanConfig, build_human
from .reporting import load_replay, parameter_record, write_bundle, write_json

MAX_EVALUATIONS = 2048


def experiment_catalog() -> dict[str, Any]:
    from .bench import integrated, load, motivation, senses, studies  # noqa: F401
    from .experiment_protocols import CATALOG

    protocols = json.loads(
        files("embodied_human.bench").joinpath("protocols.json").read_text()
    )
    return {
        "version": 1,
        "benchmarks": [
            {
                "name": name,
                "description": (function.__doc__ or "").strip(),
                "categories": protocols.get(name, {}).get("categories", []),
                "metrics": protocols.get(name, {}).get("metrics", []),
            }
            for name, function in BENCHMARKS.items()
        ]
        + CATALOG,
        "limits": {"seeds": 20, "sweep_values": 32, "evaluations": MAX_EVALUATIONS},
    }


def _integer(value: Any, name: str, low: int, high: int) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer from {low} to {high}")
    return value


def validate_request(data: Any) -> dict[str, Any]:
    allowed = {
        "version",
        "config",
        "benchmarks",
        "seeds",
        "compare_baseline",
        "sweep",
        "sensitivity",
    }
    if not isinstance(data, dict) or set(data) - allowed:
        raise ValueError("experiment must contain known fields")
    if type(data.get("version")) is not int or data["version"] != 1:
        raise ValueError("experiment version must be 1")
    config = config_from_dict(data.get("config"))
    catalog = {entry["name"] for entry in experiment_catalog()["benchmarks"]}
    names, seeds = data.get("benchmarks"), data.get("seeds")
    if (
        not isinstance(names, list)
        or not names
        or any(not isinstance(n, str) or n not in catalog for n in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("benchmarks must contain distinct registered names")
    if not isinstance(seeds, list) or not 1 <= len(seeds) <= 20:
        raise ValueError("seeds must contain 1 to 20 distinct integers")
    for seed in seeds:
        _integer(seed, "seed", 0, 2**32 - 1)
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be distinct")
    compare = data.get("compare_baseline", False)
    if type(compare) is not bool:
        raise ValueError("compare_baseline must be a boolean")
    result = {
        "version": 1,
        "config": json_safe(config.as_dict()),
        "benchmarks": names,
        "seeds": seeds,
        "compare_baseline": compare,
        "sweep": data.get("sweep"),
        "sensitivity": data.get("sensitivity"),
    }
    build_human(config, seed=seeds[0])
    count = _request_cases(result, config)
    if count * len(names) * len(seeds) > MAX_EVALUATIONS:
        raise ValueError(f"experiment exceeds {MAX_EVALUATIONS} benchmark evaluations")
    return result


def _changed_config(
    config: HumanConfig, category: str, field: str, value: Any
) -> HumanConfig:
    data = json_safe(config.as_dict())
    if category not in data:
        raise ValueError("unknown parameter category")
    data[category]["params"][field] = value
    changed = config_from_dict(data)
    build_human(changed, seed=0)
    return changed


def _request_cases(request: dict[str, Any], config: HumanConfig) -> int:
    count = 1 + int(request["compare_baseline"])
    sweep = request["sweep"]
    if sweep is not None:
        if not isinstance(sweep, dict) or set(sweep) != {
            "category",
            "parameter",
            "values",
        }:
            raise ValueError("sweep needs category, parameter and values")
        category, field, values = sweep["category"], sweep["parameter"], sweep["values"]
        if not isinstance(category, str) or not isinstance(field, str):
            raise ValueError("sweep category and parameter must be strings")
        if not isinstance(values, list) or not 1 <= len(values) <= 32:
            raise ValueError("sweep needs 1 to 32 values")
        for value in values:
            _changed_config(config, category, field, value)
        if len(set(json.dumps(v, sort_keys=True) for v in values)) != len(values):
            raise ValueError("sweep values must be distinct")
        count = len(values) + int(request["compare_baseline"])
    if request["sensitivity"] is not None:
        from .sensitivity import validate_sensitivity

        if request["compare_baseline"] or sweep is not None:
            raise ValueError("sensitivity cannot be combined with comparison or sweep")
        spec = validate_sensitivity(
            request["sensitivity"], config, request["benchmarks"]
        )
        request["sensitivity"] = spec
        count = spec["evaluations"]
    return count


def _cases(request: dict[str, Any]) -> list[tuple[str, HumanConfig, dict[str, Any]]]:
    config = config_from_dict(request["config"])
    cases: list[tuple[str, HumanConfig, dict[str, Any]]] = (
        [("baseline", HumanConfig(), {})] if request["compare_baseline"] else []
    )
    sweep = request["sweep"]
    if sweep is None:
        cases.append(("configured", config, {}))
    else:
        for index, value in enumerate(sweep["values"]):
            changed = _changed_config(
                config, sweep["category"], sweep["parameter"], value
            )
            cases.append(
                (
                    f"sweep_{index}",
                    changed,
                    {sweep["category"] + "." + sweep["parameter"]: value},
                )
            )
    return cases


def _statistics(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return json_safe(
        {
            "n": len(array),
            "mean": float(array.mean()) if len(array) else None,
            "ci95": bootstrap_ci(array),
        }
    )


def _summaries(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str], dict[int, float]] = defaultdict(dict)
    evaluations: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        for metric, value in row["metrics"].items():
            if value is not None and math.isfinite(value):
                grouped[row["case"], row["benchmark"], metric][row["seed"]] = value
                evaluations[row["case"], row["benchmark"], metric] = row.get(
                    "validation", {}
                ).get(metric, evaluation_for(metric).as_dict())
    summaries, differences = [], []
    for (case, benchmark, metric), values in grouped.items():
        summaries.append(
            {
                "case": case,
                "benchmark": benchmark,
                "metric": metric,
                "validation_role": evaluations[case, benchmark, metric]["role"],
                "validation": evaluations[case, benchmark, metric],
                **_statistics(list(values.values())),
            }
        )
        baseline = grouped.get(("baseline", benchmark, metric))
        if case != "baseline" and baseline is not None:
            paired = [
                value - baseline[seed]
                for seed, value in values.items()
                if seed in baseline
            ]
            differences.append(
                {
                    "case": case,
                    "baseline": "baseline",
                    "benchmark": benchmark,
                    "metric": metric,
                    "difference": "case_minus_baseline",
                    "validation_role": evaluations[case, benchmark, metric]["role"],
                    "baseline_validation_role": evaluations[
                        "baseline", benchmark, metric
                    ]["role"],
                    **_statistics(paired),
                }
            )
    return summaries, differences


def run_experiment(
    request: Any,
    *,
    progress: Callable[[dict[str, Any]], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    request = validate_request(request)
    started = time.perf_counter()
    sensitivity = request["sensitivity"]
    design, problem = None, None
    if sensitivity is None:
        cases = _cases(request)
    else:
        from .sensitivity import sample_design

        problem, design = sample_design(sensitivity)
        base = config_from_dict(request["config"])
        cases = []
        for index, point in enumerate(design):
            config, overrides = base, {}
            for factor, value in zip(sensitivity["parameters"], point):
                config = _changed_config(
                    config, factor["category"], factor["parameter"], float(value)
                )
                overrides[factor["category"] + "." + factor["parameter"]] = float(value)
            cases.append((f"sample_{index}", config, overrides))
    total = len(cases) * len(request["seeds"]) * len(request["benchmarks"])
    rows: list[dict[str, Any]] = []
    configs = {}
    for case, config, overrides in cases:
        configs[case] = parameter_record(build_human(config, seed=request["seeds"][0]))
        for seed in request["seeds"]:
            for name in request["benchmarks"]:
                if cancelled is not None and cancelled():
                    raise InterruptedError("experiment cancelled")
                if progress:
                    progress(
                        {
                            "completed": len(rows),
                            "total": total,
                            "message": f"{case}, seed {seed}, {name}",
                            "status": "running",
                        }
                    )
                human = build_human(config, seed=seed)
                try:
                    if name in BENCHMARKS:
                        result = run(human, [name], seed=seed)[0].as_dict()
                    else:
                        from .bench.protocol import benchmark_seed
                        from .experiment_protocols import PROTOCOLS

                        with benchmark_seed(seed):
                            result = PROTOCOLS[name](human).as_dict()
                except ImportError as exc:
                    result = {
                        "metrics": {},
                        "scores": {},
                        "validation": {},
                        "references": {},
                        "skipped": str(exc),
                        "details": {},
                    }
                rows.append(
                    {
                        "case": case,
                        "seed": seed,
                        "overrides": overrides,
                        "benchmark": name,
                        **{
                            key: result.get(key)
                            for key in (
                                "metrics",
                                "scores",
                                "validation",
                                "references",
                                "skipped",
                                "details",
                            )
                        },
                    }
                )
    summaries, differences = _summaries(rows)
    report = {
        "schema_version": 1,
        "request": request,
        "rows": rows,
        "resolved_cases": configs,
        "summaries": summaries,
        "paired_differences": differences,
        "sensitivity": None,
        "validation_summary": validation_summary(rows),
        "validation_summary_unit": "scored metric evaluations across cases and seeds",
        "unique_reference_metrics": sorted(
            {metric for row in rows for metric in row["scores"]}
        ),
        "unscored_metric_evaluations": sum(
            len(row["metrics"]) - len(row["scores"]) for row in rows
        ),
        "wall_seconds": round(time.perf_counter() - started, 3),
        "uncertainty": (
            "BCa 95% intervals across simulation seeds; undefined with fewer "
            "than three values or no spread."
        ),
    }
    if sensitivity is not None:
        report["sensitivity"] = _analyze_sensitivity(sensitivity, problem, design, rows)
    if progress:
        progress(
            {
                "completed": total,
                "total": total,
                "message": "Complete",
                "status": "complete",
            }
        )
    return json_safe(report)


def _analyze_sensitivity(
    spec: dict[str, Any], problem: Any, design: Any, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    from .sensitivity import analyze_design

    outputs = []
    for index in range(len(design)):
        values = [
            row["metrics"].get(spec["metric"])
            for row in rows
            if row["case"] == f"sample_{index}"
        ]
        if not values or any(v is None or not math.isfinite(v) for v in values):
            raise ValueError(
                "sensitivity requires a finite target metric for every "
                "design point and seed"
            )
        outputs.append(values)
    matrix = np.asarray(outputs, dtype=float)
    result = analyze_design(spec, problem, design, matrix.mean(axis=1))
    result.update(
        {
            "design": design.tolist(),
            "outputs_by_seed": matrix.tolist(),
            "analysis_target": "mean over the same declared seeds at each design point",
        }
    )
    return result


def results_csv(report: dict[str, Any]) -> str:
    stream = io.StringIO(newline="")
    columns = [
        "case",
        "seed",
        "benchmark",
        "metric",
        "value",
        "unit",
        "reference_low",
        "reference_high",
        "score",
        "validation_role",
        "skipped",
        "overrides",
    ]
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    for row in report["rows"]:
        for metric, value in (row["metrics"] or {"": None}).items():
            reference = row.get("references", {}).get(metric, {})
            low, high = reference.get("range", [None, None])
            writer.writerow(
                {
                    "case": row["case"],
                    "seed": row["seed"],
                    "benchmark": row["benchmark"],
                    "metric": metric,
                    "value": value,
                    "unit": reference.get("unit"),
                    "reference_low": low,
                    "reference_high": high,
                    "score": row["scores"].get(metric),
                    "validation_role": row["validation"]
                    .get(metric, {})
                    .get("role", "unclassified"),
                    "skipped": row["skipped"],
                    "overrides": json.dumps(row["overrides"], sort_keys=True),
                }
            )
    return stream.getvalue()


def results_svg(report: dict[str, Any]) -> str:
    """Render metric facets as portable SVG with seed confidence intervals."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for entry in report["summaries"]:
        if entry["mean"] is not None:
            groups[entry["benchmark"], entry["metric"]].append(entry)
    y, parts = 45, []
    for (benchmark, metric), entries in groups.items():
        entries = entries[:64]
        values = [v for e in entries for v in [e["mean"], *e["ci95"]] if v is not None]
        low, high = min(values), max(values)
        span = high - low or max(abs(high) * 0.1, 1.0)
        low, high = low - span * 0.1, high + span * 0.1

        def x(value: float) -> float:
            return 260 + (value - low) / (high - low) * 430

        parts.append(
            f'<text x="20" y="{y}" font-weight="bold">'
            f"{escape(benchmark + ': ' + metric)}</text>"
        )
        y += 24
        for entry in entries:
            mean, interval = entry["mean"], entry["ci95"]
            parts.append(f'<text x="20" y="{y + 4}">{escape(entry["case"])}</text>')
            if all(v is not None for v in interval):
                parts.append(
                    f'<path d="M{x(interval[0]):.2f} {y}H{x(interval[1]):.2f}" '
                    'stroke="#176e65" stroke-width="3"/>'
                )
            parts.append(
                f'<circle cx="{x(mean):.2f}" cy="{y}" r="4" fill="#173238"/>'
                f'<text x="710" y="{y + 4}">{mean:.5g} (n={entry["n"]})</text>'
            )
            y += 25
        y += 24
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="900" '
        f'height="{y + 20}" role="img">'
        "<title>Simulation means and 95% intervals across seeds</title>"
        '<rect width="100%" height="100%" fill="white"/>'
        '<g font-family="sans-serif" font-size="13">' + "".join(parts) + "</g></svg>"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--request", type=Path)
    source.add_argument("--replay", type=Path)
    parser.add_argument("--allow-source-mismatch", action="store_true")
    parser.add_argument("--allow-environment-mismatch", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("experiment-output"))
    args = parser.parse_args(argv)
    try:
        if args.replay:
            replay = load_replay(
                args.replay,
                allow_source_mismatch=args.allow_source_mismatch,
                allow_environment_mismatch=args.allow_environment_mismatch,
            )
            if replay.get("kind") != "experiment":
                raise ValueError("manifest does not contain an experiment")
            request = replay["request"]
        else:
            if args.request.stat().st_size > 256 * 1024:
                raise ValueError("request must be at most 256 KiB")
            request = json.loads(args.request.read_text(encoding="utf-8"))
        report = run_experiment(
            request, progress=lambda p: write_json(args.out / "progress.json", p)
        )
        write_bundle(
            args.out,
            report,
            replay={"kind": "experiment", "request": report["request"]},
            extra_files={
                "request.json": json.dumps(report["request"], indent=2),
                "results.csv": results_csv(report),
                "plot.svg": results_svg(report),
            },
        )
        print(str(args.out / "manifest.json"))
        return 0
    except (
        OSError,
        ValueError,
        TypeError,
        ImportError,
        AttributeError,
        RuntimeError,
    ) as exc:
        write_json(
            args.out / "progress.json",
            {"status": "failed", "error": str(exc), "message": str(exc)},
        )
        print(str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
