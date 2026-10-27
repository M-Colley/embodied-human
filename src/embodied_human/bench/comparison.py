"""Compare seeded evaluations while retaining known empirical discrepancies."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping
from html import escape
from typing import Any

from ..reporting import canonical_json, sha256
from .validation import validation_summary


def _number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("metric values and scores must be numbers or null")
    return float(value) if math.isfinite(value) else None


def _results(report: Any, label: str) -> dict[str, dict[str, Any]]:
    if not isinstance(report, dict) or not isinstance(report.get("results"), list):
        raise ValueError(f"{label} must be a benchmark report with results")
    if "schema_version" in report and (
        type(report["schema_version"]) is not int or report["schema_version"] != 1
    ):
        raise ValueError(f"{label} has an unsupported report schema")
    if not report["results"]:
        raise ValueError(f"{label} has no benchmark results to compare")
    rows = {}
    for row in report["results"]:
        if not isinstance(row, dict) or not isinstance(row.get("name"), str):
            raise ValueError(f"{label} has a result without a benchmark name")
        name = row["name"]
        if not name or name in rows:
            raise ValueError(f"{label} contains an empty or duplicate benchmark name")
        for field in ("metrics", "scores", "references", "validation"):
            if not isinstance(row.get(field, {}), Mapping):
                raise ValueError(f"{label} {name}: {field} must be a mapping")
        metrics, scores = row.get("metrics", {}), row.get("scores", {})
        if set(scores) - set(metrics):
            raise ValueError(f"{label} {name}: scores need corresponding metrics")
        for value in metrics.values():
            _number(value)
        for value in scores.values():
            score = _number(value)
            if score is not None and not 0 <= score <= 1:
                raise ValueError(f"{label} {name}: scores must be within zero and one")
        for reference in row.get("references", {}).values():
            _reference(reference)
        for metric in metrics:
            _declaration(row, metric)
        rows[name] = row
    return rows


def _status(row: Mapping[str, Any]) -> str:
    if row.get("skipped"):
        return "skipped"
    scores = row.get("scores", {})
    if not scores:
        return "unscored"
    return "pass" if all(score == 1 for score in scores.values()) else "fail"


def _reference(reference: Any) -> tuple[Any, ...] | None:
    if reference is None:
        return None
    if not isinstance(reference, Mapping):
        raise ValueError("metric references must be mappings")
    bounds = reference.get("range")
    if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
        raise ValueError("metric references need two acceptance bounds")
    low, high = map(_number, bounds)
    if low is None or high is None or low > high:
        raise ValueError("reference bounds must be finite and ordered")
    return (
        low,
        high,
        *(
            reference.get(key)
            for key in (
                "value",
                "unit",
                "citation",
                "doi",
                "conditions",
            )
        ),
    )


def _declaration(row: Mapping[str, Any], metric: str) -> tuple[Any, ...]:
    declaration = row.get("validation", {}).get(metric, {})
    if not isinstance(declaration, Mapping):
        raise ValueError("metric validation declarations must be mappings")
    if declaration.get("role", "unclassified") not in (
        "consistency",
        "calibration",
        "held_out",
        "unclassified",
    ):
        raise ValueError("unsupported validation role")
    datasets = []
    for field in ("evaluation_data", "fitting_data"):
        values = declaration.get(field, ())
        if not isinstance(values, (list, tuple)) or not all(
            isinstance(value, str) for value in values
        ):
            raise ValueError(
                "validation dataset identifiers must be sequences of names"
            )
        datasets.append(tuple(sorted(values)))
    return (declaration.get("role", "unclassified"), *datasets)


def _distance(value: float, reference: Mapping[str, Any]) -> float:
    low, high = reference["range"]
    width = high - low
    distance = max(low - value, value - high, 0.0) / max(width, 1e-12)
    if not math.isfinite(width) or not math.isfinite(distance):
        raise ValueError(
            "normalized reference error exceeds the finite comparison range"
        )
    return distance


def compare_reports(
    current: dict[str, Any],
    baseline: dict[str, Any],
    *,
    names: list[str] | None = None,
    tolerance: float = 1e-9,
) -> dict[str, Any]:
    """Compare reference error, coverage and declarations for matched protocols.

    Tolerance is in acceptance range widths. Scores remain descriptive because
    their zero floor hides error increases beyond one range width. Configuration
    changes are recorded so an explicit replacement can be compared with baseline.
    """
    if isinstance(tolerance, bool) or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("regression tolerance must be finite and nonnegative")
    old, new = _results(baseline, "baseline"), _results(current, "current report")
    for report in (baseline, current):
        seed = report.get("seed")
        if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
            raise ValueError("comparison reports need an integer seed within uint32")
    if baseline["seed"] != current["seed"]:
        raise ValueError("comparison reports must use the same seed")
    if names is not None:
        if not names or len(set(names)) != len(names):
            raise ValueError(
                "comparison selection must contain distinct benchmark names"
            )
        if set(names) - (old.keys() | new.keys()):
            raise ValueError("comparison selection contains unknown benchmarks")
        old = {name: old[name] for name in names if name in old}
        new = {name: new[name] for name in names if name in new}
    if not new:
        raise ValueError("comparison has no current benchmark results")

    result: dict[str, Any] = {
        "schema_version": 1,
        "baseline_report_sha256": sha256(canonical_json(baseline).encode()),
        "baseline_hash_encoding": "canonical_json",
        "baseline_package": baseline.get("package"),
        "current_package": current.get("package"),
        "seed": current["seed"],
        "regression_tolerance": tolerance,
        "distance_definition": "distance outside acceptance range divided by its width",
        "regressions": [],
        "incomparable": [],
        "metric_changes": [],
        "added_benchmarks": sorted(new.keys() - old.keys()),
        "removed_benchmarks": sorted(old.keys() - new.keys()),
        "configuration_changes": [],
    }
    old_config = baseline.get("resolved_config", baseline.get("config", {}))
    new_config = current.get("resolved_config", current.get("config", {}))
    if not isinstance(old_config, Mapping) or not isinstance(new_config, Mapping):
        raise ValueError("report configurations must be mappings")
    for category in sorted(old_config.keys() | new_config.keys()):
        if old_config.get(category) != new_config.get(category):
            result["configuration_changes"].append(
                {
                    "category": category,
                    "baseline": old_config.get(category),
                    "current": new_config.get(category),
                }
            )
    for name in result["removed_benchmarks"]:
        result["regressions"].append(
            {
                "benchmark": name,
                "metric": None,
                "reason": "benchmark removed",
            }
        )
    comparable = 0
    for name in sorted(new):
        before, after = old.get(name, {}), new[name]
        if before and after.get("skipped") and not before.get("skipped"):
            result["regressions"].append(
                {
                    "benchmark": name,
                    "metric": None,
                    "reason": "previously executed benchmark is now skipped",
                }
            )
        previous = before.get("metrics", {}) if before else {}
        measured = after.get("metrics", {})
        for metric in sorted(previous.keys() | measured.keys()):
            old_value = _number(previous.get(metric))
            new_value = _number(measured.get(metric))
            reference = after.get("references", {}).get(metric)
            old_reference = before.get("references", {}).get(metric) if before else None
            change = {
                "benchmark": name,
                "metric": metric,
                "baseline": old_value,
                "current": new_value,
                "delta": new_value - old_value
                if new_value is not None and old_value is not None
                else None,
                "unit": reference.get("unit", "") if reference else "",
                "baseline_unit": old_reference.get("unit", "") if old_reference else "",
                "validation_role": _declaration(after, metric)[0],
                "baseline_score": before.get("scores", {}).get(metric)
                if before
                else None,
                "current_score": after.get("scores", {}).get(metric),
                "baseline_distance": None,
                "current_distance": None,
                "status": "unchanged" if new_value == old_value else "changed",
            }
            result["metric_changes"].append(change)
            issue = {"benchmark": name, "metric": metric}
            if after.get("skipped"):
                change["status"] = "skipped"
            elif metric not in previous:
                change["status"] = "added"
            elif metric not in measured:
                change["status"] = "removed"
                result["regressions"].append({**issue, "reason": "metric removed"})
            elif new_value is None and old_value is not None:
                change["status"] = "regressed"
                result["regressions"].append(
                    {**issue, "reason": "finite result became undefined"}
                )
            elif _reference(reference) != _reference(old_reference) or _declaration(
                after, metric
            ) != _declaration(before, metric):
                change["status"] = "incomparable"
                change["delta"] = None
                result["incomparable"].append(
                    {
                        **issue,
                        "reason": "reference or validation declaration changed",
                    }
                )
            elif _number(before.get("scores", {}).get(metric)) is not None and (
                _number(after.get("scores", {}).get(metric)) is None
            ):
                change["status"] = "regressed"
                result["regressions"].append(
                    {**issue, "reason": "metric score removed"}
                )
            elif metric in after.get("scores", {}) and reference is None:
                change["status"] = "incomparable"
                result["incomparable"].append(
                    {**issue, "reason": "scored metric lacks a reference"}
                )
            elif old_value is not None and new_value is not None:
                comparable += 1
                if reference is not None:
                    old_distance = _distance(old_value, reference)
                    new_distance = _distance(new_value, reference)
                    change["baseline_distance"] = old_distance
                    change["current_distance"] = new_distance
                    if new_distance > old_distance + tolerance:
                        change["status"] = "regressed"
                        result["regressions"].append(
                            {
                                **issue,
                                "reason": "distance from acceptance range increased",
                            }
                        )
                    elif new_distance + tolerance < old_distance:
                        change["status"] = "improved"
    changes = result["metric_changes"]
    result["counts"] = {
        "matched_benchmarks": len(old.keys() & new.keys()),
        "comparable_metrics": comparable,
        "changed_metrics": sum(row["baseline"] != row["current"] for row in changes),
        "added_metrics": sum(row["status"] == "added" for row in changes),
        "removed_metrics": sum(row["status"] == "removed" for row in changes),
        "regressions": len(result["regressions"]),
        "incomparable": len(result["incomparable"]),
    }
    return result


def _cell(value: Any) -> str:
    return (
        escape(str(value), quote=False)
        .replace("|", "\\|")
        .replace("\n", " ")
        .replace("\r", " ")
    )


def _formatted(value: Any) -> str:
    if value is None:
        return "undefined"
    return f"{value:.6g}" if isinstance(value, (int, float)) else _cell(value)


def markdown_summary(
    report: dict[str, Any], comparison: dict[str, Any] | None = None
) -> str:
    """Produce a review summary with scientific roles kept separate."""
    rows = report.get("results", [])
    counts = Counter(_status(row) for row in rows)
    lines = [
        "# Benchmark evaluation",
        "",
        f"Seed: {report.get('seed')}. Protocols: {len(rows)}. "
        f"Pass: {counts['pass']}. Fail: {counts['fail']}. "
        f"Skipped: {counts['skipped']}. Unscored: {counts['unscored']}.",
        "",
        "Acceptance results describe the declared published comparisons. "
        "Regression gates compare software outcomes with the selected baseline.",
        "",
        "| Evaluation role | Scored metric evaluations | Within range |",
        "| --- | ---: | ---: |",
    ]
    roles = report.get("validation_summary", validation_summary(rows))
    for role, label in (
        ("consistency", "Consistency"),
        ("calibration", "Calibration"),
        ("held_out", "Held out"),
        ("unclassified", "Unclassified"),
    ):
        totals = roles.get(role, {})
        lines.append(
            f"| {label} | {totals.get('metrics', 0)} | {totals.get('passed', 0)} |"
        )
    lines.extend(["", "Counts can contain repeated references across protocols.", ""])
    if report.get("comparison_error"):
        lines.extend(
            ["Comparison unavailable: " + _cell(report["comparison_error"]) + ".", ""]
        )
    if comparison is not None:
        totals = comparison["counts"]
        lines.extend(
            [
                "## Baseline comparison",
                "",
                f"Regressions: {totals['regressions']}. "
                f"Incomparable: {totals['incomparable']}. "
                f"Comparable metrics: {totals['comparable_metrics']}. "
                f"Numerically changed metrics: {totals['changed_metrics']}.",
                "",
                "Reference error is measured in acceptance range widths. "
                "Regression tolerance: "
                f"{_formatted(comparison['regression_tolerance'])}. "
                "Existing discrepancies remain visible.",
                "",
            ]
        )
        if comparison["configuration_changes"]:
            categories = ", ".join(
                _cell(row["category"]) for row in comparison["configuration_changes"]
            )
            lines.extend([f"Changed configurations: {categories}.", ""])
        for key, label in (
            ("added_benchmarks", "Added protocols"),
            ("removed_benchmarks", "Removed protocols"),
        ):
            if comparison[key]:
                lines.extend(
                    [f"{label}: {', '.join(map(_cell, comparison[key]))}.", ""]
                )
        for issue in [*comparison["regressions"], *comparison["incomparable"]]:
            identifier = issue["benchmark"] + (
                " / " + issue["metric"] if issue["metric"] else ""
            )
            lines.append(f"* {_cell(identifier)}: {_cell(issue['reason'])}.")
        lines.extend(
            [
                "",
                "| Protocol / metric | Baseline | Current | Delta "
                "| Unit | Role | Change |",
                "| --- | ---: | ---: | ---: | --- | --- | --- |",
            ]
        )
        for row in comparison["metric_changes"]:
            unit = row["unit"]
            if row["baseline_unit"] != unit:
                unit = f"baseline {row['baseline_unit']}; current {unit}"
            cells = (
                _cell(row["benchmark"] + " / " + row["metric"]),
                *(_formatted(row[key]) for key in ("baseline", "current", "delta")),
                _cell(unit),
                _cell(row["validation_role"]),
                row["status"],
            )
            lines.append("| " + " | ".join(cells) + " |")
    else:
        lines.extend(
            [
                "| Protocol | Status | Metric | Value | Unit | Role |",
                "| --- | --- | --- | ---: | --- | --- |",
            ]
        )
        for row in rows:
            for metric, value in (row.get("metrics") or {"": None}).items():
                reference = row.get("references", {}).get(metric, {})
                cells = (
                    _cell(row["name"]),
                    _status(row),
                    _cell(metric),
                    _formatted(value),
                    _cell(reference.get("unit", "")),
                    _cell(_declaration(row, metric)[0]),
                )
                lines.append("| " + " | ".join(cells) + " |")
    for row in rows:
        if row.get("skipped"):
            lines.extend(
                ["", f"Skipped {_cell(row['name'])}: {_cell(row['skipped'])}."]
            )
    return "\n".join(lines) + "\n"
