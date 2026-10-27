"""Regression gates retain empirical failures and detect lost coverage."""

import json
from pathlib import Path

import pytest

from embodied_human.bench.comparison import compare_reports, markdown_summary


def report(value=0.5, *, name="fixture.protocol"):
    # Synthetic numerical range, with no human data claim.
    distance = max(-value, value - 1, 0) if value is not None else 1
    metric = "fixture.value"
    return {
        "seed": 0,
        "config": {},
        "results": [
            {
                "name": name,
                "metrics": {metric: value},
                "scores": {metric: max(0, 1 - distance)},
                "skipped": None,
                "references": {
                    metric: {
                        "range": [0.0, 1.0],
                        "unit": "fixture unit",
                        "value": 0.5,
                        "citation": "Synthetic test range",
                        "conditions": "assumed fixture",
                    }
                },
                "validation": {
                    metric: {
                        "role": "consistency",
                        "evaluation_data": ["fixture"],
                        "fitting_data": ["fixture"],
                    }
                },
            }
        ],
    }


def test_stored_baseline_retains_all_known_empirical_failures():
    path = Path(__file__).resolve().parents[1] / "evidence/benchmark-baseline.json"
    baseline = json.loads(path.read_text(encoding="utf-8"))
    result = compare_reports(baseline, baseline)
    assert not result["regressions"] and not result["incomparable"]
    assert result["counts"]["comparable_metrics"] == 53
    summary = markdown_summary(baseline, result)
    assert "Pass: 17. Fail: 11." in summary
    assert "Held out | 38 | 19" in summary


def test_zero_floor_does_not_hide_a_worsening_existing_failure():
    baseline, current = report(2.5), report(3.5)
    assert baseline["results"][0]["scores"] == current["results"][0]["scores"]
    result = compare_reports(current, baseline)
    row = result["metric_changes"][0]
    assert row["baseline_distance"] == 1.5
    assert row["current_distance"] == 2.5
    assert row["status"] == "regressed"
    assert len(result["regressions"]) == 1


@pytest.mark.parametrize(
    "before,after,status",
    [
        (0.2, 0.8, "changed"),
        (2.5, 2.5, "unchanged"),
        (2.5, 1.5, "improved"),
    ],
)
def test_changes_inside_range_and_improved_failures_do_not_regress(
    before, after, status
):
    result = compare_reports(report(after), report(before))
    assert result["metric_changes"][0]["status"] == status
    assert not result["regressions"]


def test_tolerance_is_applied_to_unclipped_reference_error():
    assert not compare_reports(report(2 + 5e-10), report(2))["regressions"]
    assert compare_reports(report(2 + 2e-9), report(2))["regressions"]


@pytest.mark.parametrize("mutation", ["unit", "range", "conditions", "role", "data"])
def test_changed_reference_or_data_use_requires_review(mutation):
    baseline, current = report(), report()
    row = current["results"][0]
    reference = row["references"]["fixture.value"]
    declaration = row["validation"]["fixture.value"]
    if mutation == "unit":
        reference["unit"] = "other units"
    elif mutation == "range":
        reference["range"] = [0, 2]
    elif mutation == "conditions":
        reference["conditions"] = "different task"
    elif mutation == "role":
        declaration["role"] = "unclassified"
    else:
        declaration["evaluation_data"] = ["different data"]
    result = compare_reports(current, baseline)
    assert result["metric_changes"][0]["status"] == "incomparable"
    assert result["metric_changes"][0]["delta"] is None
    assert len(result["incomparable"]) == 1


def test_configuration_changes_are_visible_for_intentional_model_comparisons():
    baseline, current = report(), report()
    baseline["resolved_config"] = {"memory": {"name": "old", "params": {}}}
    current["resolved_config"] = {"memory": {"name": "new", "params": {}}}
    result = compare_reports(current, baseline)
    assert result["configuration_changes"][0]["category"] == "memory"
    assert not result["regressions"] and not result["incomparable"]


@pytest.mark.parametrize("score", [None, "remove"])
def test_lost_score_is_a_regression(score):
    current = report()
    current["results"][0]["scores"] = (
        {} if score == "remove" else {"fixture.value": None}
    )
    assert compare_reports(current, report())["regressions"]


def test_removed_metrics_and_undefined_measurements_are_detected():
    baseline, current = report(), report()
    for key in ("metrics", "scores", "references", "validation"):
        current["results"][0][key] = {}
    assert (
        compare_reports(current, baseline)["metric_changes"][0]["status"] == "removed"
    )
    assert compare_reports(report(None), baseline)["regressions"]


def test_added_protocols_are_recorded_without_claiming_regression():
    baseline, current = report(), report()
    current["results"] += report(name="new.protocol")["results"]
    result = compare_reports(current, baseline)
    assert result["added_benchmarks"] == ["new.protocol"]
    assert result["counts"]["added_metrics"] == 1
    assert not result["regressions"]


def test_removed_protocols_and_explicit_subset_selection():
    baseline = report()
    baseline["results"] += report(name="other.protocol")["results"]
    current = report()
    assert compare_reports(current, baseline)["removed_benchmarks"] == [
        "other.protocol"
    ]
    subset = compare_reports(current, baseline, names=["fixture.protocol"])
    assert not subset["removed_benchmarks"] and not subset["regressions"]


def test_new_skip_is_a_regression_and_recovered_execution_is_recorded():
    baseline, skipped = report(), report()
    skipped["results"][0].update(
        skipped="optional dependency missing", metrics={}, scores={}
    )
    result = compare_reports(skipped, baseline)
    assert result["regressions"][0]["reason"].endswith("now skipped")
    assert "optional dependency missing" in markdown_summary(skipped, result)
    assert not compare_reports(baseline, skipped)["regressions"]


@pytest.mark.parametrize(
    "mutation", ["seed", "duplicate", "empty", "bad_range", "schema"]
)
def test_incompatible_or_ambiguous_reports_are_rejected(mutation):
    baseline, current = report(), report()
    if mutation == "seed":
        current["seed"] = 1
    elif mutation == "duplicate":
        current["results"] *= 2
    elif mutation == "empty":
        current["results"] = []
    elif mutation == "schema":
        current["schema_version"] = 2
    else:
        current["results"][0]["references"]["fixture.value"]["range"] = [1, 0]
    with pytest.raises(ValueError):
        compare_reports(current, baseline)


def test_summary_escapes_labels_and_handles_unscored_or_empty_reports():
    current = report(name="<unsafe>|name")
    current["results"][0]["scores"] = {}
    summary = markdown_summary(current)
    assert "&lt;unsafe&gt;\\|name" in summary
    assert "Unscored: 1" in summary
    assert "Protocols: 0" in markdown_summary({"seed": 0, "results": []})
    current["comparison_error"] = "reports must use the same seed"
    assert "Comparison unavailable: reports must use the same seed" in markdown_summary(
        current
    )


@pytest.mark.parametrize("tolerance", [-1, float("nan"), float("inf"), True])
def test_invalid_tolerance_cannot_disable_the_gate(tolerance):
    with pytest.raises(ValueError):
        compare_reports(report(), report(), tolerance=tolerance)
