"""Independent source partitions and parameter recovery from empirical summaries."""

import json
import math

import numpy as np
import pytest

from embodied_human.driving_evaluation import (
    GroupSplit,
    aggregate_summary,
    calibrate_grouped,
    compare_prediction_change,
    compare_predictions,
    group_split,
)
from embodied_human.inference import Prior


def test_partition_preserves_sources_independent_of_input_order_and_replication():
    groups = [f"study:p{i}" for i in range(12)]
    first = group_split(groups, seed=7)
    assert first == group_split(list(reversed(groups)) + groups, seed=7)
    assert set(first.train).isdisjoint(first.validation)
    assert set(first.train).isdisjoint(first.test)
    assert set(first.validation).isdisjoint(first.test)
    assert set(first.train + first.validation + first.test) == set(groups)
    with pytest.raises(ValueError, match="more than one partition"):
        GroupSplit(("same",), ("other",), ("same",))
    with pytest.raises(ValueError, match="three independent"):
        group_split(["same", "same", "other"])


def test_source_weighting_and_bootstrap_coverage_are_explicit():
    observed = {"study:a": {"speed": 0.0}, "study:b": {"speed": 0.0}}
    predicted = {"study:a": {"speed": 2.0}, "study:b": {"speed": 6.0}}
    aggregate = aggregate_summary(predicted, ["speed"])
    assert aggregate["metrics"]["speed"]["mean"] == 4.0
    report = compare_predictions(observed, predicted, ["speed"], bootstrap=99, seed=5)
    metric = report["metrics"]["speed"]
    assert metric["mae"]["value"] == 4.0
    assert metric["rmse"]["value"] == pytest.approx(math.sqrt(20))
    assert metric["mae"]["interval"] == [2.0, 6.0]
    assert report["resampling_unit"] == "original_source_group"
    assert metric["source_groups"] == ["study:a", "study:b"]
    assert report == compare_predictions(
        observed, predicted, ["speed"], bootstrap=99, seed=5
    )


def test_missing_predictions_do_not_make_a_complete_report():
    observed = {"a": {"x": None}, "b": {"x": 2.0}, "c": {"x": 3.0}}
    predicted = {"a": {"x": 0.0}, "b": {"x": float("nan")}, "c": {"x": 3.5}}
    report = compare_predictions(observed, predicted, ["x"])
    metric = report["metrics"]["x"]
    assert report["status"] == "incomplete_evaluation"
    assert metric["missing_observation_groups"] == 1
    assert metric["missing_prediction_groups"] == 1
    assert metric["paired_groups"] == 1
    assert metric["mae"] == {"value": 0.5, "interval": None}
    json.dumps(report, allow_nan=False)


def test_large_finite_measurements_do_not_overflow_comparison():
    observed = {"a": {"x": 0.0}, "b": {"x": 0.0}}
    predicted = {"a": {"x": 1e300}, "b": {"x": 1e300}}
    report = compare_predictions(observed, predicted, ["x"], bootstrap=10)
    assert report["metrics"]["x"]["rmse"]["value"] == pytest.approx(1e300)
    json.dumps(report, allow_nan=False)


def test_paired_change_uses_matched_sources_with_heterogeneous_baseline_errors():
    observed = {source: {"x": 0.0} for source in "abc"}
    baseline = {
        source: {"x": value} for source, value in zip("abc", [1, 1000, 1000000])
    }
    selected = {source: {"x": row["x"] + 1} for source, row in baseline.items()}
    result = compare_prediction_change(
        observed, baseline, selected, ["x"], bootstrap=199, seed=5
    )
    metric = result["metrics"]["x"]
    assert metric["change_in_mae"] == {"value": 1.0, "interval": [1.0, 1.0]}
    assert metric["paired_groups"] == 3
    assert "mae" not in metric and "rmse" not in metric and "bias" not in metric
    assert result["negative_change_favors"] == "selected"
    del baseline["b"]
    partial = compare_prediction_change(
        observed, baseline, selected, ["x"], bootstrap=0
    )
    assert partial["status"] == "incomplete_evaluation"
    assert partial["metrics"]["x"]["paired_groups"] == 2


def test_grouped_calibration_recovers_known_parameter_and_preserves_configuration():
    groups = [f"data:p{i}" for i in range(12)]
    observed = {g: {"response": 1.7 * (i + 1) / 10} for i, g in enumerate(groups)}
    contexts = {g: (i + 1) / 10 for i, g in enumerate(groups)}
    calls = []

    def simulate(parameters, sources, seed):
        calls.append((sources, seed))
        rng = np.random.default_rng(seed)
        return {
            source: {
                "response": parameters["gain"] * contexts[source]
                + rng.normal(0, 0.0001)
            }
            for source in sources
        }

    configuration = {
        "human": {"attention": {"name": "baseline"}},
        "driver": {},
        "reflex": {},
    }
    report = calibrate_grouped(
        observed,
        simulate,
        [Prior("gain", 0.1, 3.0)],
        {"gain": 0.8},
        metrics=["response"],
        configuration=configuration,
        provenance={"dataset": "synthetic parameter recovery"},
        samples=128,
        accept=0.1,
        bootstrap=50,
        seed=3,
    )
    assert report["profile"]["role"] == "fitted_candidate"
    assert report["profile"]["parameters"]["gain"] == pytest.approx(1.7, abs=0.03)
    test = report["evaluations"]["test"]
    assert test["selected"]["metrics"]["response"]["mae"]["value"] < 0.02
    assert test["baseline"]["metrics"]["response"]["mae"]["value"] > 0.3
    assert test["paired_change"]["metrics"]["response"]["change_in_mae"]["value"] < 0
    test_groups = set(report["partitions"]["test"])
    assert all(seed >= 100 for sources, seed in calls if test_groups & set(sources))
    assert report["profile"]["test_used_for_parameter_selection"] is False
    configuration["human"]["attention"]["name"] = "mutated"
    assert (
        report["profile"]["configuration"]["human"]["attention"]["name"] == "baseline"
    )
    json.dumps(report, allow_nan=False)


def test_test_outcomes_cannot_change_selected_parameters_or_scales():
    partition = GroupSplit(("a", "b", "c"), ("d",), ("e", "f"))
    observed = {source: {"response": 1.8} for source in "abcdef"}

    def simulate(parameters, sources, seed):
        return {source: {"response": parameters["gain"]} for source in sources}

    kwargs = {
        "metrics": ["response"],
        "split": partition,
        "scales": {"response": 1.0},
        "configuration": {"human": {}, "driver": {}, "reflex": {}},
        "provenance": {"dataset": "synthetic holdout corruption"},
        "samples": 32,
        "accept": 0.2,
        "bootstrap": 0,
    }
    before = calibrate_grouped(
        observed, simulate, [Prior("gain", 0.0, 3.0)], {"gain": 1.0}, **kwargs
    )
    observed["e"]["response"] = 900.0
    observed["f"]["response"] = -900.0
    after = calibrate_grouped(
        observed, simulate, [Prior("gain", 0.0, 3.0)], {"gain": 1.0}, **kwargs
    )
    assert before["profile"] == after["profile"]
    assert before["validation_candidates"] == after["validation_candidates"]
    assert before["posterior"] == after["posterior"]
    assert before["evaluations"]["test"] != after["evaluations"]["test"]


def test_invalid_simulator_partition_and_seed_reuse_fail_before_scoring():
    observed = {source: {"x": float(i)} for i, source in enumerate("abcdef")}
    kwargs = {
        "metrics": ["x"],
        "configuration": {"human": {}, "driver": {}, "reflex": {}},
        "provenance": {},
        "samples": 4,
    }
    with pytest.raises(ValueError, match="outside its requested partition"):
        calibrate_grouped(
            observed,
            lambda p, g, s: observed,
            [Prior("x", 0.0, 1.0)],
            {"x": 0.5},
            **kwargs,
        )
    with pytest.raises(ValueError, match="distinct nonnegative"):
        calibrate_grouped(
            observed,
            lambda p, g, s: {},
            [Prior("x", 0.0, 1.0)],
            {"x": 0.5},
            fit_seeds=(1,),
            evaluation_seeds=(1,),
            **kwargs,
        )


def test_validation_can_retain_a_better_baseline_and_failures_are_reported():
    observed = {source: {"x": 0.0} for source in "abcdef"}

    def simulate(parameters, sources, seed):
        if parameters["x"] > 0.8:
            raise RuntimeError("invalid physical trajectory")
        return {source: {"x": parameters["x"]} for source in sources}

    report = calibrate_grouped(
        observed,
        simulate,
        [Prior("x", 0.1, 1.0)],
        {"x": 0.0},
        metrics=["x"],
        scales={"x": 1.0},
        configuration={"human": {}, "driver": {}, "reflex": {}},
        provenance={"dataset": "synthetic exact baseline"},
        samples=16,
        accept=0.25,
        bootstrap=0,
    )
    assert report["profile"]["role"] == "retained_baseline"
    assert report["profile"]["parameters"] == {"x": 0.0}
    assert report["simulation_failures"]
    assert None in report["all_training_distances"]
    json.dumps(report, allow_nan=False)
