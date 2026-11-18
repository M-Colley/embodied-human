"""Regression checks for paired external site analysis."""

from pathlib import Path
from runpy import run_path

import pytest

pytest.importorskip("scipy")
pytest.importorskip("myosuite_carla")


def _example():
    path = Path(__file__).parents[1] / "examples/evaluate_headway_transfer.py"
    return run_path(str(path))["test_decision"]


def _inputs():
    return {
        "track_groups": ["a", "b", "c", "d"],
        "base_seeds": [721, 722],
        "run_by_group": {"a": "1", "b": "1", "c": "1", "d": "2"},
        "design": {
            "test_minimum_complete_tracks": 3,
            "maximum_speed_rmse_increase_mps": 0.05,
            "test_minimum_run_groups": 2,
            "bootstrap_resamples": 1000,
            "bootstrap_seed": 172,
        },
    }


def _rows(inputs, gaps, bias):
    return {
        (group, seed): {
            "seed": seed,
            "gap_rmse_m": gap,
            "signed_gap_error_m": bias,
            "speed_rmse_mps": 0.8,
            "acceleration_rmse_mps2": 0.4,
        }
        for gap, group in zip(gaps, inputs["track_groups"], strict=True)
        for seed in inputs["base_seeds"]
    }


def test_runs_receive_equal_weight_despite_unequal_track_counts():
    inputs = _inputs()
    reference = _rows(inputs, [1, 1, 1, 5], 0.5)
    candidate = _rows(inputs, [1, 1, 1, 4], 0.4)
    report = _example()({"reference": reference, "candidate": candidate}, inputs)
    assert report["conditional_metrics"]["reference"]["gap_rmse_m"] == 3.0
    assert report["conditional_metrics"]["candidate"]["gap_rmse_m"] == 2.5
    assert report["paired_gap_rmse_change_m"]["mean"] == -0.5


def test_candidate_incomplete_episode_prevents_adoption():
    inputs = _inputs()
    reference = _rows(inputs, [5, 5, 5, 5], 1.0)
    candidate = _rows(inputs, [3, 3, 3, 3], 0.5)
    del candidate["a", 721]
    report = _example()({"reference": reference, "candidate": candidate}, inputs)
    assert report["guards"]["paired_gap_rmse_reduction"]
    assert report["guards"]["complete_track_count"]
    assert not report["guards"]["candidate_complete"]
    assert report["decision"] == "rejected"


def test_pilot_coverage_cannot_pass_even_with_positive_performance():
    inputs = {**_inputs(), "protocol_eligible": False}
    reference = _rows(inputs, [5, 5, 5, 5], 1.0)
    candidate = _rows(inputs, [3, 3, 3, 3], 0.5)
    report = _example()({"reference": reference, "candidate": candidate}, inputs)
    assert report["guards"]["candidate_complete"]
    assert report["guards"]["paired_gap_rmse_reduction"]
    assert not report["guards"]["protocol_coverage"]
    assert report["decision"] == "descriptive_pilot"
