"""Explicit task protocols must remain separate from published benchmarks."""

import numpy as np
import pytest

import embodied_human as eh
from embodied_human.bench.benchmarks import BENCHMARKS
from embodied_human.bench.protocol import benchmark_seed
from embodied_human.experiment_protocols import CATALOG, PROTOCOLS


def test_protocols_do_not_change_the_published_benchmark_registry():
    assert set(PROTOCOLS) == {row["name"] for row in CATALOG}
    assert not set(PROTOCOLS) & set(BENCHMARKS)


def test_baseline_serial_position_protocol_is_seeded_and_truthfully_unscored():
    def run():
        with benchmark_seed(7):
            return PROTOCOLS["memory.serial_recall"](eh.build_human(seed=7))

    first, second = run(), run()
    assert first.as_dict() == second.as_dict()
    assert not first.scores and not first.passed
    assert first.metrics["memory.serial_recall_intrusion_rate"] == 0
    assert len(first.details["accuracy_by_serial_position"]) == 6
    assert all(row["kind"] == "assumed" for row in first.details["protocol_sources"])
    assert "human evaluation pending" in first.details["interpretation"]


def test_native_memory_protocol_scores_identity_intrusions():
    pytest.importorskip("pyactup")
    config = eh.HumanConfig().replace(
        "memory",
        "pyactup",
        capacity=0,
        noise=0,
        threshold=-5,
        mismatch_penalty=0.1,
    )
    with benchmark_seed(9):
        result = PROTOCOLS["memory.serial_recall"](eh.build_human(config, seed=9))
    assert result.skipped is None
    assert result.metrics["memory.serial_recall_intrusion_rate"] == 1
    assert result.metrics["memory.serial_recall_accuracy"] == 0
    assert all(probe["intrusion"] for row in result.details["probes"] for probe in row)


def test_choice_protocol_explicitly_skips_the_detection_only_baseline():
    result = PROTOCOLS["response.choice"](eh.build_human(seed=0))
    assert result.skipped and "response=pyddm" in result.skipped
    assert not result.metrics


def test_native_choice_protocol_reports_solver_predictions_and_repeatable_draws():
    pytest.importorskip("pyddm")
    config = eh.HumanConfig().replace("response", "pyddm")

    def run():
        with benchmark_seed(5):
            return PROTOCOLS["response.choice"](eh.build_human(config, seed=5))

    first, second = run(), run()
    assert first.as_dict() == second.as_dict()
    assert first.skipped is None and not first.scores
    rows = first.details["conditions"]
    assert len(rows) == 6
    assert np.mean([row["accuracy"] for row in rows]) == pytest.approx(
        np.mean([row["predicted_accuracy"] for row in rows]), abs=0.025
    )
    assert all(row["predicted_mean_rt_s"] is not None for row in rows)
    assert all(row["kind"] == "assumed" for row in first.details["protocol_sources"])
