"""Seeded comparisons, bounded designs and replayable worker outputs."""

import csv
import io
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pytest

from embodied_human import HumanConfig, build_human
from embodied_human.bench.benchmarks import run
from embodied_human.bench.protocol import benchmark_seed, protocol_seed
from embodied_human.experiments import (
    experiment_catalog,
    results_csv,
    results_svg,
    run_experiment,
    validate_request,
)


def request(**changes):
    config = HumanConfig().replace("eyes", "main_sequence", latency_median=0.4)
    return {
        "version": 1,
        "config": config.as_dict(),
        "benchmarks": ["eyes.latency"],
        "seeds": [7, 8, 9],
        "compare_baseline": True,
        "sweep": None,
        **changes,
    }


def test_catalog_lists_benchmarks_and_declared_metrics():
    catalog = experiment_catalog()
    latency = next(
        item for item in catalog["benchmarks"] if item["name"] == "eyes.latency"
    )
    assert latency["metrics"] == ["eyes.latency_median"]
    assert "eyes" in latency["categories"]
    assert catalog["limits"]["seeds"] == 20


def test_comparison_uses_matched_seeds_and_propagates_them_to_stochastic_protocols():
    spec = request()
    progress = []
    report = run_experiment(spec, progress=progress.append)
    repeated = run_experiment(spec)
    assert report["rows"] == repeated["rows"]
    assert report["summaries"] == repeated["summaries"]
    assert len(report["rows"]) == 6
    baseline = {
        row["seed"]: row["metrics"]["eyes.latency_median"]
        for row in report["rows"]
        if row["case"] == "baseline"
    }
    changed = {
        row["seed"]: row["metrics"]["eyes.latency_median"]
        for row in report["rows"]
        if row["case"] == "configured"
    }
    assert set(baseline) == set(changed) == {7, 8, 9}
    assert len(set(baseline.values())) == 3
    for seed in baseline:
        # Identical Gaussian draws and a doubled median imply exactly doubled RT.
        assert changed[seed] == pytest.approx(2 * baseline[seed])
    difference = report["paired_differences"][0]
    assert difference["n"] == 3
    assert difference["mean"] == pytest.approx(
        np.mean([changed[seed] - baseline[seed] for seed in baseline])
    )
    assert progress[-1]["status"] == "complete"
    assert progress[-1]["completed"] == progress[-1]["total"] == 6
    assert (
        report["resolved_cases"]["configured"]["resolved_config"]["eyes"]["params"][
            "latency_median"
        ]
        == 0.4
    )
    assert report["validation_summary"]["consistency"]["metrics"] == 6
    assert "simulation seeds" in report["uncertainty"]


def test_identical_comparison_has_zero_difference_and_explicit_undefined_interval():
    report = run_experiment(request(config=HumanConfig().as_dict()))
    difference = report["paired_differences"][0]
    assert difference["mean"] == 0.0
    assert difference["n"] == 3
    assert difference["ci95"] == [None, None]
    single = run_experiment(request(seeds=[7], compare_baseline=False))
    assert single["summaries"][0]["n"] == 1
    assert single["summaries"][0]["ci95"] == [None, None]


def test_sweep_changes_one_parameter_preserves_seeds_and_records_values():
    report = run_experiment(
        request(
            seeds=[11],
            compare_baseline=False,
            sweep={
                "category": "eyes",
                "parameter": "latency_median",
                "values": [0.1, 0.3],
            },
        )
    )
    first, second = report["rows"]
    assert [first["case"], second["case"]] == ["sweep_0", "sweep_1"]
    assert first["seed"] == second["seed"] == 11
    assert first["overrides"] == {"eyes.latency_median": 0.1}
    assert second["overrides"] == {"eyes.latency_median": 0.3}
    assert second["metrics"]["eyes.latency_median"] == pytest.approx(
        3 * first["metrics"]["eyes.latency_median"]
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"seeds": [1, 1]},
        {"seeds": [True]},
        {"seeds": [-1]},
        {"seeds": [1.5]},
        {"seeds": list(range(21))},
        {"benchmarks": ["eyes.latency", "eyes.latency"]},
        {"benchmarks": ["unregistered"]},
        {"compare_baseline": 1},
        {"sweep": {"category": "memory", "parameter": "capacity", "values": [1, 1]}},
        {"sweep": {"category": "memory", "parameter": "capacity", "values": [1.5]}},
    ],
)
def test_invalid_experiments_are_rejected_before_execution(changes):
    with pytest.raises((ValueError, TypeError)):
        validate_request(request(**changes))


def test_benchmark_budget_rejects_large_sweep_before_running():
    with pytest.raises(ValueError, match="evaluations"):
        validate_request(
            request(
                seeds=list(range(20)),
                benchmarks=[
                    "eyes.latency",
                    "eyes.main_sequence",
                    "memory.span",
                    "vision.foveation",
                ],
                sweep={
                    "category": "memory",
                    "parameter": "capacity",
                    "values": list(range(1, 33)),
                },
            )
        )


def test_cancellation_callback_stops_between_benchmark_runs():
    progress = []
    with pytest.raises(InterruptedError, match="cancelled"):
        run_experiment(
            request(),
            progress=progress.append,
            cancelled=lambda: bool(progress),
        )
    assert len(progress) == 1
    assert progress[0]["status"] == "running"


def test_explicit_seed_makes_benchmarks_independent_of_selection_order():
    names = ["eyes.latency", "eyes.main_sequence"]
    forward = run(build_human(seed=0), names, seed=15)
    reverse = run(build_human(seed=100), list(reversed(names)), seed=15)
    assert {result.name: result.metrics for result in forward} == {
        result.name: result.metrics for result in reverse
    }
    assert protocol_seed() == 0
    with pytest.raises(RuntimeError), benchmark_seed(7):
        assert protocol_seed() == 7
        assert protocol_seed(2) == protocol_seed(2)
        assert protocol_seed(2) != protocol_seed(3)
        raise RuntimeError("test restoration")
    assert protocol_seed() == 0


def test_csv_and_svg_represent_actual_results_and_escape_labels():
    report = run_experiment(request(seeds=[7], compare_baseline=False))
    rows = list(csv.DictReader(io.StringIO(results_csv(report))))
    assert len(rows) == 1
    assert rows[0]["validation_role"] == "consistency"
    assert rows[0]["unit"] == "ms"
    assert float(rows[0]["reference_low"]) == 150
    assert float(rows[0]["reference_high"]) == 250
    reference = report["rows"][0]["references"]["eyes.latency_median"]
    assert reference["conditions"] and reference["citation"]
    assert (
        float(rows[0]["value"]) == report["rows"][0]["metrics"]["eyes.latency_median"]
    )
    report["summaries"][0]["case"] = "<unsafe & label>"
    svg = results_svg(report)
    assert "&lt;unsafe &amp; label&gt;" in svg
    assert ET.fromstring(svg).tag == "{http://www.w3.org/2000/svg}svg"


def test_worker_writes_complete_bundle_and_replay_cli_preserves_seeded_outputs(
    tmp_path,
):
    spec = request(seeds=[7], compare_baseline=False)
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(spec), encoding="utf-8")
    original, replayed = tmp_path / "original", tmp_path / "replayed"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "embodied_human.experiments",
            "--request",
            str(request_path),
            "--out",
            str(original),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    expected = {
        "report.json",
        "manifest.json",
        "references.json",
        "environment.json",
        "REPLAY.txt",
        "request.json",
        "results.csv",
        "plot.svg",
        "bundle.zip",
    }
    assert expected <= {path.name for path in original.iterdir()}
    with zipfile.ZipFile(original / "bundle.zip") as archive:
        assert expected - {"bundle.zip"} <= set(archive.namelist())
    assert ET.fromstring((original / "plot.svg").read_text()).tag.endswith("svg")
    # Other agents can edit package source between subprocesses during development.
    subprocess.run(
        [
            sys.executable,
            "-m",
            "embodied_human.experiments",
            "--replay",
            str(original / "manifest.json"),
            "--allow-source-mismatch",
            "--out",
            str(replayed),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    initial = json.loads((original / "report.json").read_text())
    repeated = json.loads((replayed / "report.json").read_text())
    assert initial["rows"] == repeated["rows"]
    assert initial["summaries"] == repeated["summaries"]
