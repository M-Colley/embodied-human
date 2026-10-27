"""Benchmark inventory and comparison gates preserve scientific coverage."""

import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from embodied_human.bench.__main__ import main
from embodied_human.bench.benchmarks import BENCHMARKS, Result, benchmark
from embodied_human.bench.catalog import benchmark_names, protocol_catalog
from embodied_human.reporting import sha256


def test_inventory_profiles_do_not_construct_a_human(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("inventory constructed a human")

    monkeypatch.setattr("embodied_human.bench.__main__.build_human", forbidden)
    assert main(["--list", "--profile", "core"]) == 0
    core = capsys.readouterr().out.splitlines()
    assert core == benchmark_names("core")
    assert main(["--list", "--profile", "full"]) == 0
    full = capsys.readouterr().out.splitlines()
    assert full == benchmark_names("full")
    assert len(core) == 21 and len(full) == 28
    assert set(core) < set(full)
    assert set(full) - set(core) == {
        "driving.surt_glances", "driving.takeover", "motor.touch_tapping",
        "design.touch_key_size", "vestibular.vertical_sickness",
        "vestibular.sickness_time_course", "vestibular.roll_compensation",
    }


def test_packaged_catalog_matches_the_reviewed_baseline_coverage():
    catalog = protocol_catalog()
    baseline = json.loads(
        (Path(__file__).parents[1] / "evidence" / "benchmark-baseline.json").read_text(
            encoding="utf-8"
        )
    )
    results = {result["name"]: result for result in baseline["results"]}
    assert set(results) == set(catalog)
    for name, declaration in catalog.items():
        assert set(declaration["metrics"]) == set(results[name]["metrics"])
        assert set(declaration["categories"]) == set(results[name]["categories"])


def test_new_packaged_protocol_requires_metadata(monkeypatch):
    def new_protocol(human):
        return Result("new.example", ("memory",))

    new_protocol.__module__ = "embodied_human.bench.example"
    monkeypatch.setitem(BENCHMARKS, "new.example", new_protocol)
    with pytest.raises(ValueError, match="missing.*new.example"):
        protocol_catalog()


@pytest.mark.parametrize("change", ["profile", "category", "metric", "stale"])
def test_catalog_rejects_incomplete_or_unknown_declarations(monkeypatch, change):
    catalog = protocol_catalog()
    if change == "profile":
        del catalog["eyes.latency"]["ci_profile"]
    elif change == "category":
        catalog["eyes.latency"]["categories"] = ["unregistered"]
    elif change == "metric":
        catalog["eyes.latency"]["metrics"] = ["unreferenced.metric"]
    else:
        catalog["unregistered.protocol"] = catalog["eyes.latency"]
    resource = SimpleNamespace(read_text=lambda: json.dumps(catalog))
    monkeypatch.setattr(
        "embodied_human.bench.catalog.files",
        lambda package: SimpleNamespace(joinpath=lambda name: resource),
    )
    with pytest.raises(ValueError):
        protocol_catalog()


def test_duplicate_protocol_names_are_rejected():
    protocol_catalog()
    original = BENCHMARKS["eyes.latency"]
    with pytest.raises(ValueError, match="already registered"):
        benchmark("eyes.latency")(lambda human: Result("eyes.latency", ("eyes",)))
    assert BENCHMARKS["eyes.latency"] is original


def baseline(tmp_path, *extra):
    path = tmp_path / "baseline.json"
    assert main(["--only", "eyes.latency", "--out", str(path), *extra]) == 0
    return path


@pytest.mark.parametrize("external_summary", [True, False])
def test_regression_writes_report_summary_and_bundle_before_failing(
    tmp_path, external_summary
):
    previous = baseline(tmp_path)
    report_path = tmp_path / "current.json"
    summary_path = tmp_path / "summary" / "results.md"
    bundle_path = tmp_path / "bundle"
    args = [
        "--only", "eyes.latency", "--replace", "eyes=main_sequence:latency_median=0.4",
        "--compare", str(previous), "--fail-on-regression", "--out", str(report_path),
        "--bundle", str(bundle_path),
    ]
    if external_summary:
        args.extend(["--summary", str(summary_path)])
    code = main(args)
    assert code == 1
    report = json.loads(report_path.read_text())
    assert report["comparison"]["regressions"]
    assert report["comparison"]["configuration_changes"]
    summary = (bundle_path / "summary.md").read_bytes()
    assert summary.strip()
    if external_summary:
        assert summary_path.read_bytes() == summary
    assert (bundle_path / "bundle.zip").is_file()
    assert json.loads((bundle_path / "report.json").read_text()) == report
    archived_baseline = (bundle_path / "comparison-baseline.json").read_bytes()
    assert archived_baseline == previous.read_bytes()
    manifest = json.loads((bundle_path / "manifest.json").read_text())
    with zipfile.ZipFile(bundle_path / "bundle.zip") as archive:
        for name in ("summary.md", "comparison-baseline.json"):
            content = archive.read(name)
            assert content == (bundle_path / name).read_bytes()
            assert manifest["artifacts"][name] == sha256(content)


def test_summary_without_comparison_is_included_in_bundle(tmp_path):
    bundle = tmp_path / "bundle"
    summary = tmp_path / "summary.md"
    assert main([
        "--only", "eyes.latency", "--summary", str(summary), "--bundle", str(bundle),
    ]) == 0
    manifest = json.loads((bundle / "manifest.json").read_text())
    assert manifest["artifacts"]["summary.md"] == sha256(summary.read_bytes())
    assert "comparison-baseline.json" not in manifest["artifacts"]
    assert manifest["replay"]["benchmarks"] == ["eyes.latency"]


def test_existing_reference_failure_remains_successful_without_a_regression(tmp_path):
    replacement = ["--replace", "eyes=main_sequence:latency_median=0.4"]
    previous = baseline(tmp_path, *replacement)
    assert not json.loads(previous.read_text())["results"][0]["passed"]
    assert main([
        "--only", "eyes.latency", *replacement,
        "--compare", str(previous), "--fail-on-regression",
    ]) == 0


def test_explicit_tolerance_applies_to_the_regression_gate(tmp_path):
    previous = baseline(tmp_path)
    assert main([
        "--only", "eyes.latency", "--replace", "eyes=main_sequence:latency_median=0.4",
        "--compare", str(previous), "--fail-on-regression",
        "--regression-tolerance", "100",
    ]) == 0


def test_seed_mismatch_preserves_raw_outputs_and_reports_an_input_error(tmp_path):
    previous = baseline(tmp_path)
    current = tmp_path / "current.json"
    summary = tmp_path / "summary.md"
    bundle = tmp_path / "invalid-comparison-bundle"
    with pytest.raises(SystemExit) as error:
        main([
            "--only", "eyes.latency", "--seed", "1", "--compare", str(previous),
            "--fail-on-regression", "--out", str(current), "--summary", str(summary),
            "--bundle", str(bundle),
        ])
    assert error.value.code == 2
    assert json.loads(current.read_text())["seed"] == 1
    report = json.loads(current.read_text())
    assert "comparison" not in report
    assert "same seed" in report["comparison_error"]
    assert report["comparison_error"] in summary.read_text(encoding="utf-8")
    assert (bundle / "comparison-baseline.json").read_bytes() == previous.read_bytes()
    assert json.loads((bundle / "report.json").read_text()) == report


@pytest.mark.parametrize("contents", ["{}", "{broken json", "[]"])
def test_malformed_baselines_preserve_current_outputs(tmp_path, contents):
    previous = tmp_path / "invalid.json"
    previous.write_text(contents, encoding="utf-8")
    current = tmp_path / "current.json"
    with pytest.raises(SystemExit) as error:
        main([
            "--only", "eyes.latency", "--compare", str(previous),
            "--fail-on-regression", "--out", str(current),
        ])
    assert error.value.code == 2
    assert json.loads(current.read_text())["results"][0]["name"] == "eyes.latency"


def test_explicit_selection_compares_only_selected_baseline_protocols():
    previous = Path(__file__).parents[1] / "evidence" / "benchmark-baseline.json"
    assert main([
        "--only", "eyes.latency", "--compare", str(previous), "--fail-on-regression",
    ]) == 0


def test_profile_comparison_detects_a_missing_profile_protocol(tmp_path, monkeypatch):
    previous = Path(__file__).parents[1] / "evidence" / "benchmark-baseline.json"
    report = json.loads(previous.read_text(encoding="utf-8"))
    names = set(benchmark_names("core"))
    report["results"] = [row for row in report["results"] if row["name"] in names]
    previous = tmp_path / "core-baseline.json"
    previous.write_text(json.dumps(report), encoding="utf-8")
    results = [
        Result(
            row["name"], tuple(row["categories"]), metrics=row["metrics"],
            scores=row["scores"], details=row["details"], skipped=row["skipped"],
            validation=row["validation"],
        )
        for row in report["results"] if row["name"] != "eyes.latency"
    ]
    monkeypatch.setattr(
        "embodied_human.bench.benchmarks.run", lambda *args, **kwargs: results
    )
    output = tmp_path / "current.json"
    assert main([
        "--profile", "core", "--compare", str(previous), "--fail-on-regression",
        "--out", str(output),
    ]) == 1
    comparison = json.loads(output.read_text())["comparison"]
    assert "eyes.latency" in comparison["removed_benchmarks"]


def test_skips_write_outputs_and_fail_only_when_requested(tmp_path, monkeypatch):
    monkeypatch.setitem(
        BENCHMARKS, "optional.skipped",
        lambda human: Result("optional.skipped", ("memory",), skipped="missing extra"),
    )
    report = tmp_path / "skipped.json"
    summary = tmp_path / "skipped.md"
    args = [
        "--only", "optional.skipped", "--out", str(report), "--summary", str(summary),
    ]
    assert main(args) == 0
    assert main([*args, "--fail-on-skip"]) == 1
    assert json.loads(report.read_text())["results"][0]["skipped"] == "missing extra"
    assert summary.read_text(encoding="utf-8").strip()


@pytest.mark.parametrize("value", ["-1", "nan", "inf"])
def test_regression_tolerance_requires_a_finite_nonnegative_value(value):
    with pytest.raises(SystemExit) as error:
        main(["--regression-tolerance", value])
    assert error.value.code == 2


@pytest.mark.parametrize("args", [
    ["--fail-on-regression"],
    ["--profile", "core", "--only", "eyes.latency"],
    ["--only", "--compare", "baseline.json"],
    ["--list", "--out", "results.json"],
])
def test_invalid_gate_and_selection_combinations_are_rejected(args):
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2


def test_packaged_protocol_outside_bench_namespace_requires_metadata(monkeypatch):
    from embodied_human.bench.catalog import protocol_catalog

    def custom(human):
        return Result("extra.protocol", ("memory",))

    custom.__module__ = "embodied_human.extra_benchmarks"
    monkeypatch.setitem(BENCHMARKS, "extra.protocol", custom)
    with pytest.raises(ValueError, match="missing.*extra.protocol"):
        protocol_catalog()
