"""Campaign coverage, source integrity and independent evaluation decisions."""

import dataclasses
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def campaign(monkeypatch):
    path = Path(__file__).parents[1] / "examples/evaluate_driving_data.py"
    spec = importlib.util.spec_from_file_location("driving_campaign_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "environment_record", lambda: {"test": True})
    return module


def test_runtime_sources_record_import_origin_without_importing_optional_packages(
    campaign, monkeypatch, tmp_path
):
    package = tmp_path / "source_override" / "myosuite"
    package.mkdir(parents=True)
    origin = package / "__init__.py"
    origin.write_bytes(b"value = 1\r\n")
    helper = package / "helper.py"
    helper.write_bytes(b"value = 2\n")
    loaded = ModuleType("myosuite")
    loaded.__file__ = str(origin)
    monkeypatch.setitem(sys.modules, "myosuite", loaded)
    for name in ("embodied_human", "myosuite_carla", "embodied_ui"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    lookups = []

    def find_spec(name):
        lookups.append(name)
        return None

    monkeypatch.setattr(campaign.importlib.util, "find_spec", find_spec)

    def import_must_not_run(*args, **kwargs):
        raise AssertionError("source discovery imported an optional package")

    monkeypatch.setattr(campaign.importlib, "import_module", import_must_not_run)
    commands = []

    def git(command, **kwargs):
        commands.append(command)
        if "rev-parse" in command:
            return b"actual-runtime-checkout\n"
        return b""

    monkeypatch.setattr(campaign.subprocess, "check_output", git)
    sources = campaign._runtime_sources()
    actual = sources["myosuite"]
    assert actual["origin"] == str(origin.resolve())
    assert actual["git_revision"] == "actual-runtime-checkout"
    assert actual["git_dirty"] is False
    assert actual["python_files"] == {
        "__init__.py": hashlib.sha256(b"value = 1\n").hexdigest(),
        "helper.py": hashlib.sha256(b"value = 2\n").hexdigest(),
    }
    assert all(command[2] == str(package) for command in commands)
    assert "myosuite" not in lookups
    assert sources["myosuite_carla"] == {"status": "unavailable"}
    helper.write_bytes(b"value = 3\n")
    assert (
        campaign._runtime_sources()["myosuite"]["python_source_sha256"]
        != actual["python_source_sha256"]
    )


def test_runtime_sources_use_find_spec_and_tolerate_missing_git(
    campaign, monkeypatch, tmp_path
):
    origin = tmp_path / "__init__.py"
    origin.write_text("value = 1\n")
    for name in ("embodied_human", "myosuite_carla", "myosuite", "embodied_ui"):
        monkeypatch.delitem(sys.modules, name, raising=False)

    def find_spec(name):
        if name == "myosuite":
            return SimpleNamespace(origin=str(origin))
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(campaign.importlib.util, "find_spec", find_spec)

    def unavailable_git(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(campaign.subprocess, "check_output", unavailable_git)
    sources = campaign._runtime_sources()
    assert sources["myosuite"]["status"] == "resolved"
    assert sources["myosuite"]["git_revision"] is None
    assert sources["embodied_ui"]["status"] == "unavailable"


def _candidate(before=1.0, after=0.5):
    return {
        "status": "evaluated",
        "profile": {"role": "fitted_candidate", "parameters": {"gain": 1.5}},
        "evaluations": {
            "test": {
                "baseline": {"metrics": {"error": {"mae": {"value": before}}}},
                "selected": {"metrics": {"error": {"mae": {"value": after}}}},
            }
        },
    }


def _inputs(tmp_path, *, gated=True):
    entries = {}
    for name in (
        "lehtonen2018",
        "rightgaze",
        "expert_driving",
        "ngsim",
        "drivedna",
        "mpdb",
    ):
        if name == "drivedna" and gated:
            entries[name] = {
                "path": None,
                "status": "access_pending",
                "http_status": 403,
            }
        else:
            path = tmp_path / f"{name}.csv"
            path.write_text("public source fixture\n")
            entries[name] = {"path": path.name}
    path = tmp_path / "inputs.json"
    path.write_text(json.dumps(entries))
    return path


def _stub_protocols(campaign, monkeypatch, *, broken=None):
    calls = []

    def load(name, path, **options):
        calls.append(name)
        if name == broken:
            raise ValueError("required signal is missing")
        return SimpleNamespace(
            source=f"https://example.test/{name}",
            metadata={"source_files": [str(path)]},
            summary=lambda: {"dataset_id": name, "subjects": np.int64(4)},
        )

    protocols = SimpleNamespace(
        calibrate_gaze_allocation=lambda *a, **k: _candidate(),
        gaze_allocation_reference=lambda *a, **k: {"status": "reference_ready"},
        telemetry_reference=lambda *a, **k: {"status": "reference_ready"},
        emg_reference=lambda *a, **k: {"status": "reference_ready"},
    )
    original_import = campaign.importlib.import_module
    monkeypatch.setattr(
        campaign.importlib,
        "import_module",
        lambda name: (
            protocols
            if name == "embodied_human.driving_protocols"
            else original_import(name)
        ),
    )
    monkeypatch.setattr(campaign, "load_dataset", load)
    monkeypatch.setattr(campaign, "_fixation", lambda *a, **k: _candidate(after=2.0))
    monkeypatch.setattr(campaign, "_following", lambda *a, **k: _candidate())
    return calls


def test_campaign_retains_all_six_sources_and_explicit_gate(
    campaign, monkeypatch, tmp_path
):
    inputs = _inputs(tmp_path)
    calls = _stub_protocols(campaign, monkeypatch)
    destination = tmp_path / "out"
    result = campaign.run_campaign(inputs, destination, fit=True)
    assert result["status"] == "partial"
    assert set(result["datasets"]) == set(campaign.DATASETS)
    assert len(calls) == 5
    record = result["datasets"]["drivedna_controlled"]
    assert record["status"] == "access_pending"
    gated = json.loads((destination / record["report"]).read_text())
    assert gated["input"]["http_status"] == 403
    fixation = result["datasets"]["lehtonen2018"]
    assert fixation["status"] == "evaluated"
    assert fixation["model_update"]["decision"] == "rejected_after_test_regression"
    assert fixation["model_update"]["defaults_updated"] is False
    assert result["defaults_updated"] is False
    assert (
        result["input_registry"]["sha256"]
        == hashlib.sha256(inputs.read_bytes()).hexdigest()
    )
    for record in result["datasets"].values():
        assert (
            record["sha256"]
            == hashlib.sha256((destination / record["report"]).read_bytes()).hexdigest()
        )


def test_processing_error_is_preserved_and_other_datasets_continue(
    campaign, monkeypatch, tmp_path, capsys
):
    inputs = _inputs(tmp_path, gated=False)
    calls = _stub_protocols(campaign, monkeypatch, broken="expert_driving")
    code = campaign.main(
        [
            "--inputs",
            str(inputs),
            "--out",
            str(tmp_path / "result.json"),
            "--fit",
            "--allow-incomplete",
        ]
    )
    assert code == 1
    assert len(calls) == 6
    report = json.loads((tmp_path / "result.json").read_text())
    failed = report["datasets"]["expert_driving"]
    assert failed["status"] == "failed"
    assert failed["error"]["stage"] == "import"
    assert failed["error"]["type"] == "ValueError"
    assert "required signal is missing" in failed["error"]["traceback"]
    assert report["datasets"]["mpdb"]["status"] == "reference_ready"
    assert "ERROR during import" in capsys.readouterr().err


def test_partial_requires_explicit_allowance_and_remains_partial(
    campaign, monkeypatch, tmp_path
):
    inputs = _inputs(tmp_path)
    _stub_protocols(campaign, monkeypatch)
    arguments = ["--inputs", str(inputs), "--out", str(tmp_path / "out"), "--fit"]
    assert campaign.main(arguments) == 2
    assert campaign.main(arguments + ["--allow-incomplete"]) == 0
    assert (
        json.loads((tmp_path / "out/campaign.json").read_text())["status"] == "partial"
    )


def test_missing_inputs_and_checksum_failure_are_explicit(
    campaign, monkeypatch, tmp_path
):
    source = tmp_path / "source.csv"
    source.write_text("modified source")
    inputs = tmp_path / "inputs.json"
    inputs.write_text(
        json.dumps({"expert_driving": {"path": source.name, "sha256": "0" * 64}})
    )
    calls = _stub_protocols(campaign, monkeypatch)
    report = campaign.run_campaign(inputs, tmp_path / "out")
    assert not calls
    assert report["dataset_status_counts"] == {"failed": 1, "missing_input": 5}
    assert "checksum" in report["datasets"]["expert_driving"]["error"]["message"]


def test_json_conversion_omits_table_contents_and_large_arrays(campaign):
    @dataclasses.dataclass
    class Metadata:
        count: object
        values: object

    class Table:
        columns = ["observation"]
        index = range(3)

        def __len__(self):
            return 3

    result = campaign._safe(
        {
            "metadata": Metadata(np.int64(3), np.array([1, 2])),
            "table": Table(),
            "large": np.ones((100, 101)),
            "missing": np.nan,
        }
    )
    assert result["metadata"] == {"count": 3, "values": [1, 2]}
    assert result["table"] == {
        "rows": 3,
        "columns": ["observation"],
        "data_omitted": True,
    }
    assert result["large"] == {"shape": [100, 101], "data_omitted": True}
    assert result["missing"] is None


def test_following_closed_loop_uses_only_locked_test_sources(
    campaign, monkeypatch, tmp_path
):
    driver_models = ModuleType("myosuite_carla.driver_models")

    @dataclasses.dataclass
    class ReflexConfig:
        perception_delay: float = 0.02

    driver_models.ReflexDriverConfig = ReflexConfig
    following = ModuleType("myosuite_carla.empirical_following")
    following.METRICS = ("error",)
    following.FIXED_ERROR_SCALES = {"error": 1.0}
    parameters = {
        "time_headway": 1.2,
        "min_gap": 2.0,
        "max_acceleration": 1.5,
        "comfortable_deceleration": 2.0,
        "exponent": 4.0,
    }
    following.idm_parameters = lambda: parameters.copy()
    protocol = SimpleNamespace(
        observed={"train": {}, "validation": {}, "test": {}},
        metadata={"source": "fixture"},
    )
    following.load_following_protocol = lambda *a, **k: protocol
    closed_calls = []
    following.closed_loop_following = lambda *a, **k: (
        closed_calls.append(k) or {"status": "completed"}
    )
    parent = ModuleType("myosuite_carla")
    parent.__path__ = []
    monkeypatch.setitem(sys.modules, "myosuite_carla", parent)
    monkeypatch.setitem(sys.modules, "myosuite_carla.driver_models", driver_models)
    monkeypatch.setitem(sys.modules, "myosuite_carla.empirical_following", following)
    import embodied_human.driving_evaluation as evaluation

    captured = []

    def calibrate(*args, **kwargs):
        captured.append((args, kwargs))
        return {
            "status": "evaluated",
            "profile": {"parameters": parameters.copy()},
            "partitions": {
                "train": ["train"],
                "validation": ["validation"],
                "test": ["test"],
            },
        }

    monkeypatch.setattr(evaluation, "calibrate_grouped", calibrate)
    report = campaign._following(
        tmp_path / "ngsim.csv",
        fit=True,
        samples=64,
        seed=7,
        max_groups=96,
        closed_loop_trials=6,
    )
    assert closed_calls == [{"groups": ["test"], "max_trials": 6, "seed": 7}]
    assert {p.name for p in captured[0][0][2]} == {
        "time_headway",
        "min_gap",
        "max_acceleration",
        "comfortable_deceleration",
    }
    assert set(captured[0][1]["configuration"]) == {"human", "driver", "reflex"}
    assert report["profile"]["module"] == "reflex.idm"
    assert report["profile"]["context"]


def test_fixation_report_reuse_checks_protocol_and_preserves_original(
    campaign, monkeypatch, tmp_path
):
    import embodied_human.fixation_protocol as fixation
    from embodied_human.driving_evaluation import GroupSplit

    expected = SimpleNamespace(
        split=GroupSplit(("a",), ("b",), ("c",)),
        metadata={
            "source": "fixture",
            "seconds": 30,
            "cache_identity": {"schema_version": 1, "source": "synthetic cache"},
        },
        observed={g: dict.fromkeys(fixation.METRICS, 1.0) for g in ("a", "b", "c")},
    )
    monkeypatch.setattr(fixation, "FixationProtocol", lambda *a, **k: expected)
    path = tmp_path / "fit.json"
    report = {
        "status": "evaluated",
        "partitions": dataclasses.asdict(expected.split),
        "protocol": dict(expected.metadata),
        "posterior": {"simulated": 64},
        "evaluations": {
            role: {
                "baseline_predictions": {
                    g: dict.fromkeys(fixation.METRICS, 1.1)
                    for g in getattr(expected.split, role)
                },
                "selected_predictions": {
                    g: dict.fromkeys(fixation.METRICS, 1.0)
                    for g in getattr(expected.split, role)
                },
            }
            for role in ("train", "validation", "test")
        },
    }
    path.write_text(json.dumps(report))
    original = path.read_bytes()
    reused = campaign._fixation(
        None, fit=True, samples=64, seconds=30, seed=0, reuse=path
    )
    assert path.read_bytes() == original
    assert reused["reused_report"]["sha256"] == hashlib.sha256(original).hexdigest()
    report["protocol"]["source"] = "different source"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="protocol settings"):
        campaign._fixation(None, fit=True, samples=64, seconds=30, seed=0, reuse=path)


def test_registry_collisions_and_output_overwrite_are_rejected(campaign, tmp_path):
    path = tmp_path / "inputs.json"
    path.write_text(json.dumps({"rightgaze": {}, "rightgaze2022": {}}))
    with pytest.raises(ValueError, match="Duplicate"):
        campaign.run_campaign(path, tmp_path / "out")
    path.write_text("{}")
    with pytest.raises(ValueError, match="overwrite"):
        campaign.run_campaign(path, path)


def _fixation_cache_bundle():
    pandas = pytest.importorskip("pandas")
    from embodied_human.driving_data import DatasetBundle

    rows = []
    for subject in range(6):
        for condition, target in (("down1", (8, -12)), ("down2", (26, -24))):
            for glance in range(5):
                rows.append(
                    {
                        "subject_id": str(subject),
                        "trial_id": f"{subject}:{condition}",
                        "condition": condition,
                        "area": "target",
                        "boundary": False,
                        "duration_s": 0.5 + subject / 10 + glance / 20,
                        "speed_mps": 24.0,
                        "heading_deg": target[0],
                        "pitch_deg": target[1],
                    }
                )
    return DatasetBundle(
        "fixture",
        "real_road",
        "CC BY 4.0",
        "fixture source",
        glances=pandas.DataFrame(rows),
        trials=pandas.DataFrame([{}]),
    )


@pytest.mark.parametrize("change", ["observations", "contexts", "code", "missing"])
def test_fixation_cache_rejects_changed_observations_contexts_or_code(
    campaign, monkeypatch, tmp_path, change
):
    import embodied_human.driving_evaluation as evaluation
    import embodied_human.fixation_protocol as fixation

    bundle = _fixation_cache_bundle()
    protocol = fixation.FixationProtocol(bundle, seconds=30, seed=0)
    report = {
        "status": "evaluated",
        "partitions": dataclasses.asdict(protocol.split),
        "protocol": protocol.metadata,
        "posterior": {"simulated": 64},
    }
    if change == "missing":
        report["protocol"].pop("cache_identity")
    path = tmp_path / "cached.json"
    path.write_text(json.dumps(report))
    before = path.read_bytes()
    if change == "observations":
        glances = bundle.glances.copy()
        glances["duration_s"] *= 2
        bundle = dataclasses.replace(bundle, glances=glances)
    elif change == "contexts":
        glances = bundle.glances.copy()
        glances["speed_mps"] *= 0.5
        bundle = dataclasses.replace(bundle, glances=glances)
    elif change == "code":
        monkeypatch.setattr(fixation, "_model_source_sha256", lambda: "f" * 64)

    def augmentation_must_not_run(*args, **kwargs):
        raise AssertionError("A stale cache reached report augmentation")

    monkeypatch.setattr(
        evaluation, "compare_prediction_change", augmentation_must_not_run
    )
    with pytest.raises(ValueError, match="cache identity"):
        campaign._fixation(bundle, fit=True, samples=64, seconds=30, seed=0, reuse=path)
    assert path.read_bytes() == before


def test_model_source_identity_normalizes_newlines_and_retains_relative_paths(tmp_path):
    from embodied_human.fixation_protocol import _model_source_sha256

    path = tmp_path / "model.py"
    path.write_bytes(b"value = 1\n")
    first = _model_source_sha256(tmp_path)
    path.write_bytes(b"value = 1\r\n")
    assert _model_source_sha256(tmp_path) == first
    (tmp_path / "other.py").write_bytes(b"value = 1\n")
    assert _model_source_sha256(tmp_path) != first


def test_completed_closed_loop_regression_rejects_candidate_with_paired_source_errors(
    campaign,
):
    import copy

    report = _candidate(before=3.0, after=1.0)
    episodes = []
    for label in ("baseline", "fitted"):
        for source, trial, speed in (("a", 0, 1.0), ("a", 1, 9.0), ("b", 0, 100.0)):
            episodes.append(
                {
                    "profile": label,
                    "source_group": source,
                    "trial_index": trial,
                    "seed": trial,
                    "status": "completed",
                    "collision": False,
                    "off_road": False,
                    "coverage_fraction": 1.0,
                    "speed_rmse_mps": speed + (label == "fitted"),
                    "gap_rmse_m": 2.0,
                    "acceleration_rmse_mps2": 1.0,
                }
            )
    report["closed_loop"] = {
        "status": "completed",
        "selected_trials": 3,
        "selected_source_groups": ["a", "b"],
        "episodes": episodes,
    }
    snapshot = json.dumps(report, sort_keys=True)
    review = campaign._promotion_review(report)
    assert review["decision"] == "rejected_after_closed_loop_regression"
    assert review["closed_loop_regressions"] == [
        {"metric": "speed_rmse_mps", "baseline": 52.5, "candidate": 53.5}
    ]
    change = review["paired_source_comparison"]["metrics"]["speed_rmse_mps"]
    assert change["change_in_mae"] == {"value": 1.0, "interval": [1.0, 1.0]}
    assert change["paired_groups"] == 2
    assert review["defaults_updated"] is False
    assert review["test_used_for_retuning"] is False
    assert json.dumps(report, sort_keys=True) == snapshot
    incomplete = copy.deepcopy(report)
    incomplete["closed_loop"]["episodes"][-1]["coverage_fraction"] = 0.9
    assert (
        campaign._promotion_review(incomplete)["decision"]
        == "physical_evaluation_incomplete"
    )
    missing_pair = copy.deepcopy(report)
    missing_pair["closed_loop"]["episodes"].pop()
    assert (
        campaign._promotion_review(missing_pair)["decision"]
        == "physical_evaluation_incomplete"
    )
