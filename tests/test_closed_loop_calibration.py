"""Physical calibration preserves independent sources and complete test pairs."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from embodied_human import closed_loop_calibration as calibration


def _episode(label, source, trial, error, seed=117):
    return {
        "profile": label,
        "source_group": source,
        "trial_index": trial,
        "base_seed": seed,
        "seed": seed + trial,
        "status": "completed",
        "collision": False,
        "off_road": False,
        "surrogate_closed": True,
        "coverage_fraction": 1.0,
        "full_source_horizon": True,
        **{name: error * scale for name, scale in calibration.SCALES.items()},
    }


def _report(baseline=2.0, candidate=1.0):
    return {
        "status": "evaluated",
        "profile": {
            "role": "fitted_candidate",
            "test_used_for_parameter_selection": False,
        },
        "partitions": {"test": ["a", "b"]},
        "design": {"trial_counts": {"a": 1, "b": 1}},
        "closed_loop": {
            "status": "completed",
            "episodes": [
                _episode(label, source, 0, error)
                for label, error in (("baseline", baseline), ("fitted", candidate))
                for source in ("a", "b")
            ],
        },
    }


def _configuration(**settings):
    return calibration.ClosedLoopCalibrationConfig(
        test_sources=2, bootstrap=200, **settings
    )


def test_primary_review_averages_trials_before_squaring_and_keeps_sources_equal():
    report = _report()
    report["design"]["trial_counts"]["a"] = 2
    report["closed_loop"]["episodes"] = [
        _episode(label, source, trial, error)
        for label, values in (
            ("baseline", (("a", 0, 1.0), ("a", 1, 3.0), ("b", 0, 1.0))),
            ("fitted", (("a", 0, 0.5), ("a", 1, 1.5), ("b", 0, 0.5))),
        )
        for source, trial, error in values
    ]
    before = copy.deepcopy(report)
    review = calibration.review_closed_loop(report, _configuration(trials_per_source=2))
    assert review["decision"] == "accepted_contextual_candidate"
    change = review["paired_source_comparison"]["metrics"][calibration.PRIMARY]
    assert change["change_in_mae"]["value"] == pytest.approx(-1.875)
    assert change["paired_groups"] == 2
    assert review["source_primary_errors"]["baseline"]["a"][calibration.PRIMARY] == 4.0
    assert review["defaults_updated"] is False
    assert report == before


@pytest.mark.parametrize(
    "problem",
    [
        "collision",
        "off_road",
        "cleanup",
        "coverage",
        "horizon",
        "missing",
        "duplicate",
        "seed",
        "selection",
        "role",
        "lost_source",
    ],
)
def test_incomplete_or_unsafe_pairs_cannot_be_accepted(problem):
    report = _report()
    row = report["closed_loop"]["episodes"][-1]
    if problem in ("collision", "off_road"):
        row[problem] = True
    elif problem == "cleanup":
        row["surrogate_closed"] = False
    elif problem == "coverage":
        row["coverage_fraction"] = 0.9
    elif problem == "horizon":
        row["full_source_horizon"] = False
    elif problem == "missing":
        report["closed_loop"]["episodes"].pop()
    elif problem == "duplicate":
        report["closed_loop"]["episodes"].append(dict(row))
    elif problem == "seed":
        row["seed"] += 1
    elif problem == "selection":
        report["profile"]["test_used_for_parameter_selection"] = True
    elif problem == "role":
        report["profile"]["role"] = "unselected"
    else:
        report["partitions"]["test"] = ["a"]
        report["closed_loop"]["episodes"] = [
            item
            for item in report["closed_loop"]["episodes"]
            if item["source_group"] == "a"
        ]
    review = calibration.review_closed_loop(report, _configuration())
    assert review["decision"] == "rejected_incomplete_physical_evaluation"
    assert "paired_source_comparison" not in review


def test_primary_point_improvement_requires_a_negative_upper_bootstrap_bound():
    report = _report()
    for row in report["closed_loop"]["episodes"]:
        if row["source_group"] == "b":
            error = 1.0 if row["profile"] == "baseline" else 2**0.5
            row.update(
                {name: error * scale for name, scale in calibration.SCALES.items()}
            )
    review = calibration.review_closed_loop(report, _configuration())
    assert review["decision"] == "inconclusive_contextual_candidate"
    change = review["paired_source_comparison"]["metrics"][calibration.PRIMARY][
        "change_in_mae"
    ]
    assert change["value"] < 0 < change["interval"][1]
    assert (
        calibration.review_closed_loop(_report(1, 2), _configuration())["decision"]
        == "rejected_after_test_regression"
    )
    report["profile"]["role"] = "retained_baseline"
    assert (
        calibration.review_closed_loop(report, _configuration())["decision"]
        == "retained_baseline"
    )


def test_one_source_smoke_run_cannot_accept_a_contextual_profile():
    report = _report()
    report["partitions"]["test"] = ["a"]
    report["closed_loop"]["episodes"] = [
        row for row in report["closed_loop"]["episodes"] if row["source_group"] == "a"
    ]
    config = calibration.ClosedLoopCalibrationConfig(test_sources=1)
    assert (
        calibration.review_closed_loop(report, config)["decision"]
        == "insufficient_independent_test_sources"
    )


def test_exclusions_recover_all_previous_partitions_and_physical_request_groups(
    tmp_path,
):
    grouped = tmp_path / "grouped.json"
    grouped.write_text(
        json.dumps(
            {
                "evaluation": {
                    "partitions": {
                        "train": ["ngsim:a"],
                        "validation": ["ngsim:b"],
                        "test": ["ngsim:c"],
                    },
                    "protocol": {"source_groups": ["ngsim:a", "ngsim:b", "ngsim:c"]},
                }
            }
        )
    )
    physical = tmp_path / "physical.json"
    physical.write_text(json.dumps({"requested_source_groups": ["ngsim:d", "other:e"]}))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"datasets": {"ngsim": {"report": "grouped.json"}}}))
    assert calibration.source_exclusions([manifest, physical, grouped]) == (
        "ngsim:a",
        "ngsim:b",
        "ngsim:c",
        "ngsim:d",
    )
    empty = tmp_path / "empty.json"
    empty.write_text("{}")
    with pytest.raises(ValueError, match="no source identities"):
        calibration.source_exclusions([grouped, empty])


@pytest.mark.parametrize(
    "settings",
    [
        {"samples": True},
        {"bootstrap": 0},
        {"test_sources": 0},
        {"fit_seed": 117},
        {"seed": -1},
        {"seconds": float("nan")},
        {"seconds": 0.1},
        {"accept": 1.1},
        {"desired_speed": True},
    ],
)
def test_invalid_experiment_settings_are_rejected(settings):
    with pytest.raises(ValueError):
        calibration.ClosedLoopCalibrationConfig(**settings)


def test_orchestration_freezes_fresh_sources_before_fitting_and_locks_test(
    tmp_path, monkeypatch
):
    old = tmp_path / "old.json"
    old.write_text(json.dumps({"requested_source_groups": ["ngsim:old"]}))
    groups = tuple(f"ngsim:{index}" for index in range(15))
    log = []
    calls = []

    @dataclass
    class Reflex:
        perception_delay: float = 0.02

    @dataclass
    class PhysicalConfig:
        seconds: float = 20.0
        trials_per_source: int = 1

    class Physical:
        def __init__(self, source, *, config):
            assert log[0]["event"] == "design_frozen"
            self.groups = source.groups
            self.config = config
            self.closed = False

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.closed = True

        @property
        def observed(self):
            return {group: dict.fromkeys(calibration.METRICS, 0.0) for group in groups}

        @property
        def metadata(self):
            return {"source_groups": list(groups), "configuration": asdict(self.config)}

        @property
        def diagnostics(self):
            return {"closed": self.closed, "episode_runs": len(calls)}

        @property
        def episodes(self):
            return []

        def __call__(self, parameters, requested, seed):
            calls.append((dict(parameters), requested, seed))
            error = 0.1 + (parameters["time_headway"] - 1.0) ** 2
            return {
                group: {
                    name: error * scale for name, scale in calibration.SCALES.items()
                }
                for group in requested
            }

        def report_pair(self, baseline, fitted, *, groups, seed):
            return {
                "status": "completed",
                "episodes": [
                    _episode(
                        label,
                        group,
                        0,
                        0.1 + (parameters["time_headway"] - 1.0) ** 2,
                        seed,
                    )
                    for label, parameters in (
                        ("baseline", baseline),
                        ("fitted", fitted),
                    )
                    for group in groups
                ],
            }

    def load(source, **settings):
        assert settings["excluded_groups"] == ("ngsim:old",)
        assert settings["max_groups"] == 15
        return SimpleNamespace(groups=groups, trials_for=lambda group: (object(),))

    package = ModuleType("myosuite_carla")
    package.__path__ = []
    bridge = ModuleType("myosuite_carla.empirical_following")
    bridge.ClosedLoopConfig = PhysicalConfig
    bridge.ClosedLoopFollowingProtocol = Physical
    bridge.load_following_protocol = load
    bridge.idm_parameters = lambda: {
        "time_headway": 1.6,
        "min_gap": 2.0,
        "max_acceleration": 1.0,
        "comfortable_deceleration": 1.5,
        "exponent": 4.0,
    }
    models = ModuleType("myosuite_carla.driver_models")
    models.ReflexDriverConfig = Reflex
    monkeypatch.setitem(sys.modules, "myosuite_carla", package)
    monkeypatch.setitem(sys.modules, bridge.__name__, bridge)
    monkeypatch.setitem(sys.modules, models.__name__, models)
    config = calibration.ClosedLoopCalibrationConfig(bootstrap=200)
    report = calibration.calibrate_closed_loop_following(
        tmp_path / "unused.csv",
        config=config,
        exclude_reports=[old],
        progress=log.append,
    )
    partition = report["partitions"]
    assert [len(partition[name]) for name in ("train", "validation", "test")] == [
        6,
        3,
        6,
    ]
    assert not set(partition["train"]) & set(partition["test"])
    assert report["diagnostics"]["closed"] is True
    test_sources = set(partition["test"])
    test_calls = [
        index
        for index, (_, requested, _) in enumerate(calls)
        if set(requested) & test_sources
    ]
    assert test_calls == [len(calls) - 2, len(calls) - 1]
    assert all(seed == 117 for _, _, seed in calls[-2:])
    assert all(
        set(requested) == set(partition["train"]) and seed == 17
        for _, requested, seed in calls[:16]
    )
    assert report["model_update"]["decision"] == "accepted_contextual_candidate"
    assert report["profile"]["module"] == "reflex.idm"
    assert report["defaults_updated"] is False
    assert log[-1]["event"] == "objective_completed"


@pytest.mark.parametrize("changed", [False, True])
def test_cli_persists_design_and_refuses_results_if_runtime_source_changes(
    tmp_path, monkeypatch, changed
):
    path = Path(__file__).parents[1] / "examples/calibrate_closed_loop_following.py"
    spec = importlib.util.spec_from_file_location("closed_loop_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / "source.csv"
    source.write_text("source")
    out = tmp_path / "out"
    origins = [
        {"model": {"python_source_sha256": "first"}},
        {"model": {"python_source_sha256": "changed" if changed else "first"}},
    ]
    monkeypatch.setattr(module, "runtime_sources", lambda: origins.pop(0))
    monkeypatch.setattr(module, "environment_record", lambda: {})

    def run(source, *, config, exclude_reports, progress):
        progress({"event": "design_frozen", "design": {"fixed": True}})
        assert json.loads((out / "design.json").read_text())["fixed"] is True
        return {
            "status": "evaluated",
            "model_update": {"decision": "retained_baseline"},
        }

    monkeypatch.setattr(module, "calibrate_closed_loop_following", run)
    assert module.main(["--source", str(source), "--out", str(out)]) == (
        1 if changed else 0
    )
    assert (out / "fullreport.json").exists() is not changed
    assert (out / "progress.jsonl").exists()
    provenance = json.loads((out / "provenance.json").read_text())
    assert provenance["source_identity_verified"] is not changed
    before = (out / "provenance.json").read_bytes()
    assert module.main(["--source", str(source), "--out", str(out)]) == 1
    assert (out / "provenance.json").read_bytes() == before
