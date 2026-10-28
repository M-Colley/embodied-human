"""Fixed actuator arms preserve fresh test sources and common seed pairs."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from embodied_human import actuation_evaluation as study


def _episode(group, base_seed, value, *, seed=None):
    return {
        "source_group": group,
        "trial_index": 0,
        "base_seed": base_seed,
        "seed": base_seed if seed is None else seed,
        "status": "completed",
        "collision": False,
        "off_road": False,
        "surrogate_closed": True,
        "full_source_horizon": True,
        "coverage_fraction": 1.0,
        **{name: value * scale for name, scale in study.SCALES.items()},
    }


def _result(groups, seeds, value):
    return {
        "status": "completed",
        "episodes": [_episode(g, s, value) for g in groups for s in seeds],
    }


def test_seed_rmse_is_averaged_before_primary_squaring():
    config = study.ActuationEvaluationConfig(test_sources=2, bootstrap=200)
    groups = ("ngsim:a", "ngsim:b")
    baseline = _result(groups, config.test_seeds, 2.0)
    selected = _result(groups, config.test_seeds, 1.0)
    for row in selected["episodes"]:
        if row["source_group"] == "ngsim:a":
            value = 0.0 if row["base_seed"] == config.test_seeds[0] else 2.0
            row.update({name: value * scale for name, scale in study.SCALES.items()})
    review = study.review_actuation(baseline, selected, groups, config)
    assert review["decision"] == "accepted_contextual_candidate"
    assert review["source_primary_errors"]["fitted"]["ngsim:a"][study.PRIMARY] == 1.0
    assert review["complete_pairs"] == 4
    assert (
        review["paired_source_comparison"]["metrics"][study.PRIMARY]["change_in_mae"][
            "value"
        ]
        == -3.0
    )
    assert review["defaults_updated"] is False


@pytest.mark.parametrize(
    "problem",
    [
        "missing_seed",
        "collision",
        "short",
        "duplicate",
        "wrong_pair_seed",
        "foreign_source",
    ],
)
def test_incomplete_or_unmatched_test_pairs_cannot_accept(problem):
    config = study.ActuationEvaluationConfig(test_sources=2, bootstrap=100)
    groups = ("ngsim:a", "ngsim:b")
    before = _result(groups, config.test_seeds, 2.0)
    after = _result(groups, config.test_seeds, 1.0)
    if problem == "missing_seed":
        after["episodes"].pop()
    elif problem == "collision":
        after["episodes"][0]["collision"] = True
    elif problem == "short":
        after["episodes"][0]["coverage_fraction"] = 0.9
    elif problem == "duplicate":
        after["episodes"].append(copy.deepcopy(after["episodes"][0]))
    elif problem == "wrong_pair_seed":
        after["episodes"][0]["seed"] += 1
    else:
        after["episodes"][0]["source_group"] = "ngsim:foreign"
    assert (
        study.review_actuation(before, after, groups, config)["decision"]
        == "rejected_incomplete_physical_evaluation"
    )


def test_development_ties_keep_baseline_and_zero_error_candidate_can_win():
    groups = ("ngsim:a", "ngsim:b")
    arms = {name: _result(groups, (17,), 1.0) for name in ("A", "B", "C")}
    assert study.select_development(arms, groups, 17)["selected_arm"] == "A"
    arms["C"] = _result(groups, (17,), 0.0)
    selection = study.select_development(arms, groups, 17)
    assert selection["selected_arm"] == "C"
    assert selection["test_evaluated"] is False


@pytest.mark.parametrize(
    "settings",
    [
        {"development_sources": 0},
        {"workers": True},
        {"test_seeds": (17,)},
        {"test_seeds": (117, 117)},
        {"test_seeds": (True,)},
        {"seconds": float("nan")},
    ],
)
def test_invalid_design_settings_are_rejected(settings):
    with pytest.raises(ValueError):
        study.ActuationEvaluationConfig(**settings)


def _fake_bridge(monkeypatch):
    @dataclass
    class Idm:
        max_acceleration: float = 1.0
        comfortable_deceleration: float = 1.5
        time_headway: float = 1.6
        min_gap: float = 2.0
        exponent: float = 4.0

    @dataclass
    class Acceleration:
        gain: float = 1.0

    @dataclass
    class Reflex:
        perception_delay: float = 0.02
        idm: Idm = field(default_factory=Idm)
        acceleration_control: Acceleration | None = None

    package = ModuleType("myosuite_carla")
    package.__path__ = []
    models = ModuleType("myosuite_carla.driver_models")
    models.IdmConfig, models.AccelerationControlConfig, models.ReflexDriverConfig = (
        Idm,
        Acceleration,
        Reflex,
    )
    following = ModuleType("myosuite_carla.empirical_following")
    all_groups = tuple(f"ngsim:{i}" for i in range(12))
    following.load_following_protocol = lambda *a, **k: SimpleNamespace(
        groups=all_groups, trials_for=lambda g: (g,)
    )
    ngsim = ModuleType("myosuite_carla.ngsim")
    ngsim.save_trials = lambda trials, path: Path(path).write_text(json.dumps(trials))
    for name, value in (
        ("myosuite_carla", package),
        (models.__name__, models),
        (following.__name__, following),
        (ngsim.__name__, ngsim),
    ):
        monkeypatch.setitem(sys.modules, name, value)


@pytest.mark.parametrize("improve", [True, False])
def test_orchestration_freezes_selection_before_any_fresh_test_job(
    tmp_path, monkeypatch, improve
):
    _fake_bridge(monkeypatch)
    source = tmp_path / "raw.csv"
    source.write_text("fixed trajectory source")
    old = tmp_path / "history.json"
    inspected = [f"ngsim:{i}" for i in range(6)]
    old.write_text(json.dumps({"requested_source_groups": inspected}))
    component = tmp_path / "component.json"
    component.write_text(
        json.dumps({"evaluation": {"profile": {"parameters": {"time_headway": 0.7}}}})
    )
    out = tmp_path / "study"
    calls = []

    def run_jobs(jobs, workers):
        design = json.loads((out / "design.json").read_text())
        assert design["design_fixed_before_simulation"]
        stage = Path(jobs[0].output).parent.name
        calls.append(stage)
        if stage == "test":
            selection = json.loads((out / "selection.json").read_text())
            assert selection["selected_arm"] == "B"
            assert all(not set(job.groups).intersection(inspected) for job in jobs)
        else:
            assert not (out / "selection.json").exists()
            assert all(set(job.groups) <= set(inspected) for job in jobs)
        result = {}
        for job in jobs:
            error = (
                2.0 if job.arm == "A" else 1.0 if improve and job.arm == "B" else 3.0
            )
            result[job.arm] = {
                **_result(job.groups, job.seeds, error),
                "reflex_configuration": job.reflex,
                "metadata": {"muscles": 114, "driver_config": {"control_dt": 0.02}},
            }
        return result

    monkeypatch.setattr(study, "_run_jobs", run_jobs)
    result = study.evaluate_actuation_improvement(
        source,
        component_report=component,
        development_reports=[old],
        out=out,
        config=study.ActuationEvaluationConfig(
            development_sources=4, test_sources=2, bootstrap=100
        ),
    )
    assert calls == (["development", "test"] if improve else ["development"])
    assert result["test_evaluated"] is improve
    assert len(result["design"]["partitions"]["development"]) == 4
    assert result["profile"]["parameter_overrides"] == {}
    if improve:
        assert result["model_update"]["decision"] == "accepted_contextual_candidate"
        assert (
            result["closed_loop"]["reflex_configurations"]["baseline"][
                "acceleration_control"
            ]
            is None
        )
        assert result["closed_loop"]["reflex_configurations"]["fitted"][
            "acceleration_control"
        ] == {"gain": 1.0}
    else:
        assert result["model_update"]["fresh_test_unopened"]


def test_empty_idm_overrides_preserve_baseline_provenance():
    from embodied_human.core import HumanConfig
    from embodied_human.driving_profiles import resolve_profile

    profile = {
        "module": "reflex.idm",
        "context": "fixed mechanical correction",
        "role": "fitted_candidate",
        "parameters": {"time_headway": 1.6},
        "parameter_overrides": {},
        "test_used_for_parameter_selection": False,
        "configuration": {
            "human": HumanConfig().as_dict(),
            "driver": {},
            "reflex": {
                "idm": {"time_headway": 1.6},
                "acceleration_control": {"gain": 1.0},
            },
        },
        "provenance": {"role": "mechanical correction"},
    }
    resolved = resolve_profile(profile, context=profile["context"])
    assert resolved.parameters["time_headway"] == 1.6
    assert resolved.fitted_provenance() == []


def test_native_profile_roundtrip_preserves_nested_actuation_configuration():
    from dataclasses import asdict

    from embodied_human.core import HumanConfig
    from embodied_human.driving_profiles import resolve_profile

    bridge = pytest.importorskip("myosuite_carla.driver_models")
    native = bridge.ReflexDriverConfig(
        perception_delay=0.02,
        acceleration_control=bridge.AccelerationControlConfig(),
    )
    profile = {
        "module": "reflex.idm",
        "context": "fixed mechanical correction",
        "role": "fitted_candidate",
        "parameters": asdict(native.idm),
        "parameter_overrides": {},
        "test_used_for_parameter_selection": False,
        "configuration": {
            "human": HumanConfig().as_dict(),
            "driver": {},
            "reflex": asdict(native),
        },
        "provenance": {"role": "mechanical correction"},
    }
    resolved = resolve_profile(
        json.loads(json.dumps(profile)), context=profile["context"]
    )
    assert asdict(resolved.make_reflex_config()) == asdict(native)
    assert resolved.fitted_provenance() == []


def test_worker_retains_infeasible_episode_and_completes_other_sources(
    tmp_path, monkeypatch
):
    _fake_bridge(monkeypatch)
    following = sys.modules["myosuite_carla.empirical_following"]
    models = sys.modules["myosuite_carla.driver_models"]
    models.ReflexDriverConfig.from_dict = staticmethod(
        lambda values: SimpleNamespace(**values)
    )
    groups = ("ngsim:a", "ngsim:b")
    following.load_following_protocol = lambda *a, **k: SimpleNamespace(groups=groups)

    @dataclass
    class PhysicalConfig:
        seconds: float
        trials_per_source: int

    class Physical:
        def __init__(self, source, **kwargs):
            self.episodes = []
            self.closed = False
            self.entries = 0

        def __enter__(self):
            self.entries += 1
            return self

        def __exit__(self, *args):
            self.closed = True

        @property
        def metadata(self):
            return {"body_builds": self.entries}

        @property
        def diagnostics(self):
            return {"closed": self.closed, "episode_runs": len(self.episodes)}

        def __call__(self, parameters, sources, seed):
            row = _episode(sources[0], seed, 1.0)
            if sources[0] == "ngsim:a" and seed == 117:
                row.update(status="terminated", collision=True)
            self.episodes.append(row)
            if row["collision"]:
                raise RuntimeError("collision")
            return {sources[0]: {name: 1.0 for name in study.METRICS}}

    following.ClosedLoopConfig = PhysicalConfig
    following.ClosedLoopFollowingProtocol = Physical
    fixture = tmp_path / "fixture.npz"
    fixture.write_bytes(b"fixed source fixture")
    path = tmp_path / "worker.json"
    job = study.ArmJob(
        "B",
        {"idm": {"time_headway": 1.6}},
        str(fixture),
        groups,
        (117, 118),
        str(path),
        20,
        16,
    )
    result = study.run_arm(job)
    assert result["status"] == "infeasible"
    assert len(result["episodes"]) == 4
    assert sum(row["collision"] for row in result["episodes"]) == 1
    assert result["metadata"]["body_builds"] == 1
    assert result["diagnostics"]["closed"]
    assert len(path.with_suffix(".jsonl").read_text().splitlines()) == 4
    assert len(json.loads(path.read_text())["episodes"]) == 4


def test_cli_marks_changed_source_report_invalid(tmp_path, monkeypatch):
    path = Path(__file__).parents[1] / "examples/evaluate_actuation_improvement.py"
    spec = importlib.util.spec_from_file_location("actuation_cli_test", path)
    command = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(command)
    monkeypatch.setattr(command, "environment_record", lambda: {})
    origins = iter(
        (
            {"human": {"python_source_sha256": "before"}},
            {"human": {"python_source_sha256": "after"}},
        )
    )
    monkeypatch.setattr(command, "runtime_sources", lambda: next(origins))
    monkeypatch.setattr(
        command,
        "evaluate_actuation_improvement",
        lambda *a, **k: {
            "status": "evaluated",
            "model_update": {"decision": "accepted_contextual_candidate"},
        },
    )
    output = tmp_path / "run"
    code = command.main(
        [
            "--source",
            "unused.csv",
            "--component-report",
            "unused.json",
            "--development-report",
            "unused-history.json",
            "--out",
            str(output),
        ]
    )
    assert code == 1
    report = json.loads((output / "fullreport.json").read_text())
    assert report["status"] == "invalid_source_identity"
    assert report["runtime_source_identity_verified"] is False
    assert report["model_update"]["decision"] == "rejected_changed_runtime_sources"
