"""Evaluate fixed actuator corrections on development and fresh test sources."""

from __future__ import annotations

import json
import math
import multiprocessing
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .closed_loop_calibration import (
    METRICS,
    PRIMARY,
    SCALES,
    _complete_episode,
    file_sha256,
    source_exclusions,
)
from .core import HumanConfig
from .driving_evaluation import compare_prediction_change
from .reporting import write_json

Progress = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class ActuationEvaluationConfig:
    """Fixed source counts and common random numbers for the three arm study."""

    development_sources: int = 32
    test_sources: int = 24
    seed: int = 20261029
    development_seed: int = 17
    test_seeds: tuple[int, ...] = (117, 118)
    seconds: float = 20.0
    desired_speed: float = 16.0
    bootstrap: int = 5000
    workers: int = 3

    def __post_init__(self) -> None:
        for name in ("development_sources", "test_sources", "bootstrap", "workers"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("seed", "development_seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if isinstance(self.test_seeds, (str, bytes)):
            raise ValueError("test_seeds must contain distinct integer seeds")
        object.__setattr__(self, "test_seeds", tuple(self.test_seeds))
        if (
            not self.test_seeds
            or len(set(self.test_seeds)) != len(self.test_seeds)
            or self.development_seed in self.test_seeds
            or any(
                isinstance(s, bool) or not isinstance(s, int) or s < 0
                for s in self.test_seeds
            )
        ):
            raise ValueError("test seeds must be distinct from development")
        for name in ("seconds", "desired_speed"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if self.seconds < 0.4:
            raise ValueError("seconds must be at least 0.4")


@dataclass(frozen=True)
class ArmJob:
    """A pickle compatible job whose worker owns one physical body."""

    arm: str
    reflex: dict[str, Any]
    fixture: str
    groups: tuple[str, ...]
    seeds: tuple[int, ...]
    output: str
    seconds: float
    desired_speed: float


def run_arm(job: ArmJob) -> dict[str, Any]:
    """Run every requested source and seed while preserving every episode."""
    from myosuite_carla.driver_models import ReflexDriverConfig
    from myosuite_carla.empirical_following import (
        ClosedLoopConfig,
        ClosedLoopFollowingProtocol,
        load_following_protocol,
    )

    output = Path(job.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    progress_path = output.with_suffix(".jsonl")
    if output.exists() or progress_path.exists():
        raise ValueError(f"worker output already exists: {output}")
    source = load_following_protocol(
        job.fixture, max_groups=None, desired_speed=job.desired_speed
    )
    if set(source.groups) != set(job.groups) or len(set(job.groups)) != len(job.groups):
        raise ValueError("worker fixture differs from its fixed source partition")
    reflex = ReflexDriverConfig.from_dict(job.reflex)
    physical = ClosedLoopFollowingProtocol(
        source,
        config=ClosedLoopConfig(seconds=job.seconds, trials_per_source=1),
        reflex_config=reflex,
    )
    records: list[dict[str, Any]] = []
    result: dict[str, Any] = {
        "arm": job.arm,
        "groups": list(job.groups),
        "base_seeds": list(job.seeds),
        "reflex_configuration": job.reflex,
        "fixture_sha256": file_sha256(Path(job.fixture)),
        "status": "completed",
        "episodes": records,
    }
    started = time.perf_counter()
    try:
        with physical:
            for base_seed in job.seeds:
                for group in job.groups:
                    error = None
                    previous = len(physical.episodes)
                    try:
                        physical(job.reflex["idm"], (group,), base_seed)
                    except (ArithmeticError, RuntimeError, ValueError) as failure:
                        error = f"{type(failure).__name__}: {failure}"
                        result["status"] = "infeasible"
                    produced = physical.episodes[previous:]
                    if len(produced) != 1:
                        raise RuntimeError(
                            "one source call must produce exactly one episode"
                        )
                    record = {**produced[0], "base_seed": base_seed, "arm": job.arm}
                    records.append(record)
                    event = {
                        "arm": job.arm,
                        "source_group": group,
                        "base_seed": base_seed,
                        "episode_runs": len(records),
                        "error": error,
                        "status": record["status"],
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                    with progress_path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(event, allow_nan=False) + "\n")
                    write_json(
                        output,
                        {
                            **result,
                            "metadata": physical.metadata,
                            "diagnostics": physical.diagnostics,
                        },
                    )
    except Exception as failure:
        result["status"] = "failed"
        result["error"] = f"{type(failure).__name__}: {failure}"
    finally:
        result.update(
            metadata=physical.metadata,
            diagnostics=physical.diagnostics,
            elapsed_seconds=time.perf_counter() - started,
        )
        write_json(output, result)
    return result


def _run_jobs(jobs: Sequence[ArmJob], workers: int) -> dict[str, dict[str, Any]]:
    if workers == 1:
        return {job.arm: run_arm(job) for job in jobs}
    with ProcessPoolExecutor(
        max_workers=min(workers, len(jobs)),
        mp_context=multiprocessing.get_context("spawn"),
    ) as pool:
        results = list(pool.map(run_arm, jobs))
    return {result["arm"]: result for result in results}


def _summaries(
    result: Mapping[str, Any], groups: Sequence[str], seeds: Sequence[int]
) -> dict[str, dict[str, float]] | None:
    expected = {(group, 0, seed) for group in groups for seed in seeds}
    records: dict[tuple[str, int, int], Mapping[str, Any]] = {}
    for row in result.get("episodes", ()):
        key = (row.get("source_group"), row.get("trial_index"), row.get("base_seed"))
        if (
            key not in expected
            or key in records
            or not _complete_episode(row)
            or isinstance(row.get("seed"), bool)
            or not isinstance(row.get("seed"), int)
            or row["seed"] < 0
            or isinstance(row.get("base_seed"), bool)
            or not isinstance(row.get("base_seed"), int)
            or isinstance(row.get("trial_index"), bool)
        ):
            return None
        records[key] = row
    if not expected or set(records) != expected or result.get("status") != "completed":
        return None
    return {
        group: {
            name: float(np.mean([records[(group, 0, seed)][name] for seed in seeds]))
            for name in METRICS
        }
        for group in groups
    }


def _primary(
    summaries: Mapping[str, Mapping[str, float]],
) -> dict[str, dict[str, float]]:
    return {
        group: {
            PRIMARY: math.fsum((row[name] / SCALES[name]) ** 2 for name in METRICS)
            / len(METRICS)
        }
        for group, row in summaries.items()
    }


def select_development(
    results: Mapping[str, Mapping[str, Any]], groups: Sequence[str], seed: int
) -> dict[str, Any]:
    """Select a fixed correction using development outcomes only."""
    scores: dict[str, float | None] = {}
    for arm in ("A", "B", "C"):
        summaries = _summaries(results[arm], groups, (seed,))
        scores[arm] = (
            math.sqrt(
                float(np.mean([row[PRIMARY] for row in _primary(summaries).values()]))
            )
            if summaries is not None
            else None
        )
    selected = "A"
    if scores["A"] is not None:
        eligible = [
            (value, arm) for arm in ("B", "C") if (value := scores[arm]) is not None
        ]
        if eligible:
            best_score, best = min(eligible)
            baseline_score = scores["A"]
            assert baseline_score is not None
            if best_score < baseline_score:
                selected = best
    return {
        "selected_arm": selected,
        "development_distances": scores,
        "selection_data": "development_only",
        "test_evaluated": False,
        "test_used_for_parameter_selection": False,
        "decision": (
            "incomplete_development_baseline"
            if scores["A"] is None
            else "evaluate_selected_correction"
            if selected != "A"
            else "retained_baseline_on_development"
        ),
    }


def review_actuation(
    baseline: Mapping[str, Any],
    selected: Mapping[str, Any],
    groups: Sequence[str],
    config: ActuationEvaluationConfig,
) -> dict[str, Any]:
    """Require complete common seed pairs and an independent source improvement."""
    result: dict[str, Any] = {
        "decision": "rejected_incomplete_physical_evaluation",
        "defaults_updated": False,
        "test_used_for_retuning": False,
        "policy": "prespecified_paired_source_primary_error",
        "primary_metric": PRIMARY,
        "scales": dict(SCALES),
        "confidence_level": 0.95,
        "weighting": "mean seeds per trial; one trial per source; equal sources",
    }
    before = _summaries(baseline, groups, config.test_seeds)
    after = _summaries(selected, groups, config.test_seeds)
    if (
        before is None
        or after is None
        or len(groups) != len(set(groups))
        or len(groups) != config.test_sources
    ):
        return result
    left = {
        (r["source_group"], r["trial_index"], r["base_seed"]): r
        for r in baseline["episodes"]
    }
    right = {
        (r["source_group"], r["trial_index"], r["base_seed"]): r
        for r in selected["episodes"]
    }
    if any(left[key]["seed"] != right[key]["seed"] for key in left):
        return result
    result["complete_pairs"] = len(left)
    result["source_metric_errors"] = {"baseline": before, "fitted": after}
    result["source_primary_errors"] = {
        "baseline": _primary(before),
        "fitted": _primary(after),
    }
    if len(groups) < 2:
        result["decision"] = "insufficient_independent_test_sources"
        return result
    comparison = compare_prediction_change(
        {group: {PRIMARY: 0.0} for group in groups},
        _primary(before),
        _primary(after),
        (PRIMARY,),
        bootstrap=config.bootstrap,
        seed=config.seed,
    )
    result["paired_source_comparison"] = comparison
    change = comparison["metrics"][PRIMARY]["change_in_mae"]
    result["decision"] = (
        "rejected_after_test_regression"
        if change["value"] >= 0
        else "inconclusive_contextual_candidate"
        if change["interval"][1] >= 0
        else "accepted_contextual_candidate"
    )
    return result


def evaluate_actuation_improvement(
    source: str | Path,
    *,
    component_report: str | Path,
    development_reports: Sequence[str | Path],
    out: str | Path,
    config: ActuationEvaluationConfig = ActuationEvaluationConfig(),
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Select a fixed actuator correction before evaluating fresh source tracks."""
    from myosuite_carla.driver_models import (
        AccelerationControlConfig,
        IdmConfig,
        ReflexDriverConfig,
    )
    from myosuite_carla.empirical_following import load_following_protocol
    from myosuite_carla.ngsim import save_trials

    from .agents.driver import DriverAgentParameters

    output = Path(out)
    output.mkdir(parents=True, exist_ok=True)
    for name in (
        "design.json",
        "selection.json",
        "fullreport.json",
        "development.npz",
        "test.npz",
    ):
        if (output / name).exists():
            raise ValueError("output directory already contains study evidence")
    started = time.perf_counter()
    known = source_exclusions(development_reports)
    if len(known) < config.development_sources:
        raise ValueError("insufficient previously inspected development sources")
    recorded = load_following_protocol(
        source, max_groups=None, desired_speed=config.desired_speed
    )
    if set(known) - set(recorded.groups):
        raise ValueError(
            "development source identities are absent from the supplied dataset"
        )
    fresh = sorted(set(recorded.groups) - set(known))
    if len(fresh) < config.test_sources:
        raise ValueError("insufficient fresh test sources")
    development = tuple(
        sorted(
            str(group)
            for group in np.random.default_rng(config.seed).choice(
                known, config.development_sources, replace=False
            )
        )
    )
    test = tuple(
        sorted(
            str(group)
            for group in np.random.default_rng(config.seed + 1).choice(
                fresh, config.test_sources, replace=False
            )
        )
    )
    component_path = Path(component_report)
    component_record = json.loads(component_path.read_text(encoding="utf-8-sig"))
    component = component_record.get("evaluation", component_record)["profile"][
        "parameters"
    ]
    base = ReflexDriverConfig(perception_delay=0.02)
    corrected = replace(base, acceleration_control=AccelerationControlConfig())
    fixed_component = replace(corrected, idm=IdmConfig(**component))
    arms = {
        name: asdict(values)
        for name, values in (("A", base), ("B", corrected), ("C", fixed_component))
    }
    fixtures = {}
    for name, groups in (("development", development), ("test", test)):
        path = output / (name + ".npz")
        save_trials([recorded.trials_for(group)[0] for group in groups], path)
        fixtures[name] = {"path": str(path.resolve()), "sha256": file_sha256(path)}
    design = {
        "schema_version": 1,
        "configuration": asdict(config),
        "partitions": {"development": list(development), "test": list(test)},
        "previously_inspected_sources": list(known),
        "population_sources": len(recorded.groups),
        "source": {
            "path": str(Path(source).resolve()),
            "sha256": file_sha256(Path(source)),
        },
        "development_reports": [
            {"path": str(Path(path).resolve()), "sha256": file_sha256(Path(path))}
            for path in development_reports
        ],
        "component_report": {
            "path": str(component_path.resolve()),
            "sha256": file_sha256(component_path),
        },
        "arms": arms,
        "arm_definitions": {
            "A": "legacy actuation with baseline IDM",
            "B": "inverse actuation with baseline IDM",
            "C": "inverse actuation with the previously fitted component IDM",
        },
        "fixtures": fixtures,
        "trials_per_source": 1,
        "metrics": list(METRICS),
        "scales": dict(SCALES),
        "primary_definition": "average RMSE across seeds before normalized squaring",
        "selection": "best complete safe B or C on development if better than A",
        "acceptance": (
            "all test pairs complete and safe; "
            "paired source 95 percent upper change below zero"
        ),
        "design_fixed_before_simulation": True,
        "test_used_for_parameter_selection": False,
    }
    write_json(output / "design.json", design)
    if progress is not None:
        progress({"event": "design_frozen", "design": design})
    jobs = [
        ArmJob(
            name,
            values,
            fixtures["development"]["path"],
            development,
            (config.development_seed,),
            str(output / "development" / (name + ".json")),
            config.seconds,
            config.desired_speed,
        )
        for name, values in arms.items()
    ]
    development_results = _run_jobs(jobs, config.workers)
    selection = select_development(
        development_results, development, config.development_seed
    )
    write_json(output / "selection.json", selection)
    if progress is not None:
        progress({"event": "selection_frozen", "selection": selection})
    selected = selection["selected_arm"]
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": (
            "incomplete_evaluation"
            if selection["development_distances"]["A"] is None
            else "evaluated"
        ),
        "design": design,
        "partitions": design["partitions"],
        "selection": selection,
        "development": development_results,
        "test_evaluated": False,
        "metrics": list(METRICS),
        "scales": dict(SCALES),
        "defaults_updated": False,
        "protocol": {
            "dataset_id": "ngsim",
            "source_groups": [*development, *test],
            "role": "fixed_actuation_improvement",
        },
    }
    if selected == "A":
        report["model_update"] = {
            "decision": selection["decision"],
            "defaults_updated": False,
            "test_used_for_retuning": False,
            "fresh_test_unopened": True,
        }
    else:
        test_jobs = [
            ArmJob(
                name,
                arms[name],
                fixtures["test"]["path"],
                test,
                config.test_seeds,
                str(output / "test" / (name + ".json")),
                config.seconds,
                config.desired_speed,
            )
            for name in ("A", selected)
        ]
        test_results = _run_jobs(test_jobs, config.workers)
        report["test_evaluated"] = True
        report["test"] = test_results
        review = review_actuation(
            test_results["A"], test_results[selected], test, config
        )
        report["model_update"] = review
        if review["decision"] == "rejected_incomplete_physical_evaluation":
            report["status"] = "incomplete_evaluation"
        metadata = test_results["A"]["metadata"]
        report["closed_loop"] = {
            "status": "completed" if review.get("complete_pairs") else "failed",
            "role": "paired_physical_closed_loop",
            "group_kind": "vehicle_track",
            "requested_source_groups": list(test),
            "selected_source_groups": list(test),
            "selected_trials": len(test),
            "base_seeds": list(config.test_seeds),
            "secondary_task": None,
            "controller": "embodied_human.agents.driver.AttentiveDriver",
            "vehicle": "offline fitted CARLA 0.10.0 Nissan Patrol surrogate",
            "desired_speed_mps": config.desired_speed,
            "profiles": {"baseline": arms["A"]["idm"], "fitted": arms[selected]["idm"]},
            "reflex_configurations": {"baseline": arms["A"], "fitted": arms[selected]},
            "episodes": [
                {**row, "profile": label}
                for label, name in (("baseline", "A"), ("fitted", selected))
                for row in test_results[name]["episodes"]
            ],
            **{
                name: metadata[name]
                for name in (
                    "human_import",
                    "human_version",
                    "muscles",
                    "driver_config",
                    "vehicle_parameters",
                )
                if name in metadata
            },
        }
    parameters = arms[selected]["idm"]
    report["profile"] = {
        "schema_version": 1,
        "module": "reflex.idm",
        "context": (
            "NGSIM following through the MyoSuite body and CARLA 0.10 Nissan Patrol "
            f"surrogate at desired speed {config.desired_speed:g} m/s"
        ),
        "role": "retained_baseline" if selected == "A" else "fitted_candidate",
        "parameters": parameters,
        "parameter_overrides": {
            name: value
            for name, value in parameters.items()
            if value != arms["A"]["idm"][name]
        },
        "configuration": {
            "human": HumanConfig().as_dict(),
            "driver": asdict(DriverAgentParameters()),
            "reflex": arms[selected],
        },
        "configuration_role": "resolved_full_reflex_configuration",
        "provenance": {"design": design, "selected_arm": selected},
        "test_used_for_parameter_selection": False,
    }
    report["elapsed_seconds"] = time.perf_counter() - started
    report["limitations"] = [
        "Development sources were previously inspected.",
        "Fresh source tracks provide the independent test comparison.",
        "Vehicle tracks lack persistent human participant identities.",
        "Controller values can absorb vehicle and surrogate differences.",
        "Two test seeds provide limited stochastic coverage.",
        "The experiment uses the offline fitted vehicle surrogate.",
    ]
    write_json(output / "fullreport.json", report)
    return report
