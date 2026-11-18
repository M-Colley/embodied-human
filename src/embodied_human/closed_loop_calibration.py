"""Source grouped calibration through the complete physical driving loop."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .core import HumanConfig
from .driving_evaluation import GroupSplit, calibrate_grouped, compare_prediction_change
from .inference import Prior

METRICS = ("speed_rmse_mps", "gap_rmse_m", "acceleration_rmse_mps2")
SCALES = dict(zip(METRICS, (1.0, 5.0, 1.0)))
PRIORS = (
    Prior("time_headway", 0.6, 2.6),
    Prior("min_gap", 0.5, 4.0),
    Prior("max_acceleration", 0.5, 2.0),
    Prior("comfortable_deceleration", 0.8, 3.0),
)
PRIMARY = "normalized_trajectory_mse"
Progress = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class ClosedLoopCalibrationConfig:
    """A fixed computational design for a contextual calibration experiment."""

    samples: int = 16
    train_sources: int = 6
    validation_sources: int = 3
    test_sources: int = 6
    seconds: float = 20.0
    trials_per_source: int = 1
    seed: int = 20261028
    fit_seed: int = 17
    evaluation_seed: int = 117
    accept: float = 0.25
    bootstrap: int = 2000
    desired_speed: float = 16.0

    def __post_init__(self) -> None:
        for name in (
            "samples",
            "train_sources",
            "validation_sources",
            "test_sources",
            "trials_per_source",
            "bootstrap",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("seed", "fit_seed", "evaluation_seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.fit_seed == self.evaluation_seed:
            raise ValueError("fit and evaluation seeds must differ")
        for name in ("seconds", "desired_speed", "accept"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if self.seconds < 0.4 or self.accept > 1:
            raise ValueError("seconds must be at least 0.4 and accept at most one")


def file_sha256(path: Path) -> str:
    """Hash a source file without loading the complete dataset into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_exclusions(paths: Sequence[str | Path]) -> tuple[str, ...]:
    """Recover all previously inspected source groups from retained reports."""
    groups: set[str] = set()
    visited: set[Path] = set()
    recovered = 0

    def identities(value: Any) -> None:
        nonlocal recovered
        if not isinstance(value, (list, tuple)) or any(
            not isinstance(item, str) or not item.strip() for item in value
        ):
            raise ValueError("report source groups must be nonempty strings")
        selected = [item for item in value if item.startswith("ngsim:")]
        groups.update(selected)
        recovered += len(selected)

    def visit(value: Any, directory: Path) -> None:
        if not isinstance(value, Mapping):
            return
        partitions = value.get("partitions")
        if isinstance(partitions, Mapping):
            for role in ("train", "validation", "test"):
                if role in partitions:
                    identities(partitions[role])
        for name in ("source_groups", "requested_source_groups"):
            if name in value:
                identities(value[name])
        for name in ("evaluation", "protocol", "closed_loop"):
            visit(value.get(name), directory)
        datasets = value.get("datasets")
        if isinstance(datasets, Mapping):
            for entry in datasets.values():
                visit(entry, directory)
                if isinstance(entry, Mapping) and isinstance(entry.get("report"), str):
                    read(directory / entry["report"].replace("\\", "/"))

    def read(path: Path) -> None:
        resolved = path.resolve()
        if resolved in visited:
            return
        visited.add(resolved)
        value = json.loads(resolved.read_text(encoding="utf-8-sig"))
        visit(value, resolved.parent)

    for path in paths:
        before = recovered
        duplicate = Path(path).resolve() in visited
        read(Path(path))
        if not duplicate and recovered == before:
            raise ValueError(f"exclusion report contains no source identities: {path}")
    return tuple(sorted(groups))


def _complete_episode(row: Mapping[str, Any]) -> bool:
    coverage = row.get("coverage_fraction")
    return (
        row.get("status") == "completed"
        and row.get("collision") is False
        and row.get("off_road") is False
        and row.get("surrogate_closed") is True
        and row.get("full_source_horizon") is True
        and isinstance(coverage, (float, int))
        and not isinstance(coverage, bool)
        and math.isfinite(coverage)
        and abs(coverage - 1.0) <= 1e-8
        and all(
            isinstance(row.get(name), (float, int))
            and not isinstance(row[name], bool)
            and math.isfinite(row[name])
            and row[name] >= 0
            for name in METRICS
        )
    )


def review_closed_loop(
    report: Mapping[str, Any], config: ClosedLoopCalibrationConfig
) -> dict[str, Any]:
    """Apply the acceptance rule fixed before simulation to complete test pairs."""
    result: dict[str, Any] = {
        "decision": "rejected_incomplete_physical_evaluation",
        "policy": "prespecified_paired_source_primary_error",
        "defaults_updated": False,
        "test_used_for_retuning": False,
        "primary_metric": PRIMARY,
        "primary_definition": "mean across metrics of squared normalized RMSE",
        "scales": dict(SCALES),
        "confidence_level": 0.95,
        "acceptance_requires": [
            "complete safe paired full horizon test episodes",
            "candidate selected on validation sources",
            "paired source mean primary error change below zero",
            "upper 95 percent paired source bootstrap bound below zero",
        ],
    }
    expected_sources = tuple(report.get("partitions", {}).get("test", ()))
    counts = report.get("design", {}).get("trial_counts", {})
    expected = {
        (group, index)
        for group in expected_sources
        for index in range(counts.get(group, config.trials_per_source))
    }
    pairs: dict[str, dict[tuple[str, int], Mapping[str, Any]]] = {
        "baseline": {},
        "fitted": {},
    }
    problems: list[dict[str, Any]] = []
    closed = report.get("closed_loop", {})
    for row in closed.get("episodes", ()):
        label = row.get("profile")
        key = (row.get("source_group"), row.get("trial_index"))
        if (
            label not in pairs
            or key not in expected
            or key in pairs[label]
            or row.get("base_seed") != config.evaluation_seed
            or isinstance(row.get("seed"), bool)
            or not isinstance(row.get("seed"), int)
            or not _complete_episode(row)
        ):
            problems.append({"profile": label, "source_group": key[0], "trial": key[1]})
            continue
        pairs[label][key] = row
    complete = (
        bool(expected)
        and len(expected_sources) == config.test_sources
        and len(set(expected_sources)) == len(expected_sources)
        and not problems
        and report.get("status") == "evaluated"
        and closed.get("status") == "completed"
        and all(set(values) == expected for values in pairs.values())
        and all(
            pairs["baseline"][key]["seed"] == pairs["fitted"][key]["seed"]
            for key in expected
        )
        and report.get("profile", {}).get("test_used_for_parameter_selection") is False
        and report.get("profile", {}).get("role")
        in {"fitted_candidate", "retained_baseline"}
    )
    result["completeness"] = {
        "complete": complete,
        "expected_sources": len(expected_sources),
        "expected_trials": len(expected),
        "invalid_episodes": problems,
        "baseline_trials": len(pairs["baseline"]),
        "fitted_trials": len(pairs["fitted"]),
    }
    if not complete:
        return result
    if len(expected_sources) < 2:
        result["decision"] = "insufficient_independent_test_sources"
        return result
    primary: dict[str, dict[str, dict[str, float]]] = {}
    metric_means: dict[str, dict[str, dict[str, float]]] = {}
    for label, records in pairs.items():
        primary[label], metric_means[label] = {}, {}
        for group in expected_sources:
            rows = [row for (source, _), row in records.items() if source == group]
            metric_means[label][group] = {
                name: float(np.mean([row[name] for row in rows])) for name in METRICS
            }
            primary[label][group] = {
                PRIMARY: float(
                    np.mean(
                        [
                            (metric_means[label][group][name] / SCALES[name]) ** 2
                            for name in METRICS
                        ]
                    )
                )
            }
    comparison = compare_prediction_change(
        {group: {PRIMARY: 0.0} for group in expected_sources},
        primary["baseline"],
        primary["fitted"],
        (PRIMARY,),
        bootstrap=config.bootstrap,
        seed=config.seed,
    )
    change = comparison["metrics"][PRIMARY]["change_in_mae"]
    result.update(
        source_primary_errors=primary,
        source_metric_errors=metric_means,
        paired_source_comparison=comparison,
    )
    if report["profile"].get("role") == "retained_baseline":
        result["decision"] = "retained_baseline"
    elif change["value"] >= 0:
        result["decision"] = "rejected_after_test_regression"
    elif change["interval"][1] >= 0:
        result["decision"] = "inconclusive_contextual_candidate"
    else:
        result["decision"] = "accepted_contextual_candidate"
    return result


def calibrate_closed_loop_following(
    source: str | Path,
    *,
    config: ClosedLoopCalibrationConfig = ClosedLoopCalibrationConfig(),
    exclude_reports: Sequence[str | Path] = (),
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Fit physical trajectory errors using a fresh fixed source partition."""
    from myosuite_carla.driver_models import ReflexDriverConfig
    from myosuite_carla.empirical_following import (
        ClosedLoopConfig,
        ClosedLoopFollowingProtocol,
        idm_parameters,
        load_following_protocol,
    )

    from .agents.driver import DriverAgentParameters

    started = time.perf_counter()
    exclusions = source_exclusions(exclude_reports)
    total = config.train_sources + config.validation_sources + config.test_sources
    recorded = load_following_protocol(
        source,
        max_groups=total,
        seed=config.seed,
        desired_speed=config.desired_speed,
        excluded_groups=exclusions,
    )
    if len(recorded.groups) != total or set(recorded.groups) & set(exclusions):
        raise ValueError("insufficient fresh source groups for the requested design")
    order = (
        np.random.default_rng(config.seed).permutation(sorted(recorded.groups)).tolist()
    )
    boundary = config.train_sources + config.validation_sources
    split = GroupSplit(
        tuple(order[: config.train_sources]),
        tuple(order[config.train_sources : boundary]),
        tuple(order[boundary:]),
        config.seed,
    )
    physical_config = ClosedLoopConfig(
        seconds=config.seconds,
        trials_per_source=config.trials_per_source,
    )
    design = {
        "schema_version": 1,
        "configuration": asdict(config),
        "partitions": asdict(split),
        "trial_counts": {
            group: min(config.trials_per_source, len(recorded.trials_for(group)))
            for group in recorded.groups
        },
        "excluded_source_groups": list(exclusions),
        "exclusion_reports": [
            {"path": str(Path(path).resolve()), "sha256": file_sha256(Path(path))}
            for path in exclude_reports
        ],
        "priors": [asdict(prior) for prior in PRIORS],
        "metrics": list(METRICS),
        "scales": dict(SCALES),
        "primary_definition": (
            "equal source mean of metric mean squared normalized RMSE"
        ),
        "acceptance": {
            "safety": "all baseline and selected test episodes complete and safe",
            "improvement": (
                "paired source primary error change upper 95 percent bound below zero"
            ),
            "defaults_updated": False,
        },
    }
    if progress is not None:
        progress({"event": "design_frozen", "design": design})
    calls = 0
    baseline = idm_parameters()
    physical = ClosedLoopFollowingProtocol(recorded, config=physical_config)
    with physical:

        def simulate(
            parameters: Mapping[str, float], groups: tuple[str, ...], seed: int
        ) -> Any:
            nonlocal calls
            calls += 1
            error = None
            try:
                return physical(parameters, groups, seed)
            except Exception as failure:
                error = f"{type(failure).__name__}: {failure}"
                raise
            finally:
                if progress is not None:
                    progress(
                        {
                            "event": "objective_completed",
                            "call": calls,
                            "parameters": dict(parameters),
                            "groups": list(groups),
                            "seed": seed,
                            "diagnostics": physical.diagnostics,
                            "elapsed_seconds": time.perf_counter() - started,
                            "error": error,
                        }
                    )

        report = calibrate_grouped(
            physical.observed,
            simulate,
            PRIORS,
            baseline,
            metrics=METRICS,
            scales=SCALES,
            split=split,
            configuration={
                "human": HumanConfig().as_dict(),
                "driver": asdict(DriverAgentParameters()),
                "reflex": asdict(ReflexDriverConfig(perception_delay=0.02)),
            },
            provenance=physical.metadata,
            samples=config.samples,
            accept=config.accept,
            seed=config.seed,
            fit_seeds=(config.fit_seed,),
            evaluation_seeds=(config.evaluation_seed,),
            bootstrap=config.bootstrap,
        )
        report["profile"]["module"] = "reflex.idm"
        report["profile"]["context"] = (
            "NGSIM following through the MyoSuite body and CARLA 0.10 Nissan Patrol "
            f"surrogate at desired speed {config.desired_speed:g} m/s"
        )
        report["closed_loop"] = physical.report_pair(
            baseline,
            report["profile"]["parameters"],
            groups=split.test,
            seed=config.evaluation_seed,
        )
        report["protocol"] = physical.metadata
        report["episodes"] = physical.episodes
    report["diagnostics"] = physical.diagnostics
    report["design"] = design
    report["model_update"] = review_closed_loop(report, config)
    report["defaults_updated"] = False
    report["elapsed_seconds"] = time.perf_counter() - started
    report["limitations"] = [
        "The limited prior draws support a contextual calibration pilot.",
        "One fit seed and one evaluation seed provide limited stochastic coverage.",
        "NGSIM source groups identify vehicle tracks.",
        "Persistent participant identities are unavailable.",
        "Vehicle dynamics vary across recorded vehicles and the surrogate.",
        "Fitted controller parameters can absorb surrogate and actuator mismatch.",
        "Trajectory derivatives retain source filtering and measurement noise.",
        "This offline experiment requires separate live CARLA validation.",
    ]
    return report
