"""Evaluate six empirical driving sources and save explicit coverage reports.

Inputs are a JSON mapping from dataset identifiers to local paths and optional
loader settings. All sources are attempted. A gated or missing source leaves
the campaign partial. Fitted candidates remain separate from package defaults.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib
import json
import math
import sys
import traceback
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from embodied_human.driving_data import load_dataset
from embodied_human.reporting import environment_record, write_json
from embodied_human.reporting import runtime_sources as _runtime_sources

DATASETS = (
    "lehtonen2018",
    "rightgaze2022",
    "expert_driving",
    "ngsim",
    "drivedna_controlled",
    "mpdb",
)
ALIASES = {"rightgaze": "rightgaze2022", "drivedna": "drivedna_controlled"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    """Convert report metadata while preventing raw table serialization."""
    if hasattr(value, "columns") and hasattr(value, "index"):
        return {
            "rows": len(value),
            "columns": list(value.columns),
            "data_omitted": True,
        }
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _safe(getattr(value, item.name))
            for item in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "shape") and hasattr(value, "tolist"):
        if value.size > 10_000:
            return {"shape": list(value.shape), "data_omitted": True}
        return _safe(value.tolist())
    if hasattr(value, "item"):
        return _safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _registry(path: Path) -> dict[str, dict[str, Any]]:
    loaded = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(loaded, dict):
        raise ValueError("inputs must contain a JSON object keyed by dataset ID")
    registry = {}
    for name, entry in loaded.items():
        canonical = ALIASES.get(name, name)
        if canonical not in DATASETS:
            raise ValueError(f"Unknown dataset identifier: {name!r}")
        if canonical in registry:
            raise ValueError(f"Duplicate input for {canonical!r}")
        if isinstance(entry, str):
            entry = {"path": entry}
        if not isinstance(entry, dict):
            raise ValueError(f"Input for {name!r} must be an object or path string")
        registry[canonical] = dict(entry)
    return registry


def _promotion_review(report: Mapping[str, Any]) -> dict[str, Any]:
    """Record a conservative release review without changing test selection."""
    profile = report.get("profile")
    review: dict[str, Any] = {
        "defaults_updated": False,
        "policy": "post_hoc_conservative_release_review",
        "test_used_for_retuning": False,
        "decision": "reference_only",
        "test_mae_regressions": [],
    }
    if not isinstance(profile, Mapping):
        return review
    if profile.get("role") == "retained_baseline":
        review["decision"] = "retained_baseline"
        return review
    test = report.get("evaluations", {}).get("test", {})
    baseline = test.get("baseline", {}).get("metrics", {})
    selected = test.get("selected", {}).get("metrics", {})
    compared = 0
    for metric in sorted(baseline.keys() | selected.keys()):
        before = baseline.get(metric, {}).get("mae")
        after = selected.get(metric, {}).get("mae")
        if before is None or after is None:
            continue
        old, new = before.get("value"), after.get("value")
        if old is not None and new is not None:
            compared += 1
        if old is not None and new is not None and new > old + 1e-12:
            review["test_mae_regressions"].append(
                {"metric": metric, "baseline": old, "candidate": new}
            )
    if review["test_mae_regressions"]:
        review["decision"] = "rejected_after_test_regression"
    elif report.get("status") != "evaluated" or not test or not compared:
        review["decision"] = "independent_evaluation_required"
    else:
        review["decision"] = "candidate_for_independent_validation"
    closed = report.get("closed_loop")
    if isinstance(closed, Mapping):
        from embodied_human.driving_evaluation import compare_prediction_change

        physical_metrics = (
            "speed_rmse_mps",
            "gap_rmse_m",
            "acceleration_rmse_mps2",
        )
        # Keep seeds paired within trials, trials within original sources, and
        # give every original source equal weight in the final comparison.
        episodes: dict[str, dict[tuple[str, str, str], dict[str, float]]] = {
            "baseline": {},
            "fitted": {},
        }
        incomplete: list[dict[str, Any]] = []
        failures = 0
        for episode in closed.get("episodes", []):
            label = episode.get("profile")
            if label not in episodes:
                continue
            source = episode.get("source_group")
            identity = (
                str(source),
                str(episode.get("trial_index", 0)),
                str(episode.get("seed", "")),
            )
            failed = (
                episode.get("status") != "completed"
                or episode.get("collision")
                or episode.get("off_road")
            )
            if failed and label == "fitted":
                failures += 1
            values = {
                name: episode.get(name)
                for name in (*physical_metrics, "coverage_fraction")
            }
            valid = (
                isinstance(source, str)
                and bool(source)
                and not failed
                and all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(value)
                    and value >= 0
                    for value in values.values()
                )
                and values["coverage_fraction"] >= 1.0 - 1e-9
                and values["coverage_fraction"] <= 1.0 + 1e-9
            )
            if not valid or identity in episodes[label]:
                incomplete.append(
                    {"profile": label, "source_group": source, "trial": identity[1]}
                )
                continue
            episodes[label][identity] = {
                name: float(value) for name, value in values.items()
            }
        before_ids, after_ids = set(episodes["baseline"]), set(episodes["fitted"])
        paired = before_ids & after_ids
        unpaired = before_ids ^ after_ids
        expected_trials = closed.get("selected_trials")
        selected_sources = set(closed.get("selected_source_groups", []))
        present_sources = {identity[0] for identity in paired}
        complete = (
            closed.get("status") == "completed"
            and bool(paired)
            and not incomplete
            and not unpaired
            and (not selected_sources or selected_sources == present_sources)
            and (
                expected_trials is None
                or len({identity[:2] for identity in paired}) == expected_trials
            )
        )
        source_summaries: dict[str, dict[str, dict[str, float]]] = {}
        for label in ("baseline", "fitted"):
            per_source: dict[str, dict[str, list[dict[str, float]]]] = {}
            for source, trial, simulation_seed in sorted(paired):
                per_source.setdefault(source, {}).setdefault(trial, []).append(
                    episodes[label][(source, trial, simulation_seed)]
                )
            source_summaries[label] = {}
            for source, trials in per_source.items():
                source_summaries[label][source] = {}
                for metric in (*physical_metrics, "coverage_fraction"):
                    trial_means = [
                        math.fsum(row[metric] for row in rows) / len(rows)
                        for rows in trials.values()
                    ]
                    source_summaries[label][source][metric] = math.fsum(
                        trial_means
                    ) / len(trial_means)
        review["closed_loop_completeness"] = {
            "complete": complete,
            "paired_sources": len(present_sources),
            "paired_trial_seed_records": len(paired),
            "unpaired_trial_seed_records": len(unpaired),
            "invalid_or_incomplete_episodes": incomplete,
            "weighting": (
                "equal seeds per trial; equal trials per source; equal sources"
            ),
        }
        review["closed_loop_source_summaries"] = source_summaries
        review["closed_loop_regressions"] = []
        if paired:
            zero = {
                source: dict.fromkeys((*physical_metrics, "coverage_shortfall"), 0.0)
                for source in present_sources
            }
            errors = {
                label: {
                    source: {
                        **{metric: row[metric] for metric in physical_metrics},
                        "coverage_shortfall": max(0.0, 1.0 - row["coverage_fraction"]),
                    }
                    for source, row in summaries.items()
                }
                for label, summaries in source_summaries.items()
            }
            review["paired_source_comparison"] = compare_prediction_change(
                zero,
                errors["baseline"],
                errors["fitted"],
                (*physical_metrics, "coverage_shortfall"),
                bootstrap=1000,
                seed=0,
            )
            for metric in physical_metrics:
                old = math.fsum(
                    row[metric] for row in source_summaries["baseline"].values()
                ) / len(present_sources)
                new = math.fsum(
                    row[metric] for row in source_summaries["fitted"].values()
                ) / len(present_sources)
                if new > old + 1e-12:
                    review["closed_loop_regressions"].append(
                        {"metric": metric, "baseline": old, "candidate": new}
                    )
        if failures:
            review["decision"] = "rejected_after_closed_loop_failure"
            review["closed_loop_failed_episodes"] = failures
        elif not complete:
            review["decision"] = "physical_evaluation_incomplete"
        elif review["closed_loop_regressions"]:
            review["decision"] = "rejected_after_closed_loop_regression"
    return review


def _baseline(
    protocol: Any, parameters: dict[str, float], metrics: tuple[str, ...], seed: int
) -> dict[str, Any]:
    from embodied_human.driving_evaluation import compare_predictions, group_split

    split = getattr(protocol, "split", None) or group_split(
        tuple(protocol.observed), seed=seed
    )
    reports = {}
    for role in ("train", "validation", "test"):
        groups = getattr(split, role)
        observed = {group: protocol.observed[group] for group in groups}
        predicted = protocol(parameters, groups, 100 + seed)
        reports[role] = compare_predictions(
            observed, predicted, metrics, bootstrap=500, seed=seed
        )
    return {
        "status": "evaluated"
        if all(r["status"] == "evaluated" for r in reports.values())
        else "incomplete_evaluation",
        "role": "baseline_component_comparison",
        "parameters": parameters,
        "partitions": dataclasses.asdict(split),
        "evaluation_seed": 100 + seed,
        "evaluations": reports,
        "protocol": protocol.metadata,
    }


def _fixation(
    bundle: Any,
    *,
    fit: bool,
    samples: int,
    seconds: float,
    seed: int,
    reuse: Path | None,
) -> dict[str, Any]:
    from embodied_human.attention import SamplerParameters
    from embodied_human.fixation_protocol import (
        METRICS,
        FixationProtocol,
        calibrate_fixation,
    )

    if reuse is not None:
        from embodied_human.driving_evaluation import compare_prediction_change

        report = json.loads(reuse.read_text(encoding="utf-8-sig"))
        expected = FixationProtocol(bundle, seconds=seconds, seed=seed)
        if report.get("status") != "evaluated" or report.get("partitions") != _safe(
            expected.split
        ):
            raise ValueError(
                "Reused fixation report has incompatible source partitions or status"
            )
        protocol_fields = (
            "role",
            "backend",
            "simulated_seconds_per_trial",
            "target_centres_deg",
            "target_half_extent_deg",
            "source",
        )
        stored_protocol = report.get("protocol", {})
        cache = expected.metadata.get("cache_identity")
        if not isinstance(cache, Mapping) or not cache:
            raise ValueError("Current fixation protocol lacks a cache identity")
        if stored_protocol.get("cache_identity") != _safe(cache):
            raise ValueError(
                "Reused fixation report cache identity differs from current "
                "observations, context, settings or model source"
            )
        if any(
            stored_protocol.get(name) != _safe(expected.metadata.get(name))
            for name in protocol_fields
        ):
            raise ValueError(
                "Reused fixation report has incompatible protocol settings"
            )
        if report.get("posterior", {}).get("simulated", 0) < samples:
            raise ValueError("Reused fixation report has fewer samples than requested")
        report["reused_report"] = {
            "path": str(reuse),
            "sha256": _sha256(reuse),
            "validated_protocol_fields": [*protocol_fields, "cache_identity"],
        }
        for role in ("train", "validation", "test"):
            groups = getattr(expected.split, role)
            evaluation = report["evaluations"][role]
            evaluation["paired_change"] = compare_prediction_change(
                {group: expected.observed[group] for group in groups},
                evaluation["baseline_predictions"],
                evaluation["selected_predictions"],
                METRICS,
                bootstrap=500,
                seed=seed,
            )
        return report
    if fit:
        return calibrate_fixation(bundle, samples=samples, seconds=seconds, seed=seed)
    defaults = SamplerParameters()
    parameters = {
        name: float(getattr(defaults, name))
        for name in ("return_sd", "threshold_spread", "maximum_glance")
    }
    return _baseline(
        FixationProtocol(bundle, seconds=seconds, seed=seed), parameters, METRICS, seed
    )


def _following(
    path: Path,
    *,
    fit: bool,
    samples: int,
    seed: int,
    max_groups: int,
    closed_loop_trials: int,
) -> dict[str, Any]:
    from myosuite_carla.driver_models import ReflexDriverConfig
    from myosuite_carla.empirical_following import (
        FIXED_ERROR_SCALES,
        METRICS,
        closed_loop_following,
        idm_parameters,
        load_following_protocol,
    )

    from embodied_human.agents.driver import DriverAgentParameters
    from embodied_human.core import HumanConfig
    from embodied_human.driving_evaluation import calibrate_grouped
    from embodied_human.inference import Prior

    protocol = load_following_protocol(path, max_groups=max_groups, seed=seed)
    baseline = idm_parameters()
    if not fit:
        return _baseline(protocol, baseline, METRICS, seed)
    report = calibrate_grouped(
        protocol.observed,
        protocol,
        (
            Prior("time_headway", 0.5, 3),
            Prior("min_gap", 0.5, 5),
            Prior("max_acceleration", 0.5, 3),
            Prior("comfortable_deceleration", 0.5, 4),
        ),
        baseline,
        metrics=METRICS,
        scales=FIXED_ERROR_SCALES,
        configuration={
            "human": HumanConfig().as_dict(),
            "driver": dataclasses.asdict(DriverAgentParameters()),
            "reflex": dataclasses.asdict(ReflexDriverConfig(perception_delay=0.02)),
        },
        provenance=protocol.metadata,
        samples=samples,
        seed=seed,
        fit_seeds=(0, 1),
        evaluation_seeds=(100, 101),
        bootstrap=500,
        max_validation_candidates=8,
    )
    report["profile"]["module"] = "reflex.idm"
    report["profile"]["context"] = "NGSIM car following at desired speed 16 m/s"
    report["protocol"] = protocol.metadata
    report["closed_loop"] = closed_loop_following(
        protocol,
        baseline,
        report["profile"]["parameters"],
        groups=report["partitions"]["test"],
        max_trials=closed_loop_trials,
        seed=seed,
    )
    return report


def _source_files(bundle: Any, path: Path) -> list[dict[str, Any]]:
    files = bundle.metadata.get("source_files", [])
    if not files and path.is_file():
        files = [path]
    result = []
    for value in sorted(set(map(str, files))):
        file = Path(value)
        result.append(
            {"path": str(file), "bytes": file.stat().st_size, "sha256": _sha256(file)}
        )
    return result


def run_campaign(
    inputs: Path,
    out: Path,
    *,
    fit: bool = False,
    samples: int = 64,
    seconds: float = 30,
    seed: int = 0,
    max_following_groups: int = 96,
    closed_loop_trials: int = 6,
    fixation_report: Path | None = None,
) -> dict[str, Any]:
    """Attempt all six datasets and save independent dataset reports."""
    for name, value in (
        ("samples", samples),
        ("max_following_groups", max_following_groups),
        ("closed_loop_trials", closed_loop_trials),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if (
        not math.isfinite(seconds)
        or seconds < 0.4
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
    ):
        raise ValueError("need finite seconds >= 0.4 and a nonnegative integer seed")
    if fixation_report is not None and not fit:
        raise ValueError("fixation-report requires fitting mode")
    registry = _registry(inputs)
    combined = out if out.suffix.lower() == ".json" else out / "campaign.json"
    if combined.resolve() == inputs.resolve():
        raise ValueError("output must not overwrite the input registry")
    reports = combined.parent / (combined.stem + "-datasets")
    reports.mkdir(parents=True, exist_ok=True)
    campaign: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "input_registry": {"path": str(inputs.resolve()), "sha256": _sha256(inputs)},
        "configuration": {
            "fit": fit,
            "samples": samples,
            "seconds": seconds,
            "seed": seed,
            "max_following_groups": max_following_groups,
            "closed_loop_trials": closed_loop_trials,
        },
        "script_sha256": _sha256(Path(__file__)),
        "environment": environment_record(),
        "runtime_sources": _runtime_sources(),
        "interpretation": (
            "Component evaluations and observational references retain separate "
            "scientific roles. Evaluation completion does not establish model adequacy."
        ),
        "datasets": {},
        "defaults_updated": False,
    }
    write_json(combined, _safe(campaign))
    for dataset in DATASETS:
        print(f"{dataset}: starting", flush=True)
        record: dict[str, Any] = {"dataset_id": dataset, "status": "running"}
        stage = "input"
        try:
            entry = registry.get(dataset)
            if entry is None:
                record.update(
                    status="missing_input", reason="Dataset is absent from the registry"
                )
            elif not entry.get("path"):
                record.update(status=entry.get("status", "missing_input"), input=entry)
                if record["status"] not in ("access_pending", "missing_input"):
                    record["status"] = "missing_input"
            else:
                path = Path(entry["path"]).expanduser()
                if not path.is_absolute():
                    path = inputs.resolve().parent / path
                if not path.exists():
                    raise FileNotFoundError(f"Dataset path does not exist: {path}")
                if path.is_file() and entry.get("sha256"):
                    if _sha256(path) != entry["sha256"]:
                        raise ValueError(
                            "Source checksum differs from the input registry"
                        )
                options = dict(entry.get("options", {}))
                stage = "import"
                bundle = load_dataset(dataset, path, **options)
                record["data"] = _safe(bundle.summary())
                record["source_files"] = _source_files(bundle, path)
                record["primary_reference"] = bundle.source
                stage = "evaluation"
                protocols = importlib.import_module("embodied_human.driving_protocols")
                if dataset == "lehtonen2018":
                    evaluation = _fixation(
                        bundle,
                        fit=fit,
                        samples=samples,
                        seconds=seconds,
                        seed=seed,
                        reuse=fixation_report,
                    )
                elif dataset == "rightgaze2022":
                    evaluation = (
                        protocols.calibrate_gaze_allocation(bundle, seed=seed)
                        if fit
                        else protocols.gaze_allocation_reference(bundle)
                    )
                elif dataset == "ngsim":
                    evaluation = _following(
                        path,
                        fit=fit,
                        samples=samples,
                        seed=seed,
                        max_groups=max_following_groups,
                        closed_loop_trials=closed_loop_trials,
                    )
                elif dataset == "mpdb":
                    evaluation = protocols.emg_reference(bundle)
                else:
                    evaluation = protocols.telemetry_reference(bundle)
                record["evaluation"] = evaluation
                record["model_update"] = _promotion_review(evaluation)
                record["status"] = evaluation.get("status", "reference_only")
                closed_loop = evaluation.get("closed_loop")
                if isinstance(closed_loop, Mapping):
                    if closed_loop.get("status") != "completed":
                        record["status"] = "incomplete_evaluation"
        except Exception as error:
            record.update(
                status="failed",
                error={
                    "stage": stage,
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
            print(
                f"{dataset}: ERROR during {stage}: {type(error).__name__}: {error}",
                file=sys.stderr,
                flush=True,
            )
        path = reports / f"{dataset}.json"
        write_json(path, _safe(record))
        campaign["datasets"][dataset] = {
            "status": record["status"],
            "report": str(path.relative_to(combined.parent)),
            "sha256": _sha256(path),
            "data": record.get("data"),
            "model_update": record.get("model_update"),
            "error": record.get("error"),
        }
        print(f"{dataset}: {record['status']}", flush=True)
        write_json(combined, _safe(campaign))
    processed = {"evaluated", "reference_only", "reference", "reference_ready"}
    campaign["status"] = (
        "completed"
        if all(r["status"] in processed for r in campaign["datasets"].values())
        else "partial"
    )
    campaign["finished_utc"] = datetime.now(timezone.utc).isoformat()
    campaign["dataset_status_counts"] = {
        status: sum(r["status"] == status for r in campaign["datasets"].values())
        for status in sorted({r["status"] for r in campaign["datasets"].values()})
    }
    write_json(combined, _safe(campaign))
    return campaign


def _positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return number


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fit", action="store_true")
    parser.add_argument("--samples", type=_positive, default=64)
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-following-groups", type=_positive, default=96)
    parser.add_argument("--closed-loop-trials", type=_positive, default=6)
    parser.add_argument("--fixation-report", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args(argv)
    if not math.isfinite(args.seconds) or args.seconds < 0.4 or args.seed < 0:
        parser.error(
            "seconds must be finite and at least 0.4; seed must be nonnegative"
        )
    if args.fixation_report is not None and not args.fit:
        parser.error("fixation-report requires --fit")
    try:
        report = run_campaign(
            args.inputs,
            args.out,
            fit=args.fit,
            samples=args.samples,
            seconds=args.seconds,
            seed=args.seed,
            max_following_groups=args.max_following_groups,
            closed_loop_trials=args.closed_loop_trials,
            fixation_report=args.fixation_report,
        )
    except (OSError, ValueError) as error:
        print(f"Campaign configuration error: {error}", file=sys.stderr)
        return 1
    if any(item["status"] == "failed" for item in report["datasets"].values()):
        return 1
    return 0 if report["status"] == "completed" or args.allow_incomplete else 2


if __name__ == "__main__":
    raise SystemExit(main())
