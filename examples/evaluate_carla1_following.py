"""Evaluate two frozen CARLA 1.0 following profiles on a reserved NGSIM site."""

from __future__ import annotations

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from myosuite_carla.driver_models import (
    AccelerationControlConfig,
    ReflexDriver,
    ReflexDriverConfig,
)
from myosuite_carla.empirical_following import FollowingProtocol
from myosuite_carla.ngsim import following_trials, load_tracks, save_trials
from myosuite_carla.vehicle_model import carla_1_0_nissan_patrol

from embodied_human.actuation_evaluation import (
    ActuationEvaluationConfig,
    ArmJob,
    review_actuation,
    run_arm,
)
from embodied_human.closed_loop_calibration import METRICS, SCALES, file_sha256


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def prepare(design: dict[str, Any], data: Path, manifest: Path, out: Path) -> None:
    source_manifest = read(manifest)
    if source_manifest.get("location") != "us-101":
        raise ValueError("The source provenance must identify NGSIM US 101")
    vehicle = carla_1_0_nissan_patrol()
    delay = design["configured_reflex_delay_s"]
    baseline = ReflexDriverConfig(perception_delay=delay)
    inverse = ReflexDriverConfig(
        perception_delay=delay,
        acceleration_control=AccelerationControlConfig(vehicle=vehicle),
    )
    lag = ReflexDriver(design["control_dt_s"], baseline).perception_history_lag_s
    if lag != design["realized_history_lag_s"]:
        raise ValueError("The realized delay differs from the frozen design")
    protocol = FollowingProtocol(
        following_trials(
            load_tracks(data),
            duration=design["seconds"],
            max_speed=design["maximum_observed_speed_mps"],
            max_trials_per_driver=1,
        ),
        dataset_id=design["dataset_id"],
        desired_speed=design["desired_speed_mps"],
        seed=design["selection_seed"],
        max_groups=design["test_sources"],
    )
    if len(protocol.groups) != design["test_sources"]:
        raise ValueError("The site has too few eligible original tracks")
    out.mkdir()
    fixture = save_trials(
        [protocol.trials_for(group)[0] for group in protocol.groups], out / "trials.npz"
    )
    write(
        out / "frozen-inputs.json",
        {
            "frozen_utc": datetime.now(timezone.utc).isoformat(),
            "source_sha256": file_sha256(data),
            "source_manifest_sha256": file_sha256(manifest),
            "fixture_sha256": file_sha256(fixture),
            "design": design,
            "source_groups": list(protocol.groups),
            "selection": protocol.metadata,
            "vehicle_parameters": vehicle.as_dict(),
            "reflex": {"baseline": asdict(baseline), "inverse": asdict(inverse)},
        },
    )
    print(f"Froze {len(protocol.groups)} original US 101 tracks", flush=True)


def evaluate(design: dict[str, Any], out: Path) -> None:
    inputs = read(out / "frozen-inputs.json")
    fixture = out / "trials.npz"
    if inputs["design"] != design or file_sha256(fixture) != inputs["fixture_sha256"]:
        raise ValueError("The test requires the frozen design and trial fixture")
    if inputs["vehicle_parameters"] != carla_1_0_nissan_patrol().as_dict():
        raise ValueError("The CARLA 1.0 preset differs from the frozen parameters")
    seeds = tuple(design["test_seeds"])
    groups = tuple(inputs["source_groups"])
    metrics_config = ActuationEvaluationConfig(
        test_sources=design["test_sources"],
        test_seeds=seeds,
        seed=design["selection_seed"],
        seconds=design["seconds"],
        desired_speed=design["desired_speed_mps"],
        bootstrap=design["bootstrap_resamples"],
    )
    jobs = [
        ArmJob(
            arm=arm,
            reflex=config,
            fixture=str(fixture),
            groups=groups,
            seeds=seeds,
            output=str(out / f"{arm}.json"),
            seconds=design["seconds"],
            desired_speed=design["desired_speed_mps"],
            vehicle_parameters=inputs["vehicle_parameters"],
            dataset_id=design["dataset_id"],
        )
        for arm, config in inputs["reflex"].items()
    ]
    with ProcessPoolExecutor(
        max_workers=len(jobs), mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        results = {result["arm"]: result for result in pool.map(run_arm, jobs)}
    report = review_actuation(
        results["baseline"], results["inverse"], groups, metrics_config
    )
    report["expected_episodes_per_profile"] = len(groups) * len(seeds)
    report["failed_episodes"] = {
        arm: result["diagnostics"]["failed_episodes"] for arm, result in results.items()
    }
    report["failure_counts"] = {
        arm: len(episodes) for arm, episodes in report["failed_episodes"].items()
    }
    if "source_metric_errors" in report:
        means = {
            arm: {
                metric: mean(row[metric] for row in rows.values()) for metric in METRICS
            }
            for arm, rows in report["source_metric_errors"].items()
        }
        report["metric_means"] = means
        guards = {
            metric: means["fitted"][metric] <= means["baseline"][metric]
            for metric in METRICS
        }
        report["metric_guards"] = guards
        if report["decision"] == "accepted_contextual_candidate" and not all(
            guards.values()
        ):
            report["decision"] = "rejected_metric_regression"
    write(out / "report.json", report)
    print(report["decision"], flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "test"), required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    design = read(args.design)
    if design["scales"] != SCALES or design["dataset_id"] != "ngsim-us101":
        parser.error(
            "the design must identify the reserved US 101 site and metric scales"
        )
    if args.stage == "prepare":
        if args.source is None or args.manifest is None:
            parser.error("preparation requires the source CSV and its provenance")
        prepare(design, args.source, args.manifest, args.out)
    else:
        evaluate(design, args.out)


if __name__ == "__main__":
    main()
