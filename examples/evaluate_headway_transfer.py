"""Select desired time headway on development data and test on a fresh road site."""

from __future__ import annotations

import argparse
import json
import multiprocessing
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from myosuite_carla.driver_models import (
    AccelerationControlConfig,
    ReflexDriver,
    ReflexDriverConfig,
)
from myosuite_carla.empirical_following import FollowingProtocol
from myosuite_carla.fullbody import FullBodyDriverConfig
from myosuite_carla.ngsim import load_trials, save_trials
from myosuite_carla.vehicle_model import carla_1_0_nissan_patrol
from scipy.stats import bootstrap

from embodied_human.actuation_evaluation import ArmJob, run_arm
from embodied_human.closed_loop_calibration import (
    METRICS,
    _complete_episode,
    file_sha256,
)
from embodied_human.reporting import runtime_sources

BIAS = "signed_gap_error_m"
GAP, SPEED = "gap_rmse_m", "speed_rmse_mps"


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def runtime_identity() -> dict[str, Any]:
    configuration = asdict(FullBodyDriverConfig.fast())
    return {
        "versions": {
            name: version(name)
            for name in ("myosuite", "myo-sim", "mujoco", "numpy", "scipy")
        },
        "sources": {
            name: info.get("python_source_sha256")
            for name, info in runtime_sources().items()
        },
        "body": json.loads(json.dumps(configuration)),
        "script_sha256": file_sha256(Path(__file__)),
    }


def reflex(design: dict[str, Any], headway: float) -> ReflexDriverConfig:
    base = ReflexDriverConfig(
        perception_delay=design["configured_reflex_delay_s"],
        acceleration_control=AccelerationControlConfig(
            vehicle=carla_1_0_nissan_patrol()
        ),
    )
    config = replace(base, idm=replace(base.idm, time_headway=headway))
    assert (
        ReflexDriver(design["control_dt_s"], config).perception_history_lag_s
        == design["realized_history_lag_s"]
    )
    return config


def prepare(
    design: dict[str, Any],
    stage: str,
    fixture: Path,
    out: Path,
    manifest: Path | None,
    selection: Path | None,
) -> None:
    digest = file_sha256(fixture)
    testing = stage == "test"
    dataset = design[f"{stage}_dataset"]
    manifest_hash = selection_hash = None
    runs: dict[str, str] = {}
    if testing:
        assert manifest is not None and selection is not None
        source = read(manifest)
        if source["fixture_sha256"] != digest:
            raise ValueError("The test fixture differs from its source manifest")
        picked = read(selection)
        if picked["design"] != design or picked["selected_headway_s"] is None:
            raise ValueError("A frozen development selection is required")
        if (
            picked["runtime"] != runtime_identity()
            or picked["vehicle_parameters"] != carla_1_0_nissan_patrol().as_dict()
        ):
            raise ValueError("The test must use the development runtime and vehicle")
        runs = {
            f"{dataset}:{group}": str(run)
            for group, run in source["run_by_group"].items()
        }
        manifest_hash = file_sha256(manifest)
        selection_hash = file_sha256(selection)
        candidates = [picked["selected_headway_s"]]
        seeds = design["test_seeds"]
    else:
        if digest != design["development_original_fixture_sha256"]:
            raise ValueError("Use the frozen prior US 101 fixture for development")
        candidates = design["development_candidates_s"]
        seeds = [design["development_seed"]]
    protocol = FollowingProtocol(
        load_trials(fixture),
        dataset_id=dataset,
        desired_speed=design["desired_speed_mps"],
        max_groups=None if testing else design["development_sources"],
        seed=design["source_selection_seed"],
    )
    if len(protocol.groups) != design[f"{stage}_sources"]:
        raise ValueError("The fixture has the wrong number of eligible track groups")
    if testing and (
        set(runs) != set(protocol.groups)
        or len(set(runs.values())) < design["test_minimum_run_groups"]
    ):
        raise ValueError("External tracks must match the fixture and span six runs")
    trials = []
    for group in protocol.groups:
        windows = protocol.trials_for(group)
        if len(windows) != 1 or abs(windows[0].duration - design["seconds"]) > 1e-8:
            raise ValueError("Every source must supply exactly one full 20 s window")
        trials.append(windows[0])
    out.mkdir()
    prepared = save_trials(trials, out / "trials.npz")
    headways = {"reference": design["reference_headway_s"]}
    headways.update({f"headway-{value}": value for value in candidates})
    if (
        testing
        and picked["selected_headway_s"] not in design["development_candidates_s"]
    ):
        raise ValueError("The selected test headway is outside the frozen grid")
    write(
        out / "inputs.json",
        {
            "frozen_utc": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "design": design,
            "dataset_id": dataset,
            "source_fixture_sha256": digest,
            "fixture_sha256": file_sha256(prepared),
            "source_manifest_sha256": manifest_hash,
            "development_selection_sha256": selection_hash,
            "track_groups": list(protocol.groups),
            "run_by_group": runs,
            "base_seeds": seeds,
            "headways": headways,
            "runtime": runtime_identity(),
            "vehicle_parameters": carla_1_0_nissan_patrol().as_dict(),
            "reflex_configurations": {
                name: asdict(reflex(design, value)) for name, value in headways.items()
            },
        },
    )


def completed_rows(
    result: dict[str, Any], expected: set[tuple[str, int]]
) -> dict[tuple[str, int], dict[str, Any]]:
    accepted = {}
    observed = set()
    for row in result["episodes"]:
        key = (row["source_group"], row["base_seed"])
        if key in observed or key not in expected:
            raise ValueError("Unexpected or duplicate arm episode")
        observed.add(key)
        if _complete_episode(row):
            gap = np.asarray(row["predicted_gap_m"]) - row["observed_gap_m"]
            accepted[key] = {**row, BIAS: float(gap.mean())}
    return accepted


def summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {name: mean(row[name] for row in rows) for name in (*METRICS, BIAS)}


def run_summary(
    rows: dict[tuple[str, int], dict[str, Any]],
    groups: list[str],
    seeds: list[int],
    runs: dict[str, str],
) -> dict[str, dict[str, float]]:
    result: dict[str, list[dict[str, float]]] = {}
    for group in groups:
        per_track = summary([rows[group, seed] for seed in seeds])
        result.setdefault(runs[group], []).append(per_track)
    return {
        run: {name: mean(row[name] for row in tracks) for name in (*METRICS, BIAS)}
        for run, tracks in sorted(result.items())
    }


def test_decision(
    records: dict[str, dict[tuple[str, int], dict[str, Any]]],
    inputs: dict[str, Any],
) -> dict[str, Any]:
    design = inputs["design"]
    groups = inputs["track_groups"]
    seeds = inputs["base_seeds"]
    runs = inputs["run_by_group"]
    candidate = next(name for name in records if name != "reference")
    reference, selected = records["reference"], records[candidate]
    pairs = [
        group
        for group in groups
        if all(
            (group, seed) in reference
            and (group, seed) in selected
            and reference[group, seed]["seed"] == selected[group, seed]["seed"]
            for seed in seeds
        )
    ]
    guards = {
        "candidate_complete": len(selected) == len(groups) * len(seeds),
        "complete_track_count": len(pairs) >= design["test_minimum_complete_tracks"],
        "all_runs_represented": {runs[g] for g in pairs} == set(runs.values()),
    }
    result: dict[str, Any] = {"complete_tracks": pairs, "guards": guards}
    if pairs:
        before, after = (
            run_summary(rows, pairs, seeds, runs) for rows in (reference, selected)
        )
        metrics = (*METRICS, BIAS)
        a, b = (
            {name: mean(row[name] for row in per_run.values()) for name in metrics}
            for per_run in (before, after)
        )
        result["conditional_metrics"] = {"reference": a, "candidate": b}
        result["paired_run_means"] = {"reference": before, "candidate": after}
        guards["speed_rmse"] = (
            b[SPEED] <= a[SPEED] + design["maximum_speed_rmse_increase_mps"]
        )
        guards["absolute_signed_gap_bias"] = abs(b[BIAS]) <= abs(a[BIAS])
        if len(before) >= design["test_minimum_run_groups"]:
            delta = np.asarray([after[run][GAP] - before[run][GAP] for run in before])
            ci = bootstrap(
                (delta,),
                np.mean,
                method="percentile",
                confidence_level=0.95,
                n_resamples=design["bootstrap_resamples"],
                random_state=np.random.default_rng(design["bootstrap_seed"]),
            ).confidence_interval
            result["paired_gap_rmse_change_m"] = {
                "mean": float(delta.mean()),
                "interval": [float(ci.low), float(ci.high)],
                "unit": "collection run",
            }
            guards["paired_gap_rmse_reduction"] = bool(ci.high < 0)
    guards.setdefault("paired_gap_rmse_reduction", False)
    result["decision"] = "accepted_contextual" if all(guards.values()) else "rejected"
    return result


def evaluate(design: dict[str, Any], stage: str, out: Path) -> None:
    inputs = read(out / "inputs.json")
    fixture = out / "trials.npz"
    if (
        inputs["design"] != design
        or inputs["stage"] != stage
        or inputs["fixture_sha256"] != file_sha256(fixture)
        or inputs["runtime"] != runtime_identity()
        or inputs["vehicle_parameters"] != carla_1_0_nissan_patrol().as_dict()
    ):
        raise ValueError("Stage data, source or runtime differs from frozen inputs")
    destination = out / ("selection.json" if stage == "development" else "report.json")
    groups, seeds = tuple(inputs["track_groups"]), tuple(inputs["base_seeds"])
    jobs = [
        ArmJob(
            arm=name,
            reflex=config,
            fixture=str(fixture.resolve()),
            groups=groups,
            seeds=seeds,
            output=str((out / f"{name}.json").resolve()),
            seconds=design["seconds"],
            desired_speed=design["desired_speed_mps"],
            vehicle_parameters=inputs["vehicle_parameters"],
            dataset_id=inputs["dataset_id"],
        )
        for name, config in inputs["reflex_configurations"].items()
    ]
    paths = [destination, *(Path(job.output) for job in jobs)]
    if any(path.exists() or path.with_suffix(".jsonl").exists() for path in paths):
        raise FileExistsError("Preserve existing evaluation results")
    expected = {(group, seed) for group in groups for seed in seeds}
    with ProcessPoolExecutor(
        max_workers=len(jobs), mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        results = {row["arm"]: row for row in pool.map(run_arm, jobs)}
    if inputs["runtime"] != runtime_identity():
        raise RuntimeError("Runtime source changed during the evaluation")
    rows = {name: completed_rows(result, expected) for name, result in results.items()}
    report = {
        "input_sha256": file_sha256(out / "inputs.json"),
        "expected_per_arm": len(expected),
        "arms": {
            name: {
                "attempted": len(result["episodes"]),
                "completed": len(rows[name]),
                "failures": dict(
                    Counter(
                        episode.get("failure_kind") or "incomplete"
                        for episode in result["episodes"]
                        if not _complete_episode(episode)
                    )
                ),
            }
            for name, result in results.items()
        },
    }
    if stage == "development":
        scores = {
            name: summary(list(complete.values()))
            for name, complete in rows.items()
            if len(complete) == len(expected)
        }
        baseline = scores.get("reference")
        valid = [
            (score[GAP], -inputs["headways"][name])
            for name, score in scores.items()
            if name != "reference"
            and baseline is not None
            and score[SPEED]
            <= baseline[SPEED] + design["maximum_speed_rmse_increase_mps"]
        ]
        report.update(
            design=design,
            runtime=inputs["runtime"],
            vehicle_parameters=inputs["vehicle_parameters"],
            development_metrics=scores,
            selected_headway_s=-min(valid)[1] if valid else None,
        )
    else:
        report.update(test_decision(rows, inputs))
    write(destination, report)
    print(f"Saved {destination}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("development", "test"), required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prepare", type=Path, help="Prepare the supplied trial NPZ")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--selection", type=Path)
    args = parser.parse_args()
    design = read(args.design)
    if args.prepare:
        if args.stage == "test" and (args.manifest is None or args.selection is None):
            parser.error(
                "test preparation requires its manifest and development selection"
            )
        prepare(
            design, args.stage, args.prepare, args.out, args.manifest, args.selection
        )
    else:
        evaluate(design, args.stage, args.out)


if __name__ == "__main__":
    main()
