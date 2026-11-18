"""Diagnose an inspected following failure without fitting controller parameters."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np

from embodied_human.closed_loop_calibration import file_sha256
from embodied_human.reporting import environment_record, runtime_sources
from embodied_human.steering_diagnostics import (
    run_steering_comparison,
    signed_following_errors,
)


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: Any, *, compressed: bool = False) -> None:
    payload = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        if compressed:
            with gzip.GzipFile(fileobj=stream, mode="wb", mtime=0) as archive:
                archive.write(payload)
        else:
            stream.write(payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-group", default="ngsim-us101:1761:1")
    parser.add_argument("--base-seed", type=int, default=702)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Preserve the existing diagnostic directory")
    from myosuite_carla.driver_models import ReflexDriverConfig
    from myosuite_carla.empirical_following import load_following_protocol
    from myosuite_carla.ngsim import save_trials
    from myosuite_carla.vehicle_model import VehicleModelParams

    inputs = read(args.study / "frozen-inputs.json")
    archived = {
        arm: read(args.study / f"{arm}.json") for arm in ("baseline", "inverse")
    }
    cohort = read(args.study / "report.json")["descriptive_complete_groups"]
    if file_sha256(args.study / "trials.npz") != inputs["fixture_sha256"]:
        raise ValueError("The original source fixture changed")
    source = load_following_protocol(
        args.study / "trials.npz",
        max_groups=None,
        dataset_id=inputs["design"]["dataset_id"],
        desired_speed=inputs["design"]["desired_speed_mps"],
    )
    trial = source.trials_for(args.source_group)[0]
    expected = next(
        row
        for row in archived["baseline"]["episodes"]
        if row["source_group"] == args.source_group
        and row["base_seed"] == args.base_seed
    )
    origins = runtime_sources()
    result = run_steering_comparison(
        trial,
        VehicleModelParams.from_dict(inputs["vehicle_parameters"]),
        ReflexDriverConfig.from_dict(inputs["reflex"]["baseline"]),
        seed=expected["seed"],
        desired_speed=inputs["design"]["desired_speed_mps"],
    )
    reference = result["episodes"]["reference"]
    differences = {}
    for key, value in reference["trajectory"].items():
        actual, original = np.asarray(value), np.asarray(expected[key])
        differences[key] = (
            float(np.max(np.abs(actual - original)))
            if actual.shape == original.shape
            else None
        )
    result["archive_reproduction"] = {
        "same_failure": reference["failure_reason"] == expected["failure_kind"],
        "end_time_error_s": abs(result["comparison_horizon_s"] - expected["seconds"]),
        "maximum_absolute_trajectory_errors": differences,
        "passed": (
            reference["failure_reason"] == expected["failure_kind"]
            and abs(result["comparison_horizon_s"] - expected["seconds"]) < 1e-8
            and all(
                value is not None and value < 1e-8 for value in differences.values()
            )
        ),
    }
    result["source_group"] = args.source_group
    result["base_seed"] = args.base_seed
    args.out.mkdir(parents=True)
    save_trials([trial], args.out / "trial.npz")
    signed = signed_following_errors(
        {arm: record["episodes"] for arm, record in archived.items()}, cohort
    )
    write(args.out / "signed-following-errors.json", signed)
    summaries = {}
    for arm, episode in result["episodes"].items():
        write(args.out / f"{arm}.json.gz", episode, compressed=True)
        trace = episode["trace"]
        summaries[arm] = {
            "status": episode["status"],
            "failure_reason": episode["failure_reason"],
            "steps": len(trace),
            "seconds": trace[-1]["application_time_s"],
            "maximum_absolute_lane_offset_m": max(
                abs(r["true_lane_offset_after_m"]) for r in trace
            ),
            "wheel_tracking_rmse": float(
                np.sqrt(np.mean([r["wheel_tracking_error"] ** 2 for r in trace]))
            ),
            "surrogate_closed": episode["surrogate_closed"],
        }
    result["episodes"] = summaries
    final_origins = runtime_sources()
    result["provenance"] = {
        "runtime_sources": origins,
        "environment": environment_record(),
        "source_study_fixture_sha256": inputs["fixture_sha256"],
        "source_baseline_report_sha256": file_sha256(args.study / "baseline.json"),
        "script_sha256": file_sha256(Path(__file__)),
        "runtime_source_identity_verified": all(
            origins[name].get("python_source_sha256")
            == final_origins[name].get("python_source_sha256")
            for name in origins
        ),
    }
    write(args.out / "summary.json", result)
    print(
        json.dumps(
            {
                "reproduction": result["archive_reproduction"],
                "parity": result["parity"],
                "episodes": summaries,
            },
            indent=2,
        )
    )
    verified = (
        result["archive_reproduction"]["passed"]
        and result["provenance"]["runtime_source_identity_verified"]
        and all(all(checks.values()) for checks in result["parity"].values())
    )
    return 0 if verified else 2


if __name__ == "__main__":
    raise SystemExit(main())
