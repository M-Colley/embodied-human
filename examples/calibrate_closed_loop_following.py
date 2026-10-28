"""Calibrate following through the physical human and fitted vehicle loop."""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path
from typing import Any

from embodied_human.closed_loop_calibration import (
    ClosedLoopCalibrationConfig,
    calibrate_closed_loop_following,
    file_sha256,
)
from embodied_human.reporting import environment_record, runtime_sources, write_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--exclude-report", action="append", type=Path, default=[])
    defaults = ClosedLoopCalibrationConfig()
    for field in fields(defaults):
        value = getattr(defaults, field.name)
        parser.add_argument(
            "--" + field.name.replace("_", "-"), type=type(value), default=value
        )
    args = parser.parse_args(argv)
    try:
        configuration = ClosedLoopCalibrationConfig(
            **{field.name: getattr(args, field.name) for field in fields(defaults)}
        )
        args.out.mkdir(parents=True, exist_ok=True)
        if any(
            (args.out / name).exists()
            for name in (
                "design.json",
                "progress.jsonl",
                "provenance.json",
                "fullreport.json",
            )
        ):
            raise ValueError("output directory already contains calibration evidence")
        origins = runtime_sources()
        provenance = {
            "runtime_sources": origins,
            "environment": environment_record(),
            "script_sha256": file_sha256(Path(__file__)),
            "source": {
                "path": str(args.source.resolve()),
                "sha256": file_sha256(args.source),
            },
        }
        write_json(args.out / "provenance.json", provenance)

        def progress(event: dict[str, Any]) -> None:
            if event["event"] == "design_frozen":
                write_json(args.out / "design.json", event["design"])
            with (args.out / "progress.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, allow_nan=False) + "\n")
            if event["event"] == "objective_completed":
                print(
                    f"objective {event['call']}: {len(event['groups'])} sources, "
                    f"{event['diagnostics']['episode_runs']} physical episodes, "
                    f"{event['elapsed_seconds']:.1f} seconds",
                    flush=True,
                )

        report = calibrate_closed_loop_following(
            args.source,
            config=configuration,
            exclude_reports=args.exclude_report,
            progress=progress,
        )
        final_origins = runtime_sources()
        changed = [
            name
            for name in set(origins) | set(final_origins)
            if origins.get(name, {}).get("python_source_sha256")
            != final_origins.get(name, {}).get("python_source_sha256")
        ]
        provenance["source_identity_verified"] = not changed
        provenance["changed_runtime_sources"] = changed
        write_json(args.out / "provenance.json", provenance)
        if changed:
            raise RuntimeError(f"source changed during calibration: {changed}")
        report["runtime_source_identity_verified"] = True
        report["provenance_report"] = "provenance.json"
        write_json(args.out / "fullreport.json", report)
        print(report["model_update"]["decision"], flush=True)
        return 0 if report["status"] == "evaluated" else 2
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Calibration failed: {type(error).__name__}: {error}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
