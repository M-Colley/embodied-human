"""Compare fixed actuation corrections using fresh independent test sources."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from embodied_human.actuation_evaluation import (
    ActuationEvaluationConfig,
    evaluate_actuation_improvement,
)
from embodied_human.closed_loop_calibration import file_sha256
from embodied_human.reporting import environment_record, runtime_sources, write_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--component-report", type=Path, required=True)
    parser.add_argument(
        "--development-report", action="append", type=Path, required=True
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--development-sources", type=int, default=32)
    parser.add_argument("--test-sources", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--development-seed", type=int, default=17)
    parser.add_argument("--test-seeds", type=int, nargs="+", default=[117, 118])
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--desired-speed", type=float, default=16)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args(argv)
    try:
        configuration = ActuationEvaluationConfig(
            **{
                name: tuple(getattr(args, name))
                if name == "test_seeds"
                else getattr(args, name)
                for name in asdict(ActuationEvaluationConfig())
            }
        )
        if args.out.exists() and any(args.out.iterdir()):
            raise ValueError("output directory must be empty")
        args.out.mkdir(parents=True, exist_ok=True)
        origins = runtime_sources()
        provenance = {
            "runtime_sources": origins,
            "environment": environment_record(),
            "configuration": asdict(configuration),
            "script_sha256": file_sha256(Path(__file__)),
        }
        write_json(args.out / "provenance.json", provenance)

        def progress(event: dict[str, Any]) -> None:
            with (args.out / "progress.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, allow_nan=False) + "\n")
            print(event["event"], flush=True)

        report = evaluate_actuation_improvement(
            args.source,
            component_report=args.component_report,
            development_reports=args.development_report,
            out=args.out,
            config=configuration,
            progress=progress,
        )
        final = runtime_sources()
        changed = [
            name
            for name in set(origins) | set(final)
            if origins.get(name, {}).get("python_source_sha256")
            != final.get(name, {}).get("python_source_sha256")
        ]
        provenance["source_identity_verified"] = not changed
        provenance["changed_runtime_sources"] = changed
        write_json(args.out / "provenance.json", provenance)
        if changed:
            report["status"] = "invalid_source_identity"
            report["runtime_source_identity_verified"] = False
            report["model_update"]["decision"] = "rejected_changed_runtime_sources"
            write_json(args.out / "fullreport.json", report)
            raise RuntimeError(f"source changed during evaluation: {changed}")
        report["runtime_source_identity_verified"] = True
        report["provenance_report"] = "provenance.json"
        write_json(args.out / "fullreport.json", report)
        print(report["model_update"]["decision"], flush=True)
        return 0 if report["status"] == "evaluated" else 2
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Evaluation failed: {type(error).__name__}: {error}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
