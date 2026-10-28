"""Participant grouped calibration of self paced visual fixation.

The existing attention, estimator, vision and eye modules generate every
prediction. This component protocol holds lateral truth at zero. It supplies
no steering dynamics or muscle force validation.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .core import HumanConfig, build_human
from .driving_data import DatasetBundle
from .driving_evaluation import (
    GroupSplit,
    SubjectSummaries,
    calibrate_grouped,
    group_split,
)
from .inference import Prior

METRICS = ("mean_target_glance_s", "sd_target_glance_s", "long_target_share")


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _model_source_sha256(root: Path | None = None) -> str:
    """Bind cached predictions to relative Python paths and normalized source."""
    package = Path(__file__).resolve().parent if root is None else root
    files = {}
    for path in sorted(package.rglob("*.py")):
        content = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        files[path.relative_to(package).as_posix()] = hashlib.sha256(
            content
        ).hexdigest()
    return _canonical_sha256(files)


def _summary(durations: Any) -> dict[str, float]:
    values = np.asarray(durations, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    return dict(
        zip(
            METRICS,
            (
                float(values.mean()) if len(values) else math.nan,
                float(values.std(ddof=1)) if len(values) > 1 else math.nan,
                float(np.mean(values > 2)) if len(values) else math.nan,
            ),
        )
    )


def _average(rows: list[dict[str, float]]) -> dict[str, float]:
    return {metric: float(np.mean([row[metric] for row in rows])) for metric in METRICS}


@dataclass(frozen=True)
class Context:
    condition: str
    speed_mps: float
    target: tuple[float, float]


class FixationProtocol:
    """The actual attentive driver with the source's fixed target task.

    Target centres are estimated from training participants. Speeds are
    measured trial inputs. Target region size remains an explicit assumption.
    """

    def __init__(
        self, bundle: DatasetBundle, *, seconds: float = 30, seed: int = 0
    ) -> None:
        if bundle.glances is None or bundle.trials is None:
            raise ValueError("glance intervals and trial exposure are required")
        if not math.isfinite(seconds) or seconds < 10:
            raise ValueError("seconds must be finite and at least ten")
        self.seconds = seconds
        data = bundle.glances
        data = data[
            (data.area == "target") & data.condition.isin(("down1", "down2"))
        ].copy()
        if "source_area" in data:
            data = data[data.source_area == data.condition]
        if "boundary" in data:
            data = data[~data.boundary]
        for flag in ("left_censored", "right_censored"):
            if flag in data:
                data = data[~data[flag].fillna(False)]
        self.observed: dict[str, dict[str, float]] = {}
        trials_by_group: dict[str, list[Any]] = {}
        for subject, group in data.groupby("subject_id", sort=True):
            trials = [t for _, t in group.groupby("trial_id", sort=True)]
            summaries = [_summary(t.duration_s) for t in trials]
            if summaries and all(
                math.isfinite(v) for row in summaries for v in row.values()
            ):
                key = f"{bundle.dataset_id}:{subject}"
                self.observed[key] = _average(summaries)
                trials_by_group[key] = trials
        self.split: GroupSplit = group_split(tuple(self.observed), seed=seed)
        train_ids = {group.split(":", 1)[1] for group in self.split.train}
        geometry = {}
        for condition, group in data[data.subject_id.isin(train_ids)].groupby(
            "condition"
        ):
            centres = (
                group.groupby("subject_id")[["heading_deg", "pitch_deg"]]
                .median()
                .mean()
            )
            geometry[str(condition)] = (
                float(centres.heading_deg),
                float(centres.pitch_deg),
            )
        self.contexts = {
            group: [
                Context(
                    str(t.condition.iloc[0]),
                    float(t.speed_mps.iloc[0]),
                    geometry[str(t.condition.iloc[0])],
                )
                for t in trials
            ]
            for group, trials in trials_by_group.items()
        }
        self.metadata = {
            "role": "attention_component",
            "backend": "AttentiveDriver, FixationTask, StraightRoad, HoldLane",
            "simulated_seconds_per_trial": seconds,
            "target_centres_deg": geometry,
            "target_geometry_source": (
                "training participant median gaze centres, then equal participant mean"
            ),
            "target_half_extent_deg": [2, 2],
            "target_extent_provenance": "assumed AOI classification region",
            "measurement": (
                "complete target AOI runs; trial summaries "
                "averaged within each participant"
            ),
            "source": bundle.source,
            "available_participants": int(bundle.glances.subject_id.nunique()),
            "eligible_participants": len(self.observed),
            "limitations": [
                "Apparatus calibration is absent; training gaze estimates targets.",
                "Pursuit and outside targets are audited and excluded from this fit.",
                "Lateral truth is held at zero; physical driving needs separate tests.",
                "Human censoring is unreported; boundary intervals are excluded.",
            ],
        }
        self.metadata["cache_identity"] = {
            "schema_version": 1,
            "numerical_environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": importlib.metadata.version("scipy"),
                "pandas": importlib.metadata.version("pandas"),
            },
            "observations_contexts_settings_sha256": _canonical_sha256(
                {
                    "observed": self.observed,
                    "contexts": {
                        group: [asdict(context) for context in contexts]
                        for group, contexts in self.contexts.items()
                    },
                    "settings": {
                        "seconds": seconds,
                        "control_dt_s": 0.02,
                        "target_half_extent_deg": [2, 2],
                        "metrics": METRICS,
                        "partitions": asdict(self.split),
                        "baseline_human_config": HumanConfig().as_dict(),
                    },
                }
            ),
            "model_source_sha256": _model_source_sha256(),
        }

    def __call__(
        self, parameters: Mapping[str, float], groups: tuple[str, ...], seed: int
    ) -> SubjectSummaries:
        from .agents.driver import AttentiveDriver
        from .agents.road import HoldLane, StraightRoad
        from .tasks.fixation import FixationTask

        config = HumanConfig().replace(
            "attention", "uncertainty_sampler", **dict(parameters)
        )
        result = {}
        for group in groups:
            summaries = []
            for index, context in enumerate(self.contexts[group]):
                identity = f"{seed}/{group}/{context.condition}/{index}".encode()
                simulation_seed = int.from_bytes(
                    hashlib.sha256(identity).digest()[:4], "little"
                )
                road = StraightRoad(speed=context.speed_mps, duration=self.seconds)
                agent = AttentiveDriver(
                    build_human(config, seed=simulation_seed),
                    road,
                    FixationTask(context.target, (2, 2)),
                    reflex=HoldLane(),
                )
                road.reset(simulation_seed)
                agent.reset(simulation_seed)
                for _ in range(math.ceil(self.seconds / road.dt)):
                    road.step(agent.act())
                durations, start = [], None
                for time, area in zip(agent.log.times, agent.log.areas):
                    if area == "display" and start is None:
                        start = time
                    elif area != "display" and start is not None:
                        durations.append(time - start)
                        start = None
                summaries.append(_summary(durations))
            result[group] = _average(summaries)
        return result


def calibrate_fixation(
    bundle: DatasetBundle, *, samples: int = 64, seconds: float = 30, seed: int = 0
) -> dict[str, Any]:
    """Fit on training participants and select on validation participants."""
    from .agents.driver import DriverAgentParameters
    from .attention import SamplerParameters

    protocol = FixationProtocol(bundle, seconds=seconds, seed=seed)
    defaults = SamplerParameters()
    baseline = {
        name: float(getattr(defaults, name))
        for name in ("return_sd", "threshold_spread", "maximum_glance")
    }
    report = calibrate_grouped(
        protocol.observed,
        protocol,
        (
            Prior("return_sd", 0.02, 0.6, log=True),
            Prior("threshold_spread", 0.05, 2),
            Prior("maximum_glance", 2, 12),
        ),
        baseline,
        metrics=METRICS,
        split=protocol.split,
        samples=samples,
        accept=0.125,
        seed=seed,
        fit_seeds=(0, 1),
        evaluation_seeds=(100, 101),
        configuration={
            "human": HumanConfig().as_dict(),
            "driver": asdict(DriverAgentParameters()),
            "reflex": {"backend": "HoldLane", "vehicle_dynamics": False},
        },
        provenance=protocol.metadata,
        bootstrap=500,
        max_validation_candidates=8,
    )
    report["protocol"] = protocol.metadata
    report["profile"]["module"] = "attention.uncertainty_sampler"
    report["profile"]["context"] = "self paced in vehicle visual fixation"
    return report
