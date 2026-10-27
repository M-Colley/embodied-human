"""Small seeded task demonstrations, separate from the published benchmarks."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import ClassVar, cast

import numpy as np

from .bench.benchmarks import Result
from .bench.protocol import protocol_seed
from .core import Human, Parameters, Source
from .decision import PyDDMResponse
from .memory import Memory
from .memory_pyactup import PyACTUpMemory
from .tasks.choice import ChoiceCondition, run_choice_task
from .tasks.recall import run_serial_recall

_INTERPRETATION = (
    "Model demonstration with assumed task settings; "
    "independent human evaluation pending."
)


@dataclass(frozen=True)
class SerialRecallProtocolParameters(Parameters):
    list_length: int = 6
    repetitions: int = 12
    distractor_lists: int = 1
    interval: float = 1.0
    retention: float = 1.0

    SOURCES: ClassVar = {
        name: Source(
            "Assumed serial position demonstration protocol",
            None,
            "assumed",
            note,
        )
        for name, note in {
            "list_length": "items per list, not an empirical memory capacity",
            "repetitions": "simulation repetitions for modest interactive runtime",
            "distractor_lists": "later lists using the same positions and payloads",
            "interval": "seconds between encodings",
            "retention": "seconds after the final encoding before all probes",
        }.items()
    }


@dataclass(frozen=True)
class ChoiceProtocolParameters(Parameters):
    drifts: tuple[float, ...] = (-1.0, 0.0, 1.0)
    deadlines: tuple[float, ...] = (0.5, 2.0)
    trials: int = 1000

    SOURCES: ClassVar = {
        name: Source("Assumed choice demonstration protocol", None, "assumed", note)
        for name, note in {
            "drifts": "decision units per second; not mapped from sensory Signals",
            "deadlines": "seconds, including the model's nondecision time",
            "trials": "independent draws per condition from the native solver grid",
        }.items()
    }


def serial_recall(human: Human) -> Result:
    """Probe serial positions after an interfering list with repeated payloads."""
    result = Result("memory.serial_recall", ("memory",))
    memory = human.memory
    if not isinstance(memory, Memory):
        result.skipped = "selected memory must implement the Memory interface"
        return result
    params = SerialRecallProtocolParameters()
    items = list(range(params.list_length))
    rows = []
    traces = []
    seeds = []
    try:
        for index in range(params.repetitions):
            seed = protocol_seed(index)
            run = run_serial_recall(
                memory,
                items,
                distractors=[items.copy() for _ in range(params.distractor_lists)],
                interval=params.interval,
                retention=params.retention,
                partial=True,
                seed=seed,
            )
            seeds.append(seed)
            rows.append(run.summary())
            traces.append([asdict(probe) for probe in run.probes])
    except ImportError as exc:
        result.skipped = str(exc)
        return result
    result.metrics = {
        "memory.serial_recall_accuracy": float(np.mean([r["accuracy"] for r in rows])),
        "memory.serial_recall_intrusion_rate": float(
            np.mean([r["intrusion_rate"] for r in rows])
        ),
        "memory.serial_recall_failure_rate": float(
            np.mean([r["failure_rate"] for r in rows])
        ),
    }
    result.details = {
        "interpretation": _INTERPRETATION,
        "protocol": asdict(params),
        "protocol_sources": params.provenance(),
        "seeds": seeds,
        "accuracy_by_serial_position": np.mean(
            [row["correct_by_position"] for row in rows], axis=0
        ).tolist(),
        "repetitions": rows,
        "probes": traces,
        "retrieval_api": (
            "immediate focus, then native partial cue retrieval"
            if isinstance(memory, PyACTUpMemory)
            else "exact keyed retrieval; no content intrusions are represented"
        ),
        "scoring": "Repeated payloads from different list identities are intrusions.",
    }
    return result


def choice(human: Human) -> Result:
    """Sample explicit drift and deadline conditions from the selected choice API."""
    result = Result("response.choice", ("response",))
    response = human.response
    if not all(
        callable(getattr(response, name, None))
        for name in ("solve_choice", "sample_choices")
    ):
        result.skipped = (
            "selected response has no explicit choice API; select response=pyddm"
        )
        return result
    params = ChoiceProtocolParameters()
    conditions = [
        ChoiceCondition(drift, deadline, "lower" if drift < 0 else "upper")
        for drift in params.drifts
        for deadline in params.deadlines
    ]
    try:
        rows = run_choice_task(
            cast(PyDDMResponse, response),
            conditions,
            trials=params.trials,
            seed=protocol_seed(),
        )
    except ImportError as exc:
        result.skipped = str(exc)
        return result
    counts = [round(row["trials"] * (1 - row["timeout_rate"])) for row in rows]
    answered = sum(counts)
    mean_rt = (
        sum(row["mean_rt_s"] * count for row, count in zip(rows, counts) if count)
        / answered
        if answered
        else math.nan
    )
    result.metrics = {
        "response.choice_accuracy": float(np.mean([row["accuracy"] for row in rows])),
        "response.choice_timeout_rate": float(
            np.mean([row["timeout_rate"] for row in rows])
        ),
        "response.choice_mean_rt_s": mean_rt,
    }
    result.details = {
        "interpretation": _INTERPRETATION,
        "protocol": asdict(params),
        "protocol_sources": params.provenance(),
        "seed": protocol_seed(),
        "conditions": rows,
        "correct_boundary": (
            "lower for negative drift, upper otherwise; at zero drift upper is an "
            "assumed target label without directional evidence"
        ),
        "mean_rt_definition": "mean among answered trials, weighted by response counts",
        "pathway": "explicit choice API; sensory Signals retain TWIN detection",
    }
    return result


PROTOCOLS: dict[str, Callable[[Human], Result]] = {
    "memory.serial_recall": serial_recall,
    "response.choice": choice,
}

CATALOG = [
    {
        "name": "memory.serial_recall",
        "description": serial_recall.__doc__,
        "categories": ["memory"],
        "metrics": [
            "memory.serial_recall_accuracy",
            "memory.serial_recall_intrusion_rate",
            "memory.serial_recall_failure_rate",
        ],
    },
    {
        "name": "response.choice",
        "description": choice.__doc__,
        "categories": ["response"],
        "metrics": [
            "response.choice_accuracy",
            "response.choice_timeout_rate",
            "response.choice_mean_rt_s",
        ],
    },
]
