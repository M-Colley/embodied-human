"""Serial position probes with explicit distractor lists and intrusion scores.

This is an assumed demonstration protocol, not a published human benchmark.
All positions are probed at the same final time, without output rehearsal, so
differences reflect encoding history and cue competition rather than probe order.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..memory import Memory, Retrieval
from ..memory_pyactup import PyACTUpMemory


@dataclass(frozen=True)
class RecallProbe:
    position: int
    target_key: str
    recalled_key: str | None
    correct: bool
    intrusion: bool
    success: bool
    latency: float
    content_error: bool = False


@dataclass(frozen=True)
class SerialRecallResult:
    probes: tuple[RecallProbe, ...]

    def summary(self) -> dict[str, Any]:
        count = len(self.probes)
        return {
            "probes": count,
            "accuracy": sum(p.correct for p in self.probes) / count,
            "intrusion_rate": sum(p.intrusion for p in self.probes) / count,
            "failure_rate": sum(not p.success for p in self.probes) / count,
            "content_error_rate": sum(p.content_error for p in self.probes) / count,
            "correct_by_position": [p.correct for p in self.probes],
            "note": "Assumed serial position probe protocol, not human validation",
        }


def run_serial_recall(
    memory: Memory,
    items: Sequence[Any],
    *,
    distractors: Sequence[Sequence[Any]] = (),
    interval: float = 1.0,
    retention: float = 1.0,
    partial: bool = True,
    seed: int = 0,
) -> SerialRecallResult:
    """Encode a target list and later lists, then probe each target position.

    PyACTUp binds ``position`` and ``list_context`` features, with binary
    partial matching on context after the target leaves its immediate focus.
    Other Memory implementations use exact target keys. Correctness requires
    both the target identity and matching content. Successful keyed responses
    with wrong content are counted separately because their source identity is
    unknown. A native distractor with the same payload is still an intrusion.
    """
    if not items:
        raise ValueError("items must be nonempty")
    if not all(math.isfinite(x) and x > 0 for x in (interval, retention)):
        raise ValueError("interval and retention must be positive and finite")
    if any(len(row) != len(items) for row in distractors):
        raise ValueError("each distractor list must have the target list's length")
    memory.reset(np.random.default_rng(seed))
    native = memory if isinstance(memory, PyACTUpMemory) else None
    if native is not None:
        native.similarity("list_context")
    time = 0.0
    for context, row in enumerate((items, *distractors)):
        for position, value in enumerate(row):
            key = f"list{context}:pos{position}"
            if native is not None:
                native.encode_chunk(
                    key, value, time, {"position": position, "list_context": context}
                )
            else:
                memory.encode(key, value, time)
            time += interval
    probe_time = time - interval + retention
    probes = []
    result: Retrieval
    for position in range(len(items)):
        target = f"list0:pos{position}"
        identity_observed = native is not None and target not in native.focus
        if native is not None and identity_observed:
            cued = native.retrieve_cued(
                {"position": position, "list_context": 0}, probe_time, partial=partial
            )
            result = cued
            recalled = cued.key
        else:
            result = memory.retrieve(target, probe_time)
            recalled = target if result.success else None
        try:
            content_matches = bool(np.all(result.value == items[position]))
        except (TypeError, ValueError):
            content_matches = False
        correct = result.success and recalled == target and content_matches
        intrusion = result.success and identity_observed and recalled != target
        probes.append(
            RecallProbe(
                position,
                target,
                recalled,
                correct,
                intrusion,
                result.success,
                result.latency,
                result.success and not correct and not intrusion,
            )
        )
    return SerialRecallResult(tuple(probes))
