"""Delayed digit recall (auditory n-back), as used by the MIT AgeLab.

Spoken digits 0-9 arrive one every 2.25 s. At level n the participant
says the digit presented n positions earlier (0-back repeats the current
one). The task loads working memory without visual demand (Mehler et al.,
2009; Mehler, Reimer and Coughlin, 2012).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..memory import Memory


@dataclass(frozen=True)
class NBackConfig:
    """Delayed digit recall."""

    level: int = 2
    interval: float = 2.25
    digits: int = 10


@dataclass
class NBackResult:
    """Responses per scored stimulus."""

    correct: list[bool] = field(default_factory=list)
    latencies: list[float] = field(default_factory=list)

    def summary(self) -> dict[str, float]:
        """Accuracy and mean response latency."""
        n = len(self.correct)
        return {
            "responses": float(n),
            "accuracy": float(np.mean(self.correct)) if n else float("nan"),
            "mean_latency_s": float(np.mean(self.latencies)) if n else float("nan"),
        }


def run_nback(
    memory: Memory, config: NBackConfig, rng: np.random.Generator, start: float = 0.0
) -> NBackResult:
    """Run one block with a memory module answering from what it retrieves.

    Each digit is encoded as ``pos<k>``; the answer is the retrieval of
    ``pos<k - n>``. Retrieval failures count as errors.
    """
    result = NBackResult()
    sequence = rng.integers(0, 10, size=config.digits)
    for k, digit in enumerate(sequence):
        time = start + k * config.interval
        memory.encode(f"pos{k}", int(digit), time)
        if k >= config.level:
            retrieval = memory.retrieve(f"pos{k - config.level}", time + 0.01)
            result.correct.append(
                retrieval.success and retrieval.value == int(sequence[k - config.level])
            )
            result.latencies.append(retrieval.latency)
    return result
