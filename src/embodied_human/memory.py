"""Memory: a capacity-limited focus of attention over ACT-R declarative memory.

Items in the focus (Cowan's capacity limit of about four chunks) are
available immediately. Displaced items must be retrieved from declarative
memory, whose activation follows ACT-R's base-level learning

    A = ln(sum_j (t - t_j)^-d) + noise,

with logistic noise of scale ``s``. A retrieval succeeds when A exceeds the
threshold ``tau`` and takes ``F exp(-A)`` seconds; a failure takes
``F exp(-tau)`` (Anderson et al., 2004).
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from .core import Module, Parameters, Source, register


@dataclass(frozen=True)
class MemoryParameters(Parameters):
    """Focus capacity and ACT-R declarative-memory parameters."""

    capacity: int = 4
    decay: float = 0.5
    threshold: float = 0.0
    noise: float = 0.25
    latency_factor: float = 1.0

    SOURCES: ClassVar = {
        "capacity": Source(
            "Cowan (2001)",
            "https://doi.org/10.1017/S0140525X01003922",
            "measured",
            "about 4 chunks (3-5) when rehearsal and chunking are prevented",
        ),
        "decay": Source(
            "Anderson et al. (2004)",
            "https://doi.org/10.1037/0033-295X.111.4.1036",
            "fitted",
            "conventional base-level decay",
        ),
        "threshold": Source("ACT-R default (:rt 0)", None, "standard"),
        "noise": Source(
            "Stocco et al. (2024) example value",
            None,
            "fitted",
            "ACT-R ships with noise off",
        ),
        "latency_factor": Source("ACT-R default (:lf 1.0)", None, "standard"),
    }


@dataclass(frozen=True)
class Retrieval:
    """The outcome of one retrieval."""

    value: Any
    success: bool
    latency: float
    activation: float


class Memory(Module):
    """Category ``memory``: store and retrieve keyed items over time."""

    category = "memory"

    def encode(self, key: str, value: Any, time: float) -> None:
        """Present (or rehearse) ``key`` with ``value`` at ``time``."""
        raise NotImplementedError

    def retrieve(self, key: str, time: float) -> Retrieval:
        """Try to recall ``key`` at ``time``."""
        raise NotImplementedError


@register("memory", "act_r")
class ActRMemory(Memory):
    """Focus of attention plus ACT-R base-level declarative memory."""

    def __init__(self, **params: Any) -> None:
        super().__init__()
        self.params = MemoryParameters(**params)
        if self.params.capacity < 0:
            raise ValueError("capacity must be non-negative")
        self.reset()

    def reset(self, rng: np.random.Generator | None = None) -> None:
        super().reset(rng)
        self._values: dict[str, Any] = {}
        self._presentations: dict[str, list[float]] = {}
        self.focus: OrderedDict[str, Any] = OrderedDict()

    def encode(self, key: str, value: Any, time: float) -> None:
        self._values[key] = value
        self._presentations.setdefault(key, []).append(float(time))
        self.focus.pop(key, None)
        self.focus[key] = value
        while len(self.focus) > self.params.capacity:
            self.focus.popitem(last=False)

    def base_level(self, key: str, time: float) -> float:
        """Noise-free base-level activation of ``key`` at ``time``."""
        ages = [time - t for t in self._presentations.get(key, ()) if time > t]
        if not ages:
            return -math.inf
        return math.log(sum(age ** (-self.params.decay) for age in ages))

    def retrieve(self, key: str, time: float) -> Retrieval:
        p = self.params
        if key in self.focus:
            return Retrieval(self.focus[key], True, 0.0, math.inf)
        activation = self.base_level(key, time)
        if p.noise > 0 and math.isfinite(activation):
            u = float(self.rng.uniform(1e-12, 1 - 1e-12))
            activation += p.noise * math.log(u / (1 - u))
        if activation > p.threshold:
            return Retrieval(
                self._values[key],
                True,
                p.latency_factor * math.exp(-activation),
                activation,
            )
        return Retrieval(
            None, False, p.latency_factor * math.exp(-p.threshold), activation
        )

    def recall_probability(self, key: str, time: float) -> float:
        """Probability that a retrieval succeeds (logistic in activation)."""
        if key in self.focus:
            return 1.0
        activation = self.base_level(key, time)
        if not math.isfinite(activation):
            return 0.0
        if self.params.noise == 0:
            return float(activation > self.params.threshold)
        return 1.0 / (
            1.0 + math.exp(-(activation - self.params.threshold) / self.params.noise)
        )
