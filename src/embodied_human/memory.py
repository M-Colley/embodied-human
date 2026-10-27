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
from numbers import Integral
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

    def __post_init__(self) -> None:
        if (
            isinstance(self.capacity, bool)
            or not isinstance(self.capacity, Integral)
            or self.capacity < 0
        ):
            raise ValueError("capacity must be a non-negative integer")
        for name in ("decay", "threshold", "noise", "latency_factor"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.decay < 0 or self.noise < 0:
            raise ValueError("decay and noise must be non-negative")
        if self.latency_factor <= 0:
            raise ValueError("latency_factor must be positive")

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
            "Stocco et al. (2024), likelihood primer, Fig. 7",
            "https://arxiv.org/abs/2410.18055",
            "assumed",
            "illustrative s = 0.25 setting, not fitted to this simulator's tasks; "
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
        self.reset()

    def reset(self, rng: np.random.Generator | None = None) -> None:
        super().reset(rng)
        self._values: dict[str, Any] = {}
        self._presentations: dict[str, list[float]] = {}
        self.focus: OrderedDict[str, Any] = OrderedDict()
        self._last_encoding_time = -math.inf

    def _check_time(self, time: float) -> None:
        if not math.isfinite(time):
            raise ValueError("time must be finite")
        if time < self._last_encoding_time:
            raise ValueError("time must not precede the latest encoding; reset first")

    def encode(self, key: str, value: Any, time: float) -> None:
        self._check_time(time)
        self._last_encoding_time = float(time)
        self._values[key] = value
        self._presentations.setdefault(key, []).append(float(time))
        self.focus.pop(key, None)
        self.focus[key] = value
        while len(self.focus) > self.params.capacity:
            self.focus.popitem(last=False)

    def base_level(self, key: str, time: float) -> float:
        """Noise-free activation at or after the most recent encoding.

        Only strictly positive trace ages contribute, as the power law is
        singular at zero age. Historical queries require a separate snapshot.
        """
        self._check_time(time)
        ages = [time - t for t in self._presentations.get(key, ()) if time > t]
        if not ages:
            return -math.inf
        if self.params.decay == 0:
            return math.log(len(ages))
        # Sum in log space: direct powers can underflow to zero or overflow.
        terms = [-self.params.decay * math.log(age) for age in ages]
        largest = max(terms)
        if not math.isfinite(largest):
            return largest
        return largest + math.log(sum(math.exp(term - largest) for term in terms))

    def _latency(self, activation: float) -> float:
        # Combining the scale with activation before exponentiation also
        # avoids overflow where F * exp(-A) would still be representable.
        try:
            return math.exp(math.log(self.params.latency_factor) - activation)
        except OverflowError:
            return math.inf

    def retrieve(self, key: str, time: float) -> Retrieval:
        self._check_time(time)
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
                self._latency(activation),
                activation,
            )
        return Retrieval(
            None, False, self._latency(p.threshold), activation
        )

    def recall_probability(self, key: str, time: float) -> float:
        """Probability that a retrieval succeeds (logistic in activation)."""
        self._check_time(time)
        if key in self.focus:
            return 1.0
        activation = self.base_level(key, time)
        if activation == -math.inf:
            return 0.0
        if self.params.noise == 0:
            return float(activation > self.params.threshold)
        z = (activation - self.params.threshold) / self.params.noise
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        odds = math.exp(z)
        return odds / (1.0 + odds)
