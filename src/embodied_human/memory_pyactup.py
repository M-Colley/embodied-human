"""Optional PyACTUp declarative memory with explicit feature based retrieval.

The keyed API retains the package's immediate focus. Cue retrieval and blending
use the installed MIT PyACTUp implementation and access declarative memory
directly. They do not claim to implement the Oberauer and Lin interference model.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass, field
from importlib import import_module
from types import FunctionType
from typing import Any, ClassVar

import numpy as np

from .core import Source, register
from .memory import ActRMemory, MemoryParameters, Retrieval

_SOURCE = "https://github.com/cmu-psych-fms/pyactup/blob/main/pyactup.py"
_KEY = "_eh_key"


@dataclass(frozen=True)
class PyACTUpParameters(MemoryParameters):
    """Baseline focus and decay plus assumed settings for cue competition."""

    mismatch_penalty: float = 1.0
    blend_temperature: float = 1.0

    SOURCES: ClassVar = {
        **MemoryParameters.SOURCES,
        "mismatch_penalty": Source(
            "PyACTUp mismatch scaling; illustrative adapter setting",
            _SOURCE,
            "assumed",
            "activation penalty per unit dissimilarity, not fitted human data",
        ),
        "blend_temperature": Source(
            "PyACTUp blending temperature; illustrative adapter setting",
            _SOURCE,
            "assumed",
            "explicit temperature so noise can be set to zero independently; "
            "not the upstream noise-dependent default",
        ),
    }

    def __post_init__(self) -> None:
        super().__post_init__()
        if not math.isfinite(self.mismatch_penalty) or self.mismatch_penalty < 0:
            raise ValueError("mismatch_penalty must be finite and non-negative")
        if not math.isfinite(self.blend_temperature) or self.blend_temperature < 0.01:
            raise ValueError("blend_temperature must be finite and at least 0.01")


@dataclass(frozen=True)
class CuedRetrieval(Retrieval):
    """A native retrieval, including the identity needed to score intrusions."""

    key: str | None = None
    features: Mapping[str, Hashable] = field(default_factory=dict)


def _local_random(function: Any, stream: random.Random) -> Callable[..., Any]:
    """Run upstream bytecode with a private Python random binding.

    PyACTUp 2.2.5 uses module global random in its constructor and retrieval
    tie breaking. A new function object with copied globals changes neither
    that module nor Python's global RNG, and needs no copied implementation.
    """
    rebound = FunctionType(
        function.__code__,
        {**function.__globals__, "random": stream},
        function.__name__,
        function.__defaults__,
        function.__closure__,
    )
    rebound.__kwdefaults__ = function.__kwdefaults__
    return rebound


@register("memory", "pyactup")
class PyACTUpMemory(ActRMemory):
    """Keyed focus plus native PyACTUp cue retrieval, similarity and blending.

    Construction and reset do not import PyACTUp. First declarative use needs
    ``pip install embodied-human[memory]``. Seconds in this package map directly
    to PyACTUp's model time. Explicit cue queries must follow the latest
    encoding strictly, as upstream requires positive trace ages.
    """

    params: PyACTUpParameters

    def __init__(self, **params: Any) -> None:
        # ActRMemory's constructor accepts only the inherited parameters.
        settings = PyACTUpParameters(**params)
        super().__init__(
            **{name: getattr(settings, name) for name in MemoryParameters.SOURCES}
        )
        self.params = settings

    def reset(self, rng: np.random.Generator | None = None) -> None:
        super().reset(rng)
        self._features: dict[str, dict[str, Hashable]] = {}
        self._similarities: dict[str, tuple[Callable[..., float] | bool, float]] = dict(
            getattr(self, "_similarities", {})
        )
        self._backend: Any = None
        self._native_retrieve: Callable[..., Any] | None = None
        self._python_rng = random.Random(int(self.rng.integers(0, 2**63)))

    def _native(self) -> Any:
        if self._backend is not None:
            return self._backend
        try:
            native = import_module("pyactup")
        except ModuleNotFoundError as exc:
            if exc.name != "pyactup":
                raise
            raise ImportError(
                "PyACTUp declarative retrieval requires "
                "python -m pip install 'embodied-human[memory]'"
            ) from exc
        p = self.params
        backend = native.Memory.__new__(native.Memory)
        _local_random(native.Memory.__init__, self._python_rng)(
            backend,
            noise=p.noise,
            decay=p.decay,
            temperature=p.blend_temperature,
            threshold=p.threshold,
            mismatch=p.mismatch_penalty,
        )
        backend.noise_distribution = lambda: float(self.rng.logistic())
        for name, (function, weight) in self._similarities.items():
            backend.similarity(name, function, weight=weight)
        history = sorted(
            (time, key)
            for key, times in self._presentations.items()
            for time in times
        )
        for time, key in history:
            backend.time = time
            backend.learn({_KEY: key, **self._features[key]})
        self._native_retrieve = _local_random(native.Memory.retrieve, self._python_rng)
        self._backend = backend
        return backend

    def encode(self, key: str, value: Any, time: float) -> None:
        self.encode_chunk(key, value, time, self._features.get(key, {}))

    def encode_chunk(
        self, key: str, value: Any, time: float, features: Mapping[str, Hashable]
    ) -> None:
        """Encode a value with immutable feature bindings for this key.

        Repeating a key rehearses it and may replace its payload. Use a new
        key for a different episode or changed feature binding. Payloads may
        be arbitrary objects; only native feature values must be hashable.
        """
        self._check_time(time)
        if time < 0:
            raise ValueError("PyACTUp times must be non-negative")
        if not isinstance(key, str) or not key:
            raise ValueError("key must be a nonempty string")
        attributes = dict(features)
        for name, item in attributes.items():
            if not isinstance(name, str) or not name or name == _KEY:
                raise ValueError("feature names must be nonempty and not '_eh_key'")
            try:
                hash(item)
            except TypeError:
                raise ValueError(f"feature {name!r} must be hashable") from None
            if isinstance(item, float) and not math.isfinite(item):
                raise ValueError(f"feature {name!r} must be finite")
        if key in self._features and attributes != self._features[key]:
            raise ValueError("use a new key when feature bindings change")
        self._features[key] = attributes
        if self._backend is not None:
            self._backend.time = time
            self._backend.learn({_KEY: key, **attributes})
        super().encode(key, value, time)

    def similarity(
        self,
        attribute: str,
        function: Callable[[Hashable, Hashable], float] | None = None,
        *,
        weight: float = 1.0,
    ) -> None:
        """Configure native partial matching for one feature.

        The function must be symmetric and return a similarity in [0, 1].
        ``None`` uses native binary equality. Weight and any custom similarity
        are experimental choices requiring calibration, not human constants.
        """
        if not isinstance(attribute, str) or not attribute or attribute == _KEY:
            raise ValueError("attribute must name a non-reserved feature")
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("similarity weight must be finite and non-negative")
        if function is not None and not callable(function):
            raise ValueError("similarity function must be callable or None")
        selected: Callable[..., float] | bool = True if function is None else function
        self._similarities[attribute] = (selected, weight)
        if self._backend is not None:
            self._backend.similarity(attribute, selected, weight=weight)

    def retrieve(self, key: str, time: float) -> Retrieval:
        self._check_time(time)
        if key in self.focus:
            return Retrieval(self.focus[key], True, 0.0, math.inf)
        return self.retrieve_cued({_KEY: key}, time, partial=False)

    def retrieve_cued(
        self, cues: Mapping[str, Hashable], time: float, *, partial: bool = True
    ) -> CuedRetrieval:
        """Retrieve a native chunk, without rehearsing it or bypassing via focus."""
        self._check_query_time(time)
        backend = self._native()
        backend.time = time
        backend.activation_history = []
        assert self._native_retrieve is not None
        chunk = self._native_retrieve(backend, dict(cues), partial=partial)
        if chunk is None:
            return CuedRetrieval(
                None,
                False,
                self._latency(self.params.threshold),
                max(
                    (float(row["activation"]) for row in backend.activation_history),
                    default=-math.inf,
                ),
            )
        key = chunk[_KEY]
        record = next(
            row
            for row in backend.activation_history
            if dict(row["attributes"])[_KEY] == key
        )
        activation = float(record["activation"])
        return CuedRetrieval(
            self._values[key],
            True,
            self._latency(activation),
            activation,
            key,
            dict(self._features[key]),
        )

    def blend(
        self, attribute: str, cues: Mapping[str, Hashable], time: float
    ) -> float | None:
        """Native activation weighted blend of a numeric feature, not a payload."""
        self._check_query_time(time)
        if attribute == _KEY:
            raise ValueError("the internal identity feature cannot be blended")
        backend = self._native()
        backend.time = time
        result = backend.blend(attribute, dict(cues))
        return None if result is None else float(result)

    def _check_query_time(self, time: float) -> None:
        self._check_time(time)
        if time < 0 or time <= self._last_encoding_time:
            raise ValueError("declarative queries must follow the latest encoding")

    def recall_probability(self, key: str, time: float) -> float:
        # Analytic success probability for the one exact keyed candidate.
        self._check_time(time)
        if key in self.focus:
            return 1.0
        if self.params.noise == 0:
            return float(self.base_level(key, time) >= self.params.threshold)
        return super().recall_probability(key, time)
