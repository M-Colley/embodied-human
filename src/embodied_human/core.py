"""Module categories, the registry and cited parameters.

A simulated human is composed of one module per category. Each category
has an abstract interface:

- senses: ``eyes``, ``vision``, ``objects``, ``hearing``, ``touch``,
  ``vestibular`` and ``proprioception``;
- cognition: ``attention``, ``memory``, ``estimator`` and ``response``
  (turning detected signals into responses);
- action: ``motor`` and ``fatigue``;
- experience and motivation: ``affect``, ``risk`` and ``trust``.

Any implementation registered under a category can replace the baseline,
and the benchmark scores implementations per category and in integrated
tasks.

Every default parameter carries a :class:`Source`, so reports can separate
values measured in humans from values fitted or assumed by a model.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, TypeVar

import numpy as np

CATEGORIES = (
    "eyes",
    "vision",
    "objects",
    "attention",
    "memory",
    "proprioception",
    "motor",
    "estimator",
    "hearing",
    "touch",
    "vestibular",
    "response",
    "fatigue",
    "affect",
    "risk",
    "trust",
)

Kind = Literal["measured", "standard", "fitted", "assumed", "derived"]


@dataclass(frozen=True)
class Source:
    """Where a parameter value comes from.

    Args:
        citation: Short reference, e.g. ``"Carpenter (1988)"``.
        doi: DOI or URL of the source.
        kind: ``measured`` in humans, set by a ``standard``, ``fitted`` to
            human data by a model, ``derived`` from other parameters, or
            ``assumed`` (a modelling choice to be fitted or replaced).
        note: Conditions under which the value holds.
    """

    citation: str
    doi: str | None = None
    kind: Kind = "measured"
    note: str = ""


class Parameters:
    """Mixin for frozen parameter dataclasses with per-field sources."""

    SOURCES: ClassVar[Mapping[str, Source]] = {}

    def provenance(self) -> list[dict[str, Any]]:
        """Return each field's value and source, for reports."""
        rows = []
        for item in dataclasses.fields(self):  # type: ignore[arg-type]
            source = self.SOURCES.get(item.name)
            rows.append(
                {
                    "parameter": f"{type(self).__name__}.{item.name}",
                    "value": getattr(self, item.name),
                    "kind": source.kind if source else "assumed",
                    "citation": source.citation if source else "",
                    "doi": source.doi if source else None,
                    "note": source.note if source else "",
                }
            )
        return rows


class Module:
    """Base class of every module: a name, parameters and a random stream."""

    category: ClassVar[str] = ""

    def __init__(self) -> None:
        self.rng = np.random.default_rng()

    def reset(self, rng: np.random.Generator | None = None) -> None:
        """Return to the initial state; keep ``rng`` for all randomness."""
        if rng is not None:
            self.rng = rng

    @property
    def parameters(self) -> Parameters | None:
        """Return this module's parameters, if it has any."""
        return getattr(self, "params", None)


_M = TypeVar("_M", bound=type)
REGISTRY: dict[str, dict[str, Callable[..., Module]]] = {c: {} for c in CATEGORIES}


def register(category: str, name: str) -> Callable[[_M], _M]:
    """Class decorator: make an implementation available by category and name."""
    if category not in REGISTRY:
        raise ValueError(
            f"unknown category {category!r}; known: {', '.join(CATEGORIES)}"
        )

    def decorator(cls: _M) -> _M:
        if name in REGISTRY[category]:
            raise ValueError(f"{category} module {name!r} is already registered")
        REGISTRY[category][name] = cls
        return cls

    return decorator


def create(category: str, name: str, **params: Any) -> Module:
    """Instantiate a registered module; keyword arguments go to its parameters."""
    try:
        factory = REGISTRY[category][name]
    except KeyError:
        known = ", ".join(sorted(REGISTRY.get(category, {})))
        raise ValueError(
            f"no {category} module {name!r}; registered: {known}"
        ) from None
    return factory(**params)


def available(category: str | None = None) -> dict[str, list[str]]:
    """Return registered module names, per category."""
    _load_builtins()
    categories = [category] if category else list(CATEGORIES)
    return {c: sorted(REGISTRY[c]) for c in categories}


def _load_builtins() -> None:
    # Importing the baseline modules registers them.
    from . import (  # noqa: F401
        affect,
        attention,
        estimation,
        fatigue,
        hearing,
        memory,
        motor,
        proprioception,
        response,
        risk,
        touch,
        trust,
        vestibular,
    )
    from .vision import eyes, foveation, objects  # noqa: F401


@dataclass(frozen=True)
class ModuleChoice:
    """One category's implementation and its parameter overrides."""

    name: str
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HumanConfig:
    """Which implementation fills each category.

    The defaults are the cited baseline. Replace one entry to evaluate a new
    module with everything else held constant.
    """

    eyes: ModuleChoice = ModuleChoice("main_sequence")
    vision: ModuleChoice = ModuleChoice("foveated")
    objects: ModuleChoice = ModuleChoice("eccentricity_noise")
    attention: ModuleChoice = ModuleChoice("uncertainty_sampler")
    memory: ModuleChoice = ModuleChoice("act_r")
    proprioception: ModuleChoice = ModuleChoice("spindle")
    motor: ModuleChoice = ModuleChoice("ffitts_min_jerk")
    estimator: ModuleChoice = ModuleChoice("lane_kalman")
    hearing: ModuleChoice = ModuleChoice("psychoacoustic")
    touch: ModuleChoice = ModuleChoice("vibrotactile")
    vestibular: ModuleChoice = ModuleChoice("subjective_vertical")
    response: ModuleChoice = ModuleChoice("twin")
    fatigue: ModuleChoice = ModuleChoice("three_compartment")
    affect: ModuleChoice = ModuleChoice("appraisal")
    risk: ModuleChoice = ModuleChoice("risk_field")
    trust: ModuleChoice = ModuleChoice("decision_field")

    def replace(self, category: str, name: str, **params: Any) -> HumanConfig:
        """Return a copy with one category swapped."""
        if category not in CATEGORIES:
            raise ValueError(f"unknown category {category!r}")
        return dataclasses.replace(self, **{category: ModuleChoice(name, params)})

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable description."""
        return {
            c: {"name": getattr(self, c).name, "params": dict(getattr(self, c).params)}
            for c in CATEGORIES
        }


@dataclass
class Human:
    """One module per category, sharing a seeded random stream per module."""

    config: HumanConfig
    modules: dict[str, Module]

    def __getattr__(self, name: str) -> Module:
        modules = self.__dict__.get("modules", {})
        if name in modules:
            return modules[name]
        raise AttributeError(name)

    def reset(self, seed: int | None = None) -> None:
        """Reset every module with independent child random streams."""
        streams = np.random.SeedSequence(seed).spawn(len(self.modules))
        for stream, module in zip(streams, self.modules.values()):
            module.reset(np.random.default_rng(stream))

    def provenance(self) -> list[dict[str, Any]]:
        """Return every parameter with its source."""
        rows: list[dict[str, Any]] = []
        for category, module in self.modules.items():
            params = module.parameters
            if params is not None:
                for row in params.provenance():
                    rows.append({"category": category, **row})
        return rows


def build_human(config: HumanConfig | None = None, seed: int | None = None) -> Human:
    """Create every module named by ``config`` and reset them with ``seed``."""
    _load_builtins()
    config = config or HumanConfig()
    modules = {
        c: create(c, getattr(config, c).name, **dict(getattr(config, c).params))
        for c in CATEGORIES
    }
    human = Human(config, modules)
    human.reset(seed)
    return human
