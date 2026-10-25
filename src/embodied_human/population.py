"""Simulated participants: individual differences and group profiles.

A study needs a sample of people, not one model. A :class:`Population`
draws participants by varying module parameters between people
(:class:`Variation`) and can shift a whole group (:class:`Profile`, e.g.
older adults). Every spread and shift carries a :class:`Source`; spreads
not measured in humans are labelled as assumed.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import numpy as np

from .core import CATEGORIES, HumanConfig, ModuleChoice, Source, _load_builtins, create


@dataclass(frozen=True)
class Variation:
    """How one parameter varies between people.

    Args:
        category: Module category, e.g. ``attention``.
        parameter: Parameter field name.
        spread: Coefficient of variation (``lognormal``) or SD (``normal``).
        distribution: ``lognormal`` keeps positive parameters positive.
        source: Where the spread comes from.
    """

    category: str
    parameter: str
    spread: float
    distribution: Literal["lognormal", "normal"] = "lognormal"
    source: Source = Source("model choice", None, "assumed")

    def __post_init__(self) -> None:
        if self.category not in CATEGORIES:
            raise ValueError(f"unknown category {self.category!r}")
        if not math.isfinite(self.spread) or self.spread < 0:
            raise ValueError("spread must be finite and non-negative")

    def sample(self, value: float, rng: np.random.Generator) -> float:
        """Draw one person's value around the group value."""
        if self.distribution == "normal":
            return float(value + self.spread * rng.standard_normal())
        sigma = math.sqrt(math.log(1 + self.spread**2))
        # Median-preserving: the group value stays the typical person's.
        return float(value * math.exp(sigma * rng.standard_normal()))


@dataclass(frozen=True)
class Profile:
    """A group's parameter changes relative to the reference group.

    ``scale`` multiplies a parameter, e.g. ``{("proprioception",
    "joint_noise_deg"): 1.2}`` for 20% larger joint position errors.
    """

    name: str
    scale: Mapping[tuple[str, str], float] = field(default_factory=dict)
    sources: Mapping[tuple[str, str], Source] = field(default_factory=dict)


#: Younger adults are the reference group of the cited baselines.
YOUNG_ADULTS = Profile("young_adults")

#: Older adults (about 55-83 years) relative to younger adults (about 21-37).
OLDER_ADULTS = Profile(
    "older_adults",
    scale={
        ("proprioception", "joint_noise_deg"): 3.6 / 3.0,
        ("response", "central"): 200.0 / 131.0,
        ("response", "window"): 450.0 / 275.0,
    },
    sources={
        ("proprioception", "joint_noise_deg"): Source(
            "Talis & Levik (2016)",
            "https://doi.org/10.3389/fnins.2016.00520",
            "measured",
            "elbow matching error 3.6 deg (61-83 years) vs 3.0 deg (22-37 years)",
        ),
        ("response", "central"): Source(
            "Diederich, Colonius & Schomburg (2008), Table 3",
            "https://doi.org/10.1016/j.neuropsychologia.2008.03.026",
            "fitted",
            "TWIN central stage 200 ms (65-75 years) vs 131 ms (20-22 years), "
            "saccadic responses",
        ),
        ("response", "window"): Source(
            "Diederich, Colonius & Schomburg (2008), Table 3",
            "https://doi.org/10.1016/j.neuropsychologia.2008.03.026",
            "fitted",
            "integration window 450 vs 275 ms",
        ),
    },
)


def _value(choice: ModuleChoice, category: str, parameter: str) -> float:
    if parameter in choice.params:
        return float(choice.params[parameter])
    _load_builtins()
    module = create(category, choice.name, **dict(choice.params))
    params = module.parameters
    if params is None or not hasattr(params, parameter):
        raise ValueError(
            f"{category} module {choice.name!r} has no parameter {parameter!r}"
        )
    return float(getattr(params, parameter))


@dataclass(frozen=True)
class Population:
    """Participants drawn around a base configuration.

    Args:
        base: The group's typical person.
        variations: Between-person variation per parameter.
        profile: Group shift applied before sampling.
    """

    base: HumanConfig = field(default_factory=HumanConfig)
    variations: tuple[Variation, ...] = ()
    profile: Profile = YOUNG_ADULTS

    def typical(self) -> HumanConfig:
        """The group's typical person (profile applied, no variation)."""
        config = self.base
        for (category, parameter), factor in self.profile.scale.items():
            choice = getattr(config, category)
            value = _value(choice, category, parameter) * factor
            config = replace(config, **{category: _with(choice, parameter, value)})
        return config

    def sample(self, rng: np.random.Generator) -> HumanConfig:
        """Draw one participant."""
        config = self.typical()
        for variation in self.variations:
            choice = getattr(config, variation.category)
            value = variation.sample(
                _value(choice, variation.category, variation.parameter), rng
            )
            config = replace(
                config,
                **{variation.category: _with(choice, variation.parameter, value)},
            )
        return config

    def participants(self, count: int, seed: int = 0) -> list[HumanConfig]:
        """Draw ``count`` participants reproducibly."""
        rng = np.random.default_rng(seed)
        return [self.sample(rng) for _ in range(count)]

    def describe(self) -> dict[str, Any]:
        """Sources of every variation and profile change, for reports."""
        return {
            "profile": self.profile.name,
            "profile_changes": [
                {
                    "parameter": f"{c}.{p}",
                    "scale": s,
                    **_source(self.profile.sources.get((c, p))),
                }
                for (c, p), s in self.profile.scale.items()
            ],
            "variations": [
                {
                    "parameter": f"{v.category}.{v.parameter}",
                    "spread": v.spread,
                    "distribution": v.distribution,
                    **_source(v.source),
                }
                for v in self.variations
            ],
        }


def applicable(
    config: HumanConfig, variations: tuple[Variation, ...]
) -> tuple[Variation, ...]:
    """Keep the variations whose parameter exists in ``config``'s modules,
    so a population can be drawn around any module combination."""
    kept = []
    for variation in variations:
        try:
            _value(
                getattr(config, variation.category),
                variation.category,
                variation.parameter,
            )
        except ValueError:
            continue
        kept.append(variation)
    return tuple(kept)


def _with(choice: ModuleChoice, parameter: str, value: float) -> ModuleChoice:
    return ModuleChoice(choice.name, {**dict(choice.params), parameter: value})


def _source(source: Source | None) -> dict[str, Any]:
    if source is None:
        return {"kind": "assumed", "citation": ""}
    return {"kind": source.kind, "citation": source.citation, "doi": source.doi}


#: Between-person variation of the baseline modules. Only the joint
#: position sense spread is measured; the others are labelled assumptions.
BASELINE_VARIATIONS: tuple[Variation, ...] = (
    Variation(
        "proprioception",
        "joint_noise_deg",
        0.46 / 3.0,
        source=Source(
            "Talis & Levik (2016)",
            "https://doi.org/10.3389/fnins.2016.00520",
            "measured",
            "3.0 +- 0.46 deg elbow matching error in young adults",
        ),
    ),
    Variation("eyes", "latency_median", 0.15),
    Variation("attention", "return_sd", 0.2),
    Variation("motor", "slope", 0.15),
)
