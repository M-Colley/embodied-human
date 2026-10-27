"""Declared coverage and execution profiles for packaged benchmarks."""

from __future__ import annotations

import json
from importlib.resources import files
from typing import Any

from ..core import CATEGORIES, available
from .references import REFERENCES


def protocol_catalog() -> dict[str, dict[str, Any]]:
    """Load metadata and require complete declarations for packaged protocols."""
    from . import integrated, load, motivation, senses, studies  # noqa: F401
    from .benchmarks import BENCHMARKS

    available()  # Load packaged module registrations before checking coverage.
    catalog = json.loads(
        files("embodied_human.bench").joinpath("protocols.json").read_text()
    )
    packaged = {
        name
        for name, function in BENCHMARKS.items()
        if function.__module__.startswith("embodied_human.")
    }
    if set(catalog) != packaged:
        raise ValueError(
            "benchmark metadata differs from packaged registry: "
            f"missing {sorted(packaged - set(catalog))}; "
            f"unregistered {sorted(set(catalog) - packaged)}"
        )
    for name, declaration in catalog.items():
        if set(declaration) != {"categories", "metrics", "ci_profile"}:
            raise ValueError(
                f"{name} needs categories, metrics and ci_profile metadata"
            )
        if declaration["ci_profile"] not in ("core", "full"):
            raise ValueError(f"{name} needs ci_profile core or full")
        for field, allowed in (("categories", CATEGORIES), ("metrics", REFERENCES)):
            values = declaration[field]
            if (
                not isinstance(values, list)
                or not values
                or not all(isinstance(value, str) for value in values)
                or len(set(values)) != len(values)
                or set(values) - set(allowed)
            ):
                raise ValueError(f"{name} has invalid {field} metadata")
    return catalog


def benchmark_names(profile: str) -> list[str]:
    """Return the declared core subset or every packaged benchmark for full."""
    if profile not in ("core", "full"):
        raise ValueError("benchmark profile must be core or full")
    return [
        name
        for name, declaration in protocol_catalog().items()
        if profile == "full" or declaration["ci_profile"] == "core"
    ]
