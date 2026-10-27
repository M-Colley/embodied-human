"""Optional SALib designs and analysis with explicit simulation protocol limits.

Validation needs only core package dependencies. Sampling and analysis import
SALib lazily and use local Generator instances, including for bootstrap samples.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import json
import math
from collections.abc import Mapping
from importlib import metadata, resources
from numbers import Real
from typing import Any

import numpy as np

from .core import CATEGORIES, HumanConfig, create

_FIELDS = {
    "method",
    "parameters",
    "benchmark",
    "metric",
    "samples",
    "seed",
    "num_levels",
}
_FACTOR_FIELDS = {"category", "parameter", "bounds"}
_INDICES = {
    "morris": ("mu", "mu_star", "sigma", "mu_star_conf"),
    "sobol": ("S1", "ST", "S1_conf", "ST_conf"),
}


def _integer(value: Any, label: str, low: int, high: int | None = None) -> int:
    if type(value) is not int or value < low or (high is not None and value > high):
        raise ValueError(f"{label} must be an integer in the allowed range")
    return value


def _structure(data: Any, *, canonical: bool = False) -> dict[str, Any]:
    if not isinstance(data, Mapping):
        raise ValueError("sensitivity must be an object")
    allowed = _FIELDS | ({"evaluations"} if canonical else set())
    if set(data) - allowed:
        raise ValueError("unknown sensitivity fields")
    method = data.get("method")
    if not isinstance(method, str) or method not in _INDICES:
        raise ValueError("sensitivity method must be 'morris' or 'sobol'")
    samples = _integer(data.get("samples"), "samples", 4, 256)
    if method == "sobol" and samples & (samples - 1):
        raise ValueError("Sobol samples must be a power of two")
    seed = _integer(data.get("seed", 0), "seed", 0)
    levels = _integer(data.get("num_levels", 4), "num_levels", 4, 20)
    if method == "morris" and levels % 2:
        raise ValueError("Morris num_levels must be even, from 4 to 20")
    factors = data.get("parameters")
    if not isinstance(factors, list) or not 1 <= len(factors) <= 8:
        raise ValueError("sensitivity requires from 1 to 8 parameters")
    cleaned, seen = [], set()
    for factor in factors:
        if not isinstance(factor, Mapping) or set(factor) != _FACTOR_FIELDS:
            raise ValueError(
                "each sensitivity parameter needs category, parameter and bounds"
            )
        category, parameter = factor["category"], factor["parameter"]
        if not isinstance(category, str) or category not in CATEGORIES:
            raise ValueError("unknown sensitivity parameter category")
        if not isinstance(parameter, str) or not parameter:
            raise ValueError("sensitivity parameter names must be nonempty strings")
        if (category, parameter) in seen:
            raise ValueError("sensitivity parameters must be unique")
        seen.add((category, parameter))
        bounds = factor["bounds"]
        if (
            not isinstance(bounds, (list, tuple))
            or len(bounds) != 2
            or any(isinstance(v, bool) or not isinstance(v, Real) for v in bounds)
        ):
            raise ValueError("parameter bounds must be two finite numbers")
        low, high = (float(v) for v in bounds)
        if not math.isfinite(low) or not math.isfinite(high) or low >= high:
            raise ValueError("parameter bounds must be finite and strictly ordered")
        if not math.isfinite(high - low):
            raise ValueError("parameter bound width must be finite")
        cleaned.append(
            {"category": category, "parameter": parameter, "bounds": [low, high]}
        )
    benchmark, metric = data.get("benchmark"), data.get("metric")
    if not isinstance(benchmark, str) or not isinstance(metric, str):
        raise ValueError("sensitivity benchmark and metric must be strings")
    return {
        "method": method,
        "parameters": cleaned,
        "benchmark": benchmark,
        "metric": metric,
        "samples": samples,
        "seed": seed,
        "num_levels": levels,
        "evaluations": samples * (len(cleaned) + (2 if method == "sobol" else 1)),
    }


def validate_sensitivity(
    data: Any, config: HumanConfig, benchmarks: list[str]
) -> dict[str, Any]:
    """Validate a bounded design without importing SALib or running benchmarks."""
    # Saved experiment requests contain this derived count. Recompute it on
    # every validation so canonical requests remain valid without trusting it.
    spec = _structure(data, canonical=True)
    if len(benchmarks) != 1 or benchmarks[0] != spec["benchmark"]:
        raise ValueError("sensitivity requires exactly its one selected benchmark")
    protocols = json.loads(
        resources.files("embodied_human.bench")
        .joinpath("protocols.json")
        .read_text(encoding="utf-8")
    )
    from .experiment_protocols import CATALOG

    protocols.update({entry["name"]: entry for entry in CATALOG})
    protocol = protocols.get(spec["benchmark"])
    if protocol is None or spec["metric"] not in protocol["metrics"]:
        raise ValueError("unknown sensitivity benchmark or metric")
    parameter_types: dict[str, dict[str, Any]] = {}
    for factor in spec["parameters"]:
        category = factor["category"]
        if category not in parameter_types:
            choice = getattr(config, category)
            module = create(category, choice.name, **dict(choice.params))
            parameters = module.parameters
            if parameters is None or not dataclasses.is_dataclass(parameters):
                raise ValueError(f"{category} has no dataclass parameters to vary")
            # Python 3.10 rejects an unrelated bare ClassVar through
            # get_type_hints. Resolve class annotations without that validation
            # and retain inherited dataclass fields from alternative modules.
            hints: dict[str, Any] = {}
            for base in reversed(type(parameters).__mro__):
                hints.update(inspect.get_annotations(base, eval_str=True))
            parameter_types[category] = {
                field.name: hints.get(field.name)
                for field in dataclasses.fields(parameters)
            }
        if parameter_types[category].get(factor["parameter"]) is not float:
            raise ValueError(
                "sensitivity accepts only declared continuous float parameters"
            )
    return spec


def _problem(spec: dict[str, Any]) -> dict[str, Any]:
    factors = spec["parameters"]
    return {
        "num_vars": len(factors),
        "names": [f"{f['category']}.{f['parameter']}" for f in factors],
        "bounds": [list(f["bounds"]) for f in factors],
    }


def _load(stage: str, method: str) -> Any:
    try:
        return importlib.import_module(f"SALib.{stage}.{method}")
    except ModuleNotFoundError as exc:
        if exc.name is None or not exc.name.startswith("SALib"):
            raise
        raise ImportError(
            "sensitivity sampling and analysis require SALib: "
            "python -m pip install 'SALib>=1.6,<2'"
        ) from exc


def sample_design(spec: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray]:
    """Return the actual SALib problem and ordered design to evaluate unchanged."""
    spec = _structure(spec, canonical=True)
    problem = _problem(spec)
    sampler = _load("sample", spec["method"])
    rng = np.random.default_rng(spec["seed"])
    if spec["method"] == "morris":
        design = sampler.sample(
            problem, spec["samples"], num_levels=spec["num_levels"], seed=rng
        )
    else:
        design = sampler.sample(
            problem, spec["samples"], calc_second_order=False, scramble=True, seed=rng
        )
    return problem, np.asarray(design, dtype=float)


def analyze_design(
    spec: dict[str, Any], problem: dict[str, Any], X: np.ndarray, Y: np.ndarray
) -> dict[str, Any]:
    """Analyze one complete ordered design, with explicit undefined estimates."""
    spec = _structure(spec, canonical=True)
    expected = _problem(spec)
    for key, value in expected.items():
        if problem.get(key) != value:
            raise ValueError(
                "analysis problem does not match sensitivity specification"
            )
    design, outputs = np.asarray(X, dtype=float), np.asarray(Y, dtype=float)
    shape = (spec["evaluations"], len(spec["parameters"]))
    if design.shape != shape or not np.isfinite(design).all():
        raise ValueError("sensitivity design has an invalid shape or nonfinite inputs")
    if outputs.shape != (shape[0],):
        raise ValueError("one ordered scalar output is required per design row")
    bounds = np.asarray(expected["bounds"])
    tolerance = 32 * np.finfo(float).eps * np.maximum(1, np.abs(bounds))
    if np.any(design < bounds[:, 0] - tolerance[:, 0]) or np.any(
        design > bounds[:, 1] + tolerance[:, 1]
    ):
        raise ValueError("sensitivity design contains values outside parameter bounds")
    result: dict[str, Any] = {
        "method": spec["method"],
        "benchmark": spec["benchmark"],
        "metric": spec["metric"],
        "status": "undefined",
        "reason": None,
        "parameter_names": expected["names"],
        "indices": {key: [None] * shape[1] for key in _INDICES[spec["method"]]},
        "samples": spec["samples"],
        "evaluations": shape[0],
        "seed": spec["seed"],
        "second_order": False,
        "confidence_level": 0.95,
        "bootstrap_resamples": 100,
    }
    if not np.isfinite(outputs).all():
        result["reason"] = "one or more benchmark outputs are nonfinite"
        return result
    if np.all(outputs == outputs[0]):
        result["reason"] = "benchmark output is constant across this design"
        return result
    analyzer = _load("analyze", spec["method"])
    rng = np.random.default_rng(spec["seed"])
    if spec["method"] == "morris":
        indices = analyzer.analyze(
            problem,
            design,
            outputs,
            num_levels=spec["num_levels"],
            seed=rng,
            num_resamples=100,
            conf_level=0.95,
            print_to_console=False,
        )
    else:
        # Indices are invariant to output scaling. Preconditioning avoids
        # overflow in SALib's internal standard deviation for large metrics.
        scaled = outputs / np.max(np.abs(outputs))
        indices = analyzer.analyze(
            problem,
            scaled,
            calc_second_order=False,
            seed=rng,
            num_resamples=100,
            conf_level=0.95,
            print_to_console=False,
        )
    result["salib_version"] = metadata.version("SALib")
    undefined = False
    for key in _INDICES[spec["method"]]:
        values = np.asarray(indices[key], dtype=float)
        result["indices"][key] = [
            float(v) if math.isfinite(v) else None for v in values
        ]
        undefined |= not np.isfinite(values).all()
    result["status"] = "undefined" if undefined else "ok"
    if undefined:
        result["reason"] = "some estimates or confidence intervals are undefined"
    return result
