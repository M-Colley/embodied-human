"""Local, registry-driven configuration builder and seeded task preview.

The exported JSON describes module choices, not executable Python. Loading it
only selects implementations already registered in this Python process.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import math
from collections.abc import Mapping
from typing import Any, get_args, get_origin

import numpy as np

from ..core import CATEGORIES, HumanConfig, available, build_human, create
from ..memory import Memory
from ..tasks.nback import NBackConfig, run_nback

FORMAT_VERSION = 1


def _annotations(cls: type) -> dict[str, Any]:
    # get_type_hints validates every annotation, including unrelated bare
    # ClassVar provenance tables, which Python 3.10 rejects. Resolve annotations
    # without that validation and merge bases for inherited parameter fields.
    hints: dict[str, Any] = {}
    for base in reversed(cls.__mro__):
        hints.update(inspect.get_annotations(base, eval_str=True))
    return hints


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def catalog() -> dict[str, Any]:
    """Describe installed module choices, defaults and their published sources."""
    choices: dict[str, dict[str, Any]] = {}
    for category, names in available().items():
        choices[category] = {}
        for name in names:
            try:
                module = create(category, name)
                params = module.parameters
                choices[category][name] = {
                    "description": (type(module).__doc__ or "").strip(),
                    "parameters": _jsonable(params) if params else {},
                    "sources": _jsonable(params.provenance()) if params else [],
                }
            except ImportError as exc:
                choices[category][name] = {"unavailable": str(exc)}
    return {
        "version": FORMAT_VERSION,
        "categories": list(CATEGORIES),
        "modules": choices,
        "baseline": HumanConfig().as_dict(),
    }


def _coerce(value: Any, default: Any, path: str, annotation: Any = None) -> Any:
    """Restore typed defaults (including joint-rate dataclasses) from JSON."""
    if dataclasses.is_dataclass(default) and not isinstance(default, type):
        if not isinstance(value, dict):
            raise ValueError(f"{path} must be an object")
        fields = {field.name for field in dataclasses.fields(default)}
        if set(value) != fields:
            raise ValueError(f"{path} needs fields {', '.join(sorted(fields))}")
        hints = _annotations(type(default))
        return type(default)(
            **{
                key: _coerce(
                    item, getattr(default, key), f"{path}.{key}", hints.get(key)
                )
                for key, item in value.items()
            }
        )
    if isinstance(default, bool):
        valid = isinstance(value, bool)
    elif isinstance(default, int):
        valid = isinstance(value, int) and not isinstance(value, bool)
    elif isinstance(default, float):
        valid = (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
        )
        if valid:
            return float(value)
    elif isinstance(default, str):
        valid = isinstance(value, str)
    elif isinstance(default, (tuple, list)):
        args = get_args(annotation)
        if get_origin(annotation) is tuple and len(args) == 2 and args[1] is ...:
            if not isinstance(value, list):
                raise ValueError(f"{path} needs an array")
            if not default and value:
                raise ValueError(f"{path} has no known element type")
            return tuple(
                _coerce(item, default[0], f"{path}[{i}]", args[0])
                for i, item in enumerate(value)
            )
        if not isinstance(value, list) or len(value) != len(default):
            raise ValueError(f"{path} needs an array of length {len(default)}")
        values = [
            _coerce(item, example, f"{path}[{i}]")
            for i, (item, example) in enumerate(zip(value, default))
        ]
        return tuple(values) if isinstance(default, tuple) else values
    elif isinstance(default, dict):
        if not isinstance(value, dict) or not all(
            isinstance(key, str) for key in value
        ):
            raise ValueError(f"{path} must be an object with string keys")
        if not default and value:
            raise ValueError(f"{path} has no known value type")
        example = next(iter(default.values()), None)
        return {
            key: _coerce(item, default.get(key, example), f"{path}.{key}")
            for key, item in value.items()
        }
    else:
        valid = value is None and default is None
    if not valid:
        raise ValueError(f"{path} needs a finite {type(default).__name__} value")
    return value


def config_from_dict(data: Any) -> HumanConfig:
    """Load a complete or partial ``HumanConfig.as_dict()`` representation.

    Unknown categories, choices and parameters are rejected. Structured values
    are restored to the types of the registered implementation's defaults.
    Constructor checks remain responsible for each model's scientific domain.
    """
    if not isinstance(data, dict) or set(data) - set(CATEGORIES):
        raise ValueError("config must be an object of known module categories")
    known = available()
    config = HumanConfig()
    for category, choice in data.items():
        if not isinstance(choice, dict) or set(choice) != {"name", "params"}:
            raise ValueError(f"{category} needs exactly name and params")
        name, overrides = choice["name"], choice["params"]
        if not isinstance(name, str) or name not in known[category]:
            raise ValueError(f"unknown {category} implementation {name!r}")
        if not isinstance(overrides, dict):
            raise ValueError(f"{category}.params must be an object")
        defaults = create(category, name).parameters
        fields = (
            {field.name for field in dataclasses.fields(defaults)}  # type: ignore[arg-type]
            if defaults is not None
            else set()
        )
        if set(overrides) - fields:
            raise ValueError(f"unknown parameters for {category}/{name}")
        hints = _annotations(type(defaults)) if defaults is not None else {}
        params = {
            key: _coerce(
                value, getattr(defaults, key), f"{category}.{key}", hints.get(key)
            )
            for key, value in overrides.items()
        }
        config = config.replace(category, name, **params)
    return config


def _project(data: Any) -> tuple[HumanConfig, int]:
    if not isinstance(data, dict) or set(data) != {"version", "seed", "config"}:
        raise ValueError("project needs exactly version, seed and config")
    if type(data["version"]) is not int or data["version"] != FORMAT_VERSION:
        raise ValueError(f"unsupported project version; expected {FORMAT_VERSION}")
    seed = data["seed"]
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be an integer between 0 and 4294967295")
    return config_from_dict(data["config"]), seed


def validate_project(data: Any) -> dict[str, Any]:
    """Construct selected modules and return a canonical project and sources."""
    config, seed = _project(data)
    human = build_human(config, seed)
    return {
        "project": {
            "version": FORMAT_VERSION,
            "seed": seed,
            "config": _jsonable(config.as_dict()),
        },
        "provenance": _jsonable(human.provenance()),
        "note": "Overrides are user choices; citations describe the baseline values.",
    }


def preview_project(data: Any) -> dict[str, Any]:
    """Run a small seeded delayed-digit-recall demonstration at levels 0–5."""
    config, seed = _project(data)
    human = build_human(config, seed)
    memory = human.modules["memory"]
    if not isinstance(memory, Memory):
        raise ValueError("selected memory does not implement the Memory interface")
    rows = []
    for level in range(6):
        human.reset(seed)
        result = run_nback(
            memory,
            NBackConfig(level=level, digits=24),
            np.random.default_rng(seed),
        )
        rows.append({"level": level, **result.summary()})
    return {
        "seed": seed,
        "rows": rows,
        "note": "Seeded delayed-recall demonstration, not validation against "
        "human data. Only the memory module affects these task results; "
        "all modules are constructed.",
    }


def python_source(data: Any) -> str:
    """Export a reproducible script without interpolating executable input."""
    result = validate_project(data)
    project = result["project"]
    config_json = json.dumps(project["config"], ensure_ascii=True, allow_nan=False)
    return (
        "import json\n\n"
        "from embodied_human import build_human\n"
        "from embodied_human.builder import config_from_dict, preview_project\n\n"
        f"config_data = json.loads({config_json!r})\n"
        f"seed = {project['seed']}\n"
        "human = build_human(config_from_dict(config_data), seed=seed)\n"
        "print(json.dumps(preview_project({\n"
        '    "version": 1, "seed": seed, "config": config_data,\n'
        "}), indent=2))\n"
    )
