"""Apply contextual empirical profiles without mutating baseline defaults.

A caller supplies the intended task context explicitly. Profile provenance is
kept alongside the resolved configuration. Parameter classes retain their
original published baseline source labels in ``Human.provenance()``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from numbers import Integral, Real
from typing import Any

from .core import CATEGORIES, Human, HumanConfig, ModuleChoice, build_human

_IDM_FIELDS = {
    "max_acceleration",
    "comfortable_deceleration",
    "time_headway",
    "min_gap",
    "exponent",
}


def _copy(value: Any) -> Any:
    """Copy JSON compatible settings and reject nonfinite numeric values."""
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("configuration keys must be strings")
        return {key: _copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(_copy(item) for item in value)
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real) and math.isfinite(float(value)):
        return float(value)
    raise ValueError("profile settings must contain finite JSON compatible values")


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    result: dict[str, Any] = _copy(value)
    return result


def _version(value: Mapping[str, Any]) -> None:
    version = value.get("schema_version", 1)
    if isinstance(version, bool) or not isinstance(version, int) or version != 1:
        raise ValueError("unsupported driving profile schema version")


def _parameters(value: Any, name: str) -> dict[str, float]:
    source = _mapping(value, name)
    if not source or any(
        not key.strip() or isinstance(item, bool) or not isinstance(item, Real)
        for key, item in source.items()
    ):
        raise ValueError(f"{name} must contain named finite numeric parameters")
    return {key: float(item) for key, item in source.items()}


def _validate_idm(values: Mapping[str, Any]) -> None:
    if set(values) - _IDM_FIELDS:
        raise ValueError("IDM settings contain an unknown parameter")
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError("IDM settings must contain numeric values")
        numeric = float(value)
        if name in {"time_headway", "min_gap"}:
            valid = numeric >= 0
        else:
            valid = numeric > 0
        if not valid:
            raise ValueError(f"IDM parameter {name!r} is outside its physical domain")


@dataclass(frozen=True)
class ResolvedDrivingProfile:
    """An explicitly contextual configuration and its separate fitted sources."""

    human_config: HumanConfig
    driver_config: dict[str, Any]
    reflex_config: dict[str, Any]
    module: str
    context: str
    parameters: dict[str, float]
    fitted_parameters: tuple[str, ...]
    provenance: dict[str, Any]
    role: str

    def build_human(self, seed: int | None = None) -> Human:
        """Construct a human using the resolved module choices and overrides."""
        return build_human(self.human_config, seed=seed)

    def make_driver_parameters(self) -> Any:
        """Construct the attentive driver's geometry and task parameters."""
        from .agents.driver import DriverAgentParameters

        return DriverAgentParameters(**_copy(self.driver_config))

    def make_reflex_config(self) -> Any:
        """Construct a physical bridge reflex configuration through lazy imports."""
        values = _copy(self.reflex_config)
        backend = values.pop("backend", "ReflexDriver")
        if backend not in {"ReflexDriver", "myosuite_carla.driver_models.ReflexDriver"}:
            raise ValueError(
                "this profile does not specify a physical reflex controller"
            )
        from myosuite_carla.driver_models import (
            IdmConfig,
            ReflexDriverConfig,
            TwoPointSteeringConfig,
        )

        if "idm" in values:
            _validate_idm(values["idm"])
            values["idm"] = IdmConfig(**values["idm"])
        if "steering" in values:
            values["steering"] = TwoPointSteeringConfig(**values["steering"])
        return ReflexDriverConfig(**values)

    def fitted_provenance(self) -> list[dict[str, Any]]:
        """Return profile sources separately from module baseline citations."""
        return [
            {
                "parameter": f"{self.module}.{name}",
                "value": self.parameters[name],
                "kind": "fitted" if self.role == "fitted_candidate" else "baseline",
                "context": self.context,
                "selection_role": self.role,
                "provenance": _copy(self.provenance),
            }
            for name in self.fitted_parameters
        ]

    def as_dict(self) -> dict[str, Any]:
        """Export the resolved profile with a versioned, JSON compatible schema."""
        return {
            "schema_version": 1,
            "module": self.module,
            "context": self.context,
            "role": self.role,
            "parameters": dict(self.parameters),
            "parameter_overrides": {
                name: self.parameters[name] for name in self.fitted_parameters
            },
            "configuration": {
                "human": self.human_config.as_dict(),
                "driver": _copy(self.driver_config),
                "reflex": _copy(self.reflex_config),
            },
            "configuration_role": "resolved_with_explicit_parameter_overrides",
            "provenance": _copy(self.provenance),
            "test_used_for_parameter_selection": False,
        }


def resolve_profile(
    report_or_profile: Mapping[str, Any], *, context: str
) -> ResolvedDrivingProfile:
    """Resolve a supported profile for an exactly matching intended context.

    Complete grouped calibration reports and their direct ``profile`` objects
    are accepted. Direct objects follow schema version 1 when their version
    field is omitted. An explicitly unsupported version is rejected.
    """
    source = _mapping(report_or_profile, "report or profile")
    _version(source)
    if "profile" in source:
        if source.get("status") != "evaluated":
            raise ValueError(
                "a complete evaluated report is required to apply its profile"
            )
        profile = _mapping(source["profile"], "profile")
        _version(profile)
    else:
        profile = source
    if (
        not isinstance(context, str)
        or not context.strip()
        or profile.get("context") != context
    ):
        raise ValueError("the intended context must exactly match the fitted context")
    if profile.get("test_used_for_parameter_selection") is not False:
        raise ValueError("profile must explicitly preserve independent test selection")
    role = profile.get("role")
    if not isinstance(role, str) or role not in {
        "fitted_candidate",
        "retained_baseline",
    }:
        raise ValueError("unsupported profile selection role")
    module = profile.get("module")
    if not isinstance(module, str) or module not in {
        "attention.uncertainty_sampler",
        "attention.seev",
        "reflex.idm",
    }:
        raise ValueError("unsupported fitted driving module")
    parameters = _parameters(profile.get("parameters"), "parameters")
    overrides = _parameters(
        profile.get("parameter_overrides", parameters), "parameter_overrides"
    )
    if any(
        name not in parameters or parameters[name] != value
        for name, value in overrides.items()
    ):
        raise ValueError("parameter overrides must agree with the selected parameters")
    configuration = _mapping(profile.get("configuration"), "configuration")
    if set(configuration) != {"human", "driver", "reflex"}:
        raise ValueError(
            "configuration must preserve human, driver and reflex settings"
        )
    human = _mapping(configuration["human"], "human configuration")
    if set(human) - set(CATEGORIES):
        raise ValueError("human configuration contains an unknown module category")
    choices = {}
    for category, raw in human.items():
        choice = _mapping(raw, f"{category} module choice")
        if (
            set(choice) != {"name", "params"}
            or not isinstance(choice["name"], str)
            or not choice["name"].strip()
        ):
            raise ValueError("module choices need a name and parameter object")
        choices[category] = ModuleChoice(
            choice["name"], _mapping(choice["params"], f"{category} parameters")
        )
    resolved_human = HumanConfig(**choices)
    driver = _mapping(configuration["driver"], "driver configuration")
    reflex = _mapping(configuration["reflex"], "reflex configuration")
    from .agents.driver import DriverAgentParameters

    if set(driver) - {field.name for field in fields(DriverAgentParameters)}:
        raise ValueError("driver configuration contains an unknown setting")
    if module.startswith("attention."):
        from .attention import SamplerParameters, SEEVParameters

        implementation = module.split(".", 1)[1]
        old = resolved_human.attention
        if old.name != implementation and old.params:
            raise ValueError("baseline attention parameters belong to another module")
        values = dict(old.params) if old.name == implementation else {}
        if implementation == "uncertainty_sampler":
            values.update(parameters)
            allowed = {field.name for field in fields(SamplerParameters)}
            if set(values) - allowed:
                raise ValueError("unknown uncertainty sampler parameter")
            for name, value in values.items():
                if isinstance(value, bool) or not isinstance(value, Real):
                    raise ValueError("uncertainty sampler parameters must be numeric")
                if value < 0 or (
                    name in {"return_sd", "maximum_glance"} and value == 0
                ):
                    raise ValueError(
                        "uncertainty sampler parameter is outside its domain"
                    )
        else:
            allowed = {field.name for field in fields(SEEVParameters)}
            weights = {
                name: value
                for name, value in parameters.items()
                if not name.startswith("area_value.")
            }
            if set(weights) - (allowed - {"multiplicative"}):
                raise ValueError("unknown fitted SEEV weight")
            values.update(weights)
            if set(values) - (allowed | {"areas"}):
                raise ValueError("unknown SEEV setting")
            areas = _mapping(values.get("areas"), "SEEV area features")
            if not areas or any(
                not area.strip()
                or not isinstance(features, tuple)
                or len(features) != 4
                or any(isinstance(v, bool) or not isinstance(v, Real) for v in features)
                for area, features in areas.items()
            ):
                raise ValueError("SEEV areas need four finite feature values per area")
            for name, value in parameters.items():
                if name.startswith("area_value."):
                    area = name.removeprefix("area_value.")
                    if area not in areas:
                        raise ValueError(
                            f"fitted SEEV area {area!r} lacks fixed features"
                        )
                    areas[area] = (*areas[area][:3], value)
            values["areas"] = areas
        resolved_human = resolved_human.replace("attention", implementation, **values)
    else:
        if "idm" not in reflex:
            raise ValueError("IDM profile requires its baseline IDM configuration")
        values = _mapping(reflex["idm"], "baseline IDM configuration")
        values.update(parameters)
        _validate_idm(values)
        reflex["idm"] = values
    return ResolvedDrivingProfile(
        resolved_human,
        driver,
        reflex,
        module,
        context,
        parameters,
        tuple(overrides),
        _mapping(profile.get("provenance"), "profile provenance"),
        role,
    )
