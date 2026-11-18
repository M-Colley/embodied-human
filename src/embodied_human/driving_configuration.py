"""Explicit vehicle and controller settings for physical driving experiments."""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any

VEHICLE_PRESETS = ("legacy", "carla-1.0")


def validate_vehicle_settings(preset: str, perception_delay: float) -> None:
    """Validate experiment settings without importing physical dependencies."""
    if preset not in VEHICLE_PRESETS:
        raise ValueError(f"vehicle_preset must be one of {VEHICLE_PRESETS}")
    if (
        isinstance(perception_delay, bool)
        or not isinstance(perception_delay, (int, float))
        or not math.isfinite(perception_delay)
        or perception_delay < 0
    ):
        raise ValueError("reflex_perception_delay must be finite and nonnegative")


def resolve_vehicle_parameters(preset: str) -> Any:
    """Load one versioned bridge preset through optional physical imports."""
    validate_vehicle_settings(preset, 0.0)
    from myosuite_carla.vehicle_model import carla_nissan_patrol

    if preset == "legacy":
        return carla_nissan_patrol()
    from myosuite_carla.vehicle_model import carla_1_0_nissan_patrol

    return carla_1_0_nissan_patrol()


def resolve_reflex_configuration(
    vehicle: Any, *, perception_delay: float, acceleration_control: bool = False
) -> Any:
    """Bind an optional acceleration inverse to the selected vehicle."""
    from myosuite_carla.driver_models import (
        AccelerationControlConfig,
        ReflexDriverConfig,
    )

    return ReflexDriverConfig(
        perception_delay=perception_delay,
        acceleration_control=(
            AccelerationControlConfig(vehicle=vehicle) if acceleration_control else None
        ),
    )


def reflex_timing(configuration: Any, control_dt: float) -> dict[str, Any]:
    """Describe the oldest available sample used by ReflexDriver each step."""
    if not math.isfinite(control_dt) or control_dt <= 0:
        raise ValueError("control_dt must be finite and positive")
    samples = max(1, round(configuration.perception_delay / control_dt))
    return {
        "control_dt_s": control_dt,
        "configured_perception_delay_s": configuration.perception_delay,
        "history_samples": samples,
        "realized_history_lag_s": (samples - 1) * control_dt,
        "warmup": "oldest available sample until the history is full",
    }


def physical_configuration(preset: str, vehicle: Any, reflex: Any) -> dict[str, Any]:
    """Export complete numerical settings and discrete controller timing."""
    from myosuite_carla.fullbody import FullBodyDriverConfig

    body = FullBodyDriverConfig.fast()
    return {
        "vehicle_preset": preset,
        "vehicle_parameters": vehicle.as_dict(),
        "reflex_config": asdict(reflex),
        "driver_config": asdict(body),
        "reflex_timing": reflex_timing(reflex, body.control_dt),
    }
