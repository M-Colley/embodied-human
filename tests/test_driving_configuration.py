"""Versioned vehicles and controller timing remain explicit in physical studies."""

from dataclasses import asdict

import pytest

from embodied_human.driving_configuration import (
    physical_configuration,
    reflex_timing,
    resolve_reflex_configuration,
    resolve_vehicle_parameters,
    validate_vehicle_settings,
)


@pytest.mark.parametrize("delay", [0.0, 0.02, 0.04])
def test_timing_matches_executed_reflex_history(monkeypatch, delay):
    models = pytest.importorskip("myosuite_carla.driver_models")
    observed = []
    monkeypatch.setattr(
        models, "idm_acceleration", lambda speed, *args: observed.append(speed) or 0
    )
    configuration = models.ReflexDriverConfig(perception_delay=delay)
    controller = models.ReflexDriver(0.02, configuration)
    for index in range(4):
        controller.act(
            {
                "speed_mps": float(index),
                "target_speed_mps": 10.0,
                "near_angle_rad": 0.0,
                "far_angle_rad": 0.0,
            }
        )
    metadata = reflex_timing(configuration, 0.02)
    measured_lag = (3 - observed[-1]) * 0.02
    assert metadata["realized_history_lag_s"] == measured_lag
    assert metadata["configured_perception_delay_s"] == delay
    assert observed == ([0, 0, 1, 2] if delay == 0.04 else [0, 1, 2, 3])


def test_carla_1_configuration_binds_matching_inverse_and_preserves_legacy():
    models = pytest.importorskip("myosuite_carla.driver_models")
    vehicles = pytest.importorskip("myosuite_carla.vehicle_model")
    params = resolve_vehicle_parameters("carla-1.0")
    assert params == vehicles.carla_1_0_nissan_patrol()
    reflex = resolve_reflex_configuration(
        params, perception_delay=0.04, acceleration_control=True
    )
    assert reflex.acceleration_control.vehicle is params
    complete = physical_configuration("carla-1.0", params, reflex)
    assert complete["vehicle_parameters"] == params.as_dict()
    assert complete["reflex_timing"]["realized_history_lag_s"] == 0.02
    assert complete["driver_config"]["control_dt"] == 0.02
    assert models.ReflexDriverConfig.from_dict(complete["reflex_config"]) == reflex
    legacy = resolve_vehicle_parameters("legacy")
    baseline = resolve_reflex_configuration(legacy, perception_delay=0.02)
    assert legacy == vehicles.carla_nissan_patrol()
    assert asdict(baseline) == asdict(models.ReflexDriverConfig(perception_delay=0.02))


@pytest.mark.parametrize(
    "preset,delay", [("unknown", 0.02), ("carla-1.0", -0.1), ("legacy", True)]
)
def test_invalid_vehicle_settings_fail_without_physical_imports(preset, delay):
    with pytest.raises(ValueError):
        validate_vehicle_settings(preset, delay)
