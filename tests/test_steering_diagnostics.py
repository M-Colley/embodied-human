"""Steering interventions preserve replay controls and complete physical resets."""

import numpy as np
import pytest

from embodied_human.steering_diagnostics import (
    intervention_command,
    run_steering_comparison,
    signed_following_errors,
)


def test_intervention_changes_only_steering():
    controls = pytest.importorskip("myosuite_carla.controls")
    reference = controls.VehicleCommand(throttle=0.7, brake=0.1, reverse=True, gear=-1)
    measured = controls.VehicleCommand(throttle=0.2, steer=-0.3)
    muscle = intervention_command(measured, reference, None)
    ideal = intervention_command(measured, reference, 0.4)
    assert muscle.steer == -0.3 and ideal.steer == 0.4
    assert {k: v for k, v in muscle.as_dict().items() if k != "steer"} == {
        k: v for k, v in reference.as_dict().items() if k != "steer"
    }
    assert ideal.throttle == reference.throttle and ideal.reverse == reference.reverse


def test_native_replay_preserves_initial_state_commands_and_muscle_trajectory():
    pytest.importorskip("myosuite.core.model_builder")
    models = pytest.importorskip("myosuite_carla.driver_models")
    vehicles = pytest.importorskip("myosuite_carla.vehicle_model")
    from myosuite_carla.ngsim import FollowingTrial

    time = np.arange(11) / 10
    trial = FollowingTrial(
        driver="synthetic",
        time=time,
        position=time * 5,
        speed=np.full_like(time, 5),
        acceleration=np.zeros_like(time),
        gap=np.full_like(time, 20),
        leader_speed=np.full_like(time, 5),
    )
    result = run_steering_comparison(
        trial,
        vehicles.carla_1_0_nissan_patrol(),
        models.ReflexDriverConfig(perception_delay=0.04),
        seed=42,
        desired_speed=10,
    )
    assert result["comparison_steps"] == 50
    for arm, parity in result["parity"].items():
        assert all(parity.values()), (arm, parity)
    assert all(r["surrogate_closed"] for r in result["episodes"].values())


def test_signed_errors_average_seeds_within_sources_before_sources():
    def row(group, value):
        return {
            "source_group": group,
            **{
                f"predicted_{name}_{unit}": [value, value]
                for name, unit in (
                    ("speed", "mps"),
                    ("gap", "m"),
                    ("acceleration", "mps2"),
                )
            },
            **{
                f"observed_{name}_{unit}": [0, 0]
                for name, unit in (
                    ("speed", "mps"),
                    ("gap", "m"),
                    ("acceleration", "mps2"),
                )
            },
        }

    result = signed_following_errors(
        {"arm": [row("a", 0), row("a", 2), row("b", 5)]}, ("a", "b")
    )
    assert result["arm"]["mean"]["gap"]["mean_signed_error"] == 3
