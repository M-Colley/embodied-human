"""Evaluation-only steering interventions with replayed longitudinal commands."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from typing import Any

import numpy as np


def intervention_command(measured: Any, reference: Any, target: float | None) -> Any:
    """Replay every reference field except the selected steering intervention."""
    return replace(reference, steer=measured.steer if target is None else target)


def _physics_state(driver: Any) -> np.ndarray:
    import mujoco

    specification = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(driver.model, specification))
    mujoco.mj_getState(driver.model, driver.data, state, specification)
    return state


def _vehicle(
    parameters: Any, dt: float, commands: Sequence[Any] | None, target: Any
) -> Any:
    from myosuite_carla.session import SessionConfig
    from myosuite_carla.vehicle_model import SurrogateVehicle

    class ReplayVehicle(SurrogateVehicle):
        def __init__(self) -> None:
            self.requested_commands: list[Any] = []
            self.applied_commands: list[Any] = []
            super().__init__(parameters, config=SessionConfig(fixed_delta_seconds=dt))

        def step(self, command: Any) -> Any:
            index = len(self.applied_commands)
            applied = (
                command
                if commands is None
                else intervention_command(command, commands[index], target())
            )
            self.requested_commands.append(command)
            self.applied_commands.append(applied)
            return super().step(applied)

    return ReplayVehicle()


def _episode(
    driver: Any,
    trial: Any,
    parameters: Any,
    reflex_config: Any,
    seed: int,
    desired_speed: float,
    *,
    commands: Sequence[Any] | None = None,
    ideal: bool = False,
) -> dict[str, Any]:
    from myosuite_carla.driver_models import ReflexDriver
    from myosuite_carla.driving_env import FullBodyDrivingEnv
    from myosuite_carla.training import scenario_from_trial

    import embodied_human as eh

    from .agents.driver import AttentiveDriver

    reflex = ReflexDriver(driver.config.control_dt, reflex_config)
    vehicle = _vehicle(
        parameters,
        driver.config.control_dt,
        commands,
        lambda: reflex.demand[1] if ideal else None,
    )
    env = FullBodyDrivingEnv(
        scenario_from_trial(trial, desired_speed), driver=driver, vehicle=vehicle
    )
    records = []
    states = []
    terminated = truncated = False
    try:
        _, initial = env.reset(seed=seed)
        human = eh.build_human(seed=seed)
        agent = AttentiveDriver(human, env, reflex=reflex)
        agent.reset(seed=seed)
        initial_state = {
            "physics": _physics_state(driver).tolist(),
            "vehicle": asdict(vehicle.snapshot()),
            "human_random_states": {
                name: getattr(human, name).rng.bit_generator.state
                for name in eh.CATEGORIES
                if hasattr(getattr(human, name), "rng")
            },
        }
        states.append(initial["scenario"])
        count = (
            math.ceil(trial.duration / env.dt) if commands is None else len(commands)
        )
        for index in range(count):
            before = dict(env.state)
            request_time = env.monitor.state["time"]
            intent = agent.act()
            belief = np.asarray(getattr(human.estimator, "mean"), dtype=float).copy()
            demand = tuple(reflex.demand)
            _, _, terminated, truncated, info = env.step(intent)
            state = _physics_state(driver)
            if not np.isfinite(state).all():
                raise RuntimeError("The physical body contains nonfinite state")
            states.append(info["scenario"])
            records.append(
                {
                    "step": index,
                    "request_time_s": request_time,
                    "application_time_s": info["scenario"]["time"],
                    "vehicle_elapsed_s": vehicle.snapshot().elapsed_seconds,
                    "body_elapsed_s": float(driver.data.time),
                    "requested_acceleration_mps2": demand[0],
                    "realized_acceleration_mps2": env.state["a_forward_mps2"],
                    "requested_steer_normalized": demand[1],
                    "measured_wheel_normalized": info["inputs"]["steer"],
                    "wheel_tracking_error": demand[1] - info["inputs"]["steer"],
                    "true_lane_offset_before_m": before["lane_offset_m"],
                    "believed_lane_offset_m": float(belief[0]),
                    "true_heading_before_rad": before["heading_error_rad"],
                    "believed_heading_rad": float(belief[1]),
                    "true_lane_offset_after_m": env.state["lane_offset_m"],
                    "true_heading_after_rad": env.state["heading_error_rad"],
                    "intent": asdict(intent),
                    "measured_inputs": info["inputs"],
                    "requested_command": vehicle.requested_commands[-1].as_dict(),
                    "applied_command": vehicle.applied_commands[-1].as_dict(),
                    "vehicle": asdict(vehicle.snapshot()),
                    "physics_state_sha256": hashlib.sha256(state.tobytes()).hexdigest(),
                    "collision": info["collision"],
                    "off_road": info["off_road"],
                }
            )
            if terminated or truncated:
                break
    finally:
        env.close()
        vehicle.close()
    return {
        "initial_state": initial_state,
        "trace": records,
        "states": states,
        "commands": vehicle.applied_commands,
        "status": "terminated"
        if terminated
        else "completed"
        if truncated
        else "replay_horizon",
        "failure_reason": (
            "collision"
            if records[-1]["collision"]
            else "road_departure"
            if records[-1]["off_road"]
            else None
        ),
        "surrogate_closed": not vehicle.active,
    }


def trajectory_comparison(
    states: Sequence[Mapping[str, Any]], trial: Any
) -> dict[str, Any]:
    """Use the source processing grid for reproduction of the archived episode."""
    from myosuite_carla.ngsim import RATE, resample_following

    times = np.array([row["time"] for row in states])
    horizon = math.floor(min(times[-1], trial.duration) * RATE + 1e-8) / RATE
    keep = times <= horizon + 1e-8
    prediction = resample_following(
        times[keep],
        np.array([row["ego_front_s_m"] for row in states])[keep],
        np.array([row["lead_front_s_m"] for row in states])[keep],
        states[0]["lead_length_m"],
    )
    result: dict[str, Any] = {"time_s": prediction.time.tolist()}
    for name, suffix in (("speed", "mps"), ("gap", "m"), ("acceleration", "mps2")):
        values = getattr(prediction, name)
        observed = np.interp(prediction.time, trial.time, getattr(trial, name))
        result[f"predicted_{name}_{suffix}"] = values.tolist()
        result[f"{name}_rmse_{suffix}"] = float(
            np.sqrt(np.mean((values - observed) ** 2))
        )
    return result


def run_steering_comparison(
    trial: Any,
    parameters: Any,
    reflex_config: Any,
    *,
    seed: int,
    desired_speed: float,
) -> dict[str, Any]:
    """Compare native muscle and ideal steering using one frozen command sequence."""
    from myosuite_carla.fullbody import FullBodyDriverConfig, build_fullbody_driver

    driver = build_fullbody_driver(FullBodyDriverConfig.fast())
    reference = _episode(driver, trial, parameters, reflex_config, seed, desired_speed)
    muscle = _episode(
        driver,
        trial,
        parameters,
        reflex_config,
        seed,
        desired_speed,
        commands=reference["commands"],
    )
    ideal = _episode(
        driver,
        trial,
        parameters,
        reflex_config,
        seed,
        desired_speed,
        commands=reference["commands"],
        ideal=True,
    )
    episodes = {
        "reference": reference,
        "muscle_replay": muscle,
        "ideal_steering": ideal,
    }
    parity: dict[str, Any] = {}
    for name, episode in episodes.items():
        parity[name] = {
            "initial_physics_exact": episode["initial_state"]
            == reference["initial_state"],
            "same_horizon": len(episode["trace"]) == len(reference["trace"]),
            "longitudinal_commands_exact": all(
                {k: v for k, v in row["applied_command"].items() if k != "steer"}
                == {k: v for k, v in ref["applied_command"].items() if k != "steer"}
                for row, ref in zip(episode["trace"], reference["trace"])
            ),
            "command_times_exact": all(
                row["application_time_s"] == ref["application_time_s"]
                for row, ref in zip(episode["trace"], reference["trace"])
            ),
        }
        episode["trajectory"] = trajectory_comparison(episode.pop("states"), trial)
        episode.pop("commands")
    parity["muscle_replay"]["physical_trajectory_exact"] = (
        muscle["trace"] == reference["trace"]
    )
    return {
        "role": "inspected_source_steering_diagnostic",
        "seed": seed,
        "vehicle_parameters": parameters.as_dict(),
        "reflex_config": asdict(reflex_config),
        "driver_config": asdict(driver.config),
        "comparison_steps": len(reference["trace"]),
        "comparison_horizon_s": reference["trace"][-1]["application_time_s"],
        "parity": parity,
        "episodes": episodes,
    }


def signed_following_errors(
    episodes: Mapping[str, Sequence[Mapping[str, Any]]], groups: Sequence[str]
) -> dict[str, Any]:
    """Describe signed errors on previously inspected complete source pairs."""
    result = {}
    for arm, records in episodes.items():
        sources: dict[str, Any] = {}
        for group in groups:
            rows = [row for row in records if row["source_group"] == group]
            metrics = {}
            for name, suffix in (
                ("speed", "mps"),
                ("gap", "m"),
                ("acceleration", "mps2"),
            ):
                errors = [
                    np.array(row[f"predicted_{name}_{suffix}"])
                    - np.array(row[f"observed_{name}_{suffix}"])
                    for row in rows
                ]
                metrics[name] = {
                    "mean_signed_error": float(
                        np.mean([error.mean() for error in errors])
                    ),
                    "end_signed_error": float(np.mean([error[-1] for error in errors])),
                    "fraction_positive": float(
                        np.mean([np.mean(error > 0) for error in errors])
                    ),
                }
            sources[group] = metrics
        result[arm] = {
            "sources": sources,
            "mean": {
                metric: {
                    key: float(np.mean([row[metric][key] for row in sources.values()]))
                    for key in next(iter(sources.values()))[metric]
                }
                for metric in ("speed", "gap", "acceleration")
            },
        }
    return result
