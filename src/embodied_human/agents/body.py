"""What the simulated body feels: head motion and muscle effort.

myosuite-carla's cockpit model is fixed in the vehicle: MuJoCo's world
frame moves with the car (x left, y back, z up), its gravity is switched
off, and gravity and the vehicle's motion act on the body as d'Alembert
loads. The head's motion in the inertial world is therefore the vehicle's
motion composed with the head's motion relative to the car:

    a_head = a_vehicle + alpha x r + omega x (omega x r)
             + 2 omega x v_relative + a_relative,
    omega_head = omega_vehicle + omega_relative,

expressed in the head's own frame (x forward, y left, z up at rest, turning
with the head). The lever arm r extends from the vehicle reference point to
the head's centre of mass. The configured seat offset locates that reference
point relative to the pelvis.

Muscle effort is each muscle group's mean activation, a fraction of its
maximum, which drives the fatigue module.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ..vestibular import GRAVITY, RUB_TO_FLU, imu_to_head_kinematics

#: Muscle groups of myosuite-carla's driving muscles, by actuator name
#: (left-side actuators end in ``_l``).
MUSCLE_GROUPS = {
    "elbow": (
        "TRIlong",
        "TRIlat",
        "TRImed",
        "ANC",
        "BIClong",
        "BICshort",
        "BRA",
        "BRD",
    ),
    "hand": ("FDS5", "FDS2", "FDP5", "FDP2", "EDC5", "EDC2", "EDM", "EIP"),
    "ankle": ("gaslat", "gasmed", "soleus", "tibant"),
}

#: Forward, left, up in myosuite-carla's cockpit frame (x left, y back, z up).
_FLU_IN_COCKPIT = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


class BodySensing:
    """Read head kinematics and muscle activations from a full-body env.

    Args:
        env: A myosuite-carla ``FullBodyDrivingEnv`` (``env.driver`` holds
            the MuJoCo model, data and the vehicle motion).
        head: Name of the head body.
    """

    def __init__(self, env: Any, head: str = "head") -> None:
        import mujoco
        from myosuite_carla.vehicle_loads import carla_vector_to_cockpit

        self.driver = env.driver
        model = self.driver.model
        self.head = int(model.body(head).id)
        self._pelvis = int(model.body("pelvis").id)
        self._seat_offset = carla_vector_to_cockpit(self.driver.config.seat_offset)
        self._mujoco = mujoco
        self.groups: dict[str, list[int]] = {}
        names = {model.actuator(i).name: i for i in range(model.nu)}
        for group, muscles in MUSCLE_GROUPS.items():
            for side, suffix in (("right", ""), ("left", "_l")):
                found = []
                for muscle in muscles:
                    # Ankle muscles carry _r/_l; arm muscles only mark the left.
                    for name in (
                        muscle + suffix,
                        muscle + ("_r" if side == "right" else "_l"),
                    ):
                        if name in names:
                            found.append(int(model.actuator_actadr[names[name]]))
                            break
                if found:
                    self.groups[f"{group}_{side}"] = found
        self.reset()

    def reset(self) -> None:
        """Take the current head orientation as upright."""
        data = self.driver.data
        self._rest = np.array(data.xmat[self.head]).reshape(3, 3)

    def loads(self) -> dict[str, float]:
        """Mean activation per muscle group and side, 0-1."""
        act = self.driver.data.act
        return {
            group: float(np.mean(act[index])) for group, index in self.groups.items()
        }

    def head_motion(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Head acceleration, angular velocity and upward gravity, m/s^2 and
        rad/s, in the head frame (x forward, y left, z up)."""
        mujoco, model, data = self._mujoco, self.driver.model, self.driver.data
        relative = np.zeros(6)
        velocity = np.zeros(6)
        mujoco.mj_rnePostConstraint(model, data)
        mujoco.mj_objectAcceleration(
            model, data, mujoco.mjtObj.mjOBJ_BODY, self.head, relative, 0
        )
        mujoco.mj_objectVelocity(
            model, data, mujoco.mjtObj.mjOBJ_BODY, self.head, velocity, 0
        )
        vehicle = self.driver.motion.in_cockpit()
        origin = data.xpos[self._pelvis] - self._seat_offset
        acceleration = self.driver.motion.acceleration_at(
            data.xipos[self.head] - origin,
            relative_velocity=velocity[3:],
            relative_acceleration=relative[3:],
        )
        angular = vehicle["angular_velocity"] + velocity[:3]
        up = -np.asarray(vehicle["gravity"], dtype=float)
        # Cockpit -> head at rest (forward, left, up) -> head now.
        orientation = np.array(data.xmat[self.head]).reshape(3, 3)
        to_head = _FLU_IN_COCKPIT.T @ self._rest @ orientation.T
        return to_head @ acceleration, to_head @ angular, to_head @ up


_EXTENDED_MUSCLE_GROUPS: dict[str, tuple[str, ...]] = {
    **MUSCLE_GROUPS,
    "shoulder": ("DELT1", "DELT2", "DELT3", "SUPSP", "INFSP", "SUBSC", "TMIN"),
}


def _unwrap_ptr(obj: Any) -> Any:
    return getattr(obj, "ptr", obj)


def _resolve_model_and_data(target: Any) -> tuple[Any, Any]:
    if isinstance(target, tuple) and len(target) == 2:
        return _unwrap_ptr(target[0]), _unwrap_ptr(target[1])
    for candidate in (target, getattr(target, "unwrapped", None)):
        if candidate is None:
            continue
        sim = getattr(candidate, "sim", None)
        if sim is not None and hasattr(sim, "model") and hasattr(sim, "data"):
            return _unwrap_ptr(sim.model), _unwrap_ptr(sim.data)
        if hasattr(candidate, "model") and hasattr(candidate, "data"):
            return _unwrap_ptr(candidate.model), _unwrap_ptr(candidate.data)
    raise TypeError(
        "target must be a (MjModel, MjData) pair or an object exposing model/data"
    )


def _sensor_slice(mujoco: Any, model: Any, name: str | None, dim: int) -> slice | None:
    if not name:
        return None
    sid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name))
    if sid < 0 or int(model.sensor_dim[sid]) != dim:
        return None
    adr = int(model.sensor_adr[sid])
    return slice(adr, adr + dim)


class MujocoHeadSensing:
    """Read head kinematics (from ``head_imu`` sensors or body CoM) and muscle loads.

    Works on any standalone ``(MjModel, MjData)`` pair or Gymnasium/MyoSuite
    environment without requiring ``myosuite-carla`` or a ``pelvis`` body.
    """

    def __init__(
        self,
        target: Any,
        *,
        head: str = "head",
        imu_site: str | None = "head_imu",
        accel_sensor: str | None = "head_accel",
        gyro_sensor: str | None = "head_gyro",
        quat_sensor: str | None = "head_quat",
        sensor_to_head: np.ndarray | None = None,
        flu_in_world: np.ndarray | None = None,
        pivot: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> None:
        import mujoco

        self._target = target
        self.model, self.data = _resolve_model_and_data(target)
        self._mujoco = mujoco
        head_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, head))
        if head_id < 0:
            raise KeyError(f"Body {head!r} not found in MuJoCo model")
        self.head = head_id

        self._imu_site_id = (
            int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, imu_site))
            if imu_site
            else -1
        )
        self._accel_slice = _sensor_slice(mujoco, self.model, accel_sensor, 3)
        self._gyro_slice = _sensor_slice(mujoco, self.model, gyro_sensor, 3)
        self._quat_slice = _sensor_slice(mujoco, self.model, quat_sensor, 4)
        self.has_imu_sensors = (
            self._accel_slice is not None
            and self._gyro_slice is not None
            and self._quat_slice is not None
        )
        self._sensor_to_head = (
            RUB_TO_FLU
            if sensor_to_head is None
            else np.asarray(sensor_to_head, dtype=float)
        )
        self._explicit_flu_in_world = (
            None if flu_in_world is None else np.asarray(flu_in_world, dtype=float)
        )
        self._flu_in_world = np.eye(3, dtype=float)
        self.pivot = np.asarray(pivot, dtype=float)
        if self.pivot.shape != (3,):
            raise ValueError("pivot must have shape (3,)")

        self.groups: dict[str, list[int]] = {}
        names = {self.model.actuator(i).name: i for i in range(self.model.nu)}
        for group, muscles in _EXTENDED_MUSCLE_GROUPS.items():
            for side, suffix in (("right", ""), ("left", "_l")):
                found = []
                for muscle in muscles:
                    for cand in (
                        muscle + suffix,
                        muscle + ("_r" if side == "right" else "_l"),
                    ):
                        if cand in names:
                            act_adr = int(self.model.actuator_actadr[names[cand]])
                            if act_adr >= 0:
                                found.append(act_adr)
                            break
                if found:
                    self.groups[f"{group}_{side}"] = found
        self.reset()

    def _refresh_data_ref(self) -> None:
        self.model, self.data = _resolve_model_and_data(self._target)

    def reset(self) -> None:
        """Capture rest head orientation and world-frame FLU reference axes."""
        self._refresh_data_ref()
        self._mujoco.mj_forward(self.model, self.data)
        self._rest = np.array(self.data.xmat[self.head], dtype=float).reshape(3, 3)
        if self._explicit_flu_in_world is not None:
            self._flu_in_world = self._explicit_flu_in_world.copy()
        elif self._imu_site_id >= 0:
            site_mat = np.array(
                self.data.site_xmat[self._imu_site_id], dtype=float
            ).reshape(3, 3)
            self._flu_in_world = site_mat @ self._sensor_to_head.T
        else:
            self._flu_in_world = np.eye(3, dtype=float)

    def loads(self) -> dict[str, float]:
        """Mean muscle activation ``[0, 1]`` per discovered bilateral muscle group."""
        self._refresh_data_ref()
        if self.model.na == 0:
            return {}
        act = self.data.act
        return {
            group: float(np.mean(act[indices]))
            for group, indices in self.groups.items()
        }

    def _world_up(self) -> np.ndarray:
        grav = np.asarray(self.model.opt.gravity, dtype=float)
        if np.linalg.norm(grav) < 1e-9:
            return np.array([0.0, 0.0, GRAVITY], dtype=float)
        return -grav

    def head_motion(
        self,
        *,
        base_lin_acc: Sequence[float] | None = None,
        base_ang_vel: Sequence[float] | None = None,
        base_ang_acc: Sequence[float] | None = None,
        pivot: Sequence[float] | None = None,
        prefer_imu: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(acceleration, angular_velocity, gravity)`` in head FLU axes."""
        self._refresh_data_ref()
        mujoco, model, data = self._mujoco, self.model, self.data

        env_kin = getattr(self._target, "base_kinematics", None)
        has_base = (
            base_lin_acc is not None
            or base_ang_vel is not None
            or base_ang_acc is not None
            or env_kin is not None
        )
        up_w = self._world_up()

        if (
            prefer_imu
            and self.has_imu_sensors
            and not has_base
            and self._accel_slice is not None
            and self._gyro_slice is not None
            and self._quat_slice is not None
        ):
            mujoco.mj_rnePostConstraint(model, data)
            mujoco.mj_sensorAcc(model, data)
            f_s = np.array(data.sensordata[self._accel_slice], dtype=float)
            w_s = np.array(data.sensordata[self._gyro_slice], dtype=float)
            q_ws = np.array(data.sensordata[self._quat_slice], dtype=float)
            return imu_to_head_kinematics(
                f_s,
                w_s,
                q_ws,
                sensor_to_head=self._sensor_to_head,
                world_up=up_w,
            )

        mujoco.mj_rnePostConstraint(model, data)
        rel_acc6 = np.zeros(6, dtype=float)
        rel_vel6 = np.zeros(6, dtype=float)
        if self._imu_site_id >= 0:
            obj_type = mujoco.mjtObj.mjOBJ_SITE
            obj_id = self._imu_site_id
            locus = np.array(data.site_xpos[self._imu_site_id], dtype=float)
        else:
            obj_type = mujoco.mjtObj.mjOBJ_BODY
            obj_id = self.head
            locus = np.array(data.xipos[self.head], dtype=float)

        mujoco.mj_objectAcceleration(model, data, obj_type, obj_id, rel_acc6, 0)
        mujoco.mj_objectVelocity(model, data, obj_type, obj_id, rel_vel6, 0)
        # Subtract upward gravity that mj_objectAcceleration includes via qacc0/rne.
        grav_w = np.asarray(model.opt.gravity, dtype=float)
        a_rel = rel_acc6[3:] + grav_w
        v_rel = rel_vel6[3:]
        w_rel = rel_vel6[:3]

        a0 = (
            np.asarray(base_lin_acc, dtype=float)
            if base_lin_acc is not None
            else (
                np.asarray(env_kin.lin_acc, dtype=float)
                if env_kin is not None
                else np.zeros(3, dtype=float)
            )
        )
        w0 = (
            np.asarray(base_ang_vel, dtype=float)
            if base_ang_vel is not None
            else (
                np.asarray(env_kin.ang_vel, dtype=float)
                if env_kin is not None
                else np.zeros(3, dtype=float)
            )
        )
        alpha0 = (
            np.asarray(base_ang_acc, dtype=float)
            if base_ang_acc is not None
            else (
                np.asarray(env_kin.ang_acc, dtype=float)
                if env_kin is not None
                else np.zeros(3, dtype=float)
            )
        )
        p0 = self.pivot if pivot is None else np.asarray(pivot, dtype=float)
        r = locus - p0

        a_world = (
            a0
            + np.cross(alpha0, r)
            + np.cross(w0, np.cross(w0, r))
            + 2.0 * np.cross(w0, v_rel)
            + a_rel
        )
        w_world = w0 + w_rel

        orientation = np.array(data.xmat[self.head], dtype=float).reshape(3, 3)
        to_head = self._flu_in_world.T @ self._rest @ orientation.T
        return to_head @ a_world, to_head @ w_world, to_head @ up_w
