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

from typing import Any

import numpy as np

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
