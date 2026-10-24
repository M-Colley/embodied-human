"""Proprioception from a MuJoCo musculoskeletal model.

Muscle spindle Ia afferents fire at

    f = 4.3 |v|^0.6 sign(v) + 2 d + r0   impulses/s,

with stretch velocity ``v`` in mm/s and stretch ``d`` in mm from the
reference length (Prochazka and Gorassini, 1998; fitted to cat hamstring;
r0 = 10 imp/s for human wrist muscles per Malik et al., 2016). Joint angles
are sensed with Gaussian error and all signals arrive after the
short-latency reflex delay.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from .core import Module, Parameters, Source, register


@dataclass(frozen=True)
class SpindleParameters(Parameters):
    """Spindle rate model, joint position sense and delay."""

    velocity_gain: float = 4.3
    velocity_exponent: float = 0.6
    length_gain: float = 2.0
    baseline: float = 10.0
    joint_noise_deg: float = 3.0
    delay: float = 0.025

    SOURCES: ClassVar = {
        "velocity_gain": Source(
            "Prochazka & Gorassini (1998)",
            "https://doi.org/10.1111/j.1469-7793.1998.277bu.x",
            "fitted",
            "cat hamstring Ia afferents during treadmill walking",
        ),
        "velocity_exponent": Source(
            "Prochazka & Gorassini (1998)",
            "https://doi.org/10.1111/j.1469-7793.1998.277bu.x",
            "fitted",
        ),
        "length_gain": Source(
            "Prochazka & Gorassini (1998)",
            "https://doi.org/10.1111/j.1469-7793.1998.277bu.x",
            "fitted",
        ),
        "baseline": Source(
            "Malik, Jabakhanji & Jones (2016)",
            "https://doi.org/10.3389/fncom.2015.00154",
            "fitted",
            "human wrist; 82 imp/s for cat",
        ),
        "joint_noise_deg": Source(
            "Talis & Levik (2016)",
            "https://doi.org/10.3389/fnins.2016.00520",
            "measured",
            "elbow contralateral matching, young adults 3.0 deg "
            "absolute error; shoulder 3-6 deg",
        ),
        "delay": Source(
            "Pruszynski, Kurtzer, Lillicrap & Scott (2009)",
            "https://doi.org/10.1152/jn.00085.2009",
            "measured",
            "short-latency stretch response 20-45 ms",
        ),
    }


@dataclass(frozen=True)
class ProprioceptiveSample:
    """Delayed proprioceptive signals.

    Attributes:
        time: Time the signals arrive, s.
        sensed_time: Time the body was in this state, s.
        ia_rates: Ia firing rate per muscle, imp/s.
        joint_angles: Sensed hinge/slide joint positions (noisy).
        muscle_lengths: Muscle-tendon lengths, m.
        muscle_forces: Actuator forces, N.
    """

    time: float
    sensed_time: float
    ia_rates: NDArray[np.float64]
    joint_angles: NDArray[np.float64]
    muscle_lengths: NDArray[np.float64]
    muscle_forces: NDArray[np.float64]


class Proprioception(Module):
    """Category ``proprioception``: sense the body's own state."""

    category = "proprioception"

    def bind(self, model: Any, data: Any) -> None:
        """Resolve muscles and joints; take the current state as reference."""
        raise NotImplementedError

    def sense(self, data: Any) -> ProprioceptiveSample | None:
        """Record the current state and return the delayed sample."""
        raise NotImplementedError


@register("proprioception", "spindle")
class SpindleProprioception(Proprioception):
    """Ia spindle rates, noisy joint position sense and a fixed delay."""

    def __init__(self, **params: float) -> None:
        super().__init__()
        self.params = SpindleParameters(**params)
        self._model: Any = None
        self.reset()

    def reset(self, rng: np.random.Generator | None = None) -> None:
        super().reset(rng)
        self._buffer: deque[ProprioceptiveSample] = deque()

    def bind(self, model: Any, data: Any) -> None:
        import mujoco

        self._model = model
        self.muscles = np.flatnonzero(
            model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE
        )
        scalar = (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))
        joints = [j for j in range(model.njnt) if int(model.jnt_type[j]) in scalar]
        self.joint_qpos = np.asarray([model.jnt_qposadr[j] for j in joints], dtype=int)
        self.joint_hinge = np.asarray(
            [int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_HINGE) for j in joints]
        )
        self.reference = np.asarray(
            data.actuator_length[self.muscles], dtype=float
        ).copy()
        self.reset(self.rng)

    def ia_rates(
        self, length: NDArray[np.float64], velocity: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Ia rates for lengths and velocities in metres and m/s."""
        p = self.params
        stretch_mm = 1000.0 * (length - self.reference)
        speed_mm = 1000.0 * velocity
        rate = (
            p.velocity_gain
            * np.sign(speed_mm)
            * np.abs(speed_mm) ** p.velocity_exponent
            + p.length_gain * stretch_mm
            + p.baseline
        )
        return np.maximum(rate, 0.0)

    def sense(self, data: Any) -> ProprioceptiveSample | None:
        if self._model is None:
            raise RuntimeError("bind the proprioception module to a model first")
        time = float(data.time)
        noise = np.where(
            self.joint_hinge, math.radians(self.params.joint_noise_deg), 0.0
        )
        sample = ProprioceptiveSample(
            time + self.params.delay,
            time,
            self.ia_rates(
                data.actuator_length[self.muscles], data.actuator_velocity[self.muscles]
            ),
            data.qpos[self.joint_qpos]
            + noise * self.rng.standard_normal(len(self.joint_qpos)),
            np.asarray(data.actuator_length[self.muscles], dtype=float).copy(),
            np.asarray(data.actuator_force[self.muscles], dtype=float).copy(),
        )
        self._buffer.append(sample)
        latest = None
        while self._buffer and self._buffer[0].time <= time + 1e-12:
            latest = self._buffer.popleft()
        return latest
