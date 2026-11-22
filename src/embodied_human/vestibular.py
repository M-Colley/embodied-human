"""Self-motion perception and motion sickness.

Baseline :class:`SubjectiveVertical` is the 6-DOF subjective vertical
conflict model (Bos and Bles, 1998; Kamiji et al., 2007), with Kamiji et
al.'s parameters (2007, Table 1; the same in Wada, Fujisawa and Doi, 2018)
and the conflict scale refitted to vertical-motion data. The vestibular organs sense the
gravito-inertial acceleration ``f = g + a`` (otoliths) and angular
velocity (semicircular canals, a high-pass filter); the brain fuses them
into a sensed vertical and runs an internal model of the same organs on
its own estimate of self-motion. The conflict between the sensed and the
expected vertical, through a Hill function and a second-order lag, gives
the motion sickness incidence (MSI): the percentage of people expected to
vomit. Equations follow the state-space form in Wada and Bos (2025,
Appendix A).

Axes are head-fixed: x forward, y left, z up. ``g`` points up, so an
upright head at rest senses ``f = (0, 0, 9.81)`` m/s².

:class:`IsoDose` implements ISO 2631-1:1997: the Wf-weighted vertical
acceleration gives the motion sickness dose value MSDV_z, and Km MSDV_z
(Km = 1/3) the percentage of unadapted adults who may vomit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from .core import Module, Parameters, Source, register

GRAVITY = 9.81
Array = NDArray[np.float64]

#: Rotation from MuJoCo camera/ocular IMU axes (+X right, +Y up, +Z backward)
#: to head-fixed forward-left-up (FLU) axes.
RUB_TO_FLU: Array = np.array(
    [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float
)

_KAMIJI_2007 = (
    "Kamiji, Kurata, Wada & Doi (2007), Table 1; Wada, Fujisawa & Doi (2018), Table 1",
    "https://doi.org/10.1109/SICE.2007.4421156",
)


@dataclass(frozen=True)
class VerticalConflictParameters(Parameters):
    """6-DOF subjective vertical conflict model (Kamiji et al., 2007).

    Args:
        accel_gain: K_a, accuracy of sensed self-acceleration (internal model).
        rate_gain: K_w, accuracy of sensed angular velocity.
        rate_feedback: K_wc, canal conflict feedback.
        vertical_feedback: K_vc, vertical conflict feedback, 1/s.
        accel_feedback: K_ac, acceleration conflict feedback.
        vertical_lag: tau, otolith-canal fusion time constant, s.
        canal_lag: tau_d, semicircular canal time constant, s.
        adaptation_lag: tau_a, canal adaptation time constant, s.
        conflict_scale: b, conflict at half-maximal sickness drive, m/s².
        sickness_lag: tau_I, time constant of the sickness lag, s.
        max_incidence: P, maximum incidence, %.
    """

    accel_gain: float = 0.1
    rate_gain: float = 0.8
    rate_feedback: float = 5.0
    vertical_feedback: float = 5.0
    accel_feedback: float = 1.0
    vertical_lag: float = 5.0
    canal_lag: float = 7.0
    adaptation_lag: float = 190.0
    conflict_scale: float = 0.196237
    sickness_lag: float = 720.0
    max_incidence: float = 85.0

    SOURCES: ClassVar = {
        name: Source(*_KAMIJI_2007, "fitted")
        for name in (
            "accel_gain",
            "rate_gain",
            "rate_feedback",
            "vertical_feedback",
            "accel_feedback",
            "canal_lag",
            "adaptation_lag",
            "sickness_lag",
            "max_incidence",
        )
    } | {
        "conflict_scale": Source(
            "O'Hanlon & McCauley (1974), Tables 1-2; fitted here",
            "https://archive.org/details/DTIC_AD0768215",
            "fitted",
            "MSI after 2 h at 14 vertical conditions (examples/"
            "fit_motion_sickness.py); Wada et al. (2018) used 0.5 for car "
            "passengers, Bos & Bles (1998) fitted 0.7 in 1-D",
        ),
        "vertical_lag": Source(
            "Wada, Fujisawa & Doi (2018), p. 7; Bos & Bles (2002)",
            "https://doi.org/10.1016/j.ergon.2016.11.003",
            "measured",
            "from vertical perception studies",
        ),
    }


class Vestibular(Module):
    """Category ``vestibular``: sense self-motion; predict sickness."""

    category = "vestibular"

    def simulate(
        self,
        dt: float,
        acceleration: Array,
        angular_velocity: Array | None = None,
        gravity: Array | None = None,
    ) -> Array:
        """MSI (%) over time for head motion sampled every ``dt`` seconds.

        Args:
            acceleration: Inertial head acceleration, (..., T, 3) m/s².
            angular_velocity: Head angular velocity, (..., T, 3) rad/s.
            gravity: Upward gravity in the head frame, (..., T, 3);
                upright by default.
        """
        raise NotImplementedError


def _shape(values: Array | None, like: Array) -> Array:
    return np.zeros_like(like) if values is None else np.asarray(values, dtype=float)


@register("vestibular", "subjective_vertical")
class SubjectiveVertical(Vestibular):
    """Subjective vertical conflict model with otolith-canal interaction."""

    def __init__(self, **params: float) -> None:
        super().__init__()
        self.params = VerticalConflictParameters(**params)

    def _derivative(
        self, x: dict[str, Array], f: Array, a: Array, w: Array
    ) -> dict[str, Array]:
        p = self.params
        rate_sum = 1.0 / p.canal_lag + 1.0 / p.adaptation_lag
        rate_product = 1.0 / (p.canal_lag * p.adaptation_lag)
        # Semicircular canals: band-pass of the head's angular velocity.
        canal_acc = -rate_sum * x["canal_rate"] - rate_product * x["canal"] - w
        sensed_w = rate_sum * x["canal_rate"] + rate_product * x["canal"] + w
        # Sensed vertical from otoliths and canals (generalised Mayne equation).
        dg_s = (f - x["g_s"]) / p.vertical_lag - np.cross(sensed_w, x["g_s"])
        # Internal model of the same organs, driven by estimated motion.
        w_hat = p.rate_gain * w + x["rate_feedback"]
        w_hat_s = x["canal_hat"] + w_hat
        a_hat = p.accel_gain * a + x["accel_feedback"]
        f_hat = a_hat + x["g_hat"]
        dg_hat_s = (f_hat - x["g_hat_s"]) / p.vertical_lag - np.cross(
            w_hat_s, x["g_hat_s"]
        )
        conflict = np.linalg.norm(x["g_s"] - x["g_hat_s"], axis=-1)
        ratio = (conflict / p.conflict_scale) ** 2
        drive = p.max_incidence * ratio / (1.0 + ratio)
        lag = p.sickness_lag
        return {
            "canal": x["canal_rate"],
            "canal_rate": canal_acc,
            "g_s": dg_s,
            "rate_feedback": p.rate_feedback * (sensed_w - w_hat_s),
            "canal_hat": -(x["canal_hat"] + w_hat) / p.canal_lag,
            "accel_feedback": p.accel_feedback
            * ((f - x["g_s"]) - (f_hat - x["g_hat_s"])),
            "g_hat_s": dg_hat_s,
            "g_hat": p.vertical_feedback * (x["g_s"] - x["g_hat_s"]),
            "msi": x["msi_rate"],
            "msi_rate": (drive - x["msi"] - 2.0 * lag * x["msi_rate"]) / lag**2,
        }

    def initial(self, batch: tuple[int, ...], gravity: Array) -> dict[str, Array]:
        """Rest: every estimate of the vertical equals gravity."""
        zero = np.zeros(batch + (3,))
        g0 = np.broadcast_to(gravity, batch + (3,)).copy()
        return {
            "canal": zero.copy(),
            "canal_rate": zero.copy(),
            "g_s": g0.copy(),
            "rate_feedback": zero.copy(),
            "canal_hat": zero.copy(),
            "accel_feedback": zero.copy(),
            "g_hat_s": g0.copy(),
            "g_hat": g0.copy(),
            "msi": np.zeros(batch),
            "msi_rate": np.zeros(batch),
        }

    def simulate(
        self,
        dt: float,
        acceleration: Array,
        angular_velocity: Array | None = None,
        gravity: Array | None = None,
    ) -> Array:
        """MSI at input sample times, with the rest state at sample zero."""
        return self._run(dt, acceleration, angular_velocity, gravity)[1]

    def conflict(
        self,
        dt: float,
        acceleration: Array,
        angular_velocity: Array | None = None,
        gravity: Array | None = None,
    ) -> Array:
        """Magnitude of the vertical conflict (m/s²) over time, the input to
        the sickness stage; arguments as in :meth:`simulate`."""
        return self._run(dt, acceleration, angular_velocity, gravity)[0]

    def _run(
        self,
        dt: float,
        acceleration: Array,
        angular_velocity: Array | None,
        gravity: Array | None,
    ) -> tuple[Array, Array]:
        a_all = np.asarray(acceleration, dtype=float)
        w_all = _shape(angular_velocity, a_all)
        up = np.array([0.0, 0.0, GRAVITY])
        g_all = (
            np.broadcast_to(up, a_all.shape) if gravity is None else np.asarray(gravity)
        )
        batch, steps = a_all.shape[:-2], a_all.shape[-2]
        x = self.initial(batch, g_all[..., 0, :])
        msi = np.zeros(batch + (steps,))
        conflict = np.zeros(batch + (steps,))

        def inputs(k: float) -> tuple[Array, Array, Array]:
            lo = min(int(math.floor(k)), steps - 1)
            hi = min(lo + 1, steps - 1)
            frac = k - lo
            a = (1 - frac) * a_all[..., lo, :] + frac * a_all[..., hi, :]
            w = (1 - frac) * w_all[..., lo, :] + frac * w_all[..., hi, :]
            g = (1 - frac) * g_all[..., lo, :] + frac * g_all[..., hi, :]
            return g + a, a, w

        def shifted(
            x: dict[str, Array], d: dict[str, Array], h: float
        ) -> dict[str, Array]:
            return {key: x[key] + h * d[key] for key in x}

        for k in range(steps - 1):
            # Classic Runge-Kutta; inputs interpolated between samples.
            k1 = self._derivative(x, *inputs(k))
            k2 = self._derivative(shifted(x, k1, dt / 2), *inputs(k + 0.5))
            k3 = self._derivative(shifted(x, k2, dt / 2), *inputs(k + 0.5))
            k4 = self._derivative(shifted(x, k3, dt), *inputs(k + 1))
            x = {
                key: x[key] + dt / 6 * (k1[key] + 2 * k2[key] + 2 * k3[key] + k4[key])
                for key in x
            }
            msi[..., k + 1] = x["msi"]
            conflict[..., k + 1] = np.linalg.norm(x["g_s"] - x["g_hat_s"], axis=-1)
        return conflict, msi


@dataclass(frozen=True)
class IsoDoseParameters(Parameters):
    """ISO 2631-1:1997 motion sickness dose (Annex D)."""

    incidence_constant: float = 1.0 / 3.0

    SOURCES: ClassVar = {
        "incidence_constant": Source(
            "ISO 2631-1:1997; as reproduced by Bos, Diels & Souman (2022), Eq. 2",
            "https://doi.org/10.3390/vibration5040044",
            "standard",
            "mixed population of unadapted adults, 20 min to 6 h of vertical motion",
        )
    }


def wf_weighting(frequency: Array) -> Array:
    """|Wf| of ISO 2631-1 at frequencies in Hz (band-limited a-v filter:
    f1 0.08, f2 0.63, f4 0.25 (Q 0.86), f5 0.0625 (Q 0.8), f6 0.1 (Q 0.8),
    as reproduced by Rimell & Mansfield, 2007)."""
    return np.abs(_wf_response(2j * math.pi * np.asarray(frequency, dtype=float)))


def _wf_response(s: Any) -> Any:
    w1, w2, w4, w5, w6 = (2 * math.pi * f for f in (0.08, 0.63, 0.25, 0.0625, 0.1))
    q = 1 / math.sqrt(2)
    high = s**2 / (s**2 + w1 / q * s + w1**2)
    low = w2**2 / (s**2 + w2 / q * s + w2**2)
    transition = w4**2 / (s**2 + w4 / 0.86 * s + w4**2)
    step = (s**2 + w5 / 0.8 * s + w5**2) / (s**2 + w6 / 0.8 * s + w6**2)
    return high * low * transition * step


@register("vestibular", "iso_2631")
class IsoDose(Vestibular):
    """MSI from the ISO 2631-1 motion sickness dose value (vertical axis)."""

    def __init__(self, **params: float) -> None:
        super().__init__()
        self.params = IsoDoseParameters(**params)

    def simulate(
        self,
        dt: float,
        acceleration: Array,
        angular_velocity: Array | None = None,
        gravity: Array | None = None,
    ) -> Array:
        """Cumulative percentage who may vomit, from the z acceleration."""
        z = np.asarray(acceleration, dtype=float)[..., 2]
        spectrum = np.fft.rfft(z, axis=-1)
        frequencies = np.fft.rfftfreq(z.shape[-1], dt)
        weighted = np.fft.irfft(
            spectrum * _wf_response(2j * math.pi * frequencies), n=z.shape[-1], axis=-1
        )
        dose = np.sqrt(np.cumsum(weighted**2, axis=-1) * dt)
        return self.params.incidence_constant * dose


def _quat_wxyz_to_matrix(quat: Array) -> Array:
    """Convert unit ``(..., 4)`` ``(w, x, y, z)`` quaternions to ``(..., 3, 3)``."""
    q = np.asarray(quat, dtype=float)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm <= 0.0) or not np.all(np.isfinite(q)):
        raise ValueError("orientation_wxyz must contain finite non-zero quaternions")
    q = q / norm
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    flat = np.stack(
        (
            1.0 - 2.0 * (yy + zz),
            2.0 * (xy - wz),
            2.0 * (xz + wy),
            2.0 * (xy + wz),
            1.0 - 2.0 * (xx + zz),
            2.0 * (yz - wx),
            2.0 * (xz - wy),
            2.0 * (yz + wx),
            1.0 - 2.0 * (xx + yy),
        ),
        axis=-1,
    )
    return flat.reshape(q.shape[:-1] + (3, 3))


def imu_to_head_kinematics(
    specific_force: Array,
    angular_velocity: Array | None = None,
    orientation_wxyz: Array | None = None,
    *,
    sensor_to_head: Array | None = None,
    world_up: Array | tuple[float, float, float] = (0.0, 0.0, GRAVITY),
) -> tuple[Array, Array, Array]:
    """Convert IMU readings into head FLU ``(acceleration, angular_velocity, gravity)``.

    Args:
        specific_force: Proper acceleration ``f_s = R_ws^T (a_w + g_up_w)`` in sensor
            axes, ``(..., 3)`` m/s² (as reported by ``mjSENS_ACCELEROMETER``).
        angular_velocity: Angular velocity in sensor axes, ``(..., 3)`` rad/s.
        orientation_wxyz: Sensor-to-world unit quaternion ``(w, x, y, z)``, ``(..., 4)``
            (as reported by ``mjSENS_FRAMEQUAT``). If ``None``, gravity is assumed
            upright in the head frame.
        sensor_to_head: 3x3 rotation from sensor axes to head FLU axes
            (use :data:`RUB_TO_FLU` for MuJoCo camera/``head_imu`` sites).
        world_up: Upward gravity vector in world coordinates, ``(3,)`` m/s².

    Returns:
        Tuple ``(acceleration, angular_velocity, gravity)`` in head FLU coordinates.
    """
    f_s = np.asarray(specific_force, dtype=float)
    if f_s.shape[-1:] != (3,) or not np.all(np.isfinite(f_s)):
        raise ValueError("specific_force must have trailing shape (3,) and be finite")
    w_s = _shape(angular_velocity, f_s)
    if w_s.shape != f_s.shape or not np.all(np.isfinite(w_s)):
        raise ValueError(
            "angular_velocity must match specific_force shape and be finite"
        )
    c_sh = (
        np.eye(3) if sensor_to_head is None else np.asarray(sensor_to_head, dtype=float)
    )
    if c_sh.shape != (3, 3):
        raise ValueError("sensor_to_head must have shape (3, 3)")
    up_w = np.asarray(world_up, dtype=float)
    if up_w.shape != (3,) or not np.all(np.isfinite(up_w)):
        raise ValueError("world_up must be a finite 3-vector")

    if orientation_wxyz is None:
        g_head = np.broadcast_to(up_w, f_s.shape).copy()
        f_head = np.einsum("ij,...j->...i", c_sh, f_s)
        a_head = f_head - g_head
    else:
        r_ws = _quat_wxyz_to_matrix(orientation_wxyz)
        g_s = np.einsum("...ji,j->...i", r_ws, up_w)
        a_s = f_s - g_s
        a_head = np.einsum("ij,...j->...i", c_sh, a_s)
        g_head = np.einsum("ij,...j->...i", c_sh, g_s)

    w_head = np.einsum("ij,...j->...i", c_sh, w_s)
    return a_head, w_head, g_head


class VestibularTracker:
    """Online sample-by-sample RK4 integrator for :class:`SubjectiveVertical`.

    Matches :meth:`SubjectiveVertical.simulate` and :meth:`SubjectiveVertical.conflict`
    at every sample when fed the same ``(acceleration, angular_velocity, gravity)``
    sequence sequentially.
    """

    def __init__(
        self, model: SubjectiveVertical | None = None, **params: float
    ) -> None:
        if model is not None and params:
            raise ValueError("pass either an existing SubjectiveVertical or **params")
        self.model = model if model is not None else SubjectiveVertical(**params)
        self._state: dict[str, Array] | None = None
        self._prev: tuple[Array, Array, Array] | None = None
        self._conflict: Array | None = None
        self._msi: Array | None = None

    def reset(self) -> None:
        """Clear filter states; the next :meth:`step` anchors rest vertical."""
        self._state = None
        self._prev = None
        self._conflict = None
        self._msi = None

    @property
    def msi(self) -> Array:
        """Latest motion sickness incidence (%) after at least one :meth:`step`."""
        if self._msi is None:
            raise RuntimeError("call step() at least once before reading msi")
        return self._msi

    @property
    def conflict(self) -> Array:
        """Latest vertical conflict magnitude (m/s²) after at least one :meth:`step`."""
        if self._conflict is None:
            raise RuntimeError("call step() at least once before reading conflict")
        return self._conflict

    def step(
        self,
        dt: float,
        acceleration: Array,
        angular_velocity: Array | None = None,
        gravity: Array | None = None,
    ) -> tuple[Array, Array]:
        """Advance one sample and return ``(conflict, msi)``."""
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError(f"dt must be finite and positive, got {dt}")
        a = np.asarray(acceleration, dtype=float)
        if a.shape[-1:] != (3,) or not np.all(np.isfinite(a)):
            raise ValueError("acceleration must have trailing shape (3,) and be finite")
        w = _shape(angular_velocity, a)
        up = np.array([0.0, 0.0, GRAVITY])
        g = (
            np.broadcast_to(up, a.shape).copy()
            if gravity is None
            else np.asarray(gravity, dtype=float)
        )
        if w.shape != a.shape or g.shape != a.shape:
            raise ValueError(
                "angular_velocity and gravity must match acceleration shape"
            )
        if not (np.all(np.isfinite(w)) and np.all(np.isfinite(g))):
            raise ValueError("angular_velocity and gravity must be finite")

        batch = a.shape[:-1]
        if self._state is None or self._prev is None:
            self._state = self.model.initial(batch, g)
            self._prev = (a.copy(), w.copy(), g.copy())
            self._conflict = np.zeros(batch, dtype=float)
            self._msi = np.zeros(batch, dtype=float)
            return self._conflict.copy(), self._msi.copy()

        a0, w0, g0 = self._prev
        if a0.shape != a.shape:
            raise ValueError(f"batch shape changed from {a0.shape[:-1]} to {batch}")
        x = self._state
        a_mid = 0.5 * (a0 + a)
        w_mid = 0.5 * (w0 + w)
        g_mid = 0.5 * (g0 + g)

        def shifted(
            s: dict[str, Array], d: dict[str, Array], h: float
        ) -> dict[str, Array]:
            return {k: s[k] + h * d[k] for k in s}

        k1 = self.model._derivative(x, g0 + a0, a0, w0)
        k2 = self.model._derivative(
            shifted(x, k1, 0.5 * dt), g_mid + a_mid, a_mid, w_mid
        )
        k3 = self.model._derivative(
            shifted(x, k2, 0.5 * dt), g_mid + a_mid, a_mid, w_mid
        )
        k4 = self.model._derivative(shifted(x, k3, dt), g + a, a, w)
        x = {
            k: x[k] + (dt / 6.0) * (k1[k] + 2.0 * k2[k] + 2.0 * k3[k] + k4[k])
            for k in x
        }
        self._state = x
        self._prev = (a.copy(), w.copy(), g.copy())
        self._msi = np.asarray(x["msi"], dtype=float)
        self._conflict = np.asarray(
            np.linalg.norm(x["g_s"] - x["g_hat_s"], axis=-1), dtype=float
        )
        return self._conflict.copy(), self._msi.copy()
