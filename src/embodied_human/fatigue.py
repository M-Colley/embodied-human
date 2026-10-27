"""Muscle fatigue: how sustained effort wears a person down.

Baseline :class:`ThreeCompartment` is the three-compartment muscle fatigue
model of Xia and Frey-Law (2008): the motor units of a muscle group are
active (``MA``), fatigued (``MF``) or resting (``MR``), as fractions of the
group. A controller recruits resting units to follow the target load
(fraction of maximum voluntary contraction, MVC):

    dMA/dt = C(t) - F MA
    dMF/dt = F MA - R_eff MF
    dMR/dt = R_eff MF - C(t)

with ``C(t) = L_D (TL - MA)`` while more activity is needed and resting
units are available, ``L_D MR`` when they are not, and ``L_R (TL - MA)``
when activity exceeds the target. Recovery speeds up by ``r`` during rest
(Looft, Herkert and Frey-Law, 2018). The capacity left, ``MA + MR``,
falls as fatigue builds; the target can no longer be held once it drops
below the load, which defines the endurance time.

Any MuJoCo or MyoSuite muscle group can drive the model with its
activation as the target load. :func:`consumed_endurance` gives the
simpler Consumed Endurance metric of mid-air interaction for comparison.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from .core import Module, Parameters, Source, register


@dataclass(frozen=True)
class JointRates:
    """Fatigue and recovery rates of one joint's muscles, 1/s."""

    fatigue: float
    recovery: float


_FREY_LAW_2012 = (
    "Frey-Law, Looft & Heitsman (2012), Table 1",
    "https://doi.org/10.1016/j.jbiomech.2012.04.018",
)

#: F and R (1/s) per joint, fitted to the endurance-time meta-analysis of
#: Frey-Law and Avin (2010) at 10-90% MVC.
JOINTS = {
    "ankle": JointRates(0.00589, 0.00058),
    "knee": JointRates(0.01500, 0.00149),
    "trunk": JointRates(0.00755, 0.00075),
    "shoulder": JointRates(0.01820, 0.00168),
    "elbow": JointRates(0.00912, 0.00094),
    "hand": JointRates(0.00980, 0.00064),
    "general": JointRates(0.00970, 0.00091),
}

#: Rest multiplier r per joint (Looft, Herkert and Frey-Law, 2018). No
#: shoulder or trunk data exist; the authors suggest 15 for them.
REST_MULTIPLIERS = {"hand": 30.0}


@dataclass(frozen=True)
class FatigueParameters(Parameters):
    """Three-compartment model parameters.

    Args:
        joints: Fatigue rate F and recovery rate R per joint, 1/s.
        rest_multiplier: Factor r on R while the target load is zero.
        rest_multipliers: Joint-specific overrides of ``rest_multiplier``.
        develop: Controller gain L_D, 1/s.
        relax: Controller gain L_R, 1/s.
    """

    joints: dict[str, JointRates] = field(default_factory=lambda: dict(JOINTS))
    rest_multiplier: float = 15.0
    rest_multipliers: dict[str, float] = field(
        default_factory=lambda: dict(REST_MULTIPLIERS)
    )
    develop: float = 10.0
    relax: float = 10.0

    SOURCES: ClassVar = {
        "joints": Source(*_FREY_LAW_2012, "fitted", "per-second rates"),
        "rest_multiplier": Source(
            "Looft, Herkert & Frey-Law (2018), Tables 2-4",
            "https://doi.org/10.1016/j.jbiomech.2018.06.005",
            "fitted",
            "15 for ankle, knee, elbow and the general model; suggested for "
            "shoulder and trunk, which had no data",
        ),
        "rest_multipliers": Source(
            "Looft, Herkert & Frey-Law (2018), Tables 2-4",
            "https://doi.org/10.1016/j.jbiomech.2018.06.005",
            "fitted",
            "30 for hand/grip",
        ),
        "develop": Source(
            *_FREY_LAW_2012, "fitted", "L = 10 from Xia & Frey-Law (2008)"
        ),
        "relax": Source(*_FREY_LAW_2012, "fitted"),
    }


class Fatigue(Module):
    """Category ``fatigue``: track the effects of effort over time."""

    category = "fatigue"

    def step(self, dt: float, loads: dict[str, float]) -> dict[str, float]:
        """Advance by ``dt`` with target loads (fraction of MVC) per joint
        and return the capacity left per joint (fraction of MVC)."""
        raise NotImplementedError


@register("fatigue", "three_compartment")
class ThreeCompartment(Fatigue):
    """Xia and Frey-Law's (2008) three-compartment model per joint."""

    def __init__(self, **params: object) -> None:
        super().__init__()
        self.params = FatigueParameters(**params)  # type: ignore[arg-type]
        self.reset()

    def reset(self, rng: np.random.Generator | None = None) -> None:
        super().reset(rng)
        # Per joint: active, fatigued, resting fractions.
        self.state: dict[str, np.ndarray] = {}

    def _rates(self, joint: str) -> JointRates:
        """Rates of a joint; ``elbow_left`` uses the ``elbow`` rates."""
        joints = self.params.joints
        try:
            return joints[joint] if joint in joints else joints[joint.split("_")[0]]
        except KeyError:
            known = ", ".join(sorted(self.params.joints))
            raise KeyError(f"no fatigue rates for {joint!r}; known: {known}") from None

    def step(self, dt: float, loads: dict[str, float]) -> dict[str, float]:
        if not math.isfinite(dt) or dt < 0:
            raise ValueError("dt must be finite and non-negative")
        if any(not math.isfinite(load) for load in loads.values()):
            raise ValueError("loads must be finite")
        p = self.params
        capacity = {}
        for joint, load in loads.items():
            rates = self._rates(joint)
            ma, mf, mr = self.state.get(joint, np.array([0.0, 0.0, 1.0]))
            target = min(max(float(load), 0.0), 1.0)
            if ma < target:
                drive = (
                    p.develop * (target - ma) if mr > target - ma else p.develop * mr
                )
            else:
                drive = p.relax * (target - ma)
            kind = joint if joint in p.rest_multipliers else joint.split("_")[0]
            rest = p.rest_multipliers.get(kind, p.rest_multiplier)
            # Recovery speeds up at rest (Looft et al., 2018); it applies to
            # both the resting and fatigued pools so that units are conserved.
            recovery = rates.recovery * (rest if target == 0.0 else 1.0)
            dma = drive - rates.fatigue * ma
            dmf = rates.fatigue * ma - recovery * mf
            dmr = recovery * mf - drive
            # Explicit Euler is stable for dt well below 1 / L_D.
            ma, mf, mr = ma + dt * dma, mf + dt * dmf, mr + dt * dmr
            state = np.clip(np.array([ma, mf, mr]), 0.0, 1.0)
            self.state[joint] = state / state.sum()
            capacity[joint] = float(self.state[joint][0] + self.state[joint][2])
        return capacity

    def fatigued(self, joint: str) -> float:
        """Fraction of the joint's motor units that are fatigued."""
        return float(self.state.get(joint, np.zeros(3))[1])

    def endurance_time(
        self, joint: str, load: float, dt: float = 0.05, limit: float = 7200.0
    ) -> float:
        """Seconds a constant load (fraction of MVC) can be held from rest;
        ``inf`` if it can be held for ``limit`` seconds."""
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        if not math.isfinite(limit) or limit <= 0:
            raise ValueError("limit must be finite and positive")
        if not math.isfinite(load) or not 0 <= load <= 1:
            raise ValueError("load must be finite and lie in [0, 1]")
        saved = dict(self.state)
        self.state.pop(joint, None)
        time = 0.0
        try:
            while time < limit:
                time += dt
                if self.step(dt, {joint: load})[joint] < load - 1e-9:
                    return time
            return math.inf
        finally:
            self.state = saved


def rohmert_endurance(load: float) -> float:
    """Endurance time (s) of a static load (fraction of maximum) by the
    Rohmert-type curve used by Consumed Endurance (Hincapie-Ramos et al.,
    2014, Eq. 1); infinite below 15%."""
    percent = 100.0 * load
    if percent <= 15.0:
        return math.inf
    return 1236.5 / (percent - 15.0) ** 0.618 - 72.5


def consumed_endurance(load: float, seconds: float) -> float:
    """Consumed Endurance (%), Hincapie-Ramos et al. (2014), Eq. 7."""
    endurance = rohmert_endurance(load)
    return 0.0 if math.isinf(endurance) else 100.0 * seconds / endurance
