"""Optional native PyDDM choice distributions alongside the TWIN detection API.

The explicit choice pathway solves a constant drift diffusion model. Sensory
signals are not converted into evidence: callers provide a drift rate and a
deadline, and tasks state which boundary is the correct response.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from importlib import import_module
from typing import ClassVar, Literal

import numpy as np
from numpy.typing import NDArray

from .core import Source, register
from .response import TwinParameters, TwinResponse

Choice = Literal["upper", "lower"]
_SOURCE = "https://pyddm.readthedocs.io/en/stable/quickstart.html"


@dataclass(frozen=True)
class PyDDMParameters(TwinParameters):
    """TWIN detection parameters plus an explicitly separate choice protocol."""

    decision_bound: float = 1.0
    decision_noise: float = 1.0
    nondecision_time: float = 0.2
    starting_position: float = 0.0
    decision_dt: float = 0.005
    decision_dx: float = 0.005

    SOURCES: ClassVar = {
        **TwinParameters.SOURCES,
        **{
            name: Source("PyDDM illustrative choice protocol", _SOURCE, "assumed", note)
            for name, note in {
                "decision_bound": "symmetric bounds in decision units; not fitted",
                "decision_noise": "decision units / sqrt(s); not fitted",
                "nondecision_time": "seconds; independent of TWIN detection",
                "starting_position": "fraction of the bound; 0 is centred",
                "decision_dt": "maximum solver time step in seconds; numerical setting",
                "decision_dx": "spatial step in decision units; numerical setting",
            }.items()
        },
    }

    def __post_init__(self) -> None:
        for name in (
            "decision_bound",
            "decision_noise",
            "nondecision_time",
            "starting_position",
            "decision_dt",
            "decision_dx",
        ):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if any(
            value <= 0
            for value in (
                self.decision_bound,
                self.decision_noise,
                self.decision_dt,
                self.decision_dx,
            )
        ):
            raise ValueError("decision bounds, noise and solver steps must be positive")
        if self.nondecision_time < 0:
            raise ValueError("nondecision_time must be non-negative")
        if not -1 < self.starting_position < 1:
            raise ValueError("starting_position must lie strictly between -1 and 1")
        if self.decision_dx > self.decision_bound:
            raise ValueError("decision_dx must not exceed the bound")


@dataclass(frozen=True)
class ChoiceDistribution:
    """Joint boundary and response time density, including deadline misses.

    Density arrays integrate by the native grid rule, ``sum(pdf) * dt``.
    Their masses are unconditional. Mean response time conditions on a response;
    timeout trials have no invented reaction time or forced boundary choice.
    """

    time: NDArray[np.float64]
    upper_density: NDArray[np.float64]
    lower_density: NDArray[np.float64]
    dt: float
    deadline: float
    upper_probability: float
    lower_probability: float
    timeout_probability: float
    mean_rt: float | None
    solver_version: str

    def accuracy(self, correct: Choice, *, conditional: bool = False) -> float | None:
        if correct not in ("upper", "lower"):
            raise ValueError("correct must be 'upper' or 'lower'")
        probability = (
            self.upper_probability if correct == "upper" else self.lower_probability
        )
        if not conditional:
            return probability
        responded = self.upper_probability + self.lower_probability
        return probability / responded if responded else None


@dataclass(frozen=True)
class ChoiceReaction:
    """One sampled choice; a timeout has both fields set to None."""

    choice: Choice | None
    time: float | None

    @property
    def timeout(self) -> bool:
        return self.choice is None


@register("response", "pyddm")
class PyDDMResponse(TwinResponse):
    """TWIN signal detection plus a separate native PyDDM choice pathway.

    Inherited ``respond(signals, load)`` and ``visual_latency`` retain their
    existing TWIN meaning and parameters. Only ``solve_choice`` and ``decide``
    use PyDDM, with an explicit drift and deadline in seconds. No improvement
    in DRT or takeover is implied by selecting this implementation.
    """

    params: PyDDMParameters

    def __init__(self, **params: float) -> None:
        settings = PyDDMParameters(**params)
        super().__init__(
            **{name: getattr(settings, name) for name in TwinParameters.SOURCES}
        )
        self.params = settings

    def solve_choice(self, drift: float, deadline: float) -> ChoiceDistribution:
        """Solve an explicit two boundary experiment with the installed PyDDM.

        Positive drift favours the upper boundary. The deadline includes the
        independent nondecision time. ``mixture_coef=0`` disables PyDDM's
        default contaminant mixture, so no unexplained lapses are introduced.
        """
        if not math.isfinite(drift):
            raise ValueError("drift must be finite")
        if not math.isfinite(deadline) or deadline <= 0:
            raise ValueError("deadline must be finite and positive")
        try:
            native = import_module("pyddm")
        except ModuleNotFoundError as exc:
            if exc.name != "pyddm":
                raise
            raise ImportError(
                "Choice distributions require "
                "python -m pip install 'embodied-human[decision]'"
            ) from exc
        p = self.params
        # End exactly at the requested deadline without increasing the time step.
        dt = deadline / max(1, math.ceil(deadline / p.decision_dt))
        model = native.gddm(
            drift=float(drift),
            noise=p.decision_noise,
            bound=p.decision_bound,
            nondecision=p.nondecision_time,
            starting_position=p.starting_position,
            mixture_coef=0,
            dx=p.decision_dx,
            dt=dt,
            T_dur=float(deadline),
            choice_names=("upper", "lower"),
        )
        solution = model.solve()
        arrays = [
            np.array(values, dtype=float, copy=True)
            for values in (
                solution.t_domain,
                solution.pdf("upper"),
                solution.pdf("lower"),
            )
        ]
        for values in arrays:
            values.setflags(write=False)
        upper = float(solution.prob("upper"))
        lower = float(solution.prob("lower"))
        mean = float(solution.mean_rt()) if upper + lower > 0 else math.nan
        return ChoiceDistribution(
            arrays[0],
            arrays[1],
            arrays[2],
            dt,
            float(deadline),
            upper,
            lower,
            float(solution.prob_undecided()),
            mean if math.isfinite(mean) else None,
            str(native.__version__),
        )

    def sample_choices(
        self, distribution: ChoiceDistribution, count: int = 1
    ) -> tuple[ChoiceReaction, ...]:
        """Sample native grid masses using only this module's seeded Generator."""
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("count must be a non-negative integer")
        if distribution.time.ndim != 1:
            raise ValueError("distribution time must be a one dimensional array")
        points = len(distribution.time)
        if (
            points == 0
            or distribution.upper_density.shape != (points,)
            or distribution.lower_density.shape != (points,)
            or not np.isfinite(distribution.time).all()
            or not math.isfinite(distribution.dt)
            or distribution.dt <= 0
            or not math.isfinite(distribution.deadline)
            or distribution.deadline <= 0
            or np.any(distribution.time < 0)
            or np.any(distribution.time > distribution.deadline + 1e-12)
            or np.any(np.diff(distribution.time) < 0)
        ):
            raise ValueError("distribution needs aligned finite time and density data")
        probabilities = np.concatenate(
            (
                distribution.upper_density * distribution.dt,
                distribution.lower_density * distribution.dt,
                [distribution.timeout_probability],
            )
        )
        if (
            not np.isfinite(probabilities).all()
            or np.any(probabilities < -1e-12)
            or not math.isclose(float(probabilities.sum()), 1.0, abs_tol=1e-6)
        ):
            raise ValueError("solver distribution needs valid unit probability mass")
        probabilities = np.maximum(probabilities, 0.0)
        probabilities /= probabilities.sum()
        selected = self.rng.choice(len(probabilities), size=count, p=probabilities)
        reactions = []
        for index in selected:
            if index == 2 * points:
                reactions.append(ChoiceReaction(None, None))
            else:
                choice: Choice = "upper" if index < points else "lower"
                time = min(
                    distribution.deadline, float(distribution.time[index % points])
                )
                reactions.append(ChoiceReaction(choice, time))
        return tuple(reactions)

    def decide(self, drift: float, deadline: float) -> ChoiceReaction:
        """Solve and sample one explicit choice trial."""
        return self.sample_choices(self.solve_choice(drift, deadline))[0]
