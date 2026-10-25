"""Responding to signals: reaction times to sights, sounds and touches.

Baseline :class:`TwinResponse` is the time-window-of-integration (TWIN)
model of Colonius and Diederich for redundant signals (Colonius and
Diederich, 2012; Diederich and Colonius, 2015). Each signal is processed
peripherally in its own channel, an exponential time with a mean that
depends on the signal (the hearing, touch and visual latencies). The
first channel to finish starts the response; if another finishes within
the window ``omega`` of it, the senses integrate and the central stage is
shorter by ``Delta``:

    RT = min_i P_i + mu - Delta * 1{ another P_j within omega of it }.

Visual latency grows with the signal's eccentricity from the line of
gaze, following Osaka (1976). A signal outside the visual field is not
seen until gaze moves to it.

The central stage is a single bottleneck shared with concurrent tasks, as
in threaded cognition (Salvucci and Taatgen, 2008): driving, a memory task
or a visual search each occupy it for a while at their own pace
(:class:`CentralTask`). A response that becomes ready while the stage is
busy waits until it is free, which is how secondary tasks lengthen
detection-response times.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar, Literal

import numpy as np

from .core import Module, Parameters, Source, register

Modality = Literal["visual", "auditory", "tactile"]

_TWIN_2015 = (
    "Diederich & Colonius (2015), Table 1",
    "https://doi.org/10.1037/a0038696",
)

#: Simple visual RT by retinal eccentricity (deg), mean of nasal and
#: temporal retina over target sizes (Osaka, 1976, Table 4), ms.
OSAKA_ECCENTRICITY = (
    (0, 224.0),
    (10, 246.5),
    (20, 257.5),
    (30, 268.5),
    (40, 277.0),
    (50, 286.5),
)


@dataclass(frozen=True)
class Signal:
    """One signal reaching a sense.

    Args:
        modality: ``visual``, ``auditory`` or ``tactile``.
        onset: Time the signal starts, s.
        latency: Mean peripheral processing time, s (``inf``: not sensed).
    """

    modality: Modality
    onset: float
    latency: float


@dataclass(frozen=True)
class Reaction:
    """A response and the mechanism behind it."""

    time: float
    first: Modality | None
    integrated: bool
    waited: float = 0.0


@dataclass(frozen=True)
class CentralTask:
    """A concurrent task's demand on the central stage.

    Every ``period`` seconds (at a random phase) the task occupies the
    stage for ``duration`` seconds; ``period`` 0 means it never does.
    """

    name: str
    period: float
    duration: float

    def __post_init__(self) -> None:
        if self.period < 0 or self.duration < 0:
            raise ValueError("period and duration must be non-negative")
        if self.period and self.duration > self.period:
            raise ValueError(f"{self.name}: busy longer than its period")


def central_wait(tasks: Sequence[CentralTask], rng: np.random.Generator) -> float:
    """Time a response arriving at a random moment waits for the shared
    stage: the busy periods of all tasks, chained if they overlap."""
    intervals = []
    for task in tasks:
        if task.period <= 0 or task.duration <= 0:
            continue
        # Busy windows around the arrival at time 0, at a random phase.
        start = -float(rng.uniform(0.0, task.period))
        while start < 10.0 * task.period:
            if start + task.duration > 0:
                intervals.append((start, start + task.duration))
            start += task.period
    free = 0.0
    changed = True
    while changed:
        changed = False
        for begin, end in intervals:
            if begin <= free < end:
                free, changed = end, True
    return free


@dataclass(frozen=True)
class TwinParameters(Parameters):
    """TWIN model for manual responses.

    Args:
        central: mu, mean central (second-stage) time without integration, s.
        facilitation: Delta, reduction of the central time with
            integration, s.
        window: omega, width of the integration window, s.
        visual_latency: 1/lambda_V of a foveal light, s.
        visual_field: Largest eccentricity at which a light is noticed, deg.
    """

    central: float = 0.2171
    facilitation: float = 0.0204
    window: float = 0.3588
    visual_latency: float = 0.1026
    visual_field: float = 90.0

    SOURCES: ClassVar = {
        "central": Source(*_TWIN_2015, "fitted", "median over 11 participants"),
        "facilitation": Source(*_TWIN_2015, "fitted", "median over 11 participants"),
        "window": Source(*_TWIN_2015, "fitted", "median omega_RT"),
        "visual_latency": Source(*_TWIN_2015, "fitted", "median 1/lambda_V"),
        "visual_field": Source(
            "model choice",
            None,
            "assumed",
            "the temporal visual field extends beyond 90 deg; peripheral "
            "warning lights are rarely noticed so far out",
        ),
    }


class Response(Module):
    """Category ``response``: turn sensed signals into responses."""

    category = "response"

    def visual_latency(self, eccentricity: float) -> float:
        """Mean peripheral time of a light at an eccentricity, s."""
        raise NotImplementedError

    def respond(
        self, signals: Sequence[Signal], load: Sequence[CentralTask] = ()
    ) -> Reaction:
        """Sample a response to signals that call for the same action,
        while concurrent tasks share the central stage."""
        raise NotImplementedError


@register("response", "twin")
class TwinResponse(Response):
    """Time-window-of-integration model, redundant signals paradigm."""

    def __init__(self, **params: float) -> None:
        super().__init__()
        self.params = TwinParameters(**params)

    def visual_latency(self, eccentricity: float) -> float:
        p = self.params
        e = abs(eccentricity)
        if e > p.visual_field:
            return math.inf
        degrees, times = zip(*OSAKA_ECCENTRICITY)
        # Osaka's increase over the fovea, extrapolated linearly past 50 deg.
        if e <= degrees[-1]:
            extra = float(np.interp(e, degrees, times)) - times[0]
        else:
            slope = (times[-1] - times[-2]) / (degrees[-1] - degrees[-2])
            extra = times[-1] - times[0] + slope * (e - degrees[-1])
        return p.visual_latency + extra / 1000.0

    def respond(
        self, signals: Sequence[Signal], load: Sequence[CentralTask] = ()
    ) -> Reaction:
        p = self.params
        finished = sorted(
            (s.onset + self.rng.exponential(s.latency), s.modality)
            for s in signals
            if math.isfinite(s.latency) and s.latency > 0
        )
        if not finished:
            return Reaction(math.inf, None, False)
        (first, modality), rest = finished[0], finished[1:]
        integrated = any(time - first < p.window for time, _ in rest)
        central = p.central - (p.facilitation if integrated else 0.0)
        waited = central_wait(load, self.rng) if load else 0.0
        return Reaction(first + waited + central, modality, integrated, waited)

    def mean_reaction(
        self,
        signals: Sequence[Signal],
        samples: int = 20000,
        load: Sequence[CentralTask] = (),
    ) -> float:
        """Mean response time by simulation, s."""
        times = [self.respond(signals, load).time for _ in range(samples)]
        return float(np.mean(times))
