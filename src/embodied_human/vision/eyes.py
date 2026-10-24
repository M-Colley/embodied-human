"""Eye movements: saccade latency, the main sequence and saccadic suppression.

Gaze is expressed as ``(azimuth, elevation)`` in degrees in the head frame:
azimuth positive to the right, elevation positive upwards.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from ..core import Module, Parameters, Source, register

Gaze = tuple[float, float]


@dataclass(frozen=True)
class Saccade:
    """One executed saccade.

    Attributes:
        command: Time the gaze shift was requested, s.
        onset: Movement onset, s; ``onset - command`` is the latency.
        duration: Movement duration, s.
        origin, target: Gaze before and after, degrees.
    """

    command: float
    onset: float
    duration: float
    origin: Gaze
    target: Gaze

    @property
    def amplitude(self) -> float:
        """Angular amplitude, degrees."""
        return math.hypot(
            self.target[0] - self.origin[0], self.target[1] - self.origin[1]
        )

    @property
    def latency(self) -> float:
        """Programming latency, s."""
        return self.onset - self.command

    @property
    def end(self) -> float:
        """Landing time, s."""
        return self.onset + self.duration

    def peak_velocity(self) -> float:
        """Peak angular velocity of the minimum-jerk profile, deg/s."""
        return 1.875 * self.amplitude / self.duration if self.duration > 0 else 0.0

    def position(self, time: float) -> Gaze:
        """Gaze at ``time`` along the minimum-jerk path."""
        tau = (
            min(1.0, max(0.0, (time - self.onset) / self.duration))
            if self.duration > 0
            else 1.0
        )
        s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
        return (
            self.origin[0] + s * (self.target[0] - self.origin[0]),
            self.origin[1] + s * (self.target[1] - self.origin[1]),
        )


@dataclass(frozen=True)
class EyeState:
    """Gaze at one instant; vision is suppressed during saccades."""

    time: float
    gaze: Gaze
    in_saccade: bool
    fixation_onset: float


@dataclass(frozen=True)
class EyeParameters(Parameters):
    """Main-sequence eyes.

    Args:
        duration_slope: Saccade duration per degree of amplitude, s/deg.
        duration_intercept: Duration of a vanishing saccade, s.
        latency_median: Median latency from command to onset, s.
        latency_spread: Log-normal shape (SD of log latency).
        minimum_amplitude: Gaze shifts smaller than this, degrees, are
            made without a saccade (within fixational eye movements).
    """

    duration_slope: float = 0.0022
    duration_intercept: float = 0.021
    latency_median: float = 0.200
    latency_spread: float = 0.25
    minimum_amplitude: float = 0.5

    SOURCES: ClassVar = {
        "duration_slope": Source(
            "Carpenter (1988), Movements of the Eyes, 2nd ed.",
            None,
            "measured",
            "D = 2.2 A + 21 ms; Baloh et al. (1975) report a slope of 2.7 ms/deg",
        ),
        "duration_intercept": Source("Carpenter (1988)", None, "measured"),
        "latency_median": Source(
            "Saslow (1967)",
            "https://doi.org/10.1364/JOSA.57.001024",
            "measured",
            "step target with the fixation point extinguished; "
            "150 ms (gap) to 250 ms (overlap)",
        ),
        "latency_spread": Source(
            "model choice", None, "assumed", "right-skewed latency; fit to data"
        ),
        "minimum_amplitude": Source("model choice", None, "assumed"),
    }


class Eyes(Module):
    """Category ``eyes``: turn gaze commands into a gaze trajectory."""

    category = "eyes"

    def command(self, target: Gaze, time: float) -> None:
        """Request a gaze shift to ``target`` at ``time``."""
        raise NotImplementedError

    def step(self, time: float) -> EyeState:
        """Advance to ``time`` and return the eye state."""
        raise NotImplementedError

    @property
    def saccades(self) -> list[Saccade]:
        """Every saccade executed since the last reset."""
        raise NotImplementedError


@register("eyes", "main_sequence")
class MainSequenceEyes(Eyes):
    """Saccades with log-normal latency, linear main-sequence duration and a
    minimum-jerk velocity profile (peak velocity 1.875 times the mean).

    One saccade is programmed at a time; a new command while one is
    programmed replaces it, and a command during a saccade starts after
    landing. Vision is suppressed during the movement.
    """

    def __init__(self, **params: float) -> None:
        super().__init__()
        self.params = EyeParameters(**params)
        self._start: Gaze = (0.0, 0.0)
        self.reset()

    def reset(
        self, rng: np.random.Generator | None = None, gaze: Gaze = (0.0, 0.0)
    ) -> None:
        super().reset(rng)
        self.gaze: Gaze = (float(gaze[0]), float(gaze[1]))
        self._pending: Saccade | None = None
        self._flight: Saccade | None = None
        self._history: list[Saccade] = []
        self._fixation_onset = 0.0
        self._time = 0.0

    def duration(self, amplitude: float) -> float:
        """Main-sequence duration of a saccade, s."""
        p = self.params
        return p.duration_intercept + p.duration_slope * amplitude

    def latency(self) -> float:
        """Sample one programming latency, s."""
        p = self.params
        return float(
            p.latency_median * math.exp(p.latency_spread * self.rng.standard_normal())
        )

    def command(self, target: Gaze, time: float) -> None:
        target = (float(target[0]), float(target[1]))
        origin = self._flight.target if self._flight is not None else self.gaze
        amplitude = math.hypot(target[0] - origin[0], target[1] - origin[1])
        if amplitude < self.params.minimum_amplitude:
            self._pending = None
            return
        earliest = self._flight.end if self._flight is not None else time
        onset = max(time + self.latency(), earliest)
        self._pending = Saccade(time, onset, self.duration(amplitude), origin, target)

    def step(self, time: float) -> EyeState:
        if time < self._time:
            raise ValueError("eye time must not go backwards")
        self._time = time
        if self._flight is not None and time >= self._flight.end:
            self.gaze = self._flight.target
            self._fixation_onset = self._flight.end
            self._flight = None
        if (
            self._flight is None
            and self._pending is not None
            and time >= self._pending.onset
        ):
            pending = self._pending
            # Re-anchor at the current gaze in case the command predates landing.
            pending = Saccade(
                pending.command,
                pending.onset,
                pending.duration,
                self.gaze,
                pending.target,
            )
            self._pending = None
            self._flight = pending
            self._history.append(pending)
            if time >= pending.end:
                self.gaze = pending.target
                self._fixation_onset = pending.end
                self._flight = None
        if self._flight is not None:
            return EyeState(
                time, self._flight.position(time), True, self._fixation_onset
            )
        return EyeState(time, self.gaze, False, self._fixation_onset)

    @property
    def saccades(self) -> list[Saccade]:
        return list(self._history)

    @property
    def busy(self) -> bool:
        """Whether a saccade is programmed or in flight."""
        return self._pending is not None or self._flight is not None


def angular_distance(a: Gaze, b: Gaze) -> float:
    """Angle between two gaze directions in degrees (small-angle plane)."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


@dataclass
class Fixations:
    """Collect fixations from a stream of eye states, for analysis."""

    records: list[tuple[float, float, Gaze]] = field(default_factory=list)
    _onset: float | None = None
    _gaze: Gaze | None = None

    def add(self, state: EyeState) -> None:
        """Feed one state."""
        if state.in_saccade:
            self.close(state.time)
        elif self._onset is None:
            self._onset, self._gaze = state.time, state.gaze

    def close(self, time: float) -> None:
        """End the current fixation."""
        if self._onset is not None and self._gaze is not None:
            self.records.append((self._onset, time, self._gaze))
        self._onset = None
        self._gaze = None
