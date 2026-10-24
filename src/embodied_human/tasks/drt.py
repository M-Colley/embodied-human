"""Detection Response Task (DRT) per ISO 17488:2016.

Stimuli appear every 3-5 s (uniform), stay on for at most 1 s and switch
off at the response. A response 100-2500 ms after onset is a hit; none in
that window is a miss; one within 100 ms is premature; a second response in
the window is repeated. The standard prescribes no baseline response time.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DRTConfig:
    """ISO 17488 timing (clauses 3 and 6.3.1)."""

    interval: tuple[float, float] = (3.0, 5.0)
    duration: float = 1.0
    window: tuple[float, float] = (0.1, 2.5)


@dataclass
class DRTResult:
    """Classified responses."""

    onsets: list[float] = field(default_factory=list)
    reaction_times: list[float] = field(default_factory=list)
    misses: int = 0
    premature: int = 0
    repeated: int = 0

    def summary(self) -> dict[str, float]:
        """Hit rate and mean reaction time of hits."""
        stimuli = len(self.onsets)
        hits = len(self.reaction_times)
        return {
            "stimuli": float(stimuli),
            "hit_rate": hits / stimuli if stimuli else float("nan"),
            "mean_rt_s": float(np.mean(self.reaction_times)) if hits else float("nan"),
            "premature": float(self.premature),
            "repeated": float(self.repeated),
        }


class DRTTask:
    """DRT schedule and scoring, usable as an embodied-ui engine update.

    The stimulus is the output ``<lamp>.on`` (an embodied-ui Indicator) and
    responses are ``press`` events of the ``button`` device.
    """

    def __init__(
        self,
        config: DRTConfig = DRTConfig(),
        lamp: str = "drt_lamp",
        button: str = "drt",
        rng: np.random.Generator | None = None,
    ) -> None:
        self.config = config
        self.lamp, self.button = lamp, button
        self.rng = rng or np.random.default_rng()
        self.reset()

    def reset(self, time: float = 0.0) -> None:
        """Schedule the first stimulus."""
        self.result = DRTResult()
        self._next = time + self.rng.uniform(*self.config.interval)
        self._onset: float | None = None
        self._responded = False
        self.on = False

    def step(self, time: float) -> bool:
        """Advance the schedule; return whether the stimulus is on."""
        low, high = self.config.window
        if self._onset is not None and time - self._onset > high:
            if not self._responded:
                self.result.misses += 1
            self._onset = None
        if (
            self.on
            and self._onset is not None
            and time - self._onset >= self.config.duration
        ):
            self.on = False
        if self._onset is None and time >= self._next:
            self._onset = self._next
            self.result.onsets.append(self._onset)
            self._responded = False
            self.on = True
            self._next = self._onset + self.rng.uniform(*self.config.interval)
        return self.on

    def respond(self, time: float) -> str:
        """Classify a response at ``time``."""
        low, high = self.config.window
        if self._onset is None or time - self._onset > high:
            return "false_alarm"
        latency = time - self._onset
        if latency < low:
            self.result.premature += 1
            return "premature"
        if self._responded:
            self.result.repeated += 1
            return "repeated"
        self._responded = True
        self.on = False
        self.result.reaction_times.append(latency)
        return "hit"

    @property
    def onset(self) -> float | None:
        """Onset of the current stimulus window, if any."""
        return self._onset

    def update(self, command: Any, time: float) -> Mapping[str, Any]:
        """Engine update: score presses, return the lamp state."""
        for event in command.events:
            if event.source == self.button and event.kind == "press":
                self.respond(event.time)
        on = self.step(time)
        return {f"{self.lamp}.on": float(on)}
