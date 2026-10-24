"""Where to look: gaze allocation between the road and other areas.

Baselines:

- :class:`UncertaintySampler` looks back at the road when the believed
  lateral uncertainty exceeds a threshold, and away again once it has
  fallen. This is the uncertainty account of Senders et al. (1967).
  Jokinen, Kujala and Oulasvirta (2021,
  https://doi.org/10.1177/0018720820927687) replace the threshold by a
  supervisor policy learned with hierarchical reinforcement learning over
  the same kind of belief; such a policy can replace this module.
- :class:`SEEV` allocates dwell probability by salience, effort,
  expectancy and value (Wickens et al., 2003).
- :class:`AttenD` is the 2-s attention buffer of Kircher and Ahlström
  (2009), used as a distraction metric.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from .core import Module, Parameters, Source, register

ROAD = "road"


@dataclass(frozen=True)
class SamplerParameters(Parameters):
    """Uncertainty-threshold sampling.

    Args:
        return_sd: Median believed lateral SD (m) that sends gaze back to the
            road.
        threshold_spread: SD of the log threshold, drawn anew for every
            off-road glance; people do not return at a fixed uncertainty,
            and glance durations vary within a person.
        leave_sd: Lateral SD (m) below which gaze may leave the road again.
        minimum_road_dwell: Shortest road glance, s.
        maximum_glance: Longest off-road glance, s; a safety bound above
            the longest glances humans make in these tasks.
    """

    return_sd: float = 0.11
    threshold_spread: float = 1.0
    leave_sd: float = 0.15
    minimum_road_dwell: float = 0.3
    maximum_glance: float = 4.0

    SOURCES: ClassVar = {
        "return_sd": Source(
            "Kujala & Saariluoma (2011), Table 2; fitted here by ABC",
            "https://doi.org/10.1080/00140139.2011.592601",
            "fitted",
            "posterior median, 90% interval 0.06-0.18 m, from the mean glance "
            "and within-person SD at 50 km/h (evidence/glance-calibration.json)",
        ),
        "threshold_spread": Source(
            "Kujala & Saariluoma (2011), Table 2; fitted here by ABC",
            "https://doi.org/10.1080/00140139.2011.592601",
            "fitted",
            "posterior median, 90% interval 0.64-1.34",
        ),
        "leave_sd": Source("model choice", None, "assumed"),
        "minimum_road_dwell": Source("model choice", None, "assumed"),
        "maximum_glance": Source(
            "model choice",
            "https://doi.org/10.1080/00140139.2011.592601",
            "assumed",
            "a bound above the longest in-vehicle glances measured in simulator "
            "search tasks, 2.42-3.70 s (Kujala & Saariluoma, 2011, Table 2)",
        ),
    }


class Attention(Module):
    """Category ``attention``: choose the area of interest to look at."""

    category = "attention"

    def select(
        self,
        time: float,
        current: str,
        demands: Mapping[str, float],
        uncertainty: float,
    ) -> str:
        """Return the area to look at next.

        Args:
            time: Current time, s.
            current: Area gaze is on (or travelling to).
            demands: Areas other than the road that want attention, with a
                priority; empty when only driving.
            uncertainty: Believed lateral SD of the vehicle, m.
        """
        raise NotImplementedError


@register("attention", "uncertainty_sampler")
class UncertaintySampler(Attention):
    """Return to the road on high uncertainty; leave it once it is low.

    ``reason`` names the mechanism behind the last switch, so predictions
    can be explained: ``uncertainty`` (the belief became too uncertain),
    ``maximum_glance``, ``task_done`` (no demand left) or ``certain``
    (the road belief was certain enough to look away).
    """

    def __init__(self, **params: float) -> None:
        super().__init__()
        self.params = SamplerParameters(**params)
        self.reset()

    def reset(self, rng: np.random.Generator | None = None) -> None:
        super().reset(rng)
        self._since = 0.0
        self._current = ROAD
        self.threshold = self.params.return_sd
        self.reason = ""

    def select(
        self,
        time: float,
        current: str,
        demands: Mapping[str, float],
        uncertainty: float,
    ) -> str:
        p = self.params
        if current != self._current:
            self._current = current
            self._since = time
        dwell = time - self._since
        if current != ROAD:
            if uncertainty >= self.threshold:
                self.reason = "uncertainty"
            elif dwell >= p.maximum_glance:
                self.reason = "maximum_glance"
            elif current not in demands:
                self.reason = "task_done"
            else:
                return current
            return ROAD
        if demands and uncertainty <= p.leave_sd and dwell >= p.minimum_road_dwell:
            self.reason = "certain"
            # Each glance away has its own tolerance for uncertainty.
            spread = p.threshold_spread * float(self.rng.standard_normal())
            self.threshold = p.return_sd * float(np.exp(spread))
            return max(demands, key=lambda name: demands[name])
        return ROAD


@dataclass(frozen=True)
class SEEVParameters(Parameters):
    """Weights of the SEEV model."""

    salience: float = 0.0
    effort: float = 1.0
    expectancy: float = 1.0
    value: float = 1.0
    multiplicative: bool = True

    SOURCES: ClassVar = {
        name: Source(
            "Wickens et al. (2003)",
            "https://doi.org/10.1518/hfes.45.3.360.27250",
            "fitted",
            "weights are fitted to dwell data; salience 0 gives the top-down model",
        )
        for name in ("salience", "effort", "expectancy", "value", "multiplicative")
    }


@register("attention", "seev")
class SEEV(Attention):
    """Sample the next area with probability from the SEEV score.

    ``areas`` maps each area to its (salience, effort, expectancy, value).
    The score is ``s S - ef EF + ex EX * v V`` (multiplicative form) or with
    ``+ ex EX + v V``; scores are made positive by a softmax.
    """

    def __init__(
        self,
        areas: Mapping[str, tuple[float, float, float, float]] | None = None,
        **params: float,
    ) -> None:
        super().__init__()
        self.params = SEEVParameters(**params)  # type: ignore[arg-type]
        self.areas = dict(areas or {ROAD: (0.0, 0.0, 1.0, 1.0)})

    def probabilities(self) -> dict[str, float]:
        """Return dwell probabilities per area."""
        p = self.params
        scores = {}
        for name, (s, ef, ex, v) in self.areas.items():
            top_down = (
                (p.expectancy * ex) * (p.value * v)
                if p.multiplicative
                else p.expectancy * ex + p.value * v
            )
            scores[name] = p.salience * s - p.effort * ef + top_down
        values = np.array(list(scores.values()))
        weights = np.exp(values - values.max())
        weights /= weights.sum()
        return dict(zip(scores, weights.tolist()))

    def select(
        self,
        time: float,
        current: str,
        demands: Mapping[str, float],
        uncertainty: float,
    ) -> str:
        probabilities = self.probabilities()
        names = list(probabilities)
        return str(self.rng.choice(names, p=list(probabilities.values())))


@dataclass
class AttenD:
    """The AttenD 2-s attention buffer (Kircher and Ahlström, 2009, Table 1).

    The buffer drains at 1 s/s while gaze is outside the field relevant for
    driving (mirrors and the speedometer only after 1 s), refills at 1 s/s
    after a 0.1 s delay once gaze returns, and saturates at 2 s. A buffer
    at zero classifies the driver as distracted.
    """

    maximum: float = 2.0
    refill_delay: float = 0.1
    mirror_grace: float = 1.0
    buffer: float = 2.0
    _area: str = ROAD
    _dwell: float = 0.0
    distracted_time: float = field(default=0.0)
    minimum: float = 2.0

    def step(self, dt: float, area: str) -> float:
        """Advance with gaze on ``area`` (``road``, ``mirror``, ``speedometer``
        or anything else) and return the buffer."""
        if area != self._area:
            self._area, self._dwell = area, 0.0
        self._dwell += dt
        if area == ROAD:
            if self._dwell > self.refill_delay:
                self.buffer = min(self.maximum, self.buffer + dt)
        else:
            # Mirror and speedometer glances drain only after the grace time.
            grace = self.mirror_grace if area in ("mirror", "speedometer") else 0.0
            if self._dwell > grace:
                self.buffer = max(0.0, self.buffer - dt)
        if self.buffer <= 0.0:
            self.distracted_time += dt
        self.minimum = min(self.minimum, self.buffer)
        return self.buffer
