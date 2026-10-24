"""Benchmarks: run a module or the whole human and compare with references.

Each benchmark takes a :class:`~embodied_human.core.Human`, exercises one
category (or several, for integrated tasks), and returns a :class:`Result`
whose metrics are scored against :mod:`references`. Swap one module in
:class:`~embodied_human.core.HumanConfig` and rerun to see what it changes.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..core import Human
from .references import REFERENCES


@dataclass
class Result:
    """Outcome of one benchmark.

    Attributes:
        name: Benchmark identifier.
        categories: Module categories it exercises.
        metrics: Measured values (reference names map to scored metrics).
        scores: Score in [0, 1] per scored metric.
        details: Further measurements for reports.
        skipped: Reason, if it could not run (missing optional packages).
    """

    name: str
    categories: tuple[str, ...]
    metrics: dict[str, float] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)
    skipped: str | None = None

    @property
    def passed(self) -> bool:
        """Whether every scored metric lies within its reference range."""
        return self.skipped is None and all(s == 1.0 for s in self.scores.values())

    @property
    def score(self) -> float:
        """Mean score, or NaN when skipped."""
        return float(np.mean(list(self.scores.values()))) if self.scores else math.nan

    def score_against(self, reference: str, value: float) -> None:
        """Record ``value`` under ``reference`` and score it."""
        self.metrics[reference] = float(value)
        self.scores[reference] = REFERENCES[reference].score(float(value))

    def as_dict(self) -> dict[str, Any]:
        """JSON-serializable result with the references used."""
        return {
            "name": self.name,
            "categories": list(self.categories),
            "passed": self.passed,
            "score": self.score,
            "skipped": self.skipped,
            "metrics": self.metrics,
            "scores": self.scores,
            "references": {
                k: {
                    "value": REFERENCES[k].value,
                    "range": [REFERENCES[k].low, REFERENCES[k].high],
                    "unit": REFERENCES[k].unit,
                    "citation": REFERENCES[k].citation,
                    "doi": REFERENCES[k].doi,
                }
                for k in self.scores
            },
            "details": self.details,
        }


Benchmark = Callable[[Human], Result]
BENCHMARKS: dict[str, Benchmark] = {}


def benchmark(name: str) -> Callable[[Benchmark], Benchmark]:
    """Decorator: register a benchmark by name."""

    def decorator(function: Benchmark) -> Benchmark:
        BENCHMARKS[name] = function
        return function

    return decorator


@benchmark("eyes.main_sequence")
def main_sequence(human: Human) -> Result:
    """Saccades of 1-40 degrees: fit duration against amplitude."""
    eyes = human.eyes
    result = Result("eyes.main_sequence", ("eyes",))
    amplitudes = np.array([1, 2, 5, 10, 15, 20, 30, 40], dtype=float)
    time = 0.0
    eyes.reset(eyes.rng, gaze=(0.0, 0.0))  # type: ignore[call-arg]
    executed: list[Any] = []
    for amplitude in np.repeat(amplitudes, 5):
        target = (float(amplitude) if len(executed) % 2 == 0 else 0.0, 0.0)
        eyes.command(target, time)  # type: ignore[attr-defined]
        while True:
            time += 0.001
            eyes.step(time)  # type: ignore[attr-defined]
            if not eyes.busy:  # type: ignore[attr-defined]
                break
        executed = eyes.saccades  # type: ignore[attr-defined]
        time += 0.3
    a = np.array([s.amplitude for s in executed])
    d = np.array([s.duration for s in executed]) * 1000.0
    slope, intercept = np.polyfit(a, d, 1)
    result.score_against("eyes.duration_slope", slope)
    result.score_against("eyes.duration_intercept", intercept)
    velocity = np.array([s.peak_velocity() for s in executed])
    result.details["peak_velocity_deg_s"] = {
        f"{amp:g}": float(np.mean(velocity[a.round() == amp])) for amp in amplitudes
    }
    # Report-only comparison with Gangemi et al. (1991): V = 578 (1 - exp(-A / 23.4)).
    result.details["gangemi_1991_peak_velocity_deg_s"] = {
        f"{amp:g}": float(578 * (1 - math.exp(-amp / 23.4))) for amp in amplitudes
    }
    return result


@benchmark("eyes.latency")
def saccade_latency(human: Human) -> Result:
    """Median latency of 400 visually guided saccades."""
    eyes = human.eyes
    result = Result("eyes.latency", ("eyes",))
    eyes.reset(eyes.rng, gaze=(0.0, 0.0))  # type: ignore[call-arg]
    time = 0.0
    for k in range(400):
        eyes.command((10.0 if k % 2 == 0 else 0.0, 0.0), time)  # type: ignore[attr-defined]
        while True:
            time += 0.001
            eyes.step(time)  # type: ignore[attr-defined]
            if not eyes.busy:  # type: ignore[attr-defined]
                break
        time += 0.5
    latencies = np.array([s.latency for s in eyes.saccades]) * 1000.0  # type: ignore[attr-defined]
    result.score_against("eyes.latency_median", float(np.median(latencies)))
    result.details["latency_percentiles_ms"] = {
        str(q): float(np.percentile(latencies, q)) for q in (5, 25, 50, 75, 95)
    }
    return result


@benchmark("objects.peripheral_acuity")
def peripheral_acuity(human: Human) -> Result:
    """Estimate E2 from observation noise measured at 0-20 degrees."""
    from ..vision.objects import Item

    objects: Any = human.objects
    result = Result("objects.peripheral_acuity", ("objects",))
    eccentricities = np.array([0.0, 2.5, 5.0, 10.0, 20.0])
    sds = []
    for e in eccentricities:
        item = {"x": Item((float(e), 0.0), 0.0, 1.0)}
        samples = [objects.observe(item, (0.0, 0.0))["x"].value for _ in range(4000)]
        sds.append(float(np.std(samples)))
    ratio = np.array(sds) / sds[0] - 1.0
    # ratio = e / E2: least squares through the origin.
    e2 = float(np.dot(eccentricities, eccentricities) / np.dot(eccentricities, ratio))
    result.score_against("objects.e2", e2)
    result.details["sd_by_eccentricity"] = dict(zip(map(str, eccentricities), sds))
    return result


@benchmark("vision.foveation")
def foveation_resolution(human: Human) -> Result:
    """Eccentricity at which the visible spatial frequency halves."""
    vision = human.vision
    result = Result("vision.foveation", ("vision",))
    params: Any = vision.parameters
    eccentricities = np.linspace(0.0, 30.0, 3001)
    cutoff = params.cutoff(eccentricities)
    half = float(np.interp(-cutoff[0] / 2, -cutoff, eccentricities))
    result.score_against("vision.e2", half)
    result.details["cutoff_cycles_per_deg"] = {
        f"{e:g}": float(params.cutoff(e)) for e in (0, 2, 5, 10, 20, 40)
    }
    return result


@benchmark("memory.span")
def memory_span(human: Human) -> Result:
    """Immediate serial recall at one item per second; span is the longest
    list recalled perfectly in at least half of 200 trials."""
    memory = human.memory
    result = Result("memory.span", ("memory",))
    rates = {}
    for length in range(1, 10):
        perfect = 0
        for _ in range(200):
            memory.reset(memory.rng)
            for i in range(length):
                memory.encode(f"item{i}", i, float(i))  # type: ignore[attr-defined]
            time = float(length)
            ok = True
            for i in range(length):
                retrieval = memory.retrieve(f"item{i}", time)  # type: ignore[attr-defined]
                time += max(retrieval.latency, 0.3)
                ok = ok and retrieval.success and retrieval.value == i
            perfect += ok
        rates[length] = perfect / 200
    span = max([n for n, p in rates.items() if p >= 0.5], default=0)
    result.score_against("memory.span", span)
    result.details["perfect_recall_rate"] = {str(k): v for k, v in rates.items()}
    return result


@benchmark("motor.touch_tapping")
def touch_tapping(human: Human) -> Result:
    """ISO 9241-411 tapping with a physical fingertip on an embodied-ui
    touch surface: throughput and Fitts slope over five conditions."""
    result = Result("motor.touch_tapping", ("motor", "eyes"))
    try:
        import embodied_ui  # noqa: F401
        import mujoco  # noqa: F401
    except ImportError as error:
        result.skipped = f"needs embodied-ui and mujoco: {error}"
        return result
    from ..agents.pointer import run_tapping, tapping_summary
    from ..tasks.tapping import TappingCondition

    conditions = [(0.04, 0.01), (0.08, 0.01), (0.08, 0.005), (0.16, 0.01), (0.16, 0.02)]
    runs = [run_tapping(human, TappingCondition(a, w), rounds=2) for a, w in conditions]
    summary = tapping_summary(runs)
    result.score_against("motor.touch_throughput", summary["mean_throughput"])
    result.score_against(
        "motor.touch_slope", 1000.0 * summary["regression_slope_s_per_bit"]
    )
    result.details = summary
    return result


def run(human: Human, names: list[str] | None = None) -> list[Result]:
    """Run the named benchmarks (all by default), in registration order."""
    from . import integrated  # noqa: F401  (registers integrated benchmarks)

    chosen = names or list(BENCHMARKS)
    unknown = set(chosen) - set(BENCHMARKS)
    if unknown:
        raise ValueError(
            f"unknown benchmarks {sorted(unknown)}; known: {sorted(BENCHMARKS)}"
        )
    return [BENCHMARKS[name](human) for name in chosen]
