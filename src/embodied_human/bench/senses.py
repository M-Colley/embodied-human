"""Benchmarks of hearing, touch, multisensory responses and motion sickness.

- ``response.modality_rt``: simple reaction times to a tone, a light and a
  vibration, in the order Diederich and Colonius (2004) measured them.
- ``response.drt``: detection-response task reaction times while driving
  (visual: Conti, Dlugosch and Bengler, 2014; tactile: Cooper et al.,
  2020) and how much older drivers are slower.
- ``vestibular.vertical_sickness``: motion sickness after 2 hours of
  vertical oscillation (O'Hanlon and McCauley, 1974).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..core import Human, build_human
from ..population import OLDER_ADULTS, Population
from ..response import Signal
from .benchmarks import Result, benchmark

#: Sensation level assumed for the strong vibration of Diederich and
#: Colonius (2004) and for DRT vibrators, dB above threshold.
STRONG_VIBRATION_DB = 30.0
DRT_VIBRATION_DB = 20.0
#: Eccentricity of a head-mounted DRT light, deg (assumed: ISO 17488
#: places it in the left periphery).
DRT_LIGHT_DEG = 17.0


def _mean_rt(human: Human, signals: list[Signal], samples: int = 4000) -> float:
    return 1000.0 * float(
        np.mean([human.response.respond(signals).time for _ in range(samples)])  # type: ignore[attr-defined]
    )


@benchmark("response.modality_rt")
def modality_rt(human: Human) -> Result:
    """Simple RT to a 90 dB tone, a foveal light and a strong vibration."""
    result = Result("response.modality_rt", ("hearing", "touch", "response"))
    hearing: Any = human.hearing
    tone = hearing.sensation_level(90.0, 1000.0)
    times = {
        "auditory": _mean_rt(human, [Signal("auditory", 0.0, hearing.latency(tone))]),
        "visual": _mean_rt(
            human,
            [Signal("visual", 0.0, human.response.visual_latency(0.0))],  # type: ignore[attr-defined]
        ),
        "tactile": _mean_rt(
            human,
            [Signal("tactile", 0.0, human.touch.latency(STRONG_VIBRATION_DB))],  # type: ignore[attr-defined]
        ),
    }
    human_order = {"auditory": 132.0, "visual": 163.0, "tactile": 177.0}
    pairs = [("auditory", "visual"), ("visual", "tactile"), ("auditory", "tactile")]
    agree = [
        (times[a] < times[b]) == (human_order[a] < human_order[b]) for a, b in pairs
    ]
    result.score_against("response.modality_order", float(np.mean(agree)))
    result.details = {
        "model_ms": times,
        "human_ms": human_order,
        "redundant_av_ms": _mean_rt(
            human,
            [
                Signal("auditory", 0.0, human.hearing.latency(tone)),  # type: ignore[attr-defined]
                Signal("visual", 0.0, human.response.visual_latency(0.0)),  # type: ignore[attr-defined]
            ],
        ),
        "note": "the model's parameters come from other participants "
        "(Diederich & Colonius, 2015); only the order is compared",
    }
    return result


def _drt(human: Human) -> tuple[float, float]:
    visual = [Signal("visual", 0.0, human.response.visual_latency(DRT_LIGHT_DEG))]  # type: ignore[attr-defined]
    tactile = [Signal("tactile", 0.0, human.touch.latency(DRT_VIBRATION_DB))]  # type: ignore[attr-defined]
    return _mean_rt(human, visual), _mean_rt(human, tactile)


@benchmark("response.drt")
def drt(human: Human) -> Result:
    """Detection-response task while driving, younger and older drivers.

    The model has no effect of driving itself on the central stage, so it
    predicts the same RT with and without driving; Conti et al. measured
    279-312 ms without driving and 326-362 ms while driving.
    """
    result = Result("response.drt", ("response", "touch"))
    visual, tactile = _drt(human)
    older = build_human(
        Population(human.config, profile=OLDER_ADULTS).typical(), seed=0
    )
    _, tactile_older = _drt(older)
    result.score_against("response.drt_visual", visual)
    result.score_against("response.drt_tactile", tactile)
    result.score_against("response.older_slowing", tactile_older / tactile)
    result.details = {
        "visual_ms": visual,
        "tactile_ms": tactile,
        "tactile_older_ms": tactile_older,
        "light_eccentricity_deg": DRT_LIGHT_DEG,
        "vibration_sensation_db": DRT_VIBRATION_DB,
    }
    return result


#: O'Hanlon & McCauley (1973), Tables 1-2: frequency (Hz), mean absolute
#: acceleration per half-wave (g), % who vomited within 2 hours.
VERTICAL_SICKNESS = (
    (0.083, 0.025, 0),
    (0.083, 0.05, 5),
    (0.167, 0.025, 0),
    (0.167, 0.05, 10),
    (0.167, 0.10, 30),
    (0.167, 0.20, 60),
    (0.333, 0.05, 5),
    (0.333, 0.10, 15),
    (0.333, 0.20, 52),
    (0.333, 0.30, 52),
    (0.5, 0.10, 0),
    (0.5, 0.20, 15),
    (0.5, 0.30, 25),
    (0.5, 0.40, 30),
)


@benchmark("vestibular.vertical_sickness")
def vertical_sickness(human: Human, dt: float = 0.1) -> Result:
    """Two hours of vertical sinusoidal motion at 14 frequency-acceleration
    combinations; MSI after 2 h against the percentage who vomited."""
    result = Result("vestibular.vertical_sickness", ("vestibular",))
    time = np.arange(0.0, 7200.0, dt)
    motion = np.zeros((len(VERTICAL_SICKNESS), len(time), 3))
    for i, (frequency, mean_g, _) in enumerate(VERTICAL_SICKNESS):
        # Mean absolute acceleration is 0.637 of the peak.
        peak = mean_g / 0.637 * 9.81
        motion[i, :, 2] = peak * np.sin(2 * math.pi * frequency * time)
    predicted = human.vestibular.simulate(dt, motion)[:, -1]  # type: ignore[attr-defined]
    observed = np.array([row[2] for row in VERTICAL_SICKNESS], dtype=float)
    error = float(np.sqrt(np.mean((predicted - observed) ** 2)))
    result.score_against("vestibular.vertical_msi_error", error)
    result.details = {
        "conditions": [
            {
                "frequency_hz": f,
                "mean_accel_g": a,
                "observed_pct": o,
                "predicted_pct": float(p),
            }
            for (f, a, o), p in zip(VERTICAL_SICKNESS, predicted)
        ],
        "correlation": float(np.corrcoef(predicted, observed)[0, 1]),
    }
    return result
