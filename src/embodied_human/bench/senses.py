"""Benchmarks of hearing, touch, multisensory responses and motion sickness.

- ``response.modality_rt``: simple reaction times to a tone, a light and a
  vibration, in the order Diederich and Colonius (2004) measured them.
- ``response.eccentricity``: how much later a peripheral light is
  answered, against saccades (Diederich and Colonius, 2007).
- ``response.drt``: detection-response task reaction times while driving
  (a head-mounted light: Conti, Dlugosch and Bengler, 2014; a vibration
  and a windshield light, pooled: Cooper et al., 2020) and how much older
  drivers are slower.
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
from ..tasks.load import DRIVING
from .benchmarks import Result, benchmark

#: Sensation level assumed for the strong vibration of Diederich and
#: Colonius (2004), dB above threshold.
STRONG_VIBRATION_DB = 30.0


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


#: Diederich & Colonius (2007), Table 8: observed mean saccadic RT (ms) to a
#: light alone by eccentricity (deg), 3 participants.
SACCADE_ECCENTRICITY = {10.0: 150.0, 50.0: 189.0, 70.0: 208.0}
#: Their TWIN estimates of tactile peripheral time on the palm (Table 3), ms,
#: with the light at 10 deg at 16 ms.
PALM_VIBRATION = {"50 Hz, 0.6 V": 29.0, "5 Hz, 2.4 V": 111.0}


@benchmark("response.eccentricity")
def eccentricity(human: Human) -> Result:
    """How much later a peripheral light is responded to.

    The model's visual latency grows with eccentricity as in Osaka's (1976)
    manual reaction times up to 50 deg and is extrapolated beyond. In TWIN
    the peripheral stage does not depend on the response, so the increase
    from 10 deg should carry over to Diederich and Colonius's (2007) eye
    movements; 70 deg tests the extrapolation.
    """
    result = Result("response.eccentricity", ("response",))
    response: Any = human.response
    base = response.visual_latency(10.0)
    model = {e: 1000.0 * (response.visual_latency(e) - base) for e in (50.0, 70.0)}
    observed = {e: SACCADE_ECCENTRICITY[e] - SACCADE_ECCENTRICITY[10.0] for e in model}
    errors = [model[e] - observed[e] for e in model]
    result.score_against(
        "response.eccentricity_rmse", math.sqrt(float(np.mean(np.square(errors))))
    )
    # Their vibration amplitudes are given in volts only, so the tactile
    # times cannot test the latency curve; report what the model needs.
    touch: Any = human.touch
    result.details = {
        "model_increase_ms": model,
        "observed_increase_ms": observed,
        "tactile_minus_light_10deg_ms": {
            "observed": {k: v - 16.0 for k, v in PALM_VIBRATION.items()},
            "model_30db": 1000.0 * (touch.latency(STRONG_VIBRATION_DB) - base),
            "model_at_threshold": 1000.0 * (touch.latency(1e-9) - base),
        },
    }
    return result


@benchmark("response.drt")
def drt(human: Human) -> Result:
    """Detection-response task while driving, younger and older drivers.

    Both studies measured while driving, so driving holds the central stage
    (``response.drt_load``). Conti et al.'s light was head-mounted where ISO
    17488 puts it; Cooper et al. pooled a vibration under the collarbone
    with a light reflected in the windshield, equally often.
    """
    from .load import VIBRATION_DB, drt_rt

    result = Result("response.drt", ("response", "touch"))
    visual = drt_rt(human, ["head_light"], [DRIVING])
    on_road = drt_rt(human, ["tactile", "forward_light"], [DRIVING])
    older = build_human(
        Population(human.config, profile=OLDER_ADULTS).typical(), seed=0
    )
    on_road_older = drt_rt(older, ["tactile", "forward_light"], [DRIVING])
    result.score_against("response.drt_visual", visual)
    result.score_against("response.drt_on_road", on_road)
    result.score_against("response.older_slowing", on_road_older / on_road)
    result.details = {
        "visual_ms": visual,
        "on_road_ms": on_road,
        "on_road_older_ms": on_road_older,
        "visual_alone_ms": drt_rt(human, ["head_light"], []),
        "vibration_sensation_db": VIBRATION_DB,
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


#: McCauley et al. (1976), Appendix B: cumulative MSI over time at the nine
#: vertical conditions not in O'Hanlon & McCauley's 2-hour set, as
#: (frequency Hz, r.m.s. acceleration g, minutes, observed %, the authors'
#: descriptive model %). Row 17 reads 23 in the scan (27 by its printed
#: difference); 23 is used.
SICKNESS_TIME_COURSE = (
    (0.18, 0.170, 5, 8, 6),
    (0.18, 0.170, 15, 23, 21),
    (0.18, 0.170, 25, 40, 31),
    (0.18, 0.170, 35, 47, 37),
    (0.18, 0.170, 45, 55, 42),
    (0.18, 0.170, 85, 60, 50),
    (0.2, 0.234, 5, 11, 11),
    (0.2, 0.234, 15, 37, 32),
    (0.2, 0.234, 25, 43, 44),
    (0.2, 0.234, 35, 57, 50),
    (0.2, 0.234, 45, 60, 55),
    (0.2, 0.234, 55, 69, 58),
    (0.2, 0.234, 85, 71, 62),
    (0.25, 0.111, 5, 3, 1),
    (0.25, 0.111, 15, 7, 5),
    (0.25, 0.111, 25, 17, 9),
    (0.25, 0.111, 55, 24, 18),
    (0.25, 0.111, 65, 28, 20),
    (0.25, 0.111, 115, 31, 26),
    (0.25, 0.222, 5, 6, 6),
    (0.25, 0.222, 15, 17, 23),
    (0.25, 0.222, 25, 32, 33),
    (0.25, 0.222, 35, 39, 39),
    (0.25, 0.222, 45, 41, 44),
    (0.25, 0.222, 55, 54, 47),
    (0.25, 0.222, 65, 56, 49),
    (0.25, 0.222, 85, 57, 52),
    (0.25, 0.222, 115, 63, 55),
    (0.25, 0.333, 5, 2, 15),
    (0.25, 0.333, 15, 40, 40),
    (0.25, 0.333, 25, 51, 52),
    (0.25, 0.333, 35, 56, 59),
    (0.25, 0.333, 65, 58, 67),
    (0.25, 0.333, 75, 67, 68),
    (0.25, 0.333, 115, 69, 71),
    (0.417, 0.444, 15, 20, 17),
    (0.417, 0.444, 65, 33, 42),
    (0.417, 0.444, 115, 40, 48),
    (0.6, 0.444, 65, 4, 8),
    (0.6, 0.444, 115, 8, 12),
    (0.6, 0.555, 15, 9, 3),
    (0.6, 0.555, 115, 18, 18),
    (0.7, 0.555, 115, 4, 6),
)


def sickness_time_course(
    vestibular: object, dt: float = 0.1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predicted, observed and McCauley-model MSI at the time-course points."""
    conditions = sorted({(f, a) for f, a, *_ in SICKNESS_TIME_COURSE})
    time = np.arange(0.0, 116 * 60.0, dt)
    motion = np.zeros((len(conditions), len(time), 3))
    for i, (frequency, rms_g) in enumerate(conditions):
        peak = rms_g * math.sqrt(2) * 9.81
        motion[i, :, 2] = peak * np.sin(2 * math.pi * frequency * time)
    msi = vestibular.simulate(dt, motion)  # type: ignore[attr-defined]
    predicted, observed, descriptive = [], [], []
    for frequency, rms_g, minutes, seen, fitted in SICKNESS_TIME_COURSE:
        row = conditions.index((frequency, rms_g))
        step = min(int(round(minutes * 60.0 / dt)), len(time) - 1)
        predicted.append(msi[row, step])
        observed.append(seen)
        descriptive.append(fitted)
    return np.array(predicted), np.array(observed, float), np.array(descriptive, float)


@benchmark("vestibular.sickness_time_course")
def sickness_time_course_benchmark(human: Human, dt: float = 0.1) -> Result:
    """Cumulative MSI over up to 115 minutes at nine vertical conditions
    (0.18-0.7 Hz) that are not in the 2-hour set."""
    result = Result("vestibular.sickness_time_course", ("vestibular",))
    predicted, observed, descriptive = sickness_time_course(human.vestibular, dt)
    error = float(np.sqrt(np.mean((predicted - observed) ** 2)))
    result.score_against("vestibular.time_course_msi_error", error)
    result.details = {
        "points": len(observed),
        "correlation": float(np.corrcoef(predicted, observed)[0, 1]),
        "mccauley_model_error": float(np.sqrt(np.mean((descriptive - observed) ** 2))),
        "predicted": predicted.tolist(),
        "observed": observed.tolist(),
    }
    return result
