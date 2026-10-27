"""Detection-response times under secondary-task load.

``response.drt_load`` predicts how much driving, an n-back and SuRT slow
detection responses through the shared central stage
(:mod:`embodied_human.tasks.load`):

- The driving cost is predicted from threaded cognition's driver model
  and compared with Conti, Dlugosch and Bengler (2014): +51 ms for the
  head-mounted visual DRT while driving.
- The n-back and SuRT chunks are fitted to Conti et al.'s visual DRT, so
  those are consistency checks.
- Cooper et al. (2020) used the same 2-back and a different SuRT on the
  road, with a tactile DRT and a forward light whose reaction times they
  pooled. Their raw data give the tactile times alone
  (``examples/cooper_drt_aggregates.py``); its increases over driving are
  predictions.

During SuRT the eyes are often on the display, which moves a head-mounted
light further into the periphery; how often comes from the attentive
driver doing SuRT on a straight road.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from ..core import Human
from ..response import CentralTask, Signal
from ..tasks.drt import DRTConfig
from ..tasks.load import DRIVING, n_back, surt
from .benchmarks import Result, benchmark
from .protocol import protocol_seed

#: Head-mounted DRT light relative to straight ahead, deg (ISO 17488,
#: 6.3.2.1: 20 deg left, 10 deg up).
HEAD_LIGHT = (-20.0, 10.0)
#: Cooper et al.'s light, seen in the windshield in the forward view (deg,
#: assumed).
FORWARD_LIGHT = (0.0, 5.0)
#: Sensation level of the DRT vibrator, dB above threshold (assumed).
VIBRATION_DB = 20.0

#: Conti et al. (2014), Table 3, pooled over four groups (N = 71), ms.
CONTI = {
    "alone": 293.4,
    "driving": 344.1,
    "0-back": 479.1,
    "2-back": 636.6,
    "surt": 729.2,
}
#: Cooper et al. (2020): tactile DRT of drivers aged 21-36, means of 65
#: participants computed from their raw data
#: (evidence/cooper-drt-aggregates.json), ms.
COOPER = {"driving": 523.0, "2-back": 817.4, "surt": 715.4}

# Published analysis windows. These censor measurements without changing RT draws.
CONTI_WINDOW = (0.1, 2.0)
COOPER_WINDOW = DRTConfig().window


def display_share(human: Human, seconds: float = 60.0) -> float:
    """Share of time the attentive driver's eyes are on the SuRT display."""
    from ..agents.driver import DISPLAY
    from ..agents.road import simulate_glances

    log = simulate_glances(human, speed=80 / 3.6, seconds=seconds, seed=protocol_seed())
    return float(np.mean([area == DISPLAY for area in log.areas]))


def _eccentricity(light: tuple[float, float], gaze: tuple[float, float]) -> float:
    return math.hypot(light[0] - gaze[0], light[1] - gaze[1])


def drt_rt(
    human: Human,
    stimuli: Sequence[str],
    load: Sequence[CentralTask],
    on_display: float = 0.0,
    samples: int = 6000,
    window: tuple[float, float] = COOPER_WINDOW,
) -> float:
    """Mean response time of hits in the declared study window, in ms."""
    return drt_measurement(human, stimuli, load, on_display, samples, window)[
        "mean_rt_ms"
    ]


def drt_measurement(
    human: Human,
    stimuli: Sequence[str],
    load: Sequence[CentralTask],
    on_display: float = 0.0,
    samples: int = 6000,
    window: tuple[float, float] = COOPER_WINDOW,
) -> dict[str, Any]:
    """Measure hit conditional RT and retain uncensored model summaries.

    Stimuli are equiprobable. Gaze components have their declared time shares.
    Hit probabilities weight the component means after response censoring.
    Conti et al. (2014, Methods) retain 100 to 2000 ms responses. Cooper et
    al. (2020, Dependent Measures) retain 100 to 2500 ms responses.
    """
    from ..agents.driver import DriverAgentParameters

    if not stimuli or any(
        name not in ("tactile", "head_light", "forward_light") for name in stimuli
    ):
        raise ValueError("DRT stimuli must be tactile, head_light or forward_light")
    if not math.isfinite(on_display) or not 0 <= on_display <= 1:
        raise ValueError("on_display must be a finite share within zero and one")
    if type(samples) is not int or samples < 1:
        raise ValueError("samples must be a positive integer")
    if (
        len(window) != 2
        or not all(math.isfinite(bound) for bound in window)
        or not 0 <= window[0] < window[1]
    ):
        raise ValueError("window must contain finite ordered nonnegative bounds")
    display = DriverAgentParameters().display_position
    response: Any = human.response
    touch: Any = human.touch
    hit_weight = hit_sum = finite_weight = finite_sum = uncensored = 0.0
    premature = late = nonresponse = 0.0
    components = []
    for stimulus in stimuli:
        for gaze, weight in (((0.0, 0.0), 1.0 - on_display), (display, on_display)):
            if weight <= 0:
                continue
            if stimulus == "tactile":
                signal = Signal("tactile", 0.0, touch.latency(VIBRATION_DB))
            else:
                # Both lights stay put while the eyes turn to the display.
                light = HEAD_LIGHT if stimulus == "head_light" else FORWARD_LIGHT
                eccentricity = _eccentricity(light, gaze)
                signal = Signal("visual", 0.0, response.visual_latency(eccentricity))
            times = np.array([
                response.respond([signal], load).time for _ in range(samples)
            ])
            finite = np.isfinite(times)
            hits = finite & (times >= window[0]) & (times <= window[1])
            probability = weight / len(stimuli)
            hit_weight += probability * float(hits.mean())
            hit_sum += probability * float(times[hits].sum()) / samples
            finite_weight += probability * float(finite.mean())
            finite_sum += probability * float(times[finite].sum()) / samples
            uncensored += probability * float(times.mean())
            premature += probability * float(np.mean(finite & (times < window[0])))
            late += probability * float(np.mean(finite & (times > window[1])))
            nonresponse += probability * float(np.mean(~finite))
            components.append({
                "stimulus": stimulus, "gaze": list(gaze), "weight": probability,
                "samples": samples, "hits": int(hits.sum()),
                "uncensored_mean_rt_ms": 1000.0 * float(times.mean()),
            })
    return {
        "mean_rt_ms": 1000.0 * hit_sum / hit_weight if hit_weight else math.nan,
        "hit_rate": hit_weight,
        "premature_rate": premature,
        "late_rate": late,
        "nonresponse_rate": nonresponse,
        "uncensored_mean_rt_ms": 1000.0 * uncensored,
        "uncensored_finite_mean_rt_ms": (
            1000.0 * finite_sum / finite_weight if finite_weight else math.nan
        ),
        "window_s": list(window),
        "components": components,
    }


def conditions(on_display: float) -> dict[str, tuple[list[CentralTask], float]]:
    """Central load and share of eyes on the display per condition."""
    return {
        "alone": ([], 0.0),
        "driving": ([DRIVING], 0.0),
        "0-back": ([DRIVING, n_back(0)], 0.0),
        "2-back": ([DRIVING, n_back(2)], 0.0),
        "surt": ([DRIVING, surt()], on_display),
    }


@benchmark("response.drt_load")
def drt_load(human: Human) -> Result:
    """DRT under driving, n-back and SuRT load (see the module notes)."""
    result = Result("response.drt_load", ("response", "touch", "attention"))
    share = display_share(human)
    conti_measurements = {
        name: drt_measurement(
            human, ["head_light"], load, on_display, window=CONTI_WINDOW
        )
        for name, (load, on_display) in conditions(share).items()
    }
    cooper_measurements = {
        name: drt_measurement(
            human, ["tactile"], load, on_display, window=COOPER_WINDOW
        )
        for name, (load, on_display) in conditions(share).items()
        if name in COOPER
    }
    conti = {
        name: measurement["mean_rt_ms"]
        for name, measurement in conti_measurements.items()
    }
    cooper = {
        name: measurement["mean_rt_ms"]
        for name, measurement in cooper_measurements.items()
    }
    result.score_against("response.drt_driving_cost", conti["driving"] - conti["alone"])
    result.score_against(
        "response.drt_2back_cost", cooper["2-back"] - cooper["driving"]
    )
    result.score_against("response.drt_surt_cost", cooper["surt"] - cooper["driving"])
    result.details = {
        "display_share": share,
        "conti_model_ms": conti,
        "conti_human_ms": CONTI,
        "conti_fitted_costs_ms": {
            k: {
                "model": conti[k] - conti["driving"],
                "human": CONTI[k] - CONTI["driving"],
            }
            for k in ("0-back", "2-back", "surt")
        },
        "cooper_model_ms": cooper,
        "cooper_human_ms": COOPER,
        "conti_measurements": conti_measurements,
        "cooper_measurements": cooper_measurements,
        "measurement_note": "Mean RT includes only responses within each study's "
        "response window. Uncensored model means and hit rates remain available.",
    }
    return result
