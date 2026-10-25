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
  pooled. Their increases over driving are predictions.

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
from ..tasks.load import DRIVING, n_back, surt
from .benchmarks import Result, benchmark

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
#: Cooper et al. (2020), Table 4, drivers aged 21-36, ms.
COOPER = {"driving": 526.0, "2-back": 760.0, "surt": 703.0}


def display_share(human: Human, seconds: float = 60.0) -> float:
    """Share of time the attentive driver's eyes are on the SuRT display."""
    from ..agents.driver import DISPLAY
    from ..agents.road import simulate_glances

    log = simulate_glances(human, speed=80 / 3.6, seconds=seconds, seed=0)
    return float(np.mean([area == DISPLAY for area in log.areas]))


def _eccentricity(light: tuple[float, float], gaze: tuple[float, float]) -> float:
    return math.hypot(light[0] - gaze[0], light[1] - gaze[1])


def drt_rt(
    human: Human,
    stimuli: Sequence[str],
    load: Sequence[CentralTask],
    on_display: float = 0.0,
    samples: int = 6000,
) -> float:
    """Mean DRT response time (ms) to equiprobable stimuli, with the eyes on
    the display a share ``on_display`` of the time."""
    from ..agents.driver import DriverAgentParameters

    display = DriverAgentParameters().display_position
    response: Any = human.response
    touch: Any = human.touch
    times = []
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
            mean = response.mean_reaction([signal], samples=samples, load=load)
            times.append((weight / len(stimuli), mean))
    return 1000.0 * sum(w * m for w, m in times) / sum(w for w, _ in times)


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
    conti = {
        name: drt_rt(human, ["head_light"], load, on_display)
        for name, (load, on_display) in conditions(share).items()
    }
    cooper = {
        name: drt_rt(human, ["tactile", "forward_light"], load, on_display)
        for name, (load, on_display) in conditions(share).items()
        if name in COOPER
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
    }
    return result
