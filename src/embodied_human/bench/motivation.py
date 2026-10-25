"""Benchmarks of effort, perceived risk and emotion.

- ``fatigue.endurance``: endurance times of sustained contractions against
  the joint-specific meta-analysis of Frey-Law and Avin (2010), the data
  the fatigue rates were fitted to (a consistency check).
- ``risk.lane_width``: the speed the risk field allows on lanes of
  2.85-3.75 m, with its threshold fitted to the 3.5 m lane only, against
  Liu, Wang and Fu (2016).
- ``affect.emotion_study``: emotions in three tasks against the ratings of
  Zhang, Hilpert, Broekens and Jokinen (2024), Experiment 1.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..core import Human, create
from .benchmarks import Result, benchmark
from .metrics import kendall_tau

#: Frey-Law & Avin (2010), Table 2: ET = b0 * f^b1 (s, f = fraction of MVC).
ENDURANCE_POWER = {
    "ankle": (34.71, -2.06),
    "knee": (19.38, -1.88),
    "trunk": (22.69, -2.27),
    "shoulder": (14.86, -1.83),
    "elbow": (17.98, -2.21),
    "hand": (33.55, -1.61),
    "general": (21.92, -1.98),
}


@benchmark("fatigue.endurance")
def endurance(human: Human) -> Result:
    """Hold 20, 50 and 80% MVC to failure at every joint."""
    result = Result("fatigue.endurance", ("fatigue",))
    fatigue: Any = human.fatigue
    rows = []
    for joint, (b0, b1) in ENDURANCE_POWER.items():
        for load in (0.2, 0.5, 0.8):
            model = fatigue.endurance_time(joint, load, dt=0.02)
            meta = b0 * load**b1
            rows.append(
                {"joint": joint, "load": load, "model_s": model, "meta_s": meta}
            )
    errors = [abs(math.log(r["model_s"] / r["meta_s"])) for r in rows]
    result.score_against("fatigue.endurance_error", float(np.median(errors)))
    result.details = {
        "endurance": rows,
        "note": "consistency check: the rates were fitted to these curves",
    }
    return result


#: Liu, Wang & Fu (2016), Table 2: lane width (m) -> mean speed (km/h),
#: right lane, 0.5 m shoulder, simulator, 24 drivers.
LANE_WIDTH_SPEED = {2.85: 56.18, 3.0: 60.78, 3.25: 65.46, 3.5: 76.85, 3.75: 81.92}


@benchmark("risk.lane_width")
def lane_width(human: Human) -> Result:
    """Speed chosen on straight lanes of different widths.

    The perceived-risk threshold is fitted to the 3.5 m lane; the other
    four widths are predictions.
    """
    result = Result("risk.lane_width", ("risk",))
    risk: Any = human.risk
    fitted = risk.lane_risk(LANE_WIDTH_SPEED[3.5] / 3.6, 3.5)
    tuned = create(
        "risk",
        human.config.risk.name,
        **{**dict(human.config.risk.params), "threshold": fitted},
    )
    predicted = {w: 3.6 * tuned.preferred_speed(w) for w in LANE_WIDTH_SPEED}  # type: ignore[attr-defined]
    held_out = [w for w in LANE_WIDTH_SPEED if w != 3.5]
    error = float(np.mean([abs(predicted[w] - LANE_WIDTH_SPEED[w]) for w in held_out]))
    result.score_against("risk.lane_width_speed_error", error)
    result.details = {
        "threshold": fitted,
        "speed_kmh": {
            str(w): {"model": predicted[w], "human": LANE_WIDTH_SPEED[w]}
            for w in LANE_WIDTH_SPEED
        },
        "kendall_tau": kendall_tau(
            list(predicted.values()), list(LANE_WIDTH_SPEED.values())
        ),
    }
    return result


#: Zhang et al. (2024), Experiment 1: normalised human ratings per task
#: (happiness, boredom, irritation), means of 8-10 people per task.
EMOTION_RATINGS = {
    "happiness": (0.825, 0.164, 0.010),
    "boredom": (0.383, 0.487, 0.130),
    "irritation": (0.193, 0.204, 0.603),
}


@benchmark("affect.emotion_study")
def emotion_study(human: Human, runs: int = 3) -> Result:
    """Happiness, boredom and irritation tasks, appraised and classified by
    27 simulated participants per learner, averaged over ``runs`` learners."""
    from ..affect import emotion_study as simulate

    result = Result("affect.emotion_study", ("affect",))
    try:
        import sklearn  # noqa: F401
    except ImportError as error:
        result.skipped = f"needs scikit-learn: {error}"
        return result
    model = simulate(seed=0, runs=runs)
    predicted, observed = [], []
    for task, ratings in EMOTION_RATINGS.items():
        for emotion, value in zip(("happiness", "boredom", "irritation"), ratings):
            predicted.append(model[task][emotion])
            observed.append(value)
    p, o = np.array(predicted), np.array(observed)
    rmse = float(np.sqrt(np.mean((p - o) ** 2)))
    result.score_against("affect.emotion_rmse", rmse)
    result.details = {
        "model": model,
        "human": {
            k: dict(zip(("happiness", "boredom", "irritation"), v))
            for k, v in EMOTION_RATINGS.items()
        },
        "r_squared": float(1 - np.sum((o - p) ** 2) / np.sum((o - o.mean()) ** 2)),
        "original_model": {"r_squared": 0.78, "rmse": 0.13},
    }
    return result
