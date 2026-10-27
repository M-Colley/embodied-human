"""Benchmarks of effort, perceived risk and emotion.

- ``fatigue.endurance``: endurance times of sustained contractions against
  the joint-specific meta-analysis of Frey-Law and Avin (2010), the data
  the fatigue rates were fitted to (a consistency check).
- ``risk.lane_width``: the speed the risk field allows on lanes of
  2.85-3.75 m, with its threshold and width growth fitted to the 3.0 and
  3.5 m lanes, against Liu, Wang and Fu (2016).
- ``affect.emotion_study``: emotions in three tasks against the ratings of
  Zhang, Hilpert, Broekens and Jokinen (2024), Experiment 1.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..core import Human, ModuleChoice, create
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


#: Lane widths the risk field is fitted to; the others are predictions.
LANE_WIDTH_FIT = (3.0, 3.5)


def fit_lane_widths(
    choice: ModuleChoice, widths: tuple[float, ...] = LANE_WIDTH_FIT
) -> Any:
    """Fit the threshold and the field's width growth to the speeds chosen
    on ``widths`` (SciPy least squares); return the fitted module."""
    from scipy.optimize import least_squares

    def module(x: np.ndarray) -> Any:
        settings = {"width_rate": x[0], "threshold": math.exp(x[1])}
        return create("risk", choice.name, **{**dict(choice.params), **settings})

    def residuals(x: np.ndarray) -> list[float]:
        fitted = module(x)
        return [3.6 * fitted.preferred_speed(w) - LANE_WIDTH_SPEED[w] for w in widths]

    best = None
    for start in (0.001, 0.02, 0.05, 0.1):
        risk = module(np.array([start, 0.0])).lane_risk(
            LANE_WIDTH_SPEED[3.5] / 3.6, 3.5
        )
        solution = least_squares(
            residuals, [start, math.log(risk)], bounds=([0.0, -10.0], [1.0, 30.0])
        )
        if best is None or solution.cost < best.cost:
            best = solution
    assert best is not None
    return module(best.x)


@benchmark("risk.lane_width")
def lane_width(human: Human) -> Result:
    """Speed chosen on straight lanes of different widths.

    The perceived-risk threshold and the field's width growth are fitted
    to the 3.0 and 3.5 m lanes; 2.85, 3.25 and 3.75 m are predictions.
    """
    result = Result("risk.lane_width", ("risk",))
    tuned = fit_lane_widths(human.config.risk)
    predicted = {w: 3.6 * tuned.preferred_speed(w) for w in LANE_WIDTH_SPEED}
    held_out = [w for w in LANE_WIDTH_SPEED if w not in LANE_WIDTH_FIT]
    error = float(np.mean([abs(predicted[w] - LANE_WIDTH_SPEED[w]) for w in held_out]))
    result.score_against("risk.lane_width_speed_error", error)
    result.details = {
        "fitted_to_m": list(LANE_WIDTH_FIT),
        "threshold": tuned.params.threshold,
        "width_rate": tuned.params.width_rate,
        "published_width_rate": 0.001,
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
    from ..affect import AppraisalAffect
    from ..affect import emotion_study as simulate

    result = Result("affect.emotion_study", ("affect",))
    try:
        model = simulate(seed=0, runs=runs, choice=human.config.affect)
    except ImportError as error:
        result.skipped = f"needs the affect module's optional dependencies: {error}"
        return result
    required = {"happiness", "boredom", "irritation"}
    if any(not required <= set(values) for values in model.values()):
        result.skipped = (
            "this study requires classifier outputs for happiness, boredom "
            "and irritation; the selected affect module has different labels"
        )
        return result
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
        "affect_module": human.config.affect.name,
        "learner": (
            "selected appraisal implementation"
            if isinstance(human.affect, AppraisalAffect)
            else "baseline AppraisalAffect; selected module classifies appraisals"
        ),
    }
    return result


#: Kraus, Scholz, Stiegemeier & Baumann (2020): trust (7-point scale) at
#: each measurement point, and the events before each point (Figure 2,
#: Tables 2 and 7). "ok" is an automated overtake, "tor" a take-over
#: request, "malfunction" a stop in safe mode.
KRAUS_STUDY1 = {
    "MF-": (4.61, 4.88, 5.09, 5.51, 5.21, 5.33, 5.22, 5.49, 5.64, 5.72, 5.12, 5.02),
    "MF+": (4.58, 5.04, 5.09, 5.53, 5.35, 5.69, 5.06, 5.74, 5.84, 5.99, 5.80, 5.50),
}
KRAUS_STUDY2 = {
    "MF-/LT": (5.03, 5.66, 5.72, 5.91, 5.98, 6.09, 6.11, 6.17, 6.22, 6.22),
    "MF+/LT": (4.64, 4.95, 5.27, 5.42, 5.60, 5.13, 5.55, 5.75, 5.84, 5.96),
    "MF-/HT": (4.47, 5.13, 5.60, 5.67, 5.73, 5.61, 5.93, 5.85, 6.04, 5.98),
    "MF+/HT": (5.20, 5.25, 5.53, 5.75, 5.75, 5.67, 5.83, 5.82, 5.90, 5.90),
}


def kraus_events(study: int, group: str) -> list[list[str]]:
    """Events between consecutive measurement points (practice first)."""
    malfunction = group.startswith("MF+")
    # The practice drive counts as positive experience: its request and
    # malfunction were introduced beforehand, and trust rose over it in
    # every group of both studies.
    if study == 1:
        # Then t1, t2, request 1 (m1), t3, t4 (malfunction for MF+), t5-t7,
        # request 2 (m2), and one more overtake (tpost).
        return [
            ["ok"],
            ["ok"], ["ok"], ["tor"], ["ok"],
            ["malfunction" if malfunction else "ok"],
            ["ok"], ["ok"], ["ok"], ["tor"], ["ok"],
        ]  # fmt: skip
    later = [["ok"] for _ in range(8)]
    if malfunction:
        later[3] = ["malfunction"]
    return [["ok"], *later]


def trust_course(
    initial: float,
    events: list[list[str]],
    capability: dict[str, float],
    **params: float,
) -> list[float]:
    """Trust ratings (1-7) after each group of events, by the trust module
    with no noise, starting from ``initial`` (1-7)."""
    from ..trust import DecisionFieldTrust

    start = (initial - 1.0) / 6.0
    model = DecisionFieldTrust(noise_sd=0.0, automation_belief=start, **params)
    ratings = [initial]
    for group in events:
        for event in group:
            model.update(
                automation_capability=capability[event], manual_capability=None
            )
        ratings.append(1.0 + 6.0 * model.trust)
    return ratings


def fit_kraus(inertia: float = 0.5) -> dict[str, float]:
    """Fit event capabilities and opacity to Study 1 (SciPy least squares)."""
    from scipy.optimize import least_squares

    def residuals(x: np.ndarray) -> list[float]:
        capability = {"ok": x[0], "tor": x[1], "malfunction": x[2]}
        errors: list[float] = []
        for group, observed in KRAUS_STUDY1.items():
            course = trust_course(
                observed[0], kraus_events(1, group), capability,
                opacity=x[3], inertia=inertia,
            )  # fmt: skip
            errors += [m - o for m, o in zip(course[1:], observed[1:])]
        return errors

    solution = least_squares(
        residuals, [0.85, 0.6, 0.3, 3.0], bounds=([0, 0, 0, 1], [1, 1, 1, 50])
    )
    ok, tor, malfunction, opacity = solution.x
    return {
        "ok": float(ok),
        "tor": float(tor),
        "malfunction": float(malfunction),
        "opacity": float(opacity),
        "study1_rmse": float(np.sqrt(np.mean(np.square(solution.fun)))),
    }


@benchmark("trust.kraus_dynamics")
def kraus_dynamics(human: Human) -> Result:
    """Trust over an automated drive with take-over requests and a
    malfunction: fitted to Kraus et al.'s Study 1, predicting Study 2's
    low-transparency groups (no requests; a malfunction for one group).
    The model has no notion of transparency, so the high-transparency
    groups are reported but not scored."""
    result = Result("trust.kraus_dynamics", ("trust",))
    fitted = fit_kraus()
    capability = {k: fitted[k] for k in ("ok", "tor", "malfunction")}
    predicted = {
        group: trust_course(
            observed[0], kraus_events(2, group), capability, opacity=fitted["opacity"]
        )
        for group, observed in KRAUS_STUDY2.items()
    }
    errors = [
        m - o
        for group in ("MF-/LT", "MF+/LT")
        for m, o in zip(predicted[group][1:], KRAUS_STUDY2[group][1:])
    ]
    result.score_against(
        "trust.study2_rmse", float(np.sqrt(np.mean(np.square(errors))))
    )
    # The malfunction's transient dip (t3 to t4), which the overall error
    # can hide behind the trend.
    course = predicted["MF+/LT"]
    result.score_against("trust.malfunction_dip", course[5] - course[4])
    result.details = {
        "fitted_on_study1": fitted,
        "study2_model": predicted,
        "study2_human": KRAUS_STUDY2,
    }
    return result
