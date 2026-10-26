"""Simulated studies: samples of participants, conditions and designs.

These benchmarks run the human as a sample of simulated participants
(:mod:`embodied_human.population`), report bootstrap intervals over them,
and compare effects with human studies:

- ``driving.glance_population``: in-vehicle glance durations of 20
  participants against Kujala and Saariluoma (2011), including the
  within-person variability.
- ``driving.glance_speed``: a counterfactual: how glance duration changes
  with speed, against voluntary occlusion (Senders et al., 1967) and in-car
  search glances (Jokinen, Kujala and Oulasvirta, 2021).
- ``design.touch_key_size``: an application: replicate the analysis of
  Kim et al. (2014) with 30 simulated participants and predict the key size
  beyond which larger keys stop helping.

Glance studies run on :class:`~embodied_human.agents.road.StraightRoad`,
which needs no MyoSuite; the driving benchmarks in :mod:`integrated`
repeat them on the full body.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..attention import ROAD
from ..core import Human, HumanConfig, build_human
from ..population import BASELINE_VARIATIONS, Population, applicable
from .benchmarks import Result, benchmark
from .metrics import bootstrap_ci, elasticity

KMH = 1 / 3.6


def glance_participants(
    config: HumanConfig,
    *,
    participants: int = 20,
    speed: float = 50 * KMH,
    seconds: float = 120.0,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Glance metrics of simulated participants doing SuRT while cruising."""
    from ..agents.road import simulate_glances

    population = Population(config, applicable(config, BASELINE_VARIATIONS))
    rows = []
    for index, person in enumerate(population.participants(participants, seed)):
        log = simulate_glances(
            build_human(person, seed=seed + index),
            speed=speed,
            seconds=seconds,
            seed=seed + index,
        )
        durations = np.array(log.glance_durations())
        rows.append(
            {
                "participant": index,
                "glances": int(len(durations)),
                "mean_s": float(durations.mean()) if len(durations) else math.nan,
                "sd_s": float(durations.std(ddof=1))
                if len(durations) > 1
                else math.nan,
                "max_s": float(durations.max()) if len(durations) else math.nan,
                "long": int(np.sum(durations > 2.0)),
                "returns_by_reason": log.glance_metrics()["returns_by_reason"],
                "parameters": {
                    c: dict(getattr(person, c).params)
                    for c in ("attention", "eyes", "motor", "proprioception")
                },
            }
        )
    return rows


@benchmark("driving.glance_population")
def glance_population(
    human: Human, participants: int = 20, seconds: float = 120.0
) -> Result:
    """Twenty simulated participants do SuRT while driving at 50 km/h.

    Kujala and Saariluoma (2011) had 20 drivers search an in-vehicle
    display at 40-60 km/h; glance durations there include transitions, as
    here. Their task was menu search, not SuRT.
    """
    result = Result(
        "driving.glance_population", ("eyes", "objects", "attention", "estimator")
    )
    rows = glance_participants(human.config, participants=participants, seconds=seconds)
    means = [r["mean_s"] for r in rows]
    sds = [r["sd_s"] for r in rows]
    glances = sum(r["glances"] for r in rows)
    long_share = sum(r["long"] for r in rows) / glances if glances else math.nan
    result.score_against("driving.human_mean_glance", float(np.nanmean(means)))
    result.score_against("driving.human_glance_sd", float(np.nanmean(sds)))
    result.score_against("driving.human_long_glance_share", long_share)
    result.score_against("driving.mean_offroad_glance", float(np.nanmean(means)))
    result.score_against("driving.long_glance_share", long_share)
    reasons: dict[str, int] = {}
    for row in rows:
        for reason, count in row["returns_by_reason"].items():
            reasons[reason] = reasons.get(reason, 0) + count
    result.details = {
        "participants": participants,
        "speed_kmh": 50.0,
        "seconds_each": seconds,
        "mean_glance_ci95_s": bootstrap_ci(means),
        "within_person_sd_ci95_s": bootstrap_ci(sds),
        "between_person_sd_of_means_s": float(np.nanstd(means, ddof=1)),
        "longest_glance_s": float(np.nanmax([r["max_s"] for r in rows])),
        "returns_by_reason": reasons,
        "population": Population(
            human.config, applicable(human.config, BASELINE_VARIATIONS)
        ).describe(),
        "per_participant": [
            {k: v for k, v in r.items() if k != "returns_by_reason"} for r in rows
        ],
    }
    return result


@benchmark("driving.glance_speed")
def glance_speed(
    human: Human,
    speeds_kmh: tuple[float, ...] = (40.0, 60.0, 80.0, 100.0, 120.0),
    seconds: float = 120.0,
) -> Result:
    """How the typical participant's glance duration scales with speed.

    The elasticity (log-log slope) is compared with voluntary occlusion,
    where drivers look away for a roughly constant distance (Senders et
    al., 1967), and with in-car search glances, whose duration did not
    change with speed (Jokinen et al., 2021). The two human results differ;
    a model of glance timing by road uncertainty alone cannot match both.
    """
    from ..agents.road import simulate_glances

    result = Result("driving.glance_speed", ("attention", "estimator", "objects"))
    means = []
    for speed in speeds_kmh:
        log = simulate_glances(
            build_human(human.config, seed=0),
            speed=speed * KMH,
            seconds=seconds,
            seed=0,
        )
        means.append(float(np.mean(log.glance_durations())))
    slope = elasticity(speeds_kmh, means)
    result.score_against("driving.occlusion_speed_elasticity", slope)
    result.score_against("driving.incar_glance_speed_elasticity", slope)
    result.details = {
        "mean_glance_by_speed_s": dict(zip(map(str, speeds_kmh), means)),
        "distance_per_glance_m": {
            str(v): m * v * KMH for v, m in zip(speeds_kmh, means)
        },
    }
    return result


#: Mean distance between two different keys of a 3 x 4 phone keypad
#: (digits 1-9 in a square grid, 0 below 8), in key pitches. Derived.
KEYPAD_DISTANCE = 1.744


@benchmark("design.touch_key_size")
def touch_key_size(
    human: Human,
    sizes_mm: tuple[float, ...] = (7.5, 12.5, 17.5, 22.5, 27.5),
    participants: int = 30,
    rounds: int = 2,
    gap_mm: float = 0.0,
    alpha: float = 0.05,
) -> Result:
    """Rank touch-key sizes and find where larger keys stop helping.

    Kim et al. (2014) had 30 drivers enter digits on keys of 7.5-27.5 mm
    and found that performance stopped improving at 17.5 mm. Here 30
    simulated participants tap keys of each size with the physical
    fingertip (ISO multi-directional tapping at the mean distance between
    keypad keys; ``gap_mm`` between keys is assumed). As in the study, the
    asymptote is the smallest size whose error rate and movement time do
    not differ significantly (paired t-tests) from the largest key, with
    every larger key also not different. Stationary use only: the
    simulated participants are not driving.
    """
    result = Result("design.touch_key_size", ("motor", "eyes"))
    try:
        import embodied_ui  # noqa: F401
        import mujoco  # noqa: F401
    except ImportError as error:
        result.skipped = f"needs embodied-ui and mujoco: {error}"
        return result
    from scipy.stats import ttest_rel

    from ..agents.pointer import run_tapping
    from ..tasks.tapping import TappingCondition

    population = Population(human.config, applicable(human.config, BASELINE_VARIATIONS))
    errors = np.full((participants, len(sizes_mm)), math.nan)
    times = np.full_like(errors, math.nan)
    for i, person in enumerate(population.participants(participants, seed=0)):
        participant = build_human(person, seed=i)
        for j, size in enumerate(sizes_mm):
            width = size / 1000.0
            amplitude = KEYPAD_DISTANCE * (size + gap_mm) / 1000.0
            run = run_tapping(
                participant, TappingCondition(amplitude, width), rounds=rounds
            )
            measures = run.throughput()
            errors[i, j] = measures.error_rate
            times[i, j] = measures.movement_time
    largest = len(sizes_mm) - 1

    def differs(values: np.ndarray, j: int) -> bool:
        if j == largest or np.allclose(values[:, j], values[:, largest]):
            return False
        return bool(ttest_rel(values[:, j], values[:, largest]).pvalue < alpha)

    different = [differs(errors, j) or differs(times, j) for j in range(len(sizes_mm))]
    asymptote = sizes_mm[largest]
    for j in range(largest, -1, -1):
        if different[j]:
            break
        asymptote = sizes_mm[j]
    result.score_against("design.touch_key_asymptote", asymptote)
    result.details = {
        "participants": participants,
        "sizes_mm": list(sizes_mm),
        "error_rate": {
            str(s): {
                "mean": float(errors[:, j].mean()),
                "ci95": bootstrap_ci(errors[:, j]),
            }
            for j, s in enumerate(sizes_mm)
        },
        "movement_time_s": {
            str(s): {
                "mean": float(times[:, j].mean()),
                "ci95": bootstrap_ci(times[:, j]),
            }
            for j, s in enumerate(sizes_mm)
        },
        "differs_from_largest": dict(zip(map(str, sizes_mm), different)),
        "alpha": alpha,
        "gap_mm": gap_mm,
    }
    return result


@benchmark("driving.takeover_modality")
def takeover_modality(human: Human, runs: int = 12) -> Result:
    """Take-over requests by sound, vibration, light, or sound with light,
    during automation with SuRT on the straight road.

    The order of the modalities is compared with Petermeijer, Doubek and
    de Winter (2017) and the time to grab the wheel after a chime with an
    icon with Ko et al. (2022), whose drivers played a game on the centre
    console.
    """
    from ..agents.road import simulate_takeover

    result = Result(
        "driving.takeover_modality", ("hearing", "touch", "response", "eyes", "motor")
    )
    medians: dict[str, dict[str, float]] = {}
    for signals in (("visual",), ("auditory",), ("tactile",), ("auditory", "visual")):
        parts = [
            simulate_takeover(
                build_human(human.config, seed=seed), signals=signals, seed=seed
            ).takeover_components()
            for seed in range(runs)
        ]
        medians["+".join(signals)] = {
            key: float(np.nanmedian([p[key] for p in parts])) for key in parts[0]
        }
    visual = medians["visual"]["response_s"]
    agree = [medians[m]["response_s"] < visual for m in ("auditory", "tactile")]
    result.score_against("driving.takeover_modality_order", float(np.mean(agree)))
    result.score_against(
        "driving.takeover_hands_auditory", medians["auditory+visual"]["hand_s"]
    )
    result.details = {"median_components_s": medians, "runs": runs}
    return result


#: Gold et al. (2013), Fig. 3 and text: mean times (s) after the request
#: with 5 and 7 s to collision; braking and steering with their SDs.
GOLD = {
    "5 s": {"gaze": 0.41, "road": 0.69, "hands": 1.45, "intervention": 2.10},
    "7 s": {"gaze": 0.54, "road": 0.94, "hands": 1.79, "intervention": 2.89},
}
GOLD_INPUTS = {
    "5 s": {"braking": (2.06, 0.34), "steering": (2.27, 0.66)},
    "7 s": {"braking": (3.10, 0.66), "steering": (3.65, 1.33)},
}


@benchmark("driving.takeover_components")
def takeover_components(human: Human, runs: int = 16) -> Result:
    """The steps of a take-over against Gold et al. (2013).

    A tone with an icon interrupts SuRT during automation at 120 km/h on the
    straight road. Gold et al. timed the first saccade away from SuRT, the
    first fixation on the road, the hands on the wheel and the first
    braking or steering input; their means are compared with the model's
    gaze, eyes, hand and engagement times. None of the model's parameters
    was fitted to these data.
    """
    from ..agents.road import simulate_takeover

    result = Result(
        "driving.takeover_components",
        ("hearing", "response", "eyes", "attention", "motor"),
    )
    parts = [
        simulate_takeover(
            build_human(human.config, seed=seed), speed=120 * KMH, seed=seed
        ).takeover_components()
        for seed in range(runs)
    ]
    keys = {"gaze": "gaze_s", "road": "eyes_s", "hands": "hand_s"}
    keys["intervention"] = "engaged_s"
    model = {name: np.array([p[key] for p in parts]) for name, key in keys.items()}
    for name, values in model.items():
        result.score_against(f"driving.takeover_{name}", float(np.nanmean(values)))
    result.details = {
        "model_mean_s": {k: float(np.nanmean(v)) for k, v in model.items()},
        "model_ci95_s": {k: bootstrap_ci(v[np.isfinite(v)]) for k, v in model.items()},
        "model_response_s": float(np.mean([p["response_s"] for p in parts])),
        "runs": runs,
        "runs_with_eyes_on_task": int(np.isfinite(model["gaze"]).sum()),
        "gold_mean_s": GOLD,
        "gold_inputs_mean_sd_s": GOLD_INPUTS,
    }
    return result


#: Zeeb, Buchner & Schrauf (2015), Section 3.3: after a tone with an icon in
#: the head unit, during a visual task operated by a rotary knob.
ZEEB_TAKEOVER = {"road_s": (0.69, 0.20, 62), "hands_s": (1.14, 0.45, 88)}
#: Their Table 1 and Section 3.3 per monitoring group (n = 20, 36, 33): road
#: glances per minute, % of the minute on the road, longest eyes-off-road
#: time (s) in the minute before the request, and braking time (mean, SD).
ZEEB_GROUPS = {
    "high-risk": {
        "n": 20,
        "road_glances": 1.10,
        "road_pct": 2.15,
        "max_off_s": 44.35,
        "braking_s": (2.31, 0.67),
    },
    "medium-risk": {
        "n": 36,
        "road_glances": 4.72,
        "road_pct": 5.56,
        "max_off_s": 20.83,
        "braking_s": (1.86, 0.67),
    },
    "low-risk": {
        "n": 33,
        "road_glances": 11.79,
        "road_pct": 12.48,
        "max_off_s": 8.59,
        "braking_s": (1.63, 0.50),
    },
}


def _area_at(log: Any, time: float | None) -> str:
    """Area of interest the eyes were on at ``time`` (the start if None)."""
    index = int(np.searchsorted(np.asarray(log.times), time or 0.0))
    return str(log.areas[min(index, len(log.areas) - 1)])


@benchmark("driving.takeover_motor_readiness")
def takeover_motor_readiness(human: Human, runs: int = 16) -> Result:
    """First road fixation and hands on the wheel against Zeeb et al. (2015).

    Their drivers worked a visual task in the head unit through a rotary
    knob, so the hand lets go at once (``hand_release`` 0, assumed for a
    knob); the request was a tone with an icon in the same display. As in
    the study, runs with the eyes already on the road at the request are
    left out of the road fixation. Nothing was fitted to these data.
    """
    from dataclasses import replace

    from ..agents.driver import DriverAgentParameters
    from ..agents.road import simulate_takeover

    result = Result(
        "driving.takeover_motor_readiness", ("hearing", "response", "eyes", "motor")
    )
    base = DriverAgentParameters()
    params = replace(base, hand_release=0.0, icon_position=base.display_position)
    road, hands, engaged = [], [], []
    for seed in range(runs):
        log = simulate_takeover(
            build_human(human.config, seed=seed),
            speed=120 * KMH,
            seed=seed,
            params=params,
        )
        parts = log.takeover_components()
        if _area_at(log, log.request_at) != ROAD:
            road.append(parts["eyes_s"])
        hands.append(parts["hand_s"])
        engaged.append(parts["engaged_s"])
    result.score_against("driving.zeeb_road", float(np.mean(road)))
    result.score_against("driving.zeeb_hands", float(np.mean(hands)))
    result.details = {
        "runs": runs,
        "runs_eyes_off_road": len(road),
        "model_engaged_s": float(np.mean(engaged)),
        "human_braking_by_group_s": {k: v["braking_s"] for k, v in ZEEB_GROUPS.items()},
        "human_mean_sd_n": ZEEB_TAKEOVER,
    }
    return result


@benchmark("driving.automation_monitoring")
def automation_monitoring(human: Human, runs: int = 8) -> Result:
    """How the eyes share a visual task with the road while automation
    drives, against the minute before the request in Zeeb et al. (2015).

    The acceptance ranges span their low- to high-risk groups' means. The
    attention module knows nothing about automation: it returns to the road
    whenever the belief about the lane grows uncertain, as in manual
    driving.
    """
    from ..agents.road import simulate_takeover

    result = Result("driving.automation_monitoring", ("attention", "eyes"))
    start, end = 10.0, 70.0
    longest, road_share, returns = [], [], []
    for seed in range(runs):
        log = simulate_takeover(
            build_human(human.config, seed=seed),
            speed=120 * KMH,
            request_at=end,
            seconds=end,
            seed=seed,
        )
        glances = [
            (max(s, start), min(e, end))
            for s, e in log.glances()
            if e > start and s < end
        ]
        off = sum(e - s for s, e in glances)
        longest.append(max((e - s for s, e in glances), default=0.0))
        road_share.append(100.0 * (1.0 - off / (end - start)))
        returns.append(len(glances) * 60.0 / (end - start))
    result.score_against("driving.automation_max_offroad", float(np.mean(longest)))
    result.score_against("driving.automation_road_share", float(np.mean(road_share)))
    result.details = {
        "runs": runs,
        "window_s": [start, end],
        "model_road_glances_per_min": float(np.mean(returns)),
        "human_by_group": ZEEB_GROUPS,
    }
    return result
