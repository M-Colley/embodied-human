"""Declared evaluation roles for the published baseline protocols."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, replace
from typing import Any, Literal

Role = Literal["consistency", "calibration", "held_out", "exploratory", "unclassified"]


@dataclass(frozen=True)
class Evaluation:
    """Data use for one metric, independently of its numerical score."""

    role: Role
    evaluation_data: tuple[str, ...]
    fitting_data: tuple[str, ...] = ()
    rationale: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


_DECLARATIONS: dict[str, Evaluation] = {}


def _declare(
    metrics: tuple[str, ...],
    role: Role,
    evaluation: str,
    fitted: tuple[str, ...] = (),
    rationale: str = "",
) -> None:
    for metric in metrics:
        _DECLARATIONS[metric] = Evaluation(role, (evaluation,), fitted, rationale)


_declare(
    ("eyes.duration_slope", "eyes.duration_intercept"),
    "consistency",
    "carpenter1988.main_sequence",
    ("carpenter1988.main_sequence",),
    "The source also sets the main sequence coefficients.",
)
_declare(
    ("eyes.latency_median",),
    "consistency",
    "saslow1967.latency",
    ("saslow1967.latency",),
    "The source sets median latency.",
)
_declare(
    ("objects.e2",),
    "consistency",
    "strasburger2011.acuity",
    ("strasburger2011.acuity",),
    "The reference sets eccentricity scaling.",
)
_declare(
    ("vision.e2",),
    "consistency",
    "geisler_perry1998.foveation",
    ("geisler_perry1998.foveation",),
    "The reference sets E2.",
)
_declare(
    ("memory.span",),
    "consistency",
    "cowan2001.capacity",
    ("cowan2001.capacity",),
    "Focus capacity is set from this source.",
)
_declare(
    ("motor.touch_slope",),
    "consistency",
    "bi2013.touch_pointing",
    ("bi2013.touch_pointing",),
    "The source sets the pointing coefficients.",
)
_declare(
    ("motor.touch_throughput",),
    "held_out",
    "mackenzie2015.smartphone_throughput",
    ("bi2013.touch_pointing",),
    "MacKenzie reports a separate smartphone experiment with 16 participants. "
    "Task and throughput aggregation differ from the coefficient study.",
)
_declare(
    ("driving.mean_offroad_glance", "driving.long_glance_share"),
    "held_out",
    "nhtsa2013.glance_criteria",
    ("kujala_saariluoma2011.mean_and_sd",),
    "Design criteria are separate from glance fitting statistics.",
)
_declare(
    ("driving.human_mean_glance", "driving.human_glance_sd"),
    "calibration",
    "kujala_saariluoma2011.mean_and_sd",
    ("kujala_saariluoma2011.mean_and_sd",),
    "ABC fitted return tolerance and spread to these statistics.",
)
_declare(
    ("driving.human_long_glance_share",),
    "held_out",
    "kujala_saariluoma2011.long_glance_share",
    ("kujala_saariluoma2011.mean_and_sd",),
    "A held out statistic from the same study; participants are shared.",
)
_declare(
    ("driving.occlusion_speed_elasticity",),
    "held_out",
    "senders1967.occlusion",
    ("kujala_saariluoma2011.mean_and_sd",),
)
_declare(
    ("driving.incar_glance_speed_elasticity",),
    "held_out",
    "jokinen2021.speed_effect",
    ("kujala_saariluoma2011.mean_and_sd",),
)
_declare(
    ("design.touch_key_asymptote",),
    "held_out",
    "kim2014.key_size",
    ("bi2013.touch_pointing",),
)
_declare(
    ("response.modality_order",),
    "consistency",
    "diederich_colonius2004.modality",
    ("diederich_colonius2015.twin", "diederich_colonius2004.tactile_latency"),
    "The tactile latency uses the modality difference from this evaluation study.",
)
_declare(
    ("response.eccentricity_rmse",),
    "held_out",
    "diederich_colonius2007.eccentricity",
    ("osaka1976.visual_latency",),
)
_declare(
    ("response.drt_visual", "response.drt_driving_cost"),
    "held_out",
    "conti2014.driving_baseline",
    ("conti2014.secondary_task_costs",),
    "Secondary task costs were fitted as contrasts against this study's "
    "observed driving baseline. Baseline and driving cost were not separate "
    "fit targets, but their observations are shared with the calibration contrasts.",
)
_declare(
    (
        "response.drt_tactile",
        "response.drt_2back_cost",
        "response.drt_surt_cost",
        "response.older_slowing",
    ),
    "held_out",
    "cooper2020.tactile_drt",
    ("conti2014.secondary_task_costs", "diederich2008.age_parameters"),
)
_declare(
    ("vestibular.vertical_msi_error",),
    "calibration",
    "ohanlon_mccauley1974.vertical_endpoints",
    ("ohanlon_mccauley1974.vertical_endpoints",),
    "Conflict scale was fitted to these endpoint conditions.",
)
_declare(
    ("vestibular.time_course_msi_error",),
    "held_out",
    "mccauley1976.time_course",
    ("ohanlon_mccauley1974.vertical_endpoints",),
)
_declare(
    ("vestibular.roll_frequency_order", "vestibular.roll_compensation_effect"),
    "held_out",
    "donohew_griffin2009.roll",
    ("ohanlon_mccauley1974.vertical_endpoints",),
)
_declare(
    ("fatigue.endurance_error",),
    "consistency",
    "frey_law_avin2010.endurance",
    ("frey_law_avin2010.endurance",),
    "Published fatigue rates were fitted to these endurance curves.",
)
_declare(
    ("risk.lane_width_speed_error",),
    "held_out",
    "liu2016.widths_2.85_3.25_3.75",
    ("liu2016.widths_3.0_3.5",),
    "Conditions share the study but use different lane widths.",
)
_declare(
    ("trust.study2_rmse", "trust.malfunction_dip"),
    "held_out",
    "kraus2020.study2",
    ("kraus2020.study1",),
)
_declare(
    ("affect.emotion_rmse",),
    "unclassified",
    "zhang2024.experiment1.ratings",
    ("zhang2024.model_settings",),
    "Settings and evaluation share a publication. Independent data use is unverified.",
)
_declare(
    ("driving.takeover_time",),
    "held_out",
    "zhang2019.meta_analysis",
    ("gold2013.hands", "hergeth2016.monitoring"),
    "The meta analysis may include component studies used for calibration.",
)
_declare(
    ("driving.takeover_modality_order",),
    "held_out",
    "petermeijer2017.modality",
    ("gold2013.hands",),
)
_declare(
    ("driving.takeover_hands_auditory",),
    "held_out",
    "ko2022.hands",
    ("gold2013.hands",),
)
_declare(
    ("driving.takeover_hands",),
    "calibration",
    "gold2013.hands",
    ("gold2013.hands",),
    "Hand release time was fitted to these outcomes.",
)
_declare(
    ("driving.takeover_gaze", "driving.takeover_road", "driving.takeover_intervention"),
    "held_out",
    "gold2013.gaze_and_intervention",
    ("gold2013.hands",),
    "Other outcomes from the same study; participants are shared.",
)
_declare(
    (
        "driving.zeeb_road",
        "driving.zeeb_hands",
        "driving.automation_max_offroad",
        "driving.automation_road_share",
    ),
    "held_out",
    "zeeb2015.takeover_and_monitoring",
    ("gold2013.hands", "hergeth2016.monitoring"),
)
_declare(
    (
        "driving.zhang_no_auditory",
        "driving.zhang_vibrotactile",
        "driving.zhang_visual_task",
    ),
    "held_out",
    "zhang2019.meta_analysis_effects",
    ("gold2013.hands",),
    "The meta analysis may include calibration studies.",
)
_declare(
    ("driving.petermeijer_visual_vs_auditory", "driving.petermeijer_visual_vs_tactile"),
    "held_out",
    "petermeijer2017.eyes",
    ("gold2013.hands",),
)

# These tasks do not match the simulated SuRT protocol. Preserve their
# original scores as exploratory diagnostics until task matched protocols
# are implemented. The default driver's mechanisms retain their data roles.
for _metric in (
    "driving.occlusion_speed_elasticity", "driving.incar_glance_speed_elasticity",
):
    _previous = _DECLARATIONS[_metric]
    _DECLARATIONS[_metric] = replace(
        _previous,
        role="exploratory",
        rationale=(
            _previous.rationale + " Simulated SuRT conditions differ from the "
            "source tasks. These scores are diagnostic comparisons. No numerical "
            "acceptance range was changed."
        ).strip(),
    )


def evaluation_for(metric: str) -> Evaluation:
    """Return an explicit unclassified role for undeclared metrics."""
    return _DECLARATIONS.get(
        metric,
        Evaluation(
            "unclassified",
            (),
            rationale="No evaluation data declaration is registered.",
        ),
    )


def validation_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Count scored metrics by role, keeping held out scores separate."""
    totals: dict[str, Counter] = {
        role: Counter()
        for role in (
            "consistency", "calibration", "held_out", "exploratory", "unclassified",
        )
    }
    for result in results:
        if result.get("skipped"):
            continue
        for metric, score in result.get("scores", {}).items():
            role = (
                result.get("validation", {}).get(metric, {}).get("role", "unclassified")
            )
            totals[role]["metrics"] += 1
            totals[role]["passed"] += int(score == 1.0)
    return {
        role: {"metrics": count["metrics"], "passed": count["passed"]}
        for role, count in totals.items()
    }
