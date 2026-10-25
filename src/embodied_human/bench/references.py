"""Human reference values the benchmark compares modules against.

Only aggregate published values are stored here, each with its source and
conditions. A reference has an acceptance range: published ranges where
studies disagree, otherwise a stated tolerance around the value.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Reference:
    """One human reference.

    Args:
        name: Identifier, ``category.quantity``.
        value: Central value.
        low, high: Acceptance range.
        unit: Unit of value and range.
        citation: Short citation(s).
        doi: DOI or URL.
        conditions: Task and sample.
    """

    name: str
    value: float
    low: float
    high: float
    unit: str
    citation: str
    doi: str | None
    conditions: str

    def accepts(self, x: float) -> bool:
        """Whether ``x`` lies in the acceptance range."""
        return self.low <= x <= self.high

    def score(self, x: float) -> float:
        """1 inside the range, falling linearly to 0 one range-width outside."""
        if self.accepts(x):
            return 1.0
        width = max(self.high - self.low, 1e-12)
        distance = self.low - x if x < self.low else x - self.high
        return max(0.0, 1.0 - distance / width)


REFERENCES: dict[str, Reference] = {
    r.name: r
    for r in (
        Reference(
            "eyes.duration_slope",
            2.2,
            2.0,
            3.7,
            "ms/deg",
            "Carpenter (1988) 2.2; Baloh et al. (1975) 2.7; other fits up to 3.66",
            "https://doi.org/10.1212/wnl.25.11.1065",
            "horizontal visually guided saccades in healthy adults",
        ),
        Reference(
            "eyes.duration_intercept",
            21.0,
            13.0,
            37.0,
            "ms",
            "Carpenter (1988) 21 ms; other fits 14 ms",
            None,
            "duration of a vanishing saccade from linear main-sequence fits",
        ),
        Reference(
            "eyes.latency_median",
            200.0,
            150.0,
            250.0,
            "ms",
            "Saslow (1967)",
            "https://doi.org/10.1364/JOSA.57.001024",
            "step target: 150 ms with a gap, 200 ms simultaneous, 250 ms with overlap",
        ),
        Reference(
            "objects.e2",
            2.0,
            1.0,
            2.6,
            "deg",
            "Strasburger, Rentschler & Jüttner (2011), Table 4",
            "https://doi.org/10.1167/11.5.13",
            "letter and Landolt-C acuity; vernier 0.7, gratings 2.6-3.0",
        ),
        Reference(
            "vision.e2",
            2.3,
            1.0,
            3.0,
            "deg",
            "Strasburger et al. (2011), Table 4; Geisler & Perry (1998) 2.3",
            "https://doi.org/10.1167/11.5.13",
            "eccentricity halving the resolvable frequency; letters to gratings",
        ),
        Reference(
            "memory.span",
            4.0,
            3.0,
            5.0,
            "items",
            "Cowan (2001)",
            "https://doi.org/10.1017/S0140525X01003922",
            "capacity with rehearsal and chunking prevented",
        ),
        Reference(
            "motor.touch_throughput",
            6.95,
            5.5,
            8.5,
            "bits/s",
            "MacKenzie (2015)",
            "https://doi.org/10.1007/978-3-319-20916-6_23",
            "smartphone touch, 16 participants; method differences between studies",
        ),
        Reference(
            "motor.touch_slope",
            105.0,
            75.0,
            135.0,
            "ms/bit",
            "Bi, Li & Zhai (2013), FFitts 2-D fit",
            "https://doi.org/10.1145/2470654.2466180",
            "2-D finger touch; acceptance +-30%",
        ),
        Reference(
            "driving.mean_offroad_glance",
            2.0,
            0.0,
            2.0,
            "s",
            "NHTSA (2013) guideline criterion (b)",
            "https://www.federalregister.gov/documents/2013/04/26/2013-09883",
            "mean glance away from the road per task; a "
            "design criterion, not a human mean",
        ),
        Reference(
            "driving.long_glance_share",
            0.15,
            0.0,
            0.15,
            "fraction",
            "NHTSA (2013) guideline criterion (a)",
            "https://www.federalregister.gov/documents/2013/04/26/2013-09883",
            "share of off-road glances longer than 2 s",
        ),
        Reference(
            "driving.human_mean_glance",
            1.1,
            0.8,
            1.2,
            "s",
            "Kujala & Saariluoma (2011), Table 2; Jokinen, Kujala & Oulasvirta "
            "(2021), Appendix B",
            "https://doi.org/10.1080/00140139.2011.592601",
            "simulator, in-vehicle visual search; means 1.06-1.20 s over four "
            "interfaces (transitions included) and 1.0 +- 0.2 s by condition",
        ),
        Reference(
            "driving.human_long_glance_share",
            0.08,
            0.03,
            0.13,
            "fraction",
            "Kujala & Saariluoma (2011); Tivesten & Dozza (2014)",
            "https://doi.org/10.1016/j.trf.2014.08.004",
            "glances over 2 s: 5.7-10% in simulator search tasks; naturalistic "
            "phone tasks median 3.4%, mean 12.8%",
        ),
        Reference(
            "driving.human_glance_sd",
            0.57,
            0.48,
            0.66,
            "s",
            "Kujala & Saariluoma (2011), Table 2",
            "https://doi.org/10.1080/00140139.2011.592601",
            "mean within-person SD of in-vehicle glance durations over four "
            "interfaces, simulator at 40-60 km/h, n = 10 per interface",
        ),
        Reference(
            "driving.occlusion_speed_elasticity",
            -0.64,
            -0.72,
            -0.53,
            "1",
            "Senders et al. (1967), Table 5 (elasticities derived here)",
            "https://onlinepubs.trb.org/Onlinepubs/hrr/1967/195/195-002.pdf",
            "voluntary occlusion on a freeway at 22-60 mph, three drivers; "
            "log-log slope of occlusion time on speed per driver",
        ),
        Reference(
            "driving.incar_glance_speed_elasticity",
            0.0,
            -0.15,
            0.15,
            "1",
            "Jokinen, Kujala & Oulasvirta (2021), Appendix B, Table 6",
            "https://doi.org/10.1177/0018720820927687",
            "in-car search glances at 60 vs 120 km/h: speed coefficient -0.0 s "
            "(n.s.) at a 1.0 s intercept, 12 drivers; the range (+-0.1 s per "
            "doubling of speed) is an assumed tolerance",
        ),
        Reference(
            "design.touch_key_asymptote",
            17.5,
            12.5,
            22.5,
            "mm",
            "Kim, Kwon, Heo, Lee & Chung (2014)",
            "https://doi.org/10.1016/j.apergo.2013.05.006",
            "5-digit entry on in-vehicle touch keys of 7.5-27.5 mm in 5 mm "
            "steps, 30 drivers; usability and safety stopped improving at "
            "17.5 mm; the range spans the neighbouring sizes",
        ),
        Reference(
            "response.modality_order",
            1.0,
            1.0,
            1.0,
            "fraction",
            "Diederich & Colonius (2004), Table 2",
            "https://doi.org/10.3758/BF03195006",
            "simple RT, same 4 people: tone (90 dB) 132 ms < light 163 ms < "
            "strong vibration 177 ms; share of the three orderings reproduced",
        ),
        Reference(
            "response.drt_visual",
            344.0,
            326.0,
            362.0,
            "ms",
            "Conti, Dlugosch & Bengler (2014), Table 3",
            "https://www.hfes-europe.org/wp-content/uploads/2014/06/Conti.pdf",
            "head-mounted visual DRT while driving a simulator, n = 71; range "
            "over instruction groups",
        ),
        Reference(
            "response.drt_tactile",
            526.0,
            482.0,
            570.0,
            "ms",
            "Cooper et al. (2020), Table 4",
            "https://doi.org/10.3389/fpsyg.2020.01154",
            "tactile DRT on-road, drivers aged 21-36 without a secondary task, "
            "526 (SD 110) ms, n = 24; range is the 95% CI of the mean",
        ),
        Reference(
            "response.older_slowing",
            1.215,
            1.07,
            1.36,
            "ratio",
            "Cooper et al. (2020), Table 4",
            "https://doi.org/10.3389/fpsyg.2020.01154",
            "tactile DRT 639 ms (55-75 years) over 526 ms (21-36 years); range "
            "from the two means' standard errors (derived)",
        ),
        Reference(
            "response.drt_driving_cost",
            50.7,
            31.4,
            70.0,
            "ms",
            "Conti, Dlugosch & Bengler (2014), Table 3",
            "https://www.hfes-europe.org/wp-content/uploads/2014/06/Conti.pdf",
            "head-mounted visual DRT while driving minus alone, pooled over "
            "four groups (N = 71); range is a 95% CI from the SDs, treating "
            "the conditions as independent (derived, conservative)",
        ),
        Reference(
            "response.drt_2back_cost",
            234.0,
            152.5,
            315.5,
            "ms",
            "Cooper et al. (2020), Table 4",
            "https://doi.org/10.3389/fpsyg.2020.01154",
            "DRT with an auditory 2-back minus driving only, drivers aged "
            "21-36 on the road, tactile and light responses pooled; range is "
            "a 95% CI from the SDs with n = 24 (derived)",
        ),
        Reference(
            "response.drt_surt_cost",
            177.0,
            104.3,
            249.7,
            "ms",
            "Cooper et al. (2020), Table 4",
            "https://doi.org/10.3389/fpsyg.2020.01154",
            "DRT with SuRT on an iPad minus driving only, as above",
        ),
        Reference(
            "driving.takeover_modality_order",
            1.0,
            1.0,
            1.0,
            "fraction",
            "Petermeijer, Doubek & de Winter (2017), abstract",
            "https://doi.org/10.1109/SMC.2017.8122827",
            "101 drivers: auditory and tactile take-over requests were "
            "answered faster than visual ones; share of the two orderings "
            "reproduced",
        ),
        Reference(
            "driving.takeover_hands_auditory",
            1.87,
            1.36,
            2.38,
            "s",
            "Ko, Sanghavi, Zhang & Jeon (2022), Fig. 4",
            "https://doi.org/10.1016/j.trf.2022.08.008",
            "time to grab the wheel after an auditory warning with a visual "
            "text, game on the centre console, laps 2-3, n = 21; range is "
            "+-1 between-person SD (0.51 s)",
        ),
        Reference(
            "vestibular.vertical_msi_error",
            0.0,
            0.0,
            10.0,
            "% points",
            "O'Hanlon & McCauley (1973/1974), Tables 1-2",
            "https://archive.org/details/DTIC_AD0768215",
            "RMS error of MSI after 2 h of vertical oscillation, 14 conditions "
            "at 0.083-0.5 Hz; acceptance is an assumed tolerance (the "
            "descriptive fit of McCauley et al. 1976 reached 6.1)",
        ),
        Reference(
            "vestibular.time_course_msi_error",
            0.0,
            0.0,
            10.0,
            "% points",
            "McCauley, Royal, Wylie, O'Hanlon & Mackie (1976), Appendix B",
            "https://archive.org/details/DTIC_ADA024709",
            "cumulative MSI at 5-115 min, 43 points at nine vertical conditions "
            "(0.18-0.7 Hz) outside the 2-hour set; the authors' own descriptive "
            "fit has an RMS error of 6.1 over all 99 points; acceptance assumed",
        ),
        Reference(
            "fatigue.endurance_error",
            0.0,
            0.0,
            0.25,
            "|ln ratio|",
            "Frey-Law & Avin (2010), Table 2",
            "https://doi.org/10.1080/00140130903389068",
            "median log ratio of endurance times to the joint-specific power "
            "models at 20-80% MVC; acceptance (about 28%) assumed",
        ),
        Reference(
            "risk.lane_width_speed_error",
            0.0,
            0.0,
            9.4,
            "km/h",
            "Liu, Wang & Fu (2016), Table 2",
            "https://doi.org/10.3390/ijerph13101010",
            "simulator speeds at lane widths 2.85-3.75 m, n = 24; acceptance is "
            "the mean between-driver SD",
        ),
        Reference(
            "trust.study2_rmse",
            0.0,
            0.0,
            0.3,
            "scale points",
            "Kraus, Scholz, Stiegemeier & Baumann (2020), Table 7",
            "https://doi.org/10.1177/0018720819853686",
            "trust (1-7) at t0-t8 in the two low-transparency groups (n = 13, "
            "11), predicted after fitting Study 1; acceptance is about one "
            "standard error of a group mean (assumed)",
        ),
        Reference(
            "trust.malfunction_dip",
            -0.47,
            -0.94,
            -0.10,
            "scale points",
            "Kraus, Scholz, Stiegemeier & Baumann (2020), Table 7",
            "https://doi.org/10.1177/0018720819853686",
            "change in trust from t3 to t4 across the malfunction, low "
            "transparency (n = 11); the dip recovered by t5. Range: a dip of "
            "at least 0.1 and at most twice the observed one (assumed)",
        ),
        Reference(
            "affect.emotion_rmse",
            0.0,
            0.0,
            0.13,
            "intensity",
            "Zhang, Hilpert, Broekens & Jokinen (2024), Experiment 1",
            "https://doi.org/10.1145/3613904.3641908",
            "normalised happiness, boredom and irritation ratings in three "
            "tasks, n = 27; acceptance is the original model's RMSE",
        ),
        Reference(
            "driving.takeover_time",
            2.72,
            1.27,
            4.17,
            "s",
            "Zhang, de Winter, Varotto, Happee & Martens (2019)",
            "https://doi.org/10.1016/j.trf.2019.04.020",
            "mean of 520 study-condition means, SD 1.45 s; range is mean +- 1 SD",
        ),
    )
}
