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
