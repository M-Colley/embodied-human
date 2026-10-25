"""Touch: whether a vibration is felt, and how quickly.

Baseline :class:`VibrotactileTouch` compares the r.m.s. acceleration of a
vibration with the human perception threshold at its frequency and body
site, but only while the body touches the vibrating surface (embodied-ui's
``InterfaceRuntime.vibration`` returns zero otherwise). Thresholds:

- ``hand_grip``: a hand gripping a 32 mm cylinder, vibration along the
  axis (Brisben, Hsiao and Johnson, 1999); converted from displacement,
  ``a_rms = (2 pi f)^2 d / sqrt(2)``.
- ``hand_lateral`` and ``seat_lateral``: a hand on a 30 mm handlebar and a
  seated body on a rigid seat, lateral vibration, median thresholds
  (Morioka and Griffin, 2006).
- ``hand_plate`` and ``seat_vertical``: a hand pressing a plate and a body
  seated on a rigid seat, vertical vibration (Miwa, 1967).

The sensation level is the acceleration's level above threshold. Its
latency follows the same Piéron form as hearing; no Piéron fit of
vibrotactile intensity exists, so the gain is set from the tactile
reaction times of Diederich and Colonius (2004) under an assumed mapping
of their weakest and strongest stimuli to 10 and 30 dB above threshold.
Full-body touch with sensor density per body part, as in MIMo (Lopez et
al., 2024), and afferent-level models such as TouchSim (Saal et al.,
2017) are natural replacements.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from .core import Module, Parameters, Source, register

#: Perception thresholds, (frequency Hz, r.m.s. acceleration m/s^2).
THRESHOLDS: dict[str, tuple[tuple[float, float], ...]] = {
    "hand_grip": (
        (10, 0.039),
        (20, 0.063),
        (30, 0.035),
        (40, 0.030),
        (60, 0.028),
        (100, 0.020),
        (150, 0.019),
        (200, 0.034),
        (300, 0.21),
    ),
    "hand_lateral": (
        (8, 0.066),
        (10, 0.079),
        (12.5, 0.111),
        (16, 0.096),
        (20, 0.120),
        (25, 0.094),
        (31.5, 0.085),
        (40, 0.057),
        (50, 0.046),
        (63, 0.044),
        (80, 0.026),
        (100, 0.032),
        (125, 0.037),
        (160, 0.031),
        (200, 0.038),
        (250, 0.045),
        (315, 0.106),
    ),
    "seat_lateral": (
        (2, 0.010),
        (2.5, 0.011),
        (3.15, 0.014),
        (4, 0.017),
        (5, 0.021),
        (6.3, 0.024),
        (8, 0.033),
        (10, 0.053),
        (12.5, 0.072),
        (16, 0.075),
        (20, 0.079),
        (25, 0.077),
        (31.5, 0.068),
        (40, 0.070),
        (50, 0.056),
        (63, 0.048),
        (80, 0.051),
        (100, 0.051),
        (125, 0.053),
        (160, 0.094),
        (200, 0.109),
        (250, 0.227),
        (315, 0.319),
    ),
    "hand_plate": (
        (2, 0.0098),
        (3, 0.015),
        (5, 0.025),
        (10, 0.046),
        (20, 0.062),
        (30, 0.041),
        (56, 0.034),
        (100, 0.020),
        (200, 0.045),
        (300, 0.11),
    ),
    "seat_vertical": (
        (0.5, 0.024),
        (1, 0.0075),
        (1.5, 0.0073),
        (3, 0.0062),
        (5, 0.0068),
        (7, 0.0066),
        (10, 0.0078),
        (15, 0.012),
        (20, 0.020),
        (30, 0.028),
        (50, 0.032),
        (70, 0.028),
        (100, 0.032),
        (150, 0.043),
        (200, 0.065),
        (300, 0.095),
    ),
}

THRESHOLD_SOURCES = {
    "hand_grip": Source(
        "Brisben, Hsiao & Johnson (1999), Fig. 3",
        "https://doi.org/10.1152/jn.1999.81.4.1548",
        "measured",
        "19 adults, geometric means; read from the figure, converted",
    ),
    "hand_lateral": Source(
        "Morioka & Griffin (2006), Fig. 1",
        "https://stacks.cdc.gov/view/cdc/8400",
        "measured",
        "12 males, medians",
    ),
    "seat_lateral": Source(
        "Morioka & Griffin (2006), Fig. 1",
        "https://stacks.cdc.gov/view/cdc/8400",
        "measured",
        "12 males, medians, rigid seat",
    ),
    "hand_plate": Source(
        "Miwa (1967), Part 3, Fig. 3",
        "https://doi.org/10.2486/indhealth.5.213",
        "measured",
        "10 males, read from the figure (+-1 dB)",
    ),
    "seat_vertical": Source(
        "Miwa (1967), Part 1, Fig. 7",
        "https://doi.org/10.2486/indhealth.5.183",
        "measured",
        "10 males, rigid seat without backrest, read from the figure",
    ),
}


@dataclass(frozen=True)
class TouchParameters(Parameters):
    """Vibrotactile detection latency.

    Args:
        latency_gain: Piéron gain on the amplitude ratio above threshold, s.
        latency_exponent: Piéron exponent.
        reference_latency: Mean peripheral time of a vibration 30 dB above
            threshold, s.
    """

    latency_gain: float = 0.1175
    latency_exponent: float = -0.29
    reference_latency: float = 0.1166

    SOURCES: ClassVar = {
        "latency_gain": Source(
            "Diederich & Colonius (2004), Table 2; fitted here",
            "https://doi.org/10.3758/BF03195006",
            "assumed",
            "tactile RT 218 vs 177 ms for their weakest and strongest "
            "vibration, assumed 10 and 30 dB above threshold",
        ),
        "latency_exponent": Source(
            "Kohfeld, Santee & Wallace (1981)",
            "https://doi.org/10.3758/BF03207370",
            "assumed",
            "auditory exponent; no vibrotactile fit exists",
        ),
        "reference_latency": Source(
            "Diederich & Colonius (2004, 2015); derived",
            "https://doi.org/10.3758/BF03195006",
            "derived",
            "visual 1/lambda_V of 102.6 ms (2015) plus the 14 ms by which "
            "strong tactile RT exceeded visual RT (2004)",
        ),
    }


class Touch(Module):
    """Category ``touch``: feel contact and vibration."""

    category = "touch"

    def sensation_level(
        self, amplitude: float, frequency: float, site: str, contact: bool = True
    ) -> float:
        """dB by which an r.m.s. acceleration exceeds the threshold."""
        raise NotImplementedError

    def latency(self, sensation_level: float) -> float:
        """Mean peripheral processing time of a felt vibration, s."""
        raise NotImplementedError


@register("touch", "vibrotactile")
class VibrotactileTouch(Touch):
    """Frequency- and site-dependent vibration thresholds."""

    def __init__(self, **params: float) -> None:
        super().__init__()
        self.params = TouchParameters(**params)

    def threshold(self, frequency: float, site: str) -> float:
        """Perception threshold, m/s^2 r.m.s., interpolated in log-log."""
        try:
            points = np.array(THRESHOLDS[site])
        except KeyError:
            raise ValueError(
                f"no thresholds for {site!r}; known: {', '.join(THRESHOLDS)}"
            ) from None
        return float(
            10
            ** np.interp(
                math.log10(frequency), np.log10(points[:, 0]), np.log10(points[:, 1])
            )
        )

    def sensation_level(
        self, amplitude: float, frequency: float, site: str, contact: bool = True
    ) -> float:
        if not contact or amplitude <= 0:
            return -math.inf
        return 20 * math.log10(amplitude / self.threshold(frequency, site))

    def latency(self, sensation_level: float) -> float:
        p = self.params

        def pieron(sl: float) -> float:
            return p.latency_gain * (10 ** (sl / 20)) ** p.latency_exponent

        if sensation_level <= 0:
            return math.inf
        return max(0.005, p.reference_latency + pieron(sensation_level) - pieron(30.0))
