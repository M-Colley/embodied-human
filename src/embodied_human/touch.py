"""Touch: whether a vibration is felt, and how quickly.

Baseline :class:`VibrotactileTouch` compares the r.m.s. acceleration of a
vibration with the human perception threshold at its frequency and body
site, but only while the body touches the vibrating surface (embodied-ui's
``InterfaceRuntime.vibration`` returns zero otherwise). Thresholds are
medians of Morioka and Griffin's young male participants:

- ``{hand,seat,foot}_{fore_aft,lateral,vertical}``: the hand gripping a
  30 mm handle, the body on a rigid seat without backrest and the foot on
  a footrest, each in three axes (2008, Table 2; 2-315 Hz at the seat,
  8-315 Hz elsewhere; yes-no staircase). The lateral axis runs along the
  handle.
- ``fingertip``, ``palm`` and ``grip``: vertical vibration of a 6 mm probe
  under the middle fingertip, a flat plate under the palm and a 30 mm
  handle (2005, Fig. 2; 8-500 Hz; forced choice). Two-interval forced
  choice gives lower thresholds than yes-no, so compare sites within one
  study.

Below the lowest frequency of a site the threshold is held constant.

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
    "hand_fore_aft": (
        (8, 0.053),
        (10, 0.047),
        (12.5, 0.05),
        (16, 0.059),
        (20, 0.064),
        (25, 0.051),
        (31.5, 0.047),
        (40, 0.048),
        (50, 0.042),
        (63, 0.041),
        (80, 0.038),
        (100, 0.031),
        (125, 0.025),
        (160, 0.026),
        (200, 0.029),
        (250, 0.042),
        (315, 0.088),
    ),
    "hand_lateral": (
        (8, 0.067),
        (10, 0.079),
        (12.5, 0.113),
        (16, 0.097),
        (20, 0.121),
        (25, 0.095),
        (31.5, 0.085),
        (40, 0.058),
        (50, 0.046),
        (63, 0.045),
        (80, 0.026),
        (100, 0.032),
        (125, 0.037),
        (160, 0.031),
        (200, 0.038),
        (250, 0.045),
        (315, 0.107),
    ),
    "hand_vertical": (
        (8, 0.094),
        (10, 0.098),
        (12.5, 0.095),
        (16, 0.126),
        (20, 0.152),
        (25, 0.14),
        (31.5, 0.134),
        (40, 0.069),
        (50, 0.074),
        (63, 0.048),
        (80, 0.027),
        (100, 0.023),
        (125, 0.022),
        (160, 0.022),
        (200, 0.031),
        (250, 0.036),
        (315, 0.062),
    ),
    "seat_fore_aft": (
        (2, 0.012),
        (2.5, 0.013),
        (3.15, 0.013),
        (4, 0.013),
        (5, 0.013),
        (6.3, 0.014),
        (8, 0.025),
        (10, 0.041),
        (12.5, 0.057),
        (16, 0.086),
        (20, 0.084),
        (25, 0.087),
        (31.5, 0.071),
        (40, 0.077),
        (50, 0.075),
        (63, 0.075),
        (80, 0.077),
        (100, 0.089),
        (125, 0.089),
        (160, 0.125),
        (200, 0.171),
        (250, 0.252),
        (315, 0.436),
    ),
    "seat_lateral": (
        (2, 0.01),
        (2.5, 0.012),
        (3.15, 0.014),
        (4, 0.017),
        (5, 0.021),
        (6.3, 0.024),
        (8, 0.033),
        (10, 0.054),
        (12.5, 0.072),
        (16, 0.076),
        (20, 0.079),
        (25, 0.077),
        (31.5, 0.069),
        (40, 0.07),
        (50, 0.056),
        (63, 0.049),
        (80, 0.051),
        (100, 0.051),
        (125, 0.054),
        (160, 0.093),
        (200, 0.109),
        (250, 0.231),
        (315, 0.319),
    ),
    "seat_vertical": (
        (2, 0.014),
        (2.5, 0.016),
        (3.15, 0.018),
        (4, 0.018),
        (5, 0.015),
        (6.3, 0.015),
        (8, 0.019),
        (10, 0.022),
        (12.5, 0.022),
        (16, 0.025),
        (20, 0.025),
        (25, 0.028),
        (31.5, 0.03),
        (40, 0.027),
        (50, 0.025),
        (63, 0.025),
        (80, 0.026),
        (100, 0.025),
        (125, 0.032),
        (160, 0.027),
        (200, 0.033),
        (250, 0.044),
        (315, 0.065),
    ),
    "foot_fore_aft": (
        (8, 0.065),
        (10, 0.062),
        (12.5, 0.066),
        (16, 0.064),
        (20, 0.064),
        (25, 0.05),
        (31.5, 0.041),
        (40, 0.038),
        (50, 0.038),
        (63, 0.037),
        (80, 0.027),
        (100, 0.025),
        (125, 0.024),
        (160, 0.031),
        (200, 0.041),
        (250, 0.096),
        (315, 0.292),
    ),
    "foot_lateral": (
        (8, 0.076),
        (10, 0.077),
        (12.5, 0.077),
        (16, 0.07),
        (20, 0.076),
        (25, 0.076),
        (31.5, 0.061),
        (40, 0.053),
        (50, 0.049),
        (63, 0.039),
        (80, 0.035),
        (100, 0.047),
        (125, 0.049),
        (160, 0.055),
        (200, 0.072),
        (250, 0.124),
        (315, 0.256),
    ),
    "foot_vertical": (
        (8, 0.049),
        (10, 0.044),
        (12.5, 0.048),
        (16, 0.04),
        (20, 0.043),
        (25, 0.038),
        (31.5, 0.03),
        (40, 0.03),
        (50, 0.033),
        (63, 0.032),
        (80, 0.031),
        (100, 0.026),
        (125, 0.029),
        (160, 0.037),
        (200, 0.064),
        (250, 0.077),
        (315, 0.189),
    ),
    "fingertip": (
        (8, 0.0156),
        (16, 0.0244),
        (31.5, 0.0516),
        (63, 0.0652),
        (125, 0.0553),
        (250, 0.0949),
        (500, 0.637),
    ),
    "palm": (
        (8, 0.0848),
        (16, 0.0786),
        (31.5, 0.0339),
        (63, 0.0255),
        (125, 0.0184),
        (250, 0.0307),
        (500, 0.138),
    ),
    "grip": (
        (8, 0.093),
        (16, 0.079),
        (31.5, 0.0494),
        (63, 0.0206),
        (125, 0.0142),
        (250, 0.0244),
        (500, 0.0848),
    ),
}

_MG_2008 = (
    "Morioka & Griffin (2008), Table 2",
    "https://doi.org/10.1016/j.jsv.2007.12.036",
)
_MG_2005 = (
    "Morioka & Griffin (2005), Fig. 2",
    "https://doi.org/10.1080/08990220500420400",
)
THRESHOLD_SOURCES = {
    **{
        site: Source(
            *_MG_2008,
            "measured",
            "medians of 12 males aged 20-29 per axis and site; 3-down 1-up "
            "yes-no staircase",
        )
        for site in THRESHOLDS
        if site.split("_")[0] in ("hand", "seat", "foot")
    },
    **{
        site: Source(
            *_MG_2005,
            "measured",
            f"medians of 12 males aged 22-27; {detail}; 3-down 1-up two-"
            "interval forced choice; read from the vector figure",
        )
        for site, detail in (
            ("fingertip", "6 mm probe in a 10 mm surround, 1 N"),
            ("palm", "200 x 150 mm plate, 10 N"),
            ("grip", "30 mm handle, 10 N; 8 and 16 Hz markers overlap palm"),
        )
    },
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
