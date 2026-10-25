"""Hearing: whether a sound is heard over noise, and how quickly.

Baseline :class:`PsychoacousticHearing`:

- **Threshold in quiet** from ISO 226:2003 (free field, young listeners),
  with the 20 Hz value of ISO 226:2023.
- **Masking** by the power-spectrum model: a tone is detected when its
  level exceeds the noise power in the critical band around it by
  ``10 log10 K`` dB, with K = 0.4 (Scharf, 1970) and the critical
  bandwidth of Zwicker and Terhardt (1980). The sensation level is the
  tone's level above the higher of the two thresholds.
- **Cabin noise** as a third-octave spectrum (energy mostly below 250 Hz,
  falling about 20 dB to 1.3 kHz and 6 dB/octave above 2 kHz) scaled to
  an A-weighted level that grows with speed.
- **Latency** of the auditory channel follows Piéron's law,
  ``k p^n`` with sound pressure ``p`` above threshold, as fitted by
  Kohfeld, Santee and Wallace (1981) to Chocholle's (1940) reaction times
  at 10-100 dB, anchored at a manual-response latency for loud tones.

ISO 7731 audibility of warnings (:func:`iso7731_audible`) and loudness of
waveforms with MoSQITo (:func:`loudness`, optional dependency) are
available for design checks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from .core import Module, Parameters, Source, register

#: Third-octave centre frequencies, Hz.
THIRD_OCTAVES = np.array(
    [
        20,
        25,
        31.5,
        40,
        50,
        63,
        80,
        100,
        125,
        160,
        200,
        250,
        315,
        400,
        500,
        630,
        800,
        1000,
        1250,
        1600,
        2000,
        2500,
        3150,
        4000,
        5000,
        6300,
        8000,
        10000,
        12500,
    ],
    dtype=float,
)

#: Threshold of hearing Tf, dB SPL (ISO 226:2003 Table 1; 20 Hz from the
#: 2023 edition), as reproduced by the IoSR-Surrey toolbox and Suzuki,
#: Takeshima and Kurakata (2024).
THRESHOLD_IN_QUIET = np.array(
    [
        78.1,
        68.7,
        59.5,
        51.1,
        44.0,
        37.5,
        31.5,
        26.5,
        22.1,
        17.9,
        14.4,
        11.4,
        8.6,
        6.2,
        4.4,
        3.0,
        2.2,
        2.4,
        3.5,
        1.7,
        -1.3,
        -4.2,
        -6.0,
        -5.4,
        -1.5,
        6.0,
        12.6,
        13.9,
        12.3,
    ]
)

#: Reference pressure of 0 dB SPL in dyn/cm^2 (Kohfeld et al.'s unit).
_P0 = 0.0002


def a_weighting(frequency: NDArray[np.float64] | float) -> NDArray[np.float64]:
    """A-weighting in dB (IEC 61672-1)."""
    f2 = np.asarray(frequency, dtype=float) ** 2
    ra = (12194.0**2 * f2**2) / (
        (f2 + 20.6**2) * np.sqrt((f2 + 107.7**2) * (f2 + 737.9**2)) * (f2 + 12194.0**2)
    )
    return 20 * np.log10(ra) + 2.0


def critical_bandwidth(frequency: float) -> float:
    """Critical bandwidth in Hz (Zwicker and Terhardt, 1980)."""
    khz = frequency / 1000.0
    return 25.0 + 75.0 * (1.0 + 1.4 * khz**2) ** 0.69


def cabin_noise_spectrum(level_dba: float) -> NDArray[np.float64]:
    """Third-octave band levels (dB SPL) of car interior noise whose
    A-weighted total is ``level_dba``.

    The shape follows the summary in 3GPP TR 43.058, 4.1.1.2.1: flat to
    250 Hz, a 20 dB fall from 250 Hz to 1.3 kHz, then 6 dB/octave above
    2 kHz (between 1.3 and 2 kHz the slope continues).
    """
    octaves = np.log2(THIRD_OCTAVES / 250.0)
    fall = 20.0 / math.log2(1300.0 / 250.0)
    shape = np.where(THIRD_OCTAVES <= 250.0, 0.0, -fall * octaves)
    above = THIRD_OCTAVES > 2000.0
    shape[above] = -fall * math.log2(2000.0 / 250.0) - 6.0 * np.log2(
        THIRD_OCTAVES[above] / 2000.0
    )
    total = 10 * np.log10(np.sum(10 ** ((shape + a_weighting(THIRD_OCTAVES)) / 10)))
    return shape + (level_dba - total)


def cabin_noise_level(speed_kmh: float) -> float:
    """A-weighted interior noise of a combustion car, dB(A).

    A fit, 64.6 + 28.5 log10(v / 100 km/h), through three recordings:
    53.5 and 63.7 dB(A) at 40 and 100 km/h (Li, Parizet & Colangeli,
    2026, preprint) and about 68.6 dB(A) at 130 km/h (ETSI ES 202 396-1
    full-size car).
    """
    return 64.6 + 28.5 * math.log10(max(speed_kmh, 1.0) / 100.0)


def iso7731_audible(signal_dba: float, noise_dba: float) -> bool:
    """ISO 7731:2003 4.2.2.1-4.2.2.2: at least 65 dB(A) and more than
    15 dB(A) above the ambient noise."""
    return signal_dba >= 65.0 and signal_dba - noise_dba > 15.0


def loudness(pressure: NDArray[np.float64], rate: float) -> float:
    """Loudness in sone of a stationary sound pressure signal (Pa) by
    ISO 532-1 (Zwicker), computed with MoSQITo."""
    from mosqito.sq_metrics import loudness_zwst

    value, _, _ = loudness_zwst(np.asarray(pressure, dtype=float), rate)
    return float(value)


@dataclass(frozen=True)
class HearingParameters(Parameters):
    """Detection and latency of tones.

    Args:
        masking_ratio: K, tone power over noise power in the critical
            band at masked threshold.
        latency_gain: k of Piéron's law, s per (dyn/cm^2)^n.
        latency_exponent: n of Piéron's law.
        reference_latency: Mean peripheral auditory time of a tone 60 dB
            above threshold in a manual response task, s; Piéron's law
            shifts it for other levels.
    """

    masking_ratio: float = 0.4
    latency_gain: float = 0.00731
    latency_exponent: float = -0.33
    reference_latency: float = 0.0439

    SOURCES: ClassVar = {
        "masking_ratio": Source(
            "Scharf (1970), as summarised by Yost & Shofner (2009)",
            "https://doi.org/10.1121/1.3037232",
            "measured",
            "relative to the Zwicker critical band, tones in broadband noise",
        ),
        "latency_gain": Source(
            "Kohfeld, Santee & Wallace (1981), Fig. 5b",
            "https://doi.org/10.3758/BF03207370",
            "fitted",
            "Chocholle (1940), 1 kHz, 10-100 dB SPL; k = 7.31 ms",
        ),
        "latency_exponent": Source(
            "Kohfeld, Santee & Wallace (1981), Fig. 5b",
            "https://doi.org/10.3758/BF03207370",
            "fitted",
            "-0.33 for Chocholle's data; -0.24 to -0.33 over six data sets",
        ),
        "reference_latency": Source(
            "Diederich & Colonius (2015), Table 1",
            "https://doi.org/10.1037/a0038696",
            "fitted",
            "median 1/lambda_A of 11 participants (TWIN fit)",
        ),
    }


class Hearing(Module):
    """Category ``hearing``: detect sounds over noise."""

    category = "hearing"

    def sensation_level(
        self, level: float, frequency: float, noise_dba: float | None = None
    ) -> float:
        """dB by which a tone exceeds its threshold (negative: inaudible)."""
        raise NotImplementedError

    def latency(self, sensation_level: float) -> float:
        """Mean peripheral processing time of an audible tone, s."""
        raise NotImplementedError


@register("hearing", "psychoacoustic")
class PsychoacousticHearing(Hearing):
    """Threshold in quiet, power-spectrum masking and Piéron latency."""

    def __init__(self, **params: float) -> None:
        super().__init__()
        self.params = HearingParameters(**params)

    def quiet_threshold(self, frequency: float) -> float:
        """Threshold in quiet, dB SPL, interpolated in log frequency."""
        return float(
            np.interp(
                math.log10(frequency), np.log10(THIRD_OCTAVES), THRESHOLD_IN_QUIET
            )
        )

    def masked_threshold(self, frequency: float, noise_dba: float) -> float:
        """Masked threshold of a tone in cabin noise, dB SPL."""
        bands = cabin_noise_spectrum(noise_dba)
        band_level = float(
            np.interp(math.log10(frequency), np.log10(THIRD_OCTAVES), bands)
        )
        # Spectrum level, then the noise power within the critical band.
        density = band_level - 10 * math.log10(0.2316 * frequency)
        in_band = density + 10 * math.log10(critical_bandwidth(frequency))
        return in_band + 10 * math.log10(self.params.masking_ratio)

    def sensation_level(
        self, level: float, frequency: float, noise_dba: float | None = None
    ) -> float:
        threshold = self.quiet_threshold(frequency)
        if noise_dba is not None:
            threshold = max(threshold, self.masked_threshold(frequency, noise_dba))
        return level - threshold

    def latency(self, sensation_level: float) -> float:
        p = self.params

        def pieron(sl: float) -> float:
            return p.latency_gain * (_P0 * 10 ** (sl / 20)) ** p.latency_exponent

        if sensation_level <= 0:
            return math.inf
        # Anchor the curve at the reference latency for a 60 dB tone.
        return max(0.005, p.reference_latency + pieron(sensation_level) - pieron(60.0))
