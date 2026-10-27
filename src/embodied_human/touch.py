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
- ``steering_wheel`` and ``steering_wheel_lower``: vertical vibration of
  a steering wheel gripped 45 deg above or below its centre (Morioka and
  Griffin, 2009, Table 2; 4-250 Hz). Their 2007 figure gives the same
  upper-position thresholds within 2%.

For the steering wheel, :meth:`VibrotactileTouch.perceived_intensity`
also gives how strong a felt vibration is, by Stevens' power law with an
additive threshold and the exponents and constants of the same table.

Below the lowest frequency of a site the threshold is held constant.

The sensation level is the acceleration's level above threshold. Its
latency follows the same Piéron form as hearing; no Piéron fit of
vibrotactile intensity exists, so the gain is set from the tactile
reaction times of Diederich and Colonius (2004) under an assumed mapping
of their weakest and strongest stimuli to 10 and 30 dB above threshold.
The optional :class:`TouchSimTouch` adds afferent-level indentation responses
(Saal et al., 2017) while retaining these aggregate perceptual equations.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import import_module
from typing import Any, ClassVar

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
    "steering_wheel": (
        (4, 0.08),
        (5, 0.08),
        (6.3, 0.079),
        (8, 0.081),
        (10, 0.091),
        (12.5, 0.102),
        (16, 0.115),
        (20, 0.101),
        (25, 0.089),
        (31.5, 0.079),
        (40, 0.062),
        (50, 0.048),
        (63, 0.038),
        (80, 0.032),
        (100, 0.027),
        (125, 0.023),
        (160, 0.028),
        (200, 0.034),
        (250, 0.041),
    ),
    "steering_wheel_lower": (
        (4, 0.053),
        (5, 0.063),
        (6.3, 0.073),
        (8, 0.086),
        (10, 0.102),
        (12.5, 0.12),
        (16, 0.142),
        (20, 0.124),
        (25, 0.108),
        (31.5, 0.094),
        (40, 0.07),
        (50, 0.052),
        (63, 0.039),
        (80, 0.035),
        (100, 0.031),
        (125, 0.027),
        (160, 0.033),
        (200, 0.04),
        (250, 0.049),
    ),
}

#: Growth of perceived intensity above threshold, psi = k (a - a0)^n on a
#: scale where subjects called 0.4 m/s^2 r.m.s. at 31.5 Hz 100: (frequency
#: Hz, n, k), medians of 12 subjects (Morioka and Griffin, 2009, Table 2).
SENSATION_GROWTH: dict[str, tuple[tuple[float, float, float], ...]] = {
    "steering_wheel": (
        (4, 0.649, 223.7),
        (5, 0.779, 260.4),
        (6.3, 0.786, 234.4),
        (8, 0.71, 255.4),
        (10, 0.573, 219.8),
        (12.5, 0.595, 179.6),
        (16, 0.653, 188.5),
        (20, 0.873, 186.3),
        (25, 1.02, 211.2),
        (31.5, 0.895, 185.6),
        (40, 0.77, 162.3),
        (50, 0.623, 163.9),
        (63, 0.544, 177.9),
        (80, 0.366, 158.1),
        (100, 0.371, 161.3),
        (125, 0.283, 157.0),
        (160, 0.326, 163.8),
        (200, 0.433, 165.3),
        (250, 0.419, 143.4),
    ),
    "steering_wheel_lower": (
        (4, 0.691, 270.3),
        (5, 0.668, 234.5),
        (6.3, 0.661, 188.8),
        (8, 0.954, 230.1),
        (10, 0.881, 211.6),
        (12.5, 1.154, 204.8),
        (16, 0.698, 162.0),
        (20, 1.045, 173.2),
        (25, 1.198, 197.4),
        (31.5, 0.879, 166.8),
        (40, 0.85, 181.7),
        (50, 0.718, 162.8),
        (63, 0.503, 155.1),
        (80, 0.488, 166.2),
        (100, 0.371, 161.3),
        (125, 0.467, 156.9),
        (160, 0.401, 160.7),
        (200, 0.461, 162.5),
        (250, 0.572, 157.6),
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
    **{
        site: Source(
            "Morioka & Griffin (2009), Table 2",
            "https://doi.org/10.1016/j.apergo.2008.06.001",
            "measured",
            f"medians of 12 males; 30 mm handles on a steering-wheel rig, hands "
            f"45 deg {where} the wheel centre, comfortable grip (mean 85 N)",
        )
        for site, where in (
            ("steering_wheel", "above"),
            ("steering_wheel_lower", "below"),
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

    def perceived_intensity(
        self, amplitude: float, frequency: float, site: str
    ) -> float:
        """Perceived intensity of an r.m.s. acceleration on a magnitude-
        estimation scale; 0 below threshold. Subjects called a 0.4 m/s^2
        r.m.s. 31.5 Hz reference 100; the median exponents and constants
        give it 67."""
        raise NotImplementedError


@register("touch", "vibrotactile")
class VibrotactileTouch(Touch):
    """Frequency- and site-dependent vibration thresholds."""

    def __init__(self, **params: float) -> None:
        super().__init__()
        self.params = TouchParameters(**params)
        p = self.params
        if (
            not math.isfinite(p.latency_gain)
            or p.latency_gain < 0
            or not math.isfinite(p.latency_exponent)
            or p.latency_exponent > 0
            or not math.isfinite(p.reference_latency)
            or p.reference_latency < 0
        ):
            raise ValueError(
                "latency parameters must be finite with non-negative gain and "
                "reference_latency and a non-positive exponent"
            )

    def threshold(self, frequency: float, site: str) -> float:
        """Perception threshold, m/s^2 r.m.s., interpolated in log-log."""
        if not math.isfinite(frequency) or frequency <= 0:
            raise ValueError("frequency must be finite and positive")
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
        if not math.isfinite(amplitude):
            raise ValueError("amplitude must be finite")
        if not contact or amplitude <= 0:
            return -math.inf
        return 20 * math.log10(amplitude / self.threshold(frequency, site))

    def perceived_intensity(
        self, amplitude: float, frequency: float, site: str
    ) -> float:
        if not math.isfinite(amplitude):
            raise ValueError("amplitude must be finite")
        try:
            growth = np.array(SENSATION_GROWTH[site])
        except KeyError:
            raise ValueError(
                f"no growth of sensation for {site!r}; known: "
                f"{', '.join(SENSATION_GROWTH)}"
            ) from None
        above = amplitude - self.threshold(frequency, site)
        if above <= 0:
            return 0.0
        log_f = math.log10(frequency)
        exponent = float(np.interp(log_f, np.log10(growth[:, 0]), growth[:, 1]))
        constant = float(np.interp(log_f, np.log10(growth[:, 0]), growth[:, 2]))
        return constant * above**exponent

    def latency(self, sensation_level: float) -> float:
        p = self.params
        if math.isnan(sensation_level):
            raise ValueError("sensation_level must not be NaN")

        def pieron(sl: float) -> float:
            # Combine exponents before evaluating: the intermediate amplitude
            # ratio can overflow even though the final latency is bounded.
            if p.latency_exponent == 0:
                return p.latency_gain
            return p.latency_gain * 10 ** (sl * (p.latency_exponent / 20))

        if sensation_level <= 0:
            return math.inf
        return max(0.005, p.reference_latency + pieron(sensation_level) - pieron(30.0))


_TOUCHSIM_SOURCE = (
    "https://github.com/hsaal/touchsim/blob/"
    "c7ece893c1ec75f2de10ade34422e0090a3093b9/"
)


@dataclass(frozen=True)
class TouchSimParameters(TouchParameters):
    """Psychophysical baseline plus a deterministic TouchSim stimulus protocol.

    These protocol choices are not fitted human perceptual parameters. Native
    TouchSim membrane noise is disabled because its parallel Numba random
    streams cannot be controlled by this module's NumPy Generator.
    """

    sample_rate: float = 5000.0
    pin_radius: float = 0.5
    model_index: int = 0
    afferent_classes: tuple[str, ...] = ("SA1", "RA", "PC")

    SOURCES: ClassVar = {
        **TouchParameters.SOURCES,
        "sample_rate": Source(
            "TouchSim implementation: internal neural sampling rate",
            _TOUCHSIM_SOURCE + "touchsim/transduction.py",
            "derived",
            "Hz; matches the upstream fixed 5000 Hz neural integration",
        ),
        "pin_radius": Source(
            "TouchSim stimulus-generator default",
            _TOUCHSIM_SOURCE + "touchsim/generators.py",
            "assumed",
            "mm; experimental probe geometry, not a human constant",
        ),
        "model_index": Source(
            "Deterministic model-selection convention",
            None,
            "assumed",
            "first published fitted neuron of each class; not a population fit",
        ),
        "afferent_classes": Source(
            "TouchSim supported afferent classes",
            _TOUCHSIM_SOURCE + "touchsim/classes.py",
            "assumed",
            "one of each class per supplied location, not anatomical density",
        ),
    }


@dataclass(frozen=True)
class AfferentResponse:
    """One indentation trial's afferent spikes (s) and average rates (Hz)."""

    spikes: tuple[np.ndarray, ...]
    afferent_classes: tuple[str, ...]
    locations: np.ndarray
    duration: float

    @property
    def rates(self) -> np.ndarray:
        """Spike counts divided by trial duration, one per afferent."""
        return np.array([spikes.size / self.duration for spikes in self.spikes])


@register("touch", "touchsim")
class TouchSimTouch(VibrotactileTouch):
    """Vibrotactile psychophysics with an optional TouchSim afferent pathway.

    ``sensation_level``, ``latency`` and ``perceived_intensity`` retain the
    published aggregate baseline. Only ``afferent_response`` invokes TouchSim;
    there is no validated decoder from spikes to those perceptual quantities.
    Importing or constructing this module does not import TouchSim.
    """

    params: TouchSimParameters

    def __init__(self, **params: Any) -> None:
        super().__init__(
            **{
                key: value
                for key, value in params.items()
                if key in TouchParameters.SOURCES
            }
        )
        self.params = TouchSimParameters(**params)
        p = self.params
        if not math.isfinite(p.sample_rate) or p.sample_rate <= 0:
            raise ValueError("sample_rate must be finite and positive")
        if not math.isfinite(p.pin_radius) or p.pin_radius <= 0:
            raise ValueError("pin_radius must be finite and positive")
        if (
            isinstance(p.model_index, bool)
            or not isinstance(p.model_index, int)
            or p.model_index < 0
        ):
            raise ValueError("model_index must be a non-negative integer")
        if (
            not p.afferent_classes
            or isinstance(p.afferent_classes, str)
            or any(c not in ("SA1", "RA", "PC") for c in p.afferent_classes)
            or len(set(p.afferent_classes)) != len(p.afferent_classes)
        ):
            raise ValueError(
                "afferent_classes must select unique SA1, RA or PC classes"
            )

    def afferent_response(
        self,
        trace: np.ndarray | Sequence[float] | Sequence[Sequence[float]],
        *,
        pin_locations: np.ndarray | Sequence[Sequence[float]],
        afferent_locations: np.ndarray | Sequence[Sequence[float]],
    ) -> AfferentResponse:
        """Simulate indentation of a flat skin patch, using millimetres.

        ``trace`` has shape (pins, samples), or (samples,) for one pin, and
        contains at least three displacement samples in mm, not acceleration
        or force. Both location
        arrays have shape (n, 2), in mm on the same flat patch. Each afferent
        location gets one neuron per configured class. Trials are independent;
        neural state does not persist between calls. Noise is disabled and model
        indices fixed, so results repeat without touching global random streams.
        """
        indentation = np.atleast_2d(np.asarray(trace, dtype=float))
        pins = np.asarray(pin_locations, dtype=float)
        locations = np.asarray(afferent_locations, dtype=float)
        if (
            indentation.ndim != 2
            or indentation.shape[0] == 0
            or indentation.shape[1] < 3
            or not np.isfinite(indentation).all()
        ):
            raise ValueError(
                "trace must be a finite (pins, samples) array with at least 3 samples"
            )
        for name, points in (
            ("pin_locations", pins), ("afferent_locations", locations)
        ):
            if (
                points.ndim != 2
                or points.shape[0] == 0
                or points.shape[1] != 2
                or not np.isfinite(points).all()
            ):
                raise ValueError(f"{name} must be a finite nonempty (n, 2) array")
        if pins.shape[0] != indentation.shape[0]:
            raise ValueError("trace must contain one row per pin location")
        if np.unique(pins, axis=0).shape[0] != pins.shape[0]:
            raise ValueError("pin locations must be distinct")
        p = self.params
        if round(indentation.shape[1] / p.sample_rate * 5000) < 3:
            raise ValueError(
                "trace must span at least 3 samples at TouchSim's 5000 Hz neural rate"
            )
        if pins.shape[0] > 1:
            distances = np.linalg.norm(pins[:, None] - pins[None, :], axis=2)
            np.fill_diagonal(distances, np.inf)
            if 2 * p.pin_radius > float(distances.min()):
                raise ValueError("pin discs must not overlap; reduce pin_radius")

        # Upstream changes NumPy error modes internally. Restore the caller's
        # settings even when importing or evaluating TouchSim raises an error.
        with np.errstate():
            try:
                ts = import_module("touchsim")
            except ModuleNotFoundError as exc:
                if exc.name != "touchsim":
                    raise
                raise ImportError(
                    "TouchSim afferent responses require the optional dependency: "
                    "from a source checkout or sdist, run "
                    "python -m pip install -r requirements-touchsim.txt "
                    "(see docs/touchsim.md)"
                ) from exc
            for affclass in p.afferent_classes:
                if p.model_index >= len(ts.Afferent.affparams[affclass]):
                    raise ValueError(f"model_index is out of range for {affclass}")
            afferents = [
                ts.Afferent(
                    affclass,
                    location=location.copy(),
                    idx=p.model_index,
                    noisy=False,
                    delay=False,
                )
                for location in locations
                for affclass in p.afferent_classes
            ]
            population = ts.AfferentPopulation(*afferents)
            stimulus = ts.Stimulus(
                trace=indentation.copy(),
                location=pins.copy(),
                fs=p.sample_rate,
                pin_radius=p.pin_radius,
            )
            response = population.response(stimulus)
        return AfferentResponse(
            tuple(
                np.array(spikes, dtype=float, copy=True) for spikes in response.spikes
            ),
            tuple(affclass for _ in locations for affclass in p.afferent_classes),
            np.repeat(locations, len(p.afferent_classes), axis=0),
            indentation.shape[1] / p.sample_rate,
        )
