"""Object-level vision: what is seen of areas of interest, by eccentricity.

For tasks modelled above the pixel level, each area of interest (the road,
a display item, a mirror) carries a quantity the observer estimates. Its
observation noise grows linearly with eccentricity,

    sd(e) = sd0 * (1 + e / E2),

the form that describes acuity and positional thresholds in the periphery
(Strasburger, Rentschler and Jüttner, 2011). Nothing is seen during
saccades or beyond the field of view.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from ..core import Module, Parameters, Source, register
from .eyes import Gaze


@dataclass(frozen=True)
class Item:
    """An area of interest.

    Args:
        position: Direction from the eye, (azimuth, elevation) degrees.
        value: True quantity, a number or array.
        foveal_sd: Observation noise SD when fixated, in value units.
    """

    position: Gaze
    value: Any
    foveal_sd: float


@dataclass(frozen=True)
class Observation:
    """A noisy observation of one item."""

    value: Any
    sd: float
    eccentricity: float


@dataclass(frozen=True)
class ObjectParameters(Parameters):
    """Eccentricity scaling of observation noise.

    Args:
        e2: Eccentricity at which noise doubles, degrees.
        field_of_view: Largest observable eccentricity, degrees.
    """

    e2: float = 2.0
    field_of_view: float = 90.0

    SOURCES: ClassVar = {
        "e2": Source(
            "Strasburger, Rentschler & Jüttner (2011), Table 4",
            "https://doi.org/10.1167/11.5.13",
            "measured",
            "letter and Landolt-C acuity E2 1.0-2.6 deg; vernier 0.7; gratings 2.6-3.0",
        ),
        "field_of_view": Source(
            "model choice", None, "assumed", "monocular temporal field is wider"
        ),
    }


class Objects(Module):
    """Category ``objects``: observe areas of interest from a gaze direction."""

    category = "objects"

    def observe(
        self, items: Mapping[str, Item], gaze: Gaze, suppressed: bool = False
    ) -> dict[str, Observation]:
        """Return observations of the items visible from ``gaze``."""
        raise NotImplementedError


@register("objects", "eccentricity_noise")
class EccentricityNoise(Objects):
    """Gaussian observation noise scaled by ``1 + e / E2``."""

    def __init__(self, **params: float) -> None:
        super().__init__()
        self.params = ObjectParameters(**params)

    def sd(self, foveal_sd: float, eccentricity: float) -> float:
        """Observation SD at an eccentricity."""
        return foveal_sd * (1.0 + eccentricity / self.params.e2)

    def observe(
        self, items: Mapping[str, Item], gaze: Gaze, suppressed: bool = False
    ) -> dict[str, Observation]:
        if suppressed:
            return {}
        result = {}
        for name, item in items.items():
            eccentricity = math.hypot(
                item.position[0] - gaze[0], item.position[1] - gaze[1]
            )
            if eccentricity > self.params.field_of_view:
                continue
            sd = self.sd(item.foveal_sd, eccentricity)
            value = np.asarray(item.value, dtype=float)
            noisy = value + sd * self.rng.standard_normal(value.shape)
            result[name] = Observation(
                noisy if noisy.shape else float(noisy), sd, eccentricity
            )
        return result
