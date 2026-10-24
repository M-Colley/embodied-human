"""Foveated images: resolution falling with eccentricity from the gaze point.

Following Geisler and Perry (1998), the highest visible spatial frequency
at eccentricity ``e`` (degrees) is

    f_c(e) = e2 * ln(1 / CT0) / (alpha * (e + e2))   cycles/deg,

from the contrast threshold CT(f, e) = CT0 * exp(alpha * f * (e + e2) / e2).
Each pixel is low-pass filtered so its Gaussian transfer falls to one half
at ``f_c``: sigma = sqrt(ln 2 / (2 pi^2)) / f_c degrees. Pixels blend between
precomputed blur levels, as in multiresolution foveated rendering.

The result is a retina-like input for learned perception, a stronger
baseline than a sharp foveal crop with a uniformly coarse periphery.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from ..core import Module, Parameters, Source, register

_HALF_POWER = math.sqrt(math.log(2) / (2 * math.pi**2))


@dataclass(frozen=True)
class FoveationParameters(Parameters):
    """Contrast-threshold model of Geisler and Perry (1998).

    Args:
        alpha: Spatial-frequency decay constant.
        e2: Half-resolution eccentricity, degrees.
        ct0: Minimum contrast threshold.
        levels: Number of blur levels; each doubles the blur.
    """

    alpha: float = 0.106
    e2: float = 2.3
    ct0: float = 1.0 / 64.0
    levels: int = 7

    SOURCES: ClassVar = {
        "alpha": Source(
            "Geisler & Perry (1998), SPIE 3299",
            None,
            "fitted",
            "fit to contrast thresholds across eccentricity",
        ),
        "e2": Source("Geisler & Perry (1998)", None, "fitted"),
        "ct0": Source("Geisler & Perry (1998)", None, "fitted"),
        "levels": Source("implementation choice", None, "assumed"),
    }

    def cutoff(self, eccentricity: NDArray[np.float64] | float) -> NDArray[np.float64]:
        """Highest visible spatial frequency, cycles/deg."""
        e = np.asarray(eccentricity, dtype=float)
        return self.e2 * math.log(1.0 / self.ct0) / (self.alpha * (e + self.e2))


@dataclass(frozen=True)
class Percept:
    """A foveated image and its geometry.

    Attributes:
        image: Float image in [0, 1] (H x W x C), or ``None`` while vision is
            suppressed during a saccade.
        gaze_px: Gaze point in pixel coordinates (x, y).
        eccentricity: Per-pixel eccentricity, degrees.
        cutoff: Per-pixel highest visible frequency, cycles/deg.
    """

    image: NDArray[np.float64] | None
    gaze_px: tuple[float, float]
    eccentricity: NDArray[np.float64] | None
    cutoff: NDArray[np.float64] | None

    @property
    def suppressed(self) -> bool:
        """Whether vision was suppressed."""
        return self.image is None


class Vision(Module):
    """Category ``vision``: turn a scene image and gaze into a percept."""

    category = "vision"

    def perceive(
        self,
        image: NDArray[np.generic],
        gaze_px: tuple[float, float],
        pixels_per_degree: float,
        suppressed: bool = False,
    ) -> Percept:
        """Return the percept of ``image`` fixated at ``gaze_px``."""
        raise NotImplementedError


@register("vision", "foveated")
class FoveatedVision(Vision):
    """Multiresolution foveation with the Geisler and Perry cutoff."""

    def __init__(self, **params: Any) -> None:
        super().__init__()
        self.params = FoveationParameters(**params)

    def sigma_degrees(self, eccentricity: NDArray[np.float64]) -> NDArray[np.float64]:
        """Gaussian blur (SD, degrees) that halves contrast at the cutoff."""
        return _HALF_POWER / self.params.cutoff(eccentricity)

    def perceive(
        self,
        image: NDArray[np.generic],
        gaze_px: tuple[float, float],
        pixels_per_degree: float,
        suppressed: bool = False,
    ) -> Percept:
        from scipy.ndimage import gaussian_filter

        if pixels_per_degree <= 0 or not math.isfinite(pixels_per_degree):
            raise ValueError("pixels_per_degree must be finite and positive")
        array = np.asarray(image)
        scale = 255.0 if array.dtype == np.uint8 else 1.0
        pixels = array.astype(float) / scale
        if pixels.ndim == 2:
            pixels = pixels[:, :, None]
        height, width = pixels.shape[:2]
        ys, xs = np.mgrid[0:height, 0:width]
        eccentricity = np.hypot(xs - gaze_px[0], ys - gaze_px[1]) / pixels_per_degree
        cutoff = self.params.cutoff(eccentricity)
        if suppressed:
            return Percept(None, gaze_px, eccentricity, cutoff)
        sigma_px = self.sigma_degrees(eccentricity) * pixels_per_degree
        # Levels: 0 (sharp), then blur SD 0.5, 1, 2, ... pixels.
        sigmas = [0.0] + [0.5 * 2**k for k in range(self.params.levels - 1)]
        stack = np.stack(
            [
                pixels
                if s == 0
                else gaussian_filter(pixels, sigma=(s, s, 0), mode="nearest")
                for s in sigmas
            ]
        )
        # Fractional level index of each pixel's required blur.
        level = np.interp(sigma_px, sigmas, np.arange(len(sigmas)))
        low = np.floor(level).astype(int)
        high = np.minimum(low + 1, len(sigmas) - 1)
        weight = (level - low)[:, :, None]
        rows, cols = ys, xs
        result = (1 - weight) * stack[low, rows, cols] + weight * stack[
            high, rows, cols
        ]
        if array.ndim == 2:
            result = result[:, :, 0]
        return Percept(result, gaze_px, eccentricity, cutoff)
