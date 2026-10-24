"""Surrogate Reference Task (SuRT), a visual-manual search task.

ISO/TS 14198:2019 standardizes SuRT: find the one target circle among
distractor circles of a different size and select the zone that contains
it. The standard's exact geometry is not public, so every dimension here
is a parameter; the defaults are placeholders, not the standard's values.

The task runs as an embodied-ui engine: it draws each screen into a
display (``screen.image``) and scores touches on that touch surface.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SuRTConfig:
    """Geometry of one SuRT variant (placeholders, see the module notes).

    Args:
        resolution: Display pixels (width, height).
        zones: Equal-width columns the user selects among.
        distractors: Number of distractor circles.
        distractor_diameter: Distractor diameter, pixels.
        target_diameter: Target diameter, pixels ("easy": much larger).
    """

    resolution: tuple[int, int] = (320, 160)
    zones: int = 4
    distractors: int = 30
    distractor_diameter: int = 12
    target_diameter: int = 20


@dataclass(frozen=True)
class Circle:
    """A circle in display pixels (x right, y down)."""

    x: float
    y: float
    diameter: float
    target: bool


@dataclass
class SuRTScreen:
    """One screen: circles and the target's zone."""

    circles: list[Circle]
    target_zone: int
    shown_at: float


def make_screen(
    config: SuRTConfig, rng: np.random.Generator, time: float
) -> SuRTScreen:
    """Place circles without overlap; exactly one is the target."""
    width, height = config.resolution
    largest = max(config.target_diameter, config.distractor_diameter)
    circles: list[Circle] = []
    diameters = [config.target_diameter] + [
        config.distractor_diameter
    ] * config.distractors
    for index, diameter in enumerate(diameters):
        for _ in range(1000):
            x = rng.uniform(largest, width - largest)
            y = rng.uniform(largest, height - largest)
            if all(
                np.hypot(x - c.x, y - c.y) > (diameter + c.diameter) / 2 + 2
                for c in circles
            ):
                circles.append(Circle(float(x), float(y), float(diameter), index == 0))
                break
        else:
            raise ValueError(
                "SuRT circles do not fit the display; reduce their number or size"
            )
    target = circles[0]
    zone = min(config.zones - 1, int(target.x / width * config.zones))
    order = rng.permutation(len(circles))
    return SuRTScreen([circles[i] for i in order], zone, time)


def render(screen: SuRTScreen, config: SuRTConfig) -> NDArray[np.uint8]:
    """Draw the screen: light circles on a dark background, zone lines."""
    width, height = config.resolution
    ys, xs = np.mgrid[0:height, 0:width]
    image = np.full((height, width, 3), 20, dtype=np.uint8)
    for k in range(1, config.zones):
        image[:, int(k * width / config.zones), :] = 60
    for circle in screen.circles:
        inside = (xs - circle.x) ** 2 + (ys - circle.y) ** 2 <= (
            circle.diameter / 2
        ) ** 2
        image[inside] = 230
    return image


@dataclass
class SuRTResult:
    """Responses to completed screens."""

    times: list[float] = field(default_factory=list)
    correct: list[bool] = field(default_factory=list)

    def summary(self) -> dict[str, float]:
        """Screens, accuracy and mean time per screen."""
        n = len(self.times)
        return {
            "screens": float(n),
            "accuracy": float(np.mean(self.correct)) if n else float("nan"),
            "mean_screen_time_s": float(np.mean(self.times)) if n else float("nan"),
        }


class SuRTTask:
    """SuRT state machine: show screens, score zone selections.

    Use :meth:`update` as an embodied-ui ``LoopbackEngine`` update; it reads
    ``touch_begin`` events of ``surface`` and returns its display image.
    """

    def __init__(
        self,
        config: SuRTConfig = SuRTConfig(),
        surface: str = "screen",
        rng: np.random.Generator | None = None,
    ) -> None:
        self.config = config
        self.surface = surface
        self.rng = rng or np.random.default_rng()
        self.reset()

    def reset(self, time: float = 0.0) -> None:
        """Start with a new screen."""
        self.result = SuRTResult()
        self.screen = make_screen(self.config, self.rng, time)
        self.image = render(self.screen, self.config)

    def select(self, zone: int, time: float) -> bool:
        """Score a zone selection and show the next screen."""
        correct = zone == self.screen.target_zone
        self.result.times.append(time - self.screen.shown_at)
        self.result.correct.append(correct)
        self.screen = make_screen(self.config, self.rng, time)
        self.image = render(self.screen, self.config)
        return correct

    def zone_of(self, u: float) -> int:
        """Zone index of a horizontal touch position in [0, 1]."""
        return min(self.config.zones - 1, max(0, int(u * self.config.zones)))

    def update(self, command: Any, time: float) -> Mapping[str, Any]:
        """Engine update: score touches, return the display and counters."""
        for event in command.events:
            if event.source == self.surface and event.kind == "touch_begin":
                self.select(self.zone_of(event.data["u"]), time)
        return {
            f"{self.surface}.image": self.image,
            "surt.screens": float(len(self.result.times)),
            "surt.correct": float(sum(self.result.correct)),
        }
