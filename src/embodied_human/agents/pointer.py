"""A touchscreen user: eyes find the target, the finger taps it.

The fingertip is an embodied-ui servo probe moved along the motor module's
plans, so every selection is a physical contact registered by an
embodied-ui touch surface. Each tap is one Fitts movement from the previous
touch: the finger lifts, travels and presses along a minimum-jerk path.
The eyes saccade to each target as the movement starts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..core import Human
from ..tasks.tapping import (
    TappingCondition,
    Throughput,
    iso_order,
    target_positions,
    throughput,
)


@dataclass
class TapTrial:
    """One selection."""

    start: NDArray[np.float64]
    target: NDArray[np.float64]
    endpoint: NDArray[np.float64]
    movement_time: float
    gaze_landing: float


@dataclass
class TappingRun:
    """Every selection of one condition."""

    condition: TappingCondition
    trials: list[TapTrial] = field(default_factory=list)
    missed: int = 0

    def throughput(self, skip: int = 1) -> Throughput:
        """Effective measures, skipping the first ``skip`` warm-up taps."""
        trials = self.trials[skip:]
        return throughput(
            np.array([t.start for t in trials]),
            np.array([t.target for t in trials]),
            np.array([t.endpoint for t in trials]),
            np.array([t.movement_time for t in trials]),
            self.condition.width,
        )


def run_tapping(
    human: Human,
    condition: TappingCondition,
    *,
    rounds: int = 2,
    lift: float = 0.015,
    press: float = 0.004,
    viewing_distance: float = 0.5,
    dt: float = 0.002,
) -> TappingRun:
    """Run ISO tapping on a simulated touch surface.

    Args:
        human: Uses its ``motor`` and ``eyes`` modules.
        condition: Amplitude and width.
        rounds: Passes around the circle.
        lift: Peak finger height between taps, m.
        press: Planned depth below the surface at the end of a tap, m.
        viewing_distance: Eye to surface, m, for gaze angles.
        dt: Physics and sampling step, s.
    """
    import mujoco
    from embodied_ui import Interface, TouchSurface
    from embodied_ui.probe import Probe

    size = condition.amplitude + 4 * condition.width + 0.04
    interface = Interface(
        "tap", (TouchSurface("surface", size=(size, size), threshold=0.05),)
    )
    tip_radius = 0.008
    targets = target_positions(condition)
    order = iso_order(condition.targets)
    first = targets[order[0]]
    spec = mujoco.MjSpec()
    spec.option.timestep = dt
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.gravity = (0.0, 0.0, 0.0)
    probe = Probe(pos=(first[0], first[1], tip_radius + lift), stiffness=4000.0)
    probe.attach(spec)
    interface.attach(spec)
    model = spec.compile()
    data = mujoco.MjData(model)
    runtime = interface.bind(model)
    tip = probe.bind(model)
    motor, eyes = human.motor, human.eyes
    eyes.reset(eyes.rng, gaze=_gaze(first, viewing_distance))  # type: ignore[call-arg]

    def surface_point(xy: NDArray[np.float64]) -> NDArray[np.float64]:
        # Tip centre touching the surface at xy.
        return np.array([xy[0], xy[1], tip_radius])

    run = TappingRun(condition)
    time = 0.0
    start = first.copy()
    # The first tap starts from the hover point above the first target.
    sequence = [order[i % len(order)] for i in range(rounds * len(order) + 1)]
    axes = (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
    previous_touch = None
    for target_index in sequence:
        target = targets[target_index]
        eyes.command(_gaze(target, viewing_distance), time)  # type: ignore[attr-defined]
        plan = motor.plan(  # type: ignore[attr-defined]
            surface_point(start), surface_point(target), condition.width, time, axes
        )
        landing = math.nan
        touched = None
        steps = int(math.ceil((plan.duration + 0.15) / dt))
        for _ in range(steps):
            time += dt
            tau = min(1.0, max(0.0, (time - plan.start_time) / plan.duration))
            position = plan.position(time)
            height = lift * math.sin(math.pi * tau) - press * tau**4
            tip.target(data, (position[0], position[1], tip_radius + height))
            runtime.step(data, 1)
            reading = runtime.read(data)
            state = eyes.step(time)  # type: ignore[attr-defined]
            if (
                not state.in_saccade
                and math.isnan(landing)
                and time > plan.start_time + 0.05
            ):
                landing = time
            begins = [e for e in reading.events if e.kind == "touch_begin"]
            if begins and touched is None:
                u, v = begins[0].data["u"], begins[0].data["v"]
                touched = (time, np.array([(u - 0.5) * size, (0.5 - v) * size]))
            if touched is not None and tau >= 1.0:
                break
        if touched is None:
            run.missed += 1
            start = target
            continue
        touch_time, point = touched
        if previous_touch is not None:
            run.trials.append(
                TapTrial(
                    start.copy(),
                    target.copy(),
                    point,
                    touch_time - previous_touch,
                    landing,
                )
            )
        previous_touch = touch_time
        start = point
    return run


def _gaze(point: NDArray[np.float64], distance: float) -> tuple[float, float]:
    return (
        math.degrees(math.atan2(float(point[0]), distance)),
        math.degrees(math.atan2(float(point[1]), distance)),
    )


def tapping_summary(runs: list[TappingRun]) -> dict[str, Any]:
    """Throughput per condition and the Fitts regression on effective ID."""
    rows = [
        run.throughput().as_dict() | {"nominal_id": run.condition.nominal_id}
        for run in runs
    ]
    ids = np.array([r["effective_id"] for r in rows])
    times = np.array([r["movement_time"] for r in rows])
    slope, intercept = (
        np.polyfit(ids, times, 1) if len(rows) > 1 else (math.nan, math.nan)
    )
    return {
        "conditions": rows,
        "mean_throughput": float(np.mean([r["throughput"] for r in rows])),
        "regression_intercept_s": float(intercept),
        "regression_slope_s_per_bit": float(slope),
    }
