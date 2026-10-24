"""An attentive driver: drives from belief and shares gaze with a task.

The driver runs in myosuite-carla's ``FullBodyDrivingEnv``: its reflex
controller (two-point steering, IDM) moves the MyoSuite full body, but it
sees the road only through the human model.

- The eyes move gaze between areas of interest with latency and main-
  sequence saccades.
- Object-level vision observes lateral offset and heading with
  eccentricity-dependent noise, and nothing during saccades.
- The estimator keeps the belief, whose uncertainty grows while the eyes
  are elsewhere.
- Attention returns to the road when that uncertainty is high.
- An in-vehicle task (SuRT) is searched with fixations. Search progress is
  held in memory across road glances, and targets are touched with the
  motor module's Fitts-timed reaches.
- After a takeover request, the driver looks to the road and moves the
  hand to the wheel, and drives once the belief is certain again; the
  take-over time emerges from these modules.

Two-point steering uses the observed near- and far-point angles, corrected
by the belief error in lateral offset and heading; curvature preview is
therefore not uncertain. This approximation is documented and testable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from ..attention import ROAD, AttenD
from ..core import Human, Parameters, Source
from ..tasks.surt import SuRTTask
from ..vision.eyes import Gaze
from ..vision.objects import Item

DISPLAY = "display"


@dataclass(frozen=True)
class DriverAgentParameters(Parameters):
    """Geometry and task parameters of the attentive driver.

    Args:
        display_position: Centre of the in-vehicle display, degrees.
        display_size: Angular width and height of the display, degrees.
        road_half_extent: Half width and height of the road area, degrees.
        lateral_sd: Foveal observation SD of lateral offset, m.
        heading_sd: Foveal observation SD of heading error, rad.
        search_fixation: Mean fixation duration in visual search, s.
        inspect_radius: Radius around a fixation within which a target is
            recognised, degrees.
        response_latency: Median delay from a takeover request to the first
            action, s (log-normal with ``response_spread``).
        response_spread: SD of the log response latency.
        assessment_fixations: Road fixations needed after a takeover request
            to rebuild the situation before driving.
        scene_fixation: Mean fixation duration in scene viewing, s.
        display_distance: Hand travel from the wheel to the display, m.
        display_width: Physical width of the display, m.
        wheel_width: Target width for the hand returning to the wheel, m.
        driving_chunks: Chunks a road glance places in the focus of
            attention (lane position, heading, traffic...), displacing task
            memory.
        road_field: Largest eccentricity from which lane position and
            heading are still perceived, degrees.
        perception_interval: Time between perceptual samples, s; visual
            samples are not independent at the simulation rate.
    """

    display_position: Gaze = (20.0, -25.0)
    display_size: tuple[float, float] = (16.0, 8.0)
    road_half_extent: tuple[float, float] = (30.0, 12.0)
    lateral_sd: float = 0.05
    heading_sd: float = 0.01
    search_fixation: float = 0.275
    inspect_radius: float = 3.0
    response_latency: float = 0.3
    response_spread: float = 0.3
    assessment_fixations: int = 3
    scene_fixation: float = 0.33
    display_distance: float = 0.45
    display_width: float = 0.2
    wheel_width: float = 0.1
    driving_chunks: int = 4
    road_field: float = 20.0
    perception_interval: float = 0.1

    SOURCES: ClassVar = {
        "display_position": Source(
            "model choice", None, "assumed", "a centre-stack touchscreen"
        ),
        "display_size": Source("model choice", None, "assumed"),
        "road_half_extent": Source("model choice", None, "assumed"),
        "lateral_sd": Source(
            "model choice", None, "assumed", "fit to lane-keeping variability"
        ),
        "heading_sd": Source("model choice", None, "assumed"),
        "search_fixation": Source(
            "Rayner (1998), Table 1",
            "https://doi.org/10.1037/0033-2909.124.3.372",
            "measured",
            "mean fixation in visual search",
        ),
        "inspect_radius": Source(
            "model choice", None, "assumed", "functional field for a size oddball"
        ),
        "response_latency": Source("model choice", None, "assumed"),
        "response_spread": Source("model choice", None, "assumed"),
        "assessment_fixations": Source(
            "model choice",
            None,
            "assumed",
            "situation assessment before acting; fit to take-over data",
        ),
        "scene_fixation": Source(
            "Rayner (1998), Table 1",
            "https://doi.org/10.1037/0033-2909.124.3.372",
            "measured",
            "mean fixation in scene perception",
        ),
        "display_distance": Source("model choice", None, "assumed"),
        "display_width": Source(
            "model choice", None, "assumed", "an 8-inch centre display"
        ),
        "wheel_width": Source("model choice", None, "assumed"),
        "driving_chunks": Source(
            "model choice",
            None,
            "assumed",
            "with a focus capacity of 4, a road glance displaces task state",
        ),
        "road_field": Source(
            "model choice",
            None,
            "assumed",
            "peripheral lane keeping degrades with display "
            "eccentricity; fit to glance data",
        ),
        "perception_interval": Source(
            "model choice", None, "assumed", "about the temporal integration of vision"
        ),
    }


@dataclass
class _Search:
    """SuRT search state: grid cells already inspected."""

    cells: list[Gaze]
    inspected: int = 0
    fixation_end: float | None = None
    reach_end: float | None = None
    zone: int | None = None


@dataclass
class DrivingLog:
    """Gaze areas, AttenD and task outcomes over a run."""

    times: list[float] = field(default_factory=list)
    areas: list[str] = field(default_factory=list)
    attend: list[float] = field(default_factory=list)
    screen_times: list[tuple[float, float]] = field(default_factory=list)
    engaged_at: float | None = None
    request_at: float | None = None

    def glances(self) -> list[tuple[float, float]]:
        """Off-road glances as (start, end): gaze leaving the road area
        until it is back, saccades included."""
        result = []
        start = None
        for time, area in zip(self.times, self.areas):
            if area != ROAD and start is None:
                start = time
            elif area == ROAD and start is not None:
                result.append((start, time))
                start = None
        return result

    def glance_metrics(self) -> dict[str, float]:
        """Mean and long-glance share of off-road glances, and eyes-off-road
        time per completed SuRT screen (NHTSA, 2013)."""
        durations = np.array([end - start for start, end in self.glances()])
        per_screen = []
        for start, end in self.screen_times:
            off = sum(max(0.0, min(e, end) - max(s, start)) for s, e in self.glances())
            per_screen.append(off)
        return {
            "glances": float(len(durations)),
            "mean_offroad_glance_s": float(durations.mean())
            if len(durations)
            else math.nan,
            "p85_offroad_glance_s": float(np.percentile(durations, 85))
            if len(durations)
            else math.nan,
            "long_glance_share": float(np.mean(durations > 2.0))
            if len(durations)
            else math.nan,
            "mean_eyes_off_road_per_screen_s": float(np.mean(per_screen))
            if per_screen
            else math.nan,
            "attend_minimum_s": float(min(self.attend)) if self.attend else math.nan,
        }


class AttentiveDriver:
    """Drive ``env`` with ``human``'s modules, optionally doing SuRT.

    Args:
        human: Uses eyes, objects, estimator, attention, memory and motor.
        env: A myosuite-carla ``FullBodyDrivingEnv``.
        task: SuRT task shown on the in-vehicle display, or ``None``.
        params: Geometry and task parameters.
        reflex: Steering and speed controller; myosuite-carla's
            ``ReflexDriver`` with no perception delay by default, since
            delays come from the human model.
    """

    def __init__(
        self,
        human: Human,
        env: Any,
        task: SuRTTask | None = None,
        params: DriverAgentParameters = DriverAgentParameters(),
        reflex: Any = None,
    ) -> None:
        from myosuite_carla.driver_models import ReflexDriver, ReflexDriverConfig

        self.human = human
        self.env = env
        self.task = task
        self.params = params
        self.dt = float(env.dt)
        self.reflex = reflex or ReflexDriver(
            self.dt, ReflexDriverConfig(perception_delay=self.dt)
        )

    # Episode ------------------------------------------------------------------

    def reset(self, seed: int | None = None) -> None:
        """Reset modules and logs after ``env.reset``."""
        self.human.reset(seed)
        self.reflex.reset()
        self.time = 0.0
        self.log = DrivingLog()
        self._attend = AttenD()
        self._destination = ROAD
        self._engaged = True
        self._hand_back: float | None = None
        self._reacted = False
        self._latency: float | None = None
        self._assessment: float | None = None
        self._road_since: float | None = None
        self._next_sample = 0.0
        state = self.env.state
        self.human.estimator.reset(  # type: ignore[call-arg]
            self.human.estimator.rng,
            state=(state["lane_offset_m"], state["heading_error_rad"]),
        )
        self.human.eyes.reset(self.human.eyes.rng, gaze=(0.0, 0.0))  # type: ignore[call-arg]
        if self.task is not None:
            self.task.reset(0.0)
            self._search = self._new_search()

    # Geometry -----------------------------------------------------------------

    def area(self, gaze: Gaze) -> str:
        """The area of interest a gaze direction falls in."""
        p = self.params
        if (
            abs(gaze[0]) <= p.road_half_extent[0]
            and abs(gaze[1]) <= p.road_half_extent[1]
        ):
            return ROAD
        dx = gaze[0] - p.display_position[0]
        dy = gaze[1] - p.display_position[1]
        if (
            abs(dx) <= p.display_size[0] / 2 + 2
            and abs(dy) <= p.display_size[1] / 2 + 2
        ):
            return DISPLAY
        return "transition"

    def _cell_gaze(self, x: float, y: float) -> Gaze:
        """Display pixel coordinates to gaze angles."""
        assert self.task is not None
        width, height = self.task.config.resolution
        p = self.params
        return (
            p.display_position[0] + (x / width - 0.5) * p.display_size[0],
            p.display_position[1] - (y / height - 0.5) * p.display_size[1],
        )

    def _new_search(self) -> _Search:
        assert self.task is not None
        width, height = self.task.config.resolution
        # Scan a grid whose cells are about one inspection radius apart.
        p = self.params
        columns = max(1, math.ceil(p.display_size[0] / (1.6 * p.inspect_radius)))
        rows = max(1, math.ceil(p.display_size[1] / (1.6 * p.inspect_radius)))
        cells = [
            self._cell_gaze((c + 0.5) * width / columns, (r + 0.5) * height / rows)
            for c in range(columns)
            for r in range(rows)
        ]
        return _Search(cells)

    # Control ------------------------------------------------------------------

    def act(self) -> Any:
        """Return the next ``DriverIntent`` after perceiving and attending."""
        from myosuite_carla.fullbody import DriverIntent

        human, p = self.human, self.params
        self.time += self.dt
        t = self.time
        state = dict(self.env.state)
        eye = human.eyes.step(t)  # type: ignore[attr-defined]
        area = "transition" if eye.in_saccade else self.area(eye.gaze)
        self.log.times.append(t)
        self.log.areas.append(area)
        self.log.attend.append(self._attend.step(self.dt, area))

        # Perception and belief.
        yaw_rate = float(state.get("yaw_rate_rad_s", 0.0))
        estimator = human.estimator
        estimator.predict(self.dt, speed=state["speed_mps"], yaw_rate=yaw_rate)  # type: ignore[attr-defined]
        observed = {}
        sample_due = t >= self._next_sample
        if sample_due and math.hypot(*eye.gaze) <= p.road_field:
            observed = human.objects.observe(  # type: ignore[attr-defined]
                {
                    "offset": Item((0.0, 0.0), state["lane_offset_m"], p.lateral_sd),
                    "heading": Item(
                        (0.0, 0.0), state["heading_error_rad"], p.heading_sd
                    ),
                },
                eye.gaze,
                eye.in_saccade,
            )
        if sample_due:
            self._next_sample = t + p.perception_interval
        if observed:
            estimator.update(  # type: ignore[attr-defined]
                np.array([observed["offset"].value, observed["heading"].value]),
                np.array([observed["offset"].sd, observed["heading"].sd]),
            )
        if self._reacted and area == ROAD:
            self._road_since = self._road_since if self._road_since is not None else t
        elif area != ROAD:
            self._road_since = None
        if area == ROAD:
            # The driving situation occupies the focus of attention.
            for k in range(p.driving_chunks):
                human.memory.encode(f"road{k}", None, t)  # type: ignore[attr-defined]

        # Takeover: the request is heard, the task is dropped, the hand
        # returns to the wheel and gaze goes to the road.
        automation = bool(state.get("automation", 0.0))
        requested = bool(state.get("takeover_requested", 0.0))
        if automation and requested and self.log.request_at is None:
            self.log.request_at = t - state.get("time_since_request_s", 0.0)
        if self.log.request_at is not None and self._latency is None:
            rng = human.attention.rng
            self._latency = p.response_latency * math.exp(
                p.response_spread * rng.standard_normal()
            )
            self._assessment = float(
                sum(
                    rng.gamma(4.0, p.scene_fixation / 4.0)
                    for _ in range(p.assessment_fixations)
                )
            )
        if (
            self.log.request_at is not None
            and not self._reacted
            and self._latency is not None
            and t >= self.log.request_at + self._latency
        ):
            self._reacted = True
            movement = human.motor.plan(  # type: ignore[attr-defined]
                np.zeros(3), np.array([p.display_distance, 0.0, 0.0]), p.wheel_width, t
            )
            self._hand_back = movement.finish
        task_active = self.task is not None and not self._reacted
        demands = {DISPLAY: 1.0} if task_active else {}

        # Attention.
        uncertainty = estimator.lateral_sd  # type: ignore[attr-defined]
        target = human.attention.select(t, self._destination, demands, uncertainty)  # type: ignore[attr-defined]
        if target != self._destination:
            self._destination = target
            if target == ROAD:
                human.eyes.command((0.0, 0.0), t)  # type: ignore[attr-defined]
            elif target == DISPLAY and self.task is not None:
                self._resume_search(t)

        if task_active:
            self._task_step(t, area == DISPLAY and self._destination == DISPLAY)

        # Driving from belief.
        if automation and not self._engaged_now(t, uncertainty):
            return DriverIntent()
        believed = self._believed(state)
        return self.reflex.act(believed)

    def _engaged_now(self, t: float, uncertainty: float) -> bool:
        if self.log.request_at is None:
            return False
        assessed = (
            self._road_since is not None
            and self._assessment is not None
            and t - self._road_since >= self._assessment
        )
        ready = (
            self._hand_back is not None
            and t >= self._hand_back
            and self._destination == ROAD
            and assessed
            and uncertainty <= self.human.attention.params.leave_sd  # type: ignore[attr-defined]
        )
        if ready and self.log.engaged_at is None:
            self.log.engaged_at = t
        return self.log.engaged_at is not None

    def _believed(self, state: dict[str, float]) -> dict[str, float]:
        mean = self.human.estimator.mean  # type: ignore[attr-defined]
        error_offset = float(mean[0]) - state["lane_offset_m"]
        error_heading = float(mean[1]) - state["heading_error_rad"]
        believed = dict(state)
        believed["lane_offset_m"] = float(mean[0])
        believed["heading_error_rad"] = float(mean[1])
        near = float(self.env.task.near_distance)
        far = max(near + 2.0, state["speed_mps"] * float(self.env.task.far_preview))
        believed["near_angle_rad"] = (
            state["near_angle_rad"] - (error_offset + near * error_heading) / near
        )
        believed["far_angle_rad"] = (
            state["far_angle_rad"] - (error_offset + far * error_heading) / far
        )
        # The human model decides engagement; the reflex controller acts.
        believed["automation"] = 0.0
        believed["takeover_requested"] = 0.0
        return believed

    # SuRT search ----------------------------------------------------------------

    def _resume_search(self, t: float) -> None:
        """Look back at the display, where memory says the search stopped."""
        search = self._search
        recalled = self.human.memory.retrieve("surt_progress", t)  # type: ignore[attr-defined]
        if recalled.success and recalled.value is not None:
            search.inspected = int(recalled.value)
        else:
            search.inspected = 0
        search.fixation_end = None
        cell = search.cells[min(search.inspected, len(search.cells) - 1)]
        self.human.eyes.command(cell, t)  # type: ignore[attr-defined]

    def _task_step(self, t: float, looking: bool) -> None:
        assert self.task is not None
        human, p, search = self.human, self.params, self._search
        if search.reach_end is not None:
            # The touch lands whether or not the eyes are still on the display.
            if t >= search.reach_end:
                start = self.task.screen.shown_at
                self.task.select(int(search.zone or 0), t)
                self.log.screen_times.append((start, t))
                self.human.memory.encode("surt_progress", 0, t)  # type: ignore[attr-defined]
                self._search = self._new_search()
                if looking:
                    self.human.eyes.command(self._search.cells[0], t)  # type: ignore[attr-defined]
            return
        if not looking or human.eyes.busy:  # type: ignore[attr-defined]
            return
        if search.fixation_end is None:
            duration = p.search_fixation * float(human.eyes.rng.gamma(4.0, 0.25))
            search.fixation_end = t + duration
            return
        if t < search.fixation_end:
            return
        # Fixation done: is the target within the inspected radius?
        gaze = human.eyes.gaze  # type: ignore[attr-defined]
        target = next(c for c in self.task.screen.circles if c.target)
        position = self._cell_gaze(target.x, target.y)
        if math.hypot(position[0] - gaze[0], position[1] - gaze[1]) <= p.inspect_radius:
            pixels = self.task.config.resolution[0]
            zone_width = p.display_width / self.task.config.zones
            movement = human.motor.plan(  # type: ignore[attr-defined]
                np.zeros(3), np.array([p.display_distance, 0.0, 0.0]), zone_width, t
            )
            # The finger aims at the target; its lateral endpoint error on
            # the display decides the touched zone.
            error_px = float(movement.end[1]) / p.display_width * pixels
            zone = self.task.zone_of((target.x + error_px) / pixels)
            search.reach_end = movement.finish
            search.zone = zone
            return
        search.inspected += 1
        human.memory.encode("surt_progress", search.inspected, t)  # type: ignore[attr-defined]
        if search.inspected >= len(search.cells):
            search.inspected = 0  # missed it: scan again
        search.fixation_end = None
        human.eyes.command(search.cells[search.inspected], t)  # type: ignore[attr-defined]
