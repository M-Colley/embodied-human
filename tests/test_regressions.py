"""Regression cases found during the repository-wide review."""

import math
import subprocess
import sys

import numpy as np
import pytest

import embodied_human as eh
from embodied_human.agents.road import simulate_takeover
from embodied_human.core import ModuleChoice
from embodied_human.fatigue import ThreeCompartment
from embodied_human.inference import Prior, abc_rejection, distance
from embodied_human.tasks.drt import DRTConfig, DRTTask


def test_create_loads_builtins_in_a_fresh_interpreter():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import embodied_human as eh; "
            "assert eh.create('memory', 'act_r').params.capacity == 4",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    with pytest.raises(ValueError, match="unknown category"):
        eh.available("unknown")


def test_config_override_dictionaries_are_independent():
    first, second = eh.HumanConfig(), eh.HumanConfig()
    first.memory.params["capacity"] = 1
    assert not second.memory.params
    assert not eh.HumanConfig().memory.params
    overrides = {"capacity": 2}
    choice = ModuleChoice("act_r", overrides)
    overrides["capacity"] = 9
    assert choice.params["capacity"] == 2


@pytest.mark.parametrize("scale", [0.0, -1.0, math.nan, math.inf])
def test_abc_rejects_invalid_scales_before_simulation(scale):
    called = []
    with pytest.raises(ValueError, match="scale"):
        abc_rejection(
            lambda p: called.append(p),
            {"y": 0.0},
            {"y": scale},
            [Prior("x", 0, 1)],
        )
    assert not called


def test_abc_never_accepts_failed_simulations():
    with pytest.raises(ValueError, match="finite simulations"):
        abc_rejection(
            lambda _: {"y": math.nan},
            {"y": 0.0},
            {"y": 1.0},
            [Prior("x", 0, 1)],
            samples=4,
        )
    posterior = abc_rejection(
        lambda p: {"y": p["x"] if p["x"] < 0.5 else math.nan},
        {"y": 0.0},
        {"y": 1.0},
        [Prior("x", 0, 1)],
        samples=8,
        accept=0.5,
    )
    assert np.all(np.isfinite(posterior.distances))
    assert np.all(posterior.samples < 0.5)


def test_abc_rejects_duplicate_priors_and_truncated_mapper():
    args = (lambda p: {"y": p["x"]}, {"y": 0.0}, {"y": 1.0})
    with pytest.raises(ValueError, match="unique"):
        abc_rejection(*args, [Prior("x", 0, 1), Prior("x", 1, 2)])
    with pytest.raises(ValueError, match="one summary"):
        abc_rejection(
            *args, [Prior("x", 0, 1)], mapper=lambda f, items: [], samples=4
        )
    with pytest.raises(ValueError, match="finite"):
        Prior("x", 0, math.inf)
    assert distance({"y": 1e200}, {"y": 0.0}, {"y": 1.0}) == 1e200
    assert distance({"y": 1e308}, {"y": -1e308}, {"y": 1e308}) == 2.0
    assert Prior("x", -1e308, 1e308).scale(np.array([0.5]))[0] == 0.0


def test_drt_schedule_and_light_survive_skipped_frames():
    config = DRTConfig(interval=(3.0, 3.0))
    task = DRTTask(config, rng=np.random.default_rng(0))
    assert task.step(3.0)
    assert not task.step(5.6)
    assert task.onset is None and task.result.misses == 1
    assert not task.step(20.0)
    assert task.result.onsets == [3.0, 6.0, 9.0, 12.0, 15.0, 18.0]
    assert task.result.misses == 5
    assert task.respond(20.3) == "hit"


def test_drt_event_can_arrive_on_the_first_frame_of_a_stimulus():
    task = DRTTask(DRTConfig(interval=(3.0, 3.0)))
    assert task.respond(3.2) == "hit"
    assert task.result.reaction_times == pytest.approx([0.2])
    with pytest.raises(ValueError, match="interval"):
        DRTConfig(interval=(0.0, 0.0))


@pytest.mark.parametrize("dt", [0.0, -0.1, math.inf, math.nan])
def test_endurance_rejects_time_steps_that_cannot_advance(dt):
    with pytest.raises(ValueError, match="dt"):
        ThreeCompartment().endurance_time("hand", 0.5, dt=dt)


def test_no_detected_warning_does_not_invent_a_takeover():
    log = simulate_takeover(
        eh.build_human(seed=0), signals=(), seconds=18.0, surt=False
    )
    assert log.request_at is not None
    assert log.reacted_at is None
    assert log.engaged_at is None
