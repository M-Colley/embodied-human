"""Measurement boundaries in driving logs and takeover protocols."""

import math
from dataclasses import replace

import pytest

import embodied_human as eh
from embodied_human.agents.driver import DriverAgentParameters, DrivingLog
from embodied_human.agents.road import simulate_takeover
from embodied_human.response import Reaction


def test_unfinished_glance_counts_toward_completed_screen_exposure():
    log = DrivingLog(
        times=[0.0, 0.5, 1.0, 1.5, 2.0],
        areas=["road", "display", "display", "display", "display"],
        screen_times=[(0.0, 1.5)],
    )
    assert log.glances() == []
    assert log.glances(include_censored=True) == [(0.5, 2.0)]
    metrics = log.glance_metrics()
    assert metrics["glances"] == 0
    assert metrics["right_censored_glances"] == 1
    assert metrics["right_censored_observed_s"] == 1.5
    assert metrics["mean_eyes_off_road_per_screen_s"] == 1.0


def test_road_fixation_is_recorded_before_manual_response():
    log = simulate_takeover(
        eh.build_human(seed=0), request_at=1.0, seconds=2.0, surt=False, seed=0,
    )
    parts = log.takeover_components()
    assert parts["eyes_s"] == pytest.approx(0.0, abs=0.021)
    assert parts["response_s"] > parts["eyes_s"]
    assert math.isnan(parts["gaze_s"])


def test_planned_hand_return_after_horizon_is_censored():
    human = eh.build_human(seed=0)
    human.response.respond = lambda *args: Reaction(0.8, "auditory", False, 0.1)
    log = simulate_takeover(
        human, request_at=1.0, seconds=1.9, surt=False,
        params=replace(DriverAgentParameters(), hand_release=0.0),
    )
    assert log.planned_hand_back_at > log.times[-1]
    assert log.hand_back_at is None
    assert math.isnan(log.takeover_components()["hand_s"])


def test_parallel_coupling_reports_legacy_response_fallback():
    human = eh.build_human(seed=0)
    human.response.respond = lambda *args: Reaction(0.8, "auditory", False, 0.1)
    log = simulate_takeover(
        human, request_at=1.0, seconds=2.0, surt=False,
        params=replace(DriverAgentParameters(), parallel_gaze_preparation=True),
    )
    assert log.reaction["gaze_coupling"] == "serial_fallback"
    assert log.reaction["orient_command_latency_s"] == 0.8
