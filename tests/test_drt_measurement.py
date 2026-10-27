"""Published DRT hit windows censor measurements while retaining diagnostics."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from embodied_human import build_human
from embodied_human.bench.load import (
    CONTI_WINDOW,
    COOPER_WINDOW,
    drt_measurement,
    drt_rt,
)
from embodied_human.response import CentralTask, Reaction, Signal, TwinResponse


def fixture_human(times):
    values = iter(times)
    return SimpleNamespace(
        response=SimpleNamespace(
            respond=lambda *args: Reaction(next(values), "visual", False),
            visual_latency=lambda eccentricity: 0.1,
        ),
        touch=SimpleNamespace(latency=lambda level: 0.1),
    )


def test_study_windows_include_boundaries_and_exclude_other_responses():
    times = [0.099, 0.1, 2.0, 2.001, 2.5, 2.501, math.inf]
    conti = drt_measurement(
        fixture_human(times), ["head_light"], [], samples=7, window=CONTI_WINDOW
    )
    cooper = drt_measurement(
        fixture_human(times), ["tactile"], [], samples=7, window=COOPER_WINDOW
    )
    assert conti["mean_rt_ms"] == pytest.approx(1050)
    assert conti["hit_rate"] == pytest.approx(2 / 7)
    assert cooper["mean_rt_ms"] == pytest.approx(1000 * (0.1 + 2 + 2.001 + 2.5) / 4)
    assert cooper["hit_rate"] == pytest.approx(4 / 7)
    assert conti["premature_rate"] == pytest.approx(1 / 7)
    assert conti["late_rate"] == pytest.approx(3 / 7)
    assert conti["nonresponse_rate"] == pytest.approx(1 / 7)
    assert math.isinf(conti["uncensored_mean_rt_ms"])
    assert conti["uncensored_finite_mean_rt_ms"] == pytest.approx(
        1000 * np.mean(times[:-1])
    )


def test_hit_probability_weights_stimulus_mixture_means():
    measured = drt_measurement(
        fixture_human([0.2, 3, 0.4, 0.6]), ["tactile", "forward_light"], [],
        samples=2, window=COOPER_WINDOW,
    )
    assert measured["mean_rt_ms"] == pytest.approx(400)
    assert measured["hit_rate"] == pytest.approx(0.75)
    assert measured["uncensored_mean_rt_ms"] == pytest.approx(1050)
    assert [row["weight"] for row in measured["components"]] == [0.5, 0.5]


def test_hit_probability_weights_gaze_mixture_means():
    measured = drt_measurement(
        fixture_human([0.2, 3, 0.4, 0.6]), ["head_light"], [], on_display=0.25,
        samples=2, window=COOPER_WINDOW,
    )
    assert measured["mean_rt_ms"] == pytest.approx(1000 * 0.2 / 0.625)
    assert measured["hit_rate"] == pytest.approx(0.625)
    assert [row["weight"] for row in measured["components"]] == [0.75, 0.25]


def test_no_hits_is_undefined_and_retains_uncensored_model_times():
    measured = drt_measurement(
        fixture_human([0.05, 3]), ["tactile"], [], samples=2
    )
    assert math.isnan(measured["mean_rt_ms"])
    assert measured["hit_rate"] == 0
    assert measured["uncensored_mean_rt_ms"] == pytest.approx(1525)


def test_filtering_does_not_change_response_draws_or_the_rt_wrapper():
    human = build_human(seed=5)
    measured = drt_measurement(human, ["head_light"], [], samples=100)
    human.reset(5)
    assert drt_rt(human, ["head_light"], [], samples=100) == measured["mean_rt_ms"]
    human.reset(5)
    uncensored = drt_measurement(
        human, ["head_light"], [], samples=100, window=(0, 100)
    )
    assert uncensored["uncensored_mean_rt_ms"] == measured["uncensored_mean_rt_ms"]


def test_drt_benchmarks_report_study_windows_and_uncensored_diagnostics():
    from embodied_human.bench.load import drt_load
    from embodied_human.bench.senses import drt

    details = drt(build_human(seed=0)).details
    assert details["visual_measurement"]["window_s"] == list(CONTI_WINDOW)
    assert details["tactile_measurement"]["window_s"] == list(COOPER_WINDOW)
    assert details["older_tactile_measurement"]["window_s"] == list(COOPER_WINDOW)
    for measurement in (
        details["visual_measurement"], details["tactile_measurement"]
    ):
        assert 0 <= measurement["hit_rate"] <= 1
        assert math.isfinite(measurement["uncensored_mean_rt_ms"])
    load = drt_load(build_human(seed=0)).details
    for name, measurement in load["conti_measurements"].items():
        assert measurement["window_s"] == list(CONTI_WINDOW)
        assert load["conti_model_ms"][name] == measurement["mean_rt_ms"]
    for name, measurement in load["cooper_measurements"].items():
        assert measurement["window_s"] == list(COOPER_WINDOW)
        assert load["cooper_model_ms"][name] == measurement["mean_rt_ms"]


def test_twin_exposes_response_stages_without_changing_completion_time():
    response = TwinResponse()
    response.reset(np.random.default_rng(1))
    reaction = response.respond(
        [Signal("visual", 0.02, 0.1), Signal("auditory", 0, 0.05)],
        [CentralTask("busy", 1, 0.2)],
    )
    assert reaction.time == reaction.peripheral + reaction.waited + reaction.central
    assert reaction.central == response.params.central - (
        response.params.facilitation if reaction.integrated else 0
    )
    absent = response.respond([])
    assert math.isinf(absent.time) and math.isinf(absent.peripheral)
    legacy = Reaction(0.4, "visual", False)
    assert legacy.peripheral is None and legacy.central is None


@pytest.mark.parametrize("changes", [
    {"stimuli": []}, {"stimuli": ["unknown"]}, {"on_display": -0.1},
    {"on_display": math.nan}, {"samples": 0}, {"samples": True},
    {"window": (0.5, 0.1)}, {"window": (0.1, math.inf)},
])
def test_measurement_rejects_invalid_protocol_inputs(changes):
    inputs = {"human": fixture_human([0.2]), "stimuli": ["tactile"], "load": []}
    with pytest.raises(ValueError):
        drt_measurement(**{**inputs, **changes})
