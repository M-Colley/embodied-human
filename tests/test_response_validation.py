"""Response inputs reject impossible times and preserve nondetection signals."""

import math

import numpy as np
import pytest

from embodied_human.response import (
    CentralTask,
    Signal,
    TwinParameters,
    TwinResponse,
    central_wait,
)


@pytest.mark.parametrize("field", [
    "central", "facilitation", "window", "visual_latency", "visual_field",
])
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_twin_parameters_require_finite_values(field, value):
    with pytest.raises(ValueError, match="finite"):
        TwinParameters(**{field: value})


@pytest.mark.parametrize("values", [
    {"central": -1}, {"facilitation": -0.1}, {"facilitation": 0.3},
    {"window": -0.1}, {"visual_latency": 0}, {"visual_latency": -0.1},
    {"visual_field": -1},
])
def test_twin_constructor_rejects_nonphysical_parameters(values):
    with pytest.raises(ValueError):
        TwinResponse(**values)


def test_zero_central_window_and_visual_field_boundaries_remain_valid():
    response = TwinResponse(central=0, facilitation=0, window=0, visual_field=0)
    response.reset(np.random.default_rng(2))
    assert response.visual_latency(0) > 0
    assert math.isinf(response.visual_latency(1))
    reaction = response.respond([Signal("visual", 0, response.visual_latency(0))])
    assert reaction.time == reaction.peripheral > 0
    assert reaction.central == 0


@pytest.mark.parametrize("onset", [math.nan, math.inf, -math.inf])
def test_signal_onset_requires_a_finite_value(onset):
    with pytest.raises(ValueError, match="onset"):
        Signal("visual", onset, 0.1)


@pytest.mark.parametrize("latency", [math.nan, -math.inf, -0.1, 0])
def test_signal_latency_requires_positive_time_or_nondetection(latency):
    with pytest.raises(ValueError, match="latency"):
        Signal("auditory", 0, latency)


def test_signal_retains_negative_onset_and_positive_infinity_sentinel():
    signal = Signal("auditory", -0.02, 0.05)
    assert signal.onset == -0.02
    response = TwinResponse()
    absent = response.respond([Signal("visual", 0, math.inf)])
    assert math.isinf(absent.time)
    assert absent.first is None
    with pytest.raises(ValueError, match="modality"):
        Signal("unknown", 0, 0.1)


@pytest.mark.parametrize("period,duration", [
    (math.nan, 0.1), (1, math.nan), (math.inf, 0.1), (1, math.inf),
    (-math.inf, 0), (0, -math.inf), (-1, 0.1), (1, -0.1), (1, 1.1),
])
def test_central_task_rejects_nonphysical_intervals(period, duration):
    with pytest.raises(ValueError):
        CentralTask("invalid", period, duration)


def test_zero_period_preserves_inactive_task_and_random_stream():
    rng = np.random.default_rng(7)
    expected = np.random.default_rng(7)
    assert central_wait([CentralTask("inactive", 0, 3)], rng) == 0
    assert rng.random() == expected.random()


def test_continuously_busy_task_never_releases_the_central_stage():
    assert math.isinf(central_wait(
        [CentralTask("continuous", 0.5, 0.5)], np.random.default_rng(0)
    ))


def test_validation_preserves_seeded_valid_response_draws():
    response = TwinResponse()
    response.reset(np.random.default_rng(8))
    expected_rng = np.random.default_rng(8)
    signals = [Signal("auditory", 0, 0.05)]
    tasks = [CentralTask("driving", 0.7, 0.2)]
    for _ in range(10):
        # Existing channel draws use three modality slots, with audition first.
        peripheral = 0.05 * expected_rng.standard_exponential(3)[0]
        phase = expected_rng.uniform(0, 0.7)
        # One periodic busy interval has this exact residual waiting time.
        waited = max(0.2 - phase, 0)
        expected = peripheral + waited + response.params.central
        reaction = response.respond(signals, tasks)
        assert reaction.time == pytest.approx(expected, abs=1e-14)
        assert reaction.peripheral == peripheral
