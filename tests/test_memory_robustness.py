"""Memory boundary cases that affect simulation and parameter inference."""

import math

import numpy as np
import pytest

from embodied_human.memory import ActRMemory, MemoryParameters


@pytest.mark.parametrize("capacity", [-1, 1.5, True, math.nan])
def test_capacity_requires_a_non_negative_integer(capacity):
    with pytest.raises(ValueError, match="capacity"):
        ActRMemory(capacity=capacity)


@pytest.mark.parametrize(
    "overrides",
    [
        {"decay": -0.5},
        {"noise": -0.25},
        {"latency_factor": 0.0},
        {"latency_factor": -1.0},
        {"decay": math.inf},
        {"noise": math.nan},
        {"threshold": -math.inf},
        {"latency_factor": math.nan},
    ],
)
def test_invalid_parameters_fail_at_construction(overrides):
    with pytest.raises(ValueError):
        MemoryParameters(**overrides)


def test_recall_probability_handles_nearly_deterministic_noise():
    memory = ActRMemory(capacity=0, noise=1e-9)
    memory.encode("item", 1, 0.0)
    assert memory.recall_probability("item", 0.1) == 1.0
    assert memory.recall_probability("item", 1.0) == pytest.approx(0.5)
    assert memory.recall_probability("item", 10.0) == 0.0


def test_probability_matches_seeded_retrieval_frequency():
    memory = ActRMemory(capacity=0, noise=0.25)
    memory.reset(np.random.default_rng(42))
    memory.encode("item", 1, 0.0)
    time = 2.0
    predicted = memory.recall_probability("item", time)
    observed = np.mean([memory.retrieve("item", time).success for _ in range(4000)])
    assert observed == pytest.approx(predicted, abs=0.025)


@pytest.mark.parametrize("age", [1e-300, 1e300])
def test_base_level_is_stable_beyond_representable_power_sums(age):
    memory = ActRMemory(capacity=0, decay=2.0, noise=0.0)
    memory.encode("item", 1, 0.0)
    memory.encode("item", 1, 0.0)
    expected = math.log(2.0) - 2.0 * math.log(age)
    assert memory.base_level("item", age) == pytest.approx(expected)
    assert memory.recall_probability("item", age) == float(expected > 0.0)


def test_zero_decay_counts_traces_without_underflow():
    memory = ActRMemory(capacity=0, decay=0.0)
    memory.encode("item", 1, -1e308)
    assert memory.base_level("item", 1e308) == 0.0


def test_latency_combines_scale_before_exponentiation():
    memory = ActRMemory(capacity=0, threshold=-1000.0, latency_factor=1e-200)
    result = memory.retrieve("missing", 0.0)
    assert not result.success
    assert result.latency == pytest.approx(math.exp(math.log(1e-200) + 1000.0))
    assert math.isinf(ActRMemory(threshold=-1000.0).retrieve("missing", 0.0).latency)


@pytest.mark.parametrize("time", [math.nan, math.inf, -math.inf])
def test_nonfinite_times_cannot_poison_memory(time):
    memory = ActRMemory()
    with pytest.raises(ValueError, match="finite"):
        memory.encode("item", 1, time)
    for method in (memory.base_level, memory.retrieve, memory.recall_probability):
        with pytest.raises(ValueError, match="finite"):
            method("item", time)
    assert not memory.focus


def test_queries_cannot_retrieve_future_values_and_reset_restarts_time():
    memory = ActRMemory()
    memory.encode("item", "future", 10.0)
    for method in (memory.base_level, memory.retrieve, memory.recall_probability):
        with pytest.raises(ValueError, match="latest encoding"):
            method("item", 9.0)
    with pytest.raises(ValueError, match="latest encoding"):
        memory.encode("item", "past", 9.0)
    assert memory.retrieve("item", 10.0).value == "future"
    memory.reset()
    memory.encode("item", "new run", 0.0)
    assert memory.retrieve("item", 0.0).value == "new run"


def test_noise_provenance_distinguishes_example_from_fit():
    noise_source = MemoryParameters.SOURCES["noise"]
    assert noise_source.kind == "assumed"
    assert noise_source.doi == "https://arxiv.org/abs/2410.18055"
