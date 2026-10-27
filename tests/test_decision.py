"""Native solver consistency checks, not validation against human performance."""

import math
import random
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest

from embodied_human.decision import PyDDMResponse
from embodied_human.response import Signal, TwinResponse
from embodied_human.tasks.choice import ChoiceCondition, run_choice_task


def test_pyddm_construction_and_legacy_detection_need_no_optional_backend():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.modules['pyddm'] = None; "
            "from embodied_human.decision import PyDDMResponse; "
            "from embodied_human.response import Signal; "
            "r = PyDDMResponse(); "
            "assert r.respond([Signal('auditory', 0, .05)]).time > 0",
        ],
        check=True,
        capture_output=True,
    )


def test_missing_backend_has_an_actionable_error(monkeypatch):
    def missing(name):
        raise ModuleNotFoundError("no pyddm", name="pyddm")

    monkeypatch.setattr("embodied_human.decision.import_module", missing)
    with pytest.raises(ImportError, match=r"embodied-human\[decision\]"):
        PyDDMResponse().solve_choice(1.0, 1.0)


def test_detection_path_is_exactly_the_existing_twin_path():
    baseline, choice = TwinResponse(central=0.3), PyDDMResponse(central=0.3)
    for module in (baseline, choice):
        module.reset(np.random.default_rng(7))
    signals = [Signal("auditory", 0.0, 0.05), Signal("visual", 0.0, 0.1)]
    assert [baseline.respond(signals) for _ in range(30)] == [
        choice.respond(signals) for _ in range(30)
    ]
    assert baseline.visual_latency(30.0) == choice.visual_latency(30.0)


def test_native_constant_drift_matches_first_passage_equations():
    native = pytest.importorskip("pyddm")
    response = PyDDMResponse()
    distribution = response.solve_choice(1.0, 12.0)
    # Bounds +/- B, initial state 0, dX = v dt + sigma dW.
    # These equations are analytic consistency checks, not human references.
    expected_accuracy = 1.0 / (1.0 + math.exp(-2.0))
    expected_mean = math.tanh(1.0) + 0.2
    assert distribution.upper_probability == pytest.approx(expected_accuracy, abs=1e-5)
    assert distribution.mean_rt == pytest.approx(expected_mean, abs=0.002)
    assert distribution.timeout_probability < 1e-5
    assert float(distribution.upper_density.sum() * distribution.dt) == pytest.approx(
        distribution.upper_probability
    )
    assert distribution.solver_version == native.__version__


def test_symmetry_drift_reversal_and_deadlines():
    pytest.importorskip("pyddm")
    response = PyDDMResponse()
    unbiased = response.solve_choice(0.0, 3.0)
    assert unbiased.upper_probability == pytest.approx(unbiased.lower_probability)
    positive = response.solve_choice(1.0, 1.003)
    negative = response.solve_choice(-1.0, 1.003)
    assert positive.upper_probability == pytest.approx(negative.lower_probability)
    assert positive.time[-1] == pytest.approx(1.003)
    short = response.solve_choice(1.0, 0.1)
    assert short.timeout_probability == pytest.approx(1.0)
    assert short.mean_rt is None and short.accuracy("upper", conditional=True) is None
    assert positive.timeout_probability < short.timeout_probability
    assert positive.accuracy("upper") < positive.accuracy("upper", conditional=True)
    assert sum(
        [
            positive.upper_probability,
            positive.lower_probability,
            positive.timeout_probability,
        ]
    ) == pytest.approx(1.0)


def test_seeded_samples_agree_with_native_masses_and_preserve_global_rng():
    pytest.importorskip("pyddm")
    response = PyDDMResponse()
    python_before, numpy_before = random.getstate(), np.random.get_state()
    distribution = response.solve_choice(0.7, 1.5)
    response.reset(np.random.default_rng(13))
    first = response.sample_choices(distribution, 20000)
    response.reset(np.random.default_rng(13))
    assert first == response.sample_choices(distribution, 20000)
    assert np.mean([draw.choice == "upper" for draw in first]) == pytest.approx(
        distribution.upper_probability, abs=0.012
    )
    assert np.mean([draw.timeout for draw in first]) == pytest.approx(
        distribution.timeout_probability, abs=0.012
    )
    assert all(draw.time is None or 0 <= draw.time <= 1.5 for draw in first)
    assert random.getstate() == python_before
    for before, after in zip(numpy_before, np.random.get_state()):
        np.testing.assert_equal(before, after)


def test_choice_task_reports_timeout_accuracy_and_solver_consistency():
    pytest.importorskip("pyddm")
    conditions = [
        ChoiceCondition(1.0, 2.0, "upper"),
        ChoiceCondition(1.0, 0.1, "upper"),
    ]
    rows = run_choice_task(PyDDMResponse(), conditions, trials=10000, seed=4)
    assert rows == run_choice_task(PyDDMResponse(), conditions, trials=10000, seed=4)
    assert rows[0]["accuracy"] == pytest.approx(
        rows[0]["predicted_accuracy"], abs=0.015
    )
    assert rows[1]["timeout_rate"] == rows[1]["predicted_timeout_rate"] == 1.0
    assert rows[1]["mean_rt_s"] is None
    assert "not human validation" in rows[0]["interpretation"]


def test_sampling_rejects_malformed_distributions():
    pytest.importorskip("pyddm")
    response = PyDDMResponse()
    distribution = response.solve_choice(1.0, 1.0)
    with pytest.raises(ValueError, match="aligned"):
        response.sample_choices(replace(distribution, upper_density=np.array([1.0])))
    with pytest.raises(ValueError, match="dimensional"):
        response.sample_choices(replace(distribution, time=np.array(0.1)))


@pytest.mark.parametrize(
    "params",
    [
        {"decision_bound": 0.0},
        {"decision_noise": -1.0},
        {"nondecision_time": -0.1},
        {"starting_position": 1.0},
        {"decision_dt": math.nan},
        {"decision_dx": 2.0},
    ],
)
def test_invalid_choice_protocol_is_rejected_before_optional_import(params):
    with pytest.raises(ValueError):
        PyDDMResponse(**params)
