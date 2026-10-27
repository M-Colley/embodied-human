import copy
import json
import math
import subprocess
import sys

import numpy as np
import pytest

from embodied_human import HumanConfig
from embodied_human.sensitivity import (
    analyze_design,
    sample_design,
    validate_sensitivity,
)


def request(method="morris", samples=32):
    return {
        "method": method,
        "parameters": [
            {"category": "memory", "parameter": "decay", "bounds": [0, 1]},
            {"category": "memory", "parameter": "noise", "bounds": [0, 1]},
        ],
        "benchmark": "memory.span",
        "metric": "memory.span",
        "samples": samples,
        "seed": 0,
        "num_levels": 4,
    }


def validated(method="morris", samples=32):
    return validate_sensitivity(
        request(method, samples), HumanConfig(), ["memory.span"]
    )


def test_validation_requires_no_optional_salib_installation():
    code = (
        "import sys; sys.modules['SALib'] = None; "
        "from embodied_human import HumanConfig; "
        "from embodied_human.sensitivity import validate_sensitivity; "
        f"spec = validate_sensitivity({request()!r}, HumanConfig(), ['memory.span']); "
        "assert spec['evaluations'] == 96; "
        "assert spec['parameters'][0]['bounds'] == [0.0, 1.0]"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_canonical_validation_recomputes_untrusted_evaluation_count():
    spec = validated()
    assert validate_sensitivity(spec, HumanConfig(), ["memory.span"]) == spec
    spec["evaluations"] = 0
    assert (
        validate_sensitivity(spec, HumanConfig(), ["memory.span"])["evaluations"] == 96
    )


def test_inherited_float_fields_and_bare_classvar_annotations_are_supported():
    data = request()
    data.update(
        benchmark="response.drt_load",
        metric="response.drt_2back_cost",
        parameters=[
            {"category": "touch", "parameter": "latency_gain", "bounds": [0.1, 0.2]},
            {"category": "touch", "parameter": "spread_radius", "bounds": [0.02, 0.05]},
        ],
    )
    config = HumanConfig().replace("touch", "contact_map")
    spec = validate_sensitivity(data, config, ["response.drt_load"])
    assert spec["evaluations"] == 96


@pytest.mark.parametrize("method", ["morris", "sobol"])
def test_actual_salib_design_is_deterministic_and_keeps_global_rng(method):
    pytest.importorskip("SALib")
    spec = validated(method)
    before = np.random.get_state()
    problem, first = sample_design(spec)
    second_problem, second = sample_design(spec)
    np.testing.assert_equal(first, second)
    assert problem == second_problem
    assert first.shape == (spec["evaluations"], 2)
    assert np.all((first >= 0) & (first <= 1))
    after = np.random.get_state()
    for a, b in zip(before, after):
        np.testing.assert_equal(a, b)


@pytest.mark.parametrize("method", ["morris", "sobol"])
def test_known_additive_effects_and_bootstrap_are_reproducible(method):
    pytest.importorskip("SALib")
    spec = validated(method, 256)
    problem, design = sample_design(spec)
    outputs = design[:, 0] + 2 * design[:, 1]
    before = np.random.get_state()
    result = analyze_design(spec, problem, design, outputs)
    repeated = analyze_design(spec, problem, design, outputs)
    assert result == repeated
    assert result["status"] == "ok"
    assert result["seed"] == 0
    assert result["parameter_names"] == ["memory.decay", "memory.noise"]
    assert result["second_order"] is False
    indices = result["indices"]
    if method == "morris":
        np.testing.assert_allclose(indices["mu"], [1, 2])
        np.testing.assert_allclose(indices["mu_star"], [1, 2])
        np.testing.assert_allclose(indices["sigma"], 0, atol=1e-14)
    else:
        # Uniform independent additive factors have variance shares 1/5 and 4/5.
        np.testing.assert_allclose(indices["S1"], [0.2, 0.8], atol=0.025)
        np.testing.assert_allclose(indices["ST"], [0.2, 0.8], atol=0.025)
        assert set(indices) == {"S1", "ST", "S1_conf", "ST_conf"}
    for a, b in zip(before, np.random.get_state()):
        np.testing.assert_equal(a, b)
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("method", ["morris", "sobol"])
def test_constant_metric_returns_explicit_undefined_indices(method):
    pytest.importorskip("SALib")
    spec = validated(method)
    problem, design = sample_design(spec)
    result = analyze_design(spec, problem, design, np.ones(len(design)))
    assert result["status"] == "undefined"
    assert "constant" in result["reason"]
    assert all(
        value is None for values in result["indices"].values() for value in values
    )
    json.dumps(result, allow_nan=False)


def test_nonfinite_outputs_are_not_treated_as_zero_sensitivity():
    pytest.importorskip("SALib")
    spec = validated()
    problem, design = sample_design(spec)
    outputs = design[:, 0].copy()
    outputs[0] = np.nan
    result = analyze_design(spec, problem, design, outputs)
    assert result["status"] == "undefined"
    assert "nonfinite" in result["reason"]


@pytest.mark.parametrize(
    "changes",
    [
        {"method": "other"},
        {"samples": 3},
        {"samples": 257},
        {"samples": True},
        {"method": "sobol", "samples": 6},
        {"seed": -1},
        {"seed": True},
        {"num_levels": 3},
        {"num_levels": 5},
        {"num_levels": 22},
        {"metric": "memory.unknown"},
        {"benchmark": "does.not.exist"},
        {"parameters": []},
        {"unknown": 1},
    ],
)
def test_invalid_sensitivity_settings_fail(changes):
    data = request()
    data.update(changes)
    with pytest.raises(ValueError):
        validate_sensitivity(data, HumanConfig(), [data["benchmark"]])


@pytest.mark.parametrize(
    "factor",
    [
        {"category": "memory", "parameter": "capacity", "bounds": [1, 8]},
        {"category": "memory", "parameter": "missing", "bounds": [0, 1]},
        {"category": "missing", "parameter": "decay", "bounds": [0, 1]},
        {"category": "memory", "parameter": "decay", "bounds": [1, 1]},
        {"category": "memory", "parameter": "decay", "bounds": [1, 0]},
        {"category": "memory", "parameter": "decay", "bounds": [0, math.inf]},
        {"category": "memory", "parameter": "decay", "bounds": [False, 1]},
        {"category": "memory", "parameter": "decay", "bounds": "01"},
        {"category": "memory", "parameter": "decay"},
    ],
)
def test_invalid_and_noncontinuous_factors_fail(factor):
    data = request()
    data["parameters"] = [factor]
    with pytest.raises(ValueError):
        validate_sensitivity(data, HumanConfig(), ["memory.span"])


def test_duplicate_factors_limits_and_exact_benchmark_selection():
    data = request()
    data["parameters"] = [data["parameters"][0]] * 2
    with pytest.raises(ValueError, match="unique"):
        validate_sensitivity(data, HumanConfig(), ["memory.span"])
    data["parameters"] *= 5
    with pytest.raises(ValueError, match="1 to 8"):
        validate_sensitivity(data, HumanConfig(), ["memory.span"])
    for benchmarks in ([], ["eyes.latency"], ["memory.span", "eyes.latency"]):
        with pytest.raises(ValueError, match="exactly"):
            validate_sensitivity(request(), HumanConfig(), benchmarks)


def test_analysis_rejects_problem_row_shape_and_bounds_mismatches():
    pytest.importorskip("SALib")
    spec = validated()
    problem, design = sample_design(spec)
    outputs = design[:, 0]
    wrong = copy.deepcopy(problem)
    wrong["names"][0] = "other.parameter"
    with pytest.raises(ValueError, match="problem"):
        analyze_design(spec, wrong, design, outputs)
    with pytest.raises(ValueError, match="ordered scalar"):
        analyze_design(spec, problem, design, outputs[:-1])
    with pytest.raises(ValueError, match="shape"):
        analyze_design(spec, problem, design[:-1], outputs)
    changed = design.copy()
    changed[0, 0] = -1
    with pytest.raises(ValueError, match="outside"):
        analyze_design(spec, problem, changed, outputs)


def test_sampling_missing_dependency_error_is_actionable(monkeypatch):
    spec = validated()
    monkeypatch.setitem(sys.modules, "SALib", None)
    # Ensure cached child modules cannot hide a missing installation in this test.
    monkeypatch.setitem(sys.modules, "SALib.sample.morris", None)
    with pytest.raises(ImportError, match="pip install"):
        sample_design(spec)
