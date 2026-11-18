"""Contextual fitted configurations preserve baseline values and source labels."""

import builtins
import copy
import json
from dataclasses import asdict
from typing import Any

import pytest

from embodied_human.attention import SamplerParameters
from embodied_human.core import HumanConfig
from embodied_human.driving_profiles import resolve_profile


def _profile(module: str = "attention.uncertainty_sampler") -> dict[str, Any]:
    return {
        "module": module,
        "context": "self paced in vehicle visual fixation",
        "role": "fitted_candidate",
        "parameters": {
            "return_sd": 0.2,
            "threshold_spread": 0.7,
            "maximum_glance": 8.0,
        },
        "configuration": {
            "human": HumanConfig().as_dict(),
            "driver": {"display_position": [15.0, -20.0], "assessment_fixations": 3},
            "reflex": {"backend": "HoldLane", "vehicle_dynamics": False},
        },
        "provenance": {"dataset": "synthetic profile contract", "subjects": ["a", "b"]},
        "test_used_for_parameter_selection": False,
    }


def test_profile_roundtrip_changes_only_instance_parameters_and_preserves_sources():
    baseline = asdict(SamplerParameters())
    sources = dict(SamplerParameters.SOURCES)
    profile = _profile()
    resolved = resolve_profile(profile, context=profile["context"])
    human = resolved.build_human(seed=4)
    assert human.attention.params.return_sd == 0.2
    assert human.attention.params.threshold_spread == 0.7
    assert human.attention.params.maximum_glance == 8
    assert asdict(SamplerParameters()) == baseline
    assert SamplerParameters.SOURCES == sources
    driver = resolved.make_driver_parameters()
    assert driver.display_position == (15.0, -20.0)
    assert driver.assessment_fixations == 3
    exported = json.loads(json.dumps(resolved.as_dict(), allow_nan=False))
    replay = resolve_profile(exported, context=profile["context"])
    assert replay.human_config.as_dict() == resolved.human_config.as_dict()
    assert replay.fitted_provenance() == resolved.fitted_provenance()
    assert {source["kind"] for source in replay.fitted_provenance()} == {"fitted"}
    profile["provenance"]["dataset"] = "mutated"
    profile["configuration"]["driver"]["display_position"][0] = 99
    assert resolved.provenance["dataset"] == "synthetic profile contract"
    assert resolved.make_driver_parameters().display_position[0] == 15


def test_an_explicit_matching_context_is_required():
    profile = _profile()
    with pytest.raises(TypeError):
        resolve_profile(profile)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="exactly match"):
        resolve_profile(profile, context="ordinary driving")
    with pytest.raises(ValueError, match="exactly match"):
        resolve_profile(profile, context="")


def test_grouped_report_requires_complete_evaluation_and_supported_schema():
    profile = _profile()
    report = {"schema_version": 1, "status": "evaluated", "profile": profile}
    assert (
        resolve_profile(report, context=profile["context"]).parameters["return_sd"]
        == 0.2
    )
    report["status"] = "incomplete_evaluation"
    with pytest.raises(ValueError, match="complete evaluated"):
        resolve_profile(report, context=profile["context"])
    for version in (2, True, 1.0):
        with pytest.raises(ValueError, match="schema"):
            resolve_profile(
                dict(profile, schema_version=version), context=profile["context"]
            )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "0.2"])
def test_nonfinite_or_nonnumeric_overrides_are_rejected(value: Any):
    profile = _profile()
    profile["parameters"]["return_sd"] = value
    with pytest.raises(ValueError):
        resolve_profile(profile, context=profile["context"])


def test_unknown_categories_and_inconsistent_overrides_are_rejected():
    profile = _profile()
    profile["configuration"]["human"]["unknown"] = {"name": "custom", "params": {}}
    with pytest.raises(ValueError, match="unknown module category"):
        resolve_profile(profile, context=profile["context"])
    profile = _profile()
    profile["parameter_overrides"] = {"return_sd": 0.9}
    with pytest.raises(ValueError, match="must agree"):
        resolve_profile(profile, context=profile["context"])
    profile = _profile()
    profile["test_used_for_parameter_selection"] = True
    with pytest.raises(ValueError, match="independent test"):
        resolve_profile(profile, context=profile["context"])


def test_seev_area_values_resolve_to_the_actual_area_features():
    profile = _profile("attention.seev")
    profile["context"] = "RightGazeUrban right turns with pooled n-back conditions"
    profile["parameters"] = {"area_value.road": -0.3, "area_value.right_mirror": -1.2}
    profile["configuration"]["human"]["attention"] = {
        "name": "seev",
        "params": {
            "areas": {"road": [0, 0, 1, 0], "right_mirror": [0, 0, 1, 0]},
            "salience": 0.0,
            "effort": 1.0,
            "expectancy": 1.0,
            "value": 1.0,
            "multiplicative": True,
        },
    }
    resolved = resolve_profile(profile, context=profile["context"])
    attention = resolved.build_human(seed=0).attention
    assert attention.areas["road"] == (0, 0, 1, -0.3)
    assert attention.areas["right_mirror"] == (0, 0, 1, -1.2)
    probabilities = attention.probabilities()
    assert sum(probabilities.values()) == pytest.approx(1)
    assert probabilities["road"] / probabilities["right_mirror"] == pytest.approx(
        2.4596031111569494
    )
    profile["parameters"]["area_value.unknown"] = 1
    with pytest.raises(ValueError, match="lacks fixed features"):
        resolve_profile(profile, context=profile["context"])


def test_resolution_and_human_construction_do_not_import_optional_bridge(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.startswith(("myosuite_carla", "myosuite", "mujoco")):
            raise AssertionError("optional bridge imported during profile resolution")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    profile = _profile()
    resolved = resolve_profile(profile, context=profile["context"])
    assert resolved.build_human(seed=2).attention.params.return_sd == 0.2
    with pytest.raises(ValueError, match="physical reflex"):
        resolved.make_reflex_config()


def test_idm_overrides_remain_lazy_and_produce_native_bridge_configuration(monkeypatch):
    profile = _profile("reflex.idm")
    profile["context"] = "NGSIM car following within 16 m/s"
    profile["parameters"] = {"time_headway": 2.2, "min_gap": 2.4}
    profile["configuration"]["reflex"] = {
        "idm": {
            "max_acceleration": 1.0,
            "comfortable_deceleration": 1.5,
            "time_headway": 1.6,
            "min_gap": 2.0,
            "exponent": 4.0,
        },
        "steering": {"k_far": 6.0, "k_near": 2.0, "k_integral": 1.0},
        "throttle_map": [[0.0, 0.0], [1.0, 0.08]],
    }
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.startswith("myosuite_carla"):
            raise AssertionError("optional bridge imported before native factory")
        return original(name, *args, **kwargs)

    with monkeypatch.context() as check:
        check.setattr(builtins, "__import__", guarded)
        resolved = resolve_profile(profile, context=profile["context"])
    bridge = pytest.importorskip("myosuite_carla.driver_models")
    native = resolved.make_reflex_config()
    assert isinstance(native, bridge.ReflexDriverConfig)
    assert native.idm.time_headway == 2.2
    assert native.idm.min_gap == 2.4
    assert native.throttle_map == ((0.0, 0.0), (1.0, 0.08))
    assert resolved.human_config == HumanConfig()
    invalid = copy.deepcopy(profile)
    invalid["parameters"]["max_acceleration"] = 0.0
    with pytest.raises(ValueError, match="physical domain"):
        resolve_profile(invalid, context=invalid["context"])


def test_retained_baselines_do_not_acquire_a_new_fitted_source_label():
    profile = _profile()
    profile["role"] = "retained_baseline"
    resolved = resolve_profile(profile, context=profile["context"])
    assert {source["kind"] for source in resolved.fitted_provenance()} == {"baseline"}
