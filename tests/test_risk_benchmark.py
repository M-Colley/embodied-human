"""Risk overrides remain fixed while unspecified nuisance settings are fitted."""

import math
from dataclasses import dataclass

import numpy as np
import pytest
from scipy.optimize import least_squares

from embodied_human import REGISTRY, HumanConfig, ModuleChoice, Parameters, build_human
from embodied_human.bench.motivation import (
    LANE_WIDTH_SPEED,
    fit_lane_widths,
    lane_width,
)
from embodied_human.bench.protocol import benchmark_seed
from embodied_human.risk import DriverRiskField, Risk


def legacy_default_fit():
    """Frozen original two parameter calibration for baseline preservation."""

    def model(x):
        return DriverRiskField(width_rate=x[0], threshold=math.exp(x[1]))

    def residuals(x):
        risk = model(x)
        return [3.6 * risk.preferred_speed(w) - LANE_WIDTH_SPEED[w] for w in (3.0, 3.5)]

    best = None
    for start in (0.001, 0.02, 0.05, 0.1):
        risk = model(np.array([start, 0.0])).lane_risk(LANE_WIDTH_SPEED[3.5] / 3.6, 3.5)
        solution = least_squares(
            residuals, [start, math.log(risk)], bounds=([0.0, -10.0], [1.0, 30.0])
        )
        if best is None or solution.cost < best.cost:
            best = solution
    return model(best.x)


def test_default_risk_calibration_and_predictions_are_exactly_preserved():
    legacy = legacy_default_fit()
    result = lane_width(build_human(seed=0))
    assert result.details["width_rate"] == legacy.params.width_rate
    assert result.details["threshold"] == legacy.params.threshold
    for width, target in LANE_WIDTH_SPEED.items():
        assert result.details["speed_kmh"][str(width)] == {
            "model": 3.6 * legacy.preferred_speed(width),
            "human": target,
        }
    assert result.details["fitted_parameters"] == ["width_rate", "threshold"]
    assert result.validation["risk.lane_width_speed_error"]["role"] == "held_out"


@pytest.mark.parametrize("fixed", [{"width_rate": 0.02}, {"threshold": 5000.0}])
def test_single_override_is_fixed_while_the_other_parameter_is_fitted(fixed):
    tuned = fit_lane_widths(ModuleChoice("risk_field", fixed))
    name, value = next(iter(fixed.items()))
    assert getattr(tuned.params, name) == value
    result = lane_width(
        build_human(HumanConfig().replace("risk", "risk_field", **fixed))
    )
    assert name not in result.details["fitted_parameters"]
    assert result.details["requested_risk_parameters"] == fixed
    assert result.details["fitted_to_m"] == [3.0, 3.5]


def test_two_overrides_are_used_directly_without_calibration():
    configs = [
        HumanConfig().replace("risk", "risk_field", width_rate=0.001, threshold=1.0),
        HumanConfig().replace("risk", "risk_field", width_rate=0.1, threshold=1000.0),
    ]
    metrics = []
    for config in configs:
        human = build_human(config, seed=0)
        result = lane_width(human)
        assert result.details["fitted_parameters"] == []
        assert result.details["fitted_to_m"] == []
        assert result.details["width_rate"] == config.risk.params["width_rate"]
        assert result.details["threshold"] == config.risk.params["threshold"]
        for width in LANE_WIDTH_SPEED:
            assert result.details["speed_kmh"][str(width)]["model"] == (
                3.6 * human.risk.preferred_speed(width)
            )
        metrics.append(result.metrics["risk.lane_width_speed_error"])
    assert metrics[0] != metrics[1]


def test_selected_implementation_is_used_for_explicit_settings(monkeypatch):
    class DifferentRisk(DriverRiskField):
        def preferred_speed(self, lane_width, offset=0.0):
            return super().preferred_speed(lane_width, offset) + 1.0

    monkeypatch.setitem(REGISTRY["risk"], "different_risk", DifferentRisk)
    config = HumanConfig().replace(
        "risk", "different_risk", width_rate=0.001, threshold=1000.0
    )
    result = lane_width(build_human(config, seed=0))
    reference = DriverRiskField(width_rate=0.001, threshold=1000.0)
    assert result.details["risk_module"] == "different_risk"
    for width in LANE_WIDTH_SPEED:
        assert result.details["speed_kmh"][str(width)]["model"] == (
            3.6 * (reference.preferred_speed(width) + 1.0)
        )


def test_selected_stochastic_risk_uses_the_declared_protocol_seed(monkeypatch):
    class NoisyRisk(DriverRiskField):
        def preferred_speed(self, lane_width, offset=0.0):
            return super().preferred_speed(lane_width, offset) + self.rng.normal(
                0, 0.01
            )

    monkeypatch.setitem(REGISTRY["risk"], "seeded_risk", NoisyRisk)
    config = HumanConfig().replace(
        "risk", "seeded_risk", width_rate=0.001, threshold=1000.0
    )
    values = []
    for seed in (4, 4, 5):
        with benchmark_seed(seed):
            values.append(lane_width(build_human(config, seed=seed)).metrics)
    assert values[0] == values[1]
    assert values[0] != values[2]


def test_incompatible_calibration_is_skipped_with_an_explicit_reason(monkeypatch):
    @dataclass(frozen=True)
    class OtherParameters(Parameters):
        speed: float = 15.0

    class OtherRisk(Risk):
        def __init__(self):
            super().__init__()
            self.params = OtherParameters()

        def preferred_speed(self, lane_width, offset=0):
            return self.params.speed

    monkeypatch.setitem(REGISTRY["risk"], "other_risk", OtherRisk)
    config = HumanConfig().replace("risk", "other_risk")
    result = lane_width(build_human(config, seed=0))
    assert result.skipped and "requires width_rate and threshold" in result.skipped
    assert not result.scores


@pytest.mark.parametrize(
    "settings,error",
    [
        ({"unknown_setting": 1.0}, TypeError),
        ({"width_rate": math.nan}, ValueError),
        ({"threshold": math.inf}, ValueError),
    ],
)
def test_invalid_overrides_are_not_replaced_during_calibration(settings, error):
    with pytest.raises(error):
        fit_lane_widths(ModuleChoice("risk_field", settings))
