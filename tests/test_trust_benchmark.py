"""Selected trust models, fixed overrides and the preserved Kraus baseline."""

from dataclasses import dataclass

import numpy as np
import pytest
from scipy.optimize import least_squares

from embodied_human import REGISTRY, HumanConfig, ModuleChoice, Parameters, build_human
from embodied_human.bench.motivation import (
    KRAUS_STUDY1,
    KRAUS_STUDY2,
    fit_kraus,
    kraus_dynamics,
    kraus_events,
    trust_course,
)
from embodied_human.bench.protocol import benchmark_seed
from embodied_human.trust import DecisionFieldTrust, Trust


def legacy_default_result():
    """Frozen former baseline protocol, independent of selected model routing."""

    def course(initial, study, group, capability, opacity):
        model = DecisionFieldTrust(
            noise_sd=0.0, automation_belief=(initial - 1.0) / 6.0, opacity=opacity
        )
        ratings = [initial]
        for events in kraus_events(study, group):
            for event in events:
                model.update(capability[event], None)
            ratings.append(1.0 + 6.0 * model.trust)
        return ratings

    def residuals(x):
        capability = dict(zip(("ok", "tor", "malfunction"), x[:3]))
        return [
            predicted - observed
            for group, ratings in KRAUS_STUDY1.items()
            for predicted, observed in zip(
                course(ratings[0], 1, group, capability, x[3])[1:], ratings[1:]
            )
        ]

    solution = least_squares(
        residuals, [0.85, 0.6, 0.3, 3.0], bounds=([0, 0, 0, 1], [1, 1, 1, 50])
    )
    capability = dict(zip(("ok", "tor", "malfunction"), solution.x[:3]))
    predicted = {
        group: course(observed[0], 2, group, capability, solution.x[3])
        for group, observed in KRAUS_STUDY2.items()
    }
    errors = [
        model - observed
        for group in ("MF-/LT", "MF+/LT")
        for model, observed in zip(predicted[group][1:], KRAUS_STUDY2[group][1:])
    ]
    return (
        predicted,
        {
            "trust.study2_rmse": float(np.sqrt(np.mean(np.square(errors)))),
            "trust.malfunction_dip": predicted["MF+/LT"][5] - predicted["MF+/LT"][4],
        },
        {
            **capability,
            "opacity": solution.x[3],
            "study1_rmse": float(np.sqrt(np.mean(np.square(solution.fun)))),
        },
    )


def test_default_kraus_predictions_and_fitted_values_are_exactly_preserved():
    predicted, metrics, fitted = legacy_default_result()
    result = kraus_dynamics(build_human(seed=0))
    assert result.metrics == metrics
    assert result.details["fitted_on_study1"] == fitted
    assert result.details["study2_model"] == predicted
    assert {entry["role"] for entry in result.validation.values()} == {"held_out"}


def test_inertia_overrides_change_selected_model_predictions():
    metrics = [
        kraus_dynamics(
            build_human(
                HumanConfig().replace("trust", "decision_field", inertia=inertia),
                seed=0,
            )
        ).metrics
        for inertia in (0.1, 0.9)
    ]
    assert abs(metrics[0]["trust.study2_rmse"] - metrics[1]["trust.study2_rmse"]) > 0.01


def test_explicit_opacity_and_initial_belief_are_fixed_and_reported():
    config = HumanConfig().replace(
        "trust", "decision_field", opacity=1.25, automation_belief=0.4
    )
    result = kraus_dynamics(build_human(config, seed=0))
    assert result.details["fitted_on_study1"]["opacity"] == 1.25
    assert "opacity" not in result.details["study1_fitted_parameters"]
    assert result.details["requested_trust_parameters"] == {
        "opacity": 1.25,
        "automation_belief": 0.4,
    }
    assert all(
        course[0] == pytest.approx(3.4)
        for course in result.details["study2_model"].values()
    )


def test_registered_model_without_opacity_is_used_and_its_settings_survive(monkeypatch):
    @dataclass(frozen=True)
    class CustomParameters(Parameters):
        learning: float = 0.25
        automation_belief: float = 0.7

    class CustomTrust(Trust):
        def __init__(self, **params):
            super().__init__()
            self.params = CustomParameters(**params)
            self.reset()

        def reset(self, rng=None):
            super().reset(rng)
            self.trust = self.params.automation_belief

        def update(self, automation_capability, manual_capability):
            self.trust += self.params.learning * (automation_capability - self.trust)
            return "automation"

    monkeypatch.setitem(REGISTRY["trust"], "custom_kraus", CustomTrust)
    config = HumanConfig().replace("trust", "custom_kraus", learning=0.0)
    result = kraus_dynamics(build_human(config, seed=0))
    assert result.details["trust_module"] == "custom_kraus"
    assert "opacity" not in result.details["fitted_on_study1"]
    for group, predicted in result.details["study2_model"].items():
        np.testing.assert_allclose(predicted, KRAUS_STUDY2[group][0])


def test_explicit_noise_uses_local_protocol_seed():
    choice = ModuleChoice("decision_field", {"noise_sd": 0.1})
    args = (4.6, [["ok"]] * 5, {"ok": 0.9})
    before = np.random.get_state()
    with benchmark_seed(4):
        first = trust_course(*args, choice=choice)
    with benchmark_seed(4):
        repeated = trust_course(*args, choice=choice)
    with benchmark_seed(5):
        changed = trust_course(*args, choice=choice)
    assert first == repeated and first != changed
    for a, b in zip(before, np.random.get_state()):
        np.testing.assert_equal(a, b)


@pytest.mark.parametrize(
    "params,error",
    [
        ({"unknown_setting": 1.0}, TypeError),
        ({"inertia": 1.5}, ValueError),
        ({"noise_sd": float("nan")}, ValueError),
        ({"noise_sd": -1.0}, ValueError),
    ],
)
def test_invalid_parameters_are_not_replaced_by_protocol_defaults(params, error):
    with pytest.raises(error):
        fit_kraus(choice=ModuleChoice("decision_field", params))
