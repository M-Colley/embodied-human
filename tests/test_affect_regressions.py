"""Replacement modules must affect the study reported for that module."""

import numpy as np
import pytest

import embodied_human as eh
from embodied_human.affect import Affect, AppraisalAffect, TaskModel, emotion_study
from embodied_human.bench.motivation import emotion_study as benchmark_emotions
from embodied_human.core import ModuleChoice


@pytest.mark.parametrize("forced_action", [0, False, ""])
def test_falsy_forced_actions_are_taken(forced_action):
    task = TaskModel(
        {
            "start": {
                "other": [(1.0, "goal", -1.0)],
                forced_action: [(1.0, "goal", 2.0)],
            }
        },
        "start",
        {"goal": 0.0},
    )
    learner = AppraisalAffect(exploration=0.0)
    learner.reset(np.random.default_rng(0))
    learner.run_episode(task, forced={"start": forced_action})
    assert learner.q["start"] == {forced_action: pytest.approx(0.2)}
    assert ("start", "other") not in learner.counts


@pytest.fixture
def constant_affect(monkeypatch):
    class ConstantAffect(Affect):
        def __init__(self, happiness=0.1):
            super().__init__()
            self.happiness = happiness

        def emotions(self, appraisal):
            return {
                "happiness": self.happiness,
                "boredom": 1.0 - self.happiness,
                "irritation": 0.0,
            }

    monkeypatch.setitem(eh.REGISTRY["affect"], "test_constant", ConstantAffect)
    return ConstantAffect


def test_custom_classifier_and_overrides_change_benchmark_predictions(
    constant_affect, monkeypatch
):
    # Use the actual study with short training; classification is the
    # controlled variable, so extensive Q-learning is unnecessary here.
    def short_study(**kwargs):
        return emotion_study(classifiers=2, episodes=3, **kwargs)

    monkeypatch.setattr("embodied_human.affect.emotion_study", short_study)
    reports = []
    for happiness in (0.1, 0.9):
        config = eh.HumanConfig().replace(
            "affect", "test_constant", happiness=happiness
        )
        result = benchmark_emotions(eh.build_human(config, seed=0), runs=1)
        assert result.skipped is None
        assert result.details["affect_module"] == "test_constant"
        for values in result.details["model"].values():
            assert values["happiness"] == pytest.approx(happiness)
        assert "baseline" in result.details["learner"]
        reports.append(result.metrics["affect.emotion_rmse"])
    assert reports[0] != pytest.approx(reports[1])


def test_selected_appraisal_parameters_apply_to_learner_and_classifiers(monkeypatch):
    trained = []
    classified = []

    class RecordingAppraisal(AppraisalAffect):
        def train(self, task, episodes=20000):
            trained.append(self.params.learning_rate)
            super().train(task, episodes)

        def emotions(self, appraisal):
            classified.append(self.params.learning_rate)
            return {"happiness": self.params.learning_rate}

    monkeypatch.setitem(eh.REGISTRY["affect"], "test_recording", RecordingAppraisal)
    emotion_study(
        seed=0,
        classifiers=2,
        runs=1,
        episodes=3,
        choice=ModuleChoice("test_recording", {"learning_rate": 0.25}),
    )
    assert trained == [0.25] * 3
    assert classified == [0.25] * 6


def test_study_skips_classifiers_with_incompatible_emotion_labels(monkeypatch):
    monkeypatch.setattr(
        "embodied_human.affect.emotion_study",
        lambda **kwargs: {
            name: {"fear": 1.0} for name in ("happiness", "boredom", "irritation")
        },
    )
    result = benchmark_emotions(eh.build_human(seed=0), runs=1)
    assert result.skipped and "different labels" in result.skipped
    assert not result.passed
