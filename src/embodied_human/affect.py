"""Emotion from appraisal of what the person has learned to expect.

Baseline :class:`AppraisalAffect` reimplements the model of Zhang,
Hilpert, Broekens and Jokinen (CHI 2024; IEEE Transactions on Affective
Computing 2025) from its published equations. A reinforcement learner's
own quantities are appraised as in Scherer's component process model:

- suddenness: how unexpected the transition was under a learned
  transition model, ``1 - T(s, a, s') / sum T(s, a, .)``;
- goal relevance: ``min(1, |delta|)``;
- goal conduciveness: ``clip(delta, -1, 1) / 2 + 1/2``;
- power: how much the best (or worst) action differs from the average,
  ``|max Q - mean Q|`` or ``|min Q - mean Q|`` over the highest ``|Q|``,

where ``delta = alpha (r + gamma max Q(s') - Q(s, a))`` is the learning
rate times the TD error. Emotions are then recognised from the appraisal
vector by a linear support vector classifier (scikit-learn) trained on
appraisal profiles of modal emotions (Scherer, 2001); its class
probabilities are the emotion intensities. The released code of the
original has no licence, so nothing of it is copied here.
"""

from __future__ import annotations

import inspect
import warnings
from collections import defaultdict
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from .core import Module, Parameters, Source, register

_CHI = (
    "Zhang, Hilpert, Broekens & Jokinen (2024)",
    "https://doi.org/10.1145/3613904.3641908",
)

#: Appraisal profiles (suddenness, goal relevance, conduciveness, power)
#: of modal emotions: CHI 2024 Table 1 for the first three, the IEEE TAC
#: 2025 table (arXiv 2309.06367v2, Table II) for the others.
PROFILES: dict[str, tuple[str, str, str, str]] = {
    "happiness": ("low", "medium", "high", "open"),
    "boredom": ("very low", "low", "open", "medium"),
    "irritation": ("low", "medium", "obstruct", "medium"),
    "joy": ("high", "high", "very high", "open"),
    "pride": ("open", "high", "high", "open"),
    "fear": ("high", "high", "obstruct", "very low"),
    "sadness": ("low", "high", "obstruct", "very low"),
    "shame": ("open", "high", "obstruct", "open"),
    "anxiety": ("low", "medium", "obstruct", "low"),
    "despair": ("high", "high", "obstruct", "very low"),
    "rage": ("high", "high", "obstruct", "high"),
}
APPRAISALS = ("suddenness", "goal_relevance", "conduciveness", "power")


def sample_level(level: str, size: int, rng: np.random.Generator) -> np.ndarray:
    """Draw appraisal values for a nominal level (CHI 2024 Table 2).

    "Medium" is centred on 0.5: the printed mean of 0 is inconsistent
    with the scale and with the authors' released code.
    """
    if level in ("obstruct", "very low"):
        return np.abs(rng.normal(0.0, 0.05, size))
    if level == "low":
        return np.abs(rng.normal(0.0, 0.1, size))
    if level == "medium":
        return rng.normal(0.5, 0.05, size)
    if level == "high":
        return 1.0 - np.abs(rng.normal(0.0, 0.1, size))
    if level == "very high":
        return 1.0 - np.abs(rng.normal(0.0, 0.05, size))
    if level == "open":
        return rng.uniform(0.0, 1.0, size)
    raise ValueError(f"unknown appraisal level {level!r}")


@dataclass(frozen=True)
class Appraisal:
    """One appraisal of a transition, each component in [0, 1]."""

    suddenness: float
    goal_relevance: float
    conduciveness: float
    power: float

    def vector(self) -> np.ndarray:
        return np.array(
            [self.suddenness, self.goal_relevance, self.conduciveness, self.power]
        )


def appraise(
    delta: float,
    q_next: Sequence[float],
    transitions: Mapping[Hashable, int],
    reached: Hashable,
) -> Appraisal:
    """Appraise one transition (CHI 2024 Eqs. 5-8).

    Args:
        delta: Learning rate times the TD error of the transition.
        q_next: Action values of the state the power is appraised in.
        transitions: Counts of successor states seen after (s, a).
        reached: The successor state actually reached.
    """
    total = sum(transitions.values())
    suddenness = 1.0 - transitions.get(reached, 0) / total if total else 0.0
    q = np.asarray(q_next, dtype=float)
    scale = float(np.max(np.abs(q))) if q.size else 0.0
    if scale == 0.0:
        power = 0.0
    elif abs(q.min()) < q.max():
        power = abs(q.max() - q.mean()) / scale
    else:
        power = abs(q.min() - q.mean()) / scale
    return Appraisal(
        suddenness=float(suddenness),
        goal_relevance=float(min(1.0, abs(delta))),
        conduciveness=float(min(max(delta, -1.0), 1.0) * 0.5 + 0.5),
        power=float(power),
    )


@dataclass(frozen=True)
class AffectParameters(Parameters):
    """Appraisal reinforcement learner and emotion classifier.

    Args:
        learning_rate: alpha while learning the task.
        discount: gamma.
        exploration: epsilon of the epsilon-greedy policy.
        penalty: Mean SVM penalty C; its SD is ``penalty_sd``.
        penalty_sd: Between-person SD of C.
        training_vectors: Synthetic appraisal vectors per emotion.
        emotions: Emotions the classifier distinguishes.
    """

    learning_rate: float = 0.1
    discount: float = 0.9
    exploration: float = 0.3
    penalty: float = 0.0049
    penalty_sd: float = 0.0002
    training_vectors: int = 400
    emotions: tuple[str, ...] = ("happiness", "boredom", "irritation")

    SOURCES: ClassVar = {
        "learning_rate": Source(*_CHI, "fitted", "released model settings"),
        "discount": Source(*_CHI, "fitted"),
        "exploration": Source(*_CHI, "fitted"),
        "penalty": Source(
            *_CHI,
            "fitted",
            "chosen so that classifier precision matches human emotion "
            "recognition (mean 0.626)",
        ),
        "penalty_sd": Source(
            *_CHI,
            "fitted",
            "the text says variance 0.0004; the released model draws SD 0.0002",
        ),
        "training_vectors": Source(*_CHI, "fitted"),
        "emotions": Source(*_CHI, "standard", "Scherer (2001) modal emotions"),
    }


def _svc_has_probability() -> bool:
    """Whether scikit-learn's SVC still offers Platt probabilities (removed
    in 1.11; deprecated in 1.9)."""
    from sklearn.svm import SVC

    return "probability" in inspect.signature(SVC).parameters


@dataclass
class TaskModel:
    """A small Markov decision process for appraisal studies.

    ``transitions[state][action]`` lists ``(probability, next_state,
    reward)``, the reward being paid on entering ``next_state``. An
    absorbing state has the single action ``frwd`` looping back to itself:
    entering it pays its reward, and one more loop pays its
    ``absorbing`` reward and ends the episode, so terminal values count
    twice as in the original tasks.
    """

    transitions: dict[Hashable, dict[Hashable, list[tuple[float, Hashable, float]]]]
    start: Hashable
    absorbing: dict[Hashable, float] = field(default_factory=dict)

    def actions(self, state: Hashable) -> list[Hashable]:
        if state in self.absorbing:
            return ["frwd"]
        return list(self.transitions.get(state, {}))

    def step(
        self, state: Hashable, action: Hashable, rng: np.random.Generator
    ) -> tuple[Hashable, float]:
        outcomes = self.transitions[state][action]
        index = rng.choice(len(outcomes), p=[o[0] for o in outcomes])
        _, successor, reward = outcomes[index]
        return successor, reward


class Affect(Module):
    """Category ``affect``: emotions from what happens during a task."""

    category = "affect"

    def emotions(self, appraisal: Appraisal) -> dict[str, float]:
        """Emotion intensities, summing to 1, for an appraisal."""
        raise NotImplementedError


@register("affect", "appraisal")
class AppraisalAffect(Affect):
    """Appraisal of a Q-learner's values, recognised as emotions."""

    def __init__(self, **params: Any) -> None:
        super().__init__()
        self.params = AffectParameters(**params)
        self._classifier: Any = None
        self.reset()

    def reset(self, rng: np.random.Generator | None = None) -> None:
        super().reset(rng)
        self.q: dict[Hashable, dict[Hashable, float]] = defaultdict(dict)
        self.counts: dict[tuple[Hashable, Hashable], dict[Hashable, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self._classifier = None

    # Learning ---------------------------------------------------------------

    def value(self, state: Hashable, task: TaskModel) -> list[float]:
        """Action values of a state (0 for untried actions)."""
        return [self.q[state].get(a, 0.0) for a in task.actions(state)]

    def choose(self, state: Hashable, task: TaskModel) -> Hashable:
        """Epsilon-greedy; ties go to the first action listed."""
        actions = task.actions(state)
        if self.rng.random() < self.params.exploration:
            return actions[int(self.rng.integers(len(actions)))]
        values = self.value(state, task)
        return actions[int(np.argmax(values))]

    def learn_step(
        self,
        state: Hashable,
        action: Hashable,
        reward: float,
        successor: Hashable,
        task: TaskModel,
        learning_rate: float | None = None,
        terminal: bool = False,
    ) -> float:
        """One Q-learning update; returns delta (alpha times TD error)."""
        alpha = self.params.learning_rate if learning_rate is None else learning_rate
        future = [] if terminal else self.value(successor, task)
        target = reward + (self.params.discount * max(future) if future else 0.0)
        delta = alpha * (target - self.q[state].get(action, 0.0))
        self.q[state][action] = self.q[state].get(action, 0.0) + delta
        self.counts[(state, action)][successor] += 1
        return float(delta)

    def run_episode(
        self,
        task: TaskModel,
        forced: Mapping[Hashable, Hashable] | None = None,
        appraise_at: Mapping[tuple[Hashable, Hashable], float] | None = None,
        power_states: Mapping[tuple[Hashable, Hashable], Hashable] | None = None,
    ) -> list[Appraisal]:
        """One episode of Q-learning from the start state.

        Args:
            forced: Action taken in a state regardless of the policy.
            appraise_at: Transitions (state, successor) to appraise, with
                the learning rate used for their update.
            power_states: State whose values the power of an appraised
                transition uses (the decision state by default).
        """
        forced = forced or {}
        appraise_at = appraise_at or {}
        appraisals: list[Appraisal] = []
        state = task.start
        while True:
            action = forced.get(state) or self.choose(state, task)
            if state in task.absorbing:
                self.learn_step(
                    state, action, task.absorbing[state], state, task, terminal=True
                )
                return appraisals
            successor, reward = task.step(state, action, self.rng)
            key = (state, successor)
            if key in appraise_at:
                power_state = (power_states or {}).get(key, state)
                appraisals.append(
                    self.appraise_step(
                        state,
                        action,
                        reward,
                        successor,
                        task,
                        power_state,
                        appraise_at[key],
                    )
                )
            else:
                self.learn_step(state, action, reward, successor, task)
            state = successor

    def train(self, task: TaskModel, episodes: int = 20000) -> None:
        """Learn a task with epsilon-greedy Q-learning."""
        for _ in range(episodes):
            self.run_episode(task)

    def appraise_step(
        self,
        state: Hashable,
        action: Hashable,
        reward: float,
        successor: Hashable,
        task: TaskModel,
        power_state: Hashable | None = None,
        learning_rate: float | None = None,
    ) -> Appraisal:
        """Learn from one transition and appraise it; the transition is
        counted before its suddenness is computed."""
        delta = self.learn_step(state, action, reward, successor, task, learning_rate)
        return appraise(
            delta,
            self.value(state if power_state is None else power_state, task),
            dict(self.counts[(state, action)]),
            successor,
        )

    # Emotions -----------------------------------------------------------------

    def _fit(self) -> Any:
        from sklearn.svm import SVC

        p = self.params
        vectors, labels = [], []
        for emotion in p.emotions:
            levels = PROFILES[emotion]
            columns = [
                sample_level(level, p.training_vectors, self.rng) for level in levels
            ]
            vectors.append(np.column_stack(columns))
            labels += [emotion] * p.training_vectors
        penalty = max(
            1e-6, p.penalty + p.penalty_sd * float(self.rng.standard_normal())
        )
        seed = int(self.rng.integers(2**31))
        if _svc_has_probability():
            # Platt-scaled pairwise probabilities, as in the original model.
            classifier = SVC(
                kernel="linear", C=penalty, probability=True, random_state=seed
            )
        else:
            from sklearn.calibration import CalibratedClassifierCV

            classifier = CalibratedClassifierCV(
                SVC(kernel="linear", C=penalty, random_state=seed), ensemble=False
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            classifier.fit(np.vstack(vectors), labels)
        return classifier

    def emotions(self, appraisal: Appraisal) -> dict[str, float]:
        if self._classifier is None:
            self._classifier = self._fit()
        probabilities = self._classifier.predict_proba(appraisal.vector()[None, :])[0]
        return dict(zip(self._classifier.classes_, map(float, probabilities)))


@dataclass
class EmotionTrace:
    """Appraisals and emotions over an episode."""

    appraisals: list[Appraisal] = field(default_factory=list)
    emotions: list[dict[str, float]] = field(default_factory=list)


def _task(
    rows: Mapping[str, Mapping[str, list[tuple[float, str, float]]]],
    absorbing: Mapping[str, float],
) -> TaskModel:
    transitions: dict[Hashable, dict[Hashable, list[tuple[float, Hashable, float]]]] = {
        state: {action: list(outcomes) for action, outcomes in actions.items()}
        for state, actions in rows.items()
    }
    terminal: dict[Hashable, float] = {k: v for k, v in absorbing.items()}
    return TaskModel(transitions, "S", terminal)


def _outcome(p: float, likely: tuple[str, float], other: tuple[str, float]) -> list:
    if p >= 1.0:
        return [(1.0, *likely)]
    return [(p, *likely), (1 - p, *other)]


def emotion_study_tasks(appraisal: bool = False) -> dict[str, TaskModel]:
    """The three tasks of Zhang et al. (2024), Experiment 1.

    Rewards are paid on entering a state: +5 at the goal ``G``, -5 at the
    error ``E``, -1 elsewhere (boredom: +1 or -1 at ``F`` by the action
    taken, -1 on its final loop). With ``appraisal`` the appraised outcome
    is made certain, as in the appraisal episode of the original study.
    """
    p = 1.0 if appraisal else 0.8
    happiness = _task(
        {
            "S": {"frwd": [(1.0, "S1", -1.0)]},
            "S1": {
                "a1": _outcome(p, ("G", 5.0), ("E", -5.0)),
                "a2": [(1.0, "E", -5.0)],
            },
        },
        {"G": 5.0, "E": -5.0},
    )
    boredom = _task(
        {
            "S": {"frwd": [(1.0, "S1", -1.0)]},
            "S1": {"a1": [(1.0, "F", 1.0)], "a2": [(1.0, "F", -1.0)]},
        },
        {"F": -1.0},
    )
    irritation = _task(
        {
            "S": {"frwd": [(1.0, "S1", -1.0)]},
            "S1": {
                "a1": _outcome(p, ("P", -1.0), ("G", 5.0)),
                "a2": [(1.0, "E", -5.0)],
            },
            "P": {"frwd": [(1.0, "E", -5.0)]},
        },
        {"G": 5.0, "E": -5.0},
    )
    return {"happiness": happiness, "boredom": boredom, "irritation": irritation}


#: Appraised transition and its learning rate per Experiment 1 task.
EMOTION_STUDY_APPRAISALS = {
    "happiness": (("S1", "G"), 0.2),
    "boredom": (("S1", "F"), 0.1),
    "irritation": (("S1", "P"), 0.3),
}


def emotion_study(
    seed: int = 0, classifiers: int = 27, runs: int = 1, episodes: int = 20000
) -> dict[str, dict[str, float]]:
    """Simulate Experiment 1 of Zhang et al. (2024).

    For each task, train a fresh learner, run the appraisal episode with
    ``a1`` forced, and recognise emotions with ``classifiers`` simulated
    participants (SVMs with penalties drawn per person). Returns the mean
    intensities per task, averaged over ``runs`` learners.
    """
    rng = np.random.default_rng(seed)
    stories = emotion_study_tasks(appraisal=True)
    results: dict[str, dict[str, float]] = {}
    for name, task in emotion_study_tasks().items():
        transition, alpha = EMOTION_STUDY_APPRAISALS[name]
        totals: dict[str, float] = defaultdict(float)
        for _ in range(runs):
            learner = AppraisalAffect()
            learner.reset(np.random.default_rng(int(rng.integers(2**32))))
            learner.train(task, episodes)
            (appraisal,) = learner.run_episode(
                stories[name], forced={"S1": "a1"}, appraise_at={transition: alpha}
            )
            for _ in range(classifiers):
                person = AppraisalAffect()
                person.reset(np.random.default_rng(int(rng.integers(2**32))))
                for emotion, value in person.emotions(appraisal).items():
                    totals[emotion] += value / (classifiers * runs)
        results[name] = dict(totals)
    return results
