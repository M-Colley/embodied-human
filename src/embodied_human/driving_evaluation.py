"""Participant grouped empirical comparisons and parameter calibration.

Inputs contain one summary per original human participant or source track.
Trial summaries and simulation replicates must first be averaged within that
source. The module keeps those sources intact during partitioning, gives them
equal weight, and bootstraps paired source errors. It has no pandas dependency.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .inference import Prior, abc_rejection

SubjectSummaries = Mapping[str, Mapping[str, float | None]]
Simulator = Callable[[Mapping[str, float], tuple[str, ...], int], SubjectSummaries]


@dataclass(frozen=True)
class GroupSplit:
    """Original source identifiers assigned to three disjoint partitions."""

    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]
    seed: int = 0

    def __post_init__(self) -> None:
        for name in ("train", "validation", "test"):
            value = getattr(self, name)
            if isinstance(value, str):
                raise ValueError("partitions must contain sequences of source IDs")
            object.__setattr__(self, name, tuple(value))
        groups = self.train + self.validation + self.test
        if not self.train or not self.validation or not self.test:
            raise ValueError("train, validation and test must each contain a group")
        if any(not isinstance(g, str) or not g.strip() for g in groups):
            raise ValueError("source identifiers must be nonempty strings")
        if len(set(groups)) != len(groups):
            raise ValueError("a source group occurs in more than one partition")


def group_split(
    groups: Sequence[str],
    *,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
    seed: int = 0,
) -> GroupSplit:
    """Partition complete sources with a seeded permutation of sorted IDs."""
    if (
        not math.isfinite(train_fraction)
        or not math.isfinite(validation_fraction)
        or train_fraction <= 0
        or validation_fraction <= 0
        or train_fraction + validation_fraction >= 1
    ):
        raise ValueError("partition fractions must be positive and leave a test set")
    if any(not isinstance(g, str) or not g.strip() for g in groups):
        raise ValueError("source identifiers must be nonempty strings")
    unique = sorted(set(groups))
    if len(unique) < 3:
        raise ValueError("at least three independent source groups are required")
    order = np.random.default_rng(seed).permutation(unique).tolist()
    n_train = min(max(1, round(len(order) * train_fraction)), len(order) - 2)
    n_validation = min(
        max(1, round(len(order) * validation_fraction)), len(order) - n_train - 1
    )
    return GroupSplit(
        tuple(order[:n_train]),
        tuple(order[n_train : n_train + n_validation]),
        tuple(order[n_train + n_validation :]),
        seed,
    )


def _value(row: Mapping[str, float | None], metric: str) -> float | None:
    value = row.get(metric)
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _metrics(metrics: Sequence[str]) -> tuple[str, ...]:
    names = tuple(metrics)
    if (
        not names
        or len(set(names)) != len(names)
        or any(not isinstance(n, str) or not n.strip() for n in names)
    ):
        raise ValueError("metrics must contain unique nonempty names")
    return names


def aggregate_summary(
    summaries: SubjectSummaries, metrics: Sequence[str]
) -> dict[str, Any]:
    """Return equally weighted source means and explicit finite coverage."""
    result: dict[str, Any] = {"groups": len(summaries), "metrics": {}}
    for metric in _metrics(metrics):
        values = [
            value
            for row in summaries.values()
            if (value := _value(row, metric)) is not None
        ]
        result["metrics"][metric] = {
            "mean": float(np.mean(values)) if values else None,
            "sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            "available_groups": len(values),
            "missing_groups": len(summaries) - len(values),
        }
    return result


def _error_statistics(errors: np.ndarray, axis: int = -1) -> np.ndarray:
    scale = np.max(np.abs(errors), axis=axis, keepdims=True)
    normalized = np.divide(errors, scale, out=np.zeros_like(errors), where=scale != 0)
    magnitude = np.squeeze(scale, axis=axis)
    return np.asarray(
        [
            np.mean(np.abs(normalized), axis=axis) * magnitude,
            np.sqrt(np.mean(normalized**2, axis=axis)) * magnitude,
            np.mean(normalized, axis=axis) * magnitude,
        ]
    )


def _intervals(
    values: np.ndarray,
    statistic: Callable[..., np.ndarray],
    *,
    bootstrap: int,
    confidence: float,
    seed: int,
) -> list[list[float] | None]:
    if len(values) < 2 or bootstrap == 0:
        return [None] * len(np.atleast_1d(statistic(values)))
    from scipy.stats import bootstrap as scipy_bootstrap

    # All used statistics scale linearly. Normalization also keeps SciPy's
    # bootstrap standard error finite for large, otherwise valid measurements.
    magnitude = float(np.max(np.abs(values))) or 1.0
    estimate = scipy_bootstrap(
        (values / magnitude,),
        statistic,
        n_resamples=bootstrap,
        confidence_level=confidence,
        method="percentile",
        vectorized=True,
        batch=256,
        rng=np.random.default_rng(seed),
    )
    return [
        [float(low) * magnitude, float(high) * magnitude]
        for low, high in zip(
            np.atleast_1d(estimate.confidence_interval.low),
            np.atleast_1d(estimate.confidence_interval.high),
        )
    ]


def compare_predictions(
    observed: SubjectSummaries,
    predicted: SubjectSummaries,
    metrics: Sequence[str],
    *,
    bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, Any]:
    """Compare paired source summaries with source bootstrap intervals.

    A missing prediction reduces coverage and marks the report incomplete.
    Missing observations are counted separately and never imputed.
    """
    names = _metrics(metrics)
    if isinstance(bootstrap, bool) or not isinstance(bootstrap, int) or bootstrap < 0:
        raise ValueError("bootstrap must be a nonnegative integer")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie in (0, 1)")
    if set(predicted) - set(observed):
        raise ValueError("predictions contain sources outside the observed partition")
    result: dict[str, Any] = {
        "status": "evaluated",
        "groups": len(observed),
        "resampling_unit": "original_source_group",
        "bootstrap_resamples": bootstrap,
        "confidence_level": confidence,
        "seed": seed,
        "metrics": {},
    }
    for index, metric in enumerate(names):
        errors, paired = [], []
        available = 0
        for group, row in sorted(observed.items()):
            truth = _value(row, metric)
            if truth is None:
                continue
            available += 1
            prediction = _value(predicted.get(group, {}), metric)
            if prediction is not None and math.isfinite(prediction - truth):
                paired.append(group)
                errors.append(prediction - truth)
        if len(errors) != available or available == 0:
            result["status"] = "incomplete_evaluation"
        entry: dict[str, Any] = {
            "observed_groups": available,
            "missing_observation_groups": len(observed) - available,
            "paired_groups": len(errors),
            "missing_prediction_groups": available - len(errors),
            "source_groups": paired,
        }
        if errors:
            array = np.asarray(errors, dtype=float)
            statistics = _error_statistics(array)
            intervals = _intervals(
                array,
                _error_statistics,
                bootstrap=bootstrap,
                confidence=confidence,
                seed=seed + index,
            )
            entry.update(
                {
                    name: {"value": float(value), "interval": interval}
                    for name, value, interval in zip(
                        ("mae", "rmse", "bias"), statistics, intervals
                    )
                }
            )
        else:
            entry.update({name: None for name in ("mae", "rmse", "bias")})
        result["metrics"][metric] = entry
    return result


def compare_prediction_change(
    observed: SubjectSummaries,
    baseline: SubjectSummaries,
    selected: SubjectSummaries,
    metrics: Sequence[str],
    *,
    bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, Any]:
    """Bootstrap the paired change in absolute error across original sources.

    Negative changes favor the selected model. Each source contributes its
    selected absolute error minus its baseline absolute error. Missing values
    retain their coverage flags through :func:`compare_predictions`.
    """
    names = _metrics(metrics)
    if (set(baseline) | set(selected)) - set(observed):
        raise ValueError("predictions contain sources outside the observed partition")
    zero: dict[str, dict[str, float | None]] = {}
    differences: dict[str, dict[str, float | None]] = {}
    for group, row in observed.items():
        zero[group], differences[group] = {}, {}
        for metric in names:
            truth = _value(row, metric)
            zero[group][metric] = 0.0 if truth is not None else None
            original = _value(baseline.get(group, {}), metric)
            fitted = _value(selected.get(group, {}), metric)
            difference = None
            if truth is not None and original is not None and fitted is not None:
                change = abs(fitted - truth) - abs(original - truth)
                if math.isfinite(change):
                    difference = change
            differences[group][metric] = difference
    comparison = compare_predictions(
        zero, differences, names, bootstrap=bootstrap, confidence=confidence, seed=seed
    )
    comparison["comparison_role"] = "paired_change_in_absolute_error"
    comparison["negative_change_favors"] = "selected"
    for entry in comparison["metrics"].values():
        entry["change_in_mae"] = entry.pop("bias")
        entry.pop("mae")
        entry.pop("rmse")
    return comparison


def calibrate_grouped(
    observed: SubjectSummaries,
    simulate: Simulator,
    priors: Sequence[Prior],
    baseline_parameters: Mapping[str, float],
    *,
    metrics: Sequence[str],
    configuration: Mapping[str, Any],
    provenance: Mapping[str, Any],
    split: GroupSplit | None = None,
    scales: Mapping[str, float] | None = None,
    fit_seeds: Sequence[int] = (0, 1),
    evaluation_seeds: Sequence[int] = (100, 101),
    samples: int = 256,
    accept: float = 0.1,
    bootstrap: int = 1000,
    seed: int = 0,
    max_validation_candidates: int | None = None,
) -> dict[str, Any]:
    """Fit on training sources, select on validation, then inspect locked test.

    ``simulate`` receives parameters, original source IDs and one simulation
    seed. It returns one metric summary per requested source. Repeated seeds
    are averaged within each source. Every requested observed metric must be
    finite in every replicate during fitting and selection.

    The saved profile contains explicit overrides and the supplied human,
    driver and reflex configurations. Global parameter defaults are untouched.
    """
    names = _metrics(metrics)
    if not {"human", "driver", "reflex"} <= set(configuration):
        raise ValueError(
            "configuration must preserve human, driver and reflex settings"
        )
    if not baseline_parameters or any(
        not math.isfinite(float(value)) for value in baseline_parameters.values()
    ):
        raise ValueError("baseline parameters must contain finite values")
    if set(p.name for p in priors) - set(baseline_parameters):
        raise ValueError("every fitted parameter needs its baseline value")
    fits, checks = tuple(fit_seeds), tuple(evaluation_seeds)
    if (
        not fits
        or not checks
        or len(set(fits)) != len(fits)
        or len(set(checks)) != len(checks)
        or set(fits) & set(checks)
        or any(
            isinstance(s, bool) or not isinstance(s, int) or s < 0
            for s in fits + checks
        )
    ):
        raise ValueError(
            "fit and evaluation seeds must be distinct nonnegative integers"
        )
    if max_validation_candidates is not None and (
        isinstance(max_validation_candidates, bool)
        or not isinstance(max_validation_candidates, int)
        or max_validation_candidates < 1
    ):
        raise ValueError("max_validation_candidates must be a positive integer")
    eligible = {
        group: {metric: _value(row, metric) for metric in names}
        for group, row in observed.items()
        if any(_value(row, metric) is not None for metric in names)
    }
    partition = split if split is not None else group_split(tuple(eligible), seed=seed)
    groups = partition.train + partition.validation + partition.test
    if set(groups) != set(eligible):
        raise ValueError("partitions must cover exactly the sources with observations")
    normalization = dict(scales) if scales is not None else {}
    if scales is not None and set(scales) != set(names):
        raise ValueError("supplied scales must cover exactly the selected metrics")
    for metric in names:
        values = [
            value
            for group in partition.train
            if (value := _value(eligible[group], metric)) is not None
        ]
        if not values:
            raise ValueError(f"metric {metric!r} has no training observations")
        if metric not in normalization:
            normalization[metric] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else math.nan
            )
        if not math.isfinite(normalization[metric]) or normalization[metric] <= 0:
            raise ValueError(
                f"metric {metric!r} needs a positive training or fixed scale"
            )
    failures: list[dict[str, Any]] = []

    def predict(
        parameters: Mapping[str, float],
        source_ids: tuple[str, ...],
        seeds: tuple[int, ...],
    ) -> dict[str, dict[str, float | None]]:
        replicates = []
        for simulation_seed in seeds:
            try:
                prediction = simulate(parameters, source_ids, simulation_seed)
            except (ArithmeticError, RuntimeError, ValueError) as error:
                failures.append(
                    {
                        "groups": list(source_ids),
                        "seed": simulation_seed,
                        "parameters": dict(parameters),
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                prediction = {}
            if set(prediction) - set(source_ids):
                raise ValueError(
                    "simulator returned sources outside its requested partition"
                )
            replicates.append(prediction)
        result: dict[str, dict[str, float | None]] = {}
        for group in source_ids:
            result[group] = {}
            for metric in names:
                values = [_value(r.get(group, {}), metric) for r in replicates]
                finite = [value for value in values if value is not None]
                result[group][metric] = (
                    float(np.mean(finite)) if len(finite) == len(values) else None
                )
        return result

    def cells(source_ids: tuple[str, ...]) -> tuple[dict[str, float], dict[str, float]]:
        truth, cell_scales = {}, {}
        for index, group in enumerate(source_ids):
            available = {
                metric: value
                for metric in names
                if (value := _value(eligible[group], metric)) is not None
            }
            weight = math.sqrt(len(available) * len(source_ids))
            for metric, value in available.items():
                key = f"{index}/{metric}"
                truth[key] = value
                cell_scales[key] = normalization[metric] * weight
        return truth, cell_scales

    def flatten(
        prediction: SubjectSummaries, source_ids: tuple[str, ...]
    ) -> dict[str, float]:
        return {
            f"{index}/{metric}": value if value is not None else math.nan
            for index, group in enumerate(source_ids)
            for metric in names
            if _value(eligible[group], metric) is not None
            for value in [_value(prediction.get(group, {}), metric)]
        }

    def forward(overrides: Mapping[str, float]) -> Mapping[str, float]:
        parameters = dict(baseline_parameters) | dict(overrides)
        return flatten(predict(parameters, partition.train, fits), partition.train)

    truth, cell_scales = cells(partition.train)
    posterior = abc_rejection(
        forward, truth, cell_scales, priors, samples=samples, accept=accept, seed=seed
    )
    candidates = [dict(baseline_parameters)] + [
        dict(baseline_parameters) | dict(zip(posterior.names, row.tolist()))
        for row in posterior.samples[:max_validation_candidates]
    ]
    from .inference import distance

    validation_truth, validation_scales = cells(partition.validation)
    losses = []
    for parameters in candidates:
        prediction = predict(parameters, partition.validation, fits)
        losses.append(
            distance(
                flatten(prediction, partition.validation),
                validation_truth,
                validation_scales,
            )
        )
    if not any(math.isfinite(value) for value in losses):
        raise ValueError("no candidate provides complete validation predictions")
    selected_index = int(np.argmin(losses))
    selected = candidates[selected_index]
    evaluations: dict[str, Any] = {}
    for role in ("train", "validation", "test"):
        source_ids = getattr(partition, role)
        observations = {group: eligible[group] for group in source_ids}
        baseline = predict(baseline_parameters, source_ids, checks)
        fitted = predict(selected, source_ids, checks)
        evaluations[role] = {
            "observed": aggregate_summary(observations, names),
            "baseline": compare_predictions(
                observations, baseline, names, bootstrap=bootstrap, seed=seed
            ),
            "selected": compare_predictions(
                observations, fitted, names, bootstrap=bootstrap, seed=seed
            ),
            "paired_change": compare_prediction_change(
                observations,
                baseline,
                fitted,
                names,
                bootstrap=bootstrap,
                seed=seed,
            ),
            "baseline_predictions": baseline,
            "selected_predictions": fitted,
        }
    return {
        "schema_version": 1,
        "status": (
            "evaluated"
            if all(
                report[label]["status"] == "evaluated"
                for report in evaluations.values()
                for label in ("baseline", "selected")
            )
            else "incomplete_evaluation"
        ),
        "profile": {
            "role": "fitted_candidate" if selected_index else "retained_baseline",
            "parameters": selected,
            "configuration": deepcopy(dict(configuration)),
            "configuration_role": "baseline_with_explicit_parameter_overrides",
            "parameter_overrides": {name: selected[name] for name in posterior.names},
            "provenance": deepcopy(dict(provenance)),
            "test_used_for_parameter_selection": False,
        },
        "partitions": asdict(partition),
        "excluded_empty_sources": sorted(set(observed) - set(eligible)),
        "metrics": list(names),
        "scales": normalization,
        "scale_source": "training_groups"
        if scales is None
        else "supplied_fixed_scales",
        "fit_seeds": list(fits),
        "evaluation_seeds": list(checks),
        "priors": [asdict(prior) for prior in priors],
        "posterior": posterior.as_dict(),
        "posterior_samples": posterior.samples.tolist(),
        "posterior_training_distances": posterior.distances.tolist(),
        "all_samples": posterior.all_samples.tolist(),
        "all_training_distances": [
            float(value) if math.isfinite(value) else None
            for value in posterior.all_distances
        ],
        "validation_candidates": [
            {"parameters": p, "distance": value if math.isfinite(value) else None}
            for p, value in zip(candidates, losses)
        ],
        "selection_index": selected_index,
        "evaluations": evaluations,
        "simulation_failures": failures,
    }
