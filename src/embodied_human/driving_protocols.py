"""Empirical driving protocols with explicit measurement and model roles."""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any

import numpy as np

from .attention import SEEV, SEEVParameters
from .core import HumanConfig
from .driving_data import DatasetBundle
from .driving_evaluation import aggregate_summary, compare_predictions, group_split

GAZE_AREAS = ("road", "rear_mirror", "right_mirror", "right_window", "right_shoulder")
GAZE_LIMITATIONS = (
    "Released zero gaze IDs coincide with front gaze and have ambiguous validity.",
    "Categorical dwell allocation leaves head motion and timing unevaluated.",
    "N-back conditions and approach speeds are pooled within participants.",
    "The released turning trials contain no actual cyclist on a collision course.",
    "Fixed SEEV weights and contextual utilities do not identify cognitive weights.",
)


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _subject_means(
    rows: list[dict[str, Any]], metrics: list[str], dataset: str
) -> dict[str, dict[str, float | None]]:
    groups: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        group = groups.setdefault(f"{dataset}:{row['subject_id']}", {})
        for metric in metrics:
            value = _finite(row.get(metric))
            if value is not None:
                group.setdefault(metric, []).append(value)
    return {
        subject: {
            metric: float(np.mean(values[metric])) if values.get(metric) else None
            for metric in metrics
        }
        for subject, values in groups.items()
    }


def gaze_allocation_reference(bundle: DatasetBundle) -> dict[str, Any]:
    """Summarize observed dwell shares with equal trial and participant weights."""
    if bundle.dataset_id != "rightgaze2022":
        raise ValueError("fixed gaze categories require RightGazeUrban")
    if bundle.glances is None or bundle.glances.empty:
        raise ValueError("categorical gaze intervals are required")
    frame = bundle.glances
    needed = {"subject_id", "trial_id", "area", "duration_s"}
    if not needed <= set(frame.columns):
        raise ValueError(f"gaze intervals need {sorted(needed)}")
    rows, excluded = [], 0
    for trial_id, group in frame.groupby("trial_id", sort=False):
        if group.subject_id.nunique() != 1:
            raise ValueError("each gaze trial must identify exactly one participant")
        values = group.duration_s.to_numpy(dtype=float)
        valid = (
            np.isfinite(values) & (values > 0) & group.area.isin(GAZE_AREAS).to_numpy()
        )
        excluded += int((~valid).sum())
        observed = group.loc[valid]
        total = float(observed.duration_s.sum())
        if total <= 0:
            continue
        shares = observed.groupby("area").duration_s.sum() / total
        rows.append(
            {
                "subject_id": str(group.subject_id.iloc[0]),
                "trial_id": str(trial_id),
                "exposure_s": total,
                **{
                    f"share_{area}": float(shares.get(area, 0.0)) for area in GAZE_AREAS
                },
            }
        )
    if not rows:
        raise ValueError("no finite supported gaze exposures are available")
    metrics = [f"share_{area}" for area in GAZE_AREAS]
    subjects = _subject_means(rows, metrics, bundle.dataset_id)
    baseline = {
        subject: {f"share_{area}": float(area == "road") for area in GAZE_AREAS}
        for subject in subjects
    }
    return {
        "schema_version": 1,
        "status": "reference_ready",
        "role": "empirical_reference",
        "dataset_id": bundle.dataset_id,
        "observations": rows,
        "participant_summaries": subjects,
        "metrics": metrics,
        "summary": aggregate_summary(subjects, metrics),
        "baseline": compare_predictions(subjects, baseline, metrics, bootstrap=0),
        "diagnostics": {
            "excluded_intervals": excluded,
            "weighting": "equal trials within participant; equal participants",
        },
        "requirements": [
            "Match gaze category definitions and task context before interpretation."
        ],
        "limitations": list(GAZE_LIMITATIONS),
    }


def _speed_acceleration(
    times: np.ndarray, speed: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    """Differentiate a 0.5 s quadratic Savitzky Golay speed estimate.

    Gaps above three median sample intervals split recordings. Short gaps
    and jitter are interpolated onto the segment's native median time grid.
    """
    from scipy.signal import savgol_filter

    diagnostics: dict[str, Any] = {"long_gaps": 0, "derivative_samples": 0}
    valid = np.isfinite(times) & np.isfinite(speed)
    diagnostics["invalid_speed_samples"] = int((~valid).sum())
    t, v = times[valid], speed[valid]
    if len(t) < 5:
        diagnostics["derivative_unavailable"] = "fewer than five finite samples"
        return np.array([]), diagnostics
    delta = np.diff(t)
    if np.any(delta <= 0):
        diagnostics["derivative_unavailable"] = "duplicate or decreasing timestamps"
        return np.array([]), diagnostics
    dt = float(np.median(delta))
    diagnostics["median_interval_s"] = dt
    starts = np.r_[0, np.flatnonzero(delta > 3 * dt) + 1]
    diagnostics["long_gaps"] = int(len(starts) - 1)
    acceleration = []
    for first, stop in zip(starts, np.r_[starts[1:], len(t)]):
        local_t, local_v = t[first:stop], v[first:stop]
        if len(local_t) < 5:
            continue
        grid = local_t[0] + np.arange(int((local_t[-1] - local_t[0]) / dt) + 1) * dt
        if len(grid) < 5:
            continue
        window = max(5, 2 * int(round(0.5 / dt) / 2) + 1)
        window = min(window, len(grid) - (1 - len(grid) % 2))
        acceleration.append(
            savgol_filter(
                np.interp(grid, local_t, local_v),
                window_length=window,
                polyorder=2,
                deriv=1,
                delta=dt,
                mode="interp",
            )
        )
    result = np.concatenate(acceleration) if acceleration else np.array([])
    diagnostics["derivative_samples"] = len(result)
    return result, diagnostics


def telemetry_reference(bundle: DatasetBundle) -> dict[str, Any]:
    """Describe observed telemetry without claiming a matched model test."""
    frame = bundle.telemetry
    if frame is None or frame.empty:
        return {
            "schema_version": 1,
            "status": "unavailable",
            "role": "empirical_reference",
            "dataset_id": bundle.dataset_id,
            "observations": [],
            "participant_summaries": {},
            "requirements": ["Vehicle sample recordings are required."],
        }
    needed = {"subject_id", "trial_id", "time_s", "speed_mps"}
    if not needed <= set(frame.columns):
        raise ValueError(f"vehicle samples need {sorted(needed)}")
    observations, quality = [], []
    for trial_id, group in frame.groupby("trial_id", sort=False):
        if group.subject_id.nunique() != 1:
            raise ValueError("each vehicle trial must identify one original source")
        group = group.sort_values("time_s", kind="stable")
        t = group.time_s.to_numpy(dtype=float)
        v = group.speed_mps.to_numpy(dtype=float)
        finite_speed = v[np.isfinite(v)]
        derivative, details = _speed_acceleration(t, v)
        row = {
            "subject_id": str(group.subject_id.iloc[0]),
            "trial_id": str(trial_id),
            "condition": str(group.condition.iloc[0])
            if "condition" in group
            else "unreported",
            "speed_mean_mps": float(finite_speed.mean()) if len(finite_speed) else None,
            "acceleration_sd_mps2": float(derivative.std(ddof=1))
            if len(derivative) > 1
            else None,
            "steering_total_variation_per_s": None,
        }
        steering = next(
            (name for name in ("steer_deg", "steer_percent", "steer") if name in group),
            None,
        )
        if steering is not None:
            values, dt = group[steering].to_numpy(dtype=float), np.diff(t)
            differences = np.diff(values)
            positive = dt[np.isfinite(dt) & (dt > 0)]
            if len(positive):
                usable = (
                    np.isfinite(differences)
                    & np.isfinite(dt)
                    & (dt > 0)
                    & (dt <= 3 * np.median(positive))
                )
                exposure = float(dt[usable].sum())
                if exposure:
                    row["steering_total_variation_per_s"] = float(
                        np.abs(differences[usable]).sum() / exposure
                    )
            details["steering_source_column"] = steering
            details["constant_steering"] = bool(
                len(values) and np.isfinite(values).all() and np.ptp(values) == 0
            )
        observations.append(row)
        quality.append({"trial_id": str(trial_id), **details})
    metrics = [
        "speed_mean_mps",
        "acceleration_sd_mps2",
        "steering_total_variation_per_s",
    ]
    subjects = _subject_means(observations, metrics, bundle.dataset_id)
    gaze = bundle.gaze_samples
    coverage: dict[str, Any] = {
        "eye_tracker_rows": 0,
        "gaze_subjects": 0,
        "gaze_trials": 0,
        "valid_eye_fraction": None,
    }
    if gaze is not None and not gaze.empty:
        coverage.update(
            eye_tracker_rows=len(gaze),
            gaze_subjects=int(gaze.subject_id.nunique()),
            gaze_trials=int(gaze.trial_id.nunique()),
        )
        valid_columns = [
            name for name in ("validity_left", "validity_right") if name in gaze
        ]
        if valid_columns:
            usable = np.column_stack(
                [
                    gaze[name].astype(str).str.lower().eq("valid").to_numpy()
                    for name in valid_columns
                ]
            ).any(axis=1)
            coverage["valid_eye_fraction"] = float(usable.mean())
    return {
        "schema_version": 1,
        "status": "reference_ready",
        "role": "empirical_reference",
        "dataset_id": bundle.dataset_id,
        "observations": observations,
        "participant_summaries": subjects,
        "metrics": metrics,
        "summary": aggregate_summary(subjects, metrics),
        "diagnostics": {
            "trials": quality,
            "gaze_coverage": coverage,
            "acceleration_processing": (
                "0.5s quadratic Savitzky Golay derivative; native median grid; "
                "split gaps above three median intervals"
            ),
            "steering_units": "source column units per second; physically uncalibrated",
        },
        "measurement_model_required": True,
        "requirements": [
            "Replay matched scenes with an identified vehicle for model comparison.",
            "Calibrate source steering and pedal scales.",
            "Validate cross-modal clocks and gaze regions.",
        ],
        "limitations": list(bundle.limitations),
    }


def _emg_onset(row: Any) -> dict[str, Any]:
    """Detect an event-relative sustained rise in a causal RMS envelope."""
    from scipy.ndimage import uniform_filter1d

    t, signal = np.asarray(row.time_s, float), np.asarray(row.value, float)
    rate = float(row.sampling_hz)
    result: dict[str, Any] = {
        "onset_s": None,
        "onset_state": "invalid_signal",
        "left_censored": None,
        "right_censored": None,
        "right_censor_s": None,
        "threshold_source_units": None,
        "baseline_median_source_units": None,
        "baseline_mad_source_units": None,
    }
    if (
        not math.isfinite(rate)
        or rate <= 0
        or t.ndim != 1
        or signal.shape != t.shape
        or len(t) < 3
        or not np.isfinite(t).all()
        or not np.isfinite(signal).all()
        or not np.allclose(np.diff(t), 1 / rate, rtol=1e-4, atol=1e-9)
    ):
        return result
    if bool(row.get("baseline_contaminated", False)):
        result["onset_state"] = "contaminated_baseline"
        return result
    window = max(1, round(0.05 * rate))
    magnitude = float(np.max(np.abs(signal))) or 1.0
    envelope = (
        np.sqrt(
            np.maximum(
                0.0,
                uniform_filter1d(
                    (signal / magnitude) ** 2,
                    size=window,
                    origin=(window - 1) // 2,
                    mode="nearest",
                ),
            )
        )
        * magnitude
    )
    baseline = envelope[(t >= -0.4) & (t < -0.05)]
    if len(baseline) < max(3, window):
        result["onset_state"] = "insufficient_baseline"
        return result
    median = float(np.median(baseline))
    mad = float(np.median(np.abs(baseline - median)))
    threshold = median + 3 * 1.4826 * mad
    result.update(
        threshold_source_units=threshold,
        baseline_median_source_units=median,
        baseline_mad_source_units=mad,
    )
    response_end = min(
        float(row.get("response_end_s", t[-1] + 1 / rate)), float(t[-1] + 1 / rate)
    )
    post = np.flatnonzero((t >= 0) & (t < response_end))
    result.update(
        onset_state="right_censored",
        left_censored=False,
        right_censored=True,
        response_end_s=response_end,
        right_censor_s=float(t[post[-window]]) if len(post) >= window else 0.0,
    )
    if len(post) < window:
        return result
    above = envelope > threshold
    runs = np.convolve(
        above[post].astype(int), np.ones(window, dtype=int), mode="valid"
    )
    starts = np.flatnonzero(runs == window)
    if not len(starts):
        return result
    crossing = int(post[starts[0]])
    if crossing == post[0] and crossing > 0 and above[crossing - 1]:
        result.update(
            onset_state="left_censored",
            left_censored=True,
            right_censored=False,
            onset_upper_bound_s=0.0,
        )
    else:
        result.update(
            onset_state="detected", onset_s=float(t[crossing]), right_censored=False
        )
    return result


def emg_reference(bundle: DatasetBundle) -> dict[str, Any]:
    """Measure activation relative to source events with explicit censoring.

    These observations do not estimate reaction time or absolute muscle force.
    """
    frame = bundle.emg
    if frame is None or frame.empty:
        return {
            "schema_version": 1,
            "status": "unavailable",
            "role": "empirical_reference",
            "dataset_id": bundle.dataset_id,
            "observations": [],
            "participant_summaries": {},
            "requirements": ["EMG channel recordings with event clocks are required."],
        }
    needed = {
        "subject_id",
        "trial_id",
        "condition",
        "channel",
        "time_s",
        "value",
        "sampling_hz",
    }
    if not needed <= set(frame.columns):
        raise ValueError(f"EMG samples need {sorted(needed)}")
    observations = []
    for _, row in frame.iterrows():
        observations.append(
            {
                "subject_id": str(row.subject_id),
                "trial_id": str(row.trial_id),
                "condition": str(row.condition),
                "channel": str(row.channel),
                "units": str(row.get("units", "unverified_source_units")),
                **_emg_onset(row),
            }
        )
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in observations:
        grouped.setdefault(
            (row["subject_id"], row["channel"], row["condition"]), []
        ).append(row)
    aggregates: list[dict[str, Any]] = []
    subjects: dict[str, dict[str, Any]] = {}
    for (subject, channel, condition), rows in sorted(grouped.items()):
        detected = [row["onset_s"] for row in rows if row["onset_state"] == "detected"]
        right = sum(row["onset_state"] == "right_censored" for row in rows)
        left = sum(row["onset_state"] == "left_censored" for row in rows)
        invalid = len(rows) - len(detected) - right - left
        summary = {
            "subject_id": subject,
            "channel": channel,
            "condition": condition,
            "trials": len(rows),
            "detected_trials": len(detected),
            "right_censored_trials": right,
            "left_censored_trials": left,
            "invalid_trials": invalid,
            "detected_onset_mean_s": float(np.mean(detected)) if detected else None,
            "right_censored_fraction": right / len(rows),
            "left_censored_fraction": left / len(rows),
            "invalid_fraction": invalid / len(rows),
        }
        aggregates.append(summary)
        fields = subjects.setdefault(f"{bundle.dataset_id}:{subject}", {})
        for metric in (
            "detected_onset_mean_s",
            "right_censored_fraction",
            "left_censored_fraction",
            "invalid_fraction",
        ):
            fields[f"{channel}/{condition}/{metric}"] = summary[metric]
    return {
        "schema_version": 1,
        "status": "reference_ready",
        "role": "empirical_reference",
        "dataset_id": bundle.dataset_id,
        "observations": observations,
        "participant_channel_condition_summaries": aggregates,
        "participant_summaries": subjects,
        "measurement_model_required": True,
        "diagnostics": {
            "channels": sorted(frame.channel.astype(str).unique().tolist()),
            "participants": int(frame.subject_id.nunique()),
            "channel_epochs": len(observations),
            "processing": {
                "rms_window_s": 0.05,
                "rms_direction": "trailing",
                "baseline_window_s": [-0.4, -0.05],
                "threshold": "median+3*1.4826*MAD",
                "sustain_s": 0.05,
                "censoring": "last fully observable sustained onset window",
                "competing_events": "truncate response; flag baseline contamination",
            },
        },
        "requirements": [
            "Calibrate EMG units and anatomical channel mapping.",
            "Use an activation-to-EMG observation model before model fitting.",
            "Retain censoring when comparing onset distributions.",
        ],
        "limitations": [
            *bundle.limitations,
            "Detected onset means condition on successful detection.",
            "Activation onset does not establish human reaction time.",
            "Absolute muscle force and strength are unevaluated.",
        ],
    }


def calibrate_gaze_allocation(
    bundle: DatasetBundle, *, seed: int = 0, bootstrap: int = 500
) -> dict[str, Any]:
    """Fit contextual SEEV area values on training participants only.

    Validation selects either the existing road-only allocation or the
    training estimate. The test partition is evaluated after selection.
    Source categories are fixed by the declared RightGaze schema.
    """
    if bundle.dataset_id != "rightgaze2022":
        raise ValueError("this contextual gaze protocol requires RightGazeUrban")
    reference = gaze_allocation_reference(bundle)
    observed, metrics = reference["participant_summaries"], reference["metrics"]
    partition = group_split(tuple(observed), seed=seed)
    shares = {
        area: float(np.mean([observed[s][f"share_{area}"] for s in partition.train]))
        for area in GAZE_AREAS
    }
    fitted_areas = {
        area: (0.0, 0.0, 1.0, math.log(max(share, 1e-12)))
        for area, share in shares.items()
    }
    baseline_areas = {"road": (0.0, 0.0, 1.0, 0.0)}
    candidates = [baseline_areas, fitted_areas]
    predictions = [SEEV(areas=areas).probabilities() for areas in candidates]
    losses = [
        float(
            np.mean(
                [
                    (p.get(area, 0.0) - observed[s][f"share_{area}"]) ** 2
                    for s in partition.validation
                    for area in GAZE_AREAS
                ]
            )
        )
        for p in predictions
    ]
    selected = int(np.argmin(losses))
    evaluations = {}
    for role in ("train", "validation", "test"):
        sources = getattr(partition, role)
        truth = {s: observed[s] for s in sources}

        def predict(index: int) -> dict[str, dict[str, float]]:
            return {
                s: {
                    f"share_{area}": predictions[index].get(area, 0.0)
                    for area in GAZE_AREAS
                }
                for s in sources
            }

        evaluations[role] = {
            "observed": aggregate_summary(truth, metrics),
            "baseline": compare_predictions(
                truth, predict(0), metrics, bootstrap=bootstrap, seed=seed
            ),
            "selected": compare_predictions(
                truth, predict(selected), metrics, bootstrap=bootstrap, seed=seed
            ),
            "baseline_predictions": predict(0),
            "selected_predictions": predict(selected),
        }
    from .agents.driver import DriverAgentParameters

    fixed = asdict(SEEVParameters())
    areas = candidates[selected]
    config = HumanConfig().replace("attention", "seev", areas=areas, **fixed)
    return {
        "schema_version": 1,
        "status": "evaluated",
        "role": "component_calibration",
        "dataset_id": bundle.dataset_id,
        "metrics": metrics,
        "partitions": asdict(partition),
        "selection_index": selected,
        "validation_candidates": [
            {"name": name, "mse": loss}
            for name, loss in zip(("road_only", "training_dwell"), losses)
        ],
        "profile": {
            "module": "attention.seev",
            "role": "fitted_candidate" if selected else "retained_baseline",
            "context": "RightGazeUrban right turns with pooled n-back conditions",
            "parameters": {
                f"area_value.{area}": float(features[3])
                for area, features in areas.items()
            },
            "parameter_overrides": {
                f"area_value.{area}": float(features[3])
                for area, features in areas.items()
            },
            "fixed_global_weights": fixed,
            "configuration": {
                "human": config.as_dict(),
                "driver": asdict(DriverAgentParameters()),
                "reflex": {"role": "not_executed"},
            },
            "configuration_role": "contextual_attention_component",
            "provenance": {
                "dataset": bundle.source,
                "fitting_unit": "participant",
                "estimator": "log equally weighted training participant shares",
                "role": "fitted",
            },
            "test_used_for_parameter_selection": False,
        },
        "fitted_candidate_probabilities": predictions[1],
        "evaluations": evaluations,
        "reference": reference,
        "limitations": list(GAZE_LIMITATIONS),
    }
