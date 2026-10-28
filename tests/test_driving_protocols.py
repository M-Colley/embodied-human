"""Independent participant selection and censored empirical measurements."""

import json

import numpy as np
import pytest

from embodied_human.driving_data import DatasetBundle
from embodied_human.driving_evaluation import group_split
from embodied_human.driving_protocols import (
    calibrate_gaze_allocation,
    emg_reference,
    gaze_allocation_reference,
    telemetry_reference,
)

pd = pytest.importorskip("pandas")


def _gaze_bundle():
    names = [f"p{i}" for i in range(10)]
    split = group_split([f"rightgaze2022:{name}" for name in names])
    repeated = split.train[0].split(":", 1)[1]
    rows = []
    for name in names:
        for trial in range(100 if name == repeated else 1):
            road = 0.9 if name == repeated else 0.3
            for area, duration in (("road", road), ("right_mirror", 1 - road)):
                rows.append(
                    dict(
                        subject_id=name,
                        trial_id=f"{name}:{trial}",
                        area=area,
                        duration_s=duration,
                    )
                )
    return DatasetBundle(
        "rightgaze2022",
        "real_test_track",
        "CC BY 4.0",
        "source",
        glances=pd.DataFrame(rows),
    )


def test_gaze_fit_weights_people_and_keeps_test_out_of_selection():
    bundle = _gaze_bundle()
    original = calibrate_gaze_allocation(bundle, bootstrap=0)
    assert original["selection_index"] == 1
    assert original["fitted_candidate_probabilities"]["road"] == pytest.approx(0.4)
    changed = bundle.glances.copy()
    held = [name.split(":", 1)[1] for name in original["partitions"]["test"]]
    changed.loc[
        changed.subject_id.isin(held) & changed.area.eq("road"), "duration_s"
    ] = 0.99
    changed.loc[
        changed.subject_id.isin(held) & changed.area.eq("right_mirror"), "duration_s"
    ] = 0.01
    repeated = calibrate_gaze_allocation(
        DatasetBundle(
            "rightgaze2022", "real_test_track", "CC BY 4.0", "source", glances=changed
        ),
        bootstrap=0,
    )
    assert original["profile"] == repeated["profile"]
    assert original["validation_candidates"] == repeated["validation_candidates"]
    a = original["evaluations"]["test"]["selected"]["metrics"]["share_road"]["rmse"][
        "value"
    ]
    b = repeated["evaluations"]["test"]["selected"]["metrics"]["share_road"]["rmse"][
        "value"
    ]
    assert b > a
    assert original["profile"]["test_used_for_parameter_selection"] is False
    json.dumps(original, allow_nan=False)


def test_gaze_reference_does_not_treat_unknown_aois_as_road():
    bundle = DatasetBundle(
        "rightgaze2022",
        "real_test_track",
        "CC BY 4.0",
        "source",
        glances=pd.DataFrame(
            {
                "subject_id": ["p"],
                "trial_id": ["p:1"],
                "area": ["unmapped"],
                "duration_s": [1.0],
            }
        ),
    )
    with pytest.raises(ValueError, match="no finite"):
        gaze_allocation_reference(bundle)


def test_telemetry_averages_trials_before_people():
    rows = []
    for subject, trial, count, speed in (
        ("a", "a0", 100, 10.0),
        ("a", "a1", 5, 2.0),
        ("b", "b0", 5, 2.0),
    ):
        for t in np.arange(count) * 0.1:
            rows.append(
                dict(
                    subject_id=subject,
                    trial_id=trial,
                    time_s=t,
                    condition="normal",
                    speed_mps=speed,
                    steer_percent=10 * t,
                )
            )
    result = telemetry_reference(
        DatasetBundle(
            "expert_driving",
            "real_road",
            "CC BY 4.0",
            "source",
            telemetry=pd.DataFrame(rows),
        )
    )
    assert result["participant_summaries"]["expert_driving:a"]["speed_mean_mps"] == 6
    assert result["summary"]["metrics"]["speed_mean_mps"]["mean"] == 4
    for row in result["observations"]:
        assert row["steering_total_variation_per_s"] == pytest.approx(10)
    assert result["role"] == "empirical_reference"
    assert result["measurement_model_required"] is True


def test_telemetry_does_not_differentiate_across_long_gaps():
    t = np.r_[np.arange(6) * 0.1, 5 + np.arange(6) * 0.1]
    frame = pd.DataFrame(
        dict(
            subject_id="a",
            trial_id="t",
            time_s=t,
            speed_mps=np.r_[np.ones(6), np.full(6, 10.0)],
        )
    )
    report = telemetry_reference(
        DatasetBundle(
            "expert_driving", "real_road", "CC BY 4.0", "source", telemetry=frame
        )
    )
    assert report["diagnostics"]["trials"][0]["long_gaps"] == 1
    assert report["observations"][0]["acceleration_sd_mps2"] < 1e-8


def _emg_bundle():
    t = np.arange(-500, 1500) / 1000
    signal = (t >= 0.2).astype(float)
    rows = []
    for trial, values, end, contaminated in (
        ("onset", signal, 1.5, False),
        ("none", np.zeros_like(t), 1.5, False),
        ("competing", signal, 0.22, False),
        ("pre", (t >= -0.1).astype(float), 1.5, False),
        ("bad", signal, 1.5, True),
    ):
        rows.append(
            dict(
                subject_id="a",
                trial_id=trial,
                condition="braking",
                channel="EMG1",
                sampling_hz=1000.0,
                time_s=t,
                value=values,
                response_end_s=end,
                baseline_contaminated=contaminated,
            )
        )
    return DatasetBundle(
        "mpdb", "moving_simulator", "CC BY 4.0", "source", emg=pd.DataFrame(rows)
    )


def test_emg_causal_onsets_and_all_censoring_states_are_retained():
    report = emg_reference(_emg_bundle())
    observations = {row["trial_id"]: row for row in report["observations"]}
    assert observations["onset"]["onset_s"] == pytest.approx(0.2)
    assert observations["onset"]["onset_state"] == "detected"
    assert observations["none"]["right_censored"] is True
    assert observations["none"]["right_censor_s"] == pytest.approx(1.45)
    assert observations["competing"]["onset_s"] is None
    assert observations["competing"]["right_censor_s"] == pytest.approx(0.17)
    assert observations["pre"]["left_censored"] is True
    assert observations["pre"]["onset_s"] is None
    assert observations["bad"]["onset_state"] == "contaminated_baseline"
    assert observations["bad"]["right_censored"] is None
    summary = report["participant_channel_condition_summaries"][0]
    assert (
        summary["detected_trials"],
        summary["right_censored_trials"],
        summary["left_censored_trials"],
        summary["invalid_trials"],
    ) == (1, 2, 1, 1)
    assert report["measurement_model_required"] is True
    json.dumps(report, allow_nan=False)


def test_emg_invalid_values_are_not_reported_as_absent_activation():
    bundle = _emg_bundle()
    bundle.emg.at[0, "value"] = np.full(2000, np.nan)
    result = emg_reference(bundle)
    assert result["observations"][0]["onset_state"] == "invalid_signal"
    assert result["observations"][0]["right_censored"] is None
