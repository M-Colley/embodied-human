"""Empirical adapters preserve source groups, missing signals and event clocks."""

import io
import json
import zipfile

import numpy as np
import pytest
from scipy.io import savemat

from embodied_human.driving_data import (
    DatasetBundle,
    load_dataset,
    load_drivedna,
    load_expert_driving,
    load_lehtonen,
    load_mpdb,
    load_ngsim,
    load_rightgaze,
)

pd = pytest.importorskip("pandas")


def _lehtonen_archive(tmp_path, invalid=False):
    good, bad = 2014051702, 2014051701
    glances = pd.DataFrame(
        {
            "session_id": [good, bad, good, good],
            "run": [1] * 4,
            "condition": ["down1"] * 4,
            "direction": ["east"] * 4,
            "aoi": ["road_ahead", "down1", "down1", "road_ahead"],
            "t0": [0.0, 0.0, 1.0, 2.0],
            "t1": [1.0, 1.0, 2.0, 3.0],
            "heading": [0.0, 20.0, 20.0, 0.0],
            "pitch": [0.0, -25.0, -25.0, 0.0],
        }
    )
    if invalid:
        glances.loc[0, "t1"] = 0
    vehicle = pd.DataFrame(
        {
            "session_id": [good, bad, good],
            "run": [1, 1, 2],
            "condition": ["down1"] * 3,
            "direction": ["east"] * 3,
            "duration": [3.0] * 3,
            "speed_mean": [20.0] * 3,
        }
    )
    events = pd.DataFrame(
        {
            "session_id": [good, good, bad, good],
            "run": [1] * 4,
            "condition": ["down1"] * 4,
            "direction": ["east", "east", "east", "west"],
            "ts": [1.5, 9.0, 0.5, 1.8],
        }
    )
    path = tmp_path / "lehtonen.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for name, table in (
            ("1_glances.txt", glances),
            ("2_vehicle.txt", vehicle),
            ("3_swrs.txt", events),
        ):
            stream = io.StringIO()
            table.to_csv(stream, index=False)
            archive.writestr("data/" + name, stream.getvalue())
    return path


def test_lehtonen_exclusions_keep_matched_trials_and_event_exposure(tmp_path):
    bundle = load_lehtonen(_lehtonen_archive(tmp_path))
    assert bundle.summary()["subjects"] == 1
    assert len(bundle.glances) == 3
    assert len(bundle.trials) == 1
    assert bundle.glances.area.tolist() == ["road", "target", "road"]
    assert bundle.glances.source_area.tolist() == ["road_ahead", "down1", "road_ahead"]
    assert bundle.glances.boundary.tolist() == [True, False, True]
    assert bundle.glances.right_censored.isna().all()
    assert bundle.glances.speed_mps.tolist() == [20.0] * 3
    assert bundle.events.time_s.tolist() == [1.5]
    assert bundle.events.trial_id.iloc[0] == bundle.trials.trial_id.iloc[0]
    assert bundle.telemetry is None
    json.dumps(bundle.summary(), allow_nan=False)


def test_lehtonen_invalid_endpoints_fail(tmp_path):
    with pytest.raises(ValueError, match="endpoints"):
        load_lehtonen(_lehtonen_archive(tmp_path, invalid=True))


def test_rightgaze_gap_splits_dwell_without_interpolating(tmp_path):
    pytest.importorskip("pyarrow")
    source = tmp_path / "gazes_spatiotemporal.feather"
    pd.DataFrame(
        {
            "driver_id": [4] * 5,
            "trajectory_id": [7] * 5,
            "t": [0.0, 0.01, 0.02, 0.05, 0.06],
            "s_driver": np.arange(5),
            "v": [3.0] * 5,
            "gaze_state": ["LF", "LF", "SVM_R", "SVM_R", "SVM_R"],
            "gaze_id": [0, 0, 8, 8, 8],
        }
    ).to_feather(source)
    bundle = load_rightgaze(source)
    assert bundle.glances.area.tolist() == ["road", "right_mirror", "right_mirror"]
    np.testing.assert_allclose(bundle.glances.duration_s, [0.02, 0.01, 0.02])
    assert bundle.glances.right_censored.tolist() == [False, True, True]
    assert bundle.glances.left_censored.tolist() == [True, False, True]
    assert bundle.telemetry.source_event_id.tolist()[:2] == [0, 0]
    assert "steer" not in bundle.telemetry
    assert bundle.summary()["domain"] == "real_test_track"


def test_rightgaze_duplicate_times_fail(tmp_path):
    pytest.importorskip("pyarrow")
    source = tmp_path / "duplicate.feather"
    pd.DataFrame(
        {
            "driver_id": [1, 1],
            "trajectory_id": [1, 1],
            "t": [0.0, 0.0],
            "s_driver": [0.0, 0.0],
            "v": [2.0, 2.0],
            "gaze_state": ["LF", "LF"],
            "gaze_id": [0, 0],
        }
    ).to_feather(source)
    with pytest.raises(ValueError, match="increasing"):
        load_rightgaze(source)


def test_expert_mixed_sensors_keep_gaze_and_source_control_units(tmp_path):
    origin = 1683878362997702044
    pd.DataFrame(
        {
            "timestamp": [origin, origin + 10000000],
            "speed_mps": [8.0, 8.1],
            "steering_percentage": [2.0, 3.0],
            "brake_percentage": [20.0, 20.0],
        }
    ).to_csv(tmp_path / "CAN-E02.csv", index=False)
    pd.DataFrame(
        {
            "Recording timestamp": [1000000, 1005000, 1020000, 1040000],
            "Sensor": ["Eye Tracker", "Accelerometer", "Eye Tracker", "Eye Tracker"],
            "Eye movement type": ["Fixation", "EyesNotFound", "Fixation", "Saccade"],
            "Eye movement type index": [1, 2, 1, 3],
            "Validity left": ["Valid", "Invalid", "Valid", "Valid"],
        }
    ).to_csv(tmp_path / "EyeTracking-E02.csv", index=False)
    bundle = load_expert_driving(tmp_path, condition="enter")
    assert len(bundle.gaze_samples) == 3
    assert len(bundle.glances) == 1
    assert bundle.glances.source_event_type.tolist() == ["Fixation"]
    assert bundle.glances.area.tolist() == ["unmapped"]
    np.testing.assert_allclose(bundle.telemetry.time_s, [0.0, 0.01])
    assert bundle.telemetry.steer_percent.tolist() == [2.0, 3.0]
    assert "steer_deg" not in bundle.telemetry
    assert "heading_deg" not in bundle.gaze_samples
    json.dumps(bundle.summary(), allow_nan=False)


def test_unknown_dataset_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unknown dataset"):
        load_dataset("invented", tmp_path)


def test_bundle_summary_contains_no_observations():
    bundle = DatasetBundle(
        "test",
        "real_road",
        "CC BY 4.0",
        "source",
        telemetry=pd.DataFrame(
            {
                "subject_id": ["a", "a", "b"],
                "trial_id": ["t1", "t1", "t2"],
                "speed_mps": [1.0, 2.0, 3.0],
            }
        ),
    )
    summary = bundle.summary()
    assert summary["subjects"] == 2 and summary["trials"] == 2
    assert summary["tables"]["telemetry"]["rows"] == 3
    assert "a" not in json.dumps(summary).split('"')


def test_drivedna_removes_automation_and_preserves_driver_group(tmp_path):
    folder = tmp_path / "raw_signals" / "driver_461"
    folder.mkdir(parents=True)
    pd.DataFrame(
        {
            "time": [2000, 2100, 2200],
            "vEgo": [2.0, 3.0, 4.0],
            "is_human": [True, False, True],
            "steeringAngleDeg": [1.0, 2.0, 3.0],
            "leadOne_dRel": [10.0, 11.0, 12.0],
            "leadOne_status": [True, True, False],
        }
    ).to_csv(folder / "drive_001.csv", index=False)
    bundle = load_drivedna(tmp_path, time_column="time", time_scale=0.001)
    assert bundle.telemetry.speed_mps.tolist() == [2.0, 4.0]
    np.testing.assert_allclose(bundle.telemetry.time_s, [0.0, 0.2])
    assert bundle.telemetry.subject_id.tolist() == ["driver_461"] * 2
    assert bundle.telemetry.session_id.tolist() == ["drive_001"] * 2
    assert bundle.telemetry.lead_distance_m.iloc[0] == 10
    assert np.isnan(bundle.telemetry.lead_distance_m.iloc[1])
    assert "gap_m" not in bundle.telemetry


@pytest.mark.parametrize("mask", [None, ["unknown", "human"]])
def test_drivedna_requires_a_verified_human_mask(tmp_path, mask):
    table = pd.DataFrame({"time": [0.0, 0.1], "vEgo": [2.0, 2.0]})
    if mask is not None:
        table["is_human"] = mask
    source = tmp_path / "drive.csv"
    table.to_csv(source, index=False)
    with pytest.raises(ValueError, match="is_human"):
        load_drivedna(source, time_column="time")


@pytest.mark.parametrize("wrapped", [True, False])
def test_mpdb_keeps_epochs_and_censors_competing_responses(tmp_path, wrapped):
    path = tmp_path / "EMG_31.set"
    values = np.arange(8000, dtype=np.float32).reshape(4, 2000)
    eeg = {
        "data": values,
        "srate": 1000.0,
        "pnts": 2000,
        "nbchan": 4,
        "trials": 1,
        "xmin": -0.5,
        "chanlocs": [{"labels": f"EMG{i + 1}"} for i in range(4)],
        "epoch": {
            "eventtype": np.array(["139", "137"], dtype=object),
            "eventlatency": np.array([0.0, 1000.0]),
        },
    }
    savemat(path, {"EEG": eeg} if wrapped else eeg)
    bundle = load_mpdb(path)
    assert bundle.emg.channel.tolist() == ["EMG1", "EMG2", "EMG3", "EMG4"]
    assert bundle.emg.condition.tolist() == ["braking"] * 4
    assert bundle.events.primary.tolist() == [True, False]
    assert bundle.events.time_s.tolist() == [0.0, 1.0]
    assert bundle.trials.response_end_s.tolist() == [1.0]
    np.testing.assert_array_equal(bundle.emg.value.iloc[2], values[2])
    np.testing.assert_allclose(
        bundle.emg.time_s.iloc[0][[0, 500, -1]], [-0.5, 0.0, 1.499]
    )
    assert bundle.emg.units.tolist() == ["unverified_source_units"] * 4
    json.dumps(bundle.summary(), allow_nan=False)


def test_ngsim_reuses_original_source_group(tmp_path):
    pytest.importorskip("myosuite_carla")
    from myosuite_carla.ngsim import FollowingTrial, save_trials

    t = np.arange(21) / 10
    trials = [
        FollowingTrial(
            driver=f"recording-{i}",
            source_group="original-source",
            time=t,
            position=2 * t,
            speed=np.full(21, 2.0),
            acceleration=np.zeros(21),
            gap=np.full(21, 5.0),
            leader_speed=np.full(21, 2.0),
        )
        for i in range(2)
    ]
    path = save_trials(trials, tmp_path / "following.npz")
    bundle = load_ngsim(path)
    assert bundle.summary()["subjects"] == 1
    assert bundle.summary()["trials"] == 2
    assert len(bundle.telemetry) == 42
    assert bundle.telemetry.subject_id.unique().tolist() == ["original-source"]
    assert bundle.metadata["group_kind"] == "vehicle_track"
