"""Local empirical driving data with explicit units, source groups and gaps.

Tabular dependencies and the optional vehicle bridge are imported lazily.
Adapters retain unavailable signals as absent columns. Source restrictions
and unverified calibrations are carried into the bundle's summary.
"""

from __future__ import annotations

import importlib
import re
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

IDENTITY = ("subject_id", "session_id", "trial_id", "condition")


@dataclass(frozen=True)
class DatasetBundle:
    """Optional pandas tables from a single source.

    Glance endpoints and sample times are seconds. EMG rows may contain
    arrays for one source channel and epoch, to avoid expanding millions of
    samples into object-heavy rows. Grouping uses ``subject_id``; metadata
    identifies whether this is a participant or vehicle track.
    """

    dataset_id: str
    domain: str
    licence: str
    source: str
    glances: Any = None
    telemetry: Any = None
    emg: Any = None
    events: Any = None
    trials: Any = None
    gaze_samples: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    limitations: tuple[str, ...] = ()

    def summary(self) -> dict[str, Any]:
        """Return JSON-safe counts and available fields without observations."""
        tables: dict[str, Any] = {}
        subjects: set[str] = set()
        trials: set[str] = set()
        for name in ("glances", "telemetry", "emg", "events", "trials", "gaze_samples"):
            table = getattr(self, name)
            if table is None:
                continue
            tables[name] = {
                "rows": len(table),
                "columns": list(table.columns),
                "subjects": int(table.subject_id.nunique())
                if "subject_id" in table
                else 0,
                "trials": int(table.trial_id.nunique()) if "trial_id" in table else 0,
            }
            if "subject_id" in table:
                subjects.update(map(str, table.subject_id.dropna().unique()))
            if "trial_id" in table:
                trials.update(map(str, table.trial_id.dropna().unique()))
        return dict(
            dataset_id=self.dataset_id,
            domain=self.domain,
            licence=self.licence,
            source=self.source,
            subjects=len(subjects),
            trials=len(trials),
            tables=tables,
            metadata=self.metadata,
            limitations=list(self.limitations),
        )


def _pandas() -> Any:
    return importlib.import_module("pandas")


def _require(frame: Any, columns: tuple[str, ...], label: str) -> None:
    available = frame.columns if hasattr(frame, "columns") else frame.keys()
    missing = sorted(set(columns) - set(available))
    if missing:
        raise ValueError(f"{label}: missing columns {missing}")


def _concat(frames: list[Any]) -> Any:
    return _pandas().concat(frames, ignore_index=True) if frames else None


def _identity(frame: Any, subject: str, session: str, condition: str) -> Any:
    result = frame.copy()
    result["subject_id"], result["session_id"] = str(subject), str(session)
    result["condition"] = str(condition)
    result["trial_id"] = f"{subject}:{session}:{condition}"
    return result


def _intervals(samples: Any, area: str) -> Any:
    """Reconstruct categorical dwell intervals with boundary censoring."""
    output = []
    for identity, group in samples.groupby(list(IDENTITY), sort=False):
        group = group.sort_values("time_s", kind="stable")
        t = group.time_s.to_numpy(dtype=float)
        if not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
            raise ValueError("categorical samples need finite increasing times")
        dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.0
        values = group[area].astype(str).to_numpy()
        gaps = np.r_[False, np.diff(t) > 1.5 * dt] if dt else np.ones(len(t), bool)
        starts = np.flatnonzero(np.r_[True, values[1:] != values[:-1]] | gaps)
        for first, stop in zip(starts, np.r_[starts[1:], len(t)]):
            next_gap = stop < len(t) and gaps[stop]
            end = t[stop] if stop < len(t) and not next_gap else t[stop - 1] + dt
            output.append(
                dict(zip(IDENTITY, map(str, identity)))
                | {
                    "area": str(values[first]),
                    "start_s": float(t[first]),
                    "end_s": float(end),
                    "duration_s": float(end - t[first]),
                    "left_censored": bool(first == 0 or gaps[first]),
                    "right_censored": bool(stop == len(t) or next_gap),
                    "boundary": bool(
                        first == 0 or gaps[first] or stop == len(t) or next_gap
                    ),
                }
            )
    return _pandas().DataFrame(output)


def _lehtonen_exclusions(frame: Any) -> Any:
    """Quality exclusions from the authors' Figshare 5572636 preprocess.R."""
    keep = ~frame.session_id.isin([2014051701, 2014082301, 2013120201])
    keep &= ~(
        (frame.session_id == 2013121301)
        & (
            frame.condition.isin(["down1", "down2"])
            | ((frame.condition == "free") & (frame.direction == "west"))
        )
    )
    keep &= ~(
        (frame.session_id == 2013111801) & frame.condition.isin(["down1", "down2"])
    )
    return frame.loc[keep].copy()


def load_lehtonen(path: str | Path, *, author_exclusions: bool = True) -> DatasetBundle:
    """Read the public event archive or extracted directory."""
    pandas, path = _pandas(), Path(path)

    def read(name: str) -> Any:
        if path.is_file():
            with zipfile.ZipFile(path) as archive, archive.open("data/" + name) as f:
                return pandas.read_csv(f)
        candidate = path / "data" / name
        return pandas.read_csv(candidate if candidate.exists() else path / name)

    def identify(frame: Any) -> Any:
        result = frame.copy()
        result["subject_id"] = result.session_id.astype(str)
        result["session_id"] = result.session_id.astype(str)
        result["trial_id"] = (
            result.subject_id
            + ":"
            + result.run.astype(str)
            + ":"
            + result.condition.astype(str)
            + ":"
            + result.direction.astype(str)
        )
        return result

    raw = read("1_glances.txt")
    _require(
        raw,
        ("session_id", "run", "condition", "direction", "aoi", "t0", "t1"),
        "Lehtonen",
    )
    data = identify(_lehtonen_exclusions(raw) if author_exclusions else raw)
    glances = data.rename(
        columns={
            "aoi": "source_area",
            "t0": "start_s",
            "t1": "end_s",
            "heading": "heading_deg",
            "pitch": "pitch_deg",
        }
    )
    targets = dict.fromkeys(("down1", "down2", "up1", "up2", "up3", "lamp"), "target")
    glances["area"] = glances.source_area.replace(
        {
            "roadhead": "road",
            "road_ahead": "road",
            "road ahead": "road",
            "speedometer": "instrument",
            **targets,
        }
    )
    glances["duration_s"] = glances.end_s - glances.start_s
    if (
        not np.isfinite(glances[["start_s", "end_s"]].to_numpy(float)).all()
        or (glances.duration_s <= 0).any()
    ):
        raise ValueError("Lehtonen endpoints must be finite and ordered")
    glances = glances.sort_values(["trial_id", "start_s"], kind="stable")
    glances["boundary"] = False
    for _, group in glances.groupby("trial_id", sort=False):
        glances.loc[[group.index[0], group.index[-1]], "boundary"] = True
    for name in ("left_censored", "right_censored"):
        glances[name] = pandas.array([None] * len(glances), dtype="boolean")
    vehicle = read("2_vehicle.txt")
    vehicle = _lehtonen_exclusions(vehicle) if author_exclusions else vehicle
    trials = identify(vehicle).rename(
        columns={"duration": "duration_s", "speed_mean": "speed_mps"}
    )
    trials = trials[trials.trial_id.isin(glances.trial_id)]
    glances = glances.merge(
        trials[["trial_id", "speed_mps"]].drop_duplicates("trial_id"),
        on="trial_id",
        how="left",
        validate="many_to_one",
    )
    raw_events, mapped = read("3_swrs.txt"), []
    # Retain the source direction when supplied. The exposure check excludes
    # events outside the quality-filtered gaze recording for that trial.
    for trial, group in glances.groupby("trial_id", sort=False):
        first = group.iloc[0]
        mask = (
            (raw_events.session_id.astype(str) == first.subject_id)
            & (raw_events.run == first.run)
            & (raw_events.condition == first.condition)
            & raw_events.ts.between(group.start_s.min(), group.end_s.max())
        )
        if "direction" in raw_events:
            mask &= raw_events.direction == first.direction
        selected = raw_events[mask].copy()
        selected["subject_id"], selected["session_id"] = (
            first.subject_id,
            first.session_id,
        )
        selected["trial_id"], selected["event_type"] = trial, "steering_reversal"
        mapped.append(selected.rename(columns={"ts": "time_s"}))
    return DatasetBundle(
        "lehtonen2018",
        "real_road",
        "CC BY 4.0",
        "https://doi.org/10.6084/m9.figshare.5572636",
        glances=glances.reset_index(drop=True),
        events=_concat(mapped),
        trials=trials.reset_index(drop=True),
        metadata={
            "group_kind": "participant",
            "raw_subjects": int(raw.session_id.nunique()),
            "author_exclusions": author_exclusions,
            "censoring": "unreported; boundary intervals flagged",
            "event_clock": "seconds within participant recording",
            "source_files": [str(path)],
        },
        limitations=(
            "Public vehicle records are condition summaries.",
            "Glance targets differ from the SuRT visual search task.",
            "Continuous raw steering and pedal samples are unavailable.",
        ),
    )


def load_rightgaze(path: str | Path) -> DatasetBundle:
    """Read Zenodo 20717040 categorical gaze and vehicle trajectories."""
    pandas, path = _pandas(), Path(path)
    folder = path if path.is_dir() else path.parent
    source = folder / "gazes_spatiotemporal.feather" if path.is_dir() else path
    data = pandas.read_feather(source)
    _require(
        data,
        ("driver_id", "trajectory_id", "t", "s_driver", "v", "gaze_state", "gaze_id"),
        "RightGaze",
    )
    data = data.rename(
        columns={
            "driver_id": "subject_id",
            "trajectory_id": "trial_id",
            "t": "time_s",
            "s_driver": "position_m",
            "v": "speed_mps",
            "gaze_state": "source_area",
            "gaze_id": "source_event_id",
        }
    )
    data["subject_id"], data["trial_id"] = (
        data.subject_id.astype(str),
        data.trial_id.astype(str),
    )
    data["session_id"], data["condition"] = data.subject_id, "right_turn"
    data["area"] = data.source_area.astype(str).replace(
        {
            "LF": "road",
            "RVM": "rear_mirror",
            "SVM_R": "right_mirror",
            "WG_R": "right_window",
            "SG_R": "right_shoulder",
        }
    )
    data = data.sort_values(["trial_id", "time_s"], kind="stable").reset_index(
        drop=True
    )
    trials = data[[*IDENTITY]].drop_duplicates("trial_id").reset_index(drop=True)
    scenarios = folder / "scenario_data.xlsx"
    if scenarios.exists():
        conditions = pandas.read_excel(scenarios)
        _require(conditions, ("trajectory_id", "driver_id"), "RightGaze conditions")
        conditions["trial_id"] = conditions.trajectory_id.astype(str)
        conditions["subject_id"] = conditions.driver_id.astype(str)
        conditions = conditions.drop(columns=["trajectory_id", "driver_id"])
        trials = trials.merge(
            conditions, on=["trial_id", "subject_id"], how="left", validate="one_to_one"
        )
    return DatasetBundle(
        "rightgaze2022",
        "real_test_track",
        "CC BY 4.0",
        "https://doi.org/10.5281/zenodo.20717040",
        glances=_intervals(data, "area"),
        telemetry=data,
        trials=trials,
        metadata={
            "group_kind": "participant",
            "gaze_measurement": "video coded AOI aligned at100Hz",
            "interval_reconstruction": "runs; last sample adds median interval",
            "road_geometry": str(folder / "geometry.xodr"),
            "source_files": [str(source)],
        },
        limitations=(
            "Zero gaze IDs coincide with front gaze; zero is labelled unclassified.",
            "Vehicle control samples and angular gaze measurements are unavailable.",
            "Released turning trials contain no cyclist on a collision course.",
        ),
    )


def _expert_identity(path: Path, condition: str | None) -> tuple[str, str]:
    match = re.search(r"[EN]\d+", path.stem)
    if match is None:
        raise ValueError(f"Cannot identify ExpertDriving participant in {path.name}")
    known = path.parent.name in ("ExpertDriver", "NoviceDriver")
    return match.group(), condition or (
        path.parent.parent.name if known else "unverified"
    )


def load_expert_driving(
    path: str | Path, *, condition: str | None = None
) -> DatasetBundle:
    """Read the public ExpertDriving CSV hierarchy or audited sample pair."""
    pandas, path = _pandas(), Path(path)
    files = sorted(path.rglob("*.csv")) if path.is_dir() else [path]
    telemetry, gazes, glances, trials, used = [], [], [], [], []
    for file in files:
        lower = file.as_posix().lower()
        is_can = "/2-canbus/" in lower or file.name.startswith("CAN-")
        is_eye = "/2-eyetracking/" in lower or file.name.startswith("EyeTracking-")
        if not (is_can or is_eye):
            continue
        data = pandas.read_csv(file, low_memory=False)
        subject, task = _expert_identity(file, condition)
        used.append(str(file))
        if is_can:
            _require(data, ("timestamp", "speed_mps"), str(file))
            timestamp = pandas.to_numeric(data.timestamp, errors="raise").astype(
                "int64"
            )
            frame = pandas.DataFrame(
                {
                    "time_s": (timestamp - timestamp.min()) / 1e9,
                    "timestamp_ns": timestamp,
                    "speed_mps": pandas.to_numeric(data.speed_mps, errors="raise"),
                }
            )
            for old, new in (
                ("throttle_percentage", "throttle_percent"),
                ("brake_percentage", "brake_percent"),
                ("steering_percentage", "steer_percent"),
                ("steering_torque_nm", "steering_torque_nm"),
            ):
                if old in data:
                    frame[new] = pandas.to_numeric(data[old], errors="coerce")
            frame = _identity(frame, subject, task, task)
            telemetry.append(frame.sort_values("time_s", kind="stable"))
            trials.append(frame[[*IDENTITY]].drop_duplicates())
        else:
            _require(data, ("Recording timestamp", "Sensor"), str(file))
            timestamp = pandas.to_numeric(data["Recording timestamp"], errors="raise")
            data["time_s"] = (timestamp - timestamp.min()) / 1e6
            eye = data[
                data.Sensor.astype(str).str.replace(" ", "").str.lower() == "eyetracker"
            ]
            if eye.empty:
                continue
            mapping = {
                "Recording timestamp": "recording_timestamp_us",
                "Eye movement type": "eye_event_type",
                "Eye movement type index": "source_event_id",
                "Validity left": "validity_left",
                "Validity right": "validity_right",
                "Gaze point X": "gaze_point_x",
                "Gaze point Y": "gaze_point_y",
                "Pupil diameter left": "pupil_diameter_left",
                "Pupil diameter right": "pupil_diameter_right",
            }
            for side in ("left", "right"):
                for axis in "XYZ":
                    mapping[f"Gaze direction {side} {axis}"] = (
                        f"gaze_direction_{side}_{axis.lower()}"
                    )
            columns = ["time_s", *[c for c in mapping if c in eye]]
            eye = _identity(eye[columns].rename(columns=mapping), subject, task, task)
            eye = eye.sort_values("time_s", kind="stable")
            eye["area"] = "unmapped"
            gazes.append(eye)
            if "source_event_id" in eye:
                intervals = _intervals(eye, "source_event_id")
                intervals["source_event_id"] = intervals.area
                if "eye_event_type" in eye:
                    kinds = (
                        eye.assign(source_event_id=eye.source_event_id.astype(str))
                        .groupby("source_event_id")
                        .eye_event_type.first()
                    )
                    intervals["source_event_type"] = intervals.source_event_id.map(
                        kinds
                    )
                    intervals = intervals[
                        intervals.source_event_type.astype(str).str.lower()
                        == "fixation"
                    ].copy()
                else:
                    intervals["source_event_type"] = "unverified"
                intervals["area"] = "unmapped"
                glances.append(intervals)
    if not used:
        raise FileNotFoundError("No ExpertDriving CAN or eye CSV files found")
    return DatasetBundle(
        "expert_driving",
        "real_road",
        "CC BY 4.0",
        "https://doi.org/10.6084/m9.figshare.29664056",
        glances=_concat(glances),
        telemetry=_concat(telemetry),
        gaze_samples=_concat(gazes),
        trials=_concat(trials),
        metadata={
            "group_kind": "participant",
            "gaze_clock": "recording-relative microseconds; alignment unverified",
            "source_files": used,
        },
        limitations=(
            "Steering and pedal percentages have unverified physical scaling.",
            "CAN and gaze clock alignment is unverified.",
            "Gaze AOIs and vehicle-frame calibration are unverified.",
        ),
    )


def load_ngsim(
    path: str | Path,
    *,
    max_trials: int | None = None,
    duration: float = 20.0,
    max_speed: float = 16.0,
) -> DatasetBundle:
    """Reuse bridge SI conversion and matched following processing."""
    pandas = _pandas()
    ngsim = importlib.import_module("myosuite_carla.ngsim")
    path = Path(path)
    source = (
        ngsim.load_trials(path)
        if path.suffix.lower() == ".npz"
        else ngsim.following_trials(
            ngsim.load_tracks(path), duration=duration, max_speed=max_speed
        )
    )
    if max_trials is not None:
        if (
            isinstance(max_trials, bool)
            or not isinstance(max_trials, int)
            or max_trials < 1
        ):
            raise ValueError("max_trials must be positive")
        source = source[:max_trials]
    frames, trials = [], []
    for index, trial in enumerate(source):
        frame = pandas.DataFrame(
            {
                "time_s": trial.time,
                "position_m": trial.position,
                "speed_mps": trial.speed,
                "acceleration_mps2": trial.acceleration,
                "gap_m": trial.gap,
                "leader_speed_mps": trial.leader_speed,
            }
        )
        frame = _identity(frame, trial.group_id, trial.driver, "following")
        frame["trial_id"], frame["source_group"] = (
            f"{trial.driver}:{index}",
            trial.group_id,
        )
        frames.append(frame)
        trials.append(
            {
                **{name: frame[name].iloc[0] for name in IDENTITY},
                "source_group": trial.group_id,
                "duration_s": trial.duration,
            }
        )
    return DatasetBundle(
        "ngsim",
        "real_road",
        "CC BY-SA 3.0",
        "https://doi.org/10.21949/1504477",
        telemetry=_concat(frames),
        trials=pandas.DataFrame(trials),
        metadata={
            "group_kind": "vehicle_track",
            "sampling_hz": ngsim.RATE,
            "smoothing": list(ngsim.SMOOTHING),
            "max_trials": max_trials,
            "source_files": [str(path)],
        },
        limitations=(
            "Track identity does not establish persistent human identity.",
            "Gaze and measured steering or pedal signals are unavailable.",
        ),
    )


def load_drivedna(
    path: str | Path, *, time_column: str, time_scale: float = 1.0
) -> DatasetBundle:
    """Read approved raw signals with explicit time and human-control masks.

    Clock schema must be supplied from the approved local release. This
    adapter accepts no terms and fetches no gated files. Each file must
    contain an ``is_human`` boolean mask derived from the source segment
    index before loading.
    """
    pandas, path = _pandas(), Path(path)
    if not np.isfinite(time_scale) or time_scale <= 0:
        raise ValueError("time_scale must be finite and positive")
    base = path / "raw_signals" if (path / "raw_signals").is_dir() else path
    files = (
        sorted(base.rglob("*.parquet")) + sorted(base.rglob("*.csv"))
        if base.is_dir()
        else [base]
    )
    frames, trials = [], []
    for file in files:
        data = (
            pandas.read_parquet(file)
            if file.suffix.lower() == ".parquet"
            else pandas.read_csv(file)
        )
        _require(data, (time_column, "vEgo", "is_human"), str(file))
        if not data.is_human.isin([True, False, 0, 1]).all():
            raise ValueError("DriveDNA is_human must contain boolean values")
        data = data.loc[data.is_human.astype(bool)].copy()
        if data.empty:
            continue
        subject = str(data.driver.iloc[0]) if "driver" in data else file.parent.name
        session = str(data.drive.iloc[0]) if "drive" in data else file.stem
        if "driver" in data and data.driver.nunique() != 1:
            raise ValueError("DriveDNA raw file must identify one driver")
        t = pandas.to_numeric(data[time_column], errors="raise")
        frame = pandas.DataFrame(
            {
                "time_s": (t - t.min()) * time_scale,
                "speed_mps": pandas.to_numeric(data.vEgo, errors="raise"),
            }
        )
        for old, new in (
            ("aEgo", "acceleration_mps2"),
            ("steeringAngleDeg", "steer_deg"),
            ("steeringRateDeg", "steer_rate_deg_s"),
            ("actual_curvature", "curvature_per_m"),
            ("gas", "throttle"),
            ("brake", "brake"),
        ):
            if old in data:
                frame[new] = pandas.to_numeric(data[old], errors="coerce")
        if "leadOne_dRel" in data and "leadOne_status" in data:
            frame["lead_distance_m"] = pandas.to_numeric(
                data.leadOne_dRel, errors="coerce"
            ).where(data.leadOne_status.astype(bool))
        frame = _identity(frame, subject, session, "human_driving")
        frames.append(frame.sort_values("time_s", kind="stable"))
        trials.append(frame[[*IDENTITY]].drop_duplicates())
    if not frames:
        raise FileNotFoundError("No human DriveDNA raw signal recordings found")
    return DatasetBundle(
        "drivedna_controlled",
        "real_road",
        "DriveDNA research licence",
        "https://huggingface.co/datasets/HenryYHW/DriveDNA-Controlled",
        telemetry=_concat(frames),
        trials=_concat(trials),
        metadata={
            "group_kind": "participant",
            "time_column": time_column,
            "time_scale": time_scale,
            "source_files": [str(f) for f in files],
        },
        limitations=(
            "Use requires approved access under the source research licence.",
            "Steering depends on steering ratio; pedal availability varies.",
            "Radar lead distance is not a calibrated bumper gap.",
            "Gaze measurements are unavailable.",
        ),
    )


def load_mpdb(path: str | Path) -> DatasetBundle:
    """Read processed MPDB EEGLAB EMG files with SciPy's MAT reader.

    Channel values retain their source units. Epochs retain source event
    labels and their event-relative time. This adapter does not infer muscle
    force or normalize activation against an unmeasured maximum contraction.
    """
    pandas, path = _pandas(), Path(path)
    scipy_io = importlib.import_module("scipy.io")
    files = sorted(path.rglob("EMG_*.set")) if path.is_dir() else [path]
    if not files or any(not file.is_file() for file in files):
        raise FileNotFoundError("No MPDB processed EMG_*.set files found")
    event_names = {
        "139": "braking",
        "141": "braking",
        "145": "braking",
        "125": "turn",
        "127": "turn",
        "129": "lane_change",
        "131": "lane_change",
        "137": "acceleration",
        "143": "acceleration",
        "133": "smooth_driving",
    }
    channels, events, trials = [], [], []
    for file in files:
        match = re.fullmatch(r"EMG_(\d+)", file.stem)
        if match is None:
            raise ValueError(f"MPDB processed EMG filename expected: {file.name}")
        subject = match.group(1)
        contents = scipy_io.loadmat(file, simplify_cells=True)
        eeg = contents.get("EEG", contents)
        _require(eeg, ("data", "srate", "pnts", "nbchan"), str(file))
        rate, points = float(eeg["srate"]), int(eeg["pnts"])
        count, epochs = int(eeg["nbchan"]), int(eeg.get("trials", 1))
        if not np.isfinite(rate) or rate <= 0 or points < 1 or epochs < 1:
            raise ValueError(f"Invalid MPDB sampling dimensions in {file.name}")
        data = eeg["data"]
        if isinstance(data, str):
            binary = (file.parent / data).resolve()
            if not binary.is_relative_to(file.parent.resolve()):
                raise ValueError("EEGLAB binary samples must reside beside the set")
            values = np.fromfile(binary, dtype="<f4")
            values = values.reshape((count, points, epochs), order="F")
        else:
            values = np.asarray(data)
            if values.size != count * points * epochs:
                raise ValueError(f"MPDB sample shape differs from header: {file.name}")
            values = values.reshape((count, points, epochs))
        times = np.arange(points, dtype=float) / rate + float(eeg.get("xmin", 0.0))
        locations = eeg.get("chanlocs", [])
        locations = (
            [locations]
            if isinstance(locations, dict)
            else list(np.atleast_1d(locations))
        )
        source_epochs = eeg.get("epoch", [])
        source_epochs = (
            [source_epochs]
            if isinstance(source_epochs, dict)
            else list(np.atleast_1d(source_epochs))
        )
        for epoch in range(epochs):
            details = source_epochs[epoch] if epoch < len(source_epochs) else {}
            types = np.atleast_1d(details.get("eventtype", "unmapped")).ravel()
            latencies = np.atleast_1d(details.get("eventlatency", 0)).ravel()
            selected = 0
            if len(latencies) == len(types):
                selected = int(np.argmin(np.abs(latencies.astype(float))))
            event_type = str(types[selected])
            condition = event_names.get(event_type, "unmapped")
            trial = f"{subject}:{file.stem}:{epoch + 1}"
            identity = dict(
                subject_id=subject,
                session_id=file.stem,
                trial_id=trial,
                condition=condition,
            )
            trials.append(
                identity
                | {
                    "source_event_type": event_type,
                    "start_s": float(times[0]),
                    "end_s": float(times[-1] + 1 / rate),
                }
            )
            response_end = float(times[-1] + 1 / rate)
            baseline_contaminated = False
            for event_index, code in enumerate(types):
                latency = (
                    float(latencies[event_index]) / 1000
                    if len(latencies) == len(types)
                    else 0.0
                )
                primary = event_index == selected
                events.append(
                    identity
                    | {
                        "time_s": latency,
                        "event_type": event_names.get(str(code), "unmapped"),
                        "source_event_type": str(code),
                        "primary": primary,
                    }
                )
                if not primary:
                    if latency > 0:
                        response_end = min(response_end, latency)
                    else:
                        baseline_contaminated = True
            trials[-1]["response_end_s"] = response_end
            trials[-1]["baseline_contaminated"] = baseline_contaminated
            for channel in range(count):
                location = locations[channel] if channel < len(locations) else {}
                label = str(location.get("labels", f"EMG{channel + 1}"))
                units = str(location.get("unit", "unverified_source_units"))
                channels.append(
                    identity
                    | {
                        "channel": label,
                        "units": units,
                        "sampling_hz": rate,
                        "time_s": times,
                        "value": values[channel, :, epoch],
                        "response_end_s": response_end,
                        "baseline_contaminated": baseline_contaminated,
                    }
                )
    return DatasetBundle(
        "mpdb",
        "moving_simulator",
        "CC BY 4.0",
        "https://doi.org/10.6084/m9.figshare.22192831.v3",
        emg=pandas.DataFrame(channels),
        events=pandas.DataFrame(events),
        trials=pandas.DataFrame(trials),
        metadata={
            "group_kind": "participant",
            "clock": "source epoch-relative",
            "source_files": [str(file) for file in files],
        },
        limitations=(
            "Simulator EMG requires a separate real road evaluation.",
            "EMG units and maximal voluntary contraction normalization are unverified.",
            "Source channels are retained without an inferred anatomical mapping.",
            "Continuous vehicle controls are unavailable in the processed EMG files.",
        ),
    )


def load_dataset(dataset_id: str, path: str | Path, **options: Any) -> DatasetBundle:
    """Load a supported local source by its stable identifier."""
    loaders: dict[str, Callable[..., DatasetBundle]] = {
        "lehtonen2018": load_lehtonen,
        "rightgaze2022": load_rightgaze,
        "expert_driving": load_expert_driving,
        "ngsim": load_ngsim,
        "drivedna_controlled": load_drivedna,
        "mpdb": load_mpdb,
    }
    if dataset_id not in loaders:
        raise ValueError(f"Unknown dataset {dataset_id!r}; choose {sorted(loaders)}")
    return loaders[dataset_id](path, **options)
