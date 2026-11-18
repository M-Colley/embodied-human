"""Prepare fixed human following windows from official TGSIM lane geometries."""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import platform
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from shapely import STRtree

from myosuite_carla.ngsim import load_trials, resample_following, save_trials

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "prepared-final"
OUTPUT.mkdir(exist_ok=True)
SAMPLES = 201
SEED = 20261118
TARGET = 24
MAX_SPEED = 16.0
MAX_LATERAL_DISTANCE = 2.5
CAPTURES = [
    {"name": "i294-l1", "root": ROOT, "raw_file": "tgsim-i294-l1-all-runs.csv", "automation": "acc", "lanes": [-5, -4, -3, -2, 2, 3, 4, 5], "doi": "10.21949/1404217", "date": "2021-10-18", "direction": {8: "southbound", 18: "southbound", 20: "southbound"}},
    {"name": "i90-moving", "root": ROOT / "i90-moving", "raw_file": "all-runs.csv", "automation": "av", "lanes": [1, 2, 3, 4, 5, 6], "doi": "10.21949/1404212", "date": "2022-05-27", "direction": {}},
    {"name": "i90-stationary", "root": ROOT / "i90-stationary-next", "raw_file": "all-runs.csv", "automation": "av", "lanes": [1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15], "doi": "10.21949/1404218", "date": "2021-10-06", "direction": {}},
]


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def overpass_mask(capture: str, run: int, x: np.ndarray) -> np.ndarray:
    result = np.zeros(len(x), dtype=bool)
    if capture != "i294-l1":
        return result
    if run == 1:
        ranges = [(2130, 2170), (3330, 3370), (3730, 3770)]
    elif run in (8, 18, 20):
        ranges = [(1760, 1800), (2160, 2200), (3360, 3400)]
    else:
        ranges = [(1530, 1570), (2730, 2770), (3130, 3170)]
    for lower, upper in ranges:
        result |= (x >= lower - 6) & (x <= upper + 6)
    return result


def project_lane(frame: pd.DataFrame, geometry: pd.DataFrame, lane: int) -> tuple[pd.DataFrame, dict]:
    """Locate points exactly on original segments using Shapely's spatial index."""
    fields = [f"x_{lane}", f"y_{lane}", f"ramp_{lane}"]
    geo = geometry[fields].dropna()
    coordinates = geo[fields[:2]].to_numpy(dtype=float)
    ramps = geo[fields[2]].to_numpy(dtype=int)
    keep = np.r_[True, np.linalg.norm(np.diff(coordinates, axis=0), axis=1) > 1e-10]
    coordinates, ramps = coordinates[keep], ramps[keep]
    assert len(coordinates) >= 2
    segments = shapely.linestrings(np.stack([coordinates[:-1], coordinates[1:]], axis=1))
    lengths = shapely.length(segments)
    cumulative = np.r_[0.0, np.cumsum(lengths)]
    bad_segments = (ramps[:-1] != 0) | (ramps[1:] != 0)
    bad_prefix = np.r_[0, np.cumsum(bad_segments)]
    tree = STRtree(segments)
    result = frame.copy()
    result["raw_s"] = np.nan
    result["distance_to_lane"] = np.nan
    result["segment_index"] = -1
    valid = np.isfinite(result[["xloc_kf", "yloc_kf"]].to_numpy()).all(axis=1)
    positions = np.flatnonzero(valid)
    points = shapely.points(result.loc[valid, ["xloc_kf", "yloc_kf"]].to_numpy(dtype=float))
    pair, distances = tree.query_nearest(points, all_matches=False, return_distance=True)
    point_indices, segment_indices = pair
    local = shapely.line_locate_point(segments[segment_indices], points[point_indices])
    rows = result.index[positions[point_indices]]
    result.loc[rows, "raw_s"] = cumulative[segment_indices] + local
    result.loc[rows, "distance_to_lane"] = distances
    result.loc[rows, "segment_index"] = segment_indices
    result = result.sort_values(["id", "tick"])
    group = result.groupby("id", sort=False)
    ds = group.raw_s.diff()
    dt = group.tick.diff()
    moving = (dt == 1) & result.speed_kf.gt(1) & ds.abs().gt(.02) & result.distance_to_lane.le(MAX_LATERAL_DISTANCE)
    deltas = ds[moving].to_numpy()
    assert len(deltas), "No moving observations establish lane orientation"
    orientation = 1 if float(np.median(deltas)) > 0 else -1
    agreement = float(np.mean(orientation * deltas > 0))
    assert agreement >= .95, f"Inconsistent travel direction in lane {lane}: {agreement}"
    result["s"] = orientation * result.raw_s
    rear_s = result.raw_s.to_numpy() - orientation * result.vehicle_length.to_numpy()
    rear_indices = np.clip(np.searchsorted(cumulative, rear_s, side="right") - 1, 0, len(lengths) - 1)
    front_indices = np.maximum(result.segment_index.to_numpy(dtype=int), 0)
    low = np.minimum(front_indices, rear_indices)
    high = np.maximum(front_indices, rear_indices)
    contains_ramp = bad_prefix[high + 1] - bad_prefix[low] > 0
    result["geometry_valid"] = (
        result.raw_s.notna()
        & result.distance_to_lane.le(MAX_LATERAL_DISTANCE)
        & (rear_s >= 0)
        & (rear_s <= cumulative[-1])
    )
    result["mainline"] = result.geometry_valid & ~contains_ramp
    summary = {
        "lane": lane,
        "original_vertices": len(coordinates),
        "arc_length_m": float(cumulative[-1]),
        "orientation": orientation,
        "orientation_observations": len(deltas),
        "orientation_agreement": agreement,
        "rows": len(result),
        "valid_geometry_rows": int(result.geometry_valid.sum()),
        "mainline_rows": int(result.mainline.sum()),
        "distance_quantiles_m": np.quantile(distances, [.5, .95, .99, 1]).tolist(),
    }
    return result, summary


def necessary_tracks(frame: pd.DataFrame) -> int:
    count = 0
    for _, track in frame.groupby("id", sort=False):
        valid = track.manual_small.to_numpy() & track.speed_kf.between(0, MAX_SPEED).to_numpy() & track.xloc_kf.notna().to_numpy() & track.yloc_kf.notna().to_numpy()
        ticks, lanes = track.tick.to_numpy(), track.lane_kf.to_numpy()
        changes = np.r_[True, (np.diff(ticks) != 1) | (np.diff(lanes) != 0)] | ~valid
        blocks = np.cumsum(changes)
        sizes = pd.Series(blocks[valid]).value_counts()
        count += bool((sizes >= SAMPLES).any())
    return count


def prepare() -> None:
    design = {
        "dataset_id": "tgsim-i294-i90",
        "seed": SEED,
        "target_tracks": TARGET,
        "minimum_run_clusters": 6,
        "minimum_tracks_per_selected_run": 4,
        "seconds": 20,
        "samples": SAMPLES,
        "sample_hz": 10,
        "maximum_speed_mps": MAX_SPEED,
        "minimum_speed_mps": 0,
        "maximum_lane_projection_distance_m": MAX_LATERAL_DISTANCE,
        "projection": "Exact original lane segments indexed by Shapely STRtree; cumulative Euclidean segment lengths plus exact local projection; no geometry simplification.",
        "orientation": "Pooled same-track consecutive 0.1s moving displacements; require at least 95 percent sign agreement per lane.",
        "position_convention": "Front bumper following the extraction method described in FHWA-JPO-24-133 printed p26; data dictionary itself labels vehicle position without an explicit point convention.",
        "length_convention": "Median provider length_smoothed within the complete source track; gap = leader projected front - leader median length - follower projected front.",
        "automation_filter": "Exclude both follower and nearest leader if any source-track row is automation flagged or has unknown/non-small class. Infer nearest leader from all tracked classes first.",
        "sections": "Exclude auxiliary lanes, any vehicle footprint crossing a nonzero author ramp/weaving segment, and author-identified I294 overpasses with 6m boundary padding. No author I90 Moving overpass intervals were supplied.",
        "missing_observations": "A same-lane/time frame with any unprojectable vehicle front or body outside geometry coverage is excluded. No interpolation over missing times. Provider overpass interpolation outside explicit masks remains a limitation.",
        "candidate_windows": "Consecutive nonoverlapping 201-sample windows within complete constant-leader/lane blocks; retain the earliest accepted window per source track.",
        "smoothing": "Existing resample_following helper applies Savitzky-Golay window21,polynomial3 independently to both trajectories.",
        "selection": "Seeded per-run permutations, round-robin across eligible capture-qualified runs with at least4 tracks, one window per track, up to24. Add only the lowest numbered stationary capture yielding at least4 tracks; other stationary clips do not count as independent.",
        "adoption_threshold_changed": False,
        "model_outcomes_inspected": False,
        "original_five_cluster_manifest_sha256": digest(ROOT / "five-cluster-preparation" / "manifest.json"),
        "stationary_run1_feasibility_manifest_sha256": digest(ROOT / "stationary-run1-feasibility" / "manifest.json"),
        "preparation_environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": importlib.metadata.version("scipy"), "shapely": shapely.__version__},
        "bridge_ngsim_source": str(Path(inspect.getfile(resample_following))),
        "bridge_ngsim_source_sha256": digest(Path(inspect.getfile(resample_following))),
        "prepared_utc": datetime.now(timezone.utc).isoformat(),
    }
    design_path = OUTPUT / "eligibility-design.json"
    if (OUTPUT / "manifest.json").exists():
        raise RuntimeError("An existing prepared fixture must remain unchanged")
    design_path.write_text(json.dumps(design, indent=2) + "\n", encoding="utf-8")
    all_trials = {}
    info_by_group = {}
    run_reports = []
    sources = []
    for capture in CAPTURES:
        base = capture["root"]
        source_manifest_path = base / "source-manifest.json"
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        raw_file = base / capture["raw_file"]
        assert digest(raw_file) == source_manifest["file_sha256"]
        doi_metadata = ROOT / "alternate-probes" / (capture["doi"].split("/")[-1] + ".json")
        sources.append({"capture": capture["name"], "capture_date": capture["date"], "doi": capture["doi"], "source": "https://doi.org/" + capture["doi"], "source_manifest": str(source_manifest_path), "source_manifest_sha256": digest(source_manifest_path), "raw_file": str(raw_file), "raw_sha256": digest(raw_file), "raw_files": source_manifest["files"], "doi_metadata": str(doi_metadata), "doi_metadata_sha256": digest(doi_metadata)})
        dtype = {"id": "int32", "run_index": "int16", "lane_kf": "int8"}
        full = pd.read_csv(raw_file, dtype=dtype)
        for run, original in full.groupby("run_index", sort=True):
            run = int(run)
            run_label = f"{capture['name']}:run{run}"
            report = {"run": run_label, "source_run_index": run, "capture": capture["name"], "direction": capture["direction"].get(run, "northbound"), "raw_rows": len(original), "raw_tracks": int(original.id.nunique()), "status": "evaluated", "rejections": Counter(), "geometry": []}
            run_reports.append(report)
            frame = original.sort_values(["id", "time"]).copy()
            frame["tick"] = np.rint(frame.time * 10).astype(np.int64)
            assert np.all(np.abs(frame.time - frame.tick / 10) < 1e-6)
            duplicate = frame.duplicated(["id", "tick"], keep=False)
            if duplicate.any():
                varied = frame.loc[duplicate].groupby(["id", "tick"]).nunique(dropna=False)
                if varied.drop(columns=["time"], errors="ignore").gt(1).any(axis=None):
                    report.update(status="contradictory_duplicate_keys", duplicate_rows=int(duplicate.sum()), eligible_tracks=0)
                    continue
                frame = frame.drop_duplicates(["id", "tick"])
            report["duplicate_rows_removed"] = len(original) - len(frame)
            automation = frame[capture["automation"]].astype(str).str.strip().str.lower()
            ordinary = automation.eq("no") & frame.type_most_common.eq("small-vehicle")
            frame["manual_small"] = ordinary.groupby(frame.id).transform("all")
            frame["vehicle_length"] = frame.groupby("id").length_smoothed.transform("median")
            report["flagged_or_nonpassenger_tracks"] = int(frame.loc[~frame.manual_small, "id"].nunique())
            frame = frame.loc[frame.lane_kf.isin(capture["lanes"])].copy()
            report["necessary_speed_tracks"] = necessary_tracks(frame)
            if not report["necessary_speed_tracks"]:
                report.update(status="no_complete_low_speed_track", eligible_tracks=0)
                print(json.dumps({k:v for k,v in report.items() if k!='geometry'}), flush=True)
                continue
            geometry_files = sorted((base / "raw").glob(f"*-Run_{run}-geometry-with-ramps.csv"))
            if not geometry_files:
                report.update(status="missing_author_geometry", eligible_tracks=0)
                continue
            hashes = {digest(path) for path in geometry_files}
            if len(hashes) != 1:
                report.update(status="ambiguous_author_geometry", eligible_tracks=0)
                continue
            geometry_path = geometry_files[0]
            report.update(geometry_file=str(geometry_path), geometry_sha256=digest(geometry_path))
            geometry = pd.read_csv(geometry_path)
            projected = []
            try:
                for lane, part in frame.groupby("lane_kf", sort=True):
                    if f"x_{lane}" not in geometry.columns:
                        raise ValueError(f"No author geometry for lane{lane}")
                    result, summary = project_lane(part, geometry, int(lane))
                    projected.append(result)
                    report["geometry"].append(summary)
            except Exception as error:
                report.update(status="projection_validation_failed", error=str(error), eligible_tracks=0)
                continue
            frame = pd.concat(projected).sort_values(["lane_kf", "tick", "s"])
            scope = frame.groupby(["lane_kf", "tick"], sort=False)
            frame["ambiguous_frame"] = (~frame.geometry_valid).groupby([frame.lane_kf, frame.tick]).transform("any")
            frame["overpass"] = overpass_mask(capture["name"], run, frame.xloc_kf.to_numpy())
            leader_fields = {"id": "leader_id", "s": "leader_s", "manual_small": "leader_manual_small", "vehicle_length": "leader_length", "mainline": "leader_mainline", "overpass": "leader_overpass", "speed_kf": "leader_raw_speed"}
            leaders = scope[list(leader_fields)].shift(-1).rename(columns=leader_fields)
            frame = pd.concat([frame, leaders], axis=1)
            frame["pair_valid"] = (
                ~frame.ambiguous_frame
                & frame.manual_small
                & frame.leader_manual_small.eq(True)
                & frame.mainline
                & frame.leader_mainline.eq(True)
                & ~frame.overpass
                & frame.leader_overpass.eq(False)
                & frame.speed_kf.between(0, MAX_SPEED)
                & frame.leader_raw_speed.between(0, MAX_SPEED)
                & frame.leader_id.notna()
                & (frame.leader_s - frame.s - frame.leader_length > 0)
            )
            report["ambiguous_same_lane_frame_rows"] = int(frame.ambiguous_frame.sum())
            report["valid_pair_rows"] = int(frame.pair_valid.sum())
            windows = 0
            complete_blocks = 0
            for vehicle_id, track in frame.sort_values(["id", "tick"]).groupby("id", sort=False):
                valid = track.pair_valid.to_numpy()
                ticks = track.tick.to_numpy()
                lanes = track.lane_kf.to_numpy()
                leader_ids = track.leader_id.fillna(-1).to_numpy()
                changes = np.r_[True, (np.diff(ticks) != 1) | (np.diff(lanes) != 0) | (np.diff(leader_ids) != 0)] | ~valid
                blocks = np.cumsum(changes)
                first_kept = None
                group_id = f"{run_label}:track{int(vehicle_id)}"
                for block in np.unique(blocks[valid]):
                    segment = track.loc[valid & (blocks == block)]
                    if len(segment) < SAMPLES:
                        continue
                    complete_blocks += 1
                    for begin in range(0, len(segment) + 1 - SAMPLES, SAMPLES):
                        window = segment.iloc[begin:begin + SAMPLES]
                        assert np.all(np.diff(window.tick) == 1)
                        trial = resample_following(window.time.to_numpy(), window.s.to_numpy(), window.leader_s.to_numpy(), float(window.leader_length.iloc[0]), driver=group_id, source_group=group_id)
                        speeds = np.r_[trial.speed, trial.leader_speed]
                        if not np.isfinite(speeds).all() or speeds.min() < 0 or speeds.max() > MAX_SPEED:
                            report["rejections"]["derived_speed_outside_envelope"] += 1
                            continue
                        if not np.all(trial.gap > 0):
                            report["rejections"]["nonpositive_processed_gap"] += 1
                            continue
                        windows += 1
                        if first_kept is None:
                            first_kept = trial
                            info_by_group[group_id] = {"run": run_label, "capture": capture["name"], "source_run_index": run, "track_id": int(vehicle_id), "leader_id": int(window.leader_id.iloc[0]), "lane_id": int(window.lane_kf.iloc[0]), "direction": report["direction"], "first_time_s": float(window.time.iloc[0]), "last_time_s": float(window.time.iloc[-1]), "leader_median_length_m": float(window.leader_length.iloc[0]), "samples": len(window), "both_provider_automation_flags_no": True, "both_provider_class_small_vehicle": True}
                if first_kept is not None:
                    all_trials[group_id] = first_kept
            report["complete_pair_blocks"] = complete_blocks
            report["eligible_windows"] = windows
            report["eligible_tracks"] = sum(info["run"] == run_label for info in info_by_group.values())
            print(json.dumps({k:v for k,v in report.items() if k!='geometry'}), flush=True)
    by_run = defaultdict(list)
    for group, info in info_by_group.items():
        by_run[info["run"]].append(group)
    rng = np.random.default_rng(SEED)
    stationary_options = [run for run, groups in by_run.items() if run.startswith("i90-stationary:") and len(groups) >= 4]
    chosen_stationary = min(stationary_options, key=lambda run: int(re.search(r"run(\d+)$", run).group(1))) if stationary_options else None
    choices = {run: list(rng.permutation(sorted(groups))) for run, groups in sorted(by_run.items()) if len(groups) >= 4 and (not run.startswith("i90-stationary:") or run == chosen_stationary)}
    selected = []
    while len(selected) < TARGET and any(choices.values()):
        for run in sorted(choices):
            if choices[run] and len(selected) < TARGET:
                selected.append(choices[run].pop())
    selected.sort()
    pool_path = OUTPUT / "eligible-first-windows.npz"
    fixture_path = OUTPUT / "selected-trials.npz"
    if all_trials:
        save_trials([all_trials[group] for group in sorted(all_trials)], pool_path)
    if selected:
        save_trials([all_trials[group] for group in selected], fixture_path)
        restored = load_trials(fixture_path)
        assert [trial.group_id for trial in restored] == selected
        assert all(len(trial.time) == SAMPLES and abs(trial.duration - 20) < 1e-9 for trial in restored)
    selected_runs = sorted({info_by_group[group]["run"] for group in selected})
    l1_runs = [row for row in run_reports if row["capture"] == "i294-l1"]
    l1_eligible = [row["run"] for row in l1_runs if row.get("eligible_tracks", 0)]
    l1_result = {"status": "insufficient_independent_runs" if len(l1_eligible) < 6 else "eligible", "required_runs": 6, "eligible_runs": l1_eligible, "run_reports": l1_runs, "speed_envelope_unchanged": True, "model_outcomes_inspected": False}
    (OUTPUT / "l1-feasibility.json").write_text(json.dumps(l1_result, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "status": "fixture_ready" if len(selected) == TARGET and len(selected_runs) >= 6 else "descriptive_pilot_insufficient_independent_runs",
        **design,
        "fixture": str(fixture_path),
        "fixture_sha256": digest(fixture_path) if selected else None,
        "eligible_fixture": str(pool_path),
        "eligible_fixture_sha256": digest(pool_path) if all_trials else None,
        "source_web_dois": ["https://doi.org/" + capture["doi"] for capture in CAPTURES],
        "license": "United States Government Public Domain License",
        "license_source": "DataCite DOI registration metadata supplied by the dataset publisher",
        "sources": sources,
        "eligibility_design_sha256": digest(design_path),
        "preparation_script_sha256": digest(Path(__file__)),
        "run_reports": run_reports,
        "eligible_tracks": len(all_trials),
        "eligible_runs": sorted(by_run),
        "selected_groups": selected,
        "selected_runs": selected_runs,
        "chosen_stationary_run": chosen_stationary,
        "run_by_group": {group: info_by_group[group]["run"] for group in selected},
        "selected_trials": {group: info_by_group[group] for group in selected},
        "all_eligible_track_metadata": info_by_group,
        "limitations": ["Vehicle tracks do not identify persistent human participants.", "Ordinary vehicles may use unlabelled assistance systems; provider flags identify the test fleet.", "Leader identities are inferred from nearest tracked same-lane front positions.", "Provider lengths are image bounding-box proxies.", "I90 Moving overpass interpolation has no supplied location mask.", "Runs from the same collection day and roadway may share traffic conditions.", "The recorded geometry is reduced to a scalar longitudinal trajectory for the straight-road following experiment."],
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "prepared", "status": manifest["status"], "eligible_tracks": len(all_trials), "selected_tracks": len(selected), "selected_runs": selected_runs, "fixture": str(fixture_path), "fixture_sha256": manifest["fixture_sha256"]}), flush=True)


if __name__ == "__main__":
    prepare()
