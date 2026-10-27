"""Scientific metadata and checksummed bundles preserve reproducible inputs."""

import importlib.metadata
import json
import os
import subprocess
import threading
import zipfile

import pytest

from embodied_human import HumanConfig, build_human
from embodied_human.bench.benchmarks import Result, run
from embodied_human.bench.references import REFERENCES
from embodied_human.reporting import (
    canonical_json,
    environment_record,
    load_replay,
    make_report,
    parameter_record,
    sha256,
    source_state,
    write_bundle,
    write_json,
)


def test_resolved_parameters_include_defaults_and_mark_actual_overrides():
    config = HumanConfig().replace("memory", "act_r", capacity=2)
    record = parameter_record(build_human(config, seed=7))
    resolved = record["resolved_config"]["memory"]
    assert resolved["name"] == "act_r"
    assert resolved["params"]["capacity"] == 2
    assert resolved["params"]["decay"] == 0.5
    rows = {row["parameter"]: row for row in record["parameter_provenance"]}
    assert rows["MemoryParameters.capacity"]["overridden"]
    assert not rows["MemoryParameters.decay"]["overridden"]
    assert rows["MemoryParameters.capacity"]["source_applies_to"] == "baseline_default"


@pytest.mark.skipif(os.name != "nt", reason="Windows denies replacing open files")
def test_atomic_json_write_retries_until_windows_reader_closes(tmp_path, monkeypatch):
    path = tmp_path / "progress.json"
    write_json(path, {"completed": 0})
    denied = threading.Event()
    original_replace = type(path).replace

    def replace(source, target):
        try:
            return original_replace(source, target)
        except PermissionError:
            denied.set()
            raise

    monkeypatch.setattr(type(path), "replace", replace)
    errors = []

    def write():
        try:
            write_json(path, {"completed": 1})
        except Exception as exc:
            errors.append(exc)

    with path.open("r", encoding="utf-8") as reader:
        writer = threading.Thread(target=write)
        writer.start()
        try:
            assert denied.wait(timeout=2), "reader did not exercise replacement denial"
            assert json.load(reader) == {"completed": 0}
        finally:
            reader.close()
            writer.join(timeout=3)
    assert not writer.is_alive()
    assert not errors
    assert json.loads(path.read_text()) == {"completed": 1}
    assert not path.with_name("progress.json.tmp").exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows denies replacing open files")
def test_atomic_json_write_bounds_windows_lock_retries(tmp_path, monkeypatch):
    path = tmp_path / "progress.json"
    write_json(path, {"completed": 0})
    delays = []
    monkeypatch.setattr("embodied_human.reporting.time.sleep", delays.append)
    with path.open("r", encoding="utf-8") as reader:
        with pytest.raises(PermissionError):
            write_json(path, {"completed": 1})
        assert json.load(reader) == {"completed": 0}
    assert len(delays) == 19
    assert sum(delays) < 2
    assert not path.with_name("progress.json.tmp").exists()


def test_report_exports_reference_conditions_and_separate_validation_roles():
    result = Result("role_example", ("eyes", "vestibular"))
    metrics = (
        "eyes.latency_median",
        "vestibular.vertical_msi_error",
        "vestibular.time_course_msi_error",
    )
    for metric in metrics:
        result.score_against(metric, REFERENCES[metric].value)
    report = make_report(build_human(seed=1), [result], seed=1)
    assert report["seed"] == 1
    summary = report["validation_summary"]
    for role in ("consistency", "calibration", "held_out"):
        assert summary[role] == {"metrics": 1, "passed": 1}
    detail = report["results"][0]
    for metric in metrics:
        assert (
            detail["references"][metric]["conditions"] == REFERENCES[metric].conditions
        )
        assert detail["validation"][metric]["evaluation_data"]
        assert detail["validation"][metric]["fitting_data"]
    unknown = Result("unscored", ("memory",), metrics={"unknown.metric": 1.0})
    assert unknown.as_dict()["validation"]["unknown.metric"]["role"] == "unclassified"


@pytest.fixture
def stable_source(monkeypatch):
    # Other implementation agents can edit source while these tests execute.
    snapshot = source_state()
    monkeypatch.setattr("embodied_human.reporting.source_state", lambda: snapshot)
    return snapshot


@pytest.fixture
def bundle(tmp_path, stable_source):
    human = build_human(HumanConfig().replace("memory", "act_r", capacity=2), seed=7)
    report = make_report(human, run(human, ["memory.span"], seed=7), seed=7)
    replay = {
        "kind": "benchmark",
        "config": report["resolved_config"],
        "benchmarks": ["memory.span"],
        "seed": 7,
    }
    manifest = write_bundle(tmp_path, report, replay=replay)
    return tmp_path, report, replay, manifest


def test_bundle_hashes_files_and_contains_references_environment_and_replay(bundle):
    directory, report, replay, manifest = bundle
    assert load_replay(directory / "manifest.json") == replay
    for filename, digest in manifest["artifacts"].items():
        assert sha256((directory / filename).read_bytes()) == digest
    references = json.loads((directory / "references.json").read_text())
    assert (
        references["memory.span"]["conditions"] == REFERENCES["memory.span"].conditions
    )
    assert manifest["reference_sha256"] == sha256(
        (directory / "references.json").read_bytes()
    )
    assert "--replay" in manifest["replay_argv"]
    assert "manifest.json" in (directory / "REPLAY.txt").read_text()
    with zipfile.ZipFile(directory / "bundle.zip") as archive:
        assert set(archive.namelist()) == {*manifest["artifacts"], "manifest.json"}
        assert json.loads(archive.read("report.json")) == report


def test_replay_rejects_changed_artifact(bundle):
    directory, _, _, _ = bundle
    with (directory / "report.json").open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="checksum differs"):
        load_replay(directory / "manifest.json")


def test_replay_rejects_source_mismatch_and_requires_explicit_comparison(
    bundle, monkeypatch
):
    directory, _, replay, manifest = bundle
    changed = {**manifest["environment"]["source"], "package_sha256": "0" * 64}
    monkeypatch.setattr("embodied_human.reporting.source_state", lambda: changed)
    with pytest.raises(ValueError, match="source differs"):
        load_replay(directory / "manifest.json")
    assert (
        load_replay(directory / "manifest.json", allow_source_mismatch=True) == replay
    )


def test_bundle_rejects_paths_outside_its_directory(tmp_path, stable_source):
    with pytest.raises(ValueError, match="plain filenames"):
        write_bundle(
            tmp_path / "run",
            {},
            replay={"kind": "experiment"},
            extra_files={"../escaped.txt": "unexpected"},
        )
    assert not (tmp_path / "escaped.txt").exists()


def test_replay_checks_dependency_versions_and_requires_explicit_comparison(
    bundle, monkeypatch
):
    directory, _, replay, manifest = bundle
    recorded = manifest["environment"]
    changed = {
        **recorded,
        "dependencies": {**recorded["dependencies"], "numpy": "0.0.test"},
    }
    monkeypatch.setattr("embodied_human.reporting.environment_record", lambda: changed)
    with pytest.raises(ValueError, match="dependency versions differ: numpy"):
        load_replay(directory / "manifest.json")
    assert (
        load_replay(directory / "manifest.json", allow_environment_mismatch=True)
        == replay
    )
    newly_installed = {
        **recorded,
        "dependencies": {**recorded["dependencies"], "new-optional-backend": "1.0"},
    }
    monkeypatch.setattr(
        "embodied_human.reporting.environment_record", lambda: newly_installed
    )
    with pytest.raises(ValueError, match="dependency versions differ: new-optional"):
        load_replay(directory / "manifest.json")


def test_environment_records_versions_and_relative_source_identifiers(stable_source):
    record = environment_record()
    assert record["python"]
    assert record["dependencies"]["numpy"]
    assert len(record["source"]["package_sha256"]) == 64
    assert "memory.py" in record["source"]["package_files"]
    assert all(
        not name.startswith(("/", "C:")) for name in record["source"]["package_files"]
    )
    assert canonical_json({"b": 2, "a": float("nan")}) == '{"a":null,"b":2}'


def test_replay_checks_recorded_optional_dependencies_and_python(bundle, monkeypatch):
    directory, _, replay, manifest = bundle
    recorded = manifest["environment"]
    changed = {
        **recorded,
        "dependencies": {**recorded["dependencies"], "mujoco": "0.0.test"},
    }
    if "mujoco" in recorded["dependencies"]:
        monkeypatch.setattr(
            "embodied_human.reporting.environment_record", lambda: changed
        )
        with pytest.raises(ValueError, match="dependency versions differ: mujoco"):
            load_replay(directory / "manifest.json")
    changed = {**recorded, "python": "0.0.test"}
    monkeypatch.setattr("embodied_human.reporting.environment_record", lambda: changed)
    with pytest.raises(ValueError, match="Python runtime differs"):
        load_replay(directory / "manifest.json")
    assert (
        load_replay(directory / "manifest.json", allow_environment_mismatch=True)
        == replay
    )


@pytest.fixture
def editable_dependency(tmp_path, monkeypatch):
    directory = tmp_path / "editable dependency with spaces"
    directory.mkdir()
    (directory / "model.py").write_text("GAIN = 1\n", encoding="utf-8")
    original = importlib.metadata.distribution

    class EditableDistribution:
        version = "0.1"

        def read_text(self, filename):
            if filename == "direct_url.json":
                return json.dumps(
                    {"url": directory.as_uri(), "dir_info": {"editable": True}}
                )
            return None

    def distribution(name):
        return EditableDistribution() if name == "myosuite-carla" else original(name)

    monkeypatch.setattr(
        "embodied_human.reporting.importlib.metadata.distribution", distribution
    )
    monkeypatch.setattr(
        "embodied_human.reporting._DEPENDENCIES",
        ("embodied-human", "numpy", "scipy", "myosuite-carla"),
    )
    return directory


def _dependency_git(directory, *args):
    return subprocess.check_output(
        ["git", "-C", str(directory), *args], stderr=subprocess.DEVNULL
    )


def _commit_dependency(directory, message):
    # This is an ephemeral test repository, not a commit to the user's project.
    return _dependency_git(
        directory,
        "-c",
        "user.name=Test fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "--allow-empty",
        "-qm",
        message,
    )


def _init_dependency_git(directory):
    _dependency_git(directory, "init", "-q")
    _dependency_git(directory, "add", "model.py")
    _commit_dependency(directory, "initial fixture")


def _editable_bundle(directory):
    replay = {"kind": "experiment", "request": {"config": {}, "seeds": [0]}}
    manifest = write_bundle(directory, {}, replay=replay)
    return directory / "manifest.json", replay, manifest


def test_replay_rejects_same_version_editable_code_changes(
    tmp_path, stable_source, editable_dependency
):
    _init_dependency_git(editable_dependency)
    path, replay, manifest = _editable_bundle(tmp_path / "run")
    saved = manifest["environment"]
    assert saved["editable_sources"]["myosuite-carla"]["kind"] == "git"
    assert saved["editable_sources"]["myosuite-carla"]["directory"] == "."
    assert str(editable_dependency) not in canonical_json(saved["editable_sources"])
    assert editable_dependency.as_uri() not in canonical_json(saved["editable_sources"])
    (editable_dependency / "model.py").write_text("GAIN = 2\n", encoding="utf-8")
    assert environment_record()["dependencies"] == saved["dependencies"]
    with pytest.raises(ValueError, match="editable dependency sources differ"):
        load_replay(path)
    with pytest.raises(ValueError, match="editable dependency sources differ"):
        load_replay(path, allow_source_mismatch=True)
    _dependency_git(editable_dependency, "add", "model.py")
    with pytest.raises(ValueError, match="editable dependency sources differ"):
        load_replay(path)
    assert load_replay(path, allow_environment_mismatch=True) == replay


def test_replay_rejects_editable_revision_changes_without_version_changes(
    tmp_path, stable_source, editable_dependency
):
    _init_dependency_git(editable_dependency)
    path, _, manifest = _editable_bundle(tmp_path / "run")
    before = manifest["environment"]["editable_sources"]["myosuite-carla"]
    _commit_dependency(editable_dependency, "second fixture revision")
    after = environment_record()["editable_sources"]["myosuite-carla"]
    assert before["git_diff_sha256"] == after["git_diff_sha256"]
    assert before["git_revision"] != after["git_revision"]
    with pytest.raises(ValueError, match="editable dependency sources differ"):
        load_replay(path)


def test_editable_git_capture_hashes_untracked_source_changes(
    tmp_path, stable_source, editable_dependency
):
    _init_dependency_git(editable_dependency)
    source = editable_dependency / "extra.py"
    source.write_text("CONTROL = 1\n", encoding="utf-8")
    path, _, manifest = _editable_bundle(tmp_path / "run")
    before = manifest["environment"]["editable_sources"]["myosuite-carla"]
    assert before["untracked_source_files"] == {"extra.py": sha256(source.read_bytes())}
    source.write_text("CONTROL = 2\n", encoding="utf-8")
    after = environment_record()["editable_sources"]["myosuite-carla"]
    assert before["git_diff_sha256"] == after["git_diff_sha256"]
    assert before["untracked_source_files"] != after["untracked_source_files"]
    with pytest.raises(ValueError, match="editable dependency sources differ"):
        load_replay(path)


def test_clean_git_capture_does_not_hash_tracked_asset_trees(
    stable_source, editable_dependency, monkeypatch
):
    _init_dependency_git(editable_dependency)

    def unexpected_file_hash(path):
        raise AssertionError("clean tracked files must be identified by Git revision")

    monkeypatch.setattr("embodied_human.reporting._file_sha256", unexpected_file_hash)
    record = environment_record()["editable_sources"]["myosuite-carla"]
    assert record["kind"] == "git" and not record["git_dirty"]


def test_non_git_editable_capture_tracks_assets_and_excludes_generated_files(
    tmp_path, stable_source, editable_dependency
):
    models = editable_dependency / "models"
    models.mkdir()
    asset = models / "body.xml"
    asset.write_text("<mujoco/>\n", encoding="utf-8")
    for name in (".venv", "__pycache__", "build", "artifacts"):
        generated = editable_dependency / name
        generated.mkdir()
        (generated / "ignored.py").write_text("not a source input\n", encoding="utf-8")
    path, replay, manifest = _editable_bundle(tmp_path / "run")
    record = manifest["environment"]["editable_sources"]["myosuite-carla"]
    assert record["kind"] == "directory"
    assert set(record["files"]) == {"model.py", "models/body.xml"}
    assert str(editable_dependency) not in canonical_json(record)
    (editable_dependency / "artifacts" / "ignored.py").write_text("changed\n")
    assert load_replay(path) == replay
    asset.write_text("<mujoco model='changed'/>\n", encoding="utf-8")
    with pytest.raises(ValueError, match="editable dependency sources differ"):
        load_replay(path)


def test_unreadable_editable_sources_cannot_be_certified_for_exact_replay(
    tmp_path, stable_source, editable_dependency, monkeypatch
):
    def cannot_walk(directory, *, onerror):
        onerror(PermissionError("unreadable test source"))
        return []

    monkeypatch.setattr("embodied_human.reporting.os.walk", cannot_walk)
    path, replay, manifest = _editable_bundle(tmp_path / "run")
    record = manifest["environment"]["editable_sources"]["myosuite-carla"]
    assert record["kind"] == "unavailable"
    with pytest.raises(ValueError, match="editable dependency sources differ"):
        load_replay(path)
    assert load_replay(path, allow_environment_mismatch=True) == replay


def test_replay_rejects_noneditable_vcs_revision_changes_with_the_same_version(
    tmp_path, stable_source, monkeypatch
):
    original = importlib.metadata.distribution
    vcs = {"vcs": "git", "commit_id": "a" * 40, "requested_revision": "stable"}

    class VCSDistribution:
        version = "0.1.2"

        def read_text(self, filename):
            if filename == "direct_url.json":
                return json.dumps(
                    {
                        "url": "https://private:secret@example.invalid/model.git",
                        "vcs_info": vcs,
                    }
                )
            return None

    monkeypatch.setattr(
        "embodied_human.reporting.importlib.metadata.distribution",
        lambda name: VCSDistribution() if name == "touchsim" else original(name),
    )
    monkeypatch.setattr(
        "embodied_human.reporting._DEPENDENCIES",
        ("embodied-human", "numpy", "scipy", "touchsim"),
    )
    path, replay, manifest = _editable_bundle(tmp_path / "run")
    recorded = manifest["environment"]
    assert recorded["dependency_vcs_sources"]["touchsim"] == vcs
    assert "secret" not in canonical_json(recorded)
    assert "example.invalid" not in canonical_json(recorded)
    vcs["commit_id"] = "b" * 40
    assert environment_record()["dependencies"] == recorded["dependencies"]
    with pytest.raises(ValueError, match="dependency VCS sources differ: touchsim"):
        load_replay(path)
    assert load_replay(path, allow_environment_mismatch=True) == replay
    vcs["commit_id"] = "a" * 40
    vcs["requested_revision"] = "different-tag"
    with pytest.raises(ValueError, match="dependency VCS sources differ: touchsim"):
        load_replay(path)
