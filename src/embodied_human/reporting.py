"""Reports and replayable run bundles."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import time
import zipfile
from collections import Counter
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import url2pathname

from .bench.benchmarks import Result, json_safe
from .bench.references import REFERENCES
from .bench.validation import validation_summary
from .core import Human

SCHEMA_VERSION = 1
_DEPENDENCIES = (
    "embodied-human",
    "numpy",
    "scipy",
    "SALib",
    "pyactup",
    "pyddm",
    "mujoco",
    "embodied-ui",
    "myosuite-carla",
    "MyoSuite",
    "myo-sim",
    "touchsim",
    "numba",
    "scikit-image",
    "scikit-learn",
    "mosqito",
    "matplotlib",
    "pandas",
    "trimesh",
)

_SOURCE_EXTENSIONS = {
    ".py",
    ".pyi",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".rs",
    ".cu",
    ".so",
    ".pyd",
    ".dll",
    ".js",
    ".mjs",
    ".ts",
    ".tsx",
    ".jsx",
    ".css",
    ".html",
    ".xml",
    ".urdf",
    ".mjcf",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".txt",
    ".csv",
    ".npy",
    ".npz",
    ".mat",
    ".h5",
    ".hdf5",
    ".obj",
    ".stl",
    ".msh",
    ".mesh",
    ".ply",
    ".glb",
    ".gltf",
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".svg",
    ".ipynb",
    ".pkl",
    ".pickle",
    ".pt",
    ".pth",
    ".onnx",
    ".bin",
    ".mjbin",
}
_EXCLUDED_SOURCE_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".cache",
    ".eggs",
    ".ipynb_checkpoints",
    "node_modules",
    "build",
    "dist",
    "artifacts",
    "evidence",
    "results",
    "outputs",
    "output",
    "logs",
    "runs",
    "wandb",
}


def canonical_json(value: Any) -> str:
    return json.dumps(
        json_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_file(path: Path) -> bool:
    return not any(
        part.casefold() in _EXCLUDED_SOURCE_DIRS
        or part.casefold().endswith(".egg-info")
        for part in path.parts[:-1]
    ) and (path.suffix.casefold() in _SOURCE_EXTENSIONS or path.name == "LICENSE")


def _dependency_source(directory: Path) -> dict[str, Any]:
    """Identify source content without recording the checkout's absolute path."""
    if not directory.is_dir():
        return {"kind": "unavailable", "reason": "editable directory is missing"}

    def git(root: Path, *args: str) -> bytes:
        return subprocess.check_output(
            ["git", "-C", str(root), *args], stderr=subprocess.DEVNULL, timeout=15
        )

    try:
        checkout = Path(
            os.fsdecode(git(directory, "rev-parse", "--show-toplevel")).strip()
        )
        revision = git(checkout, "rev-parse", "HEAD").decode().strip()
        diff = git(
            checkout,
            "diff",
            "--binary",
            "--no-ext-diff",
            "--no-textconv",
            "--submodule=diff",
            "HEAD",
            "--",
        )
        untracked = {}
        for entry in git(
            checkout, "ls-files", "--others", "--exclude-standard", "-z"
        ).split(b"\0"):
            if not entry:
                continue
            relative = Path(os.fsdecode(entry))
            path = checkout / relative
            if _source_file(relative) and path.is_file():
                untracked[relative.as_posix()] = _file_sha256(path)
        return {
            "kind": "git",
            "directory": directory.relative_to(checkout.resolve()).as_posix(),
            "git_revision": revision,
            "git_dirty": bool(diff or untracked),
            "git_diff_sha256": sha256(diff),
            "untracked_source_files": dict(sorted(untracked.items())),
        }
    except (OSError, subprocess.SubprocessError, ValueError):
        # A source directory copied without Git still needs a content identity.
        hashes = {}

        def unreadable(exc: OSError) -> None:
            raise exc

        try:
            for root, directories, filenames in os.walk(directory, onerror=unreadable):
                directories[:] = sorted(
                    name
                    for name in directories
                    if name.casefold() not in _EXCLUDED_SOURCE_DIRS
                    and not name.casefold().endswith(".egg-info")
                )
                for name in sorted(filenames):
                    path = Path(root) / name
                    relative = path.relative_to(directory)
                    if _source_file(relative) and path.is_file():
                        hashes[relative.as_posix()] = _file_sha256(path)
        except OSError:
            return {"kind": "unavailable", "reason": "editable source cannot be read"}
        return {
            "kind": "directory",
            "files": hashes,
            "sha256": sha256(canonical_json(hashes).encode()),
        }


def _dependency_sources() -> tuple[dict[str, Any], dict[str, Any]]:
    editable_sources: dict[str, Any] = {}
    vcs_sources: dict[str, Any] = {}
    for name in _DEPENDENCIES:
        if name == "embodied-human":
            continue
        try:
            distribution = importlib.metadata.distribution(name)
            text = distribution.read_text("direct_url.json")
        except importlib.metadata.PackageNotFoundError:
            continue
        except (OSError, UnicodeError):
            editable_sources[name] = {
                "kind": "unavailable",
                "reason": "source metadata cannot be read",
            }
            continue
        if not text:
            continue
        try:
            direct = json.loads(text)
            if vcs := direct.get("vcs_info"):
                if not all(
                    isinstance(vcs[key], str) and vcs[key]
                    for key in ("vcs", "commit_id")
                ):
                    raise ValueError("invalid VCS metadata")
                revision = vcs.get("requested_revision")
                if revision is not None and not isinstance(revision, str):
                    raise ValueError("invalid requested revision")
                vcs_sources[name] = {
                    "vcs": vcs["vcs"],
                    "commit_id": vcs["commit_id"],
                    "requested_revision": revision,
                }
            if not direct.get("dir_info", {}).get("editable"):
                continue
            url = urlparse(direct["url"])
            if url.scheme != "file":
                raise ValueError("editable URL is not local")
            path = url.path
            if url.netloc and url.netloc != "localhost":
                path = "//" + url.netloc + path
            editable_sources[name] = _dependency_source(
                Path(url2pathname(path)).resolve()
            )
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            editable_sources[name] = {
                "kind": "unavailable",
                "reason": "editable metadata cannot be resolved",
            }
    return editable_sources, vcs_sources


def source_state() -> dict[str, Any]:
    """Hash package source and identify the checkout when present."""
    package = Path(__file__).resolve().parent
    paths = sorted(
        p
        for p in package.rglob("*")
        if p.is_file()
        and p.suffix in (".py", ".js", ".css", ".html", ".md", ".json", ".txt")
        and "__pycache__" not in p.parts
    )
    files = {p.relative_to(package).as_posix(): sha256(p.read_bytes()) for p in paths}
    state: dict[str, Any] = {
        "package_files": files,
        "package_sha256": sha256(canonical_json(files).encode()),
        "git_revision": None,
        "git_dirty": None,
        "git_diff_sha256": None,
    }
    checkout = next((p for p in package.parents if (p / ".git").exists()), None)
    if checkout is not None:

        def git(*args: str) -> bytes:
            return subprocess.check_output(
                ["git", "-C", str(checkout), *args],
                stderr=subprocess.DEVNULL,
                timeout=10,
            )

        try:
            state["git_revision"] = git("rev-parse", "HEAD").decode().strip()
            state["git_dirty"] = bool(git("status", "--porcelain").strip())
            state["git_diff_sha256"] = sha256(git("diff", "--binary", "HEAD"))
        except (OSError, subprocess.SubprocessError):
            pass
    return state


def environment_record() -> dict[str, Any]:
    """Record relevant versions without credentials or machine paths."""
    versions = {}
    for name in _DEPENDENCIES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    editable_sources, vcs_sources = _dependency_sources()
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": {"system": platform.system(), "machine": platform.machine()},
        "dependencies": versions,
        "editable_sources": editable_sources,
        "dependency_vcs_sources": vcs_sources,
        "installed_distributions": {
            name: version
            for name, version in sorted(
                (distribution.metadata["Name"], distribution.version)
                for distribution in importlib.metadata.distributions()
                if distribution.metadata["Name"]
            )
        },
        "source": source_state(),
    }


def parameter_record(human: Human) -> dict[str, Any]:
    resolved = {}
    for category, module in human.modules.items():
        params = module.parameters
        values = dataclasses.asdict(params) if params is not None else {}  # type: ignore[call-overload]
        resolved[category] = {
            "name": getattr(human.config, category).name,
            "params": values,
        }
    rows = human.provenance()
    for row in rows:
        field = row["parameter"].rsplit(".", 1)[-1]
        row["overridden"] = field in getattr(human.config, row["category"]).params
        row["source_applies_to"] = "baseline_default"
    return json_safe({"resolved_config": resolved, "parameter_provenance": rows})


def make_report(
    human: Human, results: list[Result], *, seed: int, wall_seconds: float = 0.0
) -> dict[str, Any]:
    values = [result.as_dict() for result in results]
    return json_safe(
        {
            "schema_version": SCHEMA_VERSION,
            "package": "embodied-human " + importlib.metadata.version("embodied-human"),
            "python": platform.python_version(),
            "seed": seed,
            "config": human.config.as_dict(),
            **parameter_record(human),
            "parameter_sources": dict(
                Counter(row["kind"] for row in human.provenance())
            ),
            "wall_seconds": round(wall_seconds, 3),
            "validation_summary": validation_summary(values),
            "validation_summary_unit": "scored metric evaluations",
            "validation_policy": (
                "Roles describe declared baseline data use. "
                "Held out statistics can share "
                "participants with calibration data. User fitting requires updated "
                "declarations."
            ),
            "results": values,
        }
    )


def write_json(path: Path, value: Any) -> None:
    """Replace output atomically for polling clients."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(value), indent=2, allow_nan=False), encoding="utf-8"
    )
    try:
        for attempt in range(20):
            try:
                temporary.replace(path)
                break
            except PermissionError as exc:
                # Windows readers can temporarily deny replacement of an open file.
                if (
                    os.name != "nt"
                    or getattr(exc, "winerror", None) not in (5, 32, 33)
                    or attempt == 19
                ):
                    raise
                time.sleep(min(0.01 * 2**attempt, 0.1))
    finally:
        temporary.unlink(missing_ok=True)


def write_bundle(
    directory: Path | str,
    report: dict[str, Any],
    *,
    replay: Mapping[str, Any],
    extra_files: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Write a manifest, report, references, environment and ZIP archive."""
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "report.json", report)
    write_json(
        output / "references.json",
        {name: dataclasses.asdict(reference) for name, reference in REFERENCES.items()},
    )
    environment = environment_record()
    write_json(output / "environment.json", environment)
    package = Path(__file__).resolve().parent
    with zipfile.ZipFile(
        output / "package-source.zip", "w", zipfile.ZIP_DEFLATED
    ) as archive:
        for name in environment["source"]["package_files"]:
            archive.write(package / name, "embodied_human/" + name)
        license_path = package.parent.parent / "LICENSE"
        if license_path.is_file():
            archive.write(license_path, "LICENSE")
        else:
            distribution = importlib.metadata.distribution("embodied-human")
            for item in distribution.files or []:
                if str(item).replace("\\", "/").endswith("/licenses/LICENSE"):
                    archive.write(str(distribution.locate_file(item)), "LICENSE")
                    break
    for name, content in (extra_files or {}).items():
        if Path(name).name != name or name in (
            "manifest.json",
            "bundle.zip",
            "package-source.zip",
        ):
            raise ValueError("bundle files must use plain filenames")
        (output / name).write_text(content, encoding="utf-8")
    module = (
        "embodied_human.bench"
        if replay.get("kind") == "benchmark"
        else "embodied_human.experiments"
    )
    argv = ["python", "-m", module, "--replay", "manifest.json", "--out", "replayed"]
    text = (
        "Run from this directory with the recorded source and dependency versions.\n"
        "Replay: " + " ".join(argv) + "\n"
        "Report details include available protocol traces and intermediate values.\n"
        "Seed intervals describe simulation variation across the declared seeds.\n"
    )
    (output / "REPLAY.txt").write_text(text, encoding="utf-8")
    names = [
        "report.json",
        "references.json",
        "environment.json",
        "package-source.zip",
        "REPLAY.txt",
        *(extra_files or {}),
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment": environment,
        "replay": dict(replay),
        "replay_argv": argv,
        "artifacts": {name: sha256((output / name).read_bytes()) for name in names},
        "reference_sha256": sha256((output / "references.json").read_bytes()),
    }
    write_json(output / "manifest.json", manifest)
    with zipfile.ZipFile(output / "bundle.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in [*names, "manifest.json"]:
            archive.write(output / name, name)
    return manifest


def load_replay(
    path: Path | str,
    *,
    allow_source_mismatch: bool = False,
    allow_environment_mismatch: bool = False,
) -> dict[str, Any]:
    """Validate artifact hashes before returning a data only replay request."""
    manifest_path = Path(path)
    if manifest_path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("manifest is too large")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported manifest schema")
    for name, expected in manifest.get("artifacts", {}).items():
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError("invalid artifact filename")
        if sha256((manifest_path.parent / name).read_bytes()) != expected:
            raise ValueError(f"artifact checksum differs: {name}")
    previous = manifest["environment"]["source"]["package_sha256"]
    if not allow_source_mismatch and previous != source_state()["package_sha256"]:
        raise ValueError(
            "package source differs; use --allow-source-mismatch for a comparison"
        )
    replay = manifest["replay"]
    required = {"embodied-human", "numpy", "scipy"}
    config = replay.get("config", replay.get("request", {}).get("config", {}))
    choices = {
        value.get("name") for value in config.values() if isinstance(value, dict)
    }
    for choice, dependency in (
        ("pyactup", "pyactup"),
        ("pyddm", "pyddm"),
        ("contact_map", "mujoco"),
        ("touchsim", "touchsim"),
    ):
        if choice in choices:
            required.add(dependency)
    if replay.get("request", {}).get("sensitivity"):
        required.add("SALib")
    saved = manifest["environment"]["dependencies"]
    current_environment = environment_record()
    current = current_environment["dependencies"]
    required.update(saved)
    required.update(current)
    mismatches = [
        name for name in required if saved.get(name) != current.get(name)
    ]
    if mismatches and not allow_environment_mismatch:
        raise ValueError(
            "dependency versions differ: "
            + ", ".join(sorted(mismatches))
            + "; use --allow-environment-mismatch for a comparison"
        )
    for field, label in (
        ("editable_sources", "editable dependency sources"),
        ("dependency_vcs_sources", "dependency VCS sources"),
    ):
        saved_sources = manifest["environment"].get(field, {})
        current_sources = current_environment.get(field, {})
        changed_sources = [
            name
            for name in saved_sources.keys() | current_sources.keys()
            if saved_sources.get(name) != current_sources.get(name)
            or saved_sources.get(name, {}).get("kind") == "unavailable"
            or current_sources.get(name, {}).get("kind") == "unavailable"
        ]
        if changed_sources and not allow_environment_mismatch:
            raise ValueError(
                label
                + " differ: "
                + ", ".join(sorted(changed_sources))
                + "; use --allow-environment-mismatch for a comparison"
            )
    if not allow_environment_mismatch and any(
        manifest["environment"][key] != current_environment[key]
        for key in ("python", "implementation")
    ):
        raise ValueError(
            "Python runtime differs; use --allow-environment-mismatch for a comparison"
        )
    return replay
