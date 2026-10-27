"""Owned subprocess jobs for the local experiment interface."""

from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DOWNLOADS = {
    "report.json": "application/json",
    "manifest.json": "application/json",
    "results.csv": "text/csv",
    "plot.svg": "image/svg+xml",
    "bundle.zip": "application/zip",
}
TERMINAL = {"completed", "failed", "cancelled", "timed_out"}


def worker_command(request: Path, directory: Path) -> list[str]:
    """Build an argument array for the package's experiment worker."""
    return [
        sys.executable,
        "-m",
        "embodied_human.experiments",
        "--request",
        str(request),
        "--out",
        str(directory),
    ]


@dataclass
class _Job:
    identifier: str
    directory: Path
    process: subprocess.Popen[bytes]
    started: float
    timeout: float
    state: str = "running"
    error: str | None = None
    finished: float | None = None


class JobManager:
    """Run at most two owned workers and retain files until server close."""

    def __init__(self, limit: int = 2) -> None:
        self.limit = limit
        self._temporary = tempfile.TemporaryDirectory(prefix="embodied-human-jobs-")
        self._root = Path(self._temporary.name)
        self._jobs: dict[str, _Job] = {}
        self._lock = threading.RLock()
        self._closed = False

    def start(self, spec: dict[str, Any], timeout: float = 900) -> dict[str, Any]:
        """Start a validated request without executing a shell."""
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or not 1 <= timeout <= 3600
        ):
            raise ValueError("timeout_seconds must lie between 1 and 3600")
        with self._lock:
            if self._closed:
                raise ValueError("experiment server is closed")
            self._refresh_all()
            if sum(job.state == "running" for job in self._jobs.values()) >= self.limit:
                raise ValueError("two experiments are running; wait or cancel a job")
            identifier = uuid.uuid4().hex
            directory = self._root / identifier
            directory.mkdir()
            request = directory / "request.json"
            request.write_text(json.dumps(spec, allow_nan=False), encoding="utf-8")
            with (directory / "worker.log").open("wb") as log:
                process = subprocess.Popen(
                    worker_command(request, directory),
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            job = _Job(identifier, directory, process, time.monotonic(), float(timeout))
            self._jobs[identifier] = job
            threading.Thread(target=self._monitor, args=(job,), daemon=True).start()
            return self._snapshot(job)

    def _monitor(self, job: _Job) -> None:
        while True:
            with self._lock:
                self._refresh(job)
                if job.state in TERMINAL:
                    return
            threading.Event().wait(0.1)

    def _refresh_all(self) -> None:
        for job in self._jobs.values():
            self._refresh(job)

    def _refresh(self, job: _Job) -> None:
        if job.state in TERMINAL:
            return
        code = job.process.poll()
        if code is not None:
            missing = [
                name for name in DOWNLOADS if not (job.directory / name).is_file()
            ]
            if code == 0 and not missing:
                job.state = "completed"
            else:
                job.state = "failed"
                error = self._read_json(job.directory / "progress.json").get("error")
                if error:
                    job.error = str(error)
                elif code == 0:
                    job.error = "worker did not produce: " + ", ".join(missing)
                else:
                    job.error = f"worker exited with status {code}"
                    tail = self._read_log(job.directory / "worker.log").strip()
                    if tail:
                        job.error += ": " + tail.splitlines()[-1][-2000:]
            job.finished = time.monotonic()
            return
        if time.monotonic() - job.started >= job.timeout:
            job.state = "timed_out"
            job.error = "experiment exceeded its time limit"
            self._stop(job)
            job.finished = time.monotonic()

    @staticmethod
    def _stop(job: _Job) -> None:
        if job.process.poll() is not None:
            return
        job.process.terminate()
        try:
            job.process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            job.process.kill()
            job.process.wait(timeout=2)

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        try:
            if path.stat().st_size > 256 * 1024:
                return {}
            value = json.loads(path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, ValueError):
            return {}

    @staticmethod
    def _read_log(path: Path) -> str:
        try:
            with path.open("rb") as stream:
                stream.seek(max(0, path.stat().st_size - 8192))
                return stream.read(8192).decode("utf-8", errors="replace")
        except OSError:
            return ""

    def _snapshot(self, job: _Job) -> dict[str, Any]:
        return {
            "id": job.identifier,
            "state": job.state,
            "elapsed_seconds": round(
                (job.finished if job.finished is not None else time.monotonic())
                - job.started,
                2,
            ),
            "timeout_seconds": job.timeout,
            "progress": self._read_json(job.directory / "progress.json"),
            "error": job.error,
            "diagnostic": self._read_log(job.directory / "worker.log")
            if job.state == "failed"
            else None,
            "downloads": list(DOWNLOADS) if job.state == "completed" else [],
        }

    def status(self, identifier: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs[identifier]
            self._refresh(job)
            return self._snapshot(job)

    def statuses(self) -> list[dict[str, Any]]:
        with self._lock:
            self._refresh_all()
            return [self._snapshot(job) for job in self._jobs.values()]

    def cancel(self, identifier: str) -> dict[str, Any]:
        """Terminate only the worker represented by this identifier."""
        with self._lock:
            job = self._jobs[identifier]
            self._refresh(job)
            if job.state == "running":
                job.state = "cancelled"
                self._stop(job)
                job.finished = time.monotonic()
            return self._snapshot(job)

    def artifact(self, identifier: str, name: str) -> tuple[Path, str]:
        with self._lock:
            if name not in DOWNLOADS:
                raise KeyError(name)
            job = self._jobs[identifier]
            self._refresh(job)
            if job.state != "completed":
                raise ValueError("downloads require successful completion")
            path = job.directory / name
            if not path.is_file():
                raise KeyError(name)
            return path, DOWNLOADS[name]

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for job in self._jobs.values():
                if job.state == "running":
                    job.state = "cancelled"
                    self._stop(job)
                    job.finished = time.monotonic()
            self._temporary.cleanup()
