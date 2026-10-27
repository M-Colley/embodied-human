"""Local experiments run in bounded processes and export fixed artifacts."""

import http.client
import json
import sys
import threading
import time
import zipfile

import pytest

from embodied_human import HumanConfig
from embodied_human.builder.__main__ import make_server
from embodied_human.builder.jobs import JobManager


def wait_for_terminal(manager, identifier):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        result = manager.status(identifier)
        if result["state"] != "running":
            return result
        time.sleep(0.02)
    raise AssertionError("worker failed to finish within the test deadline")


@pytest.fixture
def worker(monkeypatch):
    def command(request, directory):
        return [
            sys.executable,
            "-c",
            "import json,pathlib,sys,time,zipfile; "
            "p=pathlib.Path(sys.argv[1]); "
            "spec=json.loads((p/'request.json').read_text()); "
            "(p/'progress.json').write_text(json.dumps({'completed':0,'total':1})); "
            "time.sleep(spec.get('delay',0)); "
            "(p/'report.json').write_text(json.dumps({'schema_version':1})); "
            "(p/'manifest.json').write_text(json.dumps({'seed':spec.get('seed')})); "
            "(p/'results.csv').write_text('metric,value\\nexample,1\\n'); "
            "(p/'plot.svg').write_text('<svg xmlns=\"http://www.w3.org/2000/svg\"/>'); "
            "z=zipfile.ZipFile(p/'bundle.zip','w'); "
            "[z.write(p/n,n) for n in ('request.json','report.json','manifest.json',"
            "'results.csv','plot.svg')]; z.close()",
            str(directory),
        ]

    monkeypatch.setattr("embodied_human.builder.jobs.worker_command", command)


def test_job_runs_process_and_exports_bounded_artifacts(worker):
    manager = JobManager()
    try:
        job = manager.start({"seed": 17})
        assert wait_for_terminal(manager, job["id"])["state"] == "completed"
        elapsed = manager.status(job["id"])["elapsed_seconds"]
        time.sleep(0.04)
        assert manager.status(job["id"])["elapsed_seconds"] == elapsed
        path, mime = manager.artifact(job["id"], "manifest.json")
        assert mime == "application/json"
        assert json.loads(path.read_text()) == {"seed": 17}
        path, mime = manager.artifact(job["id"], "bundle.zip")
        assert mime == "application/zip"
        with zipfile.ZipFile(path) as archive:
            assert set(archive.namelist()) == {
                "request.json",
                "report.json",
                "manifest.json",
                "results.csv",
                "plot.svg",
            }
        with pytest.raises(KeyError):
            manager.artifact(job["id"], "../request.json")
    finally:
        manager.close()


def test_cancel_releases_concurrency_and_close_terminates_owned_workers(worker):
    manager = JobManager()
    first = manager.start({"delay": 60})
    second = manager.start({"delay": 60})
    with pytest.raises(ValueError, match="two experiments"):
        manager.start({"delay": 60})
    assert manager.cancel(first["id"])["state"] == "cancelled"
    third = manager.start({"delay": 60})
    assert manager.status(second["id"])["state"] == "running"
    processes = [job.process for job in manager._jobs.values()]
    manager.close()
    assert all(process.poll() is not None for process in processes)
    assert manager.status(third["id"])["state"] == "cancelled"


def test_job_timeout_is_enforced_without_browser_polling(worker):
    manager = JobManager()
    try:
        job = manager.start({"delay": 60}, timeout=1)
        # The manager's monitor terminates the worker even with no HTTP polls.
        process = manager._jobs[job["id"]].process
        process.wait(timeout=5)
        assert manager.status(job["id"])["state"] == "timed_out"
        with pytest.raises(ValueError, match="completion"):
            manager.artifact(job["id"], "report.json")
    finally:
        manager.close()


@pytest.mark.parametrize("timeout", [0, -1, True, float("inf"), 3601, "10"])
def test_job_rejects_invalid_time_limits(timeout):
    manager = JobManager()
    try:
        with pytest.raises(ValueError, match="timeout_seconds"):
            manager.start({}, timeout)
    finally:
        manager.close()


def test_failed_worker_is_reported(monkeypatch):
    monkeypatch.setattr(
        "embodied_human.builder.jobs.worker_command",
        lambda request, directory: [sys.executable, "-c", "raise RuntimeError('test')"],
    )
    manager = JobManager()
    try:
        job = manager.start({})
        snapshot = wait_for_terminal(manager, job["id"])
        assert snapshot["state"] == "failed"
        assert "status" in snapshot["error"]
        assert "RuntimeError: test" in snapshot["diagnostic"]
    finally:
        manager.close()


@pytest.fixture
def server():
    server = make_server(0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def request(server, method, path, value=None, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    actual_headers = {"Content-Type": "application/json", **(headers or {})}
    body = json.dumps(value) if value is not None else None
    connection.request(method, path, body=body, headers=actual_headers)
    response = connection.getresponse()
    result = (response.status, response.read(), dict(response.getheaders()))
    connection.close()
    return result


def experiment():
    return {
        "version": 1,
        "config": HumanConfig().as_dict(),
        "benchmarks": ["memory.span"],
        "seeds": [0, 1],
        "compare_baseline": True,
        "sweep": None,
    }


def test_http_runs_real_experiment_and_downloads_results(server):
    status, data, _ = request(server, "GET", "/api/experiments")
    assert status == 200
    assert "benchmarks" in json.loads(data)
    status, body, _ = request(server, "POST", "/api/jobs", {"experiment": experiment()})
    assert status == 202
    identifier = json.loads(body)["id"]
    snapshot = wait_for_terminal(server.jobs, identifier)
    assert snapshot["state"] == "completed", json.dumps(snapshot, indent=2)
    listed = json.loads(request(server, "GET", "/api/jobs")[1])["jobs"]
    assert listed[0]["id"] == identifier
    assert listed[0]["state"] == "completed"
    status, body, headers = request(
        server, "GET", f"/api/jobs/{identifier}/download/report.json"
    )
    assert status == 200
    assert "attachment" in headers["Content-Disposition"]
    report = json.loads(body)
    assert len(report["rows"]) == 4
    assert report["paired_differences"]
    assert (
        request(server, "GET", f"/api/jobs/{identifier}/download/worker.log")[0] == 404
    )


def test_http_jobs_reject_invalid_and_foreign_requests(server):
    assert request(server, "POST", "/api/jobs", {"experiment": {}})[0] == 400
    assert (
        request(
            server,
            "POST",
            "/api/jobs",
            {"experiment": experiment()},
            headers={"Origin": "https://example.org"},
        )[0]
        == 403
    )
    assert request(server, "POST", "/api/jobs/unknown/cancel", {})[0] == 404
    assert request(server, "GET", "/api/jobs/unknown")[0] == 404


def test_http_cancel_stops_the_requested_job(server, monkeypatch):
    monkeypatch.setattr(
        "embodied_human.builder.jobs.worker_command",
        lambda request, directory: [
            sys.executable,
            "-c",
            "import time; time.sleep(60)",
        ],
    )
    status, body, _ = request(server, "POST", "/api/jobs", {"experiment": experiment()})
    assert status == 202, body
    identifier = json.loads(body)["id"]
    status, body, _ = request(server, "POST", f"/api/jobs/{identifier}/cancel", {})
    assert status == 200 and json.loads(body)["state"] == "cancelled"
    assert (
        request(server, "GET", f"/api/jobs/{identifier}/download/report.json")[0] == 409
    )


def test_incomplete_successful_worker_is_not_advertised_as_completed(monkeypatch):
    monkeypatch.setattr(
        "embodied_human.builder.jobs.worker_command",
        lambda request, directory: [
            sys.executable,
            "-c",
            "import pathlib,sys; pathlib.Path(sys.argv[1]).write_text('{}')",
            str(directory / "report.json"),
        ],
    )
    manager = JobManager()
    try:
        snapshot = wait_for_terminal(manager, manager.start({})["id"])
        assert snapshot["state"] == "failed"
        assert "manifest.json" in snapshot["error"]
        assert snapshot["downloads"] == []
    finally:
        manager.close()


def test_incomplete_http_body_has_a_read_deadline(server, monkeypatch):
    monkeypatch.setattr(
        "embodied_human.builder.__main__.BuilderHandler.request_timeout", 0.1
    )
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
    connection.request(
        "POST",
        "/api/validate",
        body="{}",
        headers={"Content-Type": "application/json", "Content-Length": "10"},
    )
    response = connection.getresponse()
    assert response.status == 400
    assert "timed out" in json.loads(response.read())["error"]
    connection.close()
