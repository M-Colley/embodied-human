"""Serve the local drag-and-drop module builder: python -m embodied_human.builder."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from . import catalog, preview_project, python_source, validate_project
from .jobs import JobManager

MAX_REQUEST_BYTES = 256 * 1024


class BuilderHandler(BaseHTTPRequestHandler):
    """Serve bundled assets and data; never expose arbitrary filesystem paths."""

    request_timeout = 15.0

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(self.request_timeout)

    def _reply(
        self, status: int, data: Any, content_type: str = "application/json"
    ) -> None:
        body = (
            json.dumps(data, allow_nan=False).encode("utf-8")
            if content_type == "application/json"
            else data
        )
        self.send_response(status)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "object-src 'none'; frame-ancestors 'none'; base-uri 'none'",
        )
        self.end_headers()
        self.wfile.write(body)

    def _local(self) -> bool:
        port = self.server.server_port  # type: ignore[attr-defined]
        hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        origin = self.headers.get("Origin")
        valid = self.headers.get("Host") in hosts and (
            origin is None or origin in {f"http://{host}" for host in hosts}
        )
        if not valid:
            self._reply(403, {"error": "Use the builder's localhost URL"})
        return valid

    def do_GET(self) -> None:
        if not self._local():
            return
        path = urlsplit(self.path).path
        if path == "/api/catalog":
            self._reply(200, catalog())
            return
        if path == "/api/experiments":
            from ..experiments import experiment_catalog

            self._reply(200, experiment_catalog())
            return
        parts = path.strip("/").split("/")
        if parts[:2] == ["api", "jobs"]:
            try:
                manager = self.server.jobs  # type: ignore[attr-defined]
                if len(parts) == 2:
                    self._reply(200, {"jobs": manager.statuses()})
                    return
                if len(parts) == 3:
                    self._reply(200, manager.status(parts[2]))
                    return
                if len(parts) == 5 and parts[3] == "download":
                    source, content_type = manager.artifact(parts[2], parts[4])
                    self._download(source, content_type)
                    return
            except KeyError:
                self._reply(404, {"error": "Unknown experiment or file"})
                return
            except ValueError as exc:
                self._reply(409, {"error": str(exc)})
                return
            self._reply(404, {"error": "Not found"})
            return
        assets = {
            "/": ("index.html", "text/html"),
            "/builder.js": ("builder.js", "text/javascript"),
            "/builder.css": ("builder.css", "text/css"),
        }
        if path not in assets:
            self._reply(404, {"error": "Not found"})
            return
        name, content_type = assets[path]
        self._reply(200, files(__package__).joinpath(name).read_bytes(), content_type)

    def _download(self, source: Path, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(source.stat().st_size))
        self.send_header("Content-Disposition", f'attachment; filename="{source.name}"')
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'none'; sandbox")
        self.end_headers()
        with source.open("rb") as stream:
            while chunk := stream.read(64 * 1024):
                self.wfile.write(chunk)

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_REQUEST_BYTES:
                raise ValueError("request must contain at most 256 KiB of JSON")
            # Drain bounded requests before replying, including rejected origins.
            # Closing a Windows socket with unread data can discard the response.
            body = self.rfile.read(length)
        except (ValueError, TimeoutError) as exc:
            self.close_connection = True
            self._reply(400, {"error": str(exc)})
            return
        if not self._local():
            return
        routes = {
            "/api/validate": validate_project,
            "/api/preview": preview_project,
            "/api/python": python_source,
        }
        path = urlsplit(self.path).path
        try:
            if self.headers.get_content_type() != "application/json":
                raise ValueError("Content-Type must be application/json")
            data = json.loads(body)
            if path == "/api/jobs":
                from ..experiments import validate_request

                if (
                    not isinstance(data, dict)
                    or set(data) - {"experiment", "timeout_seconds"}
                    or "experiment" not in data
                ):
                    raise ValueError(
                        "job needs experiment and optional timeout_seconds"
                    )
                spec = validate_request(data["experiment"])
                manager = self.server.jobs  # type: ignore[attr-defined]
                self._reply(202, manager.start(spec, data.get("timeout_seconds", 900)))
                return
            parts = path.strip("/").split("/")
            if (
                len(parts) == 4
                and parts[:2] == ["api", "jobs"]
                and parts[3] == "cancel"
            ):
                if data != {}:
                    raise ValueError("cancel request must be an empty object")
                self._reply(
                    200,
                    self.server.jobs.cancel(parts[2]),  # type: ignore[attr-defined]
                )
                return
            action = routes.get(path)
            if action is None:
                self._reply(404, {"error": "Not found"})
                return
            result = action(data)
            self._reply(200, {"python": result} if isinstance(result, str) else result)
        except (ValueError, TypeError, OverflowError, ImportError) as exc:
            self._reply(400, {"error": str(exc)})
        except KeyError:
            self._reply(404, {"error": "Unknown experiment"})

    def log_message(self, format: str, *args: Any) -> None:
        # No per-request logs; scientific results are shown explicitly in the UI.
        pass


class BuilderServer(ThreadingHTTPServer):
    """Close all workers and temporary outputs when the local server closes."""

    def __init__(self, port: int) -> None:
        self.jobs = JobManager()
        try:
            super().__init__(("127.0.0.1", port), BuilderHandler)
        except BaseException:
            self.jobs.close()
            raise

    def server_close(self) -> None:
        self.jobs.close()
        super().server_close()


def make_server(port: int = 8765) -> BuilderServer:
    """Create a loopback-only server; port 0 is useful for automated checks."""
    return BuilderServer(port)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--run", type=Path, help="run an exported JSON project preview")
    args = parser.parse_args(argv)
    if args.run:
        try:
            result = preview_project(json.loads(args.run.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError, ImportError) as exc:
            parser.error(str(exc))
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    if not 0 <= args.port <= 65535:
        parser.error("port must be between 0 and 65535")
    with make_server(args.port) as server:
        print(f"Local builder: http://127.0.0.1:{server.server_port}", flush=True)
        print("Press Ctrl+C to stop.", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
