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

MAX_REQUEST_BYTES = 256 * 1024


class BuilderHandler(BaseHTTPRequestHandler):
    """Serve bundled assets and data; never expose arbitrary filesystem paths."""

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

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_REQUEST_BYTES:
                raise ValueError("request must contain at most 256 KiB of JSON")
            # Drain bounded requests before replying, including rejected origins.
            # Closing a Windows socket with unread data can discard the response.
            body = self.rfile.read(length)
        except ValueError as exc:
            self._reply(400, {"error": str(exc)})
            return
        if not self._local():
            return
        routes = {
            "/api/validate": validate_project,
            "/api/preview": preview_project,
            "/api/python": python_source,
        }
        action = routes.get(urlsplit(self.path).path)
        if action is None:
            self._reply(404, {"error": "Not found"})
            return
        try:
            if self.headers.get_content_type() != "application/json":
                raise ValueError("Content-Type must be application/json")
            data = json.loads(body)
            result = action(data)
            self._reply(200, {"python": result} if isinstance(result, str) else result)
        except (ValueError, TypeError, OverflowError, ImportError) as exc:
            self._reply(400, {"error": str(exc)})

    def log_message(self, format: str, *args: Any) -> None:
        # No per-request logs; scientific results are shown explicitly in the UI.
        pass


def make_server(port: int = 8765) -> ThreadingHTTPServer:
    """Create a loopback-only server; port 0 is useful for automated checks."""
    return ThreadingHTTPServer(("127.0.0.1", port), BuilderHandler)


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
