"""HTTP server for the dashboard (standard library only).

    GET  /                  the page (web/index.html, app.js, style.css)
    GET  /events            Server-Sent Events: a JSON delta every 100 ms
    GET  /api/state         full state as JSON (for scripts)
    GET  /api/image/<name>  latest camera frame ("left" or "depth")
    POST /api/command       {"cmd": "select_mission"|"activate"|"go"|"emergency"|"reset", "mission": "..."}
    GET  /api/setup         Vehicle setup: parameters (Table 3.1), configurations, LapSim / MATLAB / sim results
    POST /api/setup         {"action": "apply"|"save"|"delete"|"reload"|"cancel_matlab"|"validate"|"open",
                             "changes": ["m=270", "mu*1.05", ...], "name": "...", "notes": "..."}

Open http://localhost:8080 in a browser on Windows: with WSL, ports opened in
Ubuntu are reachable from Windows as localhost.
"""

from __future__ import annotations

import json
import mimetypes
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .model import ClientCursor, DashboardModel

WEB_DIR = Path(__file__).resolve().parent / "web"
mimetypes.add_type("font/woff2", ".woff2")       # older Pythons / Windows registries lack it
PUSH_PERIOD = 0.1
COMMANDS = {"select_mission", "activate", "go", "emergency", "reset"}


def _local(ip: str) -> bool:
    """A browser on this PC (WSL forwards Windows' localhost as a loopback connection)."""
    ip = (ip or "").removeprefix("::ffff:")
    return ip == "::1" or ip.startswith("127.")


def _json_default(o):
    try:
        return float(o)
    except Exception:  # noqa: BLE001
        return str(o)


class _Handler(BaseHTTPRequestHandler):
    server_version = "fsd-dashboard"
    model: DashboardModel = None  # set by make_server
    stop_event: threading.Event = None

    def log_message(self, fmt, *args):  # quiet: the page polls images often
        pass

    # ------------------------------------------------------------------ #
    def do_GET(self):  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            return self._file(WEB_DIR / "index.html")
        if path == "/events":
            return self._events()
        if path == "/api/state":
            return self._json(self.model.snapshot())
        if path == "/api/setup":
            st = self.model.setup_state()
            return self._json(st if st is not None else {"available": False}, HTTPStatus.OK)
        if path.startswith("/api/image/"):
            img = self.model.image(path.rsplit("/", 1)[-1])
            if img is None:
                return self._send(HTTPStatus.NOT_FOUND, b"no image yet", "text/plain")
            return self._send(HTTPStatus.OK, img["data"], img["mime"])
        name = path.lstrip("/")
        if name and "/" not in name and not name.startswith("."):
            return self._file(WEB_DIR / name)
        return self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")

    def do_POST(self):  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path not in ("/api/command", "/api/setup"):
            return self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")
        try:
            n = int(self.headers.get("Content-Length", "0"))
            if n > 1_000_000:
                raise ValueError("request too large")
            cmd = json.loads(self.rfile.read(n) or b"{}")
        except (ValueError, json.JSONDecodeError):
            return self._json({"ok": False, "message": "invalid JSON"}, HTTPStatus.BAD_REQUEST)
        if path == "/api/setup":
            if not isinstance(cmd, dict):
                return self._json({"ok": False, "message": "expected a JSON object"}, HTTPStatus.BAD_REQUEST)
            if cmd.get("action") == "open":
                # opens a program on the PC: only for the page itself, in a browser on this PC
                if self.headers.get("X-FSG-Dashboard") != "1":
                    return self._json({"ok": False, "message": "missing dashboard header"}, HTTPStatus.FORBIDDEN)
                if not _local(self.client_address[0]):
                    return self._json({"ok": False, "message": "files open only from the PC that runs the simulator"},
                                      HTTPStatus.FORBIDDEN)
            ok, msg = self.model.setup_handle(cmd)
            return self._json({"ok": bool(ok), "message": str(msg)})
        if not isinstance(cmd, dict) or cmd.get("cmd") not in COMMANDS:
            return self._json({"ok": False, "message": f"unknown command; use one of {sorted(COMMANDS)}"},
                              HTTPStatus.BAD_REQUEST)
        ok, msg = self.model.handle_command(cmd)
        return self._json({"ok": bool(ok), "message": str(msg)})

    # ------------------------------------------------------------------ #
    def _events(self):
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        cur = ClientCursor()
        try:
            while not self.stop_event.is_set():
                t0 = time.monotonic()
                payload = json.dumps(self.model.delta(cur), separators=(",", ":"), default=_json_default)
                self.wfile.write(b"data: " + payload.encode() + b"\n\n")
                self.wfile.flush()
                time.sleep(max(0.0, PUSH_PERIOD - (time.monotonic() - t0)))
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            return

    def _file(self, p: Path):
        try:
            data = p.read_bytes()
        except OSError:
            return self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")
        mime = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        if mime.startswith("text/") or mime.endswith("javascript"):
            mime += "; charset=utf-8"
        return self._send(HTTPStatus.OK, data, mime)

    def _json(self, obj, status=HTTPStatus.OK):
        return self._send(status, json.dumps(obj, default=_json_default).encode(), "application/json")

    def _send(self, status, body: bytes, mime: str):
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass


class DashboardServer:
    def __init__(self, model: DashboardModel, host: str = "0.0.0.0", port: int = 8080):
        self.stop_event = threading.Event()
        handler = type("Handler", (_Handler,), {"model": model, "stop_event": self.stop_event})
        ThreadingHTTPServer.allow_reuse_address = True
        self.httpd = ThreadingHTTPServer((host, port), handler)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        self._thread: threading.Thread | None = None

    def start(self) -> "DashboardServer":
        self._thread = threading.Thread(target=self.httpd.serve_forever, name="fsd-dashboard-http", daemon=True)
        self._thread.start()
        return self

    def stop(self, timeout: float = 2.0) -> None:
        """Stop serving; never waits longer than `timeout` (a second Ctrl+C from launch interrupted an endless
        httpd.shutdown() and printed a traceback, 11 Nov 2026)."""
        self.stop_event.set()
        t = threading.Thread(target=self.httpd.shutdown, name="fsd-dashboard-stop", daemon=True)
        t.start()
        t.join(timeout)
        try:
            self.httpd.server_close()
        except OSError:
            pass
