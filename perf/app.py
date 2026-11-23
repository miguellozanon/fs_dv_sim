#!/usr/bin/env python3
"""Performance Lab: the LR26 lap time simulation as a stand-alone web app (perf.bat starts it on Windows).

    python perf/app.py                 # http://localhost:8090, opens the browser
    python perf/app.py --port 8091 --no-browser

Needs Python 3.9+ with numpy, scipy and pyyaml. No ROS, no Isaac Sim, no WSL. Listens on this computer only
(127.0.0.1). Details: docs/PERF_APP.md.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from perflab import REPO  # noqa: E402
from perflab import energy, ident, logs, methods, ocp, studies, transient  # noqa: E402
from perflab.jobs import Runner  # noqa: E402
from perflab.lab import Lab, clean  # noqa: E402

WEB = HERE / "web"
BRAND = Path(REPO) / "ros2_ws" / "src" / "fsd_dashboard" / "fsd_dashboard" / "web"
DOCS = {"paper1_dv_simulator.pdf": Path(REPO) / "docs" / "paper" / "paper1_dv_simulator.pdf",
        "paper2_performance_lab.pdf": Path(REPO) / "docs" / "paper" / "paper2_performance_lab.pdf",
        "PERF_APP.md": Path(REPO) / "docs" / "PERF_APP.md",
        "PERF_METHODS.md": Path(REPO) / "docs" / "PERF_METHODS.md",
        "LOGGING_SPEC.md": Path(REPO) / "docs" / "LOGGING_SPEC.md"}      # the documents the guide links to (nothing else)
MAX_UPLOAD = 1 << 30          # 1 GB
VERSION = "1.0"

JOBS = {
    "sensitivity": lambda lab, job, a: studies.sensitivity(lab, a.get("params"), float(a.get("delta", 0.1)),
                                                           a.get("changes", []), a.get("competition"), job),
    "sweep": lambda lab, job, a: studies.sweep(lab, str(a["param"]), a["values"], a.get("changes", []), a.get("competition"),
                                               job, str(a.get("trace_event") or "")),
    "sweep2d": lambda lab, job, a: studies.sweep2d(lab, str(a["p1"]), a["v1"], str(a["p2"]), a["v2"], a.get("changes", []),
                                                   a.get("competition"), job),
    "compare": lambda lab, job, a: studies.compare(lab, a["setups"], a.get("competition"), job),
    "energy": lambda lab, job, a: energy.analyse(lab, a.get("changes", []), a.get("budget_kwh") or None,
                                                 a.get("caps_kw") or None, a.get("competition"), job),
    "transient": lambda lab, job, a: transient.run(lab, str(a["event"]), a.get("changes", []), a.get("competition"),
                                                   float(a.get("dt", 0.02)), float(a.get("track_width", 3.0)), job,
                                                   a.get("line_smoothing")),
    "regen": lambda lab, job, a: energy.regen_study(lab, a.get("changes", []), a.get("regen_kw"), a.get("budget_kwh") or None,
                                                   a.get("eta_regen"), a.get("competition"), job),
    "ocp": lambda lab, job, a: ocp.run(lab, str(a["event"]), str(a.get("model", "point_mass")), a.get("changes", []),
                                       a.get("competition"), float(a.get("track_width", 3.5)), a.get("ds") or None,
                                       bool(a.get("use_event_util", True)), bool(a.get("torque_vectoring", False)),
                                       bool(a.get("fixed_line", True)), int(a.get("max_iter", 3000)), job),
    "methods": lambda lab, job, a: methods.compare(lab, str(a["event"]), str(a.get("change") or ""), a.get("methods"),
                                                   a.get("changes", []), a.get("competition"), job),
    "ident": lambda lab, job, a: ident.run_tool(lab, str(a["tool"]), str(a["source"]), a.get("spec") or {}, a.get("args"),
                                                a.get("changes", []), a.get("competition"), job),
    "monte_carlo": lambda lab, job, a: methods.monte_carlo(lab, int(a.get("n", 200)), str(a.get("change") or ""),
                                                           a.get("uncertainty"), a.get("changes", []),
                                                           a.get("competition"), int(a.get("seed", 1)), job),
}


class App:
    def __init__(self, lab: Lab | None = None):
        self.lab = lab or Lab()
        self.runner = Runner()

    def submit(self, kind: str, args: dict):
        if kind not in JOBS:
            raise ValueError(f"unknown study {kind!r}")
        fn = JOBS[kind]
        job = self.runner.submit(kind, lambda job, **a: fn(self.lab, job, a), dict(args))
        return job.to_dict(False)


def make_handler(app: App, shutdown):
    class H(BaseHTTPRequestHandler):
        server_version = "perf-lab"

        def log_message(self, *a):
            pass

        def _send(self, code, body: bytes, mime: str, extra=None):
            self.send_response(code)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=HTTPStatus.OK):
            self._send(code, json.dumps(clean(obj), allow_nan=False).encode(), "application/json")

        def _err(self, msg, code=HTTPStatus.BAD_REQUEST):
            self._json({"ok": False, "message": msg}, code)

        def _file(self, root: Path, rel: str):
            p = (root / rel).resolve()
            if root.resolve() not in p.parents or not p.is_file():
                return self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")
            mime = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
            if p.suffix == ".js":
                mime = "text/javascript"
            elif p.suffix == ".md":
                mime = "text/plain"
            if mime.startswith("text/") or mime.endswith("json"):
                mime += "; charset=utf-8"
            self._send(HTTPStatus.OK, p.read_bytes(), mime)

        def do_GET(self):  # noqa: N802
            u = urlparse(self.path)
            path, q = unquote(u.path), parse_qs(u.query)
            comp = q.get("competition", [None])[0] or None
            try:
                if path in ("/", "/index.html"):
                    return self._file(WEB, "index.html")
                if path.startswith("/static/"):
                    return self._file(WEB, path[len("/static/"):])
                if path.startswith("/brand/"):
                    return self._file(BRAND, path[len("/brand/"):])
                if path.startswith("/docs/"):
                    name = path[len("/docs/"):]
                    if name in DOCS:
                        return self._file(DOCS[name].parent, DOCS[name].name)
                    return self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")
                if path == "/api/ping":
                    return self._json({"ok": True, "app": "perf-lab", "version": VERSION})
                if path == "/api/info":
                    return self._json(app.lab.info(comp))
                if path == "/api/methods":
                    ok, why = ocp.available()
                    return self._json({"methods": [{"name": k, "label": v[0], "family": v[1], "description": v[2]}
                                                   for k, v in methods.METHODS.items()],
                                       "default": list(methods.DEFAULT_METHODS), "uncertainty": methods.UNCERTAINTY,
                                       "guidance": methods.guidance(), "ocp": {"available": ok, "detail": why}})
                if path == "/api/logs":
                    return self._json({"logs": logs.sources(app.lab), "folder": os.path.relpath(logs.UPLOADS, REPO)})
                if path.startswith("/api/job/"):
                    j = app.runner.get(path.split("/")[3])
                    return self._json(j.to_dict()) if j else self._err("no such job", HTTPStatus.NOT_FOUND)
                if path == "/api/jobs":
                    return self._json([j.to_dict(False) for j in app.runner.jobs.values()])
            except Exception as e:  # noqa: BLE001
                return self._err(f"{type(e).__name__}: {e}")
            return self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")

        def do_POST(self):  # noqa: N802
            # only this app's page may post (it sets the header; another web page cannot without CORS)
            if self.headers.get("X-Perf-Lab") != "1":
                return self._err("missing header", HTTPStatus.FORBIDDEN)
            u = urlparse(self.path)
            path, q = u.path, parse_qs(u.query)
            n = int(self.headers.get("Content-Length") or 0)
            if path == "/api/logs/upload":
                if n <= 0 or n > MAX_UPLOAD:
                    return self._err("upload a file of up to 1 GB")
                name = q.get("name", [""])[0]
                try:
                    logs._safe_name(name)
                    data = self.rfile.read(n)
                    p = logs.save_upload(name, data)
                except ValueError as e:
                    return self._err(str(e))
                return self._json({"ok": True, "id": f"upload:{os.path.basename(p)}",
                                   "message": f"saved to {os.path.relpath(p, REPO)}"})
            try:
                req = json.loads(self.rfile.read(min(n, 1 << 22)) or b"{}")
                if not isinstance(req, dict):
                    raise ValueError
            except ValueError:
                return self._err("bad JSON")
            lab, comp = app.lab, req.get("competition") or None
            try:
                if path == "/api/run":
                    return self._json({"ok": True, "result": lab.run(req.get("changes", []), comp)})
                if path == "/api/validate":
                    return self._json({"ok": True, "changes": lab.validate(req.get("changes", []), comp)})
                if path == "/api/ggv":
                    return self._json({"ok": True, "result": energy.ggv_map(lab, req.get("changes", []), str(req.get("event") or ""), comp)})
                if path == "/api/setup/save":
                    where = lab.save_setup(str(req.get("name") or ""), req.get("changes", []), str(req.get("notes") or ""),
                                           bool(req.get("overwrite")))
                    return self._json({"ok": True, "message": f"saved to {where}", "setups": lab.setups()})
                if path == "/api/eta_map":
                    lab.set_eta_map(str(req.get("name") or ""))
                    return self._json({"ok": True, "eta_map": lab.eta_map})
                if path == "/api/job":
                    return self._json({"ok": True, "job": app.submit(str(req.get("kind")), req.get("args") or {})})
                if path == "/api/job/cancel":
                    return self._json({"ok": app.runner.cancel(str(req.get("id")))})
                if path == "/api/logs/inspect":
                    return self._json({"ok": True, "result": logs.inspect(lab, str(req.get("source")))})
                if path == "/api/logs/correlate":
                    return self._json({"ok": True, "result": logs.correlate(lab, str(req.get("source")), req.get("spec") or {},
                                                                            str(req.get("event")), req.get("changes", []), comp,
                                                                            req.get("shift"))})
                if path == "/api/logs/powertrain":
                    return self._json({"ok": True, "result": logs.powertrain(lab, str(req.get("source")), req.get("changes", []), comp,
                                                                             float(req.get("motors_per_axle", 2.0)))})
                if path == "/api/logs/save_map":
                    where = logs.save_eta_map(lab, str(req.get("source")), str(req.get("name") or ""), req.get("changes", []), comp,
                                              float(req.get("motors_per_axle", 2.0)), bool(req.get("overwrite")))
                    return self._json({"ok": True, "message": f"saved to {where}", "eta_maps": lab.eta_maps()})
                if path == "/api/logs/save_track":
                    c = req.get("closed")
                    where = ident.save_track(lab, str(req.get("source")), req.get("spec") or {}, str(req.get("name") or ""),
                                             None if c in (None, "", "auto") else bool(c), None, bool(req.get("overwrite")),
                                             req.get("changes", []), comp)
                    return self._json({"ok": True, "message": f"saved to {os.path.relpath(where, REPO)}"})
                if path == "/api/quit":
                    threading.Thread(target=shutdown, daemon=True).start()
                    return self._json({"ok": True, "message": "Performance Lab stopped; you can close this tab"})
            except (ValueError, KeyError, FileNotFoundError, FileExistsError) as e:
                return self._err(str(e.args[0]) if e.args else str(e))
            except Exception as e:  # noqa: BLE001
                return self._err(f"{type(e).__name__}: {e}", HTTPStatus.INTERNAL_SERVER_ERROR)
            return self._err("unknown action", HTTPStatus.NOT_FOUND)

    return H


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=int(os.environ.get("PERF_LAB_PORT", 8090)))
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args(argv)
    try:
        app = App()
    except FileNotFoundError as e:
        print(f"[perf] {e}")
        return 2
    try:
        httpd = ThreadingHTTPServer(("127.0.0.1", a.port), None)
    except OSError:
        print(f"[perf] port {a.port} is in use: Performance Lab may already be running (http://localhost:{a.port}),"
              f" or start it with --port <other>")
        if not a.no_browser:
            webbrowser.open(f"http://localhost:{a.port}")
        return 1
    httpd.daemon_threads = True
    httpd.RequestHandlerClass = make_handler(app, lambda: threading.Thread(target=httpd.shutdown, daemon=True).start())
    url = f"http://localhost:{a.port}"
    print(f"[perf] Performance Lab {VERSION} - {url}  (LapSim: {os.path.relpath(app.lab.lapsim_dir, REPO)})")
    print("[perf] close this window or press Quit in the page to stop")
    if not a.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    print("[perf] stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
