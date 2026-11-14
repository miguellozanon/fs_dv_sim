#!/usr/bin/env python3
"""FSG DV simulator launcher: one web page to start and stop everything, no terminals.

Runs in WSL (started by run.bat on Windows) and serves http://localhost:8070:

  * Simulation   Isaac Sim (Windows) + ROS 2 relay, the lite backend, or the offline demo; mission,
                 track seed, vehicle setup, reference or RL driver, RViz. It starts the Zenoh router,
                 the ROS 2 launch file and Isaac Sim in the right order and opens the dashboard.
  * RL / Performance tools   universal policy training (CPU or GPU), per-setup training,
                 official evaluation, setup sweeps, parameter studies - as background jobs with logs
                 and links to their reports.
  * Maintenance  build the workspace, install a new version from the Downloads zip, run the tests.

    python3 tools/launcher/launcher.py              # http://localhost:8070
    python3 tools/launcher/launcher.py --port 8071 --no-browser

Only 127.0.0.1 is served, and every action needs the page's own request header, so other web
pages cannot trigger commands. Commands are built here from validated fields, never from text.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import mimetypes
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

REPO = Path(__file__).resolve().parents[2]
WEB = Path(__file__).resolve().parent / "web"
BRAND = REPO / "ros2_ws" / "src" / "fsd_dashboard" / "fsd_dashboard" / "web"
ENV_SH = REPO / "tools" / "wsl" / "fsgsim_env.sh"
RESULTS = REPO / "results"
PY = sys.executable or "python3"
STATE_DIR = Path(os.environ.get("FSG_LAUNCHER_DIR", Path.home() / ".cache" / "fsg-dv-sim" / "launcher"))
SETTINGS_FILE = Path.home() / ".config" / "fsg-dv-sim" / "launcher.json"

MISSIONS = ["acceleration", "skidpad", "autocross", "trackdrive", "ebs_test"]   # no track for DV inspection (car on stands)
EVENTS = ["acceleration", "skidpad", "autocross", "trackdrive"]
ROUTER_PORT, RELAY_PORT = 7447, 47800
CHANGE_RE = re.compile(r"^[A-Za-z_][\w.]*\s*[=*+-]\s*[-+0-9.eE*/() ]+$")
NAME_RE = re.compile(r"^[\w.+-]{1,60}$")
mimetypes.add_type("font/woff2", ".woff2")

DEFAULT_SETTINGS = {
    "isaac_path": "C:\\isaacsim",          # folder with python.bat (standalone Isaac Sim)
    "isaac_python": "",                    # or a python.exe with isaacsim installed (Isaac Lab venv)
    "dashboard_port": 8080,
    "update_zip": "",                      # default: <Windows Downloads>\\fsg-dv-sim.zip
    "win_python": "",                      # Windows python.exe with PyTorch for GPU jobs (default: found on PATH)
    "open_dashboard": True,
    "git_github": "",                      # GitHub / GitLab push (Maintenance): repository URLs, branch, author
    "git_gitlab": "",
    "git_branch": "main",
    "git_name": "",
    "git_email": "",
}
EMAIL_RE = re.compile(r"^[^\s@\"'`$;&|<>]+@[^\s@\"'`$;&|<>]+$")
GIT_NAME_RE = re.compile(r"^[^\"'`$;&|<>\n]{1,80}$")


def _git_push_module():
    """tools/git_push.py (URL / branch rules shared with the push tool)."""
    if str(REPO / "tools") not in sys.path:
        sys.path.insert(0, str(REPO / "tools"))
    import git_push  # noqa: PLC0415

    return git_push


def colcon_base() -> Path:
    """Where colcon's build/ install/ log/ are (same rule as tools/wsl/fsgsim_env.sh and fsgbuild.sh)."""
    ws = Path(os.environ.get("FSG_WS") or REPO / "ros2_ws")
    if os.environ.get("FSG_COLCON_BASE"):
        return Path(os.environ["FSG_COLCON_BASE"])
    return Path.home() / ".fsg-dv-sim" / "colcon" if str(ws).startswith("/mnt/") else ws


def build_signature() -> str:
    """Changes that need a colcon build: files added / removed in the workspace sources, and edits to
    setup.py, setup.cfg, package.xml, CMakeLists.txt, .msg / .srv (Python, launch and config edits are
    live with --symlink-install)."""
    import hashlib  # noqa: PLC0415

    h = hashlib.sha1()
    src = REPO / "ros2_ws" / "src"
    for root, dirs, files in os.walk(src):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__" and not d.endswith(".egg-info"))
        for f in sorted(files):
            if f.endswith((".pyc", ".pyo")):
                continue
            rel = os.path.relpath(os.path.join(root, f), src)
            h.update(rel.encode())
            if f in ("setup.py", "setup.cfg", "package.xml", "CMakeLists.txt") or f.endswith((".msg", ".srv")):
                try:
                    h.update(Path(root, f).read_bytes())
                except OSError:
                    pass
    return h.hexdigest()


def stamp_file() -> Path:
    return colcon_base() / ".fsg_build_stamp"


def needs_build() -> bool:
    if not (colcon_base() / "install" / "setup.bash").exists():
        return True
    try:
        return stamp_file().read_text(encoding="utf-8").strip() != build_signature()
    except OSError:
        return True


def write_build_stamp() -> None:
    stamp_file().parent.mkdir(parents=True, exist_ok=True)
    stamp_file().write_text(build_signature(), encoding="utf-8")


def version() -> str:
    try:
        txt = (REPO / "ros2_ws" / "src" / "fsdsim" / "fsdsim" / "__init__.py").read_text(encoding="utf-8")
        return re.search(r'__version__\s*=\s*"([^"]+)"', txt).group(1)
    except (OSError, AttributeError):
        return "?"


# ---------------------------------------------------------------------------- #
# small helpers
# ---------------------------------------------------------------------------- #
def encode_start_setup(name: str, changes) -> str:
    """The setup for the dashboard (fsd_dashboard.vehicle_setup.decode_start_setup): URL-safe base64 JSON."""
    import base64  # noqa: PLC0415

    raw = json.dumps({"name": name, "changes": list(changes)}, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def gpu_query() -> dict:
    """NVIDIA GPU name, memory in use / total (MB, every program on the GPU) and load, from nvidia-smi
    (WSL ships it in /usr/lib/wsl/lib). {} without an NVIDIA GPU."""
    exe = shutil.which("nvidia-smi") or ("/usr/lib/wsl/lib/nvidia-smi" if os.path.exists("/usr/lib/wsl/lib/nvidia-smi") else "")
    if not exe:
        return {}
    try:
        r = subprocess.run([exe, "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
        name, used, total, util = (x.strip() for x in r.stdout.strip().splitlines()[0].rsplit(",", 3))
        return {"name": name, "used_mb": int(float(used)), "total_mb": int(float(total)),
                "util": int(float(util)) if util.replace(".", "").isdigit() else None}
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return {}


def port_listening(port: int) -> bool:
    """A local TCP socket in LISTEN state on this port (from /proc, no connection attempt)."""
    hexport = f":{port:04X}"
    for f in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            with open(f, encoding="ascii") as fh:
                next(fh, None)
                for line in fh:
                    parts = line.split()
                    if len(parts) > 3 and parts[1].endswith(hexport) and parts[3] == "0A":
                        return True
        except OSError:
            continue
    return False


PERCEPTION = ("none", "yolo", "lidar", "fusion", "camera")
FAULT_RE = re.compile(r"^[a-z_]+:[a-z_]+(=[-+0-9.eE]+)?@[0-9.]+(\+[0-9.]+)?(,[a-z_]+:[a-z_]+(=[-+0-9.eE]+)?@[0-9.]+(\+[0-9.]+)?)*$")


def isaac_camera_hz(o: dict) -> int | None:
    """Camera rate for Isaac Sim (run_isaac.py --camera-hz), None for the sensors.yaml rate (recording cones).

    The Windows -> WSL link carries only ~50-80 MB/s; colour + depth at 15 Hz is ~70 MB/s and filled it (the car's
    data arrived up to 2 s late, 11 Nov 2026). YOLO manages ~7 images/s on the CPU anyway: 10 Hz when the camera is
    used for cones, 5 Hz (the dashboard picture) otherwise."""
    if o.get("record_cones"):
        return None
    return 10 if o.get("perception", "none") in ("yolo", "fusion", "camera") else 5


def sim_changes(o: dict) -> list[str]:
    """Vehicle changes sent to the simulator: the setup's changes plus the handling model of this run."""
    extra = ["chassis.handling=double_track"] if o.get("handling") == "double_track" else []
    return list(o.get("changes") or []) + extra


def clean_changes(raw) -> list[str]:
    """Vehicle changes as a list ("m=270 c_z*1.05" or a list); raises ValueError on anything else."""
    if isinstance(raw, str):
        raw = re.split(r"[\s,;]+", raw.strip())
    out = []
    for c in raw or []:
        c = str(c).strip()
        if not c:
            continue
        if not CHANGE_RE.match(c) or len(c) > 60:
            raise ValueError(f"vehicle change '{c}': use name=value, name*factor, name+delta or name-delta")
        out.append(c.replace(" ", ""))
    return out


def win_path(p: Path | str) -> str:
    """Windows path of a WSL file (\\\\wsl.localhost\\...)."""
    try:
        return subprocess.run(["wslpath", "-w", str(p)], capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def wsl_path(p: str) -> str:
    try:
        return subprocess.run(["wslpath", "-u", p], capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def windows_downloads() -> str:
    """WSL path of the Windows user's Downloads folder ('' outside WSL)."""
    try:
        prof = subprocess.run(["cmd.exe", "/c", "echo", "%USERPROFILE%"], capture_output=True, text=True,
                              timeout=8, cwd="/mnt/c" if os.path.isdir("/mnt/c") else None).stdout.strip()
        return os.path.join(wsl_path(prof), "Downloads") if prof and "%" not in prof else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def ps_quote(s: str) -> str:
    return "'" + str(s).replace("'", "''") + "'"


def ros_argv(*cmd: str) -> list[str]:
    """Run a ROS 2 command with the fsgsim environment (Zenoh RMW, domain, workspace)."""
    return ["bash", "-c", 'source "$0" > /dev/null || exit 1; exec "$@"', str(ENV_SH), *cmd]


def py_env() -> dict:
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    src = REPO / "ros2_ws" / "src"
    env["PYTHONPATH"] = os.pathsep.join([str(src / "fsdsim"), str(src / "fsd_dashboard"), env.get("PYTHONPATH", "")])
    return env


# ---------------------------------------------------------------------------- #
# processes
# ---------------------------------------------------------------------------- #
class Proc:
    """A child process with its log file; `windows` processes are stopped with taskkill.exe."""

    def __init__(self, key: str, title: str, argv: list, kind: str = "service", cwd: Path = REPO, env=None,
                 windows: bool = False, links=None, meta=None, on_exit=None):
        self.key, self.title, self.argv, self.kind = key, title, list(map(str, argv)), kind
        self.cwd, self.env, self.windows = cwd, env, windows
        self.links = links or []
        self.meta = meta or {}
        self.on_exit = on_exit
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        self.log_path = STATE_DIR / f"{key}.log"
        self.proc: subprocess.Popen | None = None
        self.state, self.rc = "new", None
        self.started = self.ended = None
        self.stopping = False

    def start(self) -> None:
        self.log = open(self.log_path, "w", encoding="utf-8", errors="replace")  # noqa: SIM115
        self.log.write(f"$ {' '.join(self.argv)}\n")
        self.log.flush()
        self.proc = subprocess.Popen(self.argv, stdout=self.log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                     cwd=str(self.cwd), env=self.env, start_new_session=True)
        self.state, self.started = "running", time.time()
        threading.Thread(target=self._wait, daemon=True).start()

    def _wait(self) -> None:
        rc = self.proc.wait()
        self.rc, self.ended = rc, time.time()
        self.state = "stopped" if self.stopping else ("done" if rc == 0 else "failed")
        try:
            self.log.write(f"\n[launcher] exited with code {rc}\n")
            self.log.close()
        except (OSError, ValueError):
            pass
        if self.on_exit:
            try:
                self.on_exit(self)
            except Exception:  # noqa: BLE001
                pass

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def winpid(self) -> str | None:
        try:
            m = re.search(r"\[fsd-launcher\] winpid (\d+)", self.log_path.read_text(encoding="utf-8", errors="replace"))
            return m.group(1) if m else None
        except OSError:
            return None

    def stop(self, timeout: float = 10.0) -> None:
        if not self.alive:
            return
        self.stopping = True
        if self.windows:
            pid = self.winpid()
            if pid:
                subprocess.run(["taskkill.exe", "/T", "/F", "/PID", pid], capture_output=True, timeout=20)
        try:
            pg = os.getpgid(self.proc.pid)
            for sig, wait in ((signal.SIGINT, timeout), (signal.SIGTERM, 4.0), (signal.SIGKILL, 2.0)):
                os.killpg(pg, sig)
                t0 = time.time()
                while self.alive and time.time() - t0 < wait:
                    time.sleep(0.1)
                if not self.alive:
                    break
        except (ProcessLookupError, PermissionError):
            pass

    def tail(self, n: int = 200) -> str:
        try:
            with open(self.log_path, encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            return "".join(lines[-n:])
        except OSError:
            return ""

    def last_line(self) -> str:
        for line in reversed(self.tail(30).splitlines()):
            line = line.strip()
            if line and not line.startswith("[launcher] exited"):
                return line[-220:]
        return ""

    def info(self) -> dict:
        links = [lk for lk in self.links if (REPO / lk["path"]).exists()] if self.state in ("done", "failed", "stopped") \
            else [lk for lk in self.links if lk.get("live") and (REPO / lk["path"]).exists()]
        return {"key": self.key, "title": self.title, "kind": self.kind, "state": self.state, "rc": self.rc,
                "started": self.started, "ended": self.ended, "last": self.last_line(), "meta": self.meta,
                "links": [{"label": lk["label"], "href": "/" + lk["path"]} for lk in links]}


# ---------------------------------------------------------------------------- #
class Launcher:
    def __init__(self, open_browser_cmd=None, probe: bool = True):
        self.lock = threading.RLock()
        self.procs: dict[str, Proc] = {}
        self.session: dict = {"state": "idle", "steps": [], "opts": {}}
        self.settings = self._load_settings()
        self._status_cache: dict = {}
        self._status_t = 0.0
        self._gpu_t, self._gpu = 0.0, {}
        self._slow: dict = {"torch": None, "downloads": None}
        self._job_n = 0
        if probe:                                   # PyTorch in WSL, Windows Downloads, Windows Python (slow)
            threading.Thread(target=self._probe_slow, daemon=True).start()

    # ------------------------------------------------------------------ settings
    def _load_settings(self) -> dict:
        s = dict(DEFAULT_SETTINGS)
        try:
            s.update(json.loads(SETTINGS_FILE.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            pass
        return s

    def save_settings(self, req: dict) -> tuple[bool, str]:
        s = dict(self.settings)
        for k in ("isaac_path", "isaac_python", "update_zip", "win_python"):
            if k in req:
                v = str(req[k]).strip()
                if any(ch in v for ch in "\"'`$;&|<>\n"):
                    return False, f"{k}: invalid characters"
                s[k] = v
        if "dashboard_port" in req:
            p = int(req["dashboard_port"])
            if not 1024 <= p <= 65535:
                return False, "dashboard port: 1024 ... 65535"
            s["dashboard_port"] = p
        if "open_dashboard" in req:
            s["open_dashboard"] = bool(req["open_dashboard"])
        gp = _git_push_module()
        for k, label in (("git_github", "GitHub URL"), ("git_gitlab", "GitLab URL")):
            if k in req:
                v = str(req[k]).strip()
                if v and not gp.URL_RE.match(v):
                    return False, f"{label} not understood (https://... or git@host:group/repo.git)"
                s[k] = v
        if "git_branch" in req:
            v = str(req["git_branch"]).strip() or "main"
            if not gp.BRANCH_RE.match(v) or ".." in v:
                return False, "branch: letters, digits, - _ . /"
            s["git_branch"] = v
        if "git_name" in req:
            v = str(req["git_name"]).strip()
            if v and not GIT_NAME_RE.match(v):
                return False, "author name: invalid characters"
            s["git_name"] = v
        if "git_email" in req:
            v = str(req["git_email"]).strip()
            if v and not EMAIL_RE.match(v):
                return False, "author e-mail not understood"
            s["git_email"] = v
        self.settings = s
        SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        SETTINGS_FILE.write_text(json.dumps(s, indent=2), encoding="utf-8")
        return True, "settings saved"

    def _probe_slow(self) -> None:
        try:
            r = subprocess.run([PY, "-c", "import torch; print(torch.__version__, torch.cuda.is_available())"],
                               capture_output=True, text=True, timeout=60)
            self._slow["torch"] = r.stdout.strip() if r.returncode == 0 else ""
        except (OSError, subprocess.SubprocessError):
            self._slow["torch"] = ""
        self._slow["downloads"] = windows_downloads()
        self.probe_ort()
        self.probe_win_python()
        self._slow["repo_win"] = win_path(REPO) if str(REPO).startswith("/mnt/") or shutil.which("wslpath") else ""

    # ------------------------------------------------------------------ status
    def status(self) -> dict:
        now = time.time()
        if now - self._status_t > 2.0:
            isaac_dir = wsl_path(self.settings["isaac_path"]) if self.settings["isaac_path"] else ""
            py = self.settings["isaac_python"]
            isaac_ok = bool(py and os.path.exists(wsl_path(py))) or bool(isaac_dir and os.path.exists(os.path.join(isaac_dir, "python.bat")))
            torch = self._slow["torch"]
            self._status_cache = {
                "ros": os.path.exists("/opt/ros/jazzy/setup.bash"),
                "zenoh": os.path.isdir("/opt/ros/jazzy/share/rmw_zenoh_cpp"),
                "built": (colcon_base() / "install" / "setup.bash").exists(),
                "needs_build": needs_build(),
                "building": any(p.meta.get("type") == "build" and p.alive for p in self.procs.values()),
                "router": port_listening(ROUTER_PORT),
                "relay": port_listening(RELAY_PORT),
                "dashboard": port_listening(int(self.settings["dashboard_port"])),
                "windows": shutil.which("powershell.exe") is not None,
                "isaac": isaac_ok,
                "torch": torch, "cuda": bool(torch and torch.endswith("True")),
                "gpu": self._gpu_status(now),
                "update_zip": self.update_zip(),
                "update_zip_exists": bool(self.update_zip() and os.path.exists(self.update_zip())),
            }
            self._status_t = now
        return dict(self._status_cache)

    def _gpu_status(self, now: float) -> dict:
        if now - self._gpu_t > 5.0:
            self._gpu, self._gpu_t = gpu_query(), now
        return self._gpu

    @property
    def vehicle(self):
        """tools/launcher/vehicle.py (LapSim + the dashboard's parameter table); None if it cannot load."""
        if getattr(self, "_vehicle", None) is None:
            if getattr(self, "_vehicle_error", None):
                return None
            try:
                sys.path.insert(0, str(Path(__file__).resolve().parent))
                from vehicle import VehicleEditor  # noqa: PLC0415

                self._vehicle = VehicleEditor()
            except Exception as e:  # noqa: BLE001
                self._vehicle_error = f"{type(e).__name__}: {e}"
                return None
        return self._vehicle

    def vehicle_info(self) -> dict:
        v = self.vehicle
        if v is None:
            return {"ok": False, "message": f"vehicle setup not available ({getattr(self, '_vehicle_error', '?')})"}
        try:
            return {"ok": True, **v.info()}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "message": f"LapSim: {type(e).__name__}: {e}"}

    def vehicle_check(self, req: dict) -> dict:
        v = self.vehicle
        if v is None:
            return {"ok": False, "errors": [f"vehicle setup not available ({getattr(self, '_vehicle_error', '?')})"]}
        try:
            changes = clean_changes(req.get("changes", ""))
        except ValueError as e:
            return {"ok": False, "errors": [str(e)], "changes": []}
        return v.check(changes, str(req.get("longitudinal") or "reference"), str(req.get("policy") or "auto"))

    def gpu_jobs_alive(self) -> list:
        return [p for p in self.procs.values() if p.alive and p.meta.get("gpu")]

    def gpu_note(self) -> str:
        g = self._gpu or {}
        size = f" ({g['total_mb'] / 1024:.0f} GB)" if g.get("total_mb") else ""
        return (f"Isaac Sim and the training share the GPU{size}. If it runs out of memory, the training pauses, "
                "retries and keeps the best policy so far; stopping the simulation frees the memory.")

    def update_zip(self) -> str:
        z = self.settings.get("update_zip") or ""
        if z:
            return wsl_path(z) if re.match(r"^[A-Za-z]:\\", z) else z
        beside = REPO.parent / "fsg-dv-sim.zip"            # e.g. C:\formula_student\fsg-dv-sim.zip
        if beside.exists():
            return str(beside)
        d = self._slow.get("downloads")
        return os.path.join(d, "fsg-dv-sim.zip") if d else ""

    def policies(self) -> list[dict]:
        sys.path.insert(0, str(REPO / "ros2_ws" / "src" / "fsdsim"))
        try:
            from fsdsim.rl import registry  # noqa: PLC0415

            os.environ.setdefault("FSG_RL_POLICIES", str(REPO / "rl" / "policies"))
            return [{"name": p["dir_name"], "universal": bool(p.get("universal")), "trained": p.get("trained"),
                     "changes": p.get("changes", [])} for p in registry.list_policies(str(REPO / "rl" / "policies"))]
        except Exception:  # noqa: BLE001
            return []

    def cone_models(self) -> list[dict]:
        sys.path.insert(0, str(REPO / "ros2_ws" / "src" / "fsdsim"))
        try:
            from fsdsim.vision.detector import list_models  # noqa: PLC0415

            return [{"name": m["name"], "trained": m.get("trained"), "images": m.get("images"),
                     "mAP50": (m.get("metrics") or {}).get("mAP50")} for m in list_models(str(REPO / "models" / "cones"))]
        except Exception:  # noqa: BLE001
            return []

    def perception_info(self) -> dict:
        """Cone detectors, recorded training images (counted every 20 s) and the WSL inference runtime."""
        now = time.time()
        c = self._slow.get("perception")
        if c is None or now - c["t"] > 20.0:
            n, sessions = 0, 0
            root = REPO / "datasets" / "cones"
            if root.is_dir():
                for d in root.iterdir():
                    imgs = d / "images"
                    if (d / "frames.jsonl").exists() and not (d / "session.json").exists():
                        continue                    # recorded before the camera timing fix: not used for training
                    if imgs.is_dir():
                        k = sum(1 for e in os.scandir(imgs) if e.name.endswith((".jpg", ".png")))
                        n += k
                        sessions += 1 if k else 0
            c = {"t": now, "images": n, "sessions": sessions}
            self._slow["perception"] = c
        return {"models": self.cone_models(), "images": c["images"], "sessions": c["sessions"],
                "runtime": self._slow.get("ort", None)}

    def probe_ort(self) -> None:
        """Is ONNX Runtime (YOLO inference in WSL) installed? Else the detector uses OpenCV's DNN module."""
        try:
            r = subprocess.run([PY, "-c", "import onnxruntime as o; print(o.__version__)"], capture_output=True, text=True,
                               timeout=60)
            self._slow["ort"] = ("onnxruntime " + r.stdout.strip()) if r.returncode == 0 else ""
        except (OSError, subprocess.SubprocessError):
            self._slow["ort"] = ""

    def configs(self) -> list[dict]:
        out = []
        d = REPO / "lapsim" / "configs"
        for f in sorted(d.glob("*.json")) if d.is_dir() else []:
            try:
                c = json.loads(f.read_text(encoding="utf-8"))
                out.append({"name": c.get("name") or f.stem, "changes": c.get("changes", []), "notes": c.get("notes", "")})
            except (OSError, ValueError):
                continue
        return out

    def state(self) -> dict:
        with self.lock:
            procs = [p.info() for p in self.procs.values()]
            sess = json.loads(json.dumps(self.session))
        return {"version": version(), "repo": str(REPO), "repo_windows": self._slow.get("repo_win") or str(REPO),
                "colcon_base": str(colcon_base()), "status": self.status(), "win_python": self.win_python_info(), "procs": procs, "session": sess,
                "settings": self.settings, "policies": self.policies(), "configs": self.configs(),
                "perception": self.perception_info(),
                "missions": MISSIONS, "events": EVENTS,
                "dashboard_url": f"http://localhost:{self.settings['dashboard_port']}"}

    # ------------------------------------------------------------------ process bookkeeping
    def _start(self, p: Proc) -> Proc:
        with self.lock:
            old = self.procs.get(p.key)
            if old and old.alive:
                raise RuntimeError(f"{old.title} is already running")
            self.procs[p.key] = p
        p.start()
        return p

    def _alive(self, key: str) -> bool:
        p = self.procs.get(key)
        return bool(p and p.alive)

    def stop(self, key: str) -> tuple[bool, str]:
        p = self.procs.get(key)
        if not p or not p.alive:
            return False, "not running"
        threading.Thread(target=p.stop, daemon=True).start()
        return True, f"stopping {p.title}"

    # ------------------------------------------------------------------ router
    def start_router(self) -> tuple[bool, str]:
        if port_listening(ROUTER_PORT):
            return True, "Zenoh router already running on 127.0.0.1:7447"
        st = self.status()
        if not st["ros"] or not st["zenoh"]:
            return False, "ROS 2 Jazzy with rmw_zenoh_cpp is not installed in this WSL (docs/WINDOWS_WSL.md)"
        self._start(Proc("router", "Zenoh router", ros_argv("ros2", "run", "rmw_zenoh_cpp", "rmw_zenohd")))
        for _ in range(100):
            if port_listening(ROUTER_PORT):
                return True, "Zenoh router started"
            if not self._alive("router"):
                break
            time.sleep(0.1)
        return port_listening(ROUTER_PORT), "Zenoh router did not start (see its log)"

    # ------------------------------------------------------------------ simulation
    def start_sim(self, req: dict) -> tuple[bool, str]:
        backend = str(req.get("backend", "isaac"))
        mission = str(req.get("mission", "trackdrive"))
        if backend not in ("isaac", "lite", "demo"):
            return False, f"unknown backend '{backend}'"
        if mission not in MISSIONS:
            return False, f"unknown mission '{mission}'"
        if any(self._alive(k) for k in ("sim", "isaac")):
            return False, "a simulation is running: stop it first"
        try:
            changes = clean_changes(req.get("changes", ""))
            seed = int(req.get("seed") or -1)
            rtf = float(req.get("rtf") or 1.0)
        except (ValueError, TypeError) as e:
            return False, str(e)
        if not 0.1 <= rtf <= 20:
            return False, "real-time factor: 0.1 ... 20"
        lon = "rl" if req.get("longitudinal") == "rl" else "reference"
        policy = str(req.get("policy") or "auto")
        if not (policy in ("auto", "universal") or NAME_RE.match(policy)):
            return False, "invalid policy name"
        rviz = "true" if req.get("rviz", True) else "false"
        headless = bool(req.get("headless", False))
        dport = int(self.settings["dashboard_port"])
        lapsim_only = []
        if changes and self.vehicle is not None:
            chk = self.vehicle.check(changes)
            if chk["errors"]:
                return False, "vehicle setup: " + "; ".join(chk["errors"])
            changes, lapsim_only = chk["sim_changes"], chk["lapsim_only"]
        perception = str(req.get("perception") or "none")
        if perception not in PERCEPTION:
            return False, f"unknown cone source '{perception}'"
        record = bool(req.get("record_cones"))
        cone_model = str(req.get("cone_model") or "")
        if (perception == "yolo" or record) and backend != "isaac":
            return False, "camera cones (YOLO) and recording images need Isaac Sim (the other simulators render no images)"
        if perception != "none" and backend == "demo":
            return False, "the offline demo drives on ground-truth cones: use the lite simulator or Isaac Sim for perception"
        needs_yolo = perception == "yolo" or (backend == "isaac" and perception in ("fusion", "camera"))
        if needs_yolo:
            models = [m["name"] for m in self.cone_models()]
            if not models:
                return False, "no cone detector yet: record images, then RL & Performance -> Train the cone detector (YOLO)"
            if cone_model and cone_model not in models:
                return False, f"unknown cone detector '{cone_model}'"
        own = perception in ("lidar", "fusion", "camera")           # the fsd_perception node (LiDAR / fusion / camera + SLAM)
        slam = bool(req.get("slam")) and own
        line = "racing" if req.get("line") == "racing" else "center"
        try:
            race_after = int(req.get("race_after") or 0)
        except (ValueError, TypeError):
            return False, "race after: a number of laps (0 = off)"
        if not 0 <= race_after <= 20:
            return False, "race after: 0 ... 20 laps"
        if race_after and not (own and mission == "trackdrive"):
            return False, "'race on the mapped track' needs trackdrive and cones from LiDAR, fusion or camera"
        if line == "racing" and mission not in ("autocross", "trackdrive"):
            line = "center"                                           # only circuits have a racing line
        if line == "racing" and own and not race_after:
            return False, ("racing line with perceived cones: set 'race on the mapped track after' 1 or 2 laps "
                           "(the track is not known before the first lap)")
        if line == "racing" and lon == "rl":
            return False, "the racing line is for the reference driver (the RL policies were trained on the centre line)"
        faults = re.sub(r"\s+", "", str(req.get("faults") or ""))
        if faults:
            if len(faults) > 300 or not FAULT_RE.match(faults):
                return False, "sensor faults: sensor:kind[=value]@start[+duration], e.g. lidar:dropout@20+3,imu:bias=0.02@10"
            try:
                src = str(REPO / "ros2_ws" / "src" / "fsdsim")
                if src not in sys.path:
                    sys.path.insert(0, src)
                from fsdsim.faults import parse as parse_faults  # noqa: PLC0415

                parse_faults(faults)
            except Exception as e:  # noqa: BLE001
                return False, f"sensor faults: {e}"
        handling = "double_track" if req.get("handling") == "double_track" else "single_track"
        setup = str(req.get("setup_name") or "")[:60]
        opts = {"backend": backend, "mission": mission, "changes": changes, "seed": seed, "rtf": rtf,
                "longitudinal": lon, "policy": policy, "rviz": rviz == "true", "headless": headless,
                "setup_name": setup, "lapsim_only": lapsim_only,
                "perception": perception, "cone_model": cone_model, "record_cones": record,
                "slam": slam, "line": line, "race_after": race_after, "faults": faults, "handling": handling}
        st = self.status()
        if backend != "demo" and not st["ros"]:
            return False, "ROS 2 Jazzy is not installed in this WSL (docs/WINDOWS_WSL.md); the offline demo works without it"
        if backend == "isaac" and not st["windows"]:
            return False, "Windows interop (powershell.exe) is not available from this WSL"
        with self.lock:
            self.session = {"state": "starting", "steps": [], "opts": opts, "started": time.time()}
        threading.Thread(target=self._run_session, args=(opts, dport), daemon=True).start()
        if backend == "isaac" and self.gpu_jobs_alive():
            self._step("a GPU training is running: " + self.gpu_note())
            return True, f"starting Isaac Sim. Note: {self.gpu_note()}"
        return True, f"starting {'Isaac Sim' if backend == 'isaac' else 'the lite simulator' if backend == 'lite' else 'the offline demo'}"

    def _step(self, text: str, kind: str = "info") -> None:
        with self.lock:
            self.session["steps"].append({"t": time.time(), "text": text, "kind": kind})

    def _fail(self, text: str) -> None:
        self._step(text, "error")
        with self.lock:
            self.session["state"] = "failed"

    def _run_session(self, o: dict, dport: int) -> None:
        try:
            if o["backend"] == "demo":
                argv = [PY, "-m", "fsd_dashboard.demo", "--mission", o["mission"], "--port", dport, "--rtf", o["rtf"],
                        "--longitudinal", o["longitudinal"], "--start-setup", self._start_setup(o),
                        "--line", o.get("line", "center")]
                if o.get("faults"):
                    argv += ["--faults", o["faults"]]
                if sim_changes(o):
                    argv += ["--set", *sim_changes(o)]
                self._start(Proc("sim", "Offline demo (no ROS)", argv, env=py_env(), meta=o))
            else:
                ok, msg = self.ensure_built()
                if not ok:
                    return self._fail(msg)
                ok, msg = self.start_router()
                self._step(msg, "ok" if ok else "error")
                if not ok:
                    return self._fail("cannot continue without the Zenoh router")
                common = [f"longitudinal:={o['longitudinal']}", f"rl_policy:={o['policy']}", f"rviz:={'true' if o['rviz'] else 'false'}",
                          f"dashboard_port:={dport}", f"vehicle_setup:={self._start_setup(o)}"]
                common += [f"line:={o.get('line', 'center')}"]
                if o.get("perception", "none") != "none":
                    common += [f"perception:={o['perception']}", f"slam:={'true' if o.get('slam') else 'false'}",
                               f"race_after:={int(o.get('race_after') or 0)}"]
                    if o["backend"] == "isaac" and o["perception"] in ("yolo", "fusion", "camera"):
                        if o.get("cone_model"):          # "Newest": no argument (an empty `cone_model:=` is
                            common += ["cone_model:=" + str(REPO / "models" / "cones" / o["cone_model"])]   # rejected)
                if o.get("record_cones"):
                    common += ["record_cones:=true",
                               f"record_session:=isaac_{o['mission']}_{_dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"]
                if o["backend"] == "lite":
                    args = ["ros2", "launch", "fsd_sim_bringup", "lite.launch.py", f"mission:={o['mission']}",
                            f"realtime_factor:={o['rtf']}", *common]
                    if o["seed"] >= 0:
                        args.append(f"seed:={o['seed']}")
                    if sim_changes(o):
                        args.append("vehicle_set:=" + " ".join(sim_changes(o)))
                    if o.get("faults"):
                        args.append(f"faults:={o['faults']}")
                    self._start(Proc("sim", "Lite simulator (ROS 2)", ros_argv(*args), meta=o))
                    self._step("lite.launch.py started")
                else:
                    if isaac_camera_hz(o):                 # the dashboard expects the rate Isaac renders at
                        common += [f"camera_hz:={isaac_camera_hz(o)}"]
                    self._start(Proc("sim", "ROS 2 relay, driver, dashboard (isaac.launch.py)",
                                     ros_argv("ros2", "launch", "fsd_sim_bringup", "isaac.launch.py", *common), meta=o))
                    self._step("isaac.launch.py started, waiting for the relay on port 47800")
                    for _ in range(600):
                        if port_listening(RELAY_PORT) or not self._alive("sim"):
                            break
                        time.sleep(0.1)
                    if not port_listening(RELAY_PORT):
                        return self._fail("the relay did not open port 47800 (see the log of isaac.launch.py)")
                    self._step("relay ready", "ok")
                    self._start_isaac(o)
                    self._step("Isaac Sim starting on Windows (its window opens in 20-60 s)")
            # dashboard
            for _ in range(300):
                if port_listening(dport) or not self._alive("sim"):
                    break
                time.sleep(0.1)
            if port_listening(dport):
                self._step(f"dashboard on http://localhost:{dport}", "ok")
                with self.lock:
                    self.session["state"] = "running"
                    self.session["dashboard"] = True
            else:
                self._fail("the dashboard did not come up (see the simulation log)")
        except Exception as e:  # noqa: BLE001
            self._fail(f"{type(e).__name__}: {e}")

    @staticmethod
    def _start_setup(o: dict) -> str:
        """What the dashboard shows as the active setup: the name and every change (also LapSim-only ones)."""
        return encode_start_setup(o.get("setup_name") or "", list(o["changes"]) + list(o.get("lapsim_only") or []))

    def _start_isaac(self, o: dict) -> None:
        ps1 = win_path(REPO / "isaac" / "run_isaac.ps1")
        if not ps1:
            raise RuntimeError("cannot translate the repo path for Windows (wslpath)")
        cmd = f"Write-Output ('[fsd-launcher] winpid ' + $PID); & {ps_quote(ps1)}"
        if o["mission"]:
            cmd += f" -Mission {ps_quote(o['mission'])}"
        if sim_changes(o):
            cmd += " -Set " + ",".join(ps_quote(c) for c in sim_changes(o))
        if o.get("faults"):
            cmd += f" -Faults {ps_quote(o['faults'])}"
        if o["headless"]:
            cmd += " -Headless"
        if self.settings.get("isaac_python"):
            cmd += f" -Python {ps_quote(self.settings['isaac_python'])}"
        elif self.settings.get("isaac_path"):
            cmd += f" -IsaacPath {ps_quote(self.settings['isaac_path'])}"
        extra = []
        if o["seed"] >= 0 and o["mission"] in ("autocross", "trackdrive"):
            extra += ["--seed", str(int(o["seed"]))]
        if isaac_camera_hz(o):
            extra += ["--camera-hz", str(isaac_camera_hz(o))]
        if extra:
            cmd += " -Extra " + ",".join(ps_quote(a) for a in extra)
        argv = ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", cmd]
        self._start(Proc("isaac", "Isaac Sim (Windows)", argv, windows=True, cwd=Path("/mnt/c") if os.path.isdir("/mnt/c") else REPO,
                         meta=o))

    def _build_proc(self) -> Proc | None:
        return next((p for p in self.procs.values() if p.meta.get("type") == "build" and p.alive), None)

    def ensure_built(self) -> tuple[bool, str]:
        """Build the workspace first when its sources changed (new files, setup.py, package.xml, msgs)."""
        b = self._build_proc()
        if b is None and needs_build():
            self._step("the workspace sources changed: building it first (colcon, about a minute)")
            ok, msg = self.start_job({"type": "build"})
            if not ok:
                return False, msg
            b = self._build_proc()
        elif b is not None:
            self._step("waiting for the running workspace build")
        if b is not None:
            while b.alive:
                time.sleep(0.5)
            if b.state != "done":
                return False, "the workspace build failed (see its log under Jobs)"
            self._step("workspace built", "ok")
        return True, "workspace up to date"

    def auto_build(self) -> None:
        """At start: build when ROS 2 is installed and the sources changed since the last build."""
        if os.path.exists("/opt/ros/jazzy/setup.bash") and needs_build() and self._build_proc() is None:
            self.start_job({"type": "build"})

    def stop_sim(self) -> tuple[bool, str]:
        stopped = [k for k in ("isaac", "sim") if self._alive(k)]

        def run():
            for k in ("isaac", "sim"):
                if self._alive(k):
                    self.procs[k].stop()
            with self.lock:
                self.session["state"] = "stopped"
                self.session["dashboard"] = False
                self.session["steps"].append({"t": time.time(), "text": "simulation stopped", "kind": "info"})

        threading.Thread(target=run, daemon=True).start()
        return bool(stopped), "stopping the simulation" if stopped else "no simulation running"

    # ------------------------------------------------------------------ jobs
    def start_job(self, req: dict) -> tuple[bool, str]:
        kind = str(req.get("type", ""))
        p = req.get("params") or {}
        stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        try:
            title, argv, links, env = self._job(kind, p, stamp)
        except (ValueError, TypeError, KeyError) as e:
            return False, str(e)
        windows = bool(argv and argv[0] == "powershell.exe")
        line = " ".join(str(a) for a in argv)           # WSL argv or one PowerShell command
        gpu = ((kind in ("train_universal", "sweep") and bool(re.search(r"--backend'?\s+'?torch", line))) or
               kind == "train_cones") and not re.search(r"--device'?\s+'?cpu", line)
        self._job_n += 1
        key = f"job{self._job_n}-{kind}"
        try:
            self._start(Proc(key, title, argv, kind="job", env=env, links=links, windows=windows,
                             cwd=Path("/mnt/c") if (windows or argv[0] == "cmd.exe") and os.path.isdir("/mnt/c") else REPO,
                             meta={"type": kind, "params": p, "windows": windows, "gpu": gpu},
                             on_exit=(lambda _p: self.probe_win_python()) if kind in ("win_pip", "yolo_pip")
                             else (lambda _p: self.probe_ort()) if kind == "ort_pip" else None))
        except RuntimeError as e:
            return False, str(e)
        if gpu and self._alive("isaac"):
            return True, f"started: {title}. Note: {self.gpu_note()}"
        return True, f"started: {title}"

    def _job(self, kind: str, p: dict, stamp: str):
        env = py_env()
        T = REPO / "tools"
        if kind == "train_universal":
            backend = "torch" if p.get("backend") == "torch" else "numpy"
            name = str(p.get("name") or "universal_" + stamp)
            if not NAME_RE.match(name):
                raise ValueError("policy name: letters, digits and . _ + - only")
            it = int(p.get("iterations") or (1500 if backend == "torch" else 2000))
            n = int(p.get("num_envs") or (4096 if backend == "torch" else 256))
            if not (1 <= it <= 100000 and 8 <= n <= 262144):
                raise ValueError("iterations 1 ... 100000, cars 8 ... 262144")
            args = [T / "train_rl_universal.py", "--backend", backend, "--iterations", it, "--num-envs", n, "--name", name]
            if backend == "torch":
                dev = str(p.get("device") or "cuda:0")
                if not re.match(r"^(cuda(:\d+)?|cpu)$", dev):
                    raise ValueError("device: cuda, cuda:N or cpu")
                args += ["--device", dev]
            init = str(p.get("init") or "")
            if init:
                if not NAME_RE.match(init):
                    raise ValueError("invalid policy to continue from")
                args += ["--init", REPO / "rl" / "policies" / init]
            where = ""
            if backend == "torch":
                argv, where = self._torch_argv(args)
            else:
                argv = [PY, *args]
            return f"Universal RL training '{name}' ({backend}{where}, {it} iterations)", argv, [], env
        if kind == "train_setup":
            changes = clean_changes(p.get("changes", ""))
            g = int(p.get("generations") or 50)
            if not 1 <= g <= 1000:
                raise ValueError("generations 1 ... 1000")
            argv = [PY, T / "train_rl.py", "--generations", g]
            name = str(p.get("name") or "")
            if name:
                if not NAME_RE.match(name):
                    raise ValueError("policy name: letters, digits and . _ + - only")
                argv += ["--name", name]
            if changes:
                argv += ["--set", *changes]
            return f"RL training for {' '.join(changes) or 'the LapSim baseline'} ({g} generations)", argv, [], env
        if kind == "evaluate":
            pol = str(p.get("policy") or "universal")
            if not NAME_RE.match(pol):
                raise ValueError("invalid policy name")
            changes = clean_changes(p.get("changes", ""))
            argv = [PY, T / "train_rl_universal.py", "--evaluate", REPO / "rl" / "policies" / pol]
            if changes:
                argv += ["--set", *changes]
            return f"Official runs: {pol} vs reference ({' '.join(changes) or 'baseline'})", argv, [], env
        if kind == "sweep":
            varies = [v.strip() for v in str(p.get("vary", "")).splitlines() if v.strip()]
            if not varies:
                raise ValueError("give at least one parameter to vary, e.g. m=250:320:4")
            for v in varies:
                if not re.match(r"^[A-Za-z_]\w*[=*][-+0-9.eE:,]+$", v):
                    raise ValueError(f"vary '{v}': NAME=a:b:n, NAME=v1,v2 or NAME*a:b:n")
            tracks = int(p["tracks"]) if p.get("tracks") not in (None, "") else 6
            if not 0 <= tracks <= 200:
                raise ValueError("tracks 0 ... 200")
            out = f"results/sweep_{stamp}"
            args = [T / "rl_spec_sweep.py", "--tracks", tracks, "--out", REPO / out]
            for v in varies:
                args += ["--vary", v]
            base = clean_changes(p.get("changes", ""))
            if base:
                args += ["--set", *base]
            if p.get("compare"):
                args.append("--compare")
            where = ""
            if p.get("backend") == "torch":
                args += ["--backend", "torch"]
                argv, where = self._torch_argv(args)
            else:
                argv = [PY, *args]
            return f"Setup sweep ({', '.join(varies)}){where}", argv, [
                {"label": "Table (Markdown)", "path": f"{out}/sweep.md"}, {"label": "CSV", "path": f"{out}/sweep.csv"}], env
        if kind == "param_study":
            param = str(p.get("param") or "")
            if not re.match(r"^[A-Za-z_][\w.]*$", param):
                raise ValueError("parameter name, e.g. m, mu, c_z, i_trans")
            vals = str(p.get("values") or "").replace(" ", "")
            if not re.match(r"^[-+0-9.eE]+(,[-+0-9.eE]+)+$", vals):
                raise ValueError("values: at least two numbers separated by commas")
            mode = "--scale" if p.get("mode") == "scale" else "--values"
            evs = [e for e in (p.get("events") or ["acceleration", "skidpad", "autocross"]) if e in EVENTS]
            out = f"results/study_{param}_{stamp}"
            argv = [PY, T / "param_study.py", "--param", param, mode, vals, "--events", *evs, "--out", REPO / out,
                    "--driver", "rl" if p.get("driver") == "rl" else "reference"]
            if p.get("no_sim"):
                argv.append("--no-sim")
            base = clean_changes(p.get("changes", ""))
            if base:
                argv += ["--set", *base]
            return f"Parameter study: {param} = {vals}", argv, [{"label": "Report", "path": f"{out}/report.html"},
                                                                {"label": "CSV", "path": f"{out}/results.csv"}], env
        if kind == "regression":
            out = f"results/regression_{stamp}"
            argv = [PY, T / "regression.py", "--out", REPO / out]
            if p.get("quick"):
                argv.append("--quick")
            seeds = [int(x) for x in re.findall(r"\d+", str(p.get("seeds") or ""))][:12]
            if seeds:
                argv += ["--seeds", *seeds]
            return ("Regression run (closed loop" + (", quick)" if p.get("quick") else ", all groups)"), argv,
                    [{"label": "Report", "path": f"{out}/report.md"}, {"label": "JSON", "path": f"{out}/results.json"}], env)
        if kind == "diagnose":
            # what runs in WSL and how busy it is: left-over nodes from stopped runs, memory, load (fixed command)
            script = ("echo '== uptime / load'; uptime; echo; echo '== memory (MB)'; free -m; echo; "
                      "echo '== ROS 2 / simulator processes'; ps -eo pid,etime,%cpu,%mem,args --sort=-%cpu | "
                      "grep -E 'ros2|fsd|rviz|rmw_zenoh|python3|isaac|launch' | grep -v grep | head -40; echo; "
                      "echo '== busiest processes'; ps -eo pid,etime,%cpu,%mem,comm --sort=-%cpu | head -15")
            return "Diagnose WSL (processes, memory, load)", ["bash", "-c", script], [], env
        if kind in ("git_check", "git_push"):
            st = self.settings
            if not (st.get("git_github") or st.get("git_gitlab")):
                raise ValueError("set the GitHub and / or GitLab URL first (Maintenance -> GitHub / GitLab -> Save)")
            argv = [PY, T / "git_push.py", "--github", st.get("git_github", ""), "--gitlab", st.get("git_gitlab", ""),
                    "--branch", st.get("git_branch") or "main"]
            if st.get("git_name"):
                argv += ["--name", st["git_name"]]
            if st.get("git_email"):
                argv += ["--email", st["git_email"]]
            if kind == "git_check":
                return "Git: what would be pushed", argv + ["--check"], [], env
            msg = str(p.get("message") or "").strip()
            if len(msg) > 500 or "\0" in msg:
                raise ValueError("commit message: at most 500 characters")
            if self._git_busy():
                raise ValueError("a push is already running")
            targets = " + ".join(n for n, k in (("GitHub", "git_github"), ("GitLab", "git_gitlab")) if st.get(k))
            return f"Git: commit and push to {targets}", argv + ["--init", "--message", msg], [], env
        if kind == "check_install":
            return "Installation check", [PY, T / "check_install.py"], [], env
        if kind == "tests":
            return "Unit tests", [PY, "-m", "unittest", "discover", "-s", "tests"], [], env
        if kind == "gpu_tests":
            wp = self.win_python_info()
            if wp.get("path") and wp.get("torch"):
                return (f"PyTorch checks of the GPU code (Windows Python {wp.get('python', '')}, torch {wp['torch']})",
                        self._win_py_argv(["-m", "unittest", "-v", "tests.test_vec"]), [], None)
            return "PyTorch checks of the GPU code (WSL)", [PY, "-m", "unittest", "-v", "tests.test_vec"], [], env
        if kind == "train_cones":
            name = str(p.get("name") or "cones_" + stamp)
            if not NAME_RE.match(name):
                raise ValueError("detector name: letters, digits and . _ + - only")
            epochs = int(p.get("epochs") or 60)
            batch = int(p.get("batch") or 16)
            size = "s" if p.get("size") == "s" else "n"
            if not (1 <= epochs <= 1000 and 1 <= batch <= 256):
                raise ValueError("epochs 1 ... 1000, batch 1 ... 256")
            imgs = self.perception_info()["images"]
            if imgs < 50:
                raise ValueError(f"only {imgs} training images recorded: first record some - Simulation -> Isaac Sim, "
                                 "tick 'Record camera images for YOLO training' and drive a trackdrive or autocross run "
                                 "(one trackdrive gives several hundred images)")
            args = [T / "train_cone_yolo.py", "--name", name, "--epochs", epochs, "--batch", batch,
                    "--model", f"yolo11{size}.pt"]
            wp = self.win_python_info()
            if wp.get("path") and wp.get("torch") and not wp.get("ultralytics"):
                raise ValueError("Ultralytics is not installed for the Windows Python: Maintenance -> Install YOLO (Ultralytics)")
            argv, where = self._torch_argv(args)
            if not wp.get("cuda") and argv[0] == "powershell.exe":
                argv = self._win_py_argv([*args, "--device", "cpu"])
            return (f"Cone detector '{name}' (YOLO11{size}, {epochs} epochs){where}", argv,
                    [{"label": "Results", "path": f"models/cones/{name}/results.md"}], env)
        if kind == "yolo_pip":
            if not self.win_python_info().get("path"):
                raise ValueError("no Windows Python found: set it under Settings (e.g. C:\\Python314\\python.exe)")
            return ("Install YOLO (Ultralytics, ONNX export) for the Windows Python",
                    self._win_py_argv(["-m", "pip", "install", "--user", "ultralytics", "onnx", "onnxslim"]), [], None)
        if kind == "ort_pip":
            script = ('set -e; if ! python3 -m pip --version >/dev/null 2>&1; then '
                      'echo "pip is missing: installing it for this user (get-pip.py)"; '
                      'curl -sSL https://bootstrap.pypa.io/get-pip.py | python3 - --user --break-system-packages; fi; '
                      'python3 -m pip install --user --break-system-packages onnxruntime; '
                      'python3 -c "import onnxruntime as o; print(\'onnxruntime\', o.__version__, o.get_available_providers())"')
            return "Install ONNX Runtime in WSL (YOLO inference)", ["bash", "-c", script], [], None
        if kind == "win_pip":
            if not self.win_python_info().get("path"):
                raise ValueError("no Windows Python found: set it under Settings (e.g. C:\\Python314\\python.exe)")
            return ("Install numpy and PyYAML for the Windows Python (GPU jobs)",
                    self._win_py_argv(["-m", "pip", "install", "--user", "numpy", "pyyaml"]), [], None)
        if kind == "build":
            return "Build the ROS 2 workspace (colcon)", ["bash", REPO / "tools" / "wsl" / "fsgbuild.sh"], [], None
        if kind == "shell":
            return ("Point the WSL terminal commands (fsgsim, fsgrouter, fsgbuild) to this copy",
                    ["bash", REPO / "tools" / "wsl" / "setup_shell.sh"], [], None)
        if kind == "update":
            z = self.update_zip()
            if not z or not os.path.exists(z):
                raise ValueError(f"no update zip at '{z or 'Downloads/fsg-dv-sim.zip'}'")
            script = ('set -e; echo "unzip $0 -> $1"; unzip -o -q "$0" -d "$1"; bash "$2"; '
                      'echo; echo "Update installed. Restart the launcher (Quit, then run.bat) to use its new version."')
            return "Install update from the zip and build", \
                ["bash", "-c", script, z, REPO.parent, REPO / "tools" / "wsl" / "fsgbuild.sh"], [], None
        if kind == "perf_lab":
            # the Performance Lab runs on Windows (its own launcher window, perf/perf.bat); start it from here
            bat = win_path(REPO / "perf" / "perf.bat")
            if not bat or not shutil.which("cmd.exe"):
                raise ValueError("Windows interop (cmd.exe, wslpath) is not available: double-click perf\\perf.bat in Windows")
            return "Open the Performance Lab (its own window on Windows)", \
                ["cmd.exe", "/c", "start", "Performance Lab", bat], [], None
        if kind == "desktop":
            src = win_path(REPO / "run.bat")
            if not src or not shutil.which("powershell.exe"):
                raise ValueError("Windows interop (powershell.exe, wslpath) is not available")
            cmd = ("$d = Join-Path ([Environment]::GetFolderPath('Desktop')) 'FSG DV Launcher.bat'; "
                   f"Copy-Item -LiteralPath {ps_quote(src)} -Destination $d -Force; Write-Output ('copied to ' + $d)")
            return "Put the launcher (run.bat) on the Windows desktop", \
                ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", cmd], [], None
        raise ValueError(f"unknown job '{kind}'")

    # ------------------------------------------------------------------ Windows Python (GPU jobs)
    def _git_busy(self) -> bool:
        return any(p.alive and p.meta.get("type") == "git_push" for p in self.procs.values())

    def _find_win_python(self) -> str:
        if self.settings.get("win_python"):
            return self.settings["win_python"]
        try:
            out = subprocess.run(["cmd.exe", "/c", "where", "python"], capture_output=True, text=True, timeout=10,
                                 cwd="/mnt/c" if os.path.isdir("/mnt/c") else None).stdout
            for line in out.splitlines():
                line = line.strip()
                if line.lower().endswith("python.exe") and "windowsapps" not in line.lower():
                    return line
        except (OSError, subprocess.SubprocessError):
            pass
        return ""

    def probe_win_python(self) -> None:
        """Version, PyTorch / CUDA / GPU and missing packages of the Windows Python (background, ~5 s)."""
        info: dict = {"path": "", "checked": time.time()}
        win = self._find_win_python() if shutil.which("cmd.exe") else ""
        exe = wsl_path(win) if win else ""
        if exe and os.path.exists(exe):
            code = ("import json, sys\nr = {'python': sys.version.split()[0], 'missing': []}\n"
                    "for m in ('numpy', 'yaml'):\n"
                    "    try: __import__(m)\n"
                    "    except ImportError: r['missing'].append('pyyaml' if m == 'yaml' else m)\n"
                    "try:\n    import torch\n    r['torch'] = torch.__version__\n    r['cuda'] = torch.cuda.is_available()\n"
                    "    r['gpu'] = torch.cuda.get_device_name(0) if r['cuda'] else ''\n"
                    "except Exception as e:\n    r['torch'] = ''\n    r['torch_error'] = str(e)[:200]\n"
                    "try:\n    import importlib.metadata as md\n    r['ultralytics'] = md.version('ultralytics')\n"
                    "except Exception:\n    r['ultralytics'] = ''\n"
                    "print(json.dumps(r))\n")
            try:
                out = subprocess.run([exe, "-c", code], capture_output=True, text=True, timeout=120,
                                     cwd="/mnt/c" if os.path.isdir("/mnt/c") else None)
                info.update(json.loads(out.stdout.strip().splitlines()[-1]))
                info["path"] = win
            except (OSError, subprocess.SubprocessError, ValueError, IndexError):
                info["error"] = "could not run it"
                info["path"] = win
        self._slow["win_python"] = info

    def win_python_info(self) -> dict:
        return dict(self._slow.get("win_python") or {})

    def _win_py_argv(self, args: list) -> list:
        """powershell.exe running the Windows Python in the repo folder (stoppable with taskkill)."""
        wp = self.win_python_info()
        if not wp.get("path"):
            raise ValueError("no Windows Python found (Settings -> Windows Python for GPU jobs)")
        repo_w = win_path(REPO)
        if not repo_w:
            raise ValueError("cannot translate the repo path for Windows (wslpath)")
        conv = [win_path(a) if isinstance(a, Path) else str(a) for a in args]
        cmd = ("Write-Output ('[fsd-launcher] winpid ' + $PID); $env:PYTHONUTF8 = '1'; $env:PYTHONUNBUFFERED = '1'; "
               f"Set-Location -LiteralPath {ps_quote(repo_w)}; & {ps_quote(wp['path'])} " + " ".join(ps_quote(c) for c in conv)
               + "; exit $LASTEXITCODE")
        return ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", cmd]

    def _torch_argv(self, args: list) -> tuple[list, str]:
        """GPU jobs: the Windows Python when it has CUDA PyTorch, else WSL's python3."""
        wp = self.win_python_info()
        if wp.get("path") and wp.get("torch") and not wp.get("missing"):
            return self._win_py_argv(args), f", Windows Python + {'CUDA' if wp.get('cuda') else 'CPU'} PyTorch"
        if self._slow.get("torch"):
            return [PY, *args], ", WSL PyTorch"
        if wp.get("path") and wp.get("torch"):
            raise ValueError("the Windows Python has PyTorch but lacks " + ", ".join(wp["missing"])
                             + ": Maintenance -> Install packages for GPU jobs")
        raise ValueError("no PyTorch found (Windows Python or WSL): pip install torch")

    # ------------------------------------------------------------------ share
    def report(self, n_lines: int = 150, full: bool = False) -> str:
        """Markdown report of the system, the simulation and every process / job with its log."""
        d = self.state()
        st, sess = d["status"], d["session"]
        lines = [f"# FSG DV launcher report - {_dt.datetime.now():%Y-%m-%d %H:%M:%S}", "",
                 f"* simulator {d['version']}, project `{d['repo_windows']}` (WSL `{d['repo']}`), colcon `{d['colcon_base']}`",
                 f"* system: " + ", ".join(f"{k}={v}" for k, v in st.items() if k not in ("update_zip",)),
                 f"* Windows Python: {json.dumps(self.win_python_info())}",
                 f"* settings: {json.dumps(d['settings'])}", ""]
        if sess.get("opts"):
            lines += ["## Simulation", "", f"* state: {sess.get('state')}, options: {json.dumps(sess.get('opts'))}"]
            lines += [f"* {_dt.datetime.fromtimestamp(s['t']):%H:%M:%S} [{s['kind']}] {s['text']}" for s in sess.get("steps", [])]
            lines.append("")
        procs = list(self.procs.values())
        lines += ["## Processes and jobs", ""]
        if not procs:
            lines.append("(none since the launcher started)")
        for p in procs:
            dur = f"{(p.ended or time.time()) - p.started:.0f} s" if p.started else ""
            lines += [f"### {p.title}", "",
                      f"* {p.kind}, state {p.state}, exit code {p.rc}, {dur}, params {json.dumps(p.meta.get('params', p.meta))}",
                      f"* command: `{' '.join(p.argv)[:600]}`"]
            res = self.job_results(p.key)
            if res:
                lines += ["", "Results:", "", res]
            log = p.tail(10 ** 6 if full else n_lines)
            lines += ["", f"Log ({'complete' if full else f'last {n_lines} lines'}):", "", "```", log.rstrip(), "```", ""]
        return "\n".join(lines) + "\n"

    def job_results(self, key: str, max_lines: int = 200) -> str:
        """Text of a finished job's result files (tables, CSV) for copying."""
        p = self.procs.get(key)
        if not p:
            return ""
        out = []
        for lk in p.links:
            f = REPO / lk["path"]
            if f.suffix in (".md", ".csv", ".txt") and f.exists():
                txt = f.read_text(encoding="utf-8", errors="replace").splitlines()
                more = f"\n... ({len(txt) - max_lines} more lines in {lk['path']})" if len(txt) > max_lines else ""
                out.append(f"#### {lk['path']}\n\n" + "\n".join(txt[:max_lines]) + more)
        return "\n\n".join(out)

    def save_report(self) -> tuple[bool, str]:
        """shared/<time>/report.md + every complete log, inside the project folder (readable from Windows)."""
        stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        out = REPO / "shared" / stamp
        (out / "logs").mkdir(parents=True, exist_ok=True)
        rep = self.report(full=False, n_lines=400)
        (out / "report.md").write_text(rep, encoding="utf-8")
        for p in self.procs.values():
            if p.log_path.exists():
                shutil.copy(p.log_path, out / "logs" / p.log_path.name)
        (REPO / "shared" / "LATEST.md").write_text(rep, encoding="utf-8")
        where = win_path(out) or str(out)
        return True, f"saved to {where} (also shared\\LATEST.md): tell Claude it is there"

    # ------------------------------------------------------------------ shutdown
    def stop_all(self) -> None:
        for k in ("isaac", "sim"):
            if self._alive(k):
                self.procs[k].stop()
        for p in list(self.procs.values()):
            if p.alive:
                p.stop(timeout=5)


# ---------------------------------------------------------------------------- #
# HTTP
# ---------------------------------------------------------------------------- #
def make_handler(app: Launcher, shutdown):
    class H(BaseHTTPRequestHandler):
        server_version = "fsg-launcher"

        def log_message(self, *a):
            pass

        def _send(self, code, body: bytes, mime: str):
            self.send_response(code)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=HTTPStatus.OK):
            self._send(code, json.dumps(obj, default=str).encode(), "application/json")

        def _file(self, root: Path, rel: str):
            p = (root / rel).resolve()
            if root.resolve() not in p.parents or not p.is_file():
                return self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")
            mime = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
            if mime.startswith("text/") or mime.endswith(("javascript", "json")):
                mime += "; charset=utf-8"
            self._send(HTTPStatus.OK, p.read_bytes(), mime)

        def do_GET(self):  # noqa: N802
            u = urlparse(self.path)
            path = unquote(u.path)
            if path in ("/", "/index.html"):
                return self._file(WEB, "index.html")
            if path.startswith("/static/"):
                return self._file(WEB, path[len("/static/"):])
            if path.startswith("/brand/"):
                return self._file(BRAND, path[len("/brand/"):])
            if path.startswith("/results/"):
                return self._file(RESULTS, path[len("/results/"):])
            if path == "/api/ping":
                return self._json({"ok": True, "app": "fsg-launcher", "version": version()})
            if path == "/api/state":
                return self._json(app.state())
            if path == "/api/vehicle":
                return self._json(app.vehicle_info())
            if path == "/api/vehicle/matlab":
                v = app.vehicle
                try:
                    return self._json(v.matlab_state() if v else {"available": False, "message": "vehicle setup not available"})
                except Exception as e:  # noqa: BLE001
                    return self._json({"available": False, "message": f"{type(e).__name__}: {e}"})
            if path == "/api/report":
                q = parse_qs(u.query)
                n = max(20, min(int(q.get("n", ["150"])[0]), 20000))
                return self._send(HTTPStatus.OK, app.report(n_lines=n).encode(), "text/markdown; charset=utf-8")
            if path == "/api/results":
                q = parse_qs(u.query)
                return self._send(HTTPStatus.OK, app.job_results(q.get("key", [""])[0]).encode(), "text/plain; charset=utf-8")
            if path == "/api/log":
                q = parse_qs(u.query)
                p = app.procs.get(q.get("key", [""])[0])
                n = max(10, min(int(q.get("n", ["300"])[0]), 5000))
                return self._send(HTTPStatus.OK, (p.tail(n) if p else "").encode(), "text/plain; charset=utf-8")
            return self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")

        def do_POST(self):  # noqa: N802
            if self.headers.get("X-FSG-Launcher") != "1":
                return self._json({"ok": False, "message": "missing launcher header"}, HTTPStatus.FORBIDDEN)
            try:
                n = int(self.headers.get("Content-Length") or 0)
                req = json.loads(self.rfile.read(min(n, 1 << 20)) or b"{}")
                if not isinstance(req, dict):
                    raise ValueError
            except ValueError:
                return self._json({"ok": False, "message": "bad JSON"}, HTTPStatus.BAD_REQUEST)
            path = urlparse(self.path).path
            try:
                if path == "/api/vehicle/check":
                    return self._json(app.vehicle_check(req))
                if path == "/api/sim/start":
                    ok, msg = app.start_sim(req)
                elif path == "/api/vehicle/save":
                    v = app.vehicle
                    ok, msg = (v.save(str(req.get("name") or ""), clean_changes(req.get("changes", "")),
                                      str(req.get("notes") or "")) if v else (False, "vehicle setup not available"))
                elif path == "/api/vehicle/matlab":
                    v = app.vehicle
                    if v is None:
                        ok, msg = False, "vehicle setup not available"
                    elif req.get("action") == "cancel":
                        ok, msg = v.matlab_cancel()
                    else:
                        ok, msg = v.matlab_run(clean_changes(req.get("changes", "")))
                elif path == "/api/vehicle/open":
                    v = app.vehicle
                    ok, msg = v.open(str(req.get("target") or "")) if v else (False, "vehicle setup not available")
                elif path == "/api/vehicle/delete":
                    v = app.vehicle
                    ok, msg = v.delete(str(req.get("name") or "")) if v else (False, "vehicle setup not available")
                elif path == "/api/sim/stop":
                    ok, msg = app.stop_sim()
                elif path == "/api/router/start":
                    ok, msg = app.start_router()
                elif path == "/api/proc/stop":
                    ok, msg = app.stop(str(req.get("key", "")))
                elif path == "/api/job/start":
                    ok, msg = app.start_job(req)
                elif path == "/api/settings":
                    ok, msg = app.save_settings(req)
                    if ok and "win_python" in req:
                        threading.Thread(target=app.probe_win_python, daemon=True).start()
                elif path == "/api/report/save":
                    ok, msg = app.save_report()
                elif path == "/api/winpython/check":
                    threading.Thread(target=app.probe_win_python, daemon=True).start()
                    ok, msg = True, "checking the Windows Python ..."
                elif path == "/api/quit":
                    threading.Thread(target=shutdown, daemon=True).start()
                    ok, msg = True, "the launcher is shutting down; everything it started is stopped"
                else:
                    return self._json({"ok": False, "message": "unknown action"}, HTTPStatus.NOT_FOUND)
            except Exception as e:  # noqa: BLE001
                ok, msg = False, f"{type(e).__name__}: {e}"
            return self._json({"ok": ok, "message": msg})

    return H


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=int(os.environ.get("FSG_LAUNCHER_PORT", 8070)))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-browser", action="store_true", help="do not open the page (run.bat opens it)")
    ap.add_argument("--write-build-stamp", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.write_build_stamp:            # called by tools/wsl/fsgbuild.sh after a successful build
        write_build_stamp()
        return

    app = Launcher()
    httpd = ThreadingHTTPServer((args.host, args.port), None)
    done = threading.Event()

    def shutdown(*_):
        if done.is_set():
            return
        done.set()
        print("[launcher] stopping everything ...", flush=True)
        app.stop_all()
        threading.Thread(target=httpd.shutdown, daemon=True).start()

    httpd.RequestHandlerClass = make_handler(app, shutdown)
    app.auto_build()
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda *_: threading.Thread(target=shutdown, daemon=True).start())
    url = f"http://localhost:{args.port}"
    print(f"[launcher] FSG DV simulator {version()} - {url}  (close this window or press Quit in the page to stop)",
          flush=True)
    if not args.no_browser and shutil.which("cmd.exe"):
        subprocess.Popen(["cmd.exe", "/c", "start", "", url], cwd="/mnt/c" if os.path.isdir("/mnt/c") else None,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        httpd.serve_forever(poll_interval=0.3)
    finally:
        app.stop_all()
        print("[launcher] bye", flush=True)


if __name__ == "__main__":
    main()
