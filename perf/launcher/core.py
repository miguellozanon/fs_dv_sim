"""Performance Lab launcher: everything the window does, without the window (standard library only).

The window (perf_launcher.pyw) is a thin layer over these functions, so they can be tested without a display,
and they keep working when numpy / scipy / pyyaml are missing (that is what "Check and repair" is for).

    status = health_check()            # list of Check(name, ok, detail, fix)
    start_app(); is_running(); stop_app()
    backup(); restore(zip_path)        # the team's setups and efficiency maps (lapsim/configs)
    help_report()                      # one zip to send to whoever helps
    create_shortcuts()                 # desktop + Start menu (Windows)
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
PERF = HERE.parent
REPO = PERF.parent
APP = PERF / "app.py"
CONFIGS = REPO / "lapsim" / "configs"
LAPSIM = REPO / "lapsim" / "LR26_LapSim"
PAPERS = REPO / "docs" / "paper"
ICON = HERE / "perf_lab.ico"
DEFAULT_PORT = 8090
PACKAGES = (("numpy", "numpy"), ("scipy", "scipy"), ("yaml", "pyyaml"))
OPTIONAL = (("casadi", "casadi"),)     # optimal-control lap time models (Methods page); the Lab runs without it
WINDOWS = os.name == "nt"
NO_WINDOW = 0x08000000 if WINDOWS else 0          # CREATE_NO_WINDOW: no black console window for the server


# ---------------------------------------------------------------------- folders and settings
def user_dir() -> Path:
    """Per-user folder for settings and logs (not in the project, so an update never touches it)."""
    base = os.environ.get("APPDATA") if WINDOWS else os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    d = Path(os.environ.get("PERF_LAB_HOME") or Path(base or Path.home()) / "LionsRacing" / "PerformanceLab")
    d.mkdir(parents=True, exist_ok=True)
    return d


def documents_dir() -> Path:
    """Where backups and help reports go: Documents\\Performance Lab (falls back to the user folder)."""
    d = Path(os.environ.get("PERF_LAB_DOCS") or (Path.home() / "Documents" / "Performance Lab"))
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError:
        d = user_dir() / "exports"
        d.mkdir(parents=True, exist_ok=True)
    return d


DEFAULT_SETTINGS = {"port": DEFAULT_PORT, "auto_start": True, "stop_on_close": True}


def load_settings() -> dict:
    s = dict(DEFAULT_SETTINGS)
    try:
        with open(user_dir() / "settings.json", encoding="utf-8") as f:
            d = json.load(f)
        if isinstance(d, dict):
            s.update({k: d[k] for k in DEFAULT_SETTINGS if k in d})
    except (OSError, ValueError):
        pass
    try:
        s["port"] = int(s["port"])
        if not 1024 <= s["port"] <= 65535:
            raise ValueError
    except (TypeError, ValueError):
        s["port"] = DEFAULT_PORT
    return s


def save_settings(s: dict) -> None:
    with open(user_dir() / "settings.json", "w", encoding="utf-8") as f:
        json.dump({k: s.get(k, v) for k, v in DEFAULT_SETTINGS.items()}, f, indent=1)


def log_file() -> Path:
    d = user_dir() / "logs"
    d.mkdir(exist_ok=True)
    return d / "performance_lab.log"


def log(msg: str) -> str:
    line = f"{_dt.datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
    try:
        p = log_file()
        if p.exists() and p.stat().st_size > 2_000_000:            # keep it small
            p.replace(p.with_suffix(".old.log"))
        with open(p, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass
    return line


# ---------------------------------------------------------------------- Python
def python_exe(console: bool = True) -> str:
    """This Python (python.exe for jobs; pythonw.exe is used by the window itself)."""
    exe = sys.executable
    if WINDOWS and console and exe.lower().endswith("pythonw.exe"):
        cand = exe[:-len("pythonw.exe")] + "python.exe"
        if os.path.exists(cand):
            return cand
    return exe


def pythonw_exe() -> str:
    exe = sys.executable
    if WINDOWS and exe.lower().endswith("python.exe"):
        cand = exe[:-len("python.exe")] + "pythonw.exe"
        if os.path.exists(cand):
            return cand
    return exe


def missing_packages(packages=PACKAGES) -> list[str]:
    """pip names of the packages that this Python cannot import (checked in a fresh process).
    Default: the packages the Lab needs; OPTIONAL for the optional ones."""
    code = ("import importlib.util, json\n"
            f"print(json.dumps([p for m, p in {list(packages)!r} if importlib.util.find_spec(m) is None]))")
    try:
        r = subprocess.run([python_exe(), "-c", code], capture_output=True, text=True, timeout=60, creationflags=NO_WINDOW)
        return json.loads(r.stdout.strip().splitlines()[-1])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return [p for _, p in packages]


def pip_install(packages, on_line=None) -> bool:
    """pip install --user <packages>; streams pip's output to on_line. True when they import afterwards."""
    packages = [p for p in packages if re.fullmatch(r"[A-Za-z0-9_.\-]+", p)]
    if not packages:
        return True
    cmd = [python_exe(), "-m", "pip", "install", "--user", "--disable-pip-version-check", *packages]
    if on_line:
        on_line("Installing " + ", ".join(packages) + " (a few minutes the first time) ...")
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, creationflags=NO_WINDOW)
        for line in p.stdout:
            if on_line and line.strip():
                on_line("  " + line.rstrip())
        p.wait()
    except OSError as e:
        if on_line:
            on_line(f"pip could not be started: {e}")
        return False
    known = [x for x in PACKAGES + OPTIONAL if x[1] in packages]
    return not missing_packages(known or PACKAGES)


# ---------------------------------------------------------------------- health check
@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    fix: str = ""                 # what "Repair" does or what the person has to do
    repairable: bool = False
    extra: dict = field(default_factory=dict)
    optional: bool = False        # a failed optional check does not stop the Lab (only a feature is missing)


def self_test(timeout: float = 90.0) -> tuple[bool, str]:
    """Load the LapSim folder and run the four events once (in a separate process: no crash in the window)."""
    code = ("import json, sys, time\n"
            f"sys.path.insert(0, {str(PERF)!r})\n"
            "from perflab.lab import Lab\n"
            "t = time.time(); lab = Lab(); r = lab.raw([])\n"
            "print(json.dumps({'total': r['total'], 'events': {n: e['T'] for n, e in r['events'].items()},"
            " 'seconds': time.time() - t, 'competitions': lab.competitions()}))\n")
    try:
        r = subprocess.run([python_exe(), "-c", code], capture_output=True, text=True, timeout=timeout,
                           cwd=str(PERF), creationflags=NO_WINDOW)
    except subprocess.TimeoutExpired:
        return False, "the test run took too long"
    except OSError as e:
        return False, f"Python could not be started: {e}"
    if r.returncode != 0:
        last = (r.stderr.strip().splitlines() or ["unknown error"])[-1]
        return False, last[:300]
    try:
        d = json.loads(r.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return False, "no result"
    return True, f"{d['total']:.2f} points over {len(d['events'])} events in {d['seconds']:.1f} s"


def health_check(port: int | None = None, run_self_test: bool = True) -> list[Check]:
    port = port or load_settings()["port"]
    out = []
    v = sys.version_info
    out.append(Check("Python", v >= (3, 9), f"version {platform.python_version()}",
                     "Python 3.9 or newer is needed: run perf.bat again, it installs it."))
    miss = missing_packages()
    out.append(Check("Python packages", not miss,
                     "all installed" if not miss else "missing: " + ", ".join(miss),
                     "Repair installs them (internet needed, a few minutes).", repairable=bool(miss), extra={"missing": miss}))
    opt = missing_packages(OPTIONAL)
    out.append(Check("Optional: CasADi (optimal control)", not opt,
                     "installed" if not opt else "not installed: the Methods page's optimal-control models are off",
                     "Repair installs it (internet needed, ~40 MB). Everything else works without it.",
                     repairable=bool(opt), extra={"missing": opt}, optional=True))
    params = LAPSIM / "config" / "lr26_params.m"
    comps = sorted(p.name for p in (LAPSIM / "config").glob("competition_*.m")) if (LAPSIM / "config").is_dir() else []
    env_dir = os.environ.get("FSG_LAPSIM_DIR")
    if env_dir and (Path(env_dir) / "config" / "lr26_params.m").is_file():
        params, comps = Path(env_dir) / "config" / "lr26_params.m", ["(FSG_LAPSIM_DIR)"]
    out.append(Check("Car data (LapSim folder)", params.is_file() and bool(comps),
                     f"found ({len(comps)} competition file(s))" if params.is_file() else f"not found in {LAPSIM}",
                     "Copy the LR26_LapSim folder from the team drive into lapsim\\LR26_LapSim (it is not in git: confidential)."))
    try:
        CONFIGS.mkdir(parents=True, exist_ok=True)
        probe = CONFIGS / ".write_test"
        probe.write_text("x")
        probe.unlink()
        n = len([p for p in CONFIGS.glob("*.json")])
        out.append(Check("Saved setups folder", True, f"{n} file(s), can save"))
    except OSError as e:
        out.append(Check("Saved setups folder", False, f"cannot write to {CONFIGS}: {e}",
                         "Move the project to a folder you can write to (e.g. Documents)."))
    running = is_running(port)
    free = running or port_free(port)
    out.append(Check("Web page port", free, f"{port}, " + ("the Lab is running" if running else "free" if free else "used by another program"),
                     "Settings: choose another port (e.g. 8091)."))
    if run_self_test and not miss and out[3].ok:
        ok, detail = self_test()
        out.append(Check("Test run", ok, detail, "Save a help report and send it to the team's software contact."))
    return out


# ---------------------------------------------------------------------- the web app
def url(port: int) -> str:
    return f"http://localhost:{port}"


def is_running(port: int, timeout: float = 1.0) -> bool:
    try:
        with urllib.request.urlopen(url(port) + "/api/ping", timeout=timeout) as r:
            return json.loads(r.read().decode()).get("app") == "perf-lab"
    except (OSError, ValueError, urllib.error.URLError):
        return False


def port_free(port: int) -> bool:
    import socket  # noqa: PLC0415

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if not WINDOWS:              # as the server does (http.server): a port in TIME_WAIT is free on Linux / macOS
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        s.close()


_proc: subprocess.Popen | None = None


def start_app(port: int, wait: float = 30.0) -> tuple[bool, str]:
    """Start the server in the background (no console window); True when it answers."""
    global _proc
    if is_running(port):
        return True, f"already running at {url(port)}"
    if not port_free(port):
        return False, f"port {port} is used by another program: choose another port in Settings"
    lf = user_dir() / "logs" / "server.log"
    lf.parent.mkdir(exist_ok=True)
    out = open(lf, "a", encoding="utf-8")                    # noqa: SIM115  (kept open by the child)
    out.write(f"\n===== {_dt.datetime.now():%Y-%m-%d %H:%M:%S} start on port {port}\n")
    out.flush()
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONUTF8="1")
    try:
        _proc = subprocess.Popen([python_exe(), str(APP), "--port", str(port), "--no-browser"], cwd=str(PERF),
                                 stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env,
                                 creationflags=NO_WINDOW)
    except OSError as e:
        out.close()
        return False, f"could not start Python: {e}"
    out.close()                                               # the child keeps its own handle
    t0 = time.time()
    while time.time() - t0 < wait:
        if is_running(port, 0.5):
            log(f"started on port {port} (pid {_proc.pid})")
            return True, f"running at {url(port)}"
        if _proc.poll() is not None:
            break
        time.sleep(0.3)
    tail = server_log_tail(8)
    return False, "the Lab did not start. Last lines of its log:\n" + tail


def stop_app(port: int) -> tuple[bool, str]:
    global _proc
    stopped = False
    if is_running(port):
        try:
            req = urllib.request.Request(url(port) + "/api/quit", data=b"{}", method="POST",
                                         headers={"Content-Type": "application/json", "X-Perf-Lab": "1"})
            urllib.request.urlopen(req, timeout=3).read()
            stopped = True
        except (OSError, urllib.error.URLError):
            pass
    if _proc is not None and _proc.poll() is None:
        try:
            _proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _proc.terminate()
        stopped = True
    _proc = None
    for _ in range(20):
        if not is_running(port, 0.3):
            log("stopped")
            return True, "stopped" if stopped else "was not running"
        time.sleep(0.2)
    return False, "it still answers: close it from its own window"


def server_log_tail(n: int = 20) -> str:
    try:
        return "\n".join((user_dir() / "logs" / "server.log").read_text(encoding="utf-8", errors="replace").splitlines()[-n:])
    except OSError:
        return "(no log)"


def open_path(target: str) -> None:
    """A web page, file or folder in the default program."""
    if target.startswith("http"):
        import webbrowser  # noqa: PLC0415

        webbrowser.open(target)
    elif WINDOWS:
        os.startfile(target)  # noqa: S606
    else:
        opener = shutil.which("xdg-open") or shutil.which("open")
        if opener:
            subprocess.Popen([opener, target])


# ---------------------------------------------------------------------- backups
def _team_files() -> list[Path]:
    return sorted(p for p in CONFIGS.glob("*.json") if p.is_file()) if CONFIGS.is_dir() else []


def backup(dest: Path | None = None, include_logs: bool = False) -> Path:
    """Zip the saved setups and efficiency maps (lapsim/configs/*.json) and, optionally, the uploaded logs."""
    dest = Path(dest or documents_dir() / f"PerformanceLab_backup_{_dt.datetime.now():%Y-%m-%d_%H%M}.zip")
    files = _team_files()
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("README.txt", "Performance Lab backup: saved setups and efficiency maps (configs/).\n"
                                 "Restore it with the Performance Lab launcher: Your data -> Restore from a backup.\n")
        for p in files:
            z.write(p, f"configs/{p.name}")
        if include_logs:
            up = PERF / "uploads"
            for p in sorted(up.glob("*")) if up.is_dir() else []:
                if p.is_file():
                    z.write(p, f"uploads/{p.name}")
    log(f"backup: {len(files)} file(s) -> {dest}")
    return dest


def restore(zip_path, overwrite: bool = False) -> dict:
    """Bring the setups of a backup back. Files that exist and differ are kept; the backup's copy is added as
    '<name> (restored).json' unless overwrite. Returns {"added", "same", "renamed", "skipped"}."""
    res = {"added": [], "same": [], "renamed": [], "skipped": []}
    CONFIGS.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            name = info.filename.replace("\\", "/")
            if info.is_dir() or not name.startswith("configs/"):
                continue
            base = os.path.basename(name)
            if not re.fullmatch(r"[\w .()+\-]{1,120}\.json", base):          # no paths, nothing odd
                res["skipped"].append(base)
                continue
            data = z.read(info)
            try:
                json.loads(data.decode("utf-8"))
            except ValueError:
                res["skipped"].append(base)
                continue
            target = CONFIGS / base
            if target.exists():
                if target.read_bytes() == data:
                    res["same"].append(base)
                    continue
                if not overwrite:
                    target = CONFIGS / (base[:-5] + " (restored).json")
                    if base.startswith("eta_map"):
                        target = CONFIGS / (base[:-5] + "_restored.json")
                    d = json.loads(data.decode("utf-8"))
                    if isinstance(d, dict) and "name" in d and not base.startswith("eta_map"):
                        d["name"] = f"{d['name']} (restored)"
                        data = json.dumps(d, indent=2).encode()
                    res["renamed"].append(target.name)
                else:
                    res["added"].append(base)
            else:
                res["added"].append(base)
            target.write_bytes(data)
    log(f"restore {zip_path}: {json.dumps({k: len(v) for k, v in res.items()})}")
    return res


# ---------------------------------------------------------------------- help report
def help_report(checks: list[Check] | None = None, dest: Path | None = None) -> Path:
    """One zip with what someone helping needs: system, health check, launcher and server logs (no car data)."""
    checks = checks if checks is not None else health_check(run_self_test=True)
    dest = Path(dest or documents_dir() / f"PerformanceLab_help_{_dt.datetime.now():%Y-%m-%d_%H%M}.zip")
    lines = [f"# Performance Lab help report, {_dt.datetime.now():%Y-%m-%d %H:%M}", "",
             f"* system: {platform.platform()}", f"* Python: {sys.version.split()[0]} at {python_exe()}",
             f"* project: {REPO}", f"* car data: {LAPSIM}", f"* setups: {CONFIGS}",
             f"* settings: {json.dumps(load_settings())}", "", "## Health check", ""]
    lines += [f"* {'OK ' if c.ok else 'missing (optional)' if c.optional else 'FAILED'} {c.name}: {c.detail}" + (f"  (fix: {c.fix})" if not c.ok and c.fix else "")
              for c in checks]
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("report.md", "\n".join(lines) + "\n")
        for p in sorted((user_dir() / "logs").glob("*.log")):
            z.write(p, f"logs/{p.name}")
    log(f"help report -> {dest}")
    return dest


# ---------------------------------------------------------------------- shortcuts
def _ps_quote(s: str) -> str:
    return "'" + str(s).replace("'", "''") + "'"


def shortcut_script(target: str, args: str, workdir: str, icon: str, places=("Desktop", "Programs")) -> str:
    """PowerShell that creates 'Performance Lab.lnk' on the desktop and in the Start menu."""
    parts = ["$s = New-Object -ComObject WScript.Shell"]
    for place in places:
        parts.append(
            f"$p = Join-Path ([Environment]::GetFolderPath('{place}')) 'Performance Lab.lnk'; "
            f"$l = $s.CreateShortcut($p); $l.TargetPath = {_ps_quote(target)}; $l.Arguments = {_ps_quote(args)}; "
            f"$l.WorkingDirectory = {_ps_quote(workdir)}; $l.IconLocation = {_ps_quote(icon)}; "
            "$l.Description = 'Lions Racing Performance Lab (lap time simulation)'; $l.Save(); Write-Output $p")
    return "; ".join(parts)


def create_shortcuts() -> tuple[bool, str]:
    if not WINDOWS:
        return False, "shortcuts are made on Windows only"
    target, args = pythonw_exe(), f'"{HERE / "perf_launcher.pyw"}"'
    if not target.lower().endswith("pythonw.exe"):          # unusual Python: go through the start script
        target, args = str(PERF / "perf.bat"), ""
    cmd = shortcut_script(target, args, str(PERF), str(ICON))
    try:
        r = subprocess.run(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", cmd],
                           capture_output=True, text=True, timeout=60, creationflags=NO_WINDOW)
    except (OSError, subprocess.SubprocessError) as e:
        return False, f"PowerShell could not be started: {e}"
    if r.returncode != 0:
        return False, (r.stderr.strip() or "failed")[:300]
    log("shortcuts: " + r.stdout.strip().replace("\n", ", "))
    return True, "Shortcuts made on the desktop and in the Start menu (\"Performance Lab\")."
