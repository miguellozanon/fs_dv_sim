"""Open the MATLAB LapSim files, folders and MATLAB results on the PC (dashboard and DV launcher buttons).

Only the targets listed here can be opened (never a path from a request). With WSL they open in Windows:
files with the program Windows uses for them (.m -> MATLAB, .csv -> Excel, .txt -> Notepad), folders in
File Explorer; "matlab" starts the MATLAB desktop in the LR26_LapSim folder. Without Windows, xdg-open.
"""

from __future__ import annotations

import os
import shutil
import subprocess


def _run(cmd, timeout=10) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def win_path(path: str) -> str:
    return _run(["wslpath", "-w", path]) if shutil.which("wslpath") else ""


def _ps(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def targets(lapsim_dir: str, params_file: str = "", competition_file: str = "", runner=None) -> dict:
    """{key: {label, group, path, kind ('file'|'folder'|'matlab'), exists}} - what the buttons can open."""
    root = os.path.dirname(lapsim_dir)
    t = {
        "params": ("Vehicle parameters", "files", params_file or os.path.join(lapsim_dir, "config", "lr26_params.m"), "file"),
        "competition": ("Competition (tracks, scoring)", "files",
                        competition_file or os.path.join(lapsim_dir, "config", "competition_fsa2025.m"), "file"),
        "run_lapsim": ("run_lapsim.m", "files", os.path.join(lapsim_dir, "run_lapsim.m"), "file"),
        "folder": ("LR26_LapSim folder", "files", lapsim_dir, "folder"),
        "configs": ("Saved configurations", "files", os.path.join(root, "configs"), "folder"),
    }
    if runner is not None and getattr(runner, "available", False):
        t["matlab"] = ("Start MATLAB in LR26_LapSim", "files", runner.exe, "matlab")
        st = runner.status()
        if st.get("csv"):
            t["results_csv"] = ("Results table (CSV)", "results", st["csv"], "file")
        if st.get("log"):
            t["matlab_log"] = ("MATLAB output (log)", "results", st["log"], "file")
        if st.get("script"):
            t["job_script"] = ("Job script (.m)", "results", st["script"], "file")
        jd = runner.jobs_dir
        if jd:
            t["results_dir"] = ("MATLAB results folder", "results", jd, "folder")
    return {k: {"label": lab, "group": g, "path": p, "kind": kind, "exists": bool(p and os.path.exists(p)),
                "name": os.path.basename(p) if p else ""}
            for k, (lab, g, p, kind) in t.items()}


def open_target(key: str, tg: dict, lapsim_dir: str) -> tuple[bool, str]:
    """Open one of targets(); (ok, message)."""
    t = tg.get(key)
    if t is None:
        return False, f"nothing to open for '{key}'"
    path = t["path"]
    if not path or not os.path.exists(path):
        return False, f"{t['label']}: not there yet" + (" (run MATLAB first)" if t["group"] == "results" else "")
    windows = shutil.which("powershell.exe") is not None
    if windows:
        wp = win_path(path)
        if not wp:
            return False, "cannot translate the path for Windows (wslpath)"
        if t["kind"] == "folder":
            cmd = ["explorer.exe", wp]
        elif t["kind"] == "matlab":
            cmd = ["powershell.exe", "-NoProfile", "-Command",
                   f"Start-Process -FilePath {_ps(wp)} -ArgumentList '-sd',{_ps(chr(34) + win_path(lapsim_dir) + chr(34))}"]
        else:
            cmd = ["powershell.exe", "-NoProfile", "-Command", f"Start-Process -FilePath {_ps(wp)}"]
        cwd = "/mnt/c" if os.path.isdir("/mnt/c") else None
    elif shutil.which("xdg-open") and t["kind"] != "matlab":
        cmd, cwd = ["xdg-open", path], None
    elif t["kind"] == "matlab":
        cmd, cwd = [path, "-sd", lapsim_dir], None
    else:
        return False, "no way to open files from here (no Windows interop, no xdg-open)"
    try:
        subprocess.Popen(cmd, cwd=cwd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, start_new_session=True)
    except OSError as e:
        return False, f"could not open {t['label']}: {e}"
    what = {"folder": "opening the folder", "matlab": "starting MATLAB (it can take a minute)"}.get(t["kind"], "opening")
    return True, f"{what}: {t['name'] or t['label']}"
