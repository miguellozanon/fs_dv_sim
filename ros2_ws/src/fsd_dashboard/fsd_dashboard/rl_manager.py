"""Dashboard side of the RL driver: train a policy for the applied vehicle setup, list policies,
switch the simulator's driver between the reference driver and the RL driver. A universal policy
(tools/train_rl_universal.py, GPU / Isaac Lab) drives every setup inside its training range, so a
new setup can be driven by the RL driver without training.

Training runs tools/train_rl.py as a background process (NumPy only, all CPU cores but one by
default), with a progress file the page polls through /api/setup.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time

from fsdsim.rl import registry


def _summary(meta: dict) -> dict:
    ev = []
    for e in meta.get("evaluation") or []:
        ref, rl = e.get("reference") or {}, e.get("rl") or {}
        ev.append({"mission": e.get("mission"), "seed": e.get("seed"),
                   "ref": ref.get("raw_time"), "rl": rl.get("raw_time"),
                   "ref_pen": (ref.get("cones") or 0) + (ref.get("off_course") or 0),
                   "rl_pen": (rl.get("cones") or 0) + (rl.get("off_course") or 0), "rl_state": rl.get("state")})
    v = meta.get("validation") or {}
    uni = bool(meta.get("universal"))
    if uni and not ev:   # no official runs stored: validation runs of the baseline car (spec 0) vs the reference law
        for r in v.get("runs") or []:
            if r.get("spec", 0) == 0 and "ref_time" in r:
                ev.append({"mission": r.get("mission"), "seed": r.get("seed"), "ref": r.get("ref_time"),
                           "rl": None if r.get("failed") else r.get("time"), "ref_pen": r.get("ref_cones") or 0,
                           "rl_pen": r.get("cones") or 0, "rl_state": "FAILED" if r.get("failed") else "FINISHED"})
    return {"name": meta.get("dir_name") or meta.get("name"), "trained": meta.get("trained"),
            "changes": meta.get("changes", []), "spec_hash": meta.get("spec_hash"),
            "train_seconds": meta.get("train_seconds"), "validation": {k: v.get(k) for k in ("gain", "score", "ref_score",
                                                                                            "cones", "ref_cones", "failed")
                                                                       if k in v} if uni else v,
            "evaluation": ev, "algorithm": meta.get("algorithm"), "universal": uni,
            "ranges": _ranges_text(meta) if uni else ""}


def _ranges_text(meta: dict) -> str:
    """Short description of a universal policy's training ranges."""
    b, r = meta.get("base_spec") or {}, meta.get("spec_ranges") or {}
    out = []
    for name, unit, k in (("m", "kg", 1), ("c_z", "", 1), ("c_w", "", 1), ("h_cog", "m", 1), ("P_max", "kW", 1e-3)):
        if name in r and name in b:
            kind, lo, hi = r[name][:3]
            lo, hi = (b[name] * lo, b[name] * hi) if kind == "scale" else (lo, hi)
            out.append(f"{name} {lo * k:.3g}–{hi * k:.3g}{(' ' + unit) if unit else ''}")
    if "mu" in r:
        out.append(f"mu ×{r['mu'][1]:.2f}–{r['mu'][2]:.2f}")
    if "weight_front" in r:
        out.append(f"front weight {100 * r['weight_front'][1]:.0f}–{100 * r['weight_front'][2]:.0f} %")
    if "aero_bal_f" in r:
        out.append(f"aero balance {r['aero_bal_f'][1]:.2f}–{r['aero_bal_f'][2]:.2f}")
    if "drive_awd" in r:
        out.append("RWD / AWD")
    return ", ".join(out)


class RLManager:
    def __init__(self, config_dir: str, repo_root: str, send_driver_set=None, on_change=None, log=None):
        self.config_dir = config_dir
        self.repo_root = repo_root
        self.send_driver_set = send_driver_set
        self.on_change = on_change or (lambda: None)
        self.log = log or (lambda text, kind="info": None)
        self.script = os.path.join(repo_root, "tools", "train_rl.py")
        self.work = os.path.join(os.path.expanduser("~"), ".cache", "fsg-dv-sim", "rl")
        self._lock = threading.Lock()
        self.job: dict = {"state": "idle"}
        self._proc: subprocess.Popen | None = None
        self.driver: dict = {"requested": "reference", "longitudinal": "reference", "description": ""}
        self._spec_cache: dict = {}
        self._veh_cache: dict = {}
        self.extra_args: list = []     # extra train_rl.py arguments (tests: --pairs 1 --missions acceleration)

    # ------------------------------------------------------------------ #
    def spec_of(self, veh_changes) -> str | None:
        key = tuple(veh_changes)
        if key not in self._spec_cache:
            try:
                from fsdsim.config import load_all_configs  # noqa: PLC0415
                from fsdsim.rl.es import spec_hash  # noqa: PLC0415

                self._spec_cache[key] = spec_hash(load_all_configs(self.config_dir, vehicle_overrides=list(key))[0])
            except Exception:  # noqa: BLE001
                self._spec_cache[key] = None
        return self._spec_cache[key]

    def vehicle_of(self, veh_changes):
        key = tuple(veh_changes)
        if key not in self._veh_cache:
            try:
                from fsdsim.config import load_all_configs  # noqa: PLC0415

                self._veh_cache[key] = load_all_configs(self.config_dir, vehicle_overrides=list(key))[0]
            except Exception:  # noqa: BLE001
                self._veh_cache[key] = None
        return self._veh_cache[key]

    def state(self, veh_changes) -> dict:
        pols = [_summary(p) for p in registry.list_policies()]
        spec = self.spec_of(veh_changes)
        match = next((p["name"] for p in sorted(pols, key=lambda p: p.get("trained") or "", reverse=True)
                      if spec and p["spec_hash"] == spec), None)
        # what 'auto' drives this setup with: a policy trained for it, else a universal policy covering it
        universal, uni_why = None, ""
        veh = self.vehicle_of(veh_changes)
        if veh is not None and any(p.get("universal") for p in pols):
            try:
                u, why = registry.find_universal(veh)
                universal = u.get("dir_name") if u else None
                uni_why = "; ".join(why)
            except Exception as e:  # noqa: BLE001
                uni_why = str(e)
        with self._lock:
            job = dict(self.job)
        prog = job.get("progress_file")
        if prog and os.path.exists(prog):
            try:
                with open(prog, encoding="utf-8") as f:
                    job["progress"] = json.load(f)
            except (OSError, ValueError):
                pass
        log_p = job.get("log_file")
        if log_p and os.path.exists(log_p):
            with open(log_p, encoding="utf-8", errors="replace") as f:
                job["log_tail"] = "".join(f.readlines()[-6:])
        return {"policies": pols, "spec": spec, "match": match, "universal": universal, "universal_why": uni_why,
                "job": job, "driver": dict(self.driver),
                "available": os.path.isfile(self.script), "cpus": os.cpu_count() or 1}

    # ------------------------------------------------------------------ #
    def train(self, veh_changes, name: str, generations: int = 50, workers: int | None = None) -> tuple[bool, str]:
        if not os.path.isfile(self.script):
            return False, f"training script not found: {self.script}"
        with self._lock:
            if self.job.get("state") == "running":
                return False, f"a training run is already going ({self.job.get('name')})"
        name = re.sub(r"[^\w.+-]+", "_", name.strip()).strip("_")[:60] or registry.default_name(veh_changes)
        generations = max(1, min(int(generations), 1000))
        workers = int(workers or max(1, (os.cpu_count() or 2) - 1))
        os.makedirs(self.work, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        prog = os.path.join(self.work, f"{name}_{stamp}.progress.json")
        log_p = os.path.join(self.work, f"{name}_{stamp}.log")
        cmd = [sys.executable, self.script, "--name", name, "--generations", str(generations), "--workers", str(workers),
               "--config", self.config_dir, "--progress", prog]
        cmd += list(self.extra_args)
        if veh_changes:
            cmd += ["--set", *veh_changes]
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        with open(log_p, "w", encoding="utf-8") as log:
            self._proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=self.repo_root, env=env)
        with self._lock:
            self.job = {"state": "running", "name": name, "changes": list(veh_changes), "generations": generations,
                        "workers": workers, "started": time.time(), "progress_file": prog, "log_file": log_p}
        threading.Thread(target=self._wait, args=(self._proc,), daemon=True).start()
        self.log(f"RL training started: '{name}' ({generations} generations, {workers} workers)", "command")
        self.on_change()
        return True, f"training '{name}' started ({generations} generations on {workers} CPU workers)"

    def _wait(self, proc: subprocess.Popen) -> None:
        rc = proc.wait()
        with self._lock:
            stopped = os.path.exists(self.job.get("progress_file", "") + ".stop")
            self.job.update(state="done" if rc == 0 else "error", finished=time.time(), returncode=rc,
                            stopped=stopped)
            name = self.job.get("name")
        self.log(f"RL training '{name}' {'finished' if rc == 0 else 'failed'}" + (" (stopped early)" if stopped else ""),
                 "lap" if rc == 0 else "warning")
        if rc == 0 and self.driver.get("requested") == "rl" and self.send_driver_set:
            self.send_driver_set({"longitudinal": "rl", "policy": "auto"})   # pick up the new policy
        self.on_change()

    def stop(self) -> tuple[bool, str]:
        with self._lock:
            prog = self.job.get("progress_file")
            running = self.job.get("state") == "running"
        if not running or not prog:
            return False, "no training is running"
        open(prog + ".stop", "w").close()      # the trainer stops after this generation and saves the best policy
        return True, "stopping after the current generation (the best policy so far is kept)"

    # ------------------------------------------------------------------ #
    def set_driver(self, longitudinal: str, policy: str = "auto") -> tuple[bool, str]:
        if longitudinal not in ("reference", "rl"):
            return False, f"unknown driver '{longitudinal}'"
        if not re.fullmatch(r"[\w.+-]{1,60}|auto", policy or "auto"):
            return False, "invalid policy name"
        self.driver["requested"] = longitudinal
        self.driver["policy_requested"] = policy or "auto"
        if self.send_driver_set is None:
            return False, "no simulator connection"
        ok, msg = self.send_driver_set({"longitudinal": longitudinal, "policy": policy or "auto"})
        self.on_change()
        return ok, msg

    def on_driver(self, info: dict) -> None:
        """/sim/driver from the driver node (or the offline demo)."""
        self.driver.update({k: info.get(k) for k in ("longitudinal", "description", "policy")})
        if info.get("requested"):
            self.driver["requested"] = info["requested"]
        self.on_change()
