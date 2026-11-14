"""Dashboard "Vehicle setup": try parameter configurations in LapSim, MATLAB and the simulation.

Apply a configuration (a list of changes such as "m=270", "mu*1.05", "autocross.util_y=0.85"):
  1. LapSim (Python port of LR26_LapSim, identical results): all events, times and points, at once.
  2. MATLAB: the same changes in the real LR26_LapSim, in the background (MatlabRunner).
  3. Simulation: the vehicle is swapped in the running simulator (/sim/vehicle_set), which restarts
     the run; the DV result of every finished mission is recorded against the configuration.

The baseline values are read from lapsim/LR26_LapSim (lr26_params.m, competition file) whenever
those files change, so editing them in MATLAB changes the defaults shown here too.
Saved configurations are JSON files in lapsim/configs/ (shared with the team through the repo).
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
import time

from fsdsim.perf.lapsim import EVENT_ORDER, LapSim, find_lapsim_dir, parse_change, split_changes
from fsdsim.perf.schema import LAPSIM_SCHEMA

from .matlab_files import open_target, targets
from .matlab_runner import MatlabRunner
from .rl_manager import RLManager

# (key, label, unit, description): the LapSim groups live in fsdsim.perf.schema (shared with perf/, the
# Performance Lab); the simulator-only group is added here
SCHEMA = LAPSIM_SCHEMA + [
    ("Driverless car only (simulator)", "sim", [
        ("yaw_inertia", "Yaw inertia", "kg m²", "Not in LapSim (quasi-steady state)"),
        ("actuators.max_steer", "Max steering angle", "rad", "DV steering actuator"),
        ("actuators.max_accel", "Max acceleration command", "m/s²", "DV command limit"),
        ("actuators.max_decel", "Max deceleration command", "m/s²", "DV command limit"),
    ]),
]
DV_EVENT = {"acceleration": "acceleration", "skidpad": "skidpad", "autocross": "autocross", "trackdrive": "endurance"}
_NAME_OK = re.compile(r"^[\w .()+-]{1,60}$")


def decode_start_setup(text: str) -> dict | None:
    """The setup a simulation was started with (DV launcher, launch argument vehicle_setup:=...):
    URL-safe base64 of JSON {"name": ..., "changes": [...]} (base64 so ROS does not read it as YAML)."""
    import base64  # noqa: PLC0415

    text = (text or "").strip()
    if not text:
        return None
    try:
        d = json.loads(base64.urlsafe_b64decode(text + "=" * (-len(text) % 4)).decode("utf-8"))
        return {"name": str(d.get("name") or ""), "changes": [str(c) for c in d.get("changes") or []]}
    except (ValueError, TypeError, AttributeError):
        return None


def _f(x):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else round(float(x), 6)


class VehicleSetup:
    def __init__(self, config_dir: str = "", lapsim_dir: str | None = None, configs_dir: str | None = None,
                 send_to_sim=None, on_change=None, log=None, matlab: MatlabRunner | None = None,
                 send_driver_set=None):
        """send_to_sim(changes) -> (ok, message): swap the vehicle in the running simulator.
        send_driver_set(dict) -> (ok, message): reference / RL driver in the running simulator."""
        self.config_dir = config_dir
        self.lapsim_dir = lapsim_dir or find_lapsim_dir(config_dir or None)
        if not self.lapsim_dir:
            raise FileNotFoundError("lapsim/LR26_LapSim not found (set FSG_LAPSIM_DIR)")
        self.configs_dir = configs_dir or os.path.join(os.path.dirname(self.lapsim_dir), "configs")
        self.send_to_sim = send_to_sim
        self.on_change = on_change or (lambda: None)
        self.log = log or (lambda text, kind="info": None)
        self._lock = threading.RLock()
        self.version = 0
        self.ls: LapSim | None = None
        self._mtime = 0.0
        self.baseline: dict = {}
        self.active = {"name": "LapSim baseline", "changes": [], "applied": None}
        self.port: dict = {"state": "idle"}
        self.sim: dict = {"state": "idle", "message": "", "results": {}}
        self._load()
        self.matlab = matlab or MatlabRunner(self.lapsim_dir, self.ls.competition_function, on_change=self._changed)
        self.matlab_baseline: dict | None = None
        repo_root = os.path.dirname(os.path.dirname(self.lapsim_dir))
        self.rl = RLManager(config_dir, repo_root, send_driver_set, on_change=self._changed, log=self.log)

    # ------------------------------------------------------------------ #
    def _changed(self) -> None:
        with self._lock:
            self.version += 1
        self.on_change()

    def _load(self) -> bool:
        """(Re)read the MATLAB files when they changed; recompute the LapSim baseline."""
        ls = LapSim.load(self.lapsim_dir)
        mt = ls.mtime()
        if self.ls is not None and mt == self._mtime:
            return False
        res = ls.run()
        with self._lock:
            self.ls, self._mtime = ls, mt
            self.baseline = self._port_view(res)
        return True

    def _dv_vehicle(self):
        from fsdsim.config import load_all_configs  # noqa: PLC0415

        return load_all_configs(self.config_dir)[0] if self.config_dir else None

    # ------------------------------------------------------------------ #
    @staticmethod
    def _port_view(res: dict) -> dict:
        ev = {n: {"T": _f(r["T"]), "points": _f(r["points"]), "lap": _f(r["t_lap"]), "laps": r["laps"],
                  "v_max_kmh": _f(r["v_max"] * 3.6), "v_avg_kmh": _f(r["v_avg"] * 3.6),
                  "energy_kwh": _f(r["E_total"] / 3.6e6), "track": r["track"]}
              for n, r in res["events"].items()}
        return {"events": ev, "total": _f(res["total"])}

    def schema(self) -> list:
        veh = self.ls.veh
        dv = None
        out = []
        for title, kind, rows in SCHEMA:
            items = []
            for key, label, unit, desc in rows:
                if kind == "lapsim":
                    base = getattr(veh, key)
                elif kind == "event":
                    evn, fld = key.split(".")
                    e = self.ls.comp.events.get(evn)
                    if e is None:
                        continue
                    base = getattr(e, fld)
                else:
                    dv = dv if dv is not None else self._dv_vehicle()
                    if dv is None:
                        continue
                    obj = dv
                    for part in key.split("."):
                        obj = getattr(obj, part)
                    base = obj
                items.append({"key": key, "label": label, "unit": unit, "desc": desc, "base": _f(base),
                              "base_text": "NaN (ideal)" if isinstance(base, float) and math.isnan(base) else None})
            out.append({"title": title, "kind": kind, "items": items})
        return out

    # ------------------------------------------------------------------ #
    def validate(self, changes) -> list[str]:
        out = []
        for c in changes or []:
            c = str(c).strip()
            if not c:
                continue
            parse_change(c)            # raises ValueError on bad syntax
            out.append(c)
        veh_ch, _ = split_changes(out, tuple(self.ls.comp.events))
        self.ls.vehicle(out)          # unknown LapSim name -> KeyError
        self.ls.competition(out)
        if self.config_dir:
            from fsdsim.config import load_all_configs  # noqa: PLC0415

            load_all_configs(self.config_dir, vehicle_overrides=veh_ch)   # simulator-only names
        return out

    def apply(self, changes, name: str = "", run_matlab: bool = True, send_sim: bool = True) -> tuple[bool, str]:
        try:
            self._load()
            changes = self.validate(changes)
        except Exception as e:  # noqa: BLE001
            return False, f"{type(e).__name__}: {e}"
        label = name.strip() or ("LapSim baseline" if not changes else "custom")
        with self._lock:
            self.active = {"name": label, "changes": changes, "applied": time.time()}
            self.port = {"state": "running", "changes": changes}
            self.sim["results"] = {}
        self._changed()
        threading.Thread(target=self._run_port, args=(changes,), daemon=True).start()
        msgs = []
        if run_matlab:
            if self.matlab.available:
                self.matlab.submit(changes, label)
                msgs.append("MATLAB started")
            else:
                msgs.append("MATLAB not available")
        if send_sim and self.send_to_sim is not None:
            veh_ch, _ = split_changes(changes, tuple(self.ls.comp.events))
            with self._lock:
                self.sim.update(state="sending", message="", changes=veh_ch, sent=time.time())
            ok, text = self.send_to_sim(veh_ch)
            with self._lock:
                if not ok:
                    self.sim.update(state="error", message=text)
                elif self.sim.get("state") == "sending":       # (in-process simulators confirm at once)
                    self.sim.update(state="sent", message=text)
            msgs.append(f"simulation: {text}")
        self.log(f"Vehicle setup '{label}' applied ({' '.join(changes) or 'baseline'})", "command")
        self._changed()
        return True, "; ".join(msgs) or "applied"

    def _run_port(self, changes) -> None:
        t0 = time.time()
        try:
            res = self._port_view(self.ls.run(changes))
            with self._lock:
                if self.port.get("changes") == changes:
                    self.port = {"state": "done", "changes": changes, **res, "seconds": round(time.time() - t0, 2)}
        except Exception as e:  # noqa: BLE001
            with self._lock:
                self.port = {"state": "error", "changes": changes, "message": f"{type(e).__name__}: {e}"}
        self._changed()

    def reload(self) -> tuple[bool, str]:
        try:
            changed = self._load()
        except Exception as e:  # noqa: BLE001
            return False, f"{type(e).__name__}: {e}"
        self._changed()
        return True, "LapSim files re-read" + ("" if changed else " (unchanged)")

    def set_start(self, name: str, changes) -> None:
        """The simulation was started with this setup (DV launcher): it is the active one, the page's form
        shows it and LapSim computes it. The simulator already runs it (vehicle_set), nothing is sent."""
        try:
            changes = self.validate(changes)
        except Exception as e:  # noqa: BLE001
            self.log(f"start setup not valid ({e}); showing it anyway", "warning")
            changes = [str(c) for c in changes or []]
        label = name.strip() or ("LapSim baseline" if not changes else "set at start")
        with self._lock:
            self.active = {"name": label, "changes": changes, "applied": time.time(), "start": True}
            self.port = {"state": "running", "changes": changes}
            self.sim.update(state="active", message="the simulation was started with this vehicle")
        threading.Thread(target=self._run_port, args=(changes,), daemon=True).start()
        self.log(f"Vehicle setup '{label}' from the start ({' '.join(changes) or 'baseline'})", "info")
        self._changed()

    # ------------------------------------------------------------------ #
    # simulator feedback
    # ------------------------------------------------------------------ #
    def on_vehicle(self, info: dict) -> None:
        """/sim/vehicle arrived: the simulator now runs these changes."""
        start_setup = None
        with self._lock:
            ch = list(info.get("changes") or [])
            if self.sim.get("state") in ("sending", "sent"):
                self.sim.update(state="active", message="the simulator runs this vehicle (run restarted)",
                                active=time.time())
            elif self.active.get("applied") is None and ch:
                # the simulation was started with a setup (launcher / vehicle_set:=): show it as the active one
                start_setup = ch
                self.active = {"name": "set at start", "changes": ch, "applied": time.time()}
                self.port = {"state": "running", "changes": ch}
                self.sim.update(state="active", message="the simulation was started with this vehicle")
            self.sim["running_changes"] = ch
        if start_setup is not None:
            threading.Thread(target=self._run_port, args=(start_setup,), daemon=True).start()
        self._changed()

    def on_mission_complete(self, mission: str, laps: list, penalty: float, mission_time: float) -> None:
        """Record the DV result of a finished mission for the active configuration."""
        laps = [float(x) for x in laps if x]
        if not laps:
            return
        if mission == "skidpad":
            t = 0.5 * (laps[1] + laps[3]) if len(laps) >= 4 else sum(laps) / len(laps)   # 2nd right, 2nd left
        elif mission == "trackdrive":
            t = sum(laps)
        else:
            t = laps[0]
        with self._lock:
            self.sim.setdefault("results", {})[mission] = {
                "T": _f(t), "penalty": _f(penalty), "laps": [_f(x) for x in laps], "mission_time": _f(mission_time),
                "lapsim_event": DV_EVENT.get(mission), "at": time.time(),
                "driver": self.rl.driver.get("longitudinal") or "reference"}
        self._changed()

    # ------------------------------------------------------------------ #
    # saved configurations
    # ------------------------------------------------------------------ #
    def _cfg_path(self, name: str) -> str:
        if not _NAME_OK.match(name or "") or name.strip(" .") != name.strip():
            raise ValueError("configuration names may use letters, digits, spaces and . _ - ( ) +")
        return os.path.join(self.configs_dir, re.sub(r"\s+", "_", name.strip()) + ".json")

    def configs(self) -> list[dict]:
        out = []
        if os.path.isdir(self.configs_dir):
            for f in sorted(os.listdir(self.configs_dir)):
                if f.endswith(".json"):
                    try:
                        with open(os.path.join(self.configs_dir, f), encoding="utf-8") as fh:
                            d = json.load(fh)
                        out.append({"name": str(d.get("name", f[:-5])), "changes": [str(c) for c in d.get("changes", [])],
                                    "notes": str(d.get("notes", "")), "saved": d.get("saved")})
                    except (OSError, ValueError):
                        continue
        return out

    def save(self, name: str, changes, notes: str = "") -> tuple[bool, str]:
        try:
            path = self._cfg_path(name)
            changes = self.validate(changes)
        except Exception as e:  # noqa: BLE001
            return False, f"{type(e).__name__}: {e}"
        os.makedirs(self.configs_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"name": name.strip(), "changes": changes, "notes": str(notes)[:2000],
                       "saved": time.strftime("%Y-%m-%d %H:%M")}, f, indent=2)
        self._changed()
        return True, f"saved '{name.strip()}' ({os.path.relpath(path, os.path.dirname(self.configs_dir))})"

    def delete(self, name: str) -> tuple[bool, str]:
        try:
            path = self._cfg_path(name)
        except ValueError as e:
            return False, str(e)
        if not os.path.exists(path):
            return False, f"no configuration '{name}'"
        os.remove(path)
        self._changed()
        return True, f"deleted '{name}'"

    # ------------------------------------------------------------------ #
    def state(self) -> dict:
        try:
            self._load()
        except Exception:  # noqa: BLE001
            pass
        with self._lock:
            ls = self.ls
            return {
                "version": self.version,
                "lapsim": {"dir": ls.root, "params_file": ls.params_file, "competition_file": ls.competition_file,
                           "competition": ls.comp.name, "modified": time.strftime("%Y-%m-%d %H:%M:%S",
                                                                                 time.localtime(self._mtime)),
                           "events": list(ls.comp.events)},
                "schema": self.schema(),
                "baseline": self.baseline,
                "active": dict(self.active),
                "port": dict(self.port),
                "matlab": self.matlab.status(),
                "sim": json.loads(json.dumps(self.sim, default=str)),
                "configs": self.configs(),
                "files": self.files(),
                "dv_events": DV_EVENT,
                "rl": self.rl.state(self._veh_changes(self.active["changes"])),
            }

    def _veh_changes(self, changes) -> list:
        try:
            return split_changes(changes, tuple(self.ls.comp.events))[0]
        except Exception:  # noqa: BLE001
            return []

    def files(self) -> dict:
        """MATLAB files, folders and the last MATLAB results the page can open (matlab_files.targets)."""
        ls = self.ls
        return targets(self.lapsim_dir, getattr(ls, "params_file", ""), getattr(ls, "competition_file", ""), self.matlab)

    def open(self, key: str) -> tuple[bool, str]:
        return open_target(str(key), self.files(), self.lapsim_dir)

    def handle(self, req: dict) -> tuple[bool, str]:
        action = req.get("action")
        if action == "open":
            return self.open(str(req.get("target") or ""))
        if action == "apply":
            return self.apply(req.get("changes") or [], str(req.get("name") or ""),
                              run_matlab=bool(req.get("matlab", True)), send_sim=bool(req.get("sim", True)))
        if action == "save":
            return self.save(str(req.get("name") or ""), req.get("changes") or [], str(req.get("notes") or ""))
        if action == "delete":
            return self.delete(str(req.get("name") or ""))
        if action == "reload":
            return self.reload()
        if action == "cancel_matlab":
            self.matlab.cancel()
            return True, "MATLAB job cancelled"
        if action == "rl_train":
            name = str(req.get("name") or "")
            if not name:
                a = self.active
                name = a["name"] if a.get("applied") and a["name"] not in ("custom", "LapSim baseline") else ""
            from fsdsim.rl import registry  # noqa: PLC0415

            veh_ch = self._veh_changes(self.active["changes"])
            return self.rl.train(veh_ch, name or registry.default_name(veh_ch), int(req.get("generations") or 50),
                                 req.get("workers"))
        if action == "rl_stop":
            return self.rl.stop()
        if action == "driver":
            return self.rl.set_driver(str(req.get("longitudinal") or "reference"), str(req.get("policy") or "auto"))
        if action == "validate":
            try:
                return True, " ".join(self.validate(req.get("changes") or [])) or "no changes"
            except Exception as e:  # noqa: BLE001
                return False, f"{type(e).__name__}: {e}"
        return False, f"unknown action '{action}'"
