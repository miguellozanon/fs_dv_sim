"""Vehicle setup in the launcher: edit the car before the simulation starts.

The same parameters as the dashboard's "Vehicle setup" (documentation Table 3.1 + simulator extras,
fsd_dashboard.vehicle_setup.SCHEMA), read from lapsim/LR26_LapSim. For a list of changes
("m=270", "c_z*1.05", "autocross.util_y=0.85") it returns:

  * whether every change is valid (LapSim names, simulator names, syntax),
  * the LapSim prediction (Python port, the same numbers as LR26_LapSim): time and points per event,
  * which changes go to the simulator (vehicle changes) and which only affect LapSim (event changes),
  * which universal RL policy covers this car (or why none does).

Saved configurations are the dashboard's: JSON files in lapsim/configs/.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "ros2_ws" / "src"
for _p in (SRC / "fsdsim", SRC / "fsd_dashboard"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

CONFIG = SRC / "fsd_sim_bringup" / "config"
LAPSIM = REPO / "lapsim" / "LR26_LapSim"
CONFIGS = REPO / "lapsim" / "configs"
POLICIES = REPO / "rl" / "policies"
NAME_OK = re.compile(r"^[\w .()+-]{1,60}$")


def _f(x):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else round(float(x), 6)


class VehicleEditor:
    def __init__(self, lapsim_dir: Path = LAPSIM, config_dir: Path = CONFIG, configs_dir: Path = CONFIGS,
                 policies_dir: Path = POLICIES):
        self.lapsim_dir, self.config_dir = Path(lapsim_dir), Path(config_dir)
        self.configs_dir, self.policies_dir = Path(configs_dir), Path(policies_dir)
        self._lock = threading.Lock()
        self._ls = None
        self._mtime = None
        self._base: dict = {}
        self._schema: list = []

    # ------------------------------------------------------------------ LapSim
    def _load(self):
        """(Re)read lr26_params.m and the competition file when they changed."""
        from fsdsim.perf.lapsim import LapSim  # noqa: PLC0415

        with self._lock:
            ls = LapSim.load(str(self.lapsim_dir))
            mt = ls.mtime()
            if self._ls is None or mt != self._mtime:
                self._ls, self._mtime = ls, mt
                self._base = self._view(ls.run())
                self._schema = self._build_schema(ls)
            return self._ls

    @staticmethod
    def _view(res: dict) -> dict:
        ev = {n: {"T": _f(r["T"]), "points": _f(r["points"]), "laps": r["laps"], "v_max_kmh": _f(r["v_max"] * 3.6)}
              for n, r in res["events"].items()}
        return {"events": ev, "total": _f(res["total"])}

    def _dv_vehicle(self, changes=()):
        from fsdsim.config import load_all_configs  # noqa: PLC0415

        return load_all_configs(str(self.config_dir), vehicle_overrides=list(changes))[0]

    def _build_schema(self, ls) -> list:
        from fsd_dashboard.vehicle_setup import SCHEMA  # noqa: PLC0415

        dv = None
        out = []
        for title, kind, rows in SCHEMA:
            items = []
            for key, label, unit, desc in rows:
                if kind == "lapsim":
                    base = getattr(ls.veh, key)
                elif kind == "event":
                    evn, fld = key.split(".")
                    e = ls.comp.events.get(evn)
                    if e is None:
                        continue
                    base = getattr(e, fld)
                else:
                    try:
                        dv = dv if dv is not None else self._dv_vehicle()
                    except Exception:  # noqa: BLE001 - simulator config not readable: LapSim rows only
                        continue
                    obj = dv
                    for part in key.split("."):
                        obj = getattr(obj, part)
                    base = obj
                nan = isinstance(base, float) and math.isnan(base)
                items.append({"key": key, "label": label, "unit": unit, "desc": desc, "base": _f(base),
                              "base_text": "NaN (ideal)" if nan else None})
            out.append({"title": title, "kind": kind, "items": items})
        return out

    # ------------------------------------------------------------------ API
    def info(self) -> dict:
        ls = self._load()
        return {"schema": self._schema, "baseline": self._base, "events": list(ls.comp.events),
                "competition": ls.comp.name, "params_file": os.path.basename(ls.params_file),
                "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(self._mtime or 0)),
                "configs": self.configs()}

    def check(self, changes, longitudinal: str = "reference", policy: str = "auto") -> dict:
        """Validate the changes and predict the car with LapSim; never raises."""
        from fsdsim.perf.lapsim import parse_change, split_changes  # noqa: PLC0415

        out = {"ok": False, "changes": [], "sim_changes": [], "lapsim_only": [], "errors": []}
        try:
            ls = self._load()
        except Exception as e:  # noqa: BLE001
            out["errors"].append(f"LapSim not readable: {type(e).__name__}: {e}")
            return out
        from fsdsim.perf.lapsim import EVENT_FIELDS  # noqa: PLC0415

        good = []
        for c in [str(c).strip() for c in changes or [] if str(c).strip()]:
            name = re.split(r"[=*+-]", c, maxsplit=1)[0] or c
            try:
                pname, _, _ = parse_change(c)
            except Exception:  # noqa: BLE001
                out["errors"].append(f"'{c}': cannot read it (write name=value, name*factor, name+delta or name-delta)")
                continue
            head, _, fld = pname.partition(".")
            try:
                if head in ls.comp.events and fld in EVENT_FIELDS:
                    ls.competition([c])          # LapSim driver model of one event
                else:
                    self._dv_vehicle([c])        # LapSim and simulator vehicle parameters
                good.append(c)
            except (KeyError, AttributeError):
                out["errors"].append(f"{pname}: unknown parameter (use the names in the table)")
            except Exception as e:  # noqa: BLE001
                out["errors"].append(f"{name}: {_msg(e)}")
        veh_ch, ev_ch = split_changes(good, tuple(ls.comp.events))
        out.update(changes=good, sim_changes=veh_ch, lapsim_only=ev_ch)
        try:
            veh = self._dv_vehicle(veh_ch)
        except Exception as e:  # noqa: BLE001
            out["errors"].append(f"simulator: {_msg(e)}")
            veh = None
        try:
            out["lapsim"] = self._view(ls.run(good))
        except Exception as e:  # noqa: BLE001
            out["errors"].append(f"LapSim: {_msg(e)}")
        out["baseline"] = self._base
        if veh is not None:
            out["rl"] = self._rl(veh, longitudinal, policy)
        out["ok"] = not out["errors"]
        return out

    def _rl(self, veh, longitudinal: str, policy: str) -> dict:
        from fsdsim.rl import registry  # noqa: PLC0415

        folder = str(self.policies_dir)
        try:
            exact = registry.find_for_vehicle(veh, folder)
        except Exception:  # noqa: BLE001
            exact = None
        uni, why = registry.find_universal(veh, folder)
        r = {"exact": exact["dir_name"] if exact else None, "universal": uni["dir_name"] if uni else None,
             "why_not": why}
        if policy not in ("", "auto", "universal"):
            p = next((q for q in registry.list_policies(folder) if q["dir_name"] == policy), None)
            if p is None:
                r["chosen_problem"] = f"policy '{policy}' not found"
            elif p.get("universal"):
                reasons = registry.universal_coverage(p, veh)
                if reasons:
                    r["chosen_problem"] = f"'{policy}' does not cover this car: " + "; ".join(reasons)
        return r

    # ------------------------------------------------------------------ MATLAB (the real LR26_LapSim)
    @property
    def runner(self):
        """The dashboard's MATLAB runner, with its own work folder ('launcher')."""
        if getattr(self, "_runner", None) is None:
            from fsd_dashboard.matlab_runner import MatlabRunner  # noqa: PLC0415

            ls = self._load()
            self._runner = MatlabRunner(str(self.lapsim_dir), ls.competition_function, instance="launcher")
        return self._runner

    def files(self) -> dict:
        from fsd_dashboard.matlab_files import targets  # noqa: PLC0415

        ls = self._load()
        return targets(str(self.lapsim_dir), ls.params_file, ls.competition_file, self.runner)

    def open(self, key: str) -> tuple[bool, str]:
        from fsd_dashboard.matlab_files import open_target  # noqa: PLC0415

        return open_target(str(key), self.files(), str(self.lapsim_dir))

    def matlab_state(self) -> dict:
        r = self.runner
        st = r.status()
        st.pop("matlab", None)
        return {"available": r.available, "exe": r.exe or "", **st, "files": self.files()}

    def matlab_run(self, changes) -> tuple[bool, str]:
        r = self.runner
        if not r.available:
            return False, r.status().get("message") or "MATLAB not found"
        res = self.check(changes)
        if res["errors"]:
            return False, "fix the setup first: " + "; ".join(res["errors"])
        job = r.submit(res["changes"], "launcher")
        return True, f"MATLAB job {job} started ({' '.join(res['changes']) or 'baseline'}); it takes a minute or two"

    def matlab_cancel(self) -> tuple[bool, str]:
        self.runner.cancel()
        return True, "MATLAB job cancelled"

    # ------------------------------------------------------------------ saved configurations
    def configs(self) -> list[dict]:
        out = []
        if self.configs_dir.is_dir():
            for f in sorted(self.configs_dir.glob("*.json")):
                try:
                    d = json.loads(f.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                out.append({"name": str(d.get("name") or f.stem), "changes": [str(c) for c in d.get("changes", [])],
                            "notes": str(d.get("notes", "")), "saved": d.get("saved")})
        return out

    def _path(self, name: str) -> Path:
        name = (name or "").strip()
        if not NAME_OK.match(name) or name.strip(" .") != name:
            raise ValueError("configuration names may use letters, digits, spaces and . _ - ( ) +")
        return self.configs_dir / (re.sub(r"\s+", "_", name) + ".json")

    def save(self, name: str, changes, notes: str = "") -> tuple[bool, str]:
        try:
            path = self._path(name)
        except ValueError as e:
            return False, str(e)
        res = self.check(changes)
        if res["errors"]:
            return False, "not saved: " + "; ".join(res["errors"])
        self.configs_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"name": name.strip(), "changes": res["changes"], "notes": str(notes)[:2000],
                                    "saved": time.strftime("%Y-%m-%d %H:%M")}, indent=2), encoding="utf-8")
        return True, f"saved '{name.strip()}' (lapsim/configs/{path.name})"

    def delete(self, name: str) -> tuple[bool, str]:
        try:
            path = self._path(name)
        except ValueError as e:
            return False, str(e)
        if not path.exists():
            return False, f"no configuration '{name}'"
        path.unlink()
        return True, f"deleted '{name}'"


def _msg(e: BaseException) -> str:
    s = str(e).strip("'\"")
    return s if s else type(e).__name__
