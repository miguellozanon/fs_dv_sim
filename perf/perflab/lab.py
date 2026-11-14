"""The lab: LapSim folder, competitions, saved setups, efficiency maps and one full run with its traces.

    lab = Lab()                                  # finds lapsim/LR26_LapSim
    r = lab.run(["m-10", "autocross.util_y=0.85"], competition="competition_fsa2025.m")
    r["total"], r["events"]["autocross"]["T"]

Setups are the dashboard's saved configurations (lapsim/configs/<name>.json: name, changes, notes), so a setup
saved here appears on the dashboard's Vehicle setup page and the other way round.
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
import time

import numpy as np

from . import REPO  # noqa: F401  (sets sys.path)
from fsdsim.perf import efficiency  # noqa: E402
from fsdsim.perf.lapsim import LapSim, find_lapsim_dir, parse_change, split_changes  # noqa: E402
from fsdsim.perf.params import ALIASES, PerfParams  # noqa: E402
from fsdsim.perf.schema import LAPSIM_SCHEMA  # noqa: E402

J_PER_KWH = 3.6e6
NAME_OK = re.compile(r"^[\w .()+-]{1,60}$")
# simulator-only parameters: used by the transient lap, ignored by the QSS LapSim
TRANSIENT_KEYS = ("yaw_inertia", "actuators.max_steer", "actuators.max_accel", "actuators.max_decel")
MAX_TRACE = 1500                  # points per trace sent to the browser


def clean(x):
    """JSON-safe: numpy -> lists, NaN / inf -> None."""
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, (np.floating, float)):
        f = float(x)
        return f if math.isfinite(f) else None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def decimate(n: int, limit: int = MAX_TRACE) -> slice:
    return slice(None, None, max(1, int(math.ceil(n / limit))))


class Lab:
    def __init__(self, lapsim_dir: str | None = None, configs_dir: str | None = None):
        self.lapsim_dir = lapsim_dir or find_lapsim_dir(REPO)
        if not self.lapsim_dir:
            raise FileNotFoundError("LR26_LapSim not found: put it in lapsim/LR26_LapSim or set FSG_LAPSIM_DIR")
        self.configs_dir = configs_dir or os.path.join(os.path.dirname(self.lapsim_dir), "configs")
        self.lock = threading.RLock()          # LapSim runs one at a time (the efficiency map is global)
        self._lapsims: dict[str, tuple[float, LapSim]] = {}
        self.eta_map = ""                      # "" = constant eta_motor

    # ------------------------------------------------------------------ #
    # files
    # ------------------------------------------------------------------ #
    def competitions(self) -> list[str]:
        d = os.path.join(self.lapsim_dir, "config")
        return sorted(f for f in os.listdir(d) if re.match(r"competition_.*\.m$", f))

    def lapsim(self, competition: str | None = None) -> LapSim:
        comps = self.competitions()
        competition = competition or comps[0]
        if competition not in comps:
            raise ValueError(f"unknown competition {competition!r} (have: {', '.join(comps)})")
        path = os.path.join(self.lapsim_dir, "config", competition)
        params = os.path.join(self.lapsim_dir, "config", "lr26_params.m")
        mtime = max(os.path.getmtime(path), os.path.getmtime(params))
        hit = self._lapsims.get(competition)
        if hit is None or hit[0] != mtime:          # re-read when the MATLAB files change
            hit = (mtime, LapSim.load(self.lapsim_dir, competition=path))
            self._lapsims[competition] = hit
        return hit[1]

    def eta_maps(self) -> list[str]:
        if not os.path.isdir(self.configs_dir):
            return []
        return sorted(f for f in os.listdir(self.configs_dir) if f.startswith("eta_map") and f.endswith(".json"))

    def set_eta_map(self, name: str) -> None:
        if name and name not in self.eta_maps():
            raise ValueError(f"no efficiency map {name!r} in {self.configs_dir}")
        self.eta_map = name or ""

    def _use_eta_map(self) -> None:
        efficiency.set_active(os.path.join(self.configs_dir, self.eta_map) if self.eta_map else None)

    def setups(self) -> list[dict]:
        out = []
        if os.path.isdir(self.configs_dir):
            for f in sorted(os.listdir(self.configs_dir)):
                if not f.endswith(".json") or f.startswith("eta_map"):
                    continue
                try:
                    with open(os.path.join(self.configs_dir, f), encoding="utf-8") as fh:
                        d = json.load(fh)
                except (OSError, ValueError):
                    continue
                if isinstance(d, dict) and "changes" in d:
                    out.append({"name": str(d.get("name", f[:-5])), "changes": [str(c) for c in d["changes"]],
                                "notes": str(d.get("notes", "")), "saved": d.get("saved")})
        return out

    def setup_path(self, name: str) -> str:
        if not NAME_OK.match(name or "") or name.strip(" .") != name.strip():
            raise ValueError("setup names may use letters, digits, spaces and . _ - ( ) +")
        return os.path.join(self.configs_dir, re.sub(r"\s+", "_", name.strip()) + ".json")

    def save_setup(self, name: str, changes, notes: str = "", overwrite: bool = False) -> str:
        path = self.setup_path(name)
        changes = self.validate(changes)
        if os.path.exists(path) and not overwrite:
            raise FileExistsError(f"a setup called '{name.strip()}' exists already")
        os.makedirs(self.configs_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"name": name.strip(), "changes": changes, "notes": str(notes)[:2000],
                       "saved": time.strftime("%Y-%m-%d %H:%M"), "by": "Performance Lab"}, f, indent=2)
        return os.path.relpath(path, REPO)

    # ------------------------------------------------------------------ #
    # parameters
    # ------------------------------------------------------------------ #
    def validate(self, changes, competition: str | None = None) -> list[str]:
        """The changes as a clean list; raises ValueError on anything LapSim does not know."""
        if isinstance(changes, str):
            changes = [changes]
        out = [str(c).strip() for c in changes or [] if str(c).strip()]
        ls = self.lapsim(competition)
        known = set(PerfParams().numeric_fields()) | {a.lower() for a in ALIASES} | set(TRANSIENT_KEYS)
        try:
            veh, ev = split_changes(out, tuple(ls.comp.events))
            for c in veh:
                name = parse_change(c)[0]
                if name not in known and name.lower() not in known:
                    raise ValueError(f"unknown parameter '{name}' in '{c}'")
            for c in ev:
                parse_change(c)
            ls.vehicle(out)
            ls.competition(out)
        except (KeyError, ValueError) as e:
            raise ValueError(e.args[0] if e.args else str(e)) from None
        except (NameError, SyntaxError, TypeError) as e:
            raise ValueError(f"cannot read the value in {out}: {e}") from None
        return out

    def info(self, competition: str | None = None) -> dict:
        ls = self.lapsim(competition)
        groups = []
        for title, kind, rows in LAPSIM_SCHEMA:
            items = []
            for key, label, unit, desc in rows:
                if kind == "event":
                    ev, f = key.split(".")
                    e = ls.comp.events.get(ev)
                    if e is None:                       # event not in this competition
                        continue
                    val = getattr(e, f)
                else:
                    val = getattr(ls.veh, key, None)
                items.append({"key": key, "label": label, "unit": unit, "desc": desc, "value": val})
            groups.append({"title": title, "kind": kind, "rows": items})
        events = [{"name": n, "type": e.type, "track": e.track.name if e.track else "", "laps": e.laps,
                   "timing": e.timing, "length": e.track.L if e.track else None,
                   "closed": bool(e.track.closed) if e.track else False, "score": e.score}
                  for n, e in ls.comp.events.items()]
        return clean({"lapsim_dir": os.path.relpath(self.lapsim_dir, REPO), "competition": os.path.basename(ls.competition_file),
                      "competitions": self.competitions(), "events": events, "groups": groups, "setups": self.setups(),
                      "eta_maps": self.eta_maps(), "eta_map": self.eta_map, "params_file": os.path.relpath(ls.params_file, REPO)})

    # ------------------------------------------------------------------ #
    # runs
    # ------------------------------------------------------------------ #
    def raw(self, changes=(), competition: str | None = None, events=None) -> dict:
        """lapsim.run_all output (numpy arrays)."""
        with self.lock:
            self._use_eta_map()
            return self.lapsim(competition).run(list(changes), events)

    def run(self, changes=(), competition: str | None = None, traces: bool = True) -> dict:
        t0 = time.time()
        changes = self.validate(changes, competition)
        r = self.raw(changes, competition)
        out = {"changes": changes, "total": r["total"], "events": {}, "competition": competition or self.competitions()[0]}
        for n, e in r["events"].items():
            row = {k: e[k] for k in ("T", "t_lap", "t_first", "laps", "v_avg", "v_max", "points", "track")}
            row.update(E_total_kwh=e["E_total"] / J_PER_KWH, E_lap_kwh=e["E_lap"] / J_PER_KWH,
                       v_avg_kmh=e["v_avg"] * 3.6, v_max_kmh=e["v_max"] * 3.6, util=e["util"])
            if traces:
                row["trace"] = trace(e)
            out["events"][n] = row
        out["wall_s"] = time.time() - t0
        return clean(out)

    def event_names(self, competition: str | None = None) -> list[str]:
        return list(self.lapsim(competition).comp.events)

    def split(self, changes, competition: str | None = None):
        return split_changes(changes, tuple(self.event_names(competition)))


def trace(e: dict, limit: int = MAX_TRACE) -> dict:
    """The traces of one event result, decimated, in display units."""
    sl = decimate(len(e["s"]), limit)
    g = 9.81
    return {"s": e["s"][sl], "t": e["t"][sl], "v": e["v"][sl] * 3.6, "ax": e["ax"][sl] / g, "ay": e["ay"][sl] / g,
            "P_batt": e["P_batt"][sl] / 1e3, "E": e["E"][sl] / J_PER_KWH, "x": e["x"][sl], "y": e["y"][sl],
            "k": e["k"][sl]}
