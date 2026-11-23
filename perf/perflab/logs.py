"""Car logs against the simulation: speed / acceleration correlation on distance, and the powertrain efficiency.

Log files: .mf4 (MDF 4, read by fsdsim.logdata.mdf4: no asammdf needed) or .csv with a header row. The reference
traces of LR26_LapSim (reference/*.csv: the documentation's speed traces) can be used like logs.

Correlation (as LR26_LapSim's analysis/lts_correlate.m, documentation ch. 3.3.3): the log's distance comes from
its speed (or an s_m column), both traces are compared on distance over the part they share, and the metric is
the RMSE of the signals. The distance offset between log and simulation (timing line vs. GPS / log start) can be
given or searched (smallest speed RMSE within +-30 m).

Speed from a motor-speed channel: v = rpm / i_trans * circ_ref / 60 with the setup's gear ratio and tyre
circumference. That is only the car's speed when the wheels roll on the road (see `powertrain`'s roll-out check).
"""

from __future__ import annotations

import csv
import math
import os
import re

import numpy as np

from fsdsim.logdata import powertrain as PT
from fsdsim.logdata.mdf4 import MDF4, resample

from .lab import REPO, Lab, clean, decimate

G = 9.81
UPLOADS = os.environ.get("PERF_LAB_UPLOADS") or os.path.join(REPO, "perf", "uploads")      # env: tests, demos
RATE_HZ = 20.0
UNITS = {"kmh": 1 / 3.6, "mps": 1.0, "mph": 0.44704, "g": G, "mps2": 1.0, "rpm_motor": None,
         "deg": 1.0, "degps": math.pi / 180, "radps": 1.0, "raw": 1.0}
SIGNALS = ("speed", "ax", "ay", "yaw", "lat", "lon", "brake", "torque")       # what a spec can map
DEFAULT_UNIT = {"speed": "kmh", "ax": "g", "ay": "g", "yaw": "degps", "lat": "deg", "lon": "deg", "brake": "raw",
                "torque": "raw"}
SUGGEST = {
    "speed": re.compile(r"(^|_)(v|speed|vel|velocity|gps_speed|v_kmh|v_mps|groundspeed)(_|$)|vehicle_?speed|act_speed", re.I),
    "ax": re.compile(r"(^|_)(ax|acc_?x|accel_?x|long_?acc)", re.I),
    "ay": re.compile(r"(^|_)(ay|acc_?y|accel_?y|lat_?acc)", re.I),
    "time": re.compile(r"^(t|time|time_s|timestamp)$", re.I),
    "yaw": re.compile(r"yaw_?rate|(^|_)(r|gz|gyro_?z|wz)(_|$)", re.I),
    "lat": re.compile(r"(^|_)(lat|latitude)(_|$)", re.I),
    "lon": re.compile(r"(^|_)(lon|lng|long|longitude)(_|$)", re.I),
    "brake": re.compile(r"brake_?(press|p)|p_?brake", re.I),
}


def _safe_name(name: str) -> str:
    base = os.path.basename(str(name)).strip()
    base = re.sub(r"[^\w.\- ()+]", "_", base)[:120]
    if not base or base.startswith(".") or not base.lower().endswith((".mf4", ".mdf", ".csv")):
        raise ValueError("log files must be .mf4 or .csv")
    return base


def save_upload(name: str, data: bytes) -> str:
    os.makedirs(UPLOADS, exist_ok=True)
    path = os.path.join(UPLOADS, _safe_name(name))
    with open(path, "wb") as f:
        f.write(data)
    return path


def sources(lab: Lab) -> list[dict]:
    """Uploaded logs and LR26_LapSim's reference traces."""
    out = []
    if os.path.isdir(UPLOADS):
        for f in sorted(os.listdir(UPLOADS)):
            if f.lower().endswith((".mf4", ".mdf", ".csv")):
                p = os.path.join(UPLOADS, f)
                out.append({"id": f"upload:{f}", "name": f, "kind": "upload", "mb": os.path.getsize(p) / 1e6})
    ref = os.path.join(lab.lapsim_dir, "reference")
    if os.path.isdir(ref):
        for f in sorted(os.listdir(ref)):
            if f.lower().endswith(".csv"):
                out.append({"id": f"reference:{f}", "name": f, "kind": "reference", "mb": os.path.getsize(os.path.join(ref, f)) / 1e6})
    return out


def resolve(lab: Lab, source_id: str) -> str:
    kind, _, name = str(source_id).partition(":")
    name = os.path.basename(name)
    folder = {"upload": UPLOADS, "reference": os.path.join(lab.lapsim_dir, "reference")}.get(kind)
    if not folder or not name:
        raise ValueError(f"unknown log {source_id!r}")
    path = os.path.join(folder, name)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"log {name} not found")
    return path


# ---------------------------------------------------------------------- #
# reading
# ---------------------------------------------------------------------- #
def _read_csv(path: str) -> dict[str, np.ndarray]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    if len(rows) < 3:
        raise ValueError("the CSV needs a header row and at least two data rows")
    head = [h.strip() for h in rows[0]]
    cols = {h: [] for h in head}
    for r in rows[1:]:
        for h, v in zip(head, r):
            try:
                cols[h].append(float(v))
            except ValueError:
                cols[h].append(math.nan)
    return {h: np.asarray(v, float) for h, v in cols.items() if len(v) == len(rows) - 1 and np.isfinite(v).any()}


def inspect(lab: Lab, source_id: str) -> dict:
    path = resolve(lab, source_id)
    if path.lower().endswith(".csv"):
        cols = _read_csv(path)
        names = list(cols)
        n = len(next(iter(cols.values())))
        info = {"format": "csv", "channels": [{"name": c, "group": ""} for c in names], "rows": n}
    else:
        f = MDF4(path)
        seen, chans = set(), []
        for gi, name in f.channel_names():
            key = (name, f.groups[gi].name)
            if key in seen or name.lower() in ("time", "t"):
                continue
            seen.add(key)
            chans.append({"name": name, "group": f.groups[gi].name or f"group {gi}"})
        names = [c["name"] for c in chans]
        info = {"format": "mf4", "channels": chans}
        info["powertrain"] = sum(n in names for n in PT.SIGNALS.values())
    sug = {}
    for k, rx in SUGGEST.items():
        hit = [n for n in names if rx.search(n)]
        sug[k] = hit[0] if hit else ""
    if "s_m" in names:
        sug["distance"] = "s_m"
    info.update(suggest=sug, name=os.path.basename(path), source=source_id)
    return clean(info)


def _unit_value(x: np.ndarray, unit: str, scale: float, lab_veh) -> np.ndarray:
    if unit == "rpm_motor":
        return x / lab_veh.i_trans * lab_veh.circ_ref / 60.0 * scale
    if unit not in UNITS:
        raise ValueError(f"unknown unit {unit!r} (use {', '.join(UNITS)})")
    return x * UNITS[unit] * scale


def extract(lab: Lab, source_id: str, spec: dict, changes=(), competition=None) -> dict:
    """The log as one lap on a common time base: t [s], s [m], v [m/s], ax, ay [m/s^2] (NaN if not given).

    spec: {"speed": {"channel", "group"?, "unit", "scale"?}, "ax": {...}, "ay": {...}, "time": "<csv column>",
           "distance": "<csv column>" (optional), "t0", "t1" (window, s)}"""
    path = resolve(lab, source_id)
    veh = lab.lapsim(competition).vehicle(lab.validate(changes, competition))
    sp = spec.get("speed") or {}
    if not sp.get("channel"):
        raise ValueError("choose the speed channel")
    series, units = {}, {}
    if path.lower().endswith(".csv"):
        cols = _read_csv(path)
        tcol, dcol = spec.get("time") or "", spec.get("distance") or ""
        if tcol and tcol not in cols:
            raise ValueError(f"no column {tcol!r}")
        if dcol and dcol not in cols:
            raise ValueError(f"no column {dcol!r}")
        for key in SIGNALS:
            c = (spec.get(key) or {}).get("channel")
            if c:
                if c not in cols:
                    raise ValueError(f"no column {c!r}")
                units[key] = spec[key]
                series[key] = cols[c]
        v = _unit_value(series["speed"], units["speed"].get("unit", "mps"), float(units["speed"].get("scale", 1) or 1), veh)
        if dcol:                                    # distance-based trace (e.g. the documentation's speed traces)
            s = cols[dcol] - cols[dcol][0]
            seg = np.diff(s) / np.maximum(0.5 * (v[1:] + v[:-1]), 0.3)
            t = np.r_[0.0, np.cumsum(seg)] if not tcol else cols[tcol] - cols[tcol][0]
        elif tcol:
            t = cols[tcol] - cols[tcol][0]
            s = None
        else:
            raise ValueError("a CSV log needs a time or a distance column")
        data = {"t": t, "v": v}
        for key in SIGNALS[1:]:
            if key in series:
                data[key] = _unit_value(series[key], units[key].get("unit", DEFAULT_UNIT[key]),
                                        float(units[key].get("scale", 1) or 1), veh)
        if s is not None:
            data["s"] = s
    else:
        f = MDF4(path)
        raw = {}
        for key in SIGNALS:
            c = spec.get(key) or {}
            if c.get("channel"):
                found = [x for x in f.signals(c["channel"]) if not c.get("group") or c["group"] == x[0]]
                if not found:
                    raise ValueError(f"no channel {c['channel']!r}")
                _, t, x = max(found, key=lambda a: len(a[1]))
                raw[key] = (t, x)
                units[key] = c
        grid, d = resample(raw, RATE_HZ)
        data = {"t": grid - grid[0], "v": _unit_value(d["speed"], units["speed"].get("unit", "kmh"),
                                                      float(units["speed"].get("scale", 1) or 1), veh)}
        for key in SIGNALS[1:]:
            if key in d:
                data[key] = _unit_value(d[key], units[key].get("unit", DEFAULT_UNIT[key]),
                                        float(units[key].get("scale", 1) or 1), veh)
    # window
    t0, t1 = spec.get("t0"), spec.get("t1")
    m = np.ones(len(data["t"]), bool)
    if t0 not in (None, ""):
        m &= data["t"] >= float(t0)
    if t1 not in (None, ""):
        m &= data["t"] <= float(t1)
    if m.sum() < 5:
        raise ValueError("fewer than 5 samples in the time window")
    data = {k: np.asarray(x)[m] for k, x in data.items()}
    ok = np.isfinite(data["v"])
    data = {k: x[ok] for k, x in data.items()}
    data["t"] = data["t"] - data["t"][0]
    if "s" not in data:
        v = data["v"]
        data["s"] = np.r_[0.0, np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(data["t"]))]
    else:
        data["s"] = data["s"] - data["s"][0]
    return data


# ---------------------------------------------------------------------- #
# correlation
# ---------------------------------------------------------------------- #
def _rmse(a, b) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2))) if m.sum() else math.nan


def correlate(lab: Lab, source_id: str, spec: dict, event: str, changes=(), competition=None, shift=None) -> dict:
    changes = lab.validate(changes, competition)
    log = extract(lab, source_id, spec, changes, competition)
    e = lab.raw(changes, competition, [event])["events"][event]
    ss, sv, st = e["s"], e["v"], e["t"]
    L = float(ss[-1])

    def compare(sh):
        s = log["s"] + sh
        m = (s >= 0.0) & (s <= L)
        return m, s

    if shift in (None, "", "auto"):
        best = (math.inf, 0.0)
        for sh in np.arange(-30.0, 30.01, 0.5):
            m, s = compare(sh)
            if m.sum() > 10:
                r = _rmse(log["v"][m], np.interp(s[m], ss, sv))
                if r < best[0]:
                    best = (r, float(sh))
        shift, auto = best[1], True
    else:
        shift, auto = float(shift), False
    m, s = compare(shift)
    if m.sum() < 5:
        raise ValueError("the log and the simulated lap do not overlap on distance (check the window and offset)")
    s_c, v_log = s[m], log["v"][m]
    v_sim = np.interp(s_c, ss, sv)
    t_log = log["t"][m] - log["t"][m][0]
    t_sim = np.interp(s_c, ss, st) - np.interp(s_c[0], ss, st)
    out = {"event": event, "shift_m": shift, "shift_auto": auto, "s_from": float(s_c[0]), "s_to": float(s_c[-1]),
           "lap_length": L, "coverage": float((s_c[-1] - s_c[0]) / L),
           "rmse_v_kmh": _rmse(v_log, v_sim) * 3.6, "bias_v_kmh": float(np.nanmean(v_log - v_sim)) * 3.6,
           "t_log": float(t_log[-1]), "t_sim": float(t_sim[-1]), "v_max_log_kmh": float(np.nanmax(v_log)) * 3.6,
           "v_max_sim_kmh": float(np.nanmax(v_sim)) * 3.6}
    sim_ax = np.interp(s_c, ss, e["ax"])
    sim_ay = np.abs(np.interp(s_c, ss, e["ay"]))
    if "ax" in log:
        out["rmse_ax_g"] = _rmse(log["ax"][m], sim_ax) / G
    if "ay" in log:
        out["rmse_ay_g"] = _rmse(np.abs(log["ay"][m]), sim_ay) / G
    sl = decimate(len(s_c), 2000)
    out["trace"] = {"s": s_c[sl], "v_log": v_log[sl] * 3.6, "v_sim": v_sim[sl] * 3.6, "dt": (t_log - t_sim)[sl],
                    "ax_log": log["ax"][m][sl] / G if "ax" in log else [], "ax_sim": sim_ax[sl] / G,
                    "ay_log": np.abs(log["ay"][m][sl]) / G if "ay" in log else [], "ay_sim": sim_ay[sl] / G}
    return clean(out)


# ---------------------------------------------------------------------- #
# powertrain
# ---------------------------------------------------------------------- #
ROLLOUT_CAR_MAX = 0.6        # m/s^2 (as tools/log_correlate.py)


def powertrain(lab: Lab, source_id: str, changes=(), competition=None, motors_per_axle: float = 2.0) -> dict:
    path = resolve(lab, source_id)
    if not path.lower().endswith((".mf4", ".mdf")):
        raise ValueError("the powertrain analysis needs an .mf4 log with the inverter channels")
    veh = lab.lapsim(competition).vehicle(lab.validate(changes, competition))
    d = PT.load(path, RATE_HZ)
    s = PT.motor_samples(d)
    if len(s["eff"]) < 50:
        raise ValueError("too few driving samples (torque > 0.3 Nm and DC power > 0.5 kW) in this log")
    te, ne, tab, cnt = PT.efficiency_map(s)
    fit = PT.loss_fit(s)
    ro = PT.rollout(d, veh.i_trans, veh.circ_ref)
    dec = ro["decel_median"]
    verdict = ("no zero-torque, no-brake stretch long enough" if not math.isfinite(dec) else
               "consistent with a car on the road" if dec <= ROLLOUT_CAR_MAX else
               "far above a rolling car: the wheels were not on the road; do not fit vehicle dynamics to this log")
    sl = decimate(len(s["eff"]), 3000)
    return clean({"duration_s": float(d["t"][-1] - d["t"][0]), "samples": len(s["eff"]), "T_max": float(s["T"].max()),
                  "n_max": float(s["n"].max()), "eff_median": float(np.median(s["eff"])),
                  "eff_p10": float(np.percentile(s["eff"], 10)), "eff_p90": float(np.percentile(s["eff"], 90)),
                  "eta_motor_lapsim": veh.eta_motor, "map": {"t_edges": te, "n_edges": ne, "eff": tab, "count": cnt},
                  "fit": fit, "rollout": ro, "rollout_verdict": verdict, "motors_per_axle": motors_per_axle,
                  "scatter": {"T": s["T"][sl], "n": s["n"][sl], "eff": s["eff"][sl]}})


def save_eta_map(lab: Lab, source_id: str, name: str, changes=(), competition=None, motors_per_axle: float = 2.0,
                 overwrite: bool = False) -> str:
    from fsdsim.perf.efficiency import EfficiencyMap  # noqa: PLC0415

    if not re.fullmatch(r"[\w\-]{1,40}", name or ""):
        raise ValueError("map names may use letters, digits, _ and -")
    path = os.path.join(lab.configs_dir, f"eta_map_{name}.json")
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(f"eta_map_{name}.json exists already")
    veh = lab.lapsim(competition).vehicle(lab.validate(changes, competition))
    d = PT.load(resolve(lab, source_id), RATE_HZ)
    fit = PT.loss_fit(PT.motor_samples(d))
    t_max = max(veh.T_motor / motors_per_axle, fit["T_max"])
    emap = EfficiencyMap.from_loss_fit(fit["k"], t_max=t_max, n_max=veh.n_max, source=os.path.basename(resolve(lab, source_id)),
                                       valid={"T_max_Nm": fit["T_max"], "n_max_rpm": fit["n_max"], "rms_kw": fit["rms_kw"]},
                                       motors_per_set=motors_per_axle)
    emap.save(path)
    return os.path.relpath(path, REPO)
