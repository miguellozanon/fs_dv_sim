"""Method comparison and robustness: which lap time method answers which question.

* `compare`: one event, the baseline and one change, solved by every method (LapSim QSS, the point-mass
  QSS solver, the transient lap, the minimum-time optimal control problems). Gives the time, the time change
  of the change, the compute time and whether the methods agree on its sign.
* `monte_carlo`: the parameters the team does not know exactly are drawn from their uncertainty; QSS gives the
  distribution of the points, and with a change, the probability that the change gains points.

docs/PERF_METHODS.md explains every method, its hypotheses and limits.
"""

from __future__ import annotations

import math
import time

import numpy as np

from . import ocp, transient
from .lab import Lab, clean

# name -> (label, family, short description)
METHODS = {
    "qss": ("LapSim QSS", "quasi-steady-state",
            "LapSim's solver: ggV limit per point, forward / backward passes, driver utilisation"),
    "qss_point_mass": ("QSS point mass (friction circle)", "quasi-steady-state",
                       "first-order check: friction circle from the ggV corner, no power-dependent ellipse"),
    "transient": ("Transient lap (vehicle model + driver)", "time simulation",
                  "the simulator's dynamic vehicle and reference driver, integrated in time"),
    "ocp_point_mass": ("Optimal control: point mass", "minimum-time optimal control",
                       "point mass on the fitted ggV envelope, solved over the whole lap at once"),
    "ocp_single_track": ("Optimal control: single track", "minimum-time optimal control",
                         "dynamic bicycle model, Pacejka tyres, load transfer, steering limits"),
    "ocp_double_track": ("Optimal control: double track", "minimum-time optimal control",
                         "four wheels, lateral load transfer, equal torque left / right"),
    "ocp_double_track_tv": ("Optimal control: double track + torque vectoring", "minimum-time optimal control",
                            "as double track, with free torque per wheel"),
}
DEFAULT_METHODS = ("qss", "qss_point_mass", "transient", "ocp_point_mass", "ocp_single_track")

# 1-sigma uncertainty of the parameters (relative), editable from the app
UNCERTAINTY = {"mu": 0.03, "mass": 0.01, "cl": 0.05, "cd": 0.05, "power": 0.02, "util_y": 0.02}


def _qss(lab: Lab, changes, competition, event: str, solver: str) -> dict:
    with lab.lock:
        lab._use_eta_map()
        r = lab.lapsim(competition).run(list(changes), [event], solver=solver)
    e = r["events"][event]
    return {"T": float(e["T"]), "t_lap": float(e["t_lap"]), "points": float(e["points"])}


def solve(lab: Lab, method: str, event: str, changes=(), competition=None, track_width: float = 3.5) -> dict:
    """One method, one event. Returns T (event time), t_lap, points, ok, wall_s and a short note."""
    t0 = time.time()
    note = ""
    if method in ("qss", "qss_point_mass"):
        r = _qss(lab, changes, competition, event, "qss" if method == "qss" else "point_mass")
        ok = True
    elif method == "transient":
        t = transient.run(lab, event, changes, competition, 0.02, 3.0)
        r = {"T": t["T"], "t_lap": t["t_lap"], "points": t["points"]}
        ok = t["status"] == "finished"
        note = str(t["status"])
    elif method.startswith("ocp_"):
        model = method[4:].replace("_tv", "")
        o = ocp.run(lab, event, model, changes, competition, track_width,
                    torque_vectoring=method.endswith("_tv"), fixed_line=True)
        r = {"T": o["T"], "t_lap": o["t_lap"], "points": o["points"]}
        ok = bool(o["ok"])
        note = "; ".join(f"{s['model']}: {s['status']}" for s in o["stages"])
    else:
        raise ValueError(f"unknown method {method!r} (use {', '.join(METHODS)})")
    if not ok:
        r = {"T": None, "t_lap": None, "points": None}
    return {**r, "ok": ok, "wall_s": time.time() - t0, "note": note}


def compare(lab: Lab, event: str, change: str = "", methods=None, base=(), competition=None, job=None) -> dict:
    """Every method on `event`, for the baseline (`base`) and for the baseline plus `change`."""
    methods = list(methods or DEFAULT_METHODS)
    for m in methods:
        if m in ocp_methods() and not ocp.available()[0]:
            raise ValueError(f"{m} needs CasADi: {ocp.available()[1]}")
        if m not in METHODS:
            raise ValueError(f"unknown method {m!r}")
    base = lab.validate(base, competition)
    with_change = lab.validate(list(base) + ([change] if change else []), competition)
    if event not in lab.event_names(competition):
        raise ValueError(f"unknown event {event!r}")
    variants = [("base", base)] + ([("change", with_change)] if change else [])
    n = len(methods) * len(variants)
    rows, i = [], 0
    for m in methods:
        row = {"method": m, "label": METHODS[m][0], "family": METHODS[m][1], "description": METHODS[m][2]}
        for name, ch in variants:
            if job is not None:
                job.step(i, n, f"{METHODS[m][0]}: {name}")
            row[name] = solve(lab, m, event, ch, competition)
            i += 1
        b = row["base"]
        if change and b["ok"] and row["change"]["ok"]:
            row["dT"] = row["change"]["T"] - b["T"]
            row["dpoints"] = row["change"]["points"] - b["points"]
        else:
            row["dT"] = row["dpoints"] = None
        row["wall_s"] = sum(row[v]["wall_s"] for v, _ in variants)
        rows.append(row)
    ref = next((r for r in rows if r["method"] == "qss" and r["base"]["ok"]), None)
    for r in rows:
        r["vs_qss"] = (r["base"]["T"] / ref["base"]["T"] - 1) if ref and r["base"]["ok"] else None
    signs = {r["method"]: (0 if r["dT"] is None else int(np.sign(round(r["dT"], 4)))) for r in rows}
    used = [s for s in signs.values() if s != 0]
    agree = bool(used) and all(s == used[0] for s in used)
    if job is not None:
        job.step(n, n, "done")
    return clean({"event": event, "change": change, "base": base, "methods": rows, "signs": signs,
                  "agree": agree if change else None, "competition": competition})


def ocp_methods():
    return [m for m in METHODS if m.startswith("ocp_")]


def _draw(rng, unc: dict, events) -> list[str]:
    ch = []
    for k, sd in unc.items():
        if not sd:
            continue
        f = float(np.clip(1 + rng.normal(0, sd), 1 - 3 * sd, 1 + 3 * sd))
        if k == "util_y":                       # the driver: one factor on every event's lateral utilisation
            ch += [f"{e}.util_y*{f:.6f}" for e in events]
        else:
            ch.append(f"{k}*{f:.6f}")
    return ch


def monte_carlo(lab: Lab, n: int = 200, change: str = "", uncertainty=None, base=(), competition=None,
                seed: int = 1, job=None) -> dict:
    """QSS points under parameter uncertainty; with `change`, the gain on the same draws (paired)."""
    n = int(n)
    if not 10 <= n <= 5000:
        raise ValueError("number of draws between 10 and 5000")
    unc = {**UNCERTAINTY, **{k: float(v) for k, v in (uncertainty or {}).items()}}
    for k, v in unc.items():
        if not 0 <= v <= 0.5:
            raise ValueError(f"uncertainty of {k} between 0 and 0.5")
    base = lab.validate(base, competition)
    if change:
        lab.validate(list(base) + [change], competition)
    events = lab.event_names(competition)
    rng = np.random.default_rng(int(seed))
    tot, gain, per = [], [], {e: [] for e in events}
    t0 = time.time()
    for i in range(n):
        if job is not None and i % 5 == 0:
            job.step(i, n, f"draw {i + 1} of {n}")
        d = _draw(rng, unc, events)
        r = lab.raw(list(base) + d, competition)
        tot.append(r["total"])
        for e in events:
            per[e].append(r["events"][e]["points"])
        if change:
            gain.append(lab.raw(list(base) + d + [change], competition)["total"] - r["total"])
    tot = np.array(tot)
    nominal = lab.raw(base, competition)["total"]

    def stats(x):
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        if not len(x):
            return None
        return {"mean": float(np.mean(x)), "sd": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
                "p5": float(np.percentile(x, 5)), "p50": float(np.percentile(x, 50)),
                "p95": float(np.percentile(x, 95)), "min": float(np.min(x)), "max": float(np.max(x))}

    out = {"n": n, "seed": int(seed), "uncertainty": unc, "nominal": nominal, "total": stats(tot),
           "events": {e: stats(v) for e, v in per.items()}, "samples": tot, "wall_s": time.time() - t0,
           "change": change}
    if change:
        g = np.array(gain)
        nominal_gain = lab.raw(list(base) + [change], competition)["total"] - nominal
        se = float(np.std(g, ddof=1) / math.sqrt(len(g))) if len(g) > 1 else 0.0
        out["gain"] = {**stats(g), "nominal": nominal_gain, "p_gain": float(np.mean(g > 0)), "se_mean": se,
                       "samples": g}
    if job is not None:
        job.step(n, n, "done")
    return clean(out)


def guidance() -> list[dict]:
    """Which method for which question (shown in the app; the same table is in docs/PERF_METHODS.md)."""
    return [
        {"question": "Ranking of setup changes, sensitivity, sweeps (hundreds of runs)",
         "method": "LapSim QSS", "why": "fast (0.2 s for all events), matches MATLAB, driver utilisation calibrated"},
        {"question": "Is a gain larger than what we do not know about the car?",
         "method": "Monte-Carlo on QSS", "why": "gives the spread of the points and the probability that a change gains"},
        {"question": "A check of QSS itself (is the solver the limit?)",
         "method": "Optimal control: point mass, fixed line",
         "why": "same physics as QSS solved as one optimisation: should agree within ~0.3 %"},
        {"question": "Effect of yaw inertia, steering limits, load transfer, tyre shape, balance",
         "method": "Optimal control: single / double track",
         "why": "these quantities do not exist in QSS; the OCP is the limit of a perfect driver"},
        {"question": "Torque vectoring (and other control-allocation questions)",
         "method": "Optimal control: double track + torque vectoring", "why": "needs individual wheels"},
        {"question": "How a real (or the DV) driver gets on with the car, stability, transients",
         "method": "Transient lap", "why": "closed loop with a driver model; not the limit, but realistic behaviour"},
        {"question": "Racing line on a known track width",
         "method": "Optimal control with free line", "why": "only on a true centre line with its width (not the FSA tables)"},
    ]
