"""Studies on the QSS LapSim: sensitivities, 1-D and 2-D sweeps, setup comparison.

Same definitions as LR26_LapSim's analysis/lts_sensitivity_table.m, lts_sweep.m, lts_sweep2d.m and lts_compare.m
(documentation ch. 2), so the numbers can be checked against MATLAB:

* sensitivity: every parameter at (1 + delta) and (1 - delta) of its value, everything else unchanged.
  Point sensitivity [P/%] = (best total - baseline) / (100 delta), "linearised in the improving direction";
  per event the time change of both sides. Mass also per kg.
* sweep: the total and every event's time and points for each value.
* compare: several setups side by side, with the speed traces of one event.

Every function takes `job` (a jobs.Job or None) for progress and cancelling.
"""

from __future__ import annotations

import math

import numpy as np

from .lab import Lab, clean, trace

DEFAULT_SENS = ["mu", "m", "i_trans", "h_cog", "c_z", "c_w", "P_max", "aero_bal_f", "T_motor", "A"]


def _step(job, i, n, text):
    if job is not None:
        job.step(i, n, text)


def _row(r: dict) -> dict:
    return {"total": r["total"], "T": {n: e["T"] for n, e in r["events"].items()},
            "points": {n: e["points"] for n, e in r["events"].items()},
            "E_kwh": {n: e["E_total"] / 3.6e6 for n, e in r["events"].items()}}


def _value(lab: Lab, name: str, base, competition):
    veh = lab.lapsim(competition).vehicle(base)
    from fsdsim.perf.params import ALIASES  # noqa: PLC0415

    key = ALIASES.get(name.lower(), (name,))[-1]
    if "." in name:
        ev, f = name.split(".", 1)
        return float(getattr(lab.lapsim(competition).competition(base).events[ev], f))
    return float(getattr(veh, key))


def sensitivity(lab: Lab, params=None, delta: float = 0.10, base=(), competition=None, job=None) -> dict:
    params = list(params or DEFAULT_SENS)
    base = lab.validate(base, competition)
    if not 0 < delta < 1:
        raise ValueError("delta must be between 0 and 1 (e.g. 0.1 = +-10 %)")
    for p in params:
        lab.validate(base + [f"{p}*1"], competition)
    b = _row(lab.raw(base, competition))
    rows = []
    for i, p in enumerate(params):
        _step(job, i, len(params), f"{p} +-{100 * delta:g} %")
        val = _value(lab, p, base, competition)
        plus = _row(lab.raw(base + [f"{p}*{1 + delta}"], competition))
        minus = _row(lab.raw(base + [f"{p}*{1 - delta}"], competition))
        best = plus if plus["total"] >= minus["total"] else minus
        row = {"param": p, "value": val, "total_plus": plus["total"], "total_minus": minus["total"],
               "points_per_pct": (best["total"] - b["total"]) / (100 * delta),
               "better": "+" if best is plus else "-",
               "dT_plus": {n: plus["T"][n] - b["T"][n] for n in b["T"]},
               "dT_minus": {n: minus["T"][n] - b["T"][n] for n in b["T"]},
               "dP_plus": {n: plus["points"][n] - b["points"][n] for n in b["T"]},
               "dP_minus": {n: minus["points"][n] - b["points"][n] for n in b["T"]},
               "no_effect": val == 0.0 or (abs(plus["total"] - b["total"]) < 1e-9 and abs(minus["total"] - b["total"]) < 1e-9)}
        if p.lower() in ("m", "mass") and val:
            dkg = val * delta
            row["points_per_kg"] = (best["total"] - b["total"]) / dkg
            if "autocross" in b["T"]:
                row["autocross_s_per_kg"] = (b["T"]["autocross"] - best["T"]["autocross"]) / dkg
        rows.append(row)
    rows.sort(key=lambda r: -abs(r["points_per_pct"]))
    _step(job, len(params), len(params), "done")
    return clean({"delta": delta, "base_changes": base, "baseline": b, "rows": rows})


def values_from(spec) -> list[float]:
    """[1, 2, 3], "1,2,3" or {"from": a, "to": b, "n": k} -> list of floats (at most 60)."""
    if isinstance(spec, dict):
        n = int(spec.get("n", 10))
        if not 2 <= n <= 60:
            raise ValueError("between 2 and 60 values")
        return [float(x) for x in np.linspace(float(spec["from"]), float(spec["to"]), n)]
    if isinstance(spec, str):
        spec = [x for x in spec.replace(";", ",").split(",") if x.strip()]
    vals = [float(x) for x in spec]
    if not 1 <= len(vals) <= 60 or not all(math.isfinite(v) for v in vals):
        raise ValueError("between 1 and 60 finite values")
    return vals


def sweep(lab: Lab, param: str, values, base=(), competition=None, job=None, event_traces: str = "") -> dict:
    """One parameter over values (absolute values: '<param>=<value>'). event_traces: also the speed trace of
    that event for every value."""
    base = lab.validate(base, competition)
    vals = values_from(values)
    lab.validate(base + [f"{param}={vals[0]}"], competition)
    out = []
    for i, v in enumerate(vals):
        _step(job, i, len(vals), f"{param} = {v:g}")
        r = lab.raw(base + [f"{param}={v}"], competition)
        row = {"value": v, **_row(r)}
        if event_traces and event_traces in r["events"]:
            tr = trace(r["events"][event_traces], 600)
            row["trace"] = {"s": tr["s"], "v": tr["v"]}
        out.append(row)
    _step(job, len(vals), len(vals), "done")
    base_val = _value(lab, param, base, competition)
    return clean({"param": param, "base_value": base_val, "base_changes": base, "rows": out})


def sweep2d(lab: Lab, p1: str, v1, p2: str, v2, base=(), competition=None, job=None) -> dict:
    base = lab.validate(base, competition)
    a, b = values_from(v1), values_from(v2)
    if len(a) * len(b) > 625:
        raise ValueError("at most 625 points (25 x 25)")
    lab.validate(base + [f"{p1}={a[0]}", f"{p2}={b[0]}"], competition)
    total = np.full((len(a), len(b)), np.nan)
    per_event: dict[str, np.ndarray] = {}
    n, k = len(a) * len(b), 0
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            _step(job, k, n, f"{p1} = {x:g}, {p2} = {y:g}")
            k += 1
            r = lab.raw(base + [f"{p1}={x}", f"{p2}={y}"], competition)
            total[i, j] = r["total"]
            for ev, e in r["events"].items():
                per_event.setdefault(ev, np.full((len(a), len(b)), np.nan))[i, j] = e["points"]
    _step(job, n, n, "done")
    i, j = np.unravel_index(np.nanargmax(total), total.shape)
    return clean({"p1": p1, "p2": p2, "v1": a, "v2": b, "total": total, "points": per_event, "base_changes": base,
                  "best": {"v1": a[i], "v2": b[j], "total": total[i, j]},
                  "base_values": [_value(lab, p1, base, competition), _value(lab, p2, base, competition)]})


def compare(lab: Lab, setups: list[dict], competition=None, job=None) -> dict:
    """setups: [{name, changes}] (2 to 8). Results per setup plus the traces of every event."""
    if not 1 <= len(setups) <= 8:
        raise ValueError("compare 1 to 8 setups")
    out = []
    for i, s in enumerate(setups):
        _step(job, i, len(setups), str(s.get("name", i)))
        ch = lab.validate(s.get("changes", []), competition)
        r = lab.raw(ch, competition)
        row = {"name": str(s.get("name", f"setup {i + 1}")), "changes": ch, **_row(r), "traces": {}}
        for ev, e in r["events"].items():
            tr = trace(e, 800)
            row["traces"][ev] = {k: tr[k] for k in ("s", "t", "v", "ax", "ay")}
        out.append(row)
    _step(job, len(setups), len(setups), "done")
    return clean({"setups": out})
