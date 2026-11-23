"""Model parameters from car logs (docs/LOGGING_SPEC.md, section 6).

* `coastdown`: deceleration a(v) = c0 + c2 v^2 of a coast (no torque, no brake) -> rolling resistance and drag.
* `grip_use`: the log's accelerations against the model's ggV (tyre limit, utilisation 1) per speed -> the share of
  the limit the driver used: suggested util_y (lateral) and util_x (braking, and traction where grip-limited).
* `steady`: steady cornering (constant radius) -> maximum steady ay against speed, and the grip scale of the model
  that matches it.
* `track_from_gps`: latitude / longitude of one lap -> a LapSim track file (s, x, y, curvature).

Nothing here changes lr26_params.m: the results are suggestions shown next to the setup's values.
"""

from __future__ import annotations

import math
import os
import re

import numpy as np

from fsdsim.perf import ggv as _ggv

from . import logs
from .lab import Lab, clean, decimate

G = 9.81


def _smooth(x: np.ndarray, n: int) -> np.ndarray:
    """Centred moving average over n samples (n odd), edges by shrinking windows."""
    n = max(1, int(n) | 1)
    if n == 1 or len(x) < n:
        return np.asarray(x, float)
    k = np.ones(n) / n
    y = np.convolve(x, k, mode="same")
    norm = np.convolve(np.ones(len(x)), k, mode="same")
    return y / norm


def _deriv(t: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.gradient(x, t)


# --------------------------------------------------------------------------------------------- coast-down
def coastdown(lab: Lab, source_id: str, spec: dict, changes=(), competition=None, smooth_s: float = 0.5) -> dict:
    """Fit -dv/dt = c0 + c2 v^2 on the chosen window (speed only; the car must coast: no torque, no brake).

    With the setup's mass, rotating-mass factor, air density, frontal area and downforce:
        c0 = f_roll g m / m_eff,   c2 = (0.5 rho (cwA + f_roll czA) eff) / m_eff
    -> f_roll and c_w (for the setup's A and aero_eff)."""
    d = logs.extract(lab, source_id, spec, changes, competition)
    veh = lab.lapsim(competition).vehicle(lab.validate(changes, competition))
    t, v = d["t"], d["v"]
    dt = float(np.median(np.diff(t))) if len(t) > 2 else 0.05
    vs = _smooth(v, round(smooth_s / max(dt, 1e-3)))
    a = -_deriv(t, vs)
    m = (vs > 3.0) & np.isfinite(a)
    if "brake" in d:                                      # drop samples with the brake applied (> 2 % of its range)
        b = d["brake"]
        thr = np.nanmin(b) + 0.02 * max(np.nanmax(b) - np.nanmin(b), 1e-9)
        m &= b <= max(thr, np.nanmin(b) + 0.5)
    if "torque" in d:
        m &= np.abs(d["torque"]) < 1.0
    if m.sum() < 20:
        raise ValueError("fewer than 20 coasting samples above 3 m/s in the window (check the window, brake and torque)")
    # the longest uninterrupted coast in the window
    idx = np.flatnonzero(m)
    runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    run = max(runs, key=len)
    if len(run) < 20:
        raise ValueError("the longest uninterrupted coast has fewer than 20 samples")
    m = np.zeros(len(t), bool)
    m[run] = True
    # 1) linear fit of the smoothed deceleration (first guess), 2) the speed itself fitted with the exact
    #    solution v(t) = sqrt(c0/c2) tan(atan(v0 sqrt(c2/c0)) - sqrt(c0 c2) (t - t0)) (no differentiation of noise)
    A = np.stack([np.ones(m.sum()), vs[m] ** 2], axis=1)
    (c0, c2), *_ = np.linalg.lstsq(A, a[m], rcond=None)
    tt, vr = t[m] - t[m][0], v[m]

    def v_model(p):
        v0, k0, k2 = p
        k0, k2 = max(k0, 1e-6), max(k2, 1e-9)
        w = math.sqrt(k0 / k2)
        return w * np.tan(np.arctan(v0 / w) - math.sqrt(k0 * k2) * tt)

    try:
        from scipy.optimize import least_squares  # noqa: PLC0415

        p0 = [float(vs[m][0]), max(float(c0), 0.01), max(float(c2), 1e-5)]
        fit = least_squares(lambda p: v_model(p) - vr, p0, x_scale=[1.0, 0.05, 1e-4], method="lm")
        if fit.success and fit.x[1] > 0 and fit.x[2] > 0:
            c0, c2 = float(fit.x[1]), float(fit.x[2])
    except ImportError:  # pragma: no cover
        pass
    res = a[m] - A @ np.array([c0, c2])
    meff = veh.m * veh.e_i
    f_roll = c0 * meff / (veh.m * G)
    kz = 0.5 * veh.rho * veh.c_z * veh.A * veh.aero_eff
    cwA = (c2 * meff - f_roll * kz) / (0.5 * veh.rho * veh.aero_eff)
    c_w = cwA / veh.A
    vv = np.linspace(float(vs[m].min()), float(vs[m].max()), 40)
    model = (veh.f_roll * (veh.m * G + kz * vv ** 2) + 0.5 * veh.rho * veh.c_w * veh.A * veh.aero_eff * vv ** 2) / meff
    sl = decimate(int(m.sum()), 1500)
    warn = []
    if c0 <= 0 or c2 <= 0:
        warn.append("A coefficient came out negative: the window is probably not a clean coast (torque, brake, slope "
                    "or wind). Check the window.")
    if float(vs[m].max() - vs[m].min()) < 8:
        warn.append("The speed range is small (< 8 m/s): drag and rolling resistance cannot be separated well.")
    return clean({"c0": float(c0), "c2": float(c2), "rms": float(np.sqrt(np.mean(res ** 2))), "n": int(m.sum()),
                  "v_range_kmh": [float(vs[m].min() * 3.6), float(vs[m].max() * 3.6)],
                  "f_roll": float(f_roll), "c_w": float(c_w), "cwA": float(cwA),
                  "setup": {"f_roll": veh.f_roll, "c_w": veh.c_w, "cwA": veh.c_w * veh.A},
                  "suggest": [f"f_roll={f_roll:.4f}", f"c_w={c_w:.3f}"],
                  "points": {"v": vs[m][sl] * 3.6, "a": a[m][sl]},
                  "fit": {"v": vv * 3.6, "a": c0 + c2 * vv ** 2, "model": model}, "warnings": warn})


# --------------------------------------------------------------------------------------------- ggV use
def _limits(veh, speeds):
    """Tyre-limit ggV of the car (utilisation 1): ay_max(v), brake and drive at ay = 0, and whether drive is grip-
    limited (the motor could give more) at each speed."""
    tab = _ggv.ggv(veh, speeds, n_ay=11)
    free = _ggv.ggv(veh.copy(T_motor=1e5, P_max=1e9, P_max_endurance=1e9), speeds, n_ay=11)
    ay_max = np.array([_ggv.ay_limit(veh, float(v)) for v in speeds])
    brake = -tab["ax_dec"][:, 0]
    drive = tab["ax_acc"][:, 0]
    grip_drive = free["ax_acc"][:, 0]
    return ay_max, brake, drive, grip_drive, drive >= grip_drive - 0.05


def grip_use(lab: Lab, source_id: str, spec: dict, changes=(), competition=None, pct: float = 95.0) -> dict:
    """Share of the model's tyre limit the log reaches, per direction and speed band."""
    d = logs.extract(lab, source_id, spec, changes, competition)
    if "ax" not in d or "ay" not in d:
        raise ValueError("grip use needs the ax and ay channels")
    veh = lab.lapsim(competition).vehicle(lab.validate(changes, competition)).copy(util_x=1.0, util_y=1.0)
    v, ax, ay = d["v"], d["ax"], np.abs(d["ay"])
    ok = np.isfinite(ax) & np.isfinite(ay) & (v > 3.0)
    v, ax, ay = v[ok], ax[ok], ay[ok]
    if len(v) < 50:
        raise ValueError("fewer than 50 samples above 3 m/s with ax and ay")
    speeds = np.linspace(max(3.0, float(v.min())), float(v.max()) + 1e-6, 12)
    ay_max, brake, drive, grip_drive, grip_lim = _limits(veh, speeds)
    ayl = np.interp(v, speeds, ay_max)
    brl = np.interp(v, speeds, brake)
    grl = np.interp(v, speeds, grip_drive)
    gl = np.interp(v, speeds, grip_lim.astype(float)) > 0.5
    lat = ay > 2.0 * np.abs(ax)                          # mostly cornering
    brk = (ax < -2.0 * ay) & (ax < -2.0)                  # mostly braking
    trc = (ax > 2.0 * ay) & (ax > 2.0) & gl               # mostly accelerating, grip-limited speed
    r_lat, r_brk, r_trc = ay[lat] / ayl[lat], -ax[brk] / brl[brk], ax[trc] / grl[trc]

    def q(x):
        return float(np.percentile(x, pct)) if len(x) >= 10 else None

    bands = []
    edges = np.linspace(speeds[0], speeds[-1], 6)
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (v >= lo) & (v < hi)
        bands.append({"v_kmh": [lo * 3.6, hi * 3.6], "lat": q(ay[sel & lat] / ayl[sel & lat]),
                      "brake": q(-ax[sel & brk] / brl[sel & brk]), "n": int(sel.sum())})
    util_y, ux_b, ux_t = q(r_lat), q(r_brk), q(r_trc)
    cands = [x for x in (ux_b, ux_t) if x is not None]
    util_x = float(np.mean(cands)) if cands else None
    sl = decimate(len(v), 3000)
    sug = ([f"util_y={util_y:.3f}"] if util_y else []) + ([f"util_x={util_x:.3f}"] if util_x else [])
    warn = []
    if util_y and util_y > 1.05:
        warn.append("The car used more lateral grip than the model's tyre limit: the model's grip (muy) or downforce "
                    "is too low, or the ay channel is biased (sensor mounting, sign, units).")
    if not len(r_brk):
        warn.append("No hard braking in the window: util_x from braking is missing.")
    return clean({"pct": pct, "util_y": util_y, "util_x": util_x, "util_x_brake": ux_b, "util_x_traction": ux_t,
                  "n_lat": int(lat.sum()), "n_brake": int(brk.sum()), "n_traction": int(trc.sum()),
                  "bands": bands, "suggest": sug, "warnings": warn,
                  "setup": {"util_x": veh_util(lab, changes, competition)[0], "util_y": veh_util(lab, changes, competition)[1]},
                  "points": {"v": v[sl] * 3.6, "ax": ax[sl] / G, "ay": ay[sl] / G},
                  "model": {"v": speeds * 3.6, "ay_max": ay_max / G, "brake": brake / G, "drive": drive / G,
                            "grip_drive": grip_drive / G}})


def veh_util(lab: Lab, changes, competition):
    veh = lab.lapsim(competition).vehicle(lab.validate(changes, competition))
    return float(veh.util_x), float(veh.util_y)


# --------------------------------------------------------------------------------------------- steady cornering
def steady(lab: Lab, source_id: str, spec: dict, radius: float | None = None, changes=(), competition=None,
           window_s: float = 1.0) -> dict:
    """Steady cornering: windows of window_s where speed and ay are nearly constant. ay from the ay channel, else
    v^2/R (radius) or v*r (yaw rate). Compared with the model's steady-state limit ay_limit(v) at utilisation 1;
    `mu_scale` is the factor on mux and muy for which the model's limit matches the highest steady points."""
    d = logs.extract(lab, source_id, spec, changes, competition)
    veh = lab.lapsim(competition).vehicle(lab.validate(changes, competition)).copy(util_x=1.0, util_y=1.0)
    t, v = d["t"], d["v"]
    if "ay" in d and np.isfinite(d["ay"]).any():
        ay, src = np.abs(d["ay"]), "ay channel"
    elif "yaw" in d:
        ay, src = np.abs(v * d["yaw"]), "speed × yaw rate"
    elif radius:
        ay, src = v ** 2 / float(radius), f"v²/R (R = {float(radius):.2f} m)"
    else:
        raise ValueError("steady cornering needs the ay channel, the yaw rate or the radius")
    dt = float(np.median(np.diff(t)))
    n = max(3, round(window_s / dt))
    vs, ays = _smooth(v, n), _smooth(ay, n)
    dv = np.abs(_deriv(t, vs))
    # steady: speed changes < 0.3 m/s^2, ay above 0.5 g
    st = (dv < 0.3) & (ays > 0.5 * G) & (vs > 3.0)
    if st.sum() < 5:
        raise ValueError("no steady cornering found (speed nearly constant and ay > 0.5 g for about a second)")
    vv, aa = vs[st], ays[st]
    grid = np.linspace(float(vv.min()), float(vv.max()) + 1e-6, 15)    # ay_limit on a speed grid, interpolated

    def lim_on(car, x):
        return np.interp(x, grid, [_ggv.ay_limit(car, float(g)) for g in grid])

    lim = lim_on(veh, vv)
    ratio = aa / lim
    top = ratio >= np.percentile(ratio, 80)                # the highest steady points: at the limit

    def lim_at(scale):
        return lim_on(veh.copy(mux=veh.mux * scale, muy=veh.muy * scale), vv[top])

    lo, hi = 0.5, 1.5
    for _ in range(30):                                    # bisection on the grip scale (ay_limit grows with mu)
        mid = 0.5 * (lo + hi)
        if np.median(lim_at(mid) - aa[top]) < 0:
            lo = mid
        else:
            hi = mid
    scale = 0.5 * (lo + hi)
    sl = decimate(len(vv), 2000)
    vg = np.linspace(max(3.0, float(vv.min()) * 0.8), float(vv.max()) * 1.15, 30)
    return clean({"source": src, "n": int(st.sum()), "ay_max_g": float(aa.max() / G),
                  "v_at_max_kmh": float(vv[int(np.argmax(aa))] * 3.6), "ratio_p90": float(np.percentile(ratio, 90)),
                  "mu_scale": scale, "muy": float(veh.muy * scale), "mux": float(veh.mux * scale),
                  "setup": {"muy": veh.muy, "mux": veh.mux},
                  "suggest": [f"mu*{scale:.3f}"],
                  "points": {"v": vv[sl] * 3.6, "ay": aa[sl] / G},
                  "model": {"v": vg * 3.6, "ay": np.array([_ggv.ay_limit(veh, float(x)) for x in vg]) / G,
                            "ay_fit": np.array([_ggv.ay_limit(veh.copy(mux=veh.mux * scale, muy=veh.muy * scale), float(x))
                                                for x in vg]) / G}})


# --------------------------------------------------------------------------------------------- track from GPS
def _project(lat: np.ndarray, lon: np.ndarray):
    """WGS84 -> local east / north metres (equirectangular around the mean; < 1 mm error over 1 km)."""
    R = 6371008.8
    la0, lo0 = math.radians(float(np.mean(lat))), math.radians(float(np.mean(lon)))
    x = R * (np.radians(lon) - lo0) * math.cos(la0)
    y = R * (np.radians(lat) - la0)
    return x, y


def track_from_xy(x: np.ndarray, y: np.ndarray, closed: bool | None = None, lam: float | None = None,
                  ds: float = 0.5) -> dict:
    """A driven line (x, y samples in metres) as a track: duplicates removed, cut to one lap (closed), fitted with
    cubic smoothing splines x(s), y(s) (smoothing chosen from the data by generalised cross-validation unless
    `lam` is given; closed laps wrapped so the fit is periodic), resampled to ds. The curvature is LapSim's own
    (lts_track_csv for x, y files: smoothing over 4 m, then differentiation), so a saved x, y file gives the same
    track in LapSim as shown here. Checked on a synthetic lap: 0.02-0.3 m of GPS noise -> lap time within 1 % of
    the true line through the same pipeline."""
    from scipy.interpolate import make_smoothing_spline  # noqa: PLC0415

    from fsdsim.perf import lapsim as LS  # noqa: PLC0415

    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    keep = np.r_[True, np.hypot(np.diff(x), np.diff(y)) > 0.05]       # drop standing / held samples
    x, y = x[keep], y[keep]
    if len(x) < 20:
        raise ValueError("fewer than 20 distinct positions")
    sc = np.r_[0.0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
    if sc[-1] < 30:
        raise ValueError(f"the path is only {sc[-1]:.0f} m long")
    d0 = np.hypot(x - x[0], y - y[0])
    # the first return to the start: after the car was > 30 m away, the closest point while within 10 m
    away = np.flatnonzero(d0 > 30.0)
    j = None
    if len(away):
        near = np.flatnonzero((d0 < 10.0) & (np.arange(len(d0)) > away[0]))
        if len(near):
            back = stop = near[0]
            while stop + 1 < len(d0) and d0[stop + 1] < 10.0:
                stop += 1
            j = back + int(np.argmin(d0[back: stop + 1]))
    if closed is None:
        closed = j is not None and sc[j] > 100
    if closed:
        if j is None:
            raise ValueError("the path does not come back to its start: not a closed lap (untick closed)")
        Lp = float(sc[j])
        x, y, sc = x[:j], y[:j], sc[:j]
        P = min(80.0, 0.25 * Lp)                     # wrap: the fit sees the lap as periodic
        pre, post = sc > Lp - P, sc < P
        S_ = np.r_[sc[pre] - Lp, sc, sc[post] + Lp]
        X_, Y_ = np.r_[x[pre], x, x[post]], np.r_[y[pre], y, y[post]]
        gap = float(d0[j])
    else:
        Lp, S_, X_, Y_ = float(sc[-1]), sc, x, y
        gap = float(d0[-1])
    fx, fy = make_smoothing_spline(S_, X_, lam=lam), make_smoothing_spline(S_, Y_, lam=lam)
    n = max(8, int(round(Lp / ds)))
    u = np.linspace(0.0, Lp, n + 1)
    if closed:
        u = u[:-1]
    xs, ys = fx(u), fy(u)
    t = LS._xy_track(xs, ys, ds, 4.0, bool(closed), "measured")
    psi = np.unwrap(np.arctan2(np.gradient(t.y), np.gradient(t.x)))
    turn = float(np.round((psi[-1] - psi[0]) / (2 * math.pi), 2)) if closed else None
    return {"s": t.s, "x": t.x, "y": t.y, "kappa": t.k, "closed": bool(closed), "L": float(t.L),
            "closure_gap_m": gap, "turns": turn, "k_max": float(np.max(np.abs(t.k))), "xy": (xs, ys)}


def track_from_gps(lab: Lab, source_id: str, spec: dict, closed: bool | None = None, lam: float | None = None,
                   changes=(), competition=None) -> dict:
    d = logs.extract(lab, source_id, spec, changes, competition)
    if "lat" not in d or "lon" not in d:
        raise ValueError("choose the latitude and longitude channels")
    lat, lon = d["lat"], d["lon"]
    ok = np.isfinite(lat) & np.isfinite(lon) & (np.abs(lat) > 1e-6)
    if ok.sum() < 20:
        raise ValueError("fewer than 20 valid GPS samples")
    x, y = _project(lat[ok], lon[ok])
    tr = track_from_xy(x, y, closed, lam)
    sl = decimate(len(tr["s"]), 2500)
    out = {k: tr[k] for k in ("closed", "L", "closure_gap_m", "turns", "k_max")}
    out["trace"] = {"s": tr["s"][sl], "x": tr["x"][sl], "y": tr["y"][sl], "kappa": tr["kappa"][sl]}
    out["raw"] = {"x": x[decimate(len(x), 2500)], "y": y[decimate(len(y), 2500)]}
    out["_full"] = tr
    return out


def save_track(lab: Lab, source_id: str, spec: dict, name: str, closed: bool | None = None, lam: float | None = None,
               overwrite: bool = False, changes=(), competition=None) -> str:
    """Write the track as perf/uploads/tracks/<name>.csv with x_m, y_m (0.5 m apart; the format of the "real" FSA
    tracks: LapSim computes the curvature). Copy it into lapsim/LR26_LapSim/tracks and reference it from a
    competition file (lts_track_csv with 'closed' true or false) to use it."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_\-]{0,60}", name or ""):
        raise ValueError("track name: letters, digits, _ and -, up to 60 characters")
    r = track_from_gps(lab, source_id, spec, closed, lam, changes, competition)["_full"]
    folder = os.path.join(logs.UPLOADS, "tracks")       # with the uploaded logs: team data, kept out of git
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, name + ".csv")
    if os.path.exists(path) and not overwrite:
        raise ValueError(f"{name}.csv exists (tick replace)")
    xs, ys = r["xy"]
    with open(path, "w", newline="\n") as f:
        f.write("x_m,y_m\n")
        for x, y in zip(xs, ys):
            f.write(f"{x:.3f},{y:.3f}\n")
    return path


TOOLS = ("coastdown", "grip_use", "steady", "track")


def run_tool(lab: Lab, tool: str, source: str, spec: dict, args=None, changes=(), competition=None, job=None) -> dict:
    """One identification tool (the Logs page's Identify box)."""
    a = args or {}
    if job is not None:
        job.step(0, 1, tool)
    if tool == "coastdown":
        r = coastdown(lab, source, spec, changes, competition)
    elif tool == "grip_use":
        r = grip_use(lab, source, spec, changes, competition, float(a.get("pct", 95.0)))
    elif tool == "steady":
        rad = a.get("radius")
        r = steady(lab, source, spec, float(rad) if rad not in (None, "") else None, changes, competition)
    elif tool == "track":
        c = a.get("closed")
        r = track_from_gps(lab, source, spec, None if c in (None, "", "auto") else bool(c), None, changes, competition)
        r.pop("_full", None)
    else:
        raise ValueError(f"unknown tool {tool!r} (use {', '.join(TOOLS)})")
    r["tool"] = tool
    if job is not None:
        job.step(1, 1, "done")
    return r
