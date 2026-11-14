"""LR26 LapSim events in Python: line-by-line port of LR26_LapSim's event pipeline.

    lts_corner_speed  -> corner_speed(veh, k)            (vectorised bisection, 45 steps)
    lts_qss_solve     -> qss_solve(veh, k, ds, v0, v_end)
    lts_event_vehicle -> event_vehicle(veh, ev)
    lts_run_event     -> run_event(veh, ev)
    lts_score         -> score(T, sc)
    lts_run_all       -> run_all(veh, comp)
    lts_track_basic / lts_track_csv / competition_*.m -> load_competition(path)

It reads the MATLAB files themselves (config/lr26_params.m, config/competition_*.m, tracks/*.csv),
so MATLAB and the simulator always use the same car, tracks, driver utilisation and scoring:

    from fsdsim.perf import lapsim
    ls = lapsim.LapSim.load("lapsim/LR26_LapSim")
    res = ls.run(changes=["m=300", "mu*1.05"])        # all events, times and points, like run_lapsim
"""

from __future__ import annotations

import csv
import math
import os
import re
from dataclasses import dataclass, field

import numpy as np

from . import efficiency as _efficiency
from .params import PerfParams, _eval, matlab_statements

EVENT_ORDER = ("acceleration", "skidpad", "autocross", "endurance")


# ---------------------------------------------------------------------------- #
# vectorised vehicle model (lts_wheel_loads / lts_axle_state / lts_resistance / lts_powertrain)
# ---------------------------------------------------------------------------- #
def _resistance(veh: PerfParams, v):
    DF = 0.5 * veh.rho * veh.c_z * veh.A * veh.aero_eff * v * v
    return 0.5 * veh.rho * veh.c_w * veh.A * veh.aero_eff * v * v + veh.f_roll * (veh.m * veh.g + DF)


def _v_max(veh: PerfParams) -> float:
    r = veh.circ_ref / (2 * math.pi)
    return veh.n_max * 2 * math.pi / 60 / veh.i_trans * r


def _axle_state(veh: PerfParams, v, ax, ay):
    """Vectorised lts_axle_state: (Fx_rem N x 2, u N x 2, pxAx N x 2, pyAx N x 2)."""
    v, ax, ay = np.broadcast_arrays(np.atleast_1d(np.asarray(v, float)), np.asarray(ax, float), np.asarray(ay, float))
    l = veh.lf + veh.lr  # noqa: E741
    m = veh.m
    DF = 0.5 * veh.rho * veh.c_z * veh.A * veh.aero_eff * v * v
    Ff = m * veh.g * veh.lr / l + DF * veh.aero_bal_f - m * ax * veh.h_cog / l
    Fr = m * veh.g * veh.lf / l + DF * (1 - veh.aero_bal_f) + m * ax * veh.h_cog / l
    dyf = m * ay * veh.h_cog / veh.sf * veh.lr / l
    dyr = m * ay * veh.h_cog / veh.sr * veh.lf / l
    Fz = np.maximum(np.stack([Ff / 2 - dyf, Ff / 2 + dyf, Fr / 2 - dyr, Fr / 2 + dyr], axis=1), 0.0)
    px = np.maximum(Fz * (veh.mux + veh.dmux_dFz * (Fz - veh.Fz0)) * veh.util_x, 0.0)
    py = np.maximum(Fz * (veh.muy + veh.dmuy_dFz * (Fz - veh.Fz0)) * veh.util_y, 0.0)
    pxAx = np.stack([px[:, 0] + px[:, 1], px[:, 2] + px[:, 3]], axis=1)
    pyAx = np.stack([py[:, 0] + py[:, 1], py[:, 2] + py[:, 3]], axis=1)
    Fyt = np.abs(m * ay)
    Fy = np.stack([Fyt * veh.lr / l, Fyt * veh.lf / l], axis=1)
    u = Fy / np.maximum(pyAx, 1e-9)
    e = veh.tire_exp
    Fx_rem = pxAx * np.maximum(1 - np.minimum(u, 1) ** e, 0) ** (1 / e)
    return Fx_rem, u, pxAx, pyAx, Fz


class _Fast:
    """Scalar lts_axle_state / lts_powertrain / lts_ax_limits in plain floats (the QSS loops call
    them ~10^5 times per lap; numpy is slower than math for scalars)."""

    def __init__(self, veh: PerfParams):
        self.v = veh
        self.l = veh.lf + veh.lr
        self.kDF = 0.5 * veh.rho * veh.c_z * veh.A * veh.aero_eff
        self.kD = 0.5 * veh.rho * veh.c_w * veh.A * veh.aero_eff
        self.r = veh.circ_ref / (2 * math.pi)
        self.vmax = veh.n_max * 2 * math.pi / 60 / veh.i_trans * self.r
        self.P = veh.P_max * veh.eta_motor if veh.P_at_accumulator > 0.5 else veh.P_max
        self.meff = veh.m * veh.e_i
        self.inv_e = 1.0 / veh.tire_exp

    def rem(self, v: float, ax: float, ay: float):
        p = self.v
        m, l = p.m, self.l  # noqa: E741
        DF = self.kDF * v * v
        Ff = m * p.g * p.lr / l + DF * p.aero_bal_f - m * ax * p.h_cog / l
        Fr = m * p.g * p.lf / l + DF * (1 - p.aero_bal_f) + m * ax * p.h_cog / l
        dyf = m * ay * p.h_cog / p.sf * p.lr / l
        dyr = m * ay * p.h_cog / p.sr * p.lf / l
        out = []
        Fyt = abs(m * ay)
        for (fa, fb, share) in ((Ff / 2 - dyf, Ff / 2 + dyf, p.lr / l), (Fr / 2 - dyr, Fr / 2 + dyr, p.lf / l)):
            pxa = pya = 0.0
            for fz in (max(fa, 0.0), max(fb, 0.0)):
                pxa += max(fz * (p.mux + p.dmux_dFz * (fz - p.Fz0)) * p.util_x, 0.0)
                pya += max(fz * (p.muy + p.dmuy_dFz * (fz - p.Fz0)) * p.util_y, 0.0)
            u = Fyt * share / max(pya, 1e-9)
            out.append(pxa * max(1 - min(u, 1.0) ** p.tire_exp, 0.0) ** self.inv_e)
        return out

    def powertrain(self, v: float):
        vv = max(v, 0.1)
        T = min(self.v.T_motor, self.P / (vv / self.r * self.v.i_trans))
        F_set = T * self.v.i_trans * self.v.eta_g / self.r
        if v > self.vmax * (1 + 1e-9):
            F_set = 0.0
        return F_set, self.P * self.v.eta_g / vv

    def resistance(self, v: float) -> float:
        return self.kD * v * v + self.v.f_roll * (self.v.m * self.v.g + self.kDF * v * v)

    def ax_acc(self, v: float, k: float, ax: float) -> float:
        ay = v * v * k
        Fres = self.resistance(v)
        F_set, F_cap = self.powertrain(v)
        for _ in range(4):
            rf, rr = self.rem(v, ax, ay)
            F = min(F_set, rf) + min(F_set, rr) if self.v.drive_awd > 0.5 else min(F_set, rr)
            ax = (min(F, F_cap) - Fres) / self.meff
        return ax

    def ax_dec(self, v: float, k: float, ax: float) -> float:
        ay = v * v * k
        Fres = self.resistance(v)
        b = self.v.brake_bias
        for _ in range(4):
            rf, rr = self.rem(v, ax, ay)
            F = rf + rr if math.isnan(b) else min(rf / max(b, 1e-6), rr / max(1 - b, 1e-6))
            ax = -(F + Fres) / self.meff
        return ax


# ---------------------------------------------------------------------------- #
# solver
# ---------------------------------------------------------------------------- #
def corner_speed(veh: PerfParams, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """lts_corner_speed: (vc, limit) with limit 0 = powertrain/speed, 1 = front axle, 2 = rear axle."""
    k = np.asarray(k, float).ravel()
    v_top = min(_v_max(veh), veh.v_lim)

    def feasible(v):
        Fx_rem, u, *_ = _axle_state(veh, v, 0.0, v * v * k)
        have = Fx_rem[:, 0] + Fx_rem[:, 1] if veh.drive_awd > 0.5 else Fx_rem[:, 1]
        return np.all(u <= 1, axis=1) & (have >= _resistance(veh, v))

    lo = np.zeros_like(k)
    hi = np.full_like(k, v_top)
    ok_top = feasible(hi)
    for _ in range(45):
        mid = 0.5 * (lo + hi)
        ok = feasible(mid)
        lo = np.where(ok, mid, lo)
        hi = np.where(ok, hi, mid)
    vc = lo.copy()
    vc[ok_top] = v_top
    _, u, *_ = _axle_state(veh, vc, 0.0, vc * vc * k)
    limit = 1 + (u[:, 1] >= u[:, 0]).astype(int)
    limit[ok_top] = 0
    return vc, limit


def qss_solve(veh: PerfParams, k, ds: float, v0: float = math.nan, v_end: float = math.nan):
    """lts_qss_solve: (v, vc, limit) - forward and backward pass, minimum of both."""
    k = np.asarray(k, float).ravel()
    n = len(k)
    vc, limit = corner_speed(veh, k)
    f = _Fast(veh)
    vf = np.zeros(n)
    vf[0] = vc[0] if math.isnan(v0) else min(v0, vc[0])
    ax = 0.0
    kk, vcc = k.tolist(), vc.tolist()
    vfl = [float(vf[0])] + [0.0] * (n - 1)
    for i in range(n - 1):
        ax = f.ax_acc(vfl[i], kk[i], ax)
        vfl[i + 1] = min(math.sqrt(max(vfl[i] ** 2 + 2 * ax * ds, 0.0)), vcc[i + 1])
    vbl = [0.0] * n
    vbl[-1] = vcc[-1] if math.isnan(v_end) else min(v_end, vcc[-1])
    axb = 0.0
    for i in range(n - 1, 0, -1):
        axb = f.ax_dec(vbl[i], kk[i], axb)
        vbl[i - 1] = min(math.sqrt(vbl[i] ** 2 - 2 * axb * ds), vcc[i - 1])
    v = np.minimum(np.array(vfl), np.array(vbl))
    return v, vc, limit


def _time_from_speed(v: np.ndarray, ds: float) -> np.ndarray:
    vm = 0.5 * (v[:-1] + v[1:])
    return np.concatenate([[0.0], np.cumsum(ds / np.maximum(vm, 1e-3))])


def _signals(veh: PerfParams, v: np.ndarray, k: np.ndarray, ds: float) -> dict:
    """The parts of lts_run_event's signals() the dashboard needs: t, ax, ay, P_wheel, P_batt, E."""
    t = _time_from_speed(v, ds)
    axs = np.diff(v * v) / (2 * ds)
    ax = np.concatenate([axs, axs[-1:]])
    ay = v * v * k[: len(v)]
    Fx_tot = veh.m * veh.e_i * ax + _resistance(veh, v)
    P_wheel = Fx_tot * v
    eta_m = veh.eta_motor
    emap = _efficiency.active()
    if emap is not None:                    # efficiency map instead of the constant (perf.efficiency)
        n_sets = 2.0 if veh.drive_awd > 0.5 else 1.0
        T_mot = np.maximum(Fx_tot, 0.0) * veh.r / (veh.i_trans * veh.eta_g) / n_sets
        eta_m = emap.eta(T_mot, v / veh.r * veh.i_trans * 60.0 / (2 * math.pi))
    P_batt = P_wheel / (veh.eta_g * eta_m)
    neg = P_wheel < 0
    P_batt[neg] = -np.minimum(-P_wheel[neg], veh.P_regen) * veh.eta_regen
    E = np.concatenate([[0.0], np.cumsum(0.5 * (P_batt[:-1] + P_batt[1:]) * np.diff(t))])
    return {"t": t, "ax": ax, "ay": ay, "P_wheel": P_wheel, "P_batt": P_batt, "E": E}


# ---------------------------------------------------------------------------- #
# tracks and competition
# ---------------------------------------------------------------------------- #
@dataclass
class Track:
    name: str
    s: np.ndarray
    k: np.ndarray
    x: np.ndarray
    y: np.ndarray
    closed: bool

    @property
    def L(self) -> float:  # noqa: N802
        return float(self.s[-1])

    @property
    def ds(self) -> float:
        return float(self.s[1] - self.s[0])


def track_basic(kind: str, a: float | None = None, ds: float = 0.25) -> Track:
    """lts_track_basic('acceleration', L, ds) / ('skidpad', R, ds)."""
    if kind == "acceleration":
        L = 75.0 if a is None else float(a)
        s = np.arange(0.0, L + 1e-9, ds)
        if s[-1] < L - 1e-9:
            s = np.append(s, L)
        z = np.zeros_like(s)
        return Track(f"Acceleration {L:g} m", s, z.copy(), z.copy(), s.copy(), False)
    if kind == "skidpad":
        R = 15.25 / 2 + 1.5 if a is None else float(a)
        L = 2 * math.pi * R
        n = max(8, round(L / ds))
        s = np.arange(n + 1) * (L / n)
        return Track(f"Skidpad R = {R:.3f} m", s, -np.ones_like(s) / R, R * np.cos(s / R) - R, -R * np.sin(s / R), True)
    raise ValueError(f"unknown basic track '{kind}'")


def track_csv(path: str, closed: bool = False, ds: float = 0.5, name: str | None = None,
              smooth_window: float = 4.0, method: str = "smooth", curv_filter: float = 5.0) -> Track:
    """lts_track_csv: s,x,y,kappa (used as is) or x,y (curvature computed).

    method for x,y files: 'smooth' (default: smooth the coordinates, then differentiate) or 'spline'
    (cubic spline through the points, analytic curvature, moving average of the curvature over
    `curv_filter` metres: how the Performance department's LionsLTS prepares its tracks)."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    head = [h.strip().lower() for h in rows[0]]
    data = np.array([[float(c) for c in r] for r in rows[1:] if r and r[0].strip()], float)
    name = name or os.path.splitext(os.path.basename(path))[0]
    ki = next((i for i, h in enumerate(head) if "kappa" in h), None)
    if ki is not None:
        col = lambda p: next(i for i, h in enumerate(head) if h.startswith(p))  # noqa: E731
        s, x, y, k = data[:, col("s")], data[:, col("x")], data[:, col("y")], data[:, ki]
        if closed:
            dsf = s[1] - s[0]
            s, x, y, k = np.append(s, s[-1] + dsf), np.append(x, x[0]), np.append(y, y[0]), np.append(k, k[0])
        L = s[-1]
        n = max(2, round(L / ds))
        sg = np.linspace(0.0, L, n + 1)
        return Track(name, sg, np.interp(sg, s, k), np.interp(sg, s, x), np.interp(sg, s, y), closed)
    if str(method).lower() == "spline":
        return _spline_track(data[:, 0], data[:, 1], ds, curv_filter, closed, name)
    return _xy_track(data[:, 0], data[:, 1], ds, smooth_window, closed, name)


def _movmean(z: np.ndarray, nw: int) -> np.ndarray:
    """MATLAB movmean(z, nw) with shrinking windows at the ends (even nw: one more sample before than after)."""
    if nw <= 1:
        return z.copy()
    lo, hi = (nw - 1) // 2 + (nw - 1) % 2, (nw - 1) // 2
    c = np.concatenate([[0.0], np.cumsum(z)])
    i = np.arange(len(z))
    a, b = np.maximum(0, i - lo), np.minimum(len(z), i + hi + 1)
    return (c[b] - c[a]) / (b - a)


def _spline_track(x, y, ds, width, closed, name) -> Track:
    """lts_track_csv 'spline' method. x(s), y(s) are not-a-knot cubic splines of the arc length through all
    points (a closed track is wrapped by a few knots at both ends, so no periodic-spline toolbox is
    needed); the curvature comes from the spline derivatives and is averaged over `width` metres."""
    from scipy.interpolate import CubicSpline

    keep = np.concatenate([[True], np.hypot(np.diff(x), np.diff(y)) > 1e-6])
    x, y = x[keep], y[keep]
    if closed and math.hypot(x[-1] - x[0], y[-1] - y[0]) > 1e-6:
        x, y = np.append(x, x[0]), np.append(y, y[0])
    sc = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    L = float(sc[-1])
    n = max(2, math.ceil(L / ds))
    s = np.linspace(0.0, L, n + 1)
    if closed:
        m = min(6, len(sc) - 2)
        sx = np.concatenate([sc[-m - 1:-1] - L, sc, sc[1:m + 1] + L])
        xx = np.concatenate([x[-m - 1:-1], x, x[1:m + 1]])
        yy = np.concatenate([y[-m - 1:-1], y, y[1:m + 1]])
    else:
        sx, xx, yy = sc, x, y
    cx, cy = CubicSpline(sx, xx), CubicSpline(sx, yy)
    dx, dy, ddx, ddy = cx(s, 1), cy(s, 1), cx(s, 2), cy(s, 2)
    k = (dx * ddy - dy * ddx) / np.maximum((dx * dx + dy * dy) ** 1.5, 1e-9)
    nw = max(1, round(width / (s[1] - s[0])))
    if nw > 1:
        if closed:
            kk = k[:-1]
            kp = _movmean(np.concatenate([kk[-nw:], kk, kk[:nw]]), nw)
            k = np.append(kp[nw:-nw], kp[nw])
        else:
            k = _movmean(k, nw)
    if closed:
        k[-1] = k[0]
    return Track(name, s, k, cx(s), cy(s), closed)


def _xy_track(x, y, ds, win, closed, name) -> Track:
    keep = np.concatenate([[True], np.hypot(np.diff(x), np.diff(y)) > 1e-6])
    x, y = x[keep], y[keep]
    if closed and math.hypot(x[-1] - x[0], y[-1] - y[0]) > 1e-6:
        x, y = np.append(x, x[0]), np.append(y, y[0])

    def resample(x, y):
        sc = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
        n = max(2, math.ceil(sc[-1] / ds))
        s = np.linspace(0.0, sc[-1], n + 1)
        try:
            from scipy.interpolate import CubicSpline  # MATLAB interp1(..., 'spline')

            return s, CubicSpline(sc, x)(s), CubicSpline(sc, y)(s)
        except ImportError:  # pragma: no cover
            return s, np.interp(s, sc, x), np.interp(s, sc, y)

    s, x, y = resample(x, y)
    nw = max(1, round(win / ds))
    if nw > 1:
        def movmean(z):
            half_lo, half_hi = (nw - 1) // 2 + (nw - 1) % 2, (nw - 1) // 2
            out = np.empty_like(z)
            for i in range(len(z)):
                out[i] = z[max(0, i - half_lo): i + half_hi + 1].mean()
            return out

        if closed:
            def pmov(z):
                zz = z[:-1]
                zp = movmean(np.concatenate([zz[-nw:], zz, zz[:nw]]))
                return np.append(zp[nw:-nw], zp[nw])
            x, y = pmov(x), pmov(y)
        else:
            x, y = movmean(x), movmean(y)
        s, x, y = resample(x, y)
    h = s[1] - s[0]
    if closed:
        def pd(z):
            return (np.roll(z, -1) - np.roll(z, 1)) / (2 * h)
        zx, zy = x[:-1], y[:-1]
        dx, dy = pd(zx), pd(zy)
        k = (dx * pd(dy) - dy * pd(dx)) / np.maximum((dx * dx + dy * dy) ** 1.5, 1e-9)
        k = np.append(k, k[0])
    else:
        dx, dy = np.gradient(x, h), np.gradient(y, h)
        k = (dx * np.gradient(dy, h) - dy * np.gradient(dx, h)) / np.maximum((dx * dx + dy * dy) ** 1.5, 1e-9)
    return Track(name, s, k, x, y, closed)


@dataclass
class Event:
    name: str
    type: str
    track: Track | None = None
    util_x: float = 1.0
    util_y: float = 1.0
    v0: float = math.nan
    laps: int = 1
    timing: str = "single"
    score: dict = field(default_factory=dict)


@dataclass
class Competition:
    name: str
    events: dict           # name -> Event, in file order
    source: str = ""


def _split_args(s: str) -> list[str]:
    out, depth, cur, q = [], 0, "", False
    for ch in s:
        if ch == "'":
            q = not q
        if not q and ch in "([{":
            depth += 1
        elif not q and ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0 and not q:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def _call(expr: str):
    m = re.fullmatch(r"\s*(\w+)\s*\((.*)\)\s*", expr, re.S)
    return (m.group(1), _split_args(m.group(2))) if m else (None, [])


def _value(tok: str, env: dict):
    tok = tok.strip()
    if tok.startswith("'") and tok.endswith("'"):
        return tok[1:-1]
    if tok in ("true", "false"):
        return tok == "true"
    return _eval(tok, env)


def load_competition(path: str) -> Competition:
    """Read competition_*.m: events, tracks, driver utilisation, laps, timing and scoring constants."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    main = re.split(r"(?m)^\s*function\b", text)
    body = main[1] if len(main) > 1 else text           # the competition function only (not its helpers)
    here = os.path.dirname(os.path.abspath(path))
    paths = {"here": here, "tdir": os.path.normpath(os.path.join(here, "..", "tracks"))}
    comp_name, events, cur, env = os.path.basename(path), {}, None, {}
    for st in matlab_statements(body):
        m = re.match(r"^(\w+)\s*=\s*(.+)$", st)
        if m and m.group(1) in ("here", "tdir"):
            fn, args = _call(m.group(2))
            if fn == "fullfile":
                parts = [paths.get(a, a.strip("'")) for a in args]
                paths[m.group(1)] = os.path.normpath(os.path.join(*parts))
            continue
        m = re.match(r"^(\w+)\s*=\s*base_event\((.+)\)$", st)
        if m:
            t = m.group(2).strip().strip("'")
            cur = Event(t, t)
            continue
        m = re.match(r"^\w+\.name\s*=\s*'(.*)'$", st)
        if m and cur is None:
            comp_name = m.group(1)
            continue
        m = re.match(r"^\w+\.events\.(\w+)\s*=\s*\w+$", st)
        if m and cur is not None:
            cur.name = m.group(1)
            events[m.group(1)] = cur
            cur = Event(cur.name, cur.type, cur.track, cur.util_x, cur.util_y, cur.v0, cur.laps, cur.timing, dict(cur.score))
            continue
        m = re.match(r"^\w+\.(\w+)\s*=\s*(.+)$", st)
        if not m or cur is None:
            continue
        fld, expr = m.groups()
        if fld == "track":
            fn, args = _call(expr)
            if fn == "lts_track_basic":
                vals = [_value(a, env) for a in args]
                cur.track = track_basic(vals[0], *(float(v) for v in vals[1:3]))
            elif fn == "lts_track_csv":
                f_fn, f_args = _call(args[0])
                file = (os.path.join(*[paths.get(a, a.strip("'")) for a in f_args]) if f_fn == "fullfile"
                        else args[0].strip("'"))
                if not os.path.isabs(file):
                    file = os.path.join(here, file)
                opts = {}
                if len(args) > 1:
                    s_fn, s_args = _call(args[1])
                    if s_fn == "struct":
                        opts = {s_args[i].strip("'"): _value(s_args[i + 1], env) for i in range(0, len(s_args) - 1, 2)}
                cur.track = track_csv(os.path.normpath(file), bool(opts.get("closed", False)), float(opts.get("ds", 0.5)),
                                      opts.get("name"), float(opts.get("smooth_window", 4.0)),
                                      str(opts.get("method", "smooth")), float(opts.get("curv_filter", 5.0)))
        elif fld == "score":
            fn, args = _call(expr)
            if fn == "score_def":
                v = [float(_value(a, env)) for a in args]
                cur.score = dict(zip(("Pmax", "Pmin", "Tmin", "Tmax", "n"), v))
        elif fld == "timing":
            cur.timing = expr.strip().strip("'")
        elif fld in ("util_x", "util_y", "v0"):
            setattr(cur, fld, float(_value(expr, env)))
        elif fld == "laps":
            cur.laps = int(round(float(_value(expr, env))))
    return Competition(comp_name, events, path)


# ---------------------------------------------------------------------------- #
# events
# ---------------------------------------------------------------------------- #
def event_vehicle(veh: PerfParams, ev: Event) -> PerfParams:
    """lts_event_vehicle: driver utilisation; endurance torque scale and power cap."""
    ch = {"util_x": ev.util_x, "util_y": ev.util_y}
    if ev.type.lower() == "endurance":
        ch["T_motor"] = veh.T_motor * veh.endurance_torque_scale
        ch["P_max"] = min(veh.P_max, veh.P_max_endurance)
    return veh.copy(**ch)


def score(T: float, sc: dict) -> float:
    """lts_score: P = Pmin + (Pmax - Pmin) * ((Tmax - T)/(Tmax - Tmin))^n, clipped to [Pmin, Pmax]."""
    if not sc or "Tmin" not in sc or T is None or math.isnan(T):
        return math.nan
    if T <= sc["Tmin"]:
        return sc["Pmax"]
    if T >= sc["Tmax"]:
        return sc["Pmin"]
    return sc["Pmin"] + (sc["Pmax"] - sc["Pmin"]) * ((sc["Tmax"] - T) / (sc["Tmax"] - sc["Tmin"])) ** sc["n"]


def run_event(veh: PerfParams, ev: Event, solver: str = "qss") -> dict:
    """lts_run_event: event time T (used for scoring), lap times, speeds, energy, speed trace.
    solver: a name registered in perf.solvers ("qss" = LapSim's own)."""
    vehE = event_vehicle(veh, ev)
    if solver != "qss":
        from . import solvers as _solvers  # noqa: PLC0415

        _solve = _solvers.get(solver)
    else:
        _solve = qss_solve
    trk = ev.track
    k, ds = trk.k, trk.ds
    N = len(k) - 1
    if trk.closed:
        kr = np.concatenate([np.tile(k[:-1], 3), k[:1]])
        vr, vcr, limr = _solve(vehE, kr, ds, 0.0, math.nan)
        tr = _time_from_speed(vr, ds)
        v = vr[N: 2 * N + 1]
        t_first, t_lap = tr[N] - tr[0], tr[2 * N] - tr[N]
        E_first = _signals(vehE, vr[: N + 1], k, ds)["E"][-1]
    else:
        v, _, _ = _solve(vehE, k, ds, ev.v0, math.nan)
        tt = _time_from_speed(v, ds)
        t_first = t_lap = tt[-1]
        E_first = math.nan
    sig = _signals(vehE, v, k, ds)
    if ev.timing == "flying_lap":
        T, E_total = ev.laps * t_lap, ev.laps * sig["E"][-1]
    elif ev.timing == "standing_start":
        T, E_total = t_first + (ev.laps - 1) * t_lap, E_first + (ev.laps - 1) * sig["E"][-1]
    else:
        T, E_total = t_lap, sig["E"][-1]
    return {"event": ev.name, "type": ev.type, "track": trk.name, "T": float(T), "t_lap": float(t_lap),
            "t_first": float(t_first), "laps": ev.laps, "v_avg": float(trk.L / t_lap), "v_max": float(v.max()),
            "E_total": float(E_total), "E_lap": float(sig["E"][-1]), "util": [ev.util_x, ev.util_y],
            "s": trk.s, "v": v, "t": sig["t"], "x": trk.x, "y": trk.y, "k": k, "ax": sig["ax"], "ay": sig["ay"],
            "P_wheel": sig["P_wheel"], "P_batt": sig["P_batt"], "E": sig["E"], "veh": vehE}


def run_all(veh: PerfParams, comp: Competition, events=None, solver: str = "qss") -> dict:
    """lts_run_all: {'events': {name: result incl. 'points'}, 'total': points}."""
    names = list(events or comp.events)
    out = {}
    for n in names:
        r = run_event(veh, comp.events[n], solver)
        r["points"] = score(r["T"], comp.events[n].score)
        out[n] = r
    total = sum(r["points"] for r in out.values() if not math.isnan(r["points"]))
    return {"events": out, "total": total}


# ---------------------------------------------------------------------------- #
# parameter changes: the same strings for the port, MATLAB and the simulator
# ---------------------------------------------------------------------------- #
_CHANGE = re.compile(r"\s*([A-Za-z_][\w.]*)\s*([=*+-])\s*(.+?)\s*")
EVENT_FIELDS = ("util_x", "util_y")


def parse_change(change: str):
    """'m=300' -> ('m', '=', 300.0); 'autocross.util_y*0.9' -> ('autocross.util_y', '*', 0.9)."""
    m = _CHANGE.fullmatch(str(change))
    if not m:
        raise ValueError(f"cannot parse '{change}' (use name=value, name*factor, name+delta or name-delta)")
    name, op, raw = m.groups()
    return name, op, float(_eval(raw, {}))


def split_changes(changes, event_names=EVENT_ORDER):
    """(vehicle changes, event changes) - event changes are '<event>.util_x|util_y <op> value'
    (driver friction utilisation of a LapSim event); everything else is a vehicle change."""
    veh, ev = [], []
    for c in changes or []:
        name, _, _ = parse_change(c)
        head, _, fld = name.partition(".")
        (ev if head in event_names and fld in EVENT_FIELDS else veh).append(str(c).strip())
    return veh, ev


def lapsim_changes(changes, event_names=EVENT_ORDER) -> list[str]:
    """The vehicle changes LapSim knows (its fields and lts_set_param aliases); the rest
    (yaw_inertia, actuators.max_steer, ...) only exist in the simulator."""
    from .params import ALIASES  # noqa: PLC0415

    known = set(PerfParams().numeric_fields()) | set(ALIASES)
    veh, _ = split_changes(changes, event_names)
    return [c for c in veh if parse_change(c)[0] in known or parse_change(c)[0].lower() in ALIASES]


def apply_event_changes(comp: Competition, changes) -> Competition:
    import copy  # noqa: PLC0415

    if not changes:
        return comp
    out = Competition(comp.name, {n: copy.copy(e) for n, e in comp.events.items()}, comp.source)
    for c in changes:
        name, op, x = parse_change(c)
        evn, fld = name.split(".", 1)
        if evn not in out.events:
            raise KeyError(f"unknown event '{evn}' in '{c}'")
        e = out.events[evn]
        cur = getattr(e, fld)
        setattr(e, fld, x if op == "=" else cur * x if op == "*" else cur + x if op == "+" else cur - x)
    return out


def _num(x: float) -> str:
    return "NaN" if math.isnan(x) else ("Inf" if x > 0 else "-Inf") if math.isinf(x) else f"{x:.15g}"


def matlab_change(change: str) -> str:
    """One change as MATLAB code for LR26_LapSim (values are re-formatted numbers, never raw text)."""
    from .params import ALIASES  # noqa: PLC0415

    change = re.sub(r"[^\x20-\x7e]", " ", str(change))     # it may end up in a % comment: one line only
    try:
        name, op, x = parse_change(change)
    except (ValueError, SyntaxError, KeyError, TypeError):
        return f"% (not understood) {change}"
    head, _, fld = name.partition(".")
    if head in EVENT_ORDER and fld in EVENT_FIELDS:
        tgt = f"comp.events.{head}.{fld}"
        rhs = _num(x) if op == "=" else f"{tgt} {op} {_num(x)}"
        return f"{tgt} = {rhs};"
    fields = ALIASES.get(name.lower(), (name,))
    if "." in name or not all(f in PerfParams().numeric_fields() for f in fields):
        return f"% (simulator only, not a LapSim parameter) {change}"
    if op == "=":
        return f"veh = lts_set_param(veh, '{name}', {_num(x)});"
    if op == "*":
        return f"veh = lts_set_param(veh, '{name}', {_num(x)}, 'scale');"
    return " ".join(f"veh.{f} = veh.{f} {op} {_num(x)};" for f in fields)


# ---------------------------------------------------------------------------- #
# the LR26_LapSim folder
# ---------------------------------------------------------------------------- #
@dataclass
class LapSim:
    """The MATLAB LapSim folder: vehicle (lr26_params.m) and competition (competition_*.m)."""

    root: str
    params_file: str
    competition_file: str
    veh: PerfParams
    comp: Competition
    report: dict

    @classmethod
    def load(cls, root: str, params: str | None = None, competition: str | None = None) -> "LapSim":
        root = os.path.abspath(root)
        params = params or os.path.join(root, "config", "lr26_params.m")
        competition = competition or os.environ.get("FSG_LAPSIM_COMPETITION") or None   # e.g. competition_fsa2025_real.m
        if competition and not os.path.isabs(competition) and not os.path.exists(competition):
            competition = os.path.join(root, "config", competition)
        if competition is None:
            cands = sorted(f for f in os.listdir(os.path.join(root, "config")) if re.match(r"competition_.*\.m$", f))
            if not cands:
                raise FileNotFoundError(f"no config/competition_*.m in {root}")
            competition = os.path.join(root, "config", cands[0])
        veh, rep = PerfParams.from_matlab(params, base=PerfParams())
        comp = load_competition(competition)
        events = {n: {"util_x": e.util_x, "util_y": e.util_y} for n, e in comp.events.items()}
        veh = veh.copy(events=events, source=os.path.relpath(params, os.path.dirname(root)))
        return cls(root, params, competition, veh, comp, rep)

    def vehicle(self, changes=()) -> PerfParams:
        lapsim_ch = lapsim_changes(changes, tuple(self.comp.events))
        return self.veh.with_changes(*lapsim_ch) if lapsim_ch else self.veh

    def competition(self, changes=()) -> Competition:
        return apply_event_changes(self.comp, split_changes(changes, tuple(self.comp.events))[1])

    def run(self, changes=(), events=None, solver: str = "qss") -> dict:
        return run_all(self.vehicle(changes), self.competition(changes), events, solver)

    @property
    def competition_function(self) -> str:
        return os.path.splitext(os.path.basename(self.competition_file))[0]

    def mtime(self) -> float:
        """Latest modification time of the MATLAB files the simulator reads."""
        files = [self.params_file, self.competition_file]
        return max(os.path.getmtime(f) for f in files if os.path.exists(f))


def find_lapsim_dir(start: str | None = None) -> str | None:
    """LR26_LapSim folder: $FSG_LAPSIM_DIR, else <repo>/lapsim/LR26_LapSim found upwards from `start`
    (symlinks resolved, so it also works from a colcon --symlink-install share folder)."""
    env = os.environ.get("FSG_LAPSIM_DIR")
    if env and os.path.isfile(os.path.join(env, "config", "lr26_params.m")):
        return os.path.abspath(env)
    starts = [start] if start else []
    starts.append(os.path.dirname(os.path.abspath(__file__)))
    for st in starts:
        d = os.path.realpath(st)
        for _ in range(12):
            cand = os.path.join(d, "lapsim", "LR26_LapSim")
            if os.path.isfile(os.path.join(cand, "config", "lr26_params.m")):
                return cand
            parent = os.path.dirname(d)
            if parent == d:
                break
            d = parent
    return None
