"""Quasi-steady-state lap time solver (port of the LapSim solver, documentation Fig. 3.1).

1. Corner speed at every point: the highest speed at which both axles can hold the
   lateral force (u <= 1) and the driven axle can still overcome drag and rolling
   resistance (ax_acc >= 0).
2. Forward pass with full acceleration, backward pass with full braking, minimum of
   the three profiles.
3. Closed tracks run 3 laps back to back: lap 1 is the standing start, lap 2 the
   flying lap (as in LapSim).

Events: `acceleration` (75 m from standstill), `skidpad` (radius 9.125 m, lane centre),
and any centre line (`run_path`) such as the fsdsim autocross / trackdrive layouts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .model import ax_limits, axle_state, powertrain, resistance
from .params import PerfParams

SKIDPAD_RADIUS = 9.125
ACCEL_LENGTH = 75.0


@dataclass
class QssResult:
    event: str
    time: float                      # event time [s] (flying lap for closed tracks)
    s: np.ndarray                    # distance [m]
    v: np.ndarray                    # speed [m/s]
    k: np.ndarray                    # curvature [1/m]
    ax: np.ndarray = field(default_factory=lambda: np.zeros(0))
    ay: np.ndarray = field(default_factory=lambda: np.zeros(0))
    lap_times: list = field(default_factory=list)
    v_corner: np.ndarray = field(default_factory=lambda: np.zeros(0))
    v_first: np.ndarray = field(default_factory=lambda: np.zeros(0))   # closed tracks: standing-start lap

    @property
    def v_max(self) -> float:
        return float(self.v.max()) if len(self.v) else 0.0

    @property
    def v_mean(self) -> float:
        return float(self.s[-1] / self.time) if self.time > 0 else 0.0


# ---------------------------------------------------------------------------- #
def steady_ok(veh: PerfParams, v: float, k: float) -> bool:
    """Can the car hold speed v on curvature k (steady state, ax = 0)?"""
    ay = v * v * k
    Fx_rem, u, *_ = axle_state(veh, v, 0.0, ay)
    if u.max() > 1.0:
        return False
    F_set, F_cap, v_max = powertrain(veh, v)
    if v > v_max:
        return False
    if veh.drive_awd > 0.5:
        F_drive = min(F_set, Fx_rem[0]) + min(F_set, Fx_rem[1])
    else:
        F_drive = min(F_set, Fx_rem[1])
    return min(F_drive, F_cap) >= resistance(veh, v)


def corner_speed(veh: PerfParams, k: float, v_hi: float | None = None, tol: float = 1e-4) -> float:
    """Highest steady-state speed on curvature k (bisection)."""
    k = abs(float(k))
    _, _, v_max = powertrain(veh, 0.0)
    hi = min(v_hi or v_max, v_max)
    if steady_ok(veh, hi, k):
        return hi
    lo = 0.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if steady_ok(veh, mid, k):
            lo = mid
        else:
            hi = mid
    return lo


def solve_profile(veh: PerfParams, ds: float, k: np.ndarray, v0: float = 0.0, v_end: float | None = None,
                  corner_cache: dict | None = None, veh_brake: PerfParams | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Speed profile on a curvature profile with spacing ds. Returns (v, v_corner).

    veh_brake: other parameters for the backward (braking) pass, e.g. a driver margin."""
    vb_veh = veh_brake or veh
    n = len(k)
    # corner speeds (cached per rounded curvature: tracks repeat values a lot)
    cache = corner_cache if corner_cache is not None else {}
    vc = np.empty(n)
    for i, ki in enumerate(np.abs(k)):
        key = round(float(ki), 5)
        if key not in cache:
            cache[key] = corner_speed(veh, key)
        vc[i] = cache[key]
    # forward pass
    vf = np.empty(n)
    vf[0] = min(v0, vc[0])
    a_prev = 0.0
    for i in range(n - 1):
        a_acc, _, _, _ = ax_limits(veh, vf[i], k[i], a_prev, 0.0, mode=1)
        a_prev = a_acc
        vf[i + 1] = min(vc[i + 1], math.sqrt(max(vf[i] ** 2 + 2 * a_acc * ds, 0.0)))
    # backward pass
    vb = np.empty(n)
    vb[-1] = vf[-1] if v_end is None else min(v_end, vf[-1])
    d_prev = 0.0
    for i in range(n - 1, 0, -1):
        _, a_dec, _, _ = ax_limits(vb_veh, vb[i], k[i], 0.0, d_prev, mode=2)
        d_prev = a_dec
        vb[i - 1] = min(vf[i - 1], math.sqrt(max(vb[i] ** 2 - 2 * a_dec * ds, 0.0)))
    return vb, vc


def _segment_time(v: np.ndarray, ds: float) -> float:
    vm = 0.5 * (v[:-1] + v[1:])
    return float(np.sum(ds / np.maximum(vm, 1e-3)))


def _accels(v: np.ndarray, k: np.ndarray, ds: float):
    ax = np.zeros_like(v)
    ax[:-1] = (v[1:] ** 2 - v[:-1] ** 2) / (2 * ds)
    ax[-1] = ax[-2] if len(ax) > 1 else 0.0
    return ax, v * v * k


# ---------------------------------------------------------------------------- #
# events
# ---------------------------------------------------------------------------- #
def acceleration(veh: PerfParams, length: float = ACCEL_LENGTH, ds: float = 0.25) -> QssResult:
    veh = veh.for_event("acceleration")
    n = int(round(length / ds)) + 1
    k = np.zeros(n)
    s = np.arange(n) * ds
    v, vc = solve_profile(veh, ds, k, v0=0.0)
    ax, ay = _accels(v, k, ds)
    return QssResult("acceleration", _segment_time(v, ds), s, v, k, ax, ay, v_corner=vc)


def skidpad(veh: PerfParams, radius: float = SKIDPAD_RADIUS) -> QssResult:
    veh = veh.for_event("skidpad")
    v = corner_speed(veh, 1.0 / radius)
    t = 2 * math.pi * radius / max(v, 1e-6)
    s = np.linspace(0.0, 2 * math.pi * radius, 50)
    vv = np.full_like(s, v)
    k = np.full_like(s, 1.0 / radius)
    return QssResult("skidpad", t, s, vv, k, np.zeros_like(s), vv * vv * k, [t, t], vv.copy())


def path_curvature(xy: np.ndarray, closed: bool, ds: float = 0.5, smooth: float = 3.0):
    """Resample a centre line at ds and return (s, k) with curvature smoothed over `smooth` metres."""
    xy = np.asarray(xy, float).reshape(-1, 2)
    if closed and np.linalg.norm(xy[0] - xy[-1]) > 1e-6:
        xy = np.vstack([xy, xy[:1]])
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    keep = np.concatenate([[True], seg > 1e-6])
    xy = xy[keep]
    s_raw = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
    L = s_raw[-1]
    n = max(int(round(L / ds)), 4)
    s = np.linspace(0.0, L, n + 1)
    x, y = np.interp(s, s_raw, xy[:, 0]), np.interp(s, s_raw, xy[:, 1])
    h = s[1] - s[0]
    if closed:
        xp, yp = np.r_[x[-2], x, x[1]], np.r_[y[-2], y, y[1]]
    else:
        xp, yp = np.r_[2 * x[0] - x[1], x, 2 * x[-1] - x[-2]], np.r_[2 * y[0] - y[1], y, 2 * y[-1] - y[-2]]
    dx, dy = (xp[2:] - xp[:-2]) / (2 * h), (yp[2:] - yp[:-2]) / (2 * h)
    ddx, ddy = (xp[2:] - 2 * xp[1:-1] + xp[:-2]) / h ** 2, (yp[2:] - 2 * yp[1:-1] + yp[:-2]) / h ** 2
    k = (dx * ddy - dy * ddx) / np.maximum((dx * dx + dy * dy) ** 1.5, 1e-9)
    w = max(int(round(smooth / h)), 1)
    if w > 1:
        ker = np.ones(w) / w
        k = np.convolve(np.r_[k[-w:], k, k[:w]], ker, mode="same")[w:-w] if closed else np.convolve(k, ker, mode="same")
    return s, k, h


def run_path(veh: PerfParams, xy, closed: bool, event: str = "autocross", laps: int = 1, ds: float = 0.5,
             standing_start: bool = True, v_end: float | None = None, veh_brake: PerfParams | None = None) -> QssResult:
    """Lap time on a centre line. Closed tracks: 3 laps back to back, flying lap = lap 2."""
    veh = veh.for_event(event)
    veh_brake = veh_brake.for_event(event) if veh_brake is not None else None
    s1, k1, h = path_curvature(xy, closed, ds)
    cache: dict = {}
    if closed:
        n_lap = len(k1) - 1
        k = np.concatenate([k1[:-1]] * 3 + [k1[:1]])
        v, vc = solve_profile(veh, h, k, v0=0.0 if standing_start else 1e3, corner_cache=cache, veh_brake=veh_brake)
        laps_t = [_segment_time(v[i * n_lap:(i + 1) * n_lap + 1], h) for i in range(3)]
        sl = slice(n_lap, 2 * n_lap + 1)
        vv, kk = v[sl], k[sl]
        ax, ay = _accels(vv, kk, h)
        return QssResult(event, laps_t[1], s1, vv, kk, ax, ay, laps_t, vc[sl], v[:n_lap + 1].copy())
    v, vc = solve_profile(veh, h, k1, v0=0.0, v_end=v_end, corner_cache=cache, veh_brake=veh_brake)
    ax, ay = _accels(v, k1, h)
    t = _segment_time(v, h)
    return QssResult(event, t, s1, v, k1, ax, ay, [t], vc)


def reference_speed(veh: PerfParams, path_xy, closed: bool, util: float = 1.0, v_end: float | None = 0.0,
                    ds: float = 0.5, util_long: float = 1.0) -> np.ndarray:
    """QSS speed target at every point of `path_xy` for a driver using `util` of the grip.

    util_long (<= 1) additionally scales the longitudinal grip used for BRAKING, so braking ends
    before the corner instead of at the combined limit inside it (the QSS model is a point mass
    and does not see the rear axle losing grip under braking). Traction uses `util`.
    Closed paths get the periodic (flying-lap) profile; open paths start from standstill and
    end at v_end (None: free)."""
    pts = np.asarray(path_xy, float).reshape(-1, 2)
    q = veh.copy(util_x=veh.util_x * util, util_y=veh.util_y * util)
    qb = q.copy(util_x=q.util_x * util_long)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s_pts = np.concatenate([[0.0], np.cumsum(seg)])
    if closed:
        res = run_path(q, pts, True, event="", ds=ds, veh_brake=qb)
        return np.interp(s_pts, res.s, res.v, period=res.s[-1])
    s, k, h = path_curvature(pts, False, ds)
    v, _ = solve_profile(q, h, k, v0=0.0, v_end=v_end, veh_brake=qb)
    return np.interp(s_pts, s, v)
