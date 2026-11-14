"""Transient lap: the event driven by the simulator's vehicle model and reference driver, not the QSS model.

The QSS LapSim puts the car on the grip limit at every point, with no yaw, no steering and no load transfer in
time. Here the same car (LapSim parameters of the event, at the tyre limit) is the dynamic model of
fsdsim.dynamics (model: performance; single- or double-track as set in the simulator's vehicle.yaml) and is driven
along the event's centreline by fsdsim.control's reference driver (pure pursuit + speed targets from the QSS
profile at the event's driver utilisation). What it adds to the QSS result:

* how much time the transient behaviour costs (time delta along the lap against QSS),
* sideslip, yaw rate, steering angle and the lateral error to the line (the driver needs ~5-10 % margin),
* the effect of parameters QSS does not have: yaw inertia, steering limit, actuator limits.

Closed tracks (skidpad circle, endurance): two laps from standstill, the second one is the flying lap and its
time is scaled with the event's laps like LapSim (flying_lap timing). Open tracks (acceleration, autocross):
standstill to the end of the track.

The driver drives the centreline, as LapSim's time is computed on it (no racing line on either side).
"""

from __future__ import annotations

import math
import os
import time

import numpy as np

from fsdsim.config import apply_vehicle_overrides, load_all_configs, sync_vehicle_from_performance
from fsdsim.control import make_driver
from fsdsim.dynamics import VehicleModel
from fsdsim.perf.lapsim import event_vehicle

from . import SIM_CONFIG
from .lab import TRANSIENT_KEYS, Lab, clean, decimate, trace

RECORD_DT = 0.05


def build_vehicle(lab: Lab, changes, event: str, competition=None):
    """(VehicleParams in performance mode, event, grip_util, long_util)."""
    ls = lab.lapsim(competition)
    comp = ls.competition(changes)
    if event not in comp.events:
        raise ValueError(f"unknown event {event!r} (have: {', '.join(comp.events)})")
    ev = comp.events[event]
    perf = event_vehicle(ls.vehicle(changes), ev).copy(util_x=1.0, util_y=1.0, events={})
    veh, _, _ = load_all_configs(SIM_CONFIG)
    veh.model = "performance"
    veh.performance_source = veh.performance_file = ""
    veh.performance = perf.to_dict()
    veh.performance_changes = []
    veh.efficiency_map = os.path.join(lab.configs_dir, lab.eta_map) if lab.eta_map else ""
    sync_vehicle_from_performance(veh)
    sim_only = [c for c in changes if c.split("=")[0].split("*")[0].split("+")[0].split("-")[0].strip() in TRANSIENT_KEYS]
    if sim_only:
        apply_vehicle_overrides(veh, sim_only)
    return veh, ev


def close_loop(xy: np.ndarray) -> tuple[np.ndarray, float]:
    """A closed track's points without the repeated start point, with the closure gap spread along the lap.

    The documentation tracks (FSA2025_endurance.csv) are curvature tables; their x, y (integrated from the
    curvature) end ~17 m from the start, and LapSim repeats the start point as the last one. LapSim only uses the
    curvature, so it does not matter there, but a driven car would meet a 17 m jump. The gap between where the
    point after the last one would be and the first point is removed linearly along the lap (~1 cm per metre for
    the FSA 2025 endurance; the curvature hardly changes). Returns (points, gap [m])."""
    if len(xy) > 2 and np.linalg.norm(xy[-1] - xy[0]) < 1e-6:
        xy = xy[:-1]
    nxt = xy[-1] + (xy[-1] - xy[-2])
    err = nxt - xy[0]
    gap = float(np.hypot(*err))
    step = float(np.linalg.norm(xy[-1] - xy[-2]))
    if gap > 0.5 * max(step, 1e-3):
        s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
        xy = xy - np.outer(s / (s[-1] + step), err)
    else:
        gap = 0.0
    return xy, gap


class _Progress:
    """Arc length along a polyline, searched near the last position (laps counted on closed paths)."""

    def __init__(self, xy: np.ndarray, closed: bool):
        self.xy, self.closed = xy, closed
        d = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        self.s = np.r_[0.0, np.cumsum(d)]
        self.L = self.s[-1] + (np.linalg.norm(xy[-1] - xy[0]) if closed else 0.0)
        self.i, self.lap = 0, 0

    def update(self, x: float, y: float) -> tuple[float, float]:
        n = len(self.xy)
        idx = (self.i + np.arange(-20, 60)) % n if self.closed else np.clip(self.i + np.arange(-20, 60), 0, n - 1)
        p = self.xy[idx]
        j = int(np.argmin((p[:, 0] - x) ** 2 + (p[:, 1] - y) ** 2))
        new = int(idx[j])
        if self.closed and new < self.i - n // 2:
            self.lap += 1
        elif self.closed and new > self.i + n // 2:
            self.lap -= 1
        self.i = new
        # projection on the segment after (or before) the nearest point, signed lateral offset (left +)
        a = self.xy[new]
        b = self.xy[(new + 1) % n] if (self.closed or new + 1 < n) else a + (a - self.xy[new - 1])
        t = b - a
        tl = float(np.hypot(*t)) or 1e-9
        u = ((x - a[0]) * t[0] + (y - a[1]) * t[1]) / tl
        lat = (t[0] * (y - a[1]) - t[1] * (x - a[0])) / tl
        return self.lap * self.L + self.s[new] + u, lat


def run(lab: Lab, event: str, changes=(), competition=None, dt: float = 0.02, track_width: float = 3.0,
        job=None) -> dict:
    t_wall = time.time()
    changes = lab.validate(changes, competition)
    veh, ev = build_vehicle(lab, changes, event, competition)
    trk = ev.track
    xy = np.stack([trk.x, trk.y], axis=1).astype(float)
    closure = 0.0
    if trk.closed:
        xy, closure = close_loop(xy)
    if not 0.005 <= dt <= 0.05:
        raise ValueError("time step between 0.005 and 0.05 s")
    grip = float(ev.util_y)
    drv = make_driver(veh, "acceleration" if ev.type.lower() == "acceleration" else "", "map", track_width,
                      grip_util=grip, long_util=min(1.0, float(ev.util_x) / max(grip, 1e-6)))
    if trk.closed:
        drv.set_reference(xy, True)
    else:
        # the driver's speed profile of an open path starts at 0 m/s at its first point: give it 5 m of run-up
        # behind the start line so that the target at the line (where the car stands) is above zero
        d0 = (xy[1] - xy[0]) / max(np.linalg.norm(xy[1] - xy[0]), 1e-9)
        back = xy[0] - np.outer(np.arange(10, 0, -1) * 0.5, d0)
        drv.set_reference(np.vstack([back, xy]), False)
    car = VehicleModel(veh)
    yaw0 = math.atan2(xy[1, 1] - xy[0, 1], xy[1, 0] - xy[0, 0])
    car.reset(float(xy[0, 0]), float(xy[0, 1]), yaw0)
    prog = _Progress(xy, trk.closed)
    L = prog.L
    goal = 2 * L if trk.closed else L
    t_max = goal / 3.0 + 30.0
    rec = {k: [] for k in ("t", "s", "v", "ax", "ay", "steer", "r", "beta", "lat", "E", "P", "acc_cmd", "x", "y")}
    t, next_rec, lap_marks = 0.0, 0.0, []
    off_time, lat_max, beta_max = 0.0, 0.0, 0.0
    s_prev, last_job = 0.0, 0.0
    status = "finished"
    while True:
        st = car.state
        s, lat = prog.update(st.x, st.y)
        # crossing of every full lap (interpolated): lap times on closed tracks
        if trk.closed and int(s // L) > int(s_prev // L) and s > s_prev:
            k = int(s // L)
            frac = (k * L - s_prev) / max(s - s_prev, 1e-9)
            lap_marks.append(t - dt + frac * dt)
        if s >= goal:
            frac = (goal - s_prev) / max(s - s_prev, 1e-9)
            t_end = t - dt + frac * dt
            break
        s_prev = s
        if t > t_max:
            status, t_end = "timeout", t
            break
        if abs(lat) > 4.0 * track_width:
            status, t_end = "left the track", t
            break
        steer, acc, _ = drv.update(st.pose, st.vx)
        car.set_command(steer, acc)
        car.step(dt)
        t += dt
        beta = math.degrees(math.atan2(car.state.vy, max(car.state.vx, 1.0)))
        if abs(lat) > 0.5 * track_width:
            off_time += dt
        lat_max, beta_max = max(lat_max, abs(lat)), max(beta_max, abs(beta))
        if t >= next_rec:
            next_rec += RECORD_DT
            c = car.state
            for k, val in (("t", t), ("s", s), ("v", c.vx * 3.6), ("ax", car.ax_body / 9.81), ("ay", car.ay_body / 9.81),
                           ("steer", math.degrees(c.steer)), ("r", math.degrees(c.r)), ("beta", beta), ("lat", lat),
                           ("E", car.energy_j / 3.6e6), ("P", car.power_w / 1e3), ("acc_cmd", acc), ("x", c.x), ("y", c.y)):
                rec[k].append(val)
        if job is not None and time.time() - last_job > 0.3:
            last_job = time.time()
            job.step(min(s, goal), goal, f"{event}: {t:.1f} s simulated, {s:.0f} / {goal:.0f} m")

    # times
    if trk.closed:
        t_first = lap_marks[0] if lap_marks else math.nan
        t_lap = (lap_marks[1] - lap_marks[0]) if len(lap_marks) >= 2 else math.nan
        T = ev.laps * t_lap if ev.timing == "flying_lap" else t_first + (ev.laps - 1) * t_lap
    else:
        t_first = t_lap = T = t_end if status == "finished" else math.nan

    # the QSS reference of the same event and changes, and the time delta along the lap
    q = lab.raw(changes, competition, [event])["events"][event]
    rs, rv, rt = (np.asarray(rec[k]) for k in ("s", "v", "t"))
    if trk.closed:
        m = rs >= L                                   # flying lap (second lap), distance from its start
        lap_s, lap_t, lap_v = rs[m] - L, rt[m] - (lap_marks[0] if lap_marks else 0.0), rv[m]
    else:
        lap_s, lap_t, lap_v = rs, rt, rv
    if len(lap_s) > 2:
        q_t = np.interp(lap_s, q["s"], q["t"])
        delta = lap_t - q_t
    else:
        delta = np.array([])
    E_lap = math.nan
    if trk.closed and len(lap_s) > 2:
        E = np.asarray(rec["E"])[m]
        E_lap = float(E[-1] - E[0]) if len(E) else math.nan
    qt = trace(q, 1500)
    sl = decimate(len(rs), 2500)
    out = {"event": event, "status": status, "closed": bool(trk.closed), "laps": ev.laps, "L": L, "closure_gap_m": closure,
           "t_first": t_first, "t_lap": t_lap, "T": T, "points": _score(ev, T),
           "qss": {"T": q["T"], "t_lap": q["t_lap"], "points": q["points"], "v_max_kmh": q["v_max"] * 3.6,
                   "E_lap_kwh": q["E_lap"] / 3.6e6, "trace": {"s": qt["s"], "v": qt["v"], "t": qt["t"]}},
           "v_max_kmh": float(np.max(rv)) if len(rv) else math.nan, "E_lap_kwh": E_lap,
           "lat_max": lat_max, "beta_max": beta_max, "off_track_s": off_time,
           "driver": {"grip_util": grip, "long_util": min(1.0, float(ev.util_x) / max(grip, 1e-6)),
                      "handling": str(veh.chassis.handling), "yaw_inertia": veh.yaw_inertia,
                      "max_steer_deg": math.degrees(veh.actuators.max_steer)},
           "dt": dt, "sim_time": t, "wall_s": time.time() - t_wall,
           "trace": {k: np.asarray(v)[sl] for k, v in rec.items()},
           "lap": {"s": lap_s[decimate(len(lap_s), 1500)], "v": lap_v[decimate(len(lap_s), 1500)],
                   "dt": delta[decimate(len(delta), 1500)] if len(delta) else []},
           "track": {"x": xy[decimate(len(xy), 1500), 0], "y": xy[decimate(len(xy), 1500), 1]}}
    if job is not None:
        job.step(goal, goal, "done")
    return clean(out)


def _score(ev, T):
    from fsdsim.perf.lapsim import score  # noqa: PLC0415

    return score(T, ev.score) if T == T else math.nan
