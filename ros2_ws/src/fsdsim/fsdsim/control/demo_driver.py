"""Reference driver used to exercise the simulator end-to-end.

This is NOT a competition stack. It exists so the simulator can be validated
closed-loop on every event and so newcomers have a working baseline to replace
piece by piece with the team's own perception / SLAM / planning / control.

Two modes:
    map    follow a known reference path with pure pursuit and a
           curvature-limited speed profile (like a mapped trackdrive lap,
           acceleration or skidpad with a known layout)
    cones  local planning from cones in the vehicle frame: blue/yellow pairs ->
           midpoints -> greedy ordering -> pure pursuit (like an exploration
           lap with perfect cone detection)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..geometry import local_to_world, polyline_curvature, polyline_lengths, resample_polyline, world_to_local


@dataclass
class DriverParams:
    lookahead_gain: float = 0.30  # [s]
    lookahead_min: float = 3.0
    lookahead_max: float = 9.0
    lookahead_curv: float = 0.45  # lookahead <= this x corner radius (pure pursuit cuts corners by ~ld^2 / 8R)
    lookahead_floor: float = 2.0  # [m] absolute minimum in tight corners
    a_lat_max: float = 9.0
    a_accel_max: float = 10.0
    a_decel_max: float = 12.0
    v_max: float = 22.0
    v_max_cones: float = 9.0
    speed_kp: float = 2.5
    a_stop: float = 8.0           # [m/s^2] deceleration after the finish on a straight ...
    ay_stop_soft: float = 12.0    # ... reduced linearly with lateral acceleration (min 25 % at >= 0.75 x this)
    cone_range: float = 20.0
    cone_fov_deg: float = 160.0
    track_width_guess: float = 3.5
    wheelbase: float = 1.53
    lr: float = 0.73
    max_steer: float = 0.42
    stop_mode: str = "immediate"  # immediate | path_end (open paths: stop at the end of the reference)
    grip_util: float = 0.80       # with a performance (LapSim) vehicle: share of the tyre limit the
                                  # speed targets use (1.0 = LapSim's theoretical limit)
    long_util: float = 0.70       # extra factor on the longitudinal grip in those targets: brake before
                                  # the corner, not at the combined limit inside it (rear stays stable)
    steer_preview: float = 0.05   # [s] steer for where the car will be when the steering has moved
                                  # (actuator lag + control period); without it the car runs ~0.2 m wide on
                                  # the skidpad circles and touches a cone with the CAD front wing at 20-25 Hz
    latency: float = 0.0          # [s] age of the pose + time until a command acts (Isaac Sim: ~2 render frames).
                                  # The pose is predicted that far along the arc of the last steering command
                                  # (without it the skidpad failed in Isaac from the third lap, 11 Nov 2026)


def predict_pose(pose, v, steer, wheelbase, dt):
    """Where the car will be after dt seconds on the arc of the current steering angle (kinematic bicycle)."""
    x, y, yaw = pose
    if dt <= 0.0 or v <= 0.0:
        return x, y, yaw
    r = v * math.tan(steer) / wheelbase
    dyaw = r * dt
    if abs(dyaw) < 1e-6:
        return x + v * dt * math.cos(yaw), y + v * dt * math.sin(yaw), yaw
    R = v / r
    return (x + R * (math.sin(yaw + dyaw) - math.sin(yaw)), y - R * (math.cos(yaw + dyaw) - math.cos(yaw)), yaw + dyaw)


def speed_profile(path, closed, a_lat, a_acc, a_dec, v_max, v_end=0.0, ds=None):
    p = np.asarray(path, float)
    if ds is None:
        ds = float(np.mean(np.linalg.norm(np.diff(p, axis=0), axis=1)))
    kappa = np.abs(polyline_curvature(p, closed))
    kappa = np.convolve(np.pad(kappa, 4, mode="wrap" if closed else "edge"), np.ones(9) / 9, mode="valid")
    v = np.minimum(v_max, np.sqrt(a_lat / np.maximum(kappa, 1e-4)))
    n = len(v)
    reps = 2 if closed else 1
    if not closed:
        v[-1] = v_end
    for _ in range(reps):
        for i in range(1, n if not closed else n + 1):
            j, k = i % n, (i - 1) % n
            v[j] = min(v[j], math.sqrt(v[k] ** 2 + 2 * a_acc * ds))
        for i in range(n - 2, -1, -1) if not closed else range(2 * n - 1, -1, -1):
            j, k = i % n, (i + 1) % n
            v[j] = min(v[j], math.sqrt(v[k] ** 2 + 2 * a_dec * ds))
    return v


class PathFollower:
    def __init__(self, params: DriverParams):
        self.p = params
        self.path = None
        self.closed = False
        self.v_ref = None
        self.idx = 0
        self.ds = 0.25

    def set_path(self, path, closed: bool, v_profile=None, ds: float = 0.25):
        self.path = resample_polyline(np.asarray(path, float), ds, closed)
        self.closed = closed
        self.ds = ds
        p = self.p
        kap = np.abs(polyline_curvature(self.path, closed)) if len(self.path) >= 3 else np.zeros(len(self.path))
        w = max(int(2.0 / ds), 1)
        self.kappa = np.convolve(np.pad(kap, w, mode="wrap" if closed else "edge"), np.ones(2 * w + 1) / (2 * w + 1),
                                 mode="valid")
        self.v_ref = (
            speed_profile(self.path, closed, p.a_lat_max, p.a_accel_max, p.a_decel_max, p.v_max, 0.0, ds)
            if v_profile is None
            else np.asarray(v_profile, float)
        )
        self.idx = 0

    def _nearest(self, xy):
        n = len(self.path)
        if self.closed:
            idx = np.arange(self.idx - 20, self.idx + 160) % n
        else:
            idx = np.arange(max(0, self.idx - 20), min(n, self.idx + 160))
        d = np.hypot(self.path[idx, 0] - xy[0], self.path[idx, 1] - xy[1])
        self.idx = int(idx[int(np.argmin(d))])
        return self.idx

    def initialise(self, xy):
        d = np.hypot(self.path[:, 0] - xy[0], self.path[:, 1] - xy[1])
        self.idx = int(np.argmin(d))

    def control(self, pose, v, stop=False):
        p = self.p
        x, y, yaw = predict_pose(pose, v, getattr(self, "last_steer", 0.0), p.wheelbase, p.latency)
        pv = max(0.0, v) * p.steer_preview
        rear = (x + pv * math.cos(yaw) - p.lr * math.cos(yaw), y + pv * math.sin(yaw) - p.lr * math.sin(yaw))
        i = self._nearest((x, y))
        ld = float(np.clip(p.lookahead_gain * v + p.lookahead_min, p.lookahead_min, p.lookahead_max))
        n = len(self.path)
        # shorter lookahead in tight corners, so the car does not cut them (cone hits at speed)
        ahead = np.arange(i, i + max(int(ld / self.ds), 1)) % n if self.closed else \
            np.arange(i, min(i + max(int(ld / self.ds), 1), n))
        k_max = float(self.kappa[ahead].max()) if len(ahead) else 0.0
        if k_max > 1e-4:
            ld = min(ld, max(p.lookahead_floor, p.lookahead_curv / k_max))
        j = i
        for _ in range(n):
            nj = (j + 1) % n if self.closed else min(j + 1, n - 1)
            if nj == j:
                break
            j = nj
            if math.hypot(self.path[j, 0] - rear[0], self.path[j, 1] - rear[1]) >= ld:
                break
        target = self.path[j]
        steer = _pure_pursuit(target, rear, yaw, p.wheelbase, p.max_steer)
        self.last_steer = steer
        preview = int(max(0.0, v) * 0.4 / self.ds)
        k = (i + preview) % n if self.closed else min(i + preview, n - 1)
        v_t = float(min(self.v_ref[i], self.v_ref[k]))
        if stop and (self.closed or p.stop_mode == "immediate"):
            v_t = 0.0
            # brake firmly on straights, gently in corners: braking at the cornering limit unloads
            # the rear axle and spins the car (after the finish line the path is followed until standstill)
            ay = v * v * float(self.kappa[i])
            accel = -p.a_stop * max(0.25, 1.0 - ay / p.ay_stop_soft) if v > 0.05 else -2.0
            # ... but never faster than the speed profile allows for the corners ahead: a car that crosses
            # the finish line fast (e.g. the RL driver) must not run wide in the first corner after it
            v_prof = float(min(self.v_ref[i], self.v_ref[k]))
            accel = min(accel, _speed_control(v_prof, v, p, False))
            return steer, accel, {"target": target.tolist(), "v_target": v_t, "idx": i}
        stop = False  # path_end: the speed profile already ends at v = 0
        accel = _speed_control(v_t, v, p, stop)
        return steer, accel, {"target": target.tolist(), "v_target": v_t, "idx": i}


class ConeLocalPlanner:
    def __init__(self, params: DriverParams):
        self.p = params

    def plan(self, cones_local: np.ndarray, colors: np.ndarray, prev_local: np.ndarray | None = None) -> np.ndarray:
        """prev_local: the path planned in the last cycle, in the current car frame (guides cones without colour)."""
        p = self.p
        self._prev = prev_local
        if len(cones_local):
            ang = np.degrees(np.arctan2(cones_local[:, 1], cones_local[:, 0]))
            dist = np.linalg.norm(cones_local, axis=1)
            sel = (cones_local[:, 0] > -1.0) & (dist < p.cone_range) & (np.abs(ang) < p.cone_fov_deg / 2)
            cones_local, colors = cones_local[sel], colors[sel]
        w = p.track_width_guess
        # big orange cones mark the timing line on the track boundaries: near the car they stand in for the
        # blue (left) and yellow (right) cones there. Without them the path had a gap at the line and the car
        # drove straight on where the circuit bends (2 big orange cones hit at the finish on seed 13, 11 Nov 2026).
        big = colors == 3
        if big.any():
            near = big & (cones_local[:, 0] < 8.0) & (np.abs(cones_local[:, 1]) > 0.25 * w)
            colors = colors.copy()
            colors[near] = np.where(cones_local[near, 1] > 0.0, 0, 1)
        colors = self._side_of_unknown(cones_local, colors, w)
        blue = cones_local[colors == 0]
        yellow = cones_local[colors == 1]
        mids = []
        if len(blue) and len(yellow):
            for b in blue:
                d = np.linalg.norm(yellow - b, axis=1)
                j = int(np.argmin(d))
                if 0.4 * w < d[j] < 2.2 * w:
                    mids.append((b + yellow[j]) / 2)
            for yv in yellow:
                d = np.linalg.norm(blue - yv, axis=1)
                j = int(np.argmin(d))
                if 0.4 * w < d[j] < 2.2 * w:
                    mids.append((yv + blue[j]) / 2)
        if not mids:
            if len(blue):
                mids = [b + np.array([0.0, -w / 2]) for b in blue]
            elif len(yellow):
                mids = [yv + np.array([0.0, w / 2]) for yv in yellow]
            else:
                return np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]])
        mids = _dedupe(np.array(mids), 0.8)
        return _order_path(mids)


    def _mids(self, cones_local, colors, w):
        blue, yellow = cones_local[colors == 0], cones_local[colors == 1]
        mids = []
        for a, b in ((blue, yellow), (yellow, blue)):
            if len(a) and len(b):
                for q in a:
                    d = np.linalg.norm(b - q, axis=1)
                    j = int(np.argmin(d))
                    if 0.4 * w < d[j] < 2.2 * w:
                        mids.append((q + b[j]) / 2)
        return mids

    CHAIN_STEP = (1.0, 7.0)    # m between neighbouring cones of one boundary
    CHAIN_TURN = 70.0          # deg: most a boundary turns from one cone to the next

    def _chain_sides(self, cones_local, colors, w):
        """Colour-blind boundaries: follow each side from the car cone by cone (next cone 1-7 m on, turning at most
        70 deg), alternating left and right. The path-based rule alone left the cones after a 90 deg corner
        unknown and the car went straight on (Isaac RTX LiDAR without stripe intensities, 11 Nov 2026)."""
        n = len(cones_local)
        if n == 0:
            return colors
        lo, hi = self.CHAIN_STEP
        cos_max = math.cos(math.radians(self.CHAIN_TURN))
        chains = {}
        for col, sign in ((0, 1.0), (1, -1.0)):
            known = [i for i in range(n) if colors[i] == col and cones_local[i, 0] > -1.0]
            pts = [np.array([-2.0, sign * 0.5 * w]), np.array([0.0, sign * 0.5 * w])]    # beside the car, heading on
            left = sorted(known, key=lambda i: float(np.linalg.norm(cones_local[i])))
            while left:                          # the known cones of this side in order along the boundary
                tip = pts[-1]
                j = min(left, key=lambda i: float(np.linalg.norm(cones_local[i] - tip)))
                pts.append(cones_local[j])
                left.remove(j)
            if not known:                        # seed: the nearest unknown cone on this side
                cand = [i for i in range(n) if colors[i] == 4 and cones_local[i, 0] > -1.0
                        and 0.2 * w < sign * cones_local[i, 1] < 1.2 * w and np.linalg.norm(cones_local[i]) < 8.0]
                if not cand:
                    continue
                j = min(cand, key=lambda i: float(np.linalg.norm(cones_local[i])))
                colors[j] = col
                pts.append(cones_local[j])
            chains[col] = pts
        grown = True
        while grown:
            grown = False
            for col, pts in chains.items():
                tip, d0 = pts[-1], pts[-1] - pts[-2]
                d0 = d0 / max(float(np.linalg.norm(d0)), 1e-9)
                best, score = None, None
                for i in np.nonzero(colors == 4)[0]:
                    d = cones_local[i] - tip
                    L = float(np.linalg.norm(d))
                    if not lo < L < hi:
                        continue
                    c = float(d @ d0) / L
                    if c < cos_max:
                        continue
                    sc = L * (2.0 - c)               # near and straight on
                    if score is None or sc < score:
                        best, score = i, sc
                if best is not None:
                    colors[best] = col
                    pts.append(cones_local[best])
                    grown = True
        return colors

    def _side_of_unknown(self, cones_local, colors, w, passes: int = 4):
        """Cones without a colour (camera down, a LiDAR without stripe intensities such as Isaac's RTX LiDAR):
        left / right of the path built so far -> blue / yellow. Grows outward from the car, a few metres per pass;
        cones far from the path (another part of the track) stay unknown."""
        unk = colors == 4
        if not unk.any():
            return colors
        colors = self._chain_sides(cones_local, colors.copy(), w)
        if not (colors == 4).any():
            return colors
        prev = getattr(self, "_prev", None)
        if prev is not None:
            prev = prev[prev[:, 0] > -2.0]
        for _ in range(passes):
            mids = self._mids(cones_local, colors, w)
            ref = _order_path(_dedupe(np.array(mids), 0.8)) if len(mids) >= 2 else None
            if (ref is None or len(ref) < 3) and prev is not None and len(prev) >= 2:
                ref = prev          # the last path bends with the track; a straight line would cut corners
            if ref is None or len(ref) < 2:
                ref = np.array([[0.0, 0.0], [2.0, 0.0]])
            reach = float(np.linalg.norm(ref[-1])) + 5.0
            changed = False
            for i in np.nonzero(colors == 4)[0]:
                q = cones_local[i]
                if np.linalg.norm(q) > reach:
                    continue
                best, side = None, 0.0
                for a, b in zip(ref[:-1], ref[1:]):
                    ab = b - a
                    L2 = float(ab @ ab)
                    if L2 < 1e-9:
                        continue
                    f = float(np.clip((q - a) @ ab / L2, 0.0, 1.5))     # a little past the last point
                    d = q - (a + f * ab)
                    dist = float(np.linalg.norm(d))
                    if best is None or dist < best:
                        best, side = dist, ab[0] * d[1] - ab[1] * d[0]
                if best is not None and 0.2 * w < best < 1.1 * w:
                    colors[i] = 0 if side > 0 else 1
                    changed = True
            if not changed:
                break
        return colors


class DemoDriver:
    """Glue: picks the mode and applies the stop logic from the AS status.

    With `perf` (fsdsim.perf.PerfParams, the LapSim vehicle) the map-mode speed targets come
    from the LapSim QSS solver for this car on this path at `grip_util` of the tyre limit,
    so changing a vehicle parameter changes how fast the driver goes, as in LapSim."""

    def __init__(self, params: DriverParams | None = None, mode: str = "map", perf=None):
        self.p = params or DriverParams()
        self.mode = mode
        self.perf = perf
        self.follower = PathFollower(self.p)
        self.local = ConeLocalPlanner(self.p)
        self._initialised = False

    def set_reference(self, path, closed):
        self.follower.set_path(path, closed)
        if self.perf is not None:
            from ..perf import qss  # noqa: PLC0415

            v_end = 0.0 if (not closed and self.p.stop_mode == "path_end") else None
            self.follower.v_ref = qss.reference_speed(self.perf, self.follower.path, closed, self.p.grip_util,
                                                      v_end=v_end, util_long=self.p.long_util)
        self._initialised = False

    def update(self, pose, v, cones_world=None, cone_colors=None, stop=False):
        if self.mode == "map":
            if self.follower.path is None:
                raise RuntimeError("map mode needs set_reference(path, closed) first")
            if not self._initialised:
                self.follower.initialise(pose[:2])
                self._initialised = True
            return self.follower.control(pose, v, stop)
        # cones mode
        pose = predict_pose(pose, v, getattr(self, "_last_cone_steer", 0.0), self.p.wheelbase, self.p.latency)
        cones_world = np.asarray(cones_world if cones_world is not None else [], float).reshape(-1, 2)
        colors = np.asarray(cone_colors if cone_colors is not None else [], int)
        local = world_to_local(cones_world, pose) if len(cones_world) else cones_world
        prev = world_to_local(self._prev_path_world, pose) if getattr(self, "_prev_path_world", None) is not None else None
        path = self.local.plan(local, colors, prev)
        self._prev_path_world = local_to_world(path, pose) if len(path) >= 3 else getattr(self, "_prev_path_world", None)
        p = self.p
        ld = float(np.clip(p.lookahead_gain * v + p.lookahead_min, p.lookahead_min, p.lookahead_max))
        # path is in the CoG frame; pure pursuit from the rear axle
        path_rear = path + np.array([p.lr, 0.0])
        s = polyline_lengths(path_rear)
        target = np.array([np.interp(ld, s, path_rear[:, 0]), np.interp(ld, s, path_rear[:, 1])])
        steer = math.atan2(2.0 * p.wheelbase * target[1], max(float(target @ target), 1e-3))
        steer = float(np.clip(steer, -p.max_steer, p.max_steer))
        self._last_cone_steer = steer
        if len(path) >= 3:
            dense = resample_polyline(path, 0.5)
            kap = np.abs(polyline_curvature(dense)) if len(dense) >= 3 else np.zeros(1)
            k = float(np.max(kap[: min(len(kap), 30)]))
        else:
            k = 0.0
        v_t = 0.0 if stop else min(p.v_max_cones, math.sqrt(p.a_lat_max / max(k, 1e-3)))
        accel = _speed_control(v_t, v, p, stop)
        return steer, accel, {"path_local": path.tolist(), "v_target": v_t}


# --------------------------------------------------------------------------- #
def _pure_pursuit(target, rear, yaw, wheelbase, max_steer):
    dx, dy = target[0] - rear[0], target[1] - rear[1]
    c, s = math.cos(yaw), math.sin(yaw)
    lx, ly = c * dx + s * dy, -s * dx + c * dy
    ld2 = max(lx * lx + ly * ly, 1e-3)
    return float(np.clip(math.atan2(2.0 * wheelbase * ly, ld2), -max_steer, max_steer))


def _speed_control(v_t, v, p: DriverParams, stop: bool) -> float:
    if stop:
        return -min(p.a_decel_max, p.a_stop) if v > 0.05 else -2.0
    return float(np.clip(p.speed_kp * (v_t - v), -p.a_decel_max, p.a_accel_max))


def _dedupe(pts: np.ndarray, radius: float) -> np.ndarray:
    out = []
    for q in pts:
        if all(np.linalg.norm(q - o) > radius for o in out):
            out.append(q)
    return np.array(out)


def _order_path(mids: np.ndarray, max_step: float = 8.0) -> np.ndarray:
    cur = np.array([0.0, 0.0])
    heading = np.array([1.0, 0.0])
    remaining = [m for m in mids if m[0] > -0.5]
    path = [cur]
    while remaining:
        best, best_cost = None, None
        for k, m in enumerate(remaining):
            d = m - cur
            dist = np.linalg.norm(d)
            if dist < 1e-3 or dist > max_step:
                continue
            cosang = float(d @ heading) / dist
            if cosang < math.cos(math.radians(75)):
                continue
            cost = dist * (2.0 - cosang)
            if best_cost is None or cost < best_cost:
                best, best_cost = k, cost
        if best is None:
            break
        m = remaining.pop(best)
        heading = (m - cur) / np.linalg.norm(m - cur)
        cur = m
        path.append(m)
    if len(path) == 1:
        path.append(np.array([3.0, 0.0]))
    return np.array(path)


# --------------------------------------------------------------------------- #
RACING_MARGIN = 0.40        # [m] car side to cone line on the racing line (plus half the car width)
RACING_GRIP_UTIL = 0.75     # share of the tyre limit the driver uses on the racing line (centre line: 0.80).
# The racing line turns corners into long fast sweeps driven at the limit; with 0.80 the car ran wide on 1 of 18
# generated circuits. 0.75 / 0.40 m was clean on all of them at 20 and 50 Hz and is about 2 % faster than the
# centre line (QSS alone promises 6 %: the rest is margin this simple driver needs).


def reference_path(centerline, width: float, closed: bool, veh, line: str = "center", margin: float = RACING_MARGIN,
                   grip_util: float = RACING_GRIP_UTIL):
    """The path the map-mode driver follows: the centre line, or the racing line inside the cones.

    line "racing": perf.raceline for this car (closed circuits with a LapSim vehicle; otherwise the centre line).
    margin: distance kept between the car's side and the cone line [m] on top of half the car width; the driver
    tracks its path within about +-0.2 m. Use make_driver(..., line="racing") with it (lower grip utilisation). Returns (path, info) with info = None or the raceline dict."""
    if str(line).lower() not in ("racing", "raceline", "race") or not closed:
        return centerline, None
    perf = veh.perf() if getattr(veh, "model", "") == "performance" else None
    if perf is None:
        return centerline, None
    from ..perf.raceline import racing_line  # noqa: PLC0415

    q = perf.copy(util_x=perf.util_x * grip_util, util_y=perf.util_y * grip_util, events={})
    info = racing_line(centerline, width, q, True, car_width=veh.footprint.width, margin=margin)
    return info["xy"], info


def make_driver(veh, mission: str = "", mode: str = "map", track_width: float = 3.5,
                grip_util: float | None = None, line: str = "center", **overrides) -> DemoDriver:
    """Reference driver matched to a VehicleParams (and its LapSim set when model: performance)."""
    p = DriverParams(wheelbase=veh.wheelbase, lr=veh.lr, max_steer=veh.actuators.max_steer,
                     track_width_guess=track_width)
    if mission in ("acceleration", "ebs_test"):
        p.v_max = 26.0
    if mission == "skidpad":
        p.stop_mode = "path_end"  # stop inside the exit lane, not in the figure of eight
    perf = veh.perf() if getattr(veh, "model", "") == "performance" else None
    if perf is not None:
        # the car's limits come from the LapSim model; the driver may request up to the actuator limits
        p.a_accel_max = veh.actuators.max_accel
        p.a_decel_max = veh.actuators.max_decel
        p.v_max = veh.actuators.max_speed
        if grip_util is not None:
            p.grip_util = grip_util
        elif str(line).lower() in ("racing", "raceline", "race"):
            p.grip_util = RACING_GRIP_UTIL
    for k, v in overrides.items():
        setattr(p, k, v)
    return DemoDriver(p, mode, perf)


def make_vehicle_driver(veh, mission: str = "", mode: str = "map", track_width: float = 3.5,
                        longitudinal: str = "reference", policy: str = "auto", grip_util: float | None = None,
                        **overrides):
    """(driver, description): the reference driver, or the RL driver (learned throttle / brake) when
    longitudinal == 'rl' and a policy for this vehicle exists. Falls back to the reference driver
    with the reason in the description (cones mode, no policy, not a LapSim vehicle)."""
    ref = make_driver(veh, mission, mode, track_width, grip_util=grip_util, **overrides)
    if longitudinal != "rl":
        return ref, "reference driver"
    if mode != "map":
        return ref, "reference driver (the RL driver needs map mode)"
    if getattr(veh, "model", "") != "performance":
        return ref, "reference driver (the RL driver needs a LapSim vehicle)"
    from ..rl import registry  # noqa: PLC0415
    from ..rl.policy import MLPPolicy  # noqa: PLC0415
    from ..vec.features import FEATURES_VERSION as FEATURES_VERSION_UNIVERSAL  # noqa: PLC0415
    from .rl_driver import FEATURES_VERSION, RLDriver  # noqa: PLC0415

    folder, why = registry.resolve(policy, veh)
    if folder is None:
        return ref, f"reference driver ({why})"
    pol, meta = MLPPolicy.load(folder)
    version = meta.get("features_version")
    if version not in (FEATURES_VERSION, FEATURES_VERSION_UNIVERSAL):
        return ref, f"reference driver (policy {meta.get('name')} was trained with older features: retrain it)"
    if meta.get("universal"):
        out = registry.universal_coverage(meta, veh)
        note = "" if not out else " - OUTSIDE its training range (" + "; ".join(out[:3]) + ")"
    else:
        note = "" if meta.get("spec_hash") == _spec(veh) else " - trained for another vehicle spec"
    return RLDriver(pol, veh, mission, track_width, ref_driver=ref), f"RL driver, {why}{note}"


def _spec(veh) -> str:
    from ..rl.es import spec_hash  # noqa: PLC0415

    return spec_hash(veh)
