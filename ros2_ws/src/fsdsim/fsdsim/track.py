"""FSG track layouts: acceleration, skidpad, autocross / trackdrive.

Coordinates are in the `map` frame (ENU, metres). Cone colours follow the FS
Driverless Specification: blue = left border, yellow = right border (relative to
the driving direction), small orange = entry / exit lanes, big orange = before
and after start, finish and timekeeping lines.

Every dimension comes from `EventsConfig` so it can be checked against the
current FS Rules (chapter D) and changed without touching code.
"""

from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from .config import AccelerationEvent, EventsConfig, SkidpadEvent, TrackGenerator, VehicleParams
from .geometry import (
    polyline_curvature,
    polyline_lengths,
    polyline_tangents,
    resample_polyline,
)


class ConeColor(IntEnum):
    BLUE = 0
    YELLOW = 1
    ORANGE_SMALL = 2
    ORANGE_BIG = 3
    UNKNOWN = 4


@dataclass(frozen=True)
class ConeSpec:
    base: float  # square base plate edge [m]
    height: float  # [m]
    r_bottom: float  # body radius at the bottom [m]
    r_top: float  # body radius at the top [m]
    stripes: tuple  # ((z_from, z_to, "white"|"black"), ...) as fraction of height
    body: str  # body colour name


# FS Driverless Specification, Table 1: small cones 228 x 228 x 325 mm,
# big orange cone 285 x 285 x 505 mm. Stripe layout: verify against the spec.
CONE_SPECS = {
    ConeColor.BLUE: ConeSpec(0.228, 0.325, 0.095, 0.022, ((0.40, 0.60, "white"),), "blue"),
    ConeColor.YELLOW: ConeSpec(0.228, 0.325, 0.095, 0.022, ((0.40, 0.60, "black"),), "yellow"),
    ConeColor.ORANGE_SMALL: ConeSpec(0.228, 0.325, 0.095, 0.022, ((0.40, 0.60, "white"),), "orange"),
    ConeColor.ORANGE_BIG: ConeSpec(
        0.285, 0.505, 0.125, 0.028, ((0.30, 0.42, "white"), (0.58, 0.70, "white")), "orange"
    ),
    ConeColor.UNKNOWN: ConeSpec(0.228, 0.325, 0.095, 0.022, (), "orange"),
}

CSV_TAGS = {
    ConeColor.BLUE: "blue",
    ConeColor.YELLOW: "yellow",
    ConeColor.ORANGE_SMALL: "orange",
    ConeColor.ORANGE_BIG: "big_orange",
    ConeColor.UNKNOWN: "unknown",
}
TAG_TO_COLOR = {v: k for k, v in CSV_TAGS.items()}


@dataclass
class Cone:
    x: float
    y: float
    color: ConeColor
    id: int = -1
    knocked: bool = False

    @property
    def spec(self) -> ConeSpec:
        return CONE_SPECS[self.color]

    @property
    def hit_radius(self) -> float:
        return self.spec.base / 2.0


@dataclass
class TimingLine:
    p1: np.ndarray
    p2: np.ndarray
    forward: np.ndarray  # unit vector of the valid crossing direction
    name: str = "timing"


@dataclass
class Track:
    name: str
    mission: str
    cones: list
    start_pose: tuple  # (x, y, yaw) of base_link
    centerline: np.ndarray  # (N, 2) reference path in driving order
    closed: bool
    width: float
    laps: int
    timing_line: TimingLine | None = None
    finish_line: TimingLine | None = None
    stop_zone: np.ndarray | None = None  # polygon (M, 2)
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        for i, c in enumerate(self.cones):
            c.id = i

    # Convenience arrays -------------------------------------------------- #
    def cone_xy(self) -> np.ndarray:
        return np.array([[c.x, c.y] for c in self.cones], dtype=float).reshape(-1, 2)

    def cone_colors(self) -> np.ndarray:
        return np.array([int(c.color) for c in self.cones], dtype=int)

    @property
    def length(self) -> float:
        return float(polyline_lengths(self.centerline, self.closed)[-1])

    def summary(self) -> str:
        counts = {}
        for c in self.cones:
            counts[c.color.name] = counts.get(c.color.name, 0) + 1
        return (
            f"{self.name}: mission={self.mission} length={self.length:.1f} m "
            f"width={self.width:.2f} m laps={self.laps} cones={counts}"
        )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _line(center, tangent, half_len, name):
    t = np.asarray(tangent, float)
    t = t / np.linalg.norm(t)
    n = np.array([-t[1], t[0]])
    c = np.asarray(center, float)
    return TimingLine(c + n * half_len, c - n * half_len, t, name)


def _big_orange_pair(center, tangent, width, gap=0.5):
    """Big orange cones before and after a line, on both borders."""
    t = np.asarray(tangent, float) / np.linalg.norm(tangent)
    n = np.array([-t[1], t[0]])
    c = np.asarray(center, float)
    cones = []
    for along in (-gap, gap):
        for side in (1.0, -1.0):
            p = c + t * along + n * side * width / 2.0
            cones.append(Cone(float(p[0]), float(p[1]), ConeColor.ORANGE_BIG))
    return cones


def _drop_near(cones, points, radius):
    pts = np.asarray(points, float).reshape(-1, 2)
    keep = []
    for c in cones:
        if len(pts) == 0 or np.min(np.hypot(pts[:, 0] - c.x, pts[:, 1] - c.y)) > radius:
            keep.append(c)
    return keep


# --------------------------------------------------------------------------- #
# Acceleration (straight line)
# --------------------------------------------------------------------------- #
def build_acceleration(ev: AccelerationEvent, veh: VehicleParams, mission: str = "acceleration") -> Track:
    w, L, sp = ev.width, ev.length, ev.cone_spacing
    cones: list[Cone] = []
    # Borders between start and finish line
    for x in np.arange(0.0, L + 1e-6, sp):
        cones.append(Cone(float(x), w / 2, ConeColor.BLUE))
        cones.append(Cone(float(x), -w / 2, ConeColor.YELLOW))
    # Exit lane / braking zone (small orange)
    for x in np.arange(L + sp, L + ev.stop_length + 1e-6, sp):
        cones.append(Cone(float(x), w / 2, ConeColor.ORANGE_SMALL))
        cones.append(Cone(float(x), -w / 2, ConeColor.ORANGE_SMALL))
    big = _big_orange_pair((0.0, 0.0), (1, 0), w) + _big_orange_pair((L, 0.0), (1, 0), w)
    cones = _drop_near(cones, [[b.x, b.y] for b in big], 0.9) + big

    x0 = -(ev.staging_gap + veh.front_bumper)
    xs =np.arange(x0 - 2.0, L + ev.stop_length + 1e-6, 0.25)
    centerline = np.stack([xs, np.zeros_like(xs)], axis=1)
    stop_zone = np.array([[L, -w / 2], [L + ev.stop_length, -w / 2], [L + ev.stop_length, w / 2], [L, w / 2]])
    return Track(
        name="acceleration",
        mission=mission,
        cones=cones,
        start_pose=(x0, 0.0, 0.0),
        centerline=centerline,
        closed=False,
        width=w,
        laps=1,
        timing_line=_line((0.0, 0.0), (1, 0), w / 2 + 1.0, "start"),
        finish_line=_line((L, 0.0), (1, 0), w / 2 + 1.0, "finish"),
        stop_zone=stop_zone,
        meta={"length": L, "stop_length": ev.stop_length},
    )


# --------------------------------------------------------------------------- #
# Skidpad (figure of eight)
# --------------------------------------------------------------------------- #
def build_skidpad(ev: SkidpadEvent, veh: VehicleParams) -> Track:
    r_in = ev.inner_diameter / 2.0
    r_out = ev.outer_diameter / 2.0
    d = ev.center_distance / 2.0
    w = r_out - r_in
    r_mid = (r_in + r_out) / 2.0
    c_right = np.array([d, 0.0])
    c_left = np.array([-d, 0.0])
    corridor = w / 2.0  # entry / exit lane half width (lane is centred on x = 0)

    cones: list[Cone] = []

    def ring(center, radius, n, color, other_center):
        out = []
        for k in range(n):
            a = 2 * math.pi * k / n
            p = center + radius * np.array([math.cos(a), math.sin(a)])
            # Outer rings: drop cones on the other circle's track and in the lanes
            if radius > r_in + 1e-6:
                if np.linalg.norm(p - other_center) < r_out + 0.3:
                    continue
                if abs(p[0]) < corridor + 0.6:
                    continue
            out.append(Cone(float(p[0]), float(p[1]), color))
        return out

    n_in = max(ev.inner_cones, int(math.ceil(2 * math.pi * r_in / 5.0)))
    n_out = max(ev.outer_cones, int(math.ceil(2 * math.pi * r_out / 4.8)))
    # Right circle is driven clockwise: centre on the right -> inner = yellow, outer = blue.
    cones += ring(c_right, r_in, n_in, ConeColor.YELLOW, c_left)
    cones += ring(c_right, r_out, n_out, ConeColor.BLUE, c_left)
    # Left circle is driven counter-clockwise: inner = blue, outer = yellow.
    cones += ring(c_left, r_in, n_in, ConeColor.BLUE, c_right)
    cones += ring(c_left, r_out, n_out, ConeColor.YELLOW, c_right)

    # Entry / exit lanes (small orange) outside the outer circles
    y_lane = math.sqrt(max(r_out**2 - (d - corridor) ** 2, 0.0)) + 0.4
    for y in np.arange(-ev.entry_length, -y_lane + 1e-6, 2.5):
        cones += [Cone(-corridor, float(y), ConeColor.ORANGE_SMALL), Cone(corridor, float(y), ConeColor.ORANGE_SMALL)]
    for y in np.arange(y_lane, ev.exit_length + 1e-6, 2.5):
        cones += [Cone(-corridor, float(y), ConeColor.ORANGE_SMALL), Cone(corridor, float(y), ConeColor.ORANGE_SMALL)]

    # Timekeeping line through both centres (y = 0), crossed in +y
    big = _big_orange_pair((0.0, 0.0), (0, 1), 2 * corridor)
    cones = _drop_near(cones, [[b.x, b.y] for b in big], 0.9) + big

    # Reference path: entry -> 2 laps right (CW) -> 2 laps left (CCW) -> exit
    ds = 0.25
    y0 = -ev.entry_length - veh.front_bumper
    ys_entry = np.arange(y0 - 2.0, 0.0, ds)
    entry = np.stack([np.zeros_like(ys_entry), ys_entry], axis=1)
    n_lap = int(round(2 * math.pi * r_mid / ds))
    laps = ev.laps_per_side
    th_r = math.pi - np.arange(0, laps * n_lap) * (2 * math.pi / n_lap)
    right = c_right + r_mid * np.stack([np.cos(th_r), np.sin(th_r)], axis=1)
    th_l = np.arange(0, laps * n_lap) * (2 * math.pi / n_lap)
    left = c_left + r_mid * np.stack([np.cos(th_l), np.sin(th_l)], axis=1)
    ys = np.arange(0.0, ev.exit_length - 2.0, ds)
    exit_ = np.stack([np.zeros_like(ys), ys], axis=1)
    centerline = np.vstack([entry, right, left, exit_])

    stop_zone = np.array(
        [[-corridor, y_lane], [corridor, y_lane], [corridor, ev.exit_length], [-corridor, ev.exit_length]]
    )
    return Track(
        name="skidpad",
        mission="skidpad",
        cones=cones,
        start_pose=(0.0, -ev.entry_length - veh.front_bumper, math.pi / 2),
        centerline=centerline,
        closed=False,
        width=w,
        laps=2 * laps,
        timing_line=_line((0.0, 0.0), (0, 1), corridor, "timekeeping"),
        stop_zone=stop_zone,
        meta={"r_in": r_in, "r_out": r_out, "center_distance": 2 * d, "laps_per_side": laps},
    )


# --------------------------------------------------------------------------- #
# Random closed circuit (autocross / trackdrive)
# --------------------------------------------------------------------------- #
def _convex_hull(points: np.ndarray) -> np.ndarray:
    pts = sorted(map(tuple, points))
    if len(pts) <= 2:
        return np.array(pts)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])  # counter-clockwise


def _catmull_rom_closed(ctrl: np.ndarray, ds: float, alpha: float = 0.5) -> np.ndarray:
    """Centripetal Catmull-Rom spline through closed control polygon."""
    n = len(ctrl)
    out = []
    for i in range(n):
        p0, p1, p2, p3 = ctrl[(i - 1) % n], ctrl[i], ctrl[(i + 1) % n], ctrl[(i + 2) % n]
        t0 = 0.0
        t1 = t0 + max(np.linalg.norm(p1 - p0), 1e-6) ** alpha
        t2 = t1 + max(np.linalg.norm(p2 - p1), 1e-6) ** alpha
        t3 = t2 + max(np.linalg.norm(p3 - p2), 1e-6) ** alpha
        m = max(2, int(np.linalg.norm(p2 - p1) / ds))
        t = np.linspace(t1, t2, m, endpoint=False)[:, None]
        a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
        a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
        a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
        b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
        b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
        out.append((t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2)
    return np.vstack(out)


def _smooth_closed(p: np.ndarray, window: int, iterations: int) -> np.ndarray:
    k = np.ones(window) / window
    q = p.copy()
    half = window // 2
    for _ in range(iterations):
        ext = np.vstack([q[-half:], q, q[:half]])
        q = np.stack([np.convolve(ext[:, i], k, mode="valid") for i in range(2)], axis=1)[: len(p)]
    return q


def _longest_run(mask: np.ndarray) -> tuple[int, int]:
    """Longest cyclic run of True values -> (start index, length)."""
    n = len(mask)
    if mask.all():
        return 0, n
    if not mask.any():
        return 0, 0
    start = int(np.argmin(mask))  # begin scanning at a False
    best = (0, 0)
    run_start, run_len = None, 0
    for k in range(1, n + 1):
        i = (start + k) % n
        if mask[i]:
            if run_start is None:
                run_start, run_len = i, 0
            run_len += 1
            if run_len > best[1]:
                best = (run_start, run_len)
        else:
            run_start, run_len = None, 0
    return best


def _check_clearance(path: np.ndarray, width: float, ds: float) -> bool:
    step = max(1, int(2.0 / ds))
    sub = path[::step]
    m = len(sub)
    diff = sub[:, None, :] - sub[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    idx = np.arange(m)
    gap = np.abs(idx[:, None] - idx[None, :])
    gap = np.minimum(gap, m - gap) * step * ds  # arc distance
    far = gap > 3.0 * width + 10.0
    return bool(np.all(dist[far] > width + 3.0))


def generate_centerline(gen: TrackGenerator, rng: np.random.Generator, ds: float = 0.5) -> np.ndarray:
    """Random closed centerline that respects radius, length, straight and clearance limits."""
    W, H = gen.box
    lo, hi = gen.length_range
    k_max = 1.0 / gen.min_radius
    for _ in range(500):
        n = int(rng.integers(7, 12))
        pts = rng.uniform([-W / 2, -H / 2], [W / 2, H / 2], size=(n, 2))
        hull = _convex_hull(pts)
        if len(hull) < 4:
            continue
        ctrl = []
        for i in range(len(hull)):
            a, b = hull[i], hull[(i + 1) % len(hull)]
            ctrl.append(a)
            L = np.linalg.norm(b - a)
            if L > 22.0:
                t = (b - a) / L
                inward = np.array([-t[1], t[0]])  # hull is CCW -> left normal points inside
                disp = rng.uniform(-0.15, 0.45) * L * 0.5
                ctrl.append((a + b) / 2 + inward * disp + t * rng.uniform(-0.15, 0.15) * L)
        ctrl = np.array(ctrl)
        # drop control points that are too close to their predecessor
        keep = [ctrl[0]]
        for p in ctrl[1:]:
            if np.linalg.norm(p - keep[-1]) > 9.0:
                keep.append(p)
        if np.linalg.norm(keep[-1] - keep[0]) < 9.0:
            keep.pop()
        if len(keep) < 4:
            continue
        path = _catmull_rom_closed(np.array(keep), ds / 2)
        path = resample_polyline(path, ds, closed=True)
        path = _smooth_closed(path, window=int(6.0 / ds) | 1, iterations=3)
        path = resample_polyline(path, ds, closed=True)

        k = np.max(np.abs(polyline_curvature(path, closed=True)))
        length = polyline_lengths(path, closed=True)[-1]
        scale = 1.0
        if k > k_max:
            scale = k / k_max
        if length * scale < lo:
            scale = lo / length * 1.02
        if length * scale > hi:
            # shrinking only allowed if curvature still fine
            s2 = hi / length * 0.98
            if k / s2 > k_max:
                continue
            scale = s2
        if scale > 1.8:
            continue
        cen = path.mean(axis=0)
        path = cen + (path - cen) * scale
        path = resample_polyline(path, ds, closed=True)
        kappa = polyline_curvature(path, closed=True)
        if np.max(np.abs(kappa)) > k_max * 1.02:
            continue
        straight = np.abs(kappa) < 1.0 / 100.0
        _, run = _longest_run(straight)
        if run * ds > gen.max_straight:
            continue
        if not _check_clearance(path, gen.width, ds):
            continue
        if rng.random() < 0.5:
            path = path[::-1].copy()
        return path
    raise RuntimeError("track generator: no valid layout found, relax the TrackGenerator limits")


def _place_border_cones(border, kappa_center, side, width, gen, color):
    """Walk along a closed border and drop cones; denser in tight corners."""
    n = len(border)
    seg = np.linalg.norm(np.roll(border, -1, axis=0) - border, axis=1)
    radius_c = 1.0 / np.maximum(np.abs(kappa_center), 1e-4)
    inner = np.sign(kappa_center) == side  # left turn (k > 0) -> left border is inner
    radius_b = np.where(inner, np.maximum(radius_c - width / 2, 0.5), radius_c + width / 2)
    spacing = np.clip(0.55 * radius_b, gen.min_cone_spacing, gen.max_cone_spacing)
    cones = [Cone(float(border[0, 0]), float(border[0, 1]), color)]
    acc = 0.0
    last_sp = spacing[0]
    for i in range(1, n):
        acc += seg[i - 1]
        need = min(last_sp, spacing[i])
        if acc >= need:
            cones.append(Cone(float(border[i, 0]), float(border[i, 1]), color))
            acc = 0.0
            last_sp = spacing[i]
    acc += seg[-1]
    if acc < 0.6 * min(last_sp, spacing[0]) and len(cones) > 2:
        cones.pop()
    return cones


def build_closed_track(
    gen: TrackGenerator,
    veh: VehicleParams,
    laps: int,
    mission: str,
    seed: int | None = None,
    centerline: np.ndarray | None = None,
) -> Track:
    ds = 0.5
    rng = np.random.default_rng(gen.seed if seed is None else seed)
    path = generate_centerline(gen, rng, ds) if centerline is None else resample_polyline(centerline, ds, True)
    kappa = polyline_curvature(path, closed=True)

    # Start / finish line in the middle of the longest straight-ish section
    k_s = _smooth_closed(np.abs(kappa)[:, None].repeat(2, 1), int(10 / ds) | 1, 1)[:, 0]
    start, run = _longest_run(k_s < 0.03)
    i0 = (start + run // 2) % len(path) if run > 0 else int(np.argmin(k_s))
    path = np.roll(path, -i0, axis=0)
    kappa = np.roll(kappa, -i0)

    tang = polyline_tangents(path, closed=True)
    normal = np.stack([-tang[:, 1], tang[:, 0]], axis=1)
    w = gen.width
    left = path + normal * w / 2
    right = path - normal * w / 2
    cones = _place_border_cones(left, kappa, +1, w, gen, ConeColor.BLUE)
    cones += _place_border_cones(right, kappa, -1, w, gen, ConeColor.YELLOW)
    big = _big_orange_pair(path[0], tang[0], w)
    cones = _drop_near(cones, [[b.x, b.y] for b in big], 1.2) + big

    back = int(round((gen.staging_distance + veh.lf) / ds))
    sp = path[-back]
    yaw = math.atan2(tang[-back, 1], tang[-back, 0])
    name = mission
    return Track(
        name=name,
        mission=mission,
        cones=cones,
        start_pose=(float(sp[0]), float(sp[1]), yaw),
        centerline=path,
        closed=True,
        width=w,
        laps=laps,
        timing_line=_line(path[0], tang[0], w / 2 + 1.0, "start_finish"),
        meta={"seed": gen.seed if seed is None else seed},
    )


# --------------------------------------------------------------------------- #
# Public entry points
# --------------------------------------------------------------------------- #
MISSIONS = ("acceleration", "skidpad", "autocross", "trackdrive", "ebs_test")


def build_track(mission: str, events: EventsConfig, veh: VehicleParams, seed: int | None = None) -> Track:
    mission = mission.lower()
    if mission == "acceleration":
        return build_acceleration(events.acceleration, veh)
    if mission == "ebs_test":
        t = build_acceleration(events.acceleration, veh, mission="ebs_test")
        t.name = "ebs_test"
        return t
    if mission == "skidpad":
        return build_skidpad(events.skidpad, veh)
    if mission == "autocross":
        return build_closed_track(events.track_generator, veh, events.autocross_laps, "autocross", seed)
    if mission == "trackdrive":
        return build_closed_track(events.track_generator, veh, events.trackdrive_laps, "trackdrive", seed)
    raise ValueError(f"unknown mission '{mission}', expected one of {MISSIONS}")


# --------------------------------------------------------------------------- #
# CSV import / export (EUFS / FSSIM compatible layout)
# --------------------------------------------------------------------------- #
CSV_HEADER = ["tag", "x", "y", "direction", "x_variance", "y_variance", "xy_covariance"]


def save_csv(track: Track, path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADER)
        for c in track.cones:
            w.writerow([CSV_TAGS[c.color], f"{c.x:.4f}", f"{c.y:.4f}", 0, 0, 0, 0])
        x, y, yaw = track.start_pose
        w.writerow(["car_start", f"{x:.4f}", f"{y:.4f}", f"{yaw:.6f}", 0, 0, 0])


def _chain(points: np.ndarray, start_xy: np.ndarray, max_step: float = 8.0) -> np.ndarray:
    """Order points by greedy nearest-neighbour chaining from start_xy."""
    remaining = list(range(len(points)))
    order = []
    cur = np.asarray(start_xy, float)
    while remaining:
        d = [np.linalg.norm(points[i] - cur) for i in remaining]
        j = int(np.argmin(d))
        if order and d[j] > max_step:
            break
        order.append(remaining.pop(j))
        cur = points[order[-1]]
    return points[order]


def centerline_from_cones(blue: np.ndarray, yellow: np.ndarray, start_pose, closed: bool, width_hint: float = 3.5):
    """Reconstruct a centerline from left/right border cones."""
    x0, y0, yaw = start_pose
    head = np.array([math.cos(yaw), math.sin(yaw)])
    start = np.array([x0, y0]) + head * 2.0
    b = _chain(blue, start)
    mids = []
    for p in b:
        d = np.linalg.norm(yellow - p, axis=1)
        j = int(np.argmin(d))
        if d[j] < 2.5 * width_hint:
            mids.append((p + yellow[j]) / 2)
    mids = np.array(mids)
    if len(mids) < 3:
        raise ValueError("not enough blue/yellow cone pairs to build a centerline")
    if np.dot(mids[min(3, len(mids) - 1)] - mids[0], head) < 0:
        mids = mids[::-1]
    path = resample_polyline(mids, 0.5, closed=closed)
    if closed:
        path = _smooth_closed(path, 5, 2)
    border = np.vstack([blue, yellow])
    half = [np.min(np.linalg.norm(path - q, axis=1)) for q in border]
    width = float(2.0 * np.median(half))
    return path, width


def load_csv(path: str, mission: str, events: EventsConfig, veh: VehicleParams) -> Track:
    cones, start_pose = [], None
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tag = row["tag"].strip().lower()
            if tag == "car_start":
                start_pose = (float(row["x"]), float(row["y"]), float(row.get("direction") or 0.0))
                continue
            color = TAG_TO_COLOR.get(tag, ConeColor.UNKNOWN)
            cones.append(Cone(float(row["x"]), float(row["y"]), color))
    if start_pose is None:
        start_pose = (0.0, 0.0, 0.0)
    mission = mission.lower()
    closed = mission in ("autocross", "trackdrive")
    blue = np.array([[c.x, c.y] for c in cones if c.color == ConeColor.BLUE])
    yellow = np.array([[c.x, c.y] for c in cones if c.color == ConeColor.YELLOW])
    if mission == "skidpad":
        raise ValueError("skidpad CSV import is not supported; the layout is generated from EventsConfig")
    centerline, width = centerline_from_cones(blue, yellow, start_pose, closed, events.track_generator.width)
    # Timing line at the big orange cones closest to the start pose
    big = np.array([[c.x, c.y] for c in cones if c.color == ConeColor.ORANGE_BIG]).reshape(-1, 2)
    if len(big):
        sx, sy, _ = start_pose
        order = np.argsort(np.hypot(big[:, 0] - sx, big[:, 1] - sy))
        anchor = big[order[: min(4, len(order))]].mean(axis=0)
    else:
        anchor = centerline[0]
    if mission in ("acceleration", "ebs_test"):
        # straight line through the border midpoints, from behind the car to the last cone; one sample lies
        # exactly on the start line (a 0.25 m grid would move the timing and finish lines by up to 12 cm)
        d = centerline[-1] - centerline[0]
        d = d / np.linalg.norm(d)
        origin = centerline.mean(axis=0)
        origin = origin + float((anchor - origin) @ d) * d
        allpts = np.array([[c.x, c.y] for c in cones] + [list(start_pose[:2])])
        proj = (allpts - origin) @ d
        s = np.arange(-0.25 * math.ceil((2.0 - proj.min()) / 0.25), proj.max() + 1e-6, 0.25)
        centerline = origin + s[:, None] * d[None, :]
    tang = polyline_tangents(centerline, closed)
    i = int(np.argmin(np.linalg.norm(centerline - anchor, axis=1)))
    if closed:
        centerline = np.roll(centerline, -i, axis=0)
        tang = np.roll(tang, -i, axis=0)
        i = 0
    laps = {"autocross": events.autocross_laps, "trackdrive": events.trackdrive_laps}.get(mission, 1)
    start_pose = _stage_behind_line(start_pose, centerline[i], tang[i], veh.front_bumper,
                                    events.acceleration.staging_gap, path)
    track = Track(
        name=f"csv:{path}",
        mission=mission,
        cones=cones,
        start_pose=start_pose,
        centerline=centerline,
        closed=closed,
        width=width,
        laps=laps,
        timing_line=_line(centerline[i], tang[i], width / 2 + 1.0, "start"),
        meta={"source": path},
    )
    if mission in ("acceleration", "ebs_test"):
        L = events.acceleration.length
        s_cl = polyline_lengths(centerline)
        j = int(np.argmin(np.abs(s_cl - s_cl[i] - L)))
        track.finish_line = _line(centerline[j], tang[j], width / 2 + 1.0, "finish")
        # exit lane = finish line to the last cone (as the generated layout), so results report stopped_in_zone
        stop_len = float(s_cl[-1] - s_cl[j])
        nrm = np.array([-tang[j, 1], tang[j, 0]]) * width / 2
        a, b = centerline[j], centerline[j] + tang[j] * stop_len
        track.stop_zone = np.array([a - nrm, b - nrm, b + nrm, a + nrm])
        track.meta.update({"length": float(s_cl[j] - s_cl[i]), "stop_length": stop_len})
    return track


def _stage_behind_line(start_pose, line_p, line_t, front_bumper, gap, path=""):
    """A CSV's car_start puts the front of the car past the start line when the file was written for a car with
    a shorter nose (e.g. before the footprint followed the CAD): the start crossing is then never seen, the lap is
    never timed and the car drives on until it leaves the track. Move such a start back behind the line."""
    x, y, yaw = start_pose
    fwd = np.array([math.cos(yaw), math.sin(yaw)])
    t = np.asarray(line_t, float) / max(np.linalg.norm(line_t), 1e-9)
    if float(np.dot(fwd, t)) < 0.5:          # car not facing along the track at the line: leave it as written
        return start_pose
    ahead = float(np.dot(np.array([x, y]) + fwd * front_bumper - np.asarray(line_p, float), t))
    if not (-0.01 < ahead < 3.0):            # behind the line (as written), or clearly somewhere else
        return start_pose
    back = ahead + gap
    print(f"[fsdsim] {os.path.basename(str(path))}: car_start puts the front of the car {ahead:+.2f} m from the start "
          f"line; moved back {back:.2f} m (front {gap:.2f} m behind the line)", flush=True)
    return (x - fwd[0] * back, y - fwd[1] * back, yaw)
