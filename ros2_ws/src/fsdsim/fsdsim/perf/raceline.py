"""Racing line inside the track limits, for the driverless driver and for LapSim on real tracks.

The line is the centre line moved sideways by a(s) along its normals, |a| <= half width - half car - margin.
With the normals fixed, the line's second differences (curvature) and first differences (length) are linear in a,
so "as straight as possible, not longer than needed" is a bounded least-squares problem:

    minimise  || D2 (c + N a) ||^2  +  w * || D1 (c + N a) ||^2      s.t.  -b <= a <= b

w = 0 is the classic minimum-curvature line; a large w the shortest path. The fastest of a few w is picked with
the car's own QSS model (fsdsim.perf.qss, the LapSim port), so the choice follows the vehicle parameters: a car
with much grip and little power prefers the shorter line. This is not a full minimum-time optimal control
solution (no coupling of the speed profile into the path beyond the choice of w), but it is cheap (well under a
second per candidate), has no dependencies beyond NumPy and typically gains several percent over the centre line.

Own implementation; no code from the TUM global racetrajectory optimisation was used.
"""

from __future__ import annotations

import numpy as np

from ..geometry import polyline_curvature, resample_polyline  # noqa: F401
from . import qss

DEFAULT_WEIGHTS = (0.0, 0.003, 0.01, 0.03, 0.1)


def _bounded_lsq(H: np.ndarray, g: np.ndarray, lo: np.ndarray, hi: np.ndarray, iters: int = 60) -> np.ndarray:
    """min 0.5 a'Ha + g'a with lo <= a <= hi (H positive definite): active-set iterations on the box."""
    n = len(g)
    a = np.clip(np.linalg.solve(H, -g), lo, hi)
    for _ in range(iters):
        grad = H @ a + g
        at_lo, at_hi = a <= lo + 1e-12, a >= hi - 1e-12
        fixed = (at_lo & (grad > 0)) | (at_hi & (grad < 0))      # the bound holds them, the gradient pushes outward
        free = ~fixed
        if not free.any():
            break
        new = a.copy()
        rhs = -g[free] - H[np.ix_(free, fixed)] @ a[fixed]
        new[free] = np.linalg.solve(H[np.ix_(free, free)], rhs)
        new = np.clip(new, lo, hi)
        if np.max(np.abs(new - a)) < 1e-6:
            a = new
            break
        a = new
    return a


def _uniform(pts: np.ndarray, ds: float, closed: bool) -> np.ndarray:
    """Points at exactly equal arc length (closed: n points around the loop, the first one not repeated).
    Equal spacing matters: the difference operators below assume it, and one short segment at the seam of a
    closed line would put a kink there."""
    if closed and np.linalg.norm(pts[0] - pts[-1]) > 1e-6:
        pts = np.vstack([pts, pts[:1]])
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    keep = np.concatenate([[True], seg > 1e-9])
    pts = pts[keep]
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    n = max(int(round(s[-1] / ds)), 8)
    si = np.linspace(0.0, s[-1], n + 1)[:-1] if closed else np.linspace(0.0, s[-1], n + 1)
    return np.stack([np.interp(si, s, pts[:, 0]), np.interp(si, s, pts[:, 1])], axis=1)


def _operators(n: int, closed: bool):
    I = np.eye(n)
    if closed:
        D1 = np.roll(I, -1, axis=0) - I
        D2 = np.roll(I, -1, axis=0) - 2 * I + np.roll(I, 1, axis=0)
    else:
        D1 = (np.roll(I, -1, axis=0) - I)[:-1]
        D2 = (np.roll(I, -1, axis=0) - 2 * I + np.roll(I, 1, axis=0))[1:-1]
    return D1, D2


def offset_line(center, width: float, closed: bool = True, car_width: float = 1.5, margin: float = 0.25,
                ds: float = 1.0, weight: float = 0.0, smooth: float = 1e-3):
    """(points (n, 2), offsets a (n,), bound b): the bounded least-squares line for one length weight."""
    c = _uniform(np.asarray(center, float).reshape(-1, 2), ds, closed)
    n = len(c)
    t = (np.roll(c, -1, 0) - np.roll(c, 1, 0)) if closed else np.gradient(c, axis=0)
    t = t / np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
    N = np.stack([-t[:, 1], t[:, 0]], axis=1)                    # left normal
    b = max(width / 2 - car_width / 2 - margin, 0.0)
    if b <= 1e-6:
        return (np.vstack([c, c[:1]]) if closed else c), np.zeros(n), 0.0
    # tight corners: never move further than 0.8 x the corner radius towards the inside (the normals would cross)
    k = polyline_curvature(np.vstack([c, c[:1]]) if closed else c, closed)[:n]
    r_lim = 0.8 / np.maximum(np.abs(k), 1e-6)
    lo = -np.where(k < 0, np.minimum(b, r_lim), b)               # right-hand corner: inside is a < 0
    hi = np.where(k > 0, np.minimum(b, r_lim), b)
    D1, D2 = _operators(n, closed)
    H = np.zeros((n, n))
    g = np.zeros(n)
    for D, w in ((D2, 1.0), (D1, weight * ds * ds)):
        for j in (0, 1):                                         # x and y components
            A = D * N[:, j][None, :]
            H += w * (A.T @ A)
            g += w * (A.T @ (D @ c[:, j]))
    H += smooth * np.eye(n)
    if not closed:                                               # start and end stay on the centre line
        lo[[0, -1]] = hi[[0, -1]] = 0.0
    a = _bounded_lsq(H, g, lo, hi)
    p = c + N * a[:, None]
    return (np.vstack([p, p[:1]]) if closed else p), a, b


def racing_line(center, width: float, veh=None, closed: bool = True, car_width: float = 1.5, margin: float = 0.25,
                ds: float = 1.0, weights=DEFAULT_WEIGHTS, event: str = "") -> dict:
    """The fastest line of the family for this car.

    veh: fsdsim.perf.PerfParams (with the driver's utilisation applied, if the lap times should be the driver's);
    None: the minimum-curvature line without lap times.
    Returns {"xy", "offset", "bound", "weight", "t_center", "t_line", "candidates": [(weight, time)], "length",
    "length_center"}; "xy" is closed (first point repeated) for closed tracks."""
    center = np.asarray(center, float).reshape(-1, 2)

    def lap(xy):
        return float(qss.run_path(veh, xy, closed, event=event, standing_start=False).time)

    def length(xy):
        return float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))

    cl = resample_polyline(center, ds, closed)
    out = {"t_center": lap(cl) if veh is not None else None, "length_center": length(cl), "candidates": []}
    best = None
    for w in (weights if veh is not None else weights[:1]):
        xy, a, b = offset_line(center, width, closed, car_width, margin, ds, w)
        t = lap(xy) if veh is not None else 0.0
        out["candidates"].append((float(w), t))
        if best is None or t < best[0] - 1e-9:
            best = (t, w, xy, a, b)
    t, w, xy, a, b = best
    out.update(xy=xy, offset=a, bound=b, weight=float(w), t_line=t if veh is not None else None, length=length(xy))
    return out
