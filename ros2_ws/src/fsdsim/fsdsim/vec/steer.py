"""The reference driver's path following (control.demo_driver.PathFollower) for N cars.

Nearest point in a window around the last index (-20 ... +159 points), lookahead 0.3 s x v + 3 m
(3 ... 9 m), shortened in tight corners (<= 0.45 x corner radius, >= 2 m), pure pursuit from the
rear axle placed STEER_PREVIEW s ahead (DriverParams.steer_preview). Steering is not learned: the policy works
the throttle and the brakes.
"""

from __future__ import annotations

import math

from .backend import Backend
from .tracks import DS, TrackBank

LOOKAHEAD_GAIN, LOOKAHEAD_MIN, LOOKAHEAD_MAX = 0.30, 3.0, 9.0
LOOKAHEAD_CURV, LOOKAHEAD_FLOOR = 0.45, 2.0
STEER_PREVIEW = 0.05                        # [s] as control.demo_driver.DriverParams.steer_preview
NEAR_BACK, NEAR_FWD = 20, 160
AHEAD_K = int(LOOKAHEAD_MAX / DS)          # curvature window for the lookahead reduction
TARGET_WIN = 96                             # points searched for the pure pursuit target


class BatchFollower:
    def __init__(self, B: Backend, bank: TrackBank):
        self.B = B
        self.bank = bank
        self._near = B.arange(NEAR_FWD + NEAR_BACK)[None, :] - NEAR_BACK
        self._ahead = B.arange(AHEAD_K)[None, :]
        self._tgt = B.arange(1, TARGET_WIN + 1)[None, :]

    def nearest(self, tid, idx, x, y):
        B, bk = self.B, self.bank
        w = bk.wrap(tid, idx[:, None] + self._near)
        tt = tid[:, None] + 0 * w
        d = B.hypot(bk.px[tt, w] - x[:, None], bk.py[tt, w] - y[:, None])
        j = B.argmin(d, 1)
        return w[B.arange(len(j) if not B.is_torch else j.shape[0]), j]

    def steer(self, tid, i, x, y, yaw, v, wheelbase, lr, max_steer):
        """Pure pursuit steering command for cars at path index i (after nearest())."""
        B, bk = self.B, self.bank
        pv = B.maximum(v, 0.0) * STEER_PREVIEW
        rx, ry = x + (pv - lr) * B.cos(yaw), y + (pv - lr) * B.sin(yaw)
        ld = B.clip(LOOKAHEAD_GAIN * v + LOOKAHEAD_MIN, LOOKAHEAD_MIN, LOOKAHEAD_MAX)
        n_ld = B.maximum(B.to_long(B.floor(ld / DS)), 1)
        ahead = i[:, None] + self._ahead
        aw = bk.wrap(tid, ahead)
        tt = tid[:, None] + 0 * aw
        valid = self._ahead < n_ld[:, None]
        # open paths: the window ends at the last point
        n = bk.n[tid]
        valid = valid & (bk.closed[tid][:, None] | (ahead < n[:, None]))
        k_max = B.max(B.where(valid, bk.kap[tt, aw], 0.0), 1)
        ld = B.where(k_max > 1e-4, B.minimum(ld, B.maximum(LOOKAHEAD_FLOOR, LOOKAHEAD_CURV / B.maximum(k_max, 1e-9))), ld)
        # first point beyond ld from the rear axle
        cand = bk.wrap(tid, i[:, None] + self._tgt)
        tt = tid[:, None] + 0 * cand
        cx, cy = bk.px[tt, cand], bk.py[tt, cand]
        far = B.hypot(cx - rx[:, None], cy - ry[:, None]) >= ld[:, None]
        # open path: the last point is the target when nothing is far enough
        last = (~bk.closed[tid][:, None]) & (cand >= (n - 1)[:, None])
        hit = far | last
        first = B.argmax(B.to_float(hit), 1)
        none = ~B.any(hit, 1)
        first = B.where(none, TARGET_WIN - 1, first)
        rows = B.arange(first.shape[0])
        tx, ty = cx[rows, first], cy[rows, first]
        dx, dy = tx - rx, ty - ry
        c, s = B.cos(yaw), B.sin(yaw)
        lx, ly = c * dx + s * dy, -s * dx + c * dy
        ld2 = B.maximum(lx * lx + ly * ly, 1e-3)
        return B.clip(B.atan2(2.0 * wheelbase * ly, ld2), -max_steer, max_steer)

    def tracking_error(self, tid, i, x, y, yaw):
        """(e_lat, e_head, along) w.r.t. path point i and the tangent towards i + 4 (i - 4 at an open end);
        along: signed distance along the tangent (continuous progress between path points)."""
        B, bk = self.B, self.bank
        n = bk.n[tid]
        j = bk.wrap(tid, i + 4)
        at_end = j == i
        jb = B.maximum(i - 4, 0)
        a = B.where(at_end, jb, i)
        b = B.where(at_end, i, j)
        p0x, p0y = bk.px[tid, i], bk.py[tid, i]
        tx, ty = bk.px[tid, b] - bk.px[tid, a], bk.py[tid, b] - bk.py[tid, a]
        tn = B.maximum(B.hypot(tx, ty), 1e-9)
        tx, ty = tx / tn, ty / tn
        e_lat = (x - p0x) * -ty + (y - p0y) * tx
        along = (x - p0x) * tx + (y - p0y) * ty
        h = yaw - B.atan2(ty, tx)
        del n
        return e_lat, B.atan2(B.sin(h), B.cos(h)), along


def unwrap_progress(B: Backend, d, n, closed):
    """Index difference on closed tracks across the start (d in -n/2 ... n/2)."""
    half = n // 2
    d2 = B.where(d < -half, d + n, B.where(d > half, d - n, d))
    return B.where(closed, d2, d)


__all__ = ["BatchFollower", "unwrap_progress", "math"]
