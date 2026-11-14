"""Cone SLAM: where the car is and where the cones are, without ground truth.

    predict(t, v, yaw_rate)     dead reckoning: speed from the wheel-speed sensors, yaw rate from the gyro
    update(t, cones)            cones in the car frame (LiDAR / fusion) -> pose correction + cone map

Each observation is put on the map with the pose the car had when that cone was measured (pose history; a
spinning LiDAR hits the cones at different times within a scan). Observations are matched to mapped cones (nearest
within the gate). From the matches with established cones the rigid 2D transform between "where I see them" and
"where they are on my map" is solved (Kabsch) and a share of it applied to the pose; then the matched cones are
refined (running mean, colour vote) and unmatched observations start new cones. A cone counts once it was seen
`min_hits` times; candidates never confirmed are dropped. Closing the lap needs no special step as long as the
drift over a lap stays inside the gate (it does here: wheel speeds without drive slip + gyro).

This is a small landmark SLAM for testing the stack, not an EKF / graph SLAM: no covariance, no global
optimisation. The pose starts at the known start box (`set_pose`), as on the car.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

UNKNOWN = 4


class ConeSlam:
    def __init__(self, gate: float = 0.9, min_hits: int = 3, gain: float = 0.5, max_range: float = 18.0):
        self.gate, self.min_hits, self.gain, self.max_range = gate, min_hits, gain, max_range
        self.reset()

    def reset(self, pose=(0.0, 0.0, 0.0)) -> None:
        self.x, self.y, self.yaw = (float(v) for v in pose)
        self.v = self.vy = 0.0
        self.t = None
        self.hist = deque(maxlen=400)              # (t, x, y, yaw)
        self.lm = np.zeros((0, 2))                 # landmark positions
        self.hits = np.zeros(0, int)
        self.votes = np.zeros((0, 5), float)       # colour votes (blue, yellow, orange, big orange, unknown)
        self.first_t = np.zeros(0)
        self.last_t = np.zeros(0)
        self.corrections = 0
        self.last_correction = (0.0, 0.0, 0.0)

    def set_pose(self, x: float, y: float, yaw: float) -> None:
        self.x, self.y, self.yaw = float(x), float(y), float(yaw)
        self.hist.clear()

    # ------------------------------------------------------------------ #
    def predict(self, t: float, v: float, yaw_rate: float, vy: float = 0.0) -> None:
        """v, vy: speed of the car frame's origin along / across the car (vy matters: in a fast corner the car
        slides a few degrees, which is 0.5 m/s sideways at 13 m/s)."""
        if self.t is not None and t > self.t:
            dt = min(t - self.t, 0.2)
            yaw_mid = self.yaw + 0.5 * yaw_rate * dt
            c, s = math.cos(yaw_mid), math.sin(yaw_mid)
            self.x += (v * c - vy * s) * dt
            self.y += (v * s + vy * c) * dt
            self.yaw = (self.yaw + yaw_rate * dt + math.pi) % (2 * math.pi) - math.pi
        self.t, self.v, self.vy = t, v, vy
        self.hist.append((t, self.x, self.y, self.yaw))

    def pose_at(self, t: float) -> tuple[float, float, float]:
        """Pose at time t from the history (linear interpolation; the current pose outside it)."""
        h = self.hist
        if not h:
            return self.x, self.y, self.yaw
        if t >= h[-1][0]:                           # a little ahead of the last odometry sample: keep moving
            dt = min(t - h[-1][0], 0.1)
            c, s = math.cos(self.yaw), math.sin(self.yaw)
            return self.x + (self.v * c - self.vy * s) * dt, self.y + (self.v * s + self.vy * c) * dt, self.yaw
        if t <= h[0][0]:
            return h[0][1:]
        lo, hi = 0, len(h) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if h[mid][0] <= t:
                lo = mid
            else:
                hi = mid
        a, b = h[lo], h[hi]
        f = (t - a[0]) / max(b[0] - a[0], 1e-9)
        dyaw = (b[3] - a[3] + math.pi) % (2 * math.pi) - math.pi
        return a[1] + f * (b[1] - a[1]), a[2] + f * (b[2] - a[2]), a[3] + f * dyaw

    # ------------------------------------------------------------------ #
    def update(self, t: float, cones: list[dict], stamps: list[float] | None = None) -> dict:
        """cones: [{"x", "y" (car frame), "color"}]; stamps: time each was measured (default: t)."""
        stamps = stamps or [t] * len(cones)
        keep = [i for i, c in enumerate(cones) if math.hypot(c["x"], c["y"]) <= self.max_range]
        cones, stamps = [cones[i] for i in keep], [stamps[i] for i in keep]
        if not cones:
            self._prune(t)
            return {"matched": 0, "new": 0}
        obs = np.zeros((len(cones), 2))
        for i, (c, ts) in enumerate(zip(cones, stamps)):
            px, py, pyaw = self.pose_at(ts)
            cs, sn = math.cos(pyaw), math.sin(pyaw)
            obs[i] = (px + cs * c["x"] - sn * c["y"], py + sn * c["x"] + cs * c["y"])
        match = self._associate(obs)
        est = [(i, j) for i, j in match if self.hits[j] >= self.min_hits]
        if len(est) >= 3:
            dx, dy, dth = self._rigid(obs[[i for i, _ in est]], self.lm[[j for _, j in est]])
            g = self.gain
            dx, dy, dth = g * dx, g * dy, g * dth
            if abs(dth) < 0.1 and math.hypot(dx, dy) < 1.0:          # a wild fit (wrong matches) is not applied
                # the transform maps observed -> map about the origin: apply it to the pose and to the observations
                c, s = math.cos(dth), math.sin(dth)
                self.x, self.y = c * self.x - s * self.y + dx, s * self.x + c * self.y + dy
                self.yaw = (self.yaw + dth + math.pi) % (2 * math.pi) - math.pi
                obs = obs @ np.array([[c, s], [-s, c]]) + np.array([dx, dy])
                self.hist = deque(((ht, c * hx - s * hy + dx, s * hx + c * hy + dy, hyaw + dth)
                                   for ht, hx, hy, hyaw in self.hist), maxlen=self.hist.maxlen)
                self.corrections += 1
                self.last_correction = (dx, dy, dth)
        matched = set()
        for i, j in match:
            matched.add(i)
            n = self.hits[j]
            w = 1.0 / min(n + 1, 30)                                  # running mean, then a slow filter
            self.lm[j] += w * (obs[i] - self.lm[j])
            self.hits[j] += 1
            self.last_t[j] = t
            self.votes[j, int(cones[i].get("color", UNKNOWN))] += 1.0
        new = [i for i in range(len(cones)) if i not in matched]
        if new:
            self.lm = np.vstack([self.lm, obs[new]])
            self.hits = np.concatenate([self.hits, np.ones(len(new), int)])
            v = np.zeros((len(new), 5))
            for k, i in enumerate(new):
                v[k, int(cones[i].get("color", UNKNOWN))] = 1.0
            self.votes = np.vstack([self.votes, v])
            self.first_t = np.concatenate([self.first_t, np.full(len(new), t)])
            self.last_t = np.concatenate([self.last_t, np.full(len(new), t)])
        self._prune(t)
        return {"matched": len(match), "new": len(new), "used_for_pose": len(est)}

    def _associate(self, obs: np.ndarray) -> list[tuple[int, int]]:
        if len(self.lm) == 0:
            return []
        d = np.linalg.norm(obs[:, None, :] - self.lm[None, :, :], axis=2)
        # several observations may belong to one cone: the scan seam shows a cone at the start and at the end of
        # a revolution, and the LiDAR and a camera-only detection can both report it
        j = d.argmin(axis=1)
        return [(int(i), int(j[i])) for i in range(len(obs)) if d[i, j[i]] < self.gate]

    @staticmethod
    def _rigid(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
        """dx, dy, dtheta of the rotation + translation (about the origin) that moves points a onto b."""
        ca, cb = a.mean(axis=0), b.mean(axis=0)
        A, B = a - ca, b - cb
        th = math.atan2(float(np.sum(A[:, 0] * B[:, 1] - A[:, 1] * B[:, 0])), float(np.sum(A[:, 0] * B[:, 0] + A[:, 1] * B[:, 1])))
        c, s = math.cos(th), math.sin(th)
        return float(cb[0] - (c * ca[0] - s * ca[1])), float(cb[1] - (s * ca[0] + c * ca[1])), th

    def _prune(self, t: float) -> None:
        """Drop candidates that were not confirmed within 2 s (false detections)."""
        if len(self.lm) == 0:
            return
        keep = (self.hits >= self.min_hits) | (t - self.first_t < 2.0)
        if not keep.all():
            self.lm, self.hits, self.votes = self.lm[keep], self.hits[keep], self.votes[keep]
            self.first_t, self.last_t = self.first_t[keep], self.last_t[keep]

    # ------------------------------------------------------------------ #
    def pose(self) -> tuple[float, float, float]:
        return self.x, self.y, self.yaw

    def cones(self) -> list[dict]:
        """Confirmed cones: [{"x", "y", "color", "hits"}] in the SLAM (map) frame."""
        out = []
        for j in np.nonzero(self.hits >= self.min_hits)[0]:
            v = self.votes[j]
            col = int(np.argmax(v[:4])) if v[:4].sum() > 0 else UNKNOWN
            out.append({"x": float(self.lm[j, 0]), "y": float(self.lm[j, 1]), "color": col, "hits": int(self.hits[j])})
        return out
