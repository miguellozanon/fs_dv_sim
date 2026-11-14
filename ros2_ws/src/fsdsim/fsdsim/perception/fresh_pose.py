"""A fresh SLAM pose for the driver: its own dead reckoning, anchored on the (late) SLAM pose.

The SLAM pose (/slam/odom) arrives 0.15-0.27 s after its stamp in Isaac Sim: the perception node takes the full
LiDAR clouds and its inputs reach it late (team PC, 11 Nov 2026). Carrying that pose forward on the steering arc
for so long put the racing line off the track. Instead the driver integrates the wheel speeds and the gyro itself,
which reach it within a frame, and only takes the SLAM *correction* from the perception node: the offset between
the SLAM pose and its own dead reckoning at the SLAM pose's stamp (map -> odom, as in ROS navigation). The offset
changes slowly, so its age does not matter.

The motion model is the one of fsdsim.perception.pipeline (ground-speed sensor, else the undriven front wheels).
"""

from __future__ import annotations

import math
from collections import deque


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class FreshPose:
    def __init__(self, wheel_radius: float, lr: float, history: float = 2.0):
        self.wheel_r, self.lr = float(wheel_radius), float(lr)
        self.history = history
        self.hist: deque = deque()          # (t, x, y, yaw) in the dead-reckoning frame
        self.t = None
        self.x = self.y = self.yaw = 0.0
        self.v = self.vy = 0.0
        self.r = 0.0
        self.offset = None                  # (dx, dy, dyaw): dead-reckoning frame -> map
        self.anchor_t = None

    def reset(self) -> None:
        self.__init__(self.wheel_r, self.lr, self.history)

    # ------------------------------------------------------------------ #
    def imu(self, yaw_rate: float) -> None:
        self.r = float(yaw_rate)

    def wheels(self, t: float, wheel_rpm, steering_angle: float = 0.0, ground_speed=None) -> None:
        if ground_speed is not None and all(math.isfinite(g) for g in ground_speed):
            v, vy = max(float(ground_speed[0]), 0.0), float(ground_speed[1])
        else:
            w = 0.5 * (wheel_rpm[0] + wheel_rpm[1]) * 2 * math.pi / 60.0
            v, vy = max(w * self.wheel_r * math.cos(float(steering_angle)), 0.0), self.r * self.lr
        if self.t is not None and t > self.t:
            dt = min(t - self.t, 0.2)
            ym = self.yaw + 0.5 * self.r * dt
            c, s = math.cos(ym), math.sin(ym)
            self.x += (v * c - vy * s) * dt
            self.y += (v * s + vy * c) * dt
            self.yaw = _wrap(self.yaw + self.r * dt)
        if self.t is None or t >= self.t:
            self.t, self.v, self.vy = t, v, vy
            self.hist.append((t, self.x, self.y, self.yaw))
            while self.hist and self.hist[0][0] < t - self.history:
                self.hist.popleft()

    def _at(self, t: float):
        h = self.hist
        if not h or t < h[0][0] - 1e-6:
            return None
        if t >= h[-1][0]:
            return h[-1][1:]
        lo, hi = 0, len(h) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if h[mid][0] <= t:
                lo = mid
            else:
                hi = mid
        a, b = h[lo], h[hi]
        f = (t - a[0]) / max(b[0] - a[0], 1e-9)
        return a[1] + f * (b[1] - a[1]), a[2] + f * (b[2] - a[2]), a[3] + f * _wrap(b[3] - a[3])

    def anchor(self, t: float, x: float, y: float, yaw: float) -> bool:
        """The SLAM pose at its stamp t. False if t is outside the dead-reckoning history."""
        p = self._at(t)
        if p is None:
            return False
        dyaw = _wrap(yaw - p[2])
        c, s = math.cos(dyaw), math.sin(dyaw)
        self.offset = (x - (c * p[0] - s * p[1]), y - (s * p[0] + c * p[1]), dyaw)
        self.anchor_t = t
        return True

    def pose_at(self, t: float):
        """The pose at time t (within the history) in the map frame, or None."""
        p = self._at(t)
        if self.offset is None or p is None:
            return None
        dx, dy, dyaw = self.offset
        c, s = math.cos(dyaw), math.sin(dyaw)
        return dx + c * p[0] - s * p[1], dy + s * p[0] + c * p[1], _wrap(p[2] + dyaw)

    def pose(self):
        """The current pose in the map frame, or None before the first anchor."""
        if self.offset is None:
            return None
        dx, dy, dyaw = self.offset
        c, s = math.cos(dyaw), math.sin(dyaw)
        return dx + c * self.x - s * self.y, dy + s * self.x + c * self.y, _wrap(self.yaw + dyaw)
