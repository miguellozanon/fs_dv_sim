"""Small geometry helpers (ROS REP-103 conventions: x forward, y left, z up)."""

from __future__ import annotations

import math

import numpy as np


def wrap_angle(a: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def rot2d(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]])


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Rotation matrix R = Rz(yaw) @ Ry(pitch) @ Rx(roll) (ROS fixed-axis RPY)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def quat_from_rpy(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """Quaternion (x, y, z, w) from ROS RPY."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def quat_from_matrix(R: np.ndarray) -> tuple[float, float, float, float]:
    """Quaternion (x, y, z, w) from a 3x3 rotation matrix."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return (x, y, z, w)


def world_to_local(points_xy: np.ndarray, pose: tuple[float, float, float]) -> np.ndarray:
    """Transform (N,2) world points into the frame of pose (x, y, yaw)."""
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    d = np.asarray(points_xy, dtype=float) - np.array([x, y])
    return np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], axis=1)


def local_to_world(points_xy: np.ndarray, pose: tuple[float, float, float]) -> np.ndarray:
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    p = np.asarray(points_xy, dtype=float)
    return np.stack([x + c * p[:, 0] - s * p[:, 1], y + s * p[:, 0] + c * p[:, 1]], axis=1)


def segment_intersection(p1, p2, q1, q2):
    """Intersection of segments p1-p2 and q1-q2.

    Returns (s, u) parameters along p and q (both in [0,1]) or None.
    """
    p1 = np.asarray(p1, float)
    p2 = np.asarray(p2, float)
    q1 = np.asarray(q1, float)
    q2 = np.asarray(q2, float)
    r = p2 - p1
    d = q2 - q1
    denom = r[0] * d[1] - r[1] * d[0]
    if abs(denom) < 1e-12:
        return None
    w = q1 - p1
    s = (w[0] * d[1] - w[1] * d[0]) / denom
    u = (w[0] * r[1] - w[1] * r[0]) / denom
    if 0.0 <= s <= 1.0 and 0.0 <= u <= 1.0:
        return s, u
    return None


def polyline_lengths(pts: np.ndarray, closed: bool = False) -> np.ndarray:
    """Cumulative arc length of a polyline (starts at 0)."""
    p = np.asarray(pts, float)
    if closed:
        p = np.vstack([p, p[:1]])
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def resample_polyline(pts: np.ndarray, ds: float, closed: bool = False) -> np.ndarray:
    """Resample a polyline at (approximately) uniform spacing ds."""
    p = np.asarray(pts, float)
    if closed:
        p = np.vstack([p, p[:1]])
    s = polyline_lengths(p)
    total = s[-1]
    n = max(2, int(round(total / ds)))
    if closed:
        sn = np.linspace(0.0, total, n, endpoint=False)
    else:
        sn = np.linspace(0.0, total, n + 1)
    x = np.interp(sn, s, p[:, 0])
    y = np.interp(sn, s, p[:, 1])
    return np.stack([x, y], axis=1)


def polyline_tangents(pts: np.ndarray, closed: bool = False) -> np.ndarray:
    p = np.asarray(pts, float)
    if closed:
        t = np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0)
    else:
        t = np.gradient(p, axis=0)
    n = np.linalg.norm(t, axis=1, keepdims=True)
    n[n < 1e-12] = 1.0
    return t / n


def polyline_curvature(pts: np.ndarray, closed: bool = False) -> np.ndarray:
    """Signed curvature (1/m) of a uniformly sampled polyline (left turn > 0)."""
    p = np.asarray(pts, float)
    if closed:
        d1 = (np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0)) / 2.0
        d2 = np.roll(p, -1, axis=0) - 2 * p + np.roll(p, 1, axis=0)
    else:
        d1 = np.gradient(p, axis=0)
        d2 = np.gradient(d1, axis=0)
    num = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    den = np.power(d1[:, 0] ** 2 + d1[:, 1] ** 2, 1.5)
    den[den < 1e-12] = 1e-12
    return num / den


def circle_rect_distance(cx, cy, half_l, half_w):
    """Distance from points (cx, cy) (rectangle-local coords) to an axis-aligned
    rectangle centred at the origin; 0 if inside. Vectorised."""
    dx = np.maximum(np.abs(cx) - half_l, 0.0)
    dy = np.maximum(np.abs(cy) - half_w, 0.0)
    return np.hypot(dx, dy)
