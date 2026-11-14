"""Camera geometry: transforms, projection, back-projection (optical frame: x right, y down, z forward)."""

from __future__ import annotations

import math

import numpy as np


def quat_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def transform(translation, quaternion) -> np.ndarray:
    """4x4 from (x, y, z) and (qx, qy, qz, qw): p_parent = T @ p_child."""
    T = np.eye(4)
    T[:3, :3] = quat_to_matrix(*quaternion)
    T[:3, 3] = translation
    return T


def rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def camera_pose(car_x: float, car_y: float, car_yaw: float, mount, baseline_y: float = 0.0, car_z: float = 0.0) -> np.ndarray:
    """map -> left camera optical frame, built like the simulator's TF tree (base_link -> camera_center (mount)
    -> left_camera_frame (y + baseline / 2) -> optical (roll -90 deg, yaw -90 deg)). For tests and tools."""
    def T(R, t):
        m = np.eye(4)
        m[:3, :3], m[:3, 3] = R, t
        return m
    base = T(rpy_matrix(0.0, 0.0, car_yaw), [car_x, car_y, car_z])
    center = T(rpy_matrix(mount.roll, mount.pitch, mount.yaw), [mount.x, mount.y, mount.z])
    left = T(np.eye(3), [0.0, baseline_y, 0.0])
    optical = T(rpy_matrix(-math.pi / 2, 0.0, -math.pi / 2), [0.0, 0.0, 0.0])
    return base @ center @ left @ optical


def project(points_map: np.ndarray, T_map_cam: np.ndarray, K) -> tuple[np.ndarray, np.ndarray]:
    """(N, 3) map points -> (N, 2) pixels and (N,) depths (optical z). K = (fx, fy, cx, cy)."""
    fx, fy, cx, cy = K
    T_cam_map = np.linalg.inv(T_map_cam)
    p = points_map @ T_cam_map[:3, :3].T + T_cam_map[:3, 3]
    z = p[:, 2]
    zs = np.where(np.abs(z) < 1e-9, 1e-9, z)
    return np.stack([fx * p[:, 0] / zs + cx, fy * p[:, 1] / zs + cy], axis=1), z


def back_project(u: float, v: float, depth: float, K) -> np.ndarray:
    """Pixel + optical depth -> point in the camera optical frame."""
    fx, fy, cx, cy = K
    return np.array([(u - cx) / fx * depth, (v - cy) / fy * depth, depth])


def to_map(p_cam: np.ndarray, T_map_cam: np.ndarray) -> np.ndarray:
    return T_map_cam[:3, :3] @ p_cam + T_map_cam[:3, 3]


def ground_point(u: float, v: float, T_map_cam: np.ndarray, K, ground_z: float = 0.0) -> np.ndarray | None:
    """Where the ray through pixel (u, v) meets the ground plane z = ground_z (map), or None."""
    fx, fy, cx, cy = K
    d = T_map_cam[:3, :3] @ np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
    o = T_map_cam[:3, 3]
    if d[2] > -1e-6:
        return None
    s = (ground_z - o[2]) / d[2]
    return o + s * d if s > 0 else None


def image_from_ros(encoding: str, height: int, width: int, step: int, data: bytes):
    """sensor_msgs/Image fields -> HxWx3 uint8 RGB, or HxW float32 depth in metres; None if unsupported."""
    enc = encoding.lower()
    buf = np.frombuffer(data, np.uint8)
    if enc in ("rgb8", "bgr8", "rgba8", "bgra8"):
        ch = 4 if len(enc) == 5 else 3
        img = buf.reshape(height, step)[:, : width * ch].reshape(height, width, ch)[..., :3]
        return np.ascontiguousarray(img[..., ::-1] if enc.startswith("bgr") else img)
    if enc == "32fc1":
        return buf.view(np.float32).reshape(height, step // 4)[:, :width]
    if enc == "16uc1":
        return buf.view(np.uint16).reshape(height, step // 2)[:, :width].astype(np.float32) / 1000.0
    return None
