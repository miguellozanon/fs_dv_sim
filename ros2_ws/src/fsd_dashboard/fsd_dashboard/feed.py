"""Feed the dashboard model from fsdsim event dicts (the same events the ROS bridge publishes).

Used by the offline demo, and usable from any process that runs `fsdsim.Simulation`.
"""

from __future__ import annotations

import math

import numpy as np

from fsdsim.geometry import rpy_to_matrix

from .imaging import colorize_depth, downscale, encode_rgb
from .model import DashboardModel


def lidar_to_map(xyz: np.ndarray, base_pose, mount) -> np.ndarray:
    """LiDAR-frame points -> map frame, with the base pose (x, y, yaw) and the sensor mount."""
    p = np.asarray(xyz, np.float64).reshape(-1, 3)
    r = rpy_to_matrix(mount.roll, mount.pitch, mount.yaw)
    base = p @ r.T + np.array([mount.x, mount.y, mount.z])
    x, y, yaw = base_pose
    c, s = math.cos(yaw), math.sin(yaw)
    out = np.empty_like(base)
    out[:, 0] = x + c * base[:, 0] - s * base[:, 1]
    out[:, 1] = y + s * base[:, 0] + c * base[:, 1]
    out[:, 2] = base[:, 2]
    return out


class FsdsimFeed:
    def __init__(self, model: DashboardModel, sensors):
        self.m = model
        self.sc = sensors
        self.pose = (0.0, 0.0, 0.0)

    def handle(self, ev: dict) -> None:
        t = ev.get("type")
        stamp = float(ev.get("stamp", 0.0))
        m = self.m
        if t == "track":
            m.set_track(ev["centerline"], ev["closed"], ev.get("name", ""), ev.get("mission", ""), ev.get("laps", 0),
                        ev.get("timing_line"), ev.get("finish_line"), ev.get("width"))
        elif t == "cones":
            m.set_cones(ev["cones"])
        elif t == "as_status":
            m.tick("status", stamp)
            m.set_status(ev)
        elif t == "ground_truth":
            m.tick("ground_truth", stamp)
            x, y, yaw = ev["pose"]
            vx, vy, r = ev["twist"]
            self.pose = (x, y, yaw)
            m.set_pose(stamp, x, y, yaw, vx, vy, r)
        elif t == "vehicle_sensors":
            m.tick("vehicle", stamp)
            m.set_vehicle(stamp, ev["wheel_rpm"], ev["steering_angle"])
        elif t == "imu":
            m.tick("imu", stamp)
            a = ev["linear_acceleration"]
            m.set_imu(stamp, a[0], a[1], ev["angular_velocity"][2])
        elif t == "gnss":
            m.tick("gnss", stamp)
        elif t == "lidar":
            m.tick("lidar", stamp)
            m.set_lidar(stamp, lidar_to_map(ev["cloud"]["xyz"], self.pose, self.sc.lidar.mount))
        elif t == "camera_detections":
            m.tick("detections", stamp)
        elif t == "image":
            self.image(ev["topic"], ev["data"], stamp)

    def image(self, topic: str, data: np.ndarray, stamp: float) -> None:
        if "depth" in topic:
            self.m.tick("depth", stamp)
            img = colorize_depth(downscale(data))
            name = "depth"
        else:
            self.m.tick("camera", stamp)
            img = downscale(data)
            name = "left"
        buf, mime = encode_rgb(img)
        self.m.set_image(name, buf, mime, img.shape[1], img.shape[0], stamp)
