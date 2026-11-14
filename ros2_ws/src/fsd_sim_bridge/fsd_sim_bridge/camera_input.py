"""Shared input of the camera perception nodes: the ZED left image, its depth image, camera info,
the camera pose on the map (TF) and the ground-truth cones (for labels and accuracy)."""

from __future__ import annotations

import os
from collections import deque

import numpy as np
import rclpy
from fsd_sim_msgs.msg import ASStatus, ConeArray
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformListener

from fsdsim.vision.geometry import image_from_ros, transform

# same profiles as ros_interface (kept here so the perception nodes import only what they use)
RELIABLE = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)
LATCHED = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                     history=HistoryPolicy.KEEP_LAST)


def repo_root() -> str:
    env = os.environ.get("FSG_REPO")
    if env and os.path.isdir(env):
        return env
    from fsdsim.perf.lapsim import find_lapsim_dir  # noqa: PLC0415

    lap = find_lapsim_dir()
    return os.path.dirname(os.path.dirname(lap)) if lap else os.getcwd()


class CameraInput:
    """Latest synchronised (RGB, depth, K, T_map_cam, stamp) frame; call frame() from a timer."""

    def __init__(self, node, namespace: str = "/zed/zed_node", map_frame: str = "map"):
        self.node = node
        ns = "/" + namespace.strip("/")
        self.map_frame = map_frame
        self.tf = Buffer()
        self._tfl = TransformListener(self.tf, node)
        self.K = None
        self.size = None
        self.optical = None
        self._depth = deque(maxlen=6)
        self._rgb = None
        self._seq = 0
        self._used = -1
        self.truth = []                    # [(x, y, color, knocked)] map frame
        self.status = None
        node.create_subscription(Image, f"{ns}/left/image_rect_color", self._on_rgb, qos_profile_sensor_data)
        node.create_subscription(Image, f"{ns}/depth/depth_registered", self._on_depth, qos_profile_sensor_data)
        node.create_subscription(CameraInfo, f"{ns}/left/camera_info", self._on_info, qos_profile_sensor_data)
        node.create_subscription(ConeArray, "/ground_truth/cones", self._on_truth, LATCHED)
        node.create_subscription(ASStatus, "/as/status", self._on_status, RELIABLE)

    @staticmethod
    def _stamp(m) -> float:
        return m.header.stamp.sec + m.header.stamp.nanosec * 1e-9

    def _on_info(self, m: CameraInfo) -> None:
        self.K = (m.k[0], m.k[4], m.k[2], m.k[5])
        self.size = (m.width, m.height)
        self.optical = m.header.frame_id

    def _on_rgb(self, m: Image) -> None:
        self._rgb = m
        self._seq += 1

    def _on_depth(self, m: Image) -> None:
        self._depth.append(m)

    def _on_truth(self, m: ConeArray) -> None:
        self.truth = [(c.position.x, c.position.y, int(c.color), bool(c.knocked)) for c in m.cones]

    def _on_status(self, m: ASStatus) -> None:
        self.status = m

    def pose(self, frame: str, stamp) -> np.ndarray | None:
        try:
            t = self.tf.lookup_transform(self.map_frame, frame, stamp)
        except Exception:  # noqa: BLE001
            try:
                t = self.tf.lookup_transform(self.map_frame, frame, rclpy.time.Time())
            except Exception:  # noqa: BLE001
                return None
        tr, q = t.transform.translation, t.transform.rotation
        return transform((tr.x, tr.y, tr.z), (q.x, q.y, q.z, q.w))

    def frame(self, need_depth: bool = True) -> dict | None:
        """The newest RGB frame not handed out yet, with its depth image and camera pose; else None."""
        m = self._rgb
        if m is None or self._seq == self._used or self.K is None:
            return None
        ts = self._stamp(m)
        depth_msg = None
        for d in reversed(self._depth):
            if abs(self._stamp(d) - ts) < 1e-3:
                depth_msg = d
                break
        if need_depth and depth_msg is None:
            return None
        self._used = self._seq
        T = self.pose(m.header.frame_id or self.optical, m.header.stamp)
        if T is None:
            return None
        rgb = image_from_ros(m.encoding, m.height, m.width, m.step, bytes(m.data))
        depth = None
        if depth_msg is not None:
            depth = image_from_ros(depth_msg.encoding, depth_msg.height, depth_msg.width, depth_msg.step,
                                   bytes(depth_msg.data))
        if rgb is None or rgb.ndim != 3:
            return None
        return {"t": ts, "rgb": rgb, "depth": depth, "K": self.K, "size": (m.width, m.height), "T": T}
