"""LiDAR cone positions + camera cone colours.

The LiDAR measures where a cone is to a few centimetres but (in Isaac Sim) not its colour; the camera knows the
colour but its depth error grows with the square of the distance. Each LiDAR cone takes the colour of the camera
detection in the same direction (bearing within 2.5 deg + what 0.3 m sideways gives at that range) and at a
compatible distance (20 % + 0.5 m). Camera detections without a LiDAR cone can be kept as camera-only cones
(`keep_camera_only`), LiDAR cones without a camera match keep their own colour guess."""

from __future__ import annotations

import math

import numpy as np

from ..geometry import rpy_to_matrix

UNKNOWN = 4


def camera_to_base(dets: list[dict], cam_cfg) -> list[dict]:
    """Analytic / YOLO camera detections with `position` in the left optical frame (x right, y down, z forward)
    -> cones in the car frame: {"x", "y", "color", "confidence", "source": "camera"}."""
    m = cam_cfg.mount
    R = rpy_to_matrix(m.roll, m.pitch, m.yaw)
    t = np.array([m.x, m.y + cam_cfg.baseline / 2, m.z])
    out = []
    for d in dets:
        px, py, pz = d["position"]
        b = R @ np.array([pz, -px, -py]) + t
        out.append({"x": float(b[0]), "y": float(b[1]), "range": float(math.hypot(b[0], b[1])), "color": int(d["color"]),
                    "confidence": float(d.get("confidence", 1.0)), "source": "camera", "t": 0.0})
    return out


def fuse(lidar: list[dict], camera: list[dict], keep_camera_only: bool = False, max_camera_only_range: float = 12.0) -> list[dict]:
    out = [dict(c) for c in lidar]
    used = set()
    if camera:
        cb = np.array([math.atan2(c["y"], c["x"]) for c in camera])
        cr = np.array([c["range"] for c in camera])
        for c in out:
            b, r = math.atan2(c["y"], c["x"]), c["range"]
            gate = math.radians(2.5) + 0.3 / max(r, 1.0)
            db = np.abs((cb - b + math.pi) % (2 * math.pi) - math.pi)
            ok = (db < gate) & (np.abs(cr - r) < 0.2 * r + 0.5)
            if not ok.any():
                continue
            j = int(np.argmin(np.where(ok, db, np.inf)))
            used.add(j)
            c["color"] = camera[j]["color"]
            c["source"] = "lidar+camera"
    if keep_camera_only:
        for j, c in enumerate(camera):
            if j not in used and c["range"] <= max_camera_only_range:
                out.append(dict(c))
    return out


def fuse_world(lidar: list[dict], camera: list[dict]) -> list[dict]:
    """Colours for LiDAR cones from camera cones, both already in the world (map) frame ("wx", "wy").

    Matching in the world frame makes the camera's age harmless: each camera cone was placed with the pose at its
    own stamp, each LiDAR cone with the pose at its firing time (fsdsim.perception.pipeline). Camera cones are tried
    newest first. Gate: 0.35 m + 3 % of the range (stereo depth error)."""
    out = [dict(c) for c in lidar]
    if not camera or not out:
        return out
    cw = np.array([[c["wx"], c["wy"]] for c in camera])
    for c in out:
        d = np.hypot(cw[:, 0] - c["wx"], cw[:, 1] - c["wy"])
        gate = 0.35 + 0.03 * c.get("range", 10.0)
        j = int(np.argmin(d))
        if d[j] < gate:
            c["color"] = camera[j]["color"]
            c["source"] = "lidar+camera"
    return out
