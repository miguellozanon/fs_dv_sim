"""Analytic stereo-camera cone detector (default: Stereolabs ZED 2i).

The lightweight backend has no renderer, so instead of images it outputs what a
good camera perception pipeline would: cone detections with colour, a 2D box in
the left image and a 3D position with realistic stereo depth noise
(sigma_z = z^2 * sigma_disparity / (f * B)). Detection probability falls off
with range and nearer cones occlude farther ones.

The Isaac Sim backend renders real left/right/depth images instead.
"""

from __future__ import annotations

import math

import numpy as np

from ..config import CameraConfig
from ..geometry import rpy_to_matrix
from ..track import CONE_SPECS


class StereoCameraModel:
    def __init__(self, cfg: CameraConfig, rng: np.random.Generator | None = None):
        self.cfg = cfg
        self.rng = rng or np.random.default_rng(1)
        self.width, self.height = cfg.resolution
        self.fx = (self.width / 2.0) / math.tan(math.radians(cfg.hfov_deg) / 2.0)
        self.fy = self.fx
        self.cx, self.cy = self.width / 2.0, self.height / 2.0
        m = cfg.mount
        self.R_mount = rpy_to_matrix(m.roll, m.pitch, m.yaw)  # camera body frame -> base_link
        # left camera sits +baseline/2 along the camera body y axis
        self.t_left = np.array([m.x, m.y, m.z]) + self.R_mount @ np.array([0.0, cfg.baseline / 2, 0.0])

    def intrinsics(self) -> dict:
        return {"fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy, "width": self.width, "height": self.height}

    def _to_optical(self, pts_base: np.ndarray) -> np.ndarray:
        """base_link points (N,3) -> left optical frame (x right, y down, z forward)."""
        body = (pts_base - self.t_left) @ self.R_mount  # R^T applied row-wise
        return np.stack([-body[:, 1], -body[:, 2], body[:, 0]], axis=1)

    def detect(self, pose, cones) -> list[dict]:
        cfg = self.cfg
        x, y, yaw = pose
        c, s = math.cos(yaw), math.sin(yaw)
        cand = []
        for cone in cones:
            if cone.knocked:
                continue
            dx, dy = cone.x - x, cone.y - y
            if dx * dx + dy * dy > (cfg.detection_max_range + 2.0) ** 2:
                continue
            spec = CONE_SPECS[cone.color]
            bx, by = c * dx + s * dy, -s * dx + c * dy
            pts = np.array(
                [
                    [bx, by, spec.height / 2],
                    [bx, by + spec.r_bottom, 0.0],
                    [bx, by - spec.r_bottom, 0.0],
                    [bx, by, spec.height],
                ]
            )
            opt = self._to_optical(pts)
            z = opt[0, 2]
            if z < cfg.min_depth or z > cfg.detection_max_range:
                continue
            u = self.fx * opt[:, 0] / np.maximum(opt[:, 2], 1e-3) + self.cx
            v = self.fy * opt[:, 1] / np.maximum(opt[:, 2], 1e-3) + self.cy
            box = (u[1:].min(), v[3], u[1:].max(), max(v[1], v[2]))
            if box[2] < 0 or box[0] > self.width or box[3] < 0 or box[1] > self.height:
                continue
            cand.append((z, cone, opt[0], box))
        cand.sort(key=lambda e: e[0])

        dets, taken = [], []
        for z, cone, p_opt, box in cand:
            area = max((box[2] - box[0]) * (box[3] - box[1]), 1e-6)
            occluded = False
            for b in taken:
                ix = max(0.0, min(box[2], b[2]) - max(box[0], b[0]))
                iy = max(0.0, min(box[3], b[3]) - max(box[1], b[1]))
                if ix * iy / area > 0.6:
                    occluded = True
                    break
            if occluded:
                continue
            p_det = 0.98 * max(0.0, 1.0 - (z / cfg.detection_max_range) ** 4)
            if self.rng.random() > p_det:
                continue
            taken.append(box)
            sigma_z = z * z * cfg.disparity_noise_px / (self.fx * cfg.baseline)
            z_m = z + self.rng.normal(0.0, sigma_z)
            scale = z_m / z
            pos = np.array([p_opt[0] * scale, p_opt[1] * scale, z_m])
            dets.append(
                {
                    "id": cone.id,
                    "color": int(cone.color),
                    "position": [float(v) for v in pos],
                    "bbox": [float(b) for b in box],
                    "confidence": float(p_det),
                }
            )
        return dets
