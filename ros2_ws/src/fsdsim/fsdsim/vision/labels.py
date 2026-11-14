"""Training labels from the simulator's ground truth: every cone the camera really sees, as a YOLO box.

Each ground-truth cone (map position, colour) is projected into the image with the camera pose and
intrinsics: the box is the projection of its base plate and tip (FS Driverless Specification sizes,
fsdsim.track.CONE_SPECS). The depth image decides what is really visible: a cone behind another cone,
the car or the image border is left out, so the labels match what the camera shows.
"""

from __future__ import annotations

import math

import numpy as np

from ..track import CONE_SPECS, ConeColor
from .geometry import project

MIN_BOX_PX = 6.0          # smaller boxes are not labelled (too far to learn from)
MIN_VISIBLE = 0.35        # share of the cone's depth samples that must see the cone itself
MIN_INSIDE = 0.45         # share of the box that must be inside the image
DEPTH_TOL = 0.25          # m: a depth sample this much nearer than the cone is something in front of it


def cone_points(x: float, y: float, color: int, z0: float = 0.0) -> np.ndarray:
    """3D outline of one cone (base plate corners, body ring, tip) in the map frame."""
    spec = CONE_SPECS.get(ConeColor(int(color)) if int(color) in ConeColor._value2member_map_ else ConeColor.UNKNOWN)
    hb, h = spec.base / 2.0, spec.height
    pts = [(x + sx * hb, y + sy * hb, z0) for sx in (-1, 1) for sy in (-1, 1)]
    for k in range(8):
        a = k * math.pi / 4
        pts.append((x + spec.r_top * math.cos(a), y + spec.r_top * math.sin(a), z0 + h))
    return np.array(pts, dtype=float)


def cone_box(x: float, y: float, color: int, T_map_cam: np.ndarray, K, size: tuple[int, int], z0: float = 0.0):
    """(x1, y1, x2, y2) unclipped pixel box, nearest and farthest depth; None if (partly) behind the camera."""
    uv, z = project(cone_points(x, y, color, z0), T_map_cam, K)
    if np.any(z < 0.3):
        return None
    return float(uv[:, 0].min()), float(uv[:, 1].min()), float(uv[:, 0].max()), float(uv[:, 1].max()), float(z.min()), float(z.max())


def visible_fraction(depth: np.ndarray, box, z_near: float, z_far: float) -> float:
    """Share of depth samples on the cone body (inner part of the box) that see this cone, not something nearer."""
    x1, y1, x2, y2 = box
    h_img, w_img = depth.shape[:2]
    w, h = x2 - x1, y2 - y1
    us = np.linspace(x1 + 0.3 * w, x2 - 0.3 * w, 5)
    vs = np.linspace(y1 + 0.35 * h, y2 - 0.12 * h, 6)
    uu, vv = np.meshgrid(us, vs)
    uu, vv = np.round(uu).astype(int).ravel(), np.round(vv).astype(int).ravel()
    inside = (uu >= 0) & (uu < w_img) & (vv >= 0) & (vv < h_img)
    if not inside.any():
        return 0.0
    d = depth[vv[inside], uu[inside]].astype(float)
    ok = np.isfinite(d) & (d > 0.0)
    sees = ok & (d >= z_near - DEPTH_TOL) & (d <= z_far + 1.5)
    return float(sees.sum()) / float(len(d))


def frame_labels(cones, T_map_cam: np.ndarray, K, size: tuple[int, int], depth: np.ndarray | None = None,
                 max_range: float = 25.0, z0: float = 0.0) -> list[dict]:
    """Ground-truth cones -> boxes the camera sees. cones: iterable of (x, y, color[, knocked]).
    Returns [{cls, box (x1, y1, x2, y2) clipped, cone (x, y), z (m), visible}] sorted near to far."""
    w_img, h_img = size
    cam = T_map_cam[:3, 3]
    out = []
    for c in cones:
        x, y, color = float(c[0]), float(c[1]), int(c[2])
        if len(c) > 3 and c[3]:
            continue                                  # knocked over: not a standing cone any more
        if color not in (0, 1, 2, 3):
            continue
        if math.hypot(x - cam[0], y - cam[1]) > max_range:
            continue
        b = cone_box(x, y, color, T_map_cam, K, size, z0)
        if b is None:
            continue
        x1, y1, x2, y2, zn, zf = b
        cx1, cy1, cx2, cy2 = max(x1, 0.0), max(y1, 0.0), min(x2, w_img - 1.0), min(y2, h_img - 1.0)
        if cx2 <= cx1 or cy2 <= cy1:
            continue
        area = (x2 - x1) * (y2 - y1)
        if area <= 0 or (cx2 - cx1) * (cy2 - cy1) / area < MIN_INSIDE or (cy2 - cy1) < MIN_BOX_PX:
            continue
        vis = visible_fraction(depth, (x1, y1, x2, y2), zn, zf) if depth is not None else 1.0
        if vis < MIN_VISIBLE:
            continue
        out.append({"cls": color, "box": (cx1, cy1, cx2, cy2), "cone": (x, y), "z": zn, "visible": vis})
    out.sort(key=lambda d: d["z"])
    return out


def yolo_lines(labels: list[dict], size: tuple[int, int]) -> list[str]:
    """YOLO txt rows: class cx cy w h (normalised)."""
    w_img, h_img = size
    rows = []
    for d in labels:
        x1, y1, x2, y2 = d["box"]
        rows.append(f"{d['cls']} {(x1 + x2) / 2 / w_img:.6f} {(y1 + y2) / 2 / h_img:.6f} "
                    f"{(x2 - x1) / w_img:.6f} {(y2 - y1) / h_img:.6f}")
    return rows
