"""Cones from one LiDAR scan (Ouster-style cloud: x, y, z in the sensor frame, ring, t, intensity).

1. sensor -> car frame (base_link) with the mount from sensors.yaml
2. own-car filter: everything inside the car's outline (+ margin) is dropped. With the CAD body in Isaac Sim
   the OS1 on the nose sees the front tyres, the helmet and the rear wing; without this they become obstacles.
3. ground: a plane fitted to the low points (least squares, 3 passes). The body pitches and rolls on its
   suspension (fsdsim.chassis), so the ground is not at a fixed height in the sensor frame; the fit absorbs that.
4. points 4 cm .. 0.9 m above the plane are clustered on a 15 cm grid (8-neighbourhood)
5. a cluster is a cone if it is small (<= 0.45 m across, <= 0.75 m high) and has enough points for its range
6. centre: the points lie on the near side of the cone, so the mean is pushed back along the ray;
   `big` from the height (large orange cones are 505 mm, the others 325 mm); `t` = when the cone was hit in the
   scan (for de-skewing with the car's motion); colour guess from the intensity pattern when wanted.

Colour from intensity: blue cones have a white stripe (bright band in the middle), yellow ones a black stripe
(dark band), orange ones are bright with white stripes. This works with the simulator's analytic LiDAR and is the
usual trick on a real Ouster (reflectivity channel); the RTX LiDAR in Isaac Sim does not model the stripes'
reflectivity, so there the colours must come from the camera (fusion).
"""

from __future__ import annotations

import math

import numpy as np

from ..geometry import rpy_to_matrix

BLUE, YELLOW, ORANGE_SMALL, ORANGE_BIG, UNKNOWN = 0, 1, 2, 3, 4


class LidarConeDetector:
    def __init__(self, lidar_cfg, vehicle=None, max_range: float = 25.0, ego_margin: float = 0.25,
                 cell: float = 0.15, colour_from_intensity: bool = True):
        m = lidar_cfg.mount
        self.R = rpy_to_matrix(m.roll, m.pitch, m.yaw)
        self.t = np.array([m.x, m.y, m.z])
        self.max_range = max_range
        self.cell = cell
        self.colour_from_intensity = colour_from_intensity
        self.period = 1.0 / lidar_cfg.rate_hz
        self.ego_margin = ego_margin
        self.set_vehicle(vehicle)
        self.last = {}               # diagnostics of the last scan

    def set_vehicle(self, vehicle) -> None:
        """The car's outline in base_link: LiDAR returns inside it are the car itself (nose, tyres, wings)."""
        m = self.ego_margin
        if vehicle is not None:
            self.ego = (-vehicle.rear_bumper - m, vehicle.front_bumper + m, vehicle.footprint.width / 2 + m)
        else:
            self.ego = (-1.6, 2.2, 1.0)

    # ------------------------------------------------------------------ #
    def detect(self, cloud: dict) -> list[dict]:
        xyz = np.asarray(cloud["xyz"], float)
        if len(xyz) == 0:
            self.last = {"points": 0}
            return []
        p = xyz @ self.R.T + self.t
        inten = np.asarray(cloud.get("intensity", np.zeros(len(p))), float)
        tns = np.asarray(cloud.get("t", np.zeros(len(p))), float)
        r_xy = np.hypot(p[:, 0], p[:, 1])
        x0, x1, yw = self.ego
        own = (p[:, 0] > x0) & (p[:, 0] < x1) & (np.abs(p[:, 1]) < yw)
        keep = ~own & (r_xy < self.max_range) & (p[:, 2] < 1.5)
        n_own = int(np.count_nonzero(own))
        p, inten, tns, r_xy = p[keep], inten[keep], tns[keep], r_xy[keep]
        if len(p) < 20:
            self.last = {"points": len(xyz), "own_car": n_own, "clusters": 0}
            return []
        plane = self._ground(p)
        h = p @ plane[:3] + plane[3]                         # height above the ground plane
        obs = (h > 0.04) & (h < 0.9)
        po, ho, io, to = p[obs], h[obs], inten[obs], tns[obs]
        labels = self._cluster(po[:, :2])
        cones = []
        for lab in range(labels.max() + 1 if len(labels) else 0):
            m = labels == lab
            n = int(np.count_nonzero(m))
            q, hh = po[m], ho[m]
            c = q[:, :2].mean(axis=0)
            rng = float(np.hypot(*c))
            if n < (3 if rng < 10.0 else 2):
                continue
            ext = float(np.max(np.ptp(q[:, :2], axis=0)))
            top = float(hh.max())
            if ext > 0.45 or top > 0.75:
                continue
            c = c + 0.06 * c / max(rng, 1e-6)                # near surface -> cone axis
            big = top > 0.42
            colour = self._colour(hh, io[m], top, big) if self.colour_from_intensity else (ORANGE_BIG if big else UNKNOWN)
            cones.append({"x": float(c[0]), "y": float(c[1]), "range": rng + 0.06, "points": n, "height": top,
                          "big": bool(big), "color": int(colour), "t": float(np.mean(to[m])) * 1e-9, "source": "lidar"})
        self.last = {"points": len(xyz), "own_car": n_own, "obstacle_points": int(np.count_nonzero(obs)),
                     "clusters": int(labels.max() + 1) if len(labels) else 0, "cones": len(cones),
                     "ground_pitch_deg": math.degrees(math.asin(max(-1.0, min(1.0, -plane[0])))),
                     "ground_roll_deg": math.degrees(math.asin(max(-1.0, min(1.0, plane[1])))),
                     "ground_z": float(-plane[3])}
        return cones

    # ------------------------------------------------------------------ #
    @staticmethod
    def _ground(p: np.ndarray) -> np.ndarray:
        """Plane n.x + d = 0 with n pointing up, through the low points (the car frame's z = 0 is the first guess)."""
        plane = np.array([0.0, 0.0, 1.0, 0.0])
        tol = 0.25
        for _ in range(3):
            h = p @ plane[:3] + plane[3]
            m = np.abs(h) < tol
            if np.count_nonzero(m) < 10:
                break
            q = p[m]
            A = np.c_[q[:, 0], q[:, 1], np.ones(len(q))]
            # low points weigh more than the feet of cones: fit z = ax + by + c, then drop what sticks out
            coef, *_ = np.linalg.lstsq(A, q[:, 2], rcond=None)
            res = q[:, 2] - A @ coef
            good = res < np.percentile(res, 80)
            coef, *_ = np.linalg.lstsq(A[good], q[good, 2], rcond=None)
            n = np.array([-coef[0], -coef[1], 1.0])
            norm = np.linalg.norm(n)
            plane = np.array([n[0] / norm, n[1] / norm, n[2] / norm, -coef[2] / norm])
            tol = 0.06
        return plane

    def _cluster(self, xy: np.ndarray) -> np.ndarray:
        """Labels of connected groups of occupied grid cells (8-neighbourhood)."""
        if len(xy) == 0:
            return np.zeros(0, int)
        ij = np.floor(xy / self.cell).astype(np.int64)
        cells = {}
        for k, (i, j) in enumerate(map(tuple, ij)):
            cells.setdefault((i, j), []).append(k)
        label_of = {}
        labels = np.full(len(xy), -1, int)
        cur = 0
        for start in cells:
            if start in label_of:
                continue
            stack = [start]
            label_of[start] = cur
            while stack:
                i, j = stack.pop()
                labels[cells[(i, j)]] = cur
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        nb = (i + di, j + dj)
                        if nb in cells and nb not in label_of:
                            label_of[nb] = cur
                            stack.append(nb)
            cur += 1
        return labels

    @staticmethod
    def _colour(h: np.ndarray, inten: np.ndarray, top: float, big: bool) -> int:
        """Blue / yellow / orange from the reflectivity of the stripe band against the body."""
        if big:
            return ORANGE_BIG
        if len(h) < 4:
            return UNKNOWN
        H = 0.325
        band = (h > 0.30 * H) & (h < 0.62 * H)               # the stripe of the small cones
        body = ~band & (h > 0.05)
        if np.count_nonzero(band) < 1 or np.count_nonzero(body) < 2:
            return UNKNOWN
        ib, io = float(np.median(inten[band])), float(np.median(inten[body]))
        if ib < 0.5 * io:
            return YELLOW                                    # dark stripe on a bright body
        if ib > 1.8 * io:
            return BLUE if io < 45.0 else ORANGE_SMALL       # bright stripe: dark body = blue, bright body = orange
        return UNKNOWN
