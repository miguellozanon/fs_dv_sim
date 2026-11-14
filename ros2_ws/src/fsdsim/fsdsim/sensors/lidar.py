"""Analytic spinning-LiDAR model (default: Ouster OS1-64).

Used by the lightweight backend (no GPU) and as a reference to validate the
Isaac Sim RTX LiDAR. It ray-casts every beam against

* the ground plane (z = 0), and
* every cone, modelled as a vertical frustum with its stripe layout,

and returns points in the *sensor frame at the firing time of each column*,
exactly like the Ouster driver does. With ``motion_distortion`` enabled the
vehicle pose is interpolated per column across the scan, so fast driving skews
the cloud the same way the real sensor does (your stack has to de-skew it).

Point fields follow ouster-ros: x, y, z, intensity, t (ns since scan start),
reflectivity, ring (0 = top beam), range (mm).
"""

from __future__ import annotations

import math

import numpy as np

from ..config import LidarConfig
from ..geometry import rpy_to_matrix
from ..track import CONE_SPECS, ConeColor

# Rough reflectivity values (0-255 scale) per surface
REFLECTIVITY = {"blue": 25.0, "yellow": 90.0, "orange": 70.0, "white": 210.0, "black": 6.0, "asphalt": 12.0}


class LidarModel:
    def __init__(self, cfg: LidarConfig, rng: np.random.Generator | None = None):
        self.cfg = cfg
        self.rng = rng or np.random.default_rng(0)
        self.period = 1.0 / cfg.rate_hz
        ch, hres = int(cfg.channels), int(cfg.horizontal_resolution)
        vmin, vmax = np.deg2rad(cfg.vertical_fov_deg)
        # Ouster: beam 0 is the top beam; columns sweep clockwise seen from above.
        self.elev = np.linspace(vmax, vmin, ch)
        self.azim = -2.0 * np.pi * np.arange(hres) / hres
        ce, se = np.cos(self.elev)[:, None], np.sin(self.elev)[:, None]
        ca, sa = np.cos(self.azim)[None, :], np.sin(self.azim)[None, :]
        self.dir_sensor = np.stack(
            [ce * ca * np.ones_like(sa), ce * sa * np.ones_like(ca), se * np.ones_like(ca)], axis=-1
        )  # (ch, hres, 3) unit vectors in the sensor frame
        m = cfg.mount
        self.R_mount = rpy_to_matrix(m.roll, m.pitch, m.yaw)
        self.t_mount = np.array([m.x, m.y, m.z])
        self.dir_vehicle = self.dir_sensor @ self.R_mount.T  # (ch, hres, 3) in base_link
        self.col_time = np.arange(hres) / hres * self.period

    # ------------------------------------------------------------------ #
    def scan(self, pose_start, pose_end, cones, attitude=(0.0, 0.0, 0.0)) -> dict:
        """Simulate one full revolution.

        pose_start / pose_end: (x, y, yaw) of base_link at the scan start / end.
        cones: iterable of track.Cone (knocked cones are ignored).
        attitude: (roll, pitch, heave) of the body on its suspension during the scan (fsdsim.chassis): the
        sensor tilts and sinks with the body, so the ground and the cones move in the cloud as on the car.
        Returns a dict of numpy arrays (one entry per valid return).
        """
        cfg = self.cfg
        ch, hres = self.dir_sensor.shape[:2]
        x0, y0, yaw0 = pose_start
        x1, y1, yaw1 = pose_end if cfg.motion_distortion else pose_start
        dyaw = (yaw1 - yaw0 + math.pi) % (2 * math.pi) - math.pi
        frac = np.arange(hres) / hres
        cx = x0 + (x1 - x0) * frac
        cy = y0 + (y1 - y0) * frac
        cyaw = yaw0 + dyaw * frac
        cos_y, sin_y = np.cos(cyaw), np.sin(cyaw)

        # Sensor origin per column; the body rotates about base_link (on the ground under the CoG) and heaves
        roll, pitch, heave = attitude
        if roll or pitch or heave:
            R_att = rpy_to_matrix(roll, pitch, 0.0)
            tm = R_att @ self.t_mount + np.array([0.0, 0.0, heave])
            dv = self.dir_vehicle @ R_att.T
        else:
            tm, dv = self.t_mount, self.dir_vehicle
        ox = cx + cos_y * tm[0] - sin_y * tm[1]
        oy = cy + sin_y * tm[0] + cos_y * tm[1]
        oz = tm[2]

        Dx = cos_y[None, :] * dv[..., 0] - sin_y[None, :] * dv[..., 1]
        Dy = sin_y[None, :] * dv[..., 0] + cos_y[None, :] * dv[..., 1]
        Dz = dv[..., 2]

        rng_out = np.full((ch, hres), np.inf)
        refl = np.zeros((ch, hres))

        # Ground plane
        down = Dz < -1e-6
        t_ground = np.where(down, -oz / np.where(down, Dz, -1.0), np.inf)
        ground = t_ground <= cfg.max_range
        rng_out[ground] = t_ground[ground]
        refl[ground] = REFLECTIVITY["asphalt"]

        # Cones
        mx, my = 0.5 * (ox[0] + ox[-1]), 0.5 * (oy[0] + oy[-1])
        yaw_mid = yaw0 + 0.5 * dyaw + self._mount_yaw()
        travel = math.hypot(ox[-1] - ox[0], oy[-1] - oy[0])
        # all cones in one array pass: (cone, column) pairs, then the nearest hit per beam
        sel_cols, sel_cone = [], []
        geo = []
        for cone in cones:
            if cone.knocked:
                continue
            dxc, dyc = cone.x - mx, cone.y - my
            dist = math.hypot(dxc, dyc)
            if dist > cfg.max_cone_range or dist < 0.05:
                continue
            spec = CONE_SPECS[cone.color]
            # widen the column window by the bearing change caused by rotation and translation
            sweep = abs(dyaw) + travel / max(dist - spec.r_bottom, 0.3)
            cols = self._column_window(math.atan2(dyc, dxc) - yaw_mid, spec.r_bottom, dist, sweep)
            if cols.size == 0:
                continue
            sel_cols.append(cols)
            sel_cone.append(np.full(cols.size, len(geo)))
            geo.append((cone.x, cone.y, spec))
        if geo:
            self._hit_cones(np.concatenate(sel_cols), np.concatenate(sel_cone), geo, ox, oy, oz, Dx, Dy, Dz, rng_out, refl)

        valid = (rng_out >= cfg.min_range) & (rng_out <= cfg.max_range)
        if cfg.dropout_prob > 0:
            valid &= self.rng.random(valid.shape) >= cfg.dropout_prob
        rows, cols = np.nonzero(valid)
        r = rng_out[rows, cols] + self.rng.normal(0.0, cfg.range_noise_std, rows.size)
        pts = self.dir_sensor[rows, cols] * r[:, None]
        reflectivity = np.clip(refl[rows, cols] * self.rng.normal(1.0, 0.08, rows.size), 0, 255)
        return {
            "xyz": pts.astype(np.float32),
            "intensity": reflectivity.astype(np.float32),
            "reflectivity": reflectivity.astype(np.uint16),
            "ring": rows.astype(np.uint16),
            "t": (self.col_time[cols] * 1e9).astype(np.uint32),
            "range": (np.maximum(r, 0.0) * 1000.0).astype(np.uint32),
        }

    # ------------------------------------------------------------------ #
    def _mount_yaw(self) -> float:
        return math.atan2(self.R_mount[1, 0], self.R_mount[0, 0])

    def _column_window(self, rel_az: float, radius: float, dist: float, dyaw: float) -> np.ndarray:
        hres = self.dir_sensor.shape[1]
        half = math.atan2(radius + 0.05, max(dist - radius, 0.05)) + dyaw + 2 * (2 * math.pi / hres)
        # columns have azimuth -2*pi*j/hres -> j = -az * hres / (2 pi)
        c_center = (-rel_az) * hres / (2 * math.pi)
        c_half = half * hres / (2 * math.pi)
        lo, hi = int(math.floor(c_center - c_half)), int(math.ceil(c_center + c_half))
        if hi - lo >= hres:
            return np.arange(hres)
        return np.arange(lo, hi + 1) % hres

    def _hit_cones(self, cols, idx, geo, ox, oy, oz, Dx, Dy, Dz, rng_out, refl):
        """Ray / cone-frustum intersection for many (column, cone) pairs at once; keeps the nearest hit per beam."""
        H = np.array([g[2].height for g in geo])[idx]
        r0 = np.array([g[2].r_bottom for g in geo])[idx]
        r1 = np.array([g[2].r_top for g in geo])[idx]
        k = (r0 - r1) / H
        dx, dy, dz = Dx[:, cols], Dy[:, cols], Dz[:, cols]
        px = (ox[cols] - np.array([g[0] for g in geo])[idx])[None, :]
        py = (oy[cols] - np.array([g[1] for g in geo])[idx])[None, :]
        c = (r0 - k * oz)[None, :]
        kk = k[None, :]
        A = dx * dx + dy * dy - (kk * dz) ** 2
        B = 2.0 * (px * dx + py * dy) + 2.0 * c * kk * dz
        C = px * px + py * py - c * c
        disc = B * B - 4.0 * A * C
        ok = (disc >= 0.0) & (np.abs(A) > 1e-12)
        sq = np.sqrt(np.where(ok, disc, 0.0))
        A_safe = np.where(ok, A, 1.0)
        Hn = H[None, :]
        best = np.full(dx.shape, np.inf)
        for t in ((-B - sq) / (2 * A_safe), (-B + sq) / (2 * A_safe)):
            z = oz + t * dz
            hit = ok & (t > 0.0) & (z >= 0.0) & (z <= Hn) & ((c - kk * dz * t) >= 0.0)
            best = np.where(hit & (t < best), t, best)
        rows, j = np.nonzero(best < rng_out[:, cols])
        if rows.size == 0:
            return
        t = best[rows, j]
        rel = (oz + t * dz[rows, j]) / H[j]
        values = np.empty(t.size)
        cone_of = idx[j]
        for g, (_, _, spec) in enumerate(geo):
            m = cone_of == g
            if not m.any():
                continue
            v = np.full(int(m.sum()), REFLECTIVITY[spec.body])
            rr = rel[m]
            for z_from, z_to, colour in spec.stripes:
                v = np.where((rr >= z_from) & (rr <= z_to), REFLECTIVITY[colour], v)
            values[m] = v
        # nearest hit wins where two cones share a beam: write farthest first, nearest last
        order = np.argsort(-t, kind="stable")
        rr, cc = rows[order], cols[j[order]]
        rng_out[rr, cc] = t[order]
        refl[rr, cc] = values[order]
