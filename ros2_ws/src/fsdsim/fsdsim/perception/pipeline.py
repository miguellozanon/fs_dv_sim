"""Perception pipeline fed with the simulator's events: what the driver gets instead of the ground truth.

    pipe = PerceptionPipeline(vehicle, sensors, mode="fusion", slam=True)
    pipe.start(start_pose)
    for ev in sim.step(dt): pipe.handle(ev)
    pose, v = pipe.pose(), pipe.speed()
    cones = pipe.cones()                 # [(x, y, colour)] in the map frame of the pipeline

mode   "lidar"   LiDAR cones, colours from the intensity stripes
       "fusion"  LiDAR cones, colours from the camera detections (analytic ZED detector or YOLO)
       "camera"  camera detections only (position from stereo depth)
slam   True: pose from wheel speeds + gyro, corrected on the cone map (ConeSlam); the ground truth is not used
       False: the cone map is built with the ground-truth pose (perception only)

Events used: lidar, camera_detections, vehicle_sensors, imu, ground_truth (only with slam=False).
"""

from __future__ import annotations

import math

import numpy as np

from collections import deque

from .fusion import camera_to_base, fuse_world
from .lidar_cones import LidarConeDetector
from .slam import ConeSlam


class PerceptionPipeline:
    def __init__(self, vehicle, sensors, mode: str = "fusion", slam: bool = True, max_range: float = 18.0,
                 lidar_colours: bool | None = None):
        """lidar_colours: cone colours from the LiDAR stripe intensities (default: in "lidar" mode). Off for a
        LiDAR whose intensities do not show the stripes (Isaac Sim's RTX LiDAR): the cones stay unknown."""
        if mode not in ("lidar", "fusion", "camera"):
            raise ValueError("perception mode must be lidar, fusion or camera")
        self.mode, self.use_slam = mode, slam
        self.veh, self.sens = vehicle, sensors
        self.det = LidarConeDetector(sensors.lidar, vehicle, max_range=max_range + 4.0,
                                     colour_from_intensity=(mode == "lidar") if lidar_colours is None else
                                     (bool(lidar_colours) and mode == "lidar"))
        if mode == "camera":      # stereo depth is noisy (0.2 m at 8 m, 0.4 m at 12 m): near cones only, gentle pose correction
            self.slam = ConeSlam(max_range=8.0, min_hits=3, gate=1.0, gain=0.05)
        else:
            self.slam = ConeSlam(max_range=max_range, min_hits=3)
        self.wheel_r = vehicle.wheel_radius
        self._start = (0.0, 0.0, 0.0)
        self.reset()

    def reset(self) -> None:
        self.slam.reset()
        self._yaw_rate = 0.0
        self._v = 0.0
        self._cam = []            # last camera cones (car frame)
        self._cam_t = -1e9
        self._gt_hist = deque(maxlen=3000)     # (t, x, y, yaw) ground truth: only for the diagnostics below
        self._diag = deque(maxlen=40)          # recent LiDAR frames (stamp, [(x, y, t)]) for the time-alignment check
        self._last_cloud, self._last_cam_fp = None, None
        self.frozen_frames = 0    # repeated (frozen) sensor frames skipped
        self._cam_buf = deque()   # (stamp, camera cones with world positions "wx", "wy") of the last CAM_KEEP s
        self._gt = None
        self.frames = 0
        self.last_frame = []      # cones of the last frame in the car frame (after fusion)
        self.traj = []            # where the car has been (every 0.5 m), in the pipeline's map frame

    def start(self, pose) -> None:
        """The start box is known (as on the car): the SLAM frame starts there."""
        self.reset()
        self._start = tuple(float(v) for v in pose)
        self.slam.set_pose(*pose)

    # ------------------------------------------------------------------ #
    CAM_KEEP = 0.8            # s: camera frames used for the colours of a LiDAR scan
    lidar_time_offset = 0.0   # s added to the scan stamp (a renderer that stamps its scans late or early)

    def set_vehicle(self, vehicle) -> None:
        """Another car (dashboard Vehicle setup): wheel radius for the odometry, outline for the own-car filter."""
        self.veh, self.wheel_r = vehicle, vehicle.wheel_radius
        self.det.set_vehicle(vehicle)

    _ORDER = {"ground_truth": 0, "imu": 1, "vehicle_sensors": 2, "camera_detections": 3, "lidar": 4}

    def handle_all(self, events: list) -> None:
        """The events of one simulation step: motion first, then the camera, then the LiDAR scan that ends now."""
        for ev in sorted((e for e in events if e.get("type") in self._ORDER), key=lambda e: self._ORDER[e["type"]]):
            self.handle(ev)

    def handle(self, ev: dict) -> None:
        kind = ev["type"]
        if kind == "imu":
            self._yaw_rate = float(ev["angular_velocity"][2])
        elif kind == "vehicle_sensors":
            # undriven front wheels (RWD): no drive slip; their speed along the car is wheel speed x cos(steer)
            # the optical ground-speed sensor measures vx and vy over ground; else the undriven front wheels
            # (RWD: no drive slip; speed along the car = wheel speed x cos(steer)) and a kinematic guess for vy
            gs = ev.get("ground_speed")
            if gs is not None:
                self._v, vy = max(float(gs[0]), 0.0), float(gs[1])
            else:
                rpm = ev["wheel_rpm"]
                w = 0.5 * (rpm[0] + rpm[1]) * 2 * math.pi / 60.0
                self._v = max(w * self.wheel_r * math.cos(float(ev.get("steering_angle", 0.0))), 0.0)
                vy = self._yaw_rate * self.veh.lr
            if self.use_slam:
                self.slam.predict(float(ev["stamp"]), self._v, self._yaw_rate, vy)
            self._track_traj()
        elif kind == "ground_truth":
            self._gt = ev
            gx, gy, gyaw = ev["pose"]
            self._gt_hist.append((float(ev["stamp"]), float(gx), float(gy), float(gyaw)))
            if not self.use_slam:
                x, y, yaw = ev["pose"]
                self.slam.x, self.slam.y, self.slam.yaw = float(x), float(y), float(yaw)
                self.slam.t = float(ev["stamp"])
                self.slam.hist.append((float(ev["stamp"]), x, y, yaw))
                self._v = float(ev["twist"][0])
        elif kind == "camera_detections":
            fp = tuple((round(float(v), 4) for d in ev["detections"] for v in d["position"]))
            if fp and fp == self._last_cam_fp:            # a frozen camera: the same detections again
                self.frozen_frames += 1
                return
            self._last_cam_fp = fp
            self._cam = camera_to_base(ev["detections"], self.sens.camera)
            self._cam_t = float(ev["stamp"])
            if self.mode == "fusion":
                # place the camera cones with the pose at the image's own stamp: a late image (latency) still
                # lands in the right place (before 11 Nov 2026 images older than 0.25 s were dropped, and a
                # 0.25 s camera latency left 70 % of the map without colours: the car left the track)
                x, y, yaw = self.slam.pose_at(self._cam_t)
                c, s = math.cos(yaw), math.sin(yaw)
                for d in self._cam:
                    d["wx"], d["wy"] = x + c * d["x"] - s * d["y"], y + s * d["x"] + c * d["y"]
                self._cam_buf.append((self._cam_t, self._cam))
                while self._cam_buf and self._cam_buf[0][0] < self._cam_t - self.CAM_KEEP:
                    self._cam_buf.popleft()
            if self.mode == "camera":
                near = [c for c in self._cam if c["range"] <= 8.0]      # stereo depth error grows with range^2
                self.last_frame = near
                self.frames += 1
                self.slam.update(self._cam_t, near)
        elif kind == "lidar" and self.mode != "camera":
            self.frame_from_cloud(float(ev["stamp"]), ev["cloud"])

    @staticmethod
    def _fingerprint(arr) -> tuple:
        a = np.asarray(arr, float)
        if a.size == 0:
            return (0,)
        flat = a.reshape(-1)
        return (a.shape, float(flat[:: max(1, flat.size // 997)].sum()), float(flat[-1]))

    def frame_from_cloud(self, stamp: float, cloud: dict) -> list[dict]:
        """One LiDAR scan (stamp = scan start) -> fused cones in the car frame, and the map / pose update."""
        det = self.detect_cloud(stamp, cloud)
        return self.last_frame if det is None else self.integrate_cloud(stamp, det)

    def detect_cloud(self, stamp: float, cloud: dict) -> list[dict] | None:
        """The slow half of frame_from_cloud (cone detection in the car frame). It changes no map or pose state,
        so a ROS node can run it while odometry keeps updating the pose (perception_node, 11 Nov 2026: the pose
        stood still for the ~0.2 s of each scan). None: a frozen (repeated) scan, to be skipped."""
        # Watchdog: a hung LiDAR driver repeats its last scan with new stamps. Matching the same cones again would
        # tell SLAM the car stands still (SLAM error 25 m, 4 cones hit with a 2 s freeze, 11 Nov 2026): skip it,
        # odometry carries the pose until real scans come back.
        fp = self._fingerprint(cloud["xyz"])
        if fp == self._last_cloud and fp != (0,):
            self.frozen_frames += 1
            return None
        self._last_cloud = fp
        return self.det.detect(cloud)

    def integrate_cloud(self, stamp: float, cones: list[dict]) -> list[dict]:
        """The fast half: place the detected cones with the pose at their firing time, fuse colours, update SLAM."""
        raw_stamp, stamp = stamp, stamp + self.lidar_time_offset
        if self._v > 3.0 and cones:
            self._diag.append((raw_stamp, [(c["x"], c["y"], c.get("t", 0.0)) for c in cones]))
        if self.mode == "fusion":
            t_end = stamp + self.det.period
            cam = [d for t, frame in reversed(self._cam_buf) if t_end - t < self.CAM_KEEP for d in frame]   # newest first
            for c in cones:                       # each LiDAR cone with the pose at its firing time
                x, y, yaw = self.slam.pose_at(stamp + c.get("t", 0.0))
                cs, sn = math.cos(yaw), math.sin(yaw)
                c["wx"], c["wy"] = x + cs * c["x"] - sn * c["y"], y + sn * c["x"] + cs * c["y"]
            cones = fuse_world(cones, cam)
        self.last_frame = cones
        self.frames += 1
        self.slam.update(stamp + self.det.period, cones, [stamp + c["t"] for c in cones])
        return cones

    def _track_traj(self) -> None:
        x, y, _ = self.slam.pose()
        if not self.traj or math.hypot(x - self.traj[-1][0], y - self.traj[-1][1]) >= 0.5:
            self.traj.append((x, y))

    # ------------------------------------------------------------------ #
    def pose(self) -> tuple[float, float, float]:
        return self.slam.pose()

    def speed(self) -> float:
        return self._v

    def cones(self) -> list[tuple[float, float, int]]:
        return [(c["x"], c["y"], c["color"]) for c in self.slam.cones()]

    def mapped_track(self, start_pose=None, width_hint: float = 3.5):
        """(centre line (n, 2), width) of the closed circuit from the cone map and the driven lap: the input for
        driving the following laps on a planned line (control.reference_path) instead of following cones."""
        cs = self.slam.cones()
        blue = np.array([[c["x"], c["y"]] for c in cs if c["color"] == 0]).reshape(-1, 2)
        yellow = np.array([[c["x"], c["y"]] for c in cs if c["color"] == 1]).reshape(-1, 2)
        return track_from_map(np.array(self.traj), blue, yellow)

    def lidar_time_alignment(self, truth_cones, offsets=None) -> dict | None:
        """Which time shift of the LiDAR scans fits the ground truth best (needs the ground-truth pose and cones).

        Each recent cone detection is placed with the true pose at (scan stamp + its firing time + shift); the
        shift with the smallest distance to the true cones is the timing error of the LiDAR stamps. A correct
        setup gives ~0. In Isaac Sim this measures how the RTX LiDAR's scans are stamped (11 Nov 2026: cones
        placed 0.3 m off while driving, 0.02 m at standstill). Set the result as lidar_time_offset."""
        if len(self._diag) < 10 or len(self._gt_hist) < 20 or not truth_cones:
            return None
        T = np.array([[c[0], c[1]] for c in truth_cones], float)
        h = list(self._gt_hist)
        ht = np.array([r[0] for r in h])

        def pose(t):
            i = int(np.clip(np.searchsorted(ht, t), 1, len(h) - 1))
            a, b = h[i - 1], h[i]
            f = float(np.clip((t - a[0]) / max(b[0] - a[0], 1e-9), 0.0, 1.0))
            dyaw = (b[3] - a[3] + math.pi) % (2 * math.pi) - math.pi
            return a[1] + f * (b[1] - a[1]), a[2] + f * (b[2] - a[2]), a[3] + f * dyaw

        offs = np.round(np.arange(-0.2, 0.2001, 0.01), 3) if offsets is None else np.asarray(offsets, float)
        rms = []
        diag = list(self._diag)                   # a copy: the ROS node may append from another thread
        for o in offs:
            err = []
            for stamp, cs in diag:
                if stamp - 0.25 < ht[0] or stamp + 0.35 > ht[-1]:
                    continue                      # outside the ground-truth history: the pose would be clamped
                for x, y, t in cs:
                    px, py, yaw = pose(stamp + t + o)
                    c, sn = math.cos(yaw), math.sin(yaw)
                    wx, wy = px + c * x - sn * y, py + sn * x + c * y
                    d = float(np.min(np.hypot(T[:, 0] - wx, T[:, 1] - wy)))
                    if d < 1.5:
                        err.append(min(d, 1.0))
            rms.append(math.sqrt(sum(e * e for e in err) / len(err)) if err else float("inf"))
        k = int(np.argmin(rms))
        k0 = int(np.argmin(np.abs(offs - self.lidar_time_offset)))
        out = {"best_offset": float(offs[k]), "rms_best": round(rms[k], 3), "rms_now": round(rms[k0], 3),
               "now": self.lidar_time_offset}
        # what is left at the best shift, in the car frame: a constant offset points at the LiDAR mounting (x, y),
        # a sideways error growing with range at its yaw, an error along the ray at the cone-centre estimate
        res = []
        o = offs[k]
        for stamp, cs in diag:
            if stamp - 0.25 < ht[0] or stamp + 0.35 > ht[-1]:
                continue
            for x, y, t in cs:
                px, py, yaw = pose(stamp + t + o)
                c, sn = math.cos(yaw), math.sin(yaw)
                wx, wy = px + c * x - sn * y, py + sn * x + c * y
                j = int(np.argmin(np.hypot(T[:, 0] - wx, T[:, 1] - wy)))
                ex, ey = T[j, 0] - wx, T[j, 1] - wy
                if math.hypot(ex, ey) < 0.7:
                    rng = math.hypot(x, y)
                    res.append((c * ex + sn * ey, -sn * ex + c * ey, rng,
                                ((c * ex + sn * ey) * x + (-sn * ex + c * ey) * y) / max(rng, 1e-6)))
        if len(res) >= 20:
            r = np.array(res)
            yaw_fit = float(np.sum(r[:, 1] * r[:, 2]) / np.sum(r[:, 2] ** 2))   # lateral error ~ range x yaw
            out["bias"] = {"x": round(float(np.mean(r[:, 0])), 3), "y": round(float(np.mean(r[:, 1])), 3),
                           "along_ray": round(float(np.mean(r[:, 3])), 3), "yaw_deg": round(math.degrees(yaw_fit), 2),
                           "n": len(res)}
        return out

    def accuracy(self, truth_cones, gate: float = 0.7) -> dict:
        """Map against the ground truth: truth_cones = [(x, y, colour)]. Position errors of matched cones,
        cones of the truth that were mapped, mapped cones that do not exist, colour agreement."""
        import numpy as np

        est = self.slam.cones()
        if not est or not truth_cones:
            return {"mapped": len(est), "truth": len(truth_cones), "matched": 0}
        E = np.array([[c["x"], c["y"]] for c in est])
        T = np.array([[c[0], c[1]] for c in truth_cones])
        d = np.linalg.norm(E[:, None] - T[None], axis=2)
        j = d.argmin(axis=1)
        ok = d[np.arange(len(E)), j] < gate
        err = d[np.arange(len(E)), j][ok]
        col = [est[i]["color"] == truth_cones[j[i]][2] for i in np.nonzero(ok)[0]]
        return {"mapped": len(est), "truth": len(truth_cones), "matched": int(ok.sum()),
                "false": int((~ok).sum()), "seen_share": float(len(set(j[ok].tolist())) / len(T)),
                "pos_rms": float(np.sqrt(np.mean(err ** 2))) if len(err) else float("nan"),
                "pos_max": float(err.max()) if len(err) else float("nan"),
                "colour_ok": float(np.mean(col)) if col else float("nan")}


def _dist_to_polyline(p: "np.ndarray", poly: "np.ndarray"):
    """Distance from points p (n, 2) to the closed polyline poly (m, 2)."""
    import numpy as np

    a, b = poly, np.roll(poly, -1, axis=0)
    ab = b - a
    t = np.clip(np.einsum("nmk,mk->nm", p[:, None, :] - a[None], ab) / np.maximum(np.einsum("mk,mk->m", ab, ab), 1e-12), 0, 1)
    proj = a[None] + t[..., None] * ab[None]
    return np.linalg.norm(p[:, None, :] - proj, axis=2).min(axis=1)


def recentre(path, blue, yellow, iters: int = 4):
    """Move a rough centre line (closed, from cone midpoints) to the middle between the two cone borders.

    Cone-pair midpoints lean towards the inside of corners (the nearest yellow cone of a blue one is not opposite
    it), by up to 0.4 m on a 3.5 m track: too much for a racing line that plans to within 0.4 m of the cones.
    The borders are the cones ordered along the path; every path point is shifted along its normal by half the
    difference of its distances to them, then the line is smoothed. Returns (path, width)."""
    import numpy as np

    from ..geometry import resample_polyline

    path = np.asarray(path, float)
    if np.linalg.norm(path[0] - path[-1]) < 1e-6:
        path = path[:-1]

    def border(c):
        idx = np.array([int(np.argmin(np.linalg.norm(path - q, axis=1))) for q in c])
        return c[np.argsort(idx)]

    B, Y = border(np.asarray(blue, float)), border(np.asarray(yellow, float))
    width = 0.0
    for _ in range(iters):
        t = np.roll(path, -1, 0) - np.roll(path, 1, 0)
        t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
        n = np.stack([-t[:, 1], t[:, 0]], 1)                       # left = blue side
        db, dy = _dist_to_polyline(path, B), _dist_to_polyline(path, Y)
        width = float(np.median(db + dy))
        path = path + n * (0.5 * (db - dy))[:, None]
        k = 7                                                       # smooth over ~3.5 m (0.5 m spacing)
        pad = np.vstack([path[-k:], path, path[:k]])
        ker = np.ones(k) / k
        path = np.stack([np.convolve(pad[:, 0], ker, "same"), np.convolve(pad[:, 1], ker, "same")], 1)[k:-k]
        path = resample_polyline(np.vstack([path, path[:1]]), 0.5, True)
        if np.linalg.norm(path[0] - path[-1]) < 1e-6:
            path = path[:-1]
    return np.vstack([path, path[:1]]), width


def last_lap(traj: np.ndarray, close: float = 4.0, away: float = 25.0) -> np.ndarray:
    """The last closed lap of a driven trajectory (n, 2): back from the newest point until the path returns to it."""
    traj = np.asarray(traj, float).reshape(-1, 2)
    if len(traj) < 20:
        raise ValueError("no lap driven yet")
    d = np.linalg.norm(traj - traj[-1], axis=1)
    far = np.nonzero(d > away)[0]
    if len(far) == 0:
        raise ValueError("no lap driven yet")
    back = np.nonzero(d[: far[-1]] < close)[0]          # before the last time the car was far away
    if len(back) == 0:
        raise ValueError("the driven path does not close")
    # the first point of that return (walking backwards from the far point) closest to the newest point
    seg = back[back >= (np.nonzero(d[: back[-1]] > away)[0][-1] if np.any(d[: back[-1]] > away) else 0)]
    i0 = int(seg[np.argmin(d[seg])])
    return traj[i0:]


def track_from_map(traj, blue, yellow):
    """(centre line, width) of a circuit from the path driven on the exploration lap and the mapped border cones.

    The driven lap is a closed line inside the track; recentre() moves it to the middle between the borders. This
    does not depend on chaining cones (which breaks at the start gate, where big orange cones replace the border).
    Raises ValueError when the result is not a consistent track (then keep following cones)."""
    from ..geometry import resample_polyline

    blue, yellow = np.asarray(blue, float).reshape(-1, 2), np.asarray(yellow, float).reshape(-1, 2)
    if len(blue) < 4 or len(yellow) < 4:
        raise ValueError("not enough blue / yellow cones on the map")
    loop = resample_polyline(last_lap(traj), 0.5, True)
    path, width = recentre(loop, blue, yellow)
    d = np.array([np.min(np.linalg.norm(path - q, axis=1)) for q in np.vstack([blue, yellow])])
    if d.max() > 0.5 * width + 0.6 or not 2.5 <= width <= 6.0:
        raise ValueError(f"mapped track not consistent (a border cone is {d.max():.1f} m from the line, width {width:.1f} m)")
    return path, width
