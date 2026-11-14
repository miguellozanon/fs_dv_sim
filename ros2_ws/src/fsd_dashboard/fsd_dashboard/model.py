"""Dashboard state: everything the web page shows, independent of ROS.

Adapters feed it:
  * `fsd_dashboard.node`  - ROS 2 topics (Isaac relay, lite simulator, later the real car)
  * `fsd_dashboard.demo`  - fsdsim directly, no ROS needed (offline demos, tests)

The HTTP server asks for `delta(client)` ~10x per second and streams it to the
browser. Every update method is thread-safe.
"""

from __future__ import annotations

import base64
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

STATE_NAMES = {1: "Off", 2: "Ready", 3: "Driving", 4: "Emergency", 5: "Finished"}
MISSION_NAMES = {0: "None", 1: "Acceleration", 2: "Skidpad", 3: "Trackdrive", 4: "EBS test",
                 5: "Inspection", 6: "Autocross"}
MISSION_KEYS = {"acceleration": 1, "skidpad": 2, "trackdrive": 3, "ebs_test": 4, "inspection": 5, "autocross": 6}
MISSION_NAME_KEYS = {v: k for k, v in MISSION_KEYS.items()}

LIDAR_RES = 0.02          # m per int16 step in the LiDAR stream (+-655 m)
LIDAR_MAX_POINTS = 6000
TELEMETRY_SPAN = 60.0     # s of sim time kept for the charts
TELEMETRY_DT = 0.05       # s between telemetry samples (20 Hz)


@dataclass
class Stream:
    """Health of one input stream. Rate is measured in *sim time* (message stamps), so a
    simulator running below real time still reports the rate the autonomy stack sees."""

    key: str
    label: str
    topic: str
    expected_hz: float | None = None
    optional: bool = False           # hidden until the first message (e.g. camera in the lite backend)
    active_states: tuple = ()        # only expected in these AS states (e.g. control commands: Ready, Driving)
    stamps: deque = field(default_factory=lambda: deque(maxlen=400))
    last_wall: float | None = None
    count: int = 0
    delays: deque = field(default_factory=lambda: deque(maxlen=60))   # sim time at arrival - stamp [s]

    def tick(self, stamp: float, wall: float, now: float | None = None) -> None:
        if self.stamps and stamp + 1.0 < self.stamps[-1]:
            self.stamps.clear()      # simulator restarted
            self.delays.clear()
        if now is not None and -0.05 <= now - stamp < 5.0:
            self.delays.append(max(0.0, now - stamp))
        self.stamps.append(stamp)
        self.last_wall = wall
        self.count += 1

    def rate(self) -> float | None:
        if len(self.stamps) < 3:
            return None
        newest = self.stamps[-1]
        recent = [s for s in self.stamps if s >= newest - 2.0]
        span = recent[-1] - recent[0]
        return (len(recent) - 1) / span if span > 1e-6 and len(recent) >= 3 else None

    def delay(self) -> float | None:
        """Median age of the data when it arrived [s]: sensor latency + transport (sim time, 1 clock step resolution)."""
        if len(self.delays) < 5:
            return None
        return sorted(self.delays)[len(self.delays) // 2]

    def status(self, wall: float, as_state: int | None = None) -> tuple[str, str]:
        if self.last_wall is None:
            return "none", "No data"
        idle = bool(self.active_states) and as_state not in self.active_states
        if idle and wall - self.last_wall > 1.5:
            return "none", "Idle"
        if wall - self.last_wall > 1.5:
            return "critical", "Stalled"
        r = self.rate()
        if self.expected_hz and r is not None and r < 0.8 * self.expected_hz:
            return "warning", "Low rate"
        return "good", "OK"

    def view(self, wall: float, as_state: int | None = None) -> dict:
        level, text = self.status(wall, as_state)
        return {
            "key": self.key, "label": self.label, "topic": self.topic,
            "expected": self.expected_hz, "rate": _r(self.rate(), 1),
            "age": None if self.last_wall is None else round(wall - self.last_wall, 2),
            "delay_ms": None if self.delay() is None else round(1000 * self.delay()),
            "level": level, "text": text, "optional": self.optional, "seen": self.count > 0,
        }


def default_streams(camera_hz: float = 15.0, lidar_hz: float = 10.0, imu_hz: float = 200.0, gnss_hz: float = 10.0,
                    wheel_hz: float = 100.0) -> list[Stream]:
    return [
        Stream("lidar", "LiDAR", "/lidar/points", lidar_hz),
        Stream("camera", "ZED colour", "/zed/zed_node/left/image_rect_color", camera_hz, optional=True),
        Stream("depth", "ZED depth", "/zed/zed_node/depth/depth_registered", camera_hz, optional=True),
        Stream("detections", "Cone detections", "/zed/detections", camera_hz, optional=True),
        Stream("imu", "IMU", "/imu/data", imu_hz),
        Stream("gnss", "GNSS", "/gnss/fix", gnss_hz),
        Stream("vehicle", "Wheel speed & steering", "/vehicle/sensors", wheel_hz),
        Stream("status", "AS status", "/as/status", 10.0),
        Stream("ground_truth", "Ground truth", "/ground_truth/odom", None, optional=True),
        Stream("control", "Control command", "/control/command", None, active_states=(2, 3)),
    ]


def _r(v, nd=3):
    return None if v is None or (isinstance(v, float) and not math.isfinite(v)) else round(float(v), nd)


class ClientCursor:
    """What one browser has already received."""

    def __init__(self):
        self.versions: dict[str, int] = {}
        self.tele_seq = -1
        self.event_seq = -1
        self.first = True


class DashboardModel:
    def __init__(self, source: str = "", streams: list[Stream] | None = None):
        self._lock = threading.RLock()
        self.source = source
        self.streams = {s.key: s for s in (streams or default_streams())}
        self.versions = {"track": 0, "cones": 0, "lidar": 0, "trail": 0, "pcones": 0}
        self.perception: dict = {}          # camera cone perception (YOLO): last frame's boxes and stats
        self.pcones: list = []              # the cone map it built
        self.track: dict | None = None
        self.cones: list = []
        self.status: dict = {}
        self.pose: dict | None = None
        self.vehicle: dict = {}
        self.imu: dict = {}
        self.command: dict = {}
        self.sim_t = 0.0
        self.trail: deque = deque(maxlen=6000)
        self.lidar: dict | None = None
        self.images: dict[str, dict] = {}
        self.laps: list[dict] = []
        self._tele: deque = deque(maxlen=int(TELEMETRY_SPAN / TELEMETRY_DT))
        self._tele_seq = 0
        self._next_tele = 0.0
        self._events: deque = deque(maxlen=300)
        self._event_seq = 0
        self._prev_status: dict = {}
        self._stream_levels: dict[str, str] = {}
        self.car = {"front": 1.55, "rear": 1.35, "width": 1.40}   # bumpers from base_link [m]
        self.vehicle_info: dict = {}   # name, model, changes (from /sim/vehicle or the demo)
        self.prediction: dict = {}     # LapSim prediction for the running event (fsd_dashboard.predict)
        self.command_handler = None   # callable(dict) -> (ok, message); set by the adapter
        self.setup = None             # VehicleSetup (Vehicle setup page), set by the adapter
        self.setup_version = 0
        self.on_mission_complete = None   # callable(mission, laps, penalty, mission_time)

    def bump_setup(self) -> None:
        with self._lock:
            self.setup_version += 1

    def setup_state(self) -> dict | None:
        return self.setup.state() if self.setup is not None else None

    def setup_handle(self, req: dict) -> tuple[bool, str]:
        if self.setup is None:
            return False, "the Vehicle setup is not available (lapsim/LR26_LapSim not found)"
        try:
            return self.setup.handle(req)
        except Exception as e:  # noqa: BLE001
            return False, f"{type(e).__name__}: {e}"

    def set_vehicle_info(self, info: dict) -> None:
        with self._lock:
            if info != self.vehicle_info and self.vehicle_info:
                ch = " ".join(info.get("changes") or []) or "none"
                self.log(f"Vehicle parameters changed: {ch}", "info")
            self.vehicle_info = dict(info)

    def set_prediction(self, pred: dict) -> None:
        with self._lock:
            self.prediction = {k: v for k, v in pred.items()}

    def set_car(self, front_bumper: float, rear_bumper: float, width: float) -> None:
        with self._lock:
            self.car = {"front": _r(front_bumper, 3), "rear": _r(rear_bumper, 3), "width": _r(width, 3)}

    # ------------------------------------------------------------------ #
    # inputs
    # ------------------------------------------------------------------ #
    def tick(self, key: str, stamp: float) -> None:
        s = self.streams.get(key)
        if s is not None:
            with self._lock:
                s.tick(stamp, time.monotonic(), self.sim_t if self.sim_t > 0 else None)

    def log(self, text: str, kind: str = "info") -> None:
        with self._lock:
            self._event_seq += 1
            self._events.append({"seq": self._event_seq, "t": _r(self.sim_t, 2), "wall": time.time(),
                                 "kind": kind, "text": text})

    def set_track(self, centerline, closed: bool, name: str = "", mission: str = "", laps: int = 0,
                  timing_line=None, finish_line=None, width: float | None = None) -> None:
        cl = np.asarray(centerline, float).reshape(-1, 2)
        with self._lock:
            self.track = {
                "name": name, "mission": mission, "closed": bool(closed), "laps": int(laps), "width": _r(width, 2),
                "centerline": np.round(cl, 2).tolist(),
                "timing_line": timing_line, "finish_line": finish_line,
            }
            self.versions["track"] += 1

    def set_cones(self, cones) -> None:
        """cones: iterable of (id, color, x, y, knocked); color uses fsd_sim_msgs/Cone codes."""
        out = [[int(c[0]), int(c[1]), round(float(c[2]), 3), round(float(c[3]), 3), bool(c[4])] for c in cones]
        with self._lock:
            if out != self.cones:
                self.cones = out
                self.versions["cones"] += 1

    def set_perception(self, key: str, value: dict) -> None:
        """key 'dets' (boxes of the last frame) or 'stats' (speed, accuracy) from the cone_perception node (YOLO);
        'map' (cone map and pose accuracy) from the fsd_perception node (LiDAR / fusion / camera + SLAM)."""
        with self._lock:
            self.perception[key] = value

    def set_perception_cones(self, cones) -> None:
        out = [[int(c[0]), int(c[1]), round(float(c[2]), 3), round(float(c[3]), 3)] for c in cones]
        with self._lock:
            if out != self.pcones:
                self.pcones = out
                self.versions["pcones"] += 1

    def set_status(self, st: dict) -> None:
        with self._lock:
            prev = self._prev_status
            if prev and st.get("mission_time", 0) + 1.0 < prev.get("mission_time", 0) and st.get("lap_count", 0) == 0:
                self.laps = []
            state, pstate = st.get("state"), prev.get("state")
            if state != pstate:
                self.log(st.get("message") or f"AS {STATE_NAMES.get(state, state)}",
                         "critical" if state == 4 else "state")
                if state in (1, 2) and pstate in (3, 4, 5):
                    self.laps = []
                    self.trail.clear()
                    self.versions["trail"] += 1
            if prev:
                if st.get("lap_count", 0) > prev.get("lap_count", 0):
                    lap_t = float(st.get("last_lap_time", 0.0))
                    self.laps.append({"lap": int(st["lap_count"]), "time": _r(lap_t, 3)})
                    self.log(f"Lap {st['lap_count']} completed in {lap_t:.3f} s", "lap")
                if st.get("cones_hit", 0) > prev.get("cones_hit", 0):
                    self.log(f"Cone hit - {st['cones_hit']} in total (+{st.get('penalty_time', 0):.0f} s penalties)",
                             "warning")
                if st.get("off_course", 0) > prev.get("off_course", 0):
                    self.log(f"Off course - {st['off_course']} in total", "warning")
                if st.get("mission_complete") and not prev.get("mission_complete"):
                    self.log("Mission complete", "lap")
                    cb = self.on_mission_complete
                    if cb is not None:
                        name = MISSION_NAME_KEYS.get(st.get("mission"), "")
                        laps = [x["time"] for x in self.laps]
                        threading.Thread(target=cb, args=(name, laps, float(st.get("penalty_time") or 0.0),
                                                          float(st.get("mission_time") or 0.0)), daemon=True).start()
            self._prev_status = dict(st)
            self.status = {
                **{k: st.get(k) for k in ("state", "mission", "lap_count", "laps_target", "cones_hit", "off_course",
                                          "mission_complete", "message")},
                "state_name": STATE_NAMES.get(st.get("state"), "Unknown"),
                "mission_name": MISSION_NAMES.get(st.get("mission"), "Unknown"),
                **{k: _r(st.get(k), 3) for k in ("mission_time", "last_lap_time", "best_lap_time", "penalty_time")},
            }

    def set_pose(self, t: float, x: float, y: float, yaw: float, vx: float, vy: float, r: float) -> None:
        with self._lock:
            if t + 1.0 < self.sim_t:      # simulator restarted
                self._restart()
            self.sim_t = t
            self.pose = {"x": _r(x), "y": _r(y), "yaw": _r(yaw, 4), "vx": _r(vx), "vy": _r(vy), "r": _r(r, 4)}
            if not self.trail or math.hypot(x - self.trail[-1][0], y - self.trail[-1][1]) > 0.25:
                self.trail.append((round(x, 2), round(y, 2)))
            self._sample(t)

    def set_vehicle(self, t: float, wheel_rpm, steering: float) -> None:
        with self._lock:
            self.vehicle = {"wheel_rpm": [_r(w, 1) for w in wheel_rpm], "steering": _r(steering, 4)}
            if self.pose is None:
                self.sim_t = max(self.sim_t, t)
                self._sample(t)

    def set_imu(self, t: float, ax: float, ay: float, yaw_rate: float) -> None:
        with self._lock:
            # low-pass the IMU a little so the chart shows vehicle motion, not sensor noise
            a = 0.2
            old = self.imu or {"ax": ax, "ay": ay, "r": yaw_rate}
            self.imu = {"ax": old["ax"] + a * (ax - old["ax"]), "ay": old["ay"] + a * (ay - old["ay"]),
                        "r": old["r"] + a * (yaw_rate - old["r"])}

    def set_command(self, t: float, steer: float, accel: float) -> None:
        with self._lock:
            self.command = {"steer": _r(steer, 4), "accel": _r(accel, 3), "t": t}

    def set_lidar(self, t: float, xyz_map: np.ndarray) -> None:
        """Points already transformed into the map frame (N x 3)."""
        pts = np.asarray(xyz_map, np.float32).reshape(-1, 3)
        pts = pts[np.isfinite(pts).all(axis=1)]
        if len(pts) > LIDAR_MAX_POINTS:
            pts = pts[np.linspace(0, len(pts) - 1, LIDAR_MAX_POINTS).astype(np.int64)]
        q = np.clip(np.round(pts / LIDAR_RES), -32767, 32767).astype("<i2")
        with self._lock:
            self.lidar = {"t": _r(t, 3), "n": int(len(pts)), "res": LIDAR_RES,
                          "data": base64.b64encode(q.tobytes()).decode("ascii")}
            self.versions["lidar"] += 1

    def set_image(self, name: str, data: bytes, mime: str, width: int, height: int, stamp: float) -> None:
        with self._lock:
            seq = self.images.get(name, {}).get("seq", 0) + 1
            self.images[name] = {"data": data, "mime": mime, "w": width, "h": height, "t": stamp, "seq": seq}

    def image(self, name: str) -> dict | None:
        with self._lock:
            return self.images.get(name)

    # ------------------------------------------------------------------ #
    def _restart(self) -> None:
        self.trail.clear()
        self.versions["trail"] += 1
        self._tele.clear()
        self._next_tele = 0.0
        self.laps = []
        self.log("Simulation restarted", "state")

    def _sample(self, t: float) -> None:
        if t + 1e-9 < self._next_tele:
            return
        self._next_tele = t + TELEMETRY_DT
        p, v, c, i = self.pose or {}, self.vehicle, self.command, self.imu
        speed = math.hypot(p.get("vx") or 0.0, p.get("vy") or 0.0) if p else None
        if speed is None and v.get("wheel_rpm"):
            rpm = v["wheel_rpm"]
            speed = sum(rpm[2:]) / 2 * 2 * math.pi / 60 * 0.2   # rough fallback without ground truth
        self._tele_seq += 1
        self._tele.append([
            self._tele_seq, _r(t, 3), _r(speed, 3), v.get("steering"), c.get("steer"), c.get("accel"),
            _r(i.get("ax"), 3), _r(i.get("ay"), 3),
        ])

    # ------------------------------------------------------------------ #
    # output
    # ------------------------------------------------------------------ #
    def handle_command(self, cmd: dict) -> tuple[bool, str]:
        handler = self.command_handler
        if handler is None:
            return False, "commands are not available in this mode"
        try:
            ok, msg = handler(cmd)
        except Exception as e:  # noqa: BLE001
            ok, msg = False, f"{type(e).__name__}: {e}"
        label = {"select_mission": f"Select mission {cmd.get('mission', '')}", "activate": "Activate", "go": "Go",
                 "emergency": "Emergency stop", "reset": "Reset"}.get(cmd.get("cmd"), str(cmd.get("cmd")))
        self.log(f"{label}: {msg}" if msg else label, "command" if ok else "warning")
        return ok, msg

    def delta(self, cur: ClientCursor) -> dict:
        wall = time.monotonic()
        with self._lock:
            out: dict = {"source": self.source, "t": _r(self.sim_t, 2), "status": self.status, "pose": self.pose,
                         "car": self.car, "vehicle_info": self.vehicle_info, "prediction": self.prediction,
                         "vehicle": self.vehicle, "command": self.command,
                         "imu": {k: _r(v, 3) for k, v in self.imu.items()},
                         "laps": self.laps, "setup_version": self.setup_version if self.setup is not None else None}
            if cur.first:
                out["reset"] = True
                cur.first = False
            for key in ("track", "cones", "trail"):
                if cur.versions.get(key) != self.versions[key]:
                    cur.versions[key] = self.versions[key]
                    out[key] = (self.track if key == "track" else self.cones if key == "cones" else list(self.trail))
            if self.trail and "trail" not in out:
                out["trail_tail"] = list(self.trail)[-8:]
            if self.lidar and cur.versions.get("lidar") != self.versions["lidar"]:
                cur.versions["lidar"] = self.versions["lidar"]
                out["lidar"] = self.lidar
            new_tele = [row for row in self._tele if row[0] > cur.tele_seq]
            if new_tele:
                cur.tele_seq = new_tele[-1][0]
                out["telemetry"] = [row[1:] for row in new_tele]
            new_ev = [e for e in self._events if e["seq"] > cur.event_seq]
            if new_ev:
                cur.event_seq = new_ev[-1]["seq"]
                out["events"] = new_ev
            out["images"] = {k: {"seq": v["seq"], "w": v["w"], "h": v["h"], "t": _r(v["t"], 3)} for k, v in self.images.items()}
            if self.perception:
                out["perception"] = self.perception
            if cur.versions.get("pcones") != self.versions["pcones"]:
                cur.versions["pcones"] = self.versions["pcones"]
                out["pcones"] = self.pcones
            out["health"] = [s.view(wall, self.status.get("state")) for s in self.streams.values()]
            self._watch_health(out["health"])
            return out

    def _watch_health(self, views: list[dict]) -> None:
        for v in views:
            prev = self._stream_levels.get(v["key"])
            lvl = v["level"]
            if prev is not None and prev != lvl:
                if lvl == "critical":
                    self.log(f"{v['label']} stalled - no data for {v['age']:.1f} s", "warning")
                elif prev == "critical" and lvl in ("good", "warning"):
                    self.log(f"{v['label']} is back", "info")
            self._stream_levels[v["key"]] = lvl

    def snapshot(self) -> dict:
        """Full state as JSON-able dict (GET /api/state) - for scripts and future tools."""
        cur = ClientCursor()
        d = self.delta(cur)
        d.pop("lidar", None)
        return d
