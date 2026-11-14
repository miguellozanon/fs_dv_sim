"""Perception without ground truth: LiDAR cones (+ camera colours) -> cone map and pose, for the driver.

    ros2 run fsd_sim_bridge perception --ros-args -p mode:=fusion -p slam:=true

    mode    lidar   cones from /lidar/points, colours from the intensity stripes (lite backend / real Ouster)
            fusion  cones from /lidar/points, colours from the camera cones (colour_topic)
            camera  camera cones only
    slam    true: pose from /vehicle/sensors + /imu/data, corrected on the cone map; false: ground-truth pose

Subscribes  /lidar/points, /imu/data, /vehicle/sensors, /as/status, <colour_topic> (fsd_sim_msgs/ConeArray in the
            left camera optical frame: /zed/detections of the lite backend, /perception/frame_cones of the YOLO node),
            /ground_truth/odom and /ground_truth/cones (only for the start pose, the accuracy numbers, and the pose
            when slam is false)
Publishes   /perception/cones   fsd_sim_msgs/ConeArray (map)    the cone map for the driver (cones_topic)
            /slam/odom          nav_msgs/Odometry (map)         the estimated pose and speed (driver: odom_topic)
            /perception/map_stats std_msgs/String (JSON, 1 Hz)    accuracy against the ground truth, for the dashboard
The pipeline is fsdsim.perception (the same code as tools/closed_loop_eval.py --perception ... --slam).
"""

from __future__ import annotations

import heapq
import json
import math
import os
import threading
import time

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from fsd_sim_msgs.msg import ASStatus, Cone, ConeArray, VehicleSensors
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Imu, PointCloud2
from std_msgs.msg import Header, String

from fsdsim.config import load_all_configs
from fsdsim.perception import PerceptionPipeline
from fsdsim.perception.fresh_pose import FreshPose

from .common import vehicle_from_summary
from .ros_interface import LATCHED, RELIABLE
from .common import shutdown_quietly


def cloud_from_msg(msg: PointCloud2) -> dict:
    """PointCloud2 with ouster-ros fields -> the arrays fsdsim.perception uses (xyz, intensity, t [ns], ring)."""
    dt = np.dtype({"names": [f.name for f in msg.fields],
                   "formats": [{1: "i1", 2: "u1", 3: "<i2", 4: "<u2", 5: "<i4", 6: "<u4", 7: "<f4", 8: "<f8"}[f.datatype] for f in msg.fields],
                   "offsets": [f.offset for f in msg.fields], "itemsize": msg.point_step})
    a = np.frombuffer(msg.data, dtype=dt, count=msg.width * msg.height)
    n = len(a)
    out = {"xyz": np.stack([a["x"], a["y"], a["z"]], axis=1).astype(np.float64)}
    names = dt.names
    out["intensity"] = a["intensity"].astype(np.float64) if "intensity" in names else np.zeros(n)
    out["t"] = a["t"].astype(np.float64) if "t" in names else np.zeros(n)
    out["ring"] = a["ring"] if "ring" in names else np.zeros(n, np.uint16)
    ok = np.isfinite(out["xyz"]).all(axis=1)
    return {k: v[ok] for k, v in out.items()} if not ok.all() else out


def stamp_of(msg) -> float:
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


class PerceptionNode(Node):
    def __init__(self):
        super().__init__("fsd_perception")
        self.declare_parameter("mode", "fusion")
        self.declare_parameter("slam", True)
        self.declare_parameter("config_dir", "")
        self.declare_parameter("colour_topic", "/zed/detections")
        self.declare_parameter("cones_topic", "/perception/cones")
        self.declare_parameter("lidar_time_offset", 0.0)        # s added to the LiDAR stamps (see map_stats lidar_timing)
        self.declare_parameter("lidar_colours", True)           # mode lidar: colours from the stripe intensities
        gp = self.get_parameter
        cfg = gp("config_dir").value or os.path.join(get_package_share_directory("fsd_sim_bringup"), "config")
        self.veh, self.sens, _ = load_all_configs(cfg)
        self.mode, self.use_slam = str(gp("mode").value), bool(gp("slam").value)
        self.pipe = PerceptionPipeline(self.veh, self.sens, self.mode, self.use_slam,
                                       lidar_colours=bool(gp("lidar_colours").value))
        self.pipe.lidar_time_offset = float(gp("lidar_time_offset").value)
        self._timing, self._timing_t = None, 0.0
        self.started = False
        self.gt = None                # last ground-truth odometry (start pose, accuracy, pose when slam is off)
        self.truth = []
        self._run_key = None
        self._scan_ms = []
        self._pose_err = []
        self.cones_pub = self.create_publisher(ConeArray, gp("cones_topic").value, LATCHED)
        self.odom_pub = self.create_publisher(Odometry, "/slam/odom", 20)
        self.stats_pub = self.create_publisher(String, "/perception/map_stats", 5)
        # Scans (and the 1 s stats) run on their own thread: detecting the cones of a scan takes 0.1-0.2 s, and
        # while it ran in the one executor thread no odometry was handled, so /slam/odom stood still and the driver
        # steered on a 0.2-0.3 s old pose (Isaac Sim, 11 Nov 2026). self._lock guards the pipeline's map and pose.
        self._lock = threading.Lock()
        self._heap, self._last_odo = [], {}
        self._corr = []
        # odometry alone (wheel speeds + gyro), restarted on the ground truth every second: how far it is off
        # after 1 s tells odometry errors apart from map / correction errors (map_stats odometry_err_1s_m)
        self._dr = FreshPose(self.veh.wheel_radius, self.veh.lr)
        self._dr_err = []
        self._slow = MutuallyExclusiveCallbackGroup()
        self.create_subscription(PointCloud2, "/lidar/points", self._on_cloud, qos_profile_sensor_data,
                                 callback_group=self._slow)
        # bursts: in Isaac ~7 IMU samples arrive per 30 Hz frame; the default depth 5 dropped a third of them
        burst = QoSProfile(depth=50, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(Imu, "/imu/data", self._on_imu, burst)    # every sample: they are used in stamp order
        self.create_subscription(VehicleSensors, "/vehicle/sensors", self._on_wheels, burst)
        self.create_subscription(ConeArray, gp("colour_topic").value, self._on_camera, qos_profile_sensor_data)
        self.create_subscription(Odometry, "/ground_truth/odom", self._on_gt, 20)
        self.create_subscription(ConeArray, "/ground_truth/cones", self._on_truth, LATCHED)
        self.create_subscription(ASStatus, "/as/status", self._on_status, RELIABLE)
        self.create_subscription(String, "/sim/vehicle", self._on_vehicle, LATCHED)
        self.create_timer(1.0, self._stats, callback_group=self._slow)
        self.get_logger().info(f"perception: {self.mode}, {'SLAM pose' if self.use_slam else 'ground-truth pose'}; "
                               f"cones on {gp('cones_topic').value}, pose on /slam/odom")

    # ------------------------------------------------------------------ #
    def _on_vehicle(self, msg: String) -> None:
        try:
            veh = vehicle_from_summary(msg.data)
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"/sim/vehicle not understood: {e}")
            return
        with self._lock:
            self.veh = veh
            self.pipe.set_vehicle(veh)
            self._dr = FreshPose(veh.wheel_radius, veh.lr)

    def _on_status(self, st: ASStatus) -> None:
        key = (st.mission, st.state in (ASStatus.AS_READY, ASStatus.AS_OFF))
        if self._run_key is not None and key != self._run_key and key[1]:
            with self._lock:
                self.started = False             # new run: map and pose start again at the start box
            self.get_logger().info("new run: cone map and pose reset")
        self._run_key = key

    def _on_truth(self, msg: ConeArray) -> None:
        self.truth = [(c.position.x, c.position.y, int(c.color)) for c in msg.cones if not c.knocked]

    def _on_gt(self, msg: Odometry) -> None:
        q, p = msg.pose.pose.orientation, msg.pose.pose.position
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.gt = (stamp_of(msg), p.x, p.y, yaw, msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.angular.z)
        with self._lock:
            if not self.started:
                self.pipe.start((p.x, p.y, yaw))     # the start box is known
                self.started = True
                self._heap, self._last_odo = [], {}
                self._dr.reset()
            # with slam the pipeline keeps the ground truth for the diagnostics only (lidar_time_alignment)
            self.pipe.handle({"type": "ground_truth", "stamp": self.gt[0], "pose": [p.x, p.y, yaw],
                              "twist": list(self.gt[4:7])})
            dr_t = self._dr.anchor_t
            if dr_t is None or self.gt[0] - dr_t >= 1.0:
                q = self._dr.pose_at(self.gt[0]) if dr_t is not None else None
                if q is not None and self.gt[0] - dr_t < 1.5:
                    self._dr_err.append(math.hypot(q[0] - p.x, q[1] - p.y))
                self._dr.anchor(self.gt[0], p.x, p.y, yaw)
            if self.use_slam:
                # SLAM pose at the ground truth's own stamp: compared with the newest wheel-speed pose instead, the
                # 30 Hz ground truth and 100 Hz wheel speeds of Isaac were up to a frame apart, which alone read as
                # 0.3 m of "SLAM error" at 10 m/s (11 Nov 2026)
                sx, sy, _ = self.pipe.slam.pose_at(self.gt[0])
                self._pose_err.append(math.hypot(sx - p.x, sy - p.y))

    # ------------------------------------------------------------------ #
    # Odometry, camera and scans go through the pipeline in stamp order. Isaac's relay publishes a whole render
    # frame at once (~7 IMU, ~3 wheel-speed samples, maybe a scan) and the executor takes one message per topic in
    # turn, so wheel speeds were integrated with yaw rates from other times and scans were applied before the
    # odometry had reached them: the SLAM pose drifted 0.8 m in one autocross lap (0.05 m in stamp order; same
    # drift reproduced in tools/closed_loop_eval.py, 11 Nov 2026).
    HOLD_MAX = 0.1       # s: never wait longer than this for the slower odometry stream (e.g. an IMU dropout)

    def _push(self, stamp: float, kind: str, item) -> None:
        self._seq = getattr(self, "_seq", 0) + 1
        heapq.heappush(self._heap, (stamp, self._seq, kind, item))

    def _flush(self) -> None:
        """Process everything up to the time both odometry streams have reached (call with self._lock held)."""
        last = self._last_odo
        if not last:
            return
        wm = min(last.values()) if len(last) == 2 else max(last.values()) - self.HOLD_MAX
        wm = max(wm, max(last.values()) - self.HOLD_MAX)
        wheel_t = None
        while self._heap and self._heap[0][0] <= wm:
            stamp, _, kind, item = heapq.heappop(self._heap)
            if kind == "imu":
                self.pipe.handle(item)
                self._dr.imu(item["angular_velocity"][2])
            elif kind == "wheels":
                self.pipe.handle(item)
                self._dr.wheels(stamp, item["wheel_rpm"], item["steering_angle"], item.get("ground_speed"))
                wheel_t = stamp
            elif kind == "camera":
                self.pipe.handle(item)
                if self.mode == "camera":
                    self._publish_cones(stamp)
            elif kind == "scan":
                raw, cones = item
                if cones is not None:
                    n0 = self.pipe.slam.corrections
                    self.pipe.integrate_cloud(raw, cones)
                    if self.pipe.slam.corrections != n0:     # how far each scan moves the pose: odometry drift
                        dx, dy, dth = self.pipe.slam.last_correction
                        self._corr.append((math.hypot(dx, dy), abs(dth)))
                self._publish_cones(stamp)
        if wheel_t is not None and self.use_slam:
            self._publish_pose(wheel_t)
            self._pub_lag = max(getattr(self, "_pub_lag", 0.0), self.get_clock().now().nanoseconds * 1e-9 - wheel_t)

    def _on_imu(self, msg: Imu) -> None:
        t = stamp_of(msg)
        with self._lock:
            self._last_odo["imu"] = max(self._last_odo.get("imu", t), t)
            self._push(t, "imu", {"type": "imu", "stamp": t, "angular_velocity": [0.0, 0.0, msg.angular_velocity.z]})
            self._flush()

    def _on_wheels(self, msg: VehicleSensors) -> None:
        if not self.started:
            return
        lag = self.get_clock().now().nanoseconds * 1e-9 - stamp_of(msg)     # how old the wheel speeds are here
        self._wheel_lag = max(getattr(self, "_wheel_lag", 0.0), lag)
        ev = {"type": "vehicle_sensors", "stamp": stamp_of(msg), "steering_angle": msg.steering_angle,
              "wheel_rpm": [msg.wheel_rpm_fl, msg.wheel_rpm_fr, msg.wheel_rpm_rl, msg.wheel_rpm_rr]}
        if math.isfinite(msg.ground_speed_x):
            ev["ground_speed"] = [msg.ground_speed_x, msg.ground_speed_y]
        with self._lock:
            self._last_odo["wheels"] = max(self._last_odo.get("wheels", ev["stamp"]), ev["stamp"])
            self._push(ev["stamp"], "wheels", ev)
            self._flush()

    def _on_camera(self, msg: ConeArray) -> None:
        if not self.started:
            return
        dets = [{"position": [c.position.x, c.position.y, c.position.z], "color": int(c.color), "confidence": c.confidence}
                for c in msg.cones]
        with self._lock:
            self._push(stamp_of(msg), "camera", {"type": "camera_detections", "stamp": stamp_of(msg), "detections": dets})
            self._flush()

    def _on_cloud(self, msg: PointCloud2) -> None:
        if not self.started or self.mode == "camera":
            return
        t0 = time.perf_counter()                 # wall time: the node's clock is the simulator's
        cones = self.pipe.detect_cloud(stamp_of(msg), cloud_from_msg(msg))   # slow part, without the lock
        with self._lock:
            self._scan_ms.append(1000.0 * (time.perf_counter() - t0))
            if self.started:            # applied once the odometry has reached the end of the scan
                self._push(stamp_of(msg) + self.pipe.det.period, "scan", (stamp_of(msg), cones))
                self._flush()

    # ------------------------------------------------------------------ #
    def _publish_cones(self, t: float) -> None:
        msg = ConeArray(header=self._header(t))
        for i, (x, y, col) in enumerate(self.pipe.cones()):
            msg.cones.append(Cone(id=i + 1, color=int(col), position=Point(x=x, y=y, z=0.0), confidence=1.0, knocked=False))
        self.cones_pub.publish(msg)
        if not self.use_slam:
            self._publish_pose(t)

    def _publish_pose(self, t: float) -> None:
        x, y, yaw = self.pipe.pose()
        od = Odometry(header=self._header(t), child_frame_id="base_link")
        od.pose.pose.position.x, od.pose.pose.position.y = float(x), float(y)
        od.pose.pose.orientation.z, od.pose.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        od.twist.twist.linear.x = float(self.pipe.speed())
        od.twist.twist.linear.y = float(getattr(self.pipe.slam, "vy", 0.0))
        od.twist.twist.angular.z = float(self.pipe._yaw_rate)
        self.odom_pub.publish(od)


    def _header(self, t: float) -> Header:
        h = Header(frame_id="map")
        h.stamp.sec = int(t)
        h.stamp.nanosec = int((t - int(t)) * 1e9)
        return h

    def _stats(self) -> None:
        if not self.started:
            return
        with self._lock:
            self._stats_locked()

    def _stats_locked(self) -> None:
        acc = self.pipe.accuracy(self.truth) if self.truth else {"mapped": len(self.pipe.cones())}
        ms = sorted(self._scan_ms)[len(self._scan_ms) // 2] if self._scan_ms else None
        err = self._pose_err
        d = self.pipe.det.last
        out = {"mode": self.mode, "slam": self.use_slam, "map": acc, "scan_ms": None if ms is None else round(ms, 1),
               "pose_err": None if not err else round(float(np.mean(err)), 3),
               "pose_err_max": None if not err else round(float(np.max(err)), 3),
               "corrections": self.pipe.slam.corrections, "frame_cones": len(self.pipe.last_frame),
               "own_car_points": d.get("own_car"), "ground_pitch_deg": d.get("ground_pitch_deg"),
               "ground_roll_deg": d.get("ground_roll_deg"),
               "wheel_lag_ms": round(1000 * getattr(self, "_wheel_lag", 0.0)),       # wheel speeds on arrival
               "pose_lag_ms": round(1000 * getattr(self, "_pub_lag", 0.0)),          # /slam/odom when published
               "correction_max_m": round(max((c[0] for c in self._corr), default=0.0), 3),   # per scan, last second
               "correction_max_deg": round(math.degrees(max((c[1] for c in self._corr), default=0.0)), 2),
               "odometry_err_1s_m": round(max(self._dr_err), 3) if self._dr_err else None}
        self._dr_err = []
        self._corr = []
        self._wheel_lag = self._pub_lag = 0.0
        self._scan_ms, self._pose_err = [], []
        if self.truth and time.monotonic() - self._timing_t > 5.0:       # costs ~0.1 s: every 5 s, without the lock
            self._timing_t = time.monotonic()
            self._lock.release()                 # it works on copies of the histories
            try:
                self._timing = self.pipe.lidar_time_alignment(self.truth) or self._timing
            finally:
                self._lock.acquire()
        if self._timing:
            out["lidar_timing"] = self._timing
            out["frozen_frames"] = self.pipe.frozen_frames
        out["map"] = {k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in acc.items()}   # no NaN in JSON
        self.stats_pub.publish(String(data=json.dumps(out, separators=(",", ":"), default=float)))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PerceptionNode()
    ex = MultiThreadedExecutor(num_threads=2)     # odometry / camera on one thread, scans + stats on the other
    ex.add_node(node)
    try:
        ex.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass  # Ctrl+C / launch shutdown
    finally:
        node.destroy_node()
        shutdown_quietly()  # the SIGINT handler may already have shut the context down


if __name__ == "__main__":
    main()
