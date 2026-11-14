"""ROS 2 dashboard node: subscribes to the simulator / car topics and serves the web dashboard.

    ros2 run fsd_dashboard dashboard                      # http://localhost:8080
    ros2 launch fsd_sim_bringup isaac.launch.py           # starts it too (dashboard:=false to skip)

It only uses the standard topic contract (docs/ARCHITECTURE.md), so the same
dashboard works with the lite simulator, Isaac Sim and later the real car. Ground
truth is optional: without /ground_truth/odom the car pose comes from TF
(map -> base_link, e.g. from SLAM).
"""

from __future__ import annotations

import json

import math
import threading
import time

import numpy as np
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.clock import Clock, ClockType
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time

from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import CameraInfo, Image, Imu, NavSatFix, PointCloud2
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener
from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray

from fsd_sim_msgs.msg import ASStatus, ConeArray, VehicleSensors
from fsd_sim_msgs.srv import SelectMission

from fsdsim.config import load_all_configs, vehicle_from_dict
from fsdsim.mission import NAME_BY_MISSION

from . import imaging
from .model import MISSION_KEYS, DashboardModel, default_streams
from .predict import predict
from .server import DashboardServer
from .vehicle_setup import VehicleSetup, decode_start_setup
from fsd_sim_bridge.common import shutdown_quietly

LATCHED = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                     history=HistoryPolicy.KEEP_LAST)
RELIABLE = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)


def _stamp(h) -> float:
    return h.stamp.sec + h.stamp.nanosec * 1e-9


def _yaw(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def _quat_matrix(q) -> np.ndarray:
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def cloud_xyz(msg: PointCloud2) -> np.ndarray:
    """x, y, z float32 fields of a PointCloud2 -> N x 3 array (no sensor_msgs_py needed)."""
    offs = {f.name: f.offset for f in msg.fields}
    if not all(k in offs for k in ("x", "y", "z")):
        return np.zeros((0, 3), np.float32)
    n = msg.width * msg.height
    dtype = np.dtype({"names": ["x", "y", "z"], "formats": ["<f4"] * 3,
                      "offsets": [offs["x"], offs["y"], offs["z"]], "itemsize": msg.point_step})
    arr = np.frombuffer(bytes(msg.data), dtype=dtype, count=n)
    return np.stack([arr["x"], arr["y"], arr["z"]], axis=1)


class DashboardNode(Node):
    def __init__(self):
        super().__init__("fsd_dashboard")
        from fsd_sim_bridge.common import default_config_dir  # noqa: PLC0415

        self.declare_parameter("host", "0.0.0.0")
        self.declare_parameter("port", 8080)
        self.declare_parameter("config_dir", default_config_dir())
        self.declare_parameter("image_hz", 8.0)       # max camera frames/s encoded for the page
        self.declare_parameter("lidar_hz", 10.0)      # max LiDAR clouds/s sent to the page
        self.declare_parameter("source", "ROS 2")
        self.declare_parameter("start_setup", "")     # the setup the launcher started the simulation with
        self.declare_parameter("camera_hz", 0.0)      # > 0: the rate the simulator really renders (Isaac --camera-hz)
        veh, sens, _ = load_all_configs(self.get_parameter("config_dir").value)
        self.sc = sens
        self.map_frame = sens.map_frame
        self.base_frame = sens.base_frame
        ns = "/" + sens.camera.namespace.strip("/")
        self.topics = {
            "lidar": sens.lidar.topic, "imu": sens.imu.topic, "gnss": sens.gnss.topic,
            "vehicle": sens.vehicle_sensors.topic, "detections": sens.camera.detections_topic,
            "camera": f"{ns}/left/image_rect_color", "depth": f"{ns}/depth/depth_registered",
            "status": "/as/status", "ground_truth": "/ground_truth/odom", "control": "/control/command",
        }
        cam_hz = float(self.get_parameter("camera_hz").value) or sens.camera.rate_hz   # was "Low rate" at 5-10 Hz
        streams = default_streams(cam_hz, sens.lidar.rate_hz, sens.imu.rate_hz, sens.gnss.rate_hz,
                                  sens.vehicle_sensors.rate_hz)
        for s in streams:
            s.topic = self.topics.get(s.key, s.topic)
        self.model = DashboardModel(source=str(self.get_parameter("source").value), streams=streams)
        self.model.set_car(veh.front_bumper, veh.rear_bumper, veh.footprint.width)
        self.model.command_handler = self._command

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self._last_odom_wall = 0.0
        self._tf_pose = None
        self._last_img = {}
        self._last_lidar = 0.0
        self._image_period = 1.0 / max(float(self.get_parameter("image_hz").value), 0.1)
        self._lidar_period = 1.0 / max(float(self.get_parameter("lidar_hz").value), 0.1)
        self._scan_period = 1.0 / max(sens.lidar.rate_hz, 1.0)
        self._track_meta = {}
        self._centerline = None

        t = self.topics
        cb = ReentrantCallbackGroup()
        sub = lambda typ, topic, fn, qos: self.create_subscription(typ, topic, fn, qos, callback_group=cb)  # noqa: E731
        sub(ASStatus, t["status"], self._on_status, RELIABLE)
        sub(Odometry, t["ground_truth"], self._on_odom, 20)
        # IMU / wheels come in bursts (Isaac: ~7 IMU samples per 30 Hz frame): a depth-5 queue lost a third of them
        burst = QoSProfile(depth=50, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        sub(VehicleSensors, t["vehicle"], self._on_vehicle, burst)
        sub(Imu, t["imu"], self._on_imu, burst)
        sub(NavSatFix, t["gnss"], lambda m: self.model.tick("gnss", _stamp(m.header)), qos_profile_sensor_data)
        # the bridge's thinned copy (x, y, z, <= 6000 points, every scan): the full ~1 MB clouds slowed Zenoh
        # delivery of odometry and /clock to all nodes (11 Nov 2026). Same as the camera previews.
        sub(PointCloud2, t["lidar"] + "/preview", self._on_lidar, qos_profile_sensor_data)
        # images: the small previews of the bridge (5 Hz, 320 px) for the picture; the full-rate camera_info for the
        # stream health. Full images through the Zenoh router delayed the driver's data in Isaac (11 Nov 2026)
        ns_cam = t["camera"].rsplit("/", 1)[0]
        ns_depth = t["depth"].rsplit("/", 1)[0]
        sub(Image, t["camera"] + "/preview", lambda m: self._on_image("left", m, tick=False), qos_profile_sensor_data)
        sub(Image, t["depth"] + "/preview", lambda m: self._on_image("depth", m, tick=False), qos_profile_sensor_data)
        sub(CameraInfo, ns_cam + "/camera_info", lambda m: self.model.tick("camera", _stamp(m.header)), qos_profile_sensor_data)
        sub(CameraInfo, ns_depth + "/camera_info", lambda m: self.model.tick("depth", _stamp(m.header)), qos_profile_sensor_data)
        sub(ConeArray, t["detections"], lambda m: self.model.tick("detections", _stamp(m.header)), qos_profile_sensor_data)
        sub(AckermannDriveStamped, t["control"], self._on_control, 10)
        sub(ConeArray, "/ground_truth/cones", self._on_cones, LATCHED)
        # camera cone perception (YOLO), when it runs (perception:=yolo)
        sub(String, "/perception/detections", lambda m: self._on_perception("dets", m), 5)
        sub(String, "/perception/stats", lambda m: self._on_perception("stats", m), 5)
        # LiDAR / fusion / camera cone map and SLAM pose (perception:=lidar | fusion | camera)
        sub(String, "/perception/map_stats", lambda m: self._on_perception("map", m), 5)
        sub(ConeArray, "/perception/cones", lambda m: self.model.set_perception_cones(
            [(c.id, c.color, c.position.x, c.position.y) for c in m.cones]), LATCHED)
        sub(Path, "/ground_truth/centerline", self._on_centerline, LATCHED)
        sub(MarkerArray, "/ground_truth/markers", self._on_markers, LATCHED)
        sub(String, "/sim/vehicle", self._on_vehicle_msg, LATCHED)
        self._perf = veh.perf() if veh.model == "performance" else None
        self._pred_key = None
        self._pred_lock = threading.Lock()

        self._srv_select = self.create_client(SelectMission, "/as/select_mission", callback_group=cb)
        self._srv = {name: self.create_client(Trigger, path, callback_group=cb) for name, path in
                     (("activate", "/as/activate"), ("go", "/as/go"), ("emergency", "/as/emergency"),
                      ("reset", "/sim/reset"))}
        # wall-clock timer: pose from TF when there is no ground truth (real car / SLAM)
        self.create_timer(0.05, self._tf_pose_fallback, callback_group=cb, clock=Clock(clock_type=ClockType.STEADY_TIME))

        # Vehicle setup page: LapSim (port) + MATLAB + swap the vehicle in the running simulator
        self._veh_set_pub = self.create_publisher(String, "/sim/vehicle_set", RELIABLE)
        self._drv_set_pub = self.create_publisher(String, "/sim/driver_set", RELIABLE)
        self.setup = None
        try:
            self.setup = VehicleSetup(self.get_parameter("config_dir").value, send_to_sim=self._send_vehicle_set,
                                      on_change=self.model.bump_setup, log=self.model.log,
                                      send_driver_set=self._send_driver_set)
            sub(String, "/sim/driver", self._on_driver_msg, LATCHED)
            self.model.setup = self.setup
            self.model.on_mission_complete = self.setup.on_mission_complete
            start = decode_start_setup(str(self.get_parameter("start_setup").value or ""))
            if start is not None:
                self.setup.set_start(start["name"], start["changes"])
            m = self.setup.matlab
            self.get_logger().info(f"vehicle setup: LapSim {self.setup.lapsim_dir} | MATLAB "
                                   f"{m.exe if m.available else 'not found (set FSG_MATLAB)'}")
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"vehicle setup page disabled: {e}")

        host, port = self.get_parameter("host").value, int(self.get_parameter("port").value)
        self.server = DashboardServer(self.model, host, port).start()
        self.get_logger().info(f"dashboard on http://localhost:{self.server.port} (listening on {host}:{self.server.port}; "
                               f"images encoded with {imaging.encoder_name()})")

    # ------------------------------------------------------------------ #
    def _on_vehicle_msg(self, m: String) -> None:
        import json  # noqa: PLC0415

        try:
            d = json.loads(m.data)
            veh = vehicle_from_dict(d["vehicle"])
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"/sim/vehicle not understood: {e}")
            return
        info = {"name": veh.name, "model": veh.model, "changes": list(veh.performance_changes)}
        self.model.set_vehicle_info(info)
        if self.setup is not None:
            self.setup.on_vehicle(info)
        self.model.set_car(veh.front_bumper, veh.rear_bumper, veh.footprint.width)
        self._perf = veh.perf() if veh.model == "performance" else None
        self._pred_key = None
        self._maybe_predict()

    def _maybe_predict(self, mission_id: int | None = None, laps: int | None = None) -> None:
        """LapSim prediction for the running event; recomputed when vehicle, event or track change."""
        st = self.model.status
        mission = NAME_BY_MISSION.get(mission_id if mission_id is not None else st.get("mission"), "")
        cl = self._centerline
        key = (id(self._perf), mission, len(cl) if cl else 0, laps)
        if self._perf is None or not mission or key == self._pred_key:
            return
        self._pred_key = key
        closed = bool(cl) and math.dist(cl[0], cl[-1]) < 3.0

        def work(perf=self._perf):
            if not self._pred_lock.acquire(blocking=False):
                return
            try:
                self.model.set_prediction(predict(perf, mission, cl, closed, laps or 1))
            except Exception as e:  # noqa: BLE001
                self.get_logger().warn(f"LapSim prediction failed: {e}")
            finally:
                self._pred_lock.release()

        threading.Thread(target=work, daemon=True).start()

    def _on_status(self, m: ASStatus) -> None:
        self._maybe_predict(int(m.mission), int(m.laps_target))
        self.model.tick("status", _stamp(m.header))
        self.model.set_status({
            "state": int(m.state), "mission": int(m.mission), "lap_count": int(m.lap_count),
            "laps_target": int(m.laps_target), "mission_complete": bool(m.mission_complete),
            "cones_hit": int(m.cones_hit), "off_course": int(m.off_course), "mission_time": float(m.mission_time),
            "last_lap_time": float(m.last_lap_time), "best_lap_time": float(m.best_lap_time),
            "penalty_time": float(m.penalty_time), "message": m.message,
        })

    def _on_odom(self, m: Odometry) -> None:
        t = _stamp(m.header)
        self.model.tick("ground_truth", t)
        self._last_odom_wall = time.monotonic()
        p, tw = m.pose.pose, m.twist.twist
        self.model.set_pose(t, p.position.x, p.position.y, _yaw(p.orientation), tw.linear.x, tw.linear.y, tw.angular.z)

    def _tf_pose_fallback(self) -> None:
        if time.monotonic() - self._last_odom_wall < 1.0:
            return
        try:
            tf = self.tf_buffer.lookup_transform(self.map_frame, self.base_frame, Time())
        except Exception:  # noqa: BLE001
            return
        t = _stamp(tf.header)
        tr = tf.transform.translation
        yaw = _yaw(tf.transform.rotation)
        vx = vy = r = 0.0
        if self._tf_pose is not None and t > self._tf_pose[0] + 1e-3:
            t0, x0, y0, yaw0 = self._tf_pose
            dt = t - t0
            gx, gy = (tr.x - x0) / dt, (tr.y - y0) / dt
            c, s = math.cos(yaw), math.sin(yaw)
            vx, vy = c * gx + s * gy, -s * gx + c * gy
            r = math.atan2(math.sin(yaw - yaw0), math.cos(yaw - yaw0)) / dt
        elif self._tf_pose is not None:
            return
        self._tf_pose = (t, tr.x, tr.y, yaw)
        self.model.set_pose(t, tr.x, tr.y, yaw, vx, vy, r)

    def _on_vehicle(self, m: VehicleSensors) -> None:
        t = _stamp(m.header)
        self.model.tick("vehicle", t)
        self.model.set_vehicle(t, [m.wheel_rpm_fl, m.wheel_rpm_fr, m.wheel_rpm_rl, m.wheel_rpm_rr], m.steering_angle)

    def _on_imu(self, m: Imu) -> None:
        t = _stamp(m.header)
        self.model.tick("imu", t)
        a = m.linear_acceleration
        self.model.set_imu(t, a.x, a.y, m.angular_velocity.z)

    def _on_control(self, m: AckermannDriveStamped) -> None:
        t = _stamp(m.header)
        self.model.tick("control", t)
        self.model.set_command(t, m.drive.steering_angle, m.drive.acceleration)

    def _on_lidar(self, m: PointCloud2) -> None:
        t = _stamp(m.header)
        self.model.tick("lidar", t)
        now = time.monotonic()
        if now - self._last_lidar < self._lidar_period:
            return
        self._last_lidar = now
        # the cloud is stamped at scan start; place it with the pose at mid-scan, else the latest pose
        mid = Time(nanoseconds=int((t + 0.5 * self._scan_period) * 1e9))
        try:
            tf = self.tf_buffer.lookup_transform(self.map_frame, m.header.frame_id, mid)
        except Exception:  # noqa: BLE001
            try:
                tf = self.tf_buffer.lookup_transform(self.map_frame, m.header.frame_id, Time())
            except Exception:  # noqa: BLE001
                return
        xyz = cloud_xyz(m)
        if len(xyz) > 12000:                     # thin before transforming; the page shows <= 6000
            xyz = xyz[:: int(np.ceil(len(xyz) / 12000))]
        rot = _quat_matrix(tf.transform.rotation)
        tr = tf.transform.translation
        self.model.set_lidar(t, xyz.astype(np.float64) @ rot.T + np.array([tr.x, tr.y, tr.z]))

    def _on_image(self, name: str, m: Image, tick: bool = True) -> None:
        t = _stamp(m.header)
        if tick:
            self.model.tick("camera" if name == "left" else "depth", t)
        now = time.monotonic()
        if now - self._last_img.get(name, 0.0) < self._image_period:
            return
        self._last_img[name] = now
        conv = imaging.from_ros_image(m.encoding, m.height, m.width, m.step, bytes(m.data))
        if conv is None:
            return
        kind, arr = conv
        img = imaging.colorize_depth(imaging.downscale(arr)) if kind == "depth" else imaging.downscale(arr)
        buf, mime = imaging.encode_rgb(img)
        self.model.set_image(name, buf, mime, img.shape[1], img.shape[0], t)

    def _on_cones(self, m: ConeArray) -> None:
        self.model.set_cones([(c.id, c.color, c.position.x, c.position.y, c.knocked) for c in m.cones])

    def _on_perception(self, key: str, m: String) -> None:
        try:
            self.model.set_perception(key, json.loads(m.data))
        except ValueError:
            pass

    def _on_centerline(self, m: Path) -> None:
        self._centerline = [(p.pose.position.x, p.pose.position.y) for p in m.poses]
        self._push_track()

    def _on_markers(self, m: MarkerArray) -> None:
        for mk in m.markers:
            if mk.ns == "timing" and len(mk.points) >= 2:
                pts = [[p.x, p.y] for p in mk.points]
                self._track_meta["timing_line"] = pts[0:2]
                self._track_meta["finish_line"] = pts[2:4] if len(pts) >= 4 else None
            elif mk.ns == "vehicle":
                self.model.set_car(mk.pose.position.x + mk.scale.x / 2, mk.scale.x / 2 - mk.pose.position.x, mk.scale.y)
        self._push_track()

    def _push_track(self) -> None:
        cl = self._centerline
        if not cl or len(cl) < 2:
            return
        closed = math.dist(cl[0], cl[-1]) < 3.0
        self.model.set_track(cl, closed, timing_line=self._track_meta.get("timing_line"),
                             finish_line=self._track_meta.get("finish_line"))
        if hasattr(self, "_perf"):
            self._maybe_predict(laps=self.model.status.get("laps_target"))

    def _send_vehicle_set(self, changes):
        import json  # noqa: PLC0415

        if self._veh_set_pub.get_subscription_count() == 0:
            return False, "no simulator is listening on /sim/vehicle_set (start the Isaac relay or the lite simulator)"
        self._veh_set_pub.publish(String(data=json.dumps({"changes": list(changes)})))
        return True, "sent to the simulator (the run restarts with the new vehicle)"

    def _send_driver_set(self, req: dict):
        import json  # noqa: PLC0415

        if self._drv_set_pub.get_subscription_count() == 0:
            return False, "no driver node is listening on /sim/driver_set (start the launch with driver:=true)"
        self._drv_set_pub.publish(String(data=json.dumps(req)))
        return True, "sent to the driver node"

    def _on_driver_msg(self, m: String) -> None:
        import json  # noqa: PLC0415

        try:
            info = json.loads(m.data)
        except ValueError:
            return
        if self.setup is not None:
            self.setup.rl.on_driver(info)
        self.model.log(f"Driver: {info.get('description') or info.get('longitudinal')}", "info")

    # ------------------------------------------------------------------ #
    def _command(self, cmd: dict):
        c = cmd["cmd"]
        if c == "select_mission":
            key = str(cmd.get("mission", "")).lower()
            if key not in MISSION_KEYS:
                return False, f"unknown mission '{key}'"
            client, req = self._srv_select, SelectMission.Request(mission=MISSION_KEYS[key])
        else:
            client, req = self._srv[c], Trigger.Request()
        if not client.wait_for_service(timeout_sec=0.5):
            return False, f"service {client.srv_name} is not available (is the simulator running?)"
        done = threading.Event()
        fut = client.call_async(req)
        fut.add_done_callback(lambda _f: done.set())
        if not done.wait(3.0):
            return False, f"{client.srv_name} did not answer within 3 s"
        res = fut.result()
        return bool(res.success), str(res.message)


def main(args=None):
    rclpy.init(args=args)
    node = DashboardNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass  # Ctrl+C / launch shutdown
    finally:
        try:
            node.server.stop()
            node.destroy_node()
        except KeyboardInterrupt:      # a second Ctrl+C while stopping
            pass
        shutdown_quietly()  # the SIGINT handler may already have shut the context down


if __name__ == "__main__":
    main()
