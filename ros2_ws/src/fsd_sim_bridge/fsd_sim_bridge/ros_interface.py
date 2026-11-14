"""ROS 2 side of the simulator: converts fsdsim events into messages and
exposes the control / AS interfaces. Shared by the lite backend node and the
Isaac Sim relay node so both backends present exactly the same topic contract.

Topic contract (defaults; sensor topics come from sensors.yaml):

    /clock                          rosgraph_msgs/Clock          (lite backend; Isaac publishes its own)
    /lidar/points                   sensor_msgs/PointCloud2      ouster-ros field layout
    /zed/detections                 fsd_sim_msgs/ConeArray       lite backend camera stand-in
    /zed/zed_node/left|right/image_rect_color, .../camera_info, /zed/zed_node/depth/depth_registered
                                    sensor_msgs/Image, CameraInfo (Isaac backend)
    /imu/data                       sensor_msgs/Imu
    /gnss/fix, /gnss/vel            sensor_msgs/NavSatFix, geometry_msgs/TwistWithCovarianceStamped (ENU)
    /vehicle/sensors                fsd_sim_msgs/VehicleSensors  wheel rpm, steering, ground speed
    /as/status                      fsd_sim_msgs/ASStatus
    /ground_truth/odom              nav_msgs/Odometry            map -> base_link
    /ground_truth/cones             fsd_sim_msgs/ConeArray       map frame
    /ground_truth/centerline        nav_msgs/Path                latched
    /ground_truth/markers           visualization_msgs/MarkerArray

    /control/command   (sub)        ackermann_msgs/AckermannDriveStamped
                                    drive.steering_angle [rad], drive.acceleration [m/s^2]
    /as/select_mission (srv)        fsd_sim_msgs/SelectMission
    /as/activate, /as/go, /as/emergency, /sim/reset (srv)   std_srvs/Trigger
"""

from __future__ import annotations

import math
import time

import numpy as np
from ackermann_msgs.msg import AckermannDriveStamped
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Point, PoseStamped, TransformStamped, TwistWithCovarianceStamped
from nav_msgs.msg import Odometry, Path
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CameraInfo, Image, Imu, NavSatFix, NavSatStatus, PointCloud2, PointField
from std_msgs.msg import ColorRGBA, Header, String
from std_srvs.srv import Trigger
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

from fsd_sim_msgs.msg import ASStatus, Cone, ConeArray, VehicleSensors
from fsd_sim_msgs.srv import SelectMission
from fsdsim.config import Mount
from fsdsim.geometry import quat_from_rpy
from fsdsim.timeline import MonotonicTimeline

CONE_RGBA = {
    0: (0.10, 0.35, 0.95, 1.0),
    1: (1.00, 0.85, 0.00, 1.0),
    2: (1.00, 0.55, 0.05, 1.0),
    3: (0.95, 0.30, 0.05, 1.0),
    4: (0.60, 0.60, 0.60, 1.0),
}
CONE_SIZE = {0: (0.228, 0.325), 1: (0.228, 0.325), 2: (0.228, 0.325), 3: (0.285, 0.505), 4: (0.228, 0.325)}

CLOUD_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("intensity", "<f4"),
        ("t", "<u4"),
        ("reflectivity", "<u2"),
        ("ring", "<u2"),
        ("range", "<u4"),
    ]
)
CLOUD_FIELDS = [
    PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
    PointField(name="t", offset=16, datatype=PointField.UINT32, count=1),
    PointField(name="reflectivity", offset=20, datatype=PointField.UINT16, count=1),
    PointField(name="ring", offset=22, datatype=PointField.UINT16, count=1),
    PointField(name="range", offset=24, datatype=PointField.UINT32, count=1),
]

RELIABLE = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)
LATCHED = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
)


def to_time(t: float) -> Time:
    sec = int(math.floor(t))
    nsec = int(round((t - sec) * 1e9))
    if nsec >= 1_000_000_000:
        sec, nsec = sec + 1, nsec - 1_000_000_000
    return Time(sec=sec, nanosec=nsec)


def yaw_quat(yaw: float):
    return (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))


def cloud_to_msg(header: Header, cloud: dict) -> PointCloud2:
    n = len(cloud["xyz"])
    arr = np.empty(n, dtype=CLOUD_DTYPE)
    xyz = np.asarray(cloud["xyz"], dtype=np.float32)
    arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    for key in ("intensity", "t", "reflectivity", "ring", "range"):
        arr[key] = cloud[key]
    return PointCloud2(
        header=header,
        height=1,
        width=n,
        fields=CLOUD_FIELDS,
        is_bigendian=False,
        point_step=CLOUD_DTYPE.itemsize,
        row_step=CLOUD_DTYPE.itemsize * n,
        data=arr.tobytes(),
        is_dense=True,
    )


class SimRosInterface:
    """Publishes simulator events and forwards commands to `hooks`.

    `hooks` must provide: on_control(steer, accel, stamp_sec), select_mission(m),
    activate(), go(), emergency(), reset() - the last five return (ok, message).
    Optional: set_vehicle(changes) -> (ok, message), for /sim/vehicle_set.
    """

    def __init__(self, node, sensors, vehicle, hooks, publish_clock: bool, publish_gt: bool, publish_gt_tf: bool):
        self.node = node
        self.sc = sensors
        self.vp = vehicle
        self.hooks = hooks
        self.map_frame = sensors.map_frame
        self.base_frame = sensors.base_frame
        self.publish_gt = publish_gt
        self.publish_gt_tf = publish_gt_tf
        self.last_status: dict = {}

        self.clock_pub = node.create_publisher(Clock, "/clock", 10) if publish_clock else None
        self.lidar_pub = node.create_publisher(PointCloud2, sensors.lidar.topic, 5)
        self.lidar_preview_topic, self.lidar_preview_pub = sensors.lidar.topic.rstrip("/") + "/preview", None
        self.det_pub = node.create_publisher(ConeArray, sensors.camera.detections_topic, RELIABLE)
        self.imu_pub = node.create_publisher(Imu, sensors.imu.topic, 50)
        self.fix_pub = node.create_publisher(NavSatFix, sensors.gnss.topic, 10)
        self.vel_pub = node.create_publisher(TwistWithCovarianceStamped, sensors.gnss.velocity_topic, 10)
        self.veh_pub = node.create_publisher(VehicleSensors, sensors.vehicle_sensors.topic, 20)
        self.status_pub = node.create_publisher(ASStatus, "/as/status", RELIABLE)
        self.odom_pub = node.create_publisher(Odometry, "/ground_truth/odom", 20)
        self.cones_pub = node.create_publisher(ConeArray, "/ground_truth/cones", LATCHED)
        self.path_pub = node.create_publisher(Path, "/ground_truth/centerline", LATCHED)
        self.markers_pub = node.create_publisher(MarkerArray, "/ground_truth/markers", LATCHED)
        # active vehicle (incl. LapSim parameters and run-time changes) as JSON, for the driver and dashboard
        self.vehicle_pub = node.create_publisher(String, "/sim/vehicle", LATCHED)
        self.publish_vehicle(vehicle)
        self.tf = TransformBroadcaster(node)
        self.static_tf = StaticTransformBroadcaster(node)
        self._image_pubs: dict = {}
        self._preview_t: dict = {}
        self._last_clock = None
        # ROS time = sim time + offset, kept monotonic across simulator restarts / resets.
        # Only when we own /clock: with Isaac's ros2 transport, Isaac stamps its own topics.
        self.timeline = MonotonicTimeline(log=lambda m: node.get_logger().warn(m))
        self._rebase = publish_clock
        self.publish_static_tfs()

        # own callback group: with a multi-threaded executor (Isaac relay) commands never wait behind sensor publishing
        from rclpy.callback_groups import MutuallyExclusiveCallbackGroup  # noqa: PLC0415

        self._control_group = MutuallyExclusiveCallbackGroup()
        node.create_subscription(AckermannDriveStamped, "/control/command", self._on_control, 10,
                                 callback_group=self._control_group)
        node.create_service(SelectMission, "/as/select_mission", self._srv_select)
        node.create_service(Trigger, "/as/activate", self._trigger(lambda: hooks.activate()))
        node.create_service(Trigger, "/as/go", self._trigger(lambda: hooks.go()))
        node.create_service(Trigger, "/as/emergency", self._trigger(lambda: hooks.emergency()))
        node.create_service(Trigger, "/sim/reset", self._trigger(lambda: hooks.reset()))
        # new vehicle parameters at run time (dashboard "Vehicle setup"): JSON {"changes": [...]}
        if hasattr(hooks, "set_vehicle"):
            node.create_subscription(String, "/sim/vehicle_set", self._on_vehicle_set, RELIABLE)

        self._track_markers: list[Marker] = []
        self._cone_markers: list[Marker] = []

    # ------------------------------------------------------------------ #
    # Static TF tree from sensors.yaml (single source of truth for mounts)
    # ------------------------------------------------------------------ #
    def static_transforms(self) -> list:
        sc = self.sc
        base = sc.base_frame
        out = []

        def add(parent, child, m: Mount):
            t = TransformStamped()
            t.header.stamp = self.node.get_clock().now().to_msg()
            t.header.frame_id, t.child_frame_id = parent, child
            t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = m.x, m.y, m.z
            q = quat_from_rpy(m.roll, m.pitch, m.yaw)
            t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = q
            out.append(t)

        if sc.lidar.enabled:
            add(base, sc.lidar.frame_id, sc.lidar.mount)
        if sc.imu.enabled:
            add(base, sc.imu.frame_id, sc.imu.mount)
        if sc.gnss.enabled:
            add(base, sc.gnss.frame_id, sc.gnss.mount)
        if sc.camera.enabled:
            p = sc.camera.frame_prefix
            center = f"{p}_camera_center"
            add(base, center, sc.camera.mount)
            half = sc.camera.baseline / 2.0
            for side, y in (("left", half), ("right", -half)):
                frame = f"{p}_{side}_camera_frame"
                add(center, frame, Mount(y=y))
                add(frame, f"{p}_{side}_camera_optical_frame", Mount(roll=-math.pi / 2, yaw=-math.pi / 2))
        return out

    def publish_vehicle(self, vehicle) -> None:
        import json

        from .common import vehicle_summary

        try:
            self.vehicle_pub.publish(String(data=json.dumps(vehicle_summary(vehicle), default=str)))
        except Exception as e:  # noqa: BLE001
            self.node.get_logger().warn(f"could not publish /sim/vehicle: {e}")

    def _on_vehicle_set(self, msg: String) -> None:
        import json

        try:
            req = json.loads(msg.data or "{}")
            changes = [str(c) for c in req.get("changes") or []]
        except (ValueError, AttributeError) as e:
            self.node.get_logger().warn(f"/sim/vehicle_set: invalid JSON ({e})")
            return
        ok, text = self.hooks.set_vehicle(changes)
        log = self.node.get_logger().info if ok else self.node.get_logger().warn
        log(f"vehicle set ({' '.join(changes) or 'LapSim baseline'}): {text}")

    def publish_static_tfs(self) -> None:
        self.static_tf.sendTransform(self.static_transforms())

    # ------------------------------------------------------------------ #
    # Commands in
    # ------------------------------------------------------------------ #
    def _on_control(self, msg: AckermannDriveStamped) -> None:
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.hooks.on_control(float(msg.drive.steering_angle), float(msg.drive.acceleration),
                              self.timeline.to_sim(stamp))

    def _srv_select(self, req, res):
        ok, text = self.hooks.select_mission(int(req.mission))
        res.success, res.message = bool(ok), str(text)
        return res

    @staticmethod
    def _trigger(fn):
        def cb(_req, res):
            ok, text = fn()
            res.success, res.message = bool(ok), str(text)
            return res

        return cb

    # ------------------------------------------------------------------ #
    # Events out
    # ------------------------------------------------------------------ #
    def new_session(self) -> None:
        """A (re)connected simulator starts its clock at 0 again."""
        self.timeline.new_session()

    def begin_frame(self, t: float) -> None:
        """Call once per simulator frame, before handle(): advances the ROS timeline and /clock."""
        ros_t = self.timeline.advance(t) if self._rebase else t
        if self.clock_pub is None:
            return
        if self._last_clock is not None and ros_t <= self._last_clock:
            return  # out-of-order frame: /clock never goes backwards
        self._last_clock = ros_t
        self.clock_pub.publish(Clock(clock=to_time(ros_t)))

    def handle(self, ev: dict) -> None:
        handler = getattr(self, "_ev_" + ev["type"], None)
        if handler is not None:
            handler(ev)

    def _header(self, ev, frame=None) -> Header:
        # never negative: rclcpp throws on negative stamps (e.g. a LiDAR scan stamped at its start, t < 0)
        return Header(stamp=to_time(max(0.0, self.timeline.to_ros(ev["stamp"]))), frame_id=frame or ev.get("frame_id", self.base_frame))

    PREVIEW_POINTS = 6000    # <lidar topic>/preview: x, y, z of at most this many points per scan (the dashboard)

    def _ev_lidar(self, ev):
        h = self._header(ev)
        if self.lidar_pub.get_subscription_count() > 0:      # perception, RViz, a recorder
            self.lidar_pub.publish(cloud_to_msg(h, ev["cloud"]))
        self._publish_cloud_preview(h, ev["cloud"])

    def _publish_cloud_preview(self, h, cloud: dict) -> None:
        """Every scan, thinned to x, y, z of <= 6000 points (~70 kB instead of ~1 MB). With the dashboard on the full
        cloud, Zenoh delivered odometry and /clock to every node up to 0.4 s late (found 11 Nov 2026)."""
        if self.lidar_preview_pub is None:
            self.lidar_preview_pub = self.node.create_publisher(PointCloud2, self.lidar_preview_topic, 2)
        xyz = np.asarray(cloud["xyz"], dtype=np.float32)
        if len(xyz) > self.PREVIEW_POINTS:
            xyz = xyz[np.linspace(0, len(xyz) - 1, self.PREVIEW_POINTS).astype(np.int64)]
        xyz = np.ascontiguousarray(xyz)
        self.lidar_preview_pub.publish(PointCloud2(
            header=h, height=1, width=len(xyz), fields=CLOUD_FIELDS[:3], is_bigendian=False, point_step=12,
            row_step=12 * len(xyz), data=xyz.tobytes(), is_dense=True))

    def _ev_camera_detections(self, ev):
        msg = ConeArray(header=self._header(ev))
        for d in ev["detections"]:
            p = d["position"]
            msg.cones.append(
                Cone(id=int(max(d.get("id", 0), 0)), color=int(d["color"]), position=Point(x=p[0], y=p[1], z=p[2]),
                     confidence=float(d["confidence"]))
            )
        self.det_pub.publish(msg)

    def _ev_image(self, ev):
        """Rendered camera image (Isaac backend over the TCP link)."""
        topic = ev["topic"]
        pub = self._image_pubs.get(topic)
        if pub is None:
            pub = self._image_pubs[topic] = self.node.create_publisher(Image, topic, 5)
        h = self._header(ev)
        data = np.ascontiguousarray(ev["data"])
        if ev.get("depth_mm"):                   # Isaac sends depth as uint16 mm (link bandwidth): back to metres
            data = data.astype(np.float32) * 0.001
            data[data <= 0.0] = np.nan
        height, width = data.shape[:2]
        enc = ev["encoding"]
        step = width * data.dtype.itemsize * (data.shape[2] if data.ndim == 3 else 1)
        # a full image costs the relay ~10-30 ms of Python: only when someone listens (YOLO, RViz, a recorder)
        if pub.get_subscription_count() > 0:
            pub.publish(Image(header=h, height=height, width=width, encoding=enc, is_bigendian=0, step=step,
                              data=data.tobytes()))
        self._publish_preview(topic, h, data, enc)
        info_topic = ev.get("info_topic")
        if info_topic:
            ipub = self._image_pubs.get(info_topic)
            if ipub is None:
                ipub = self._image_pubs[info_topic] = self.node.create_publisher(CameraInfo, info_topic, 5)
            fx, fy, cx, cy = ev["K"]
            tx = -fx * float(ev.get("baseline", 0.0))  # right camera of a stereo pair
            ipub.publish(CameraInfo(
                header=h, height=height, width=width, distortion_model="plumb_bob", d=[0.0] * 5,
                k=[fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0],
                r=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                p=[fx, 0.0, cx, tx, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0],
            ))

    PREVIEW_HZ = 5.0         # small copies of the images for the dashboard: <topic>/preview, <= 320 px wide
    PREVIEW_WIDTH = 320

    def _publish_preview(self, topic: str, h, data, enc: str) -> None:
        """The dashboard watches <topic>/preview (5 Hz, 320 px) instead of the full images: 1280 x 720 colour and
        float depth at 15 Hz are ~100 MB/s through the Zenoh router, which delayed odometry and commands by up to
        2 s in Isaac Sim (the car left the skidpad on ground truth, 11 Nov 2026)."""
        now = time.monotonic()
        last = self._preview_t.get(topic, 0.0)
        if now - last < 1.0 / self.PREVIEW_HZ:
            return
        self._preview_t[topic] = now
        pt = topic.rstrip("/") + "/preview"
        pub = self._image_pubs.get(pt)
        if pub is None:
            pub = self._image_pubs[pt] = self.node.create_publisher(Image, pt, 2)
        k = max(1, int(np.ceil(data.shape[1] / self.PREVIEW_WIDTH)))
        small = np.ascontiguousarray(data[::k, ::k])
        sh, sw = small.shape[:2]
        step = sw * small.dtype.itemsize * (small.shape[2] if small.ndim == 3 else 1)
        pub.publish(Image(header=h, height=sh, width=sw, encoding=enc, is_bigendian=0, step=step, data=small.tobytes()))

    def _ev_imu(self, ev):
        msg = Imu(header=self._header(ev))
        msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z = ev["angular_velocity"]
        a = ev["linear_acceleration"]
        msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z = a
        gv, av = ev["gyro_var"], ev["accel_var"]
        msg.angular_velocity_covariance = [gv, 0.0, 0.0, 0.0, gv, 0.0, 0.0, 0.0, gv]
        msg.linear_acceleration_covariance = [av, 0.0, 0.0, 0.0, av, 0.0, 0.0, 0.0, av]
        if "orientation" in ev:
            q = ev["orientation"]
            msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w = q
            ov = ev["orientation_var"]
            msg.orientation_covariance = [ov, 0.0, 0.0, 0.0, ov, 0.0, 0.0, 0.0, ov]
        else:
            msg.orientation_covariance[0] = -1.0
        self.imu_pub.publish(msg)

    def _ev_gnss(self, ev):
        h = self._header(ev)
        fix = NavSatFix(header=h, latitude=ev["lat"], longitude=ev["lon"], altitude=ev["alt"])
        fix.status.status = NavSatStatus.STATUS_GBAS_FIX
        fix.status.service = NavSatStatus.SERVICE_GPS | NavSatStatus.SERVICE_GALILEO
        pv = ev["pos_var"]
        fix.position_covariance = [pv[0], 0.0, 0.0, 0.0, pv[1], 0.0, 0.0, 0.0, pv[2]]
        fix.position_covariance_type = NavSatFix.COVARIANCE_TYPE_DIAGONAL_KNOWN
        self.fix_pub.publish(fix)
        vel = TwistWithCovarianceStamped(header=h)
        vel.twist.twist.linear.x, vel.twist.twist.linear.y, vel.twist.twist.linear.z = ev["vel_enu"]
        cov = [0.0] * 36
        cov[0] = cov[7] = cov[14] = ev["vel_var"]
        vel.twist.covariance = cov
        self.vel_pub.publish(vel)

    def _ev_vehicle_sensors(self, ev):
        rpm = ev["wheel_rpm"]
        gs = ev.get("ground_speed", [float("nan"), float("nan")])
        self.veh_pub.publish(
            VehicleSensors(
                header=self._header(ev),
                wheel_rpm_fl=rpm[0],
                wheel_rpm_fr=rpm[1],
                wheel_rpm_rl=rpm[2],
                wheel_rpm_rr=rpm[3],
                steering_angle=ev["steering_angle"],
                ground_speed_x=gs[0],
                ground_speed_y=gs[1],
            )
        )

    def _ev_as_status(self, ev):
        self.last_status = ev
        msg = ASStatus(header=self._header(ev, self.base_frame))
        for key in ("state", "mission", "lap_count", "laps_target", "cones_hit", "off_course"):
            setattr(msg, key, int(ev[key]))
        msg.mission_complete = bool(ev["mission_complete"])
        for key in ("mission_time", "last_lap_time", "best_lap_time", "penalty_time"):
            setattr(msg, key, float(ev[key]))
        msg.message = str(ev["message"])
        self.status_pub.publish(msg)

    def _ev_ground_truth(self, ev):
        if not self.publish_gt:
            return
        x, y, yaw = ev["pose"]
        vx, vy, r = ev["twist"]
        roll, pitch, heave = ev.get("attitude") or (0.0, 0.0, 0.0)     # body on its suspension (fsdsim.chassis)
        q = quat_from_rpy(roll, pitch, yaw) if (roll or pitch) else yaw_quat(yaw)
        odom = Odometry(header=self._header(ev, self.map_frame), child_frame_id=self.base_frame)
        odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z = x, y, float(heave)
        odom.pose.pose.orientation.x, odom.pose.pose.orientation.y, odom.pose.pose.orientation.z, odom.pose.pose.orientation.w = q
        odom.twist.twist.linear.x, odom.twist.twist.linear.y = vx, vy
        odom.twist.twist.angular.z = r
        self.odom_pub.publish(odom)
        if self.publish_gt_tf:
            tf = TransformStamped(header=odom.header, child_frame_id=self.base_frame)
            tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = x, y, float(heave)
            tf.transform.rotation = odom.pose.pose.orientation
            self.tf.sendTransform(tf)

    def _ev_cones(self, ev):
        h = self._header(ev, self.map_frame)
        msg = ConeArray(header=h)
        markers = []
        for cid, color, x, y, knocked in ev["cones"]:
            msg.cones.append(Cone(id=int(cid), color=int(color), position=Point(x=x, y=y, z=0.0), confidence=1.0,
                                  knocked=bool(knocked)))
            d, hgt = CONE_SIZE[int(color)]
            m = Marker(header=h, ns="cones", id=int(cid), type=Marker.CYLINDER, action=Marker.ADD)
            m.pose.position.x, m.pose.position.y = x, y
            if knocked:
                m.pose.position.z = d / 2
                m.pose.orientation.x, m.pose.orientation.w = math.sin(math.pi / 4), math.cos(math.pi / 4)
                rgba = (0.35, 0.35, 0.35, 0.8)
            else:
                m.pose.position.z = hgt / 2
                m.pose.orientation.w = 1.0
                rgba = CONE_RGBA[int(color)]
            m.scale.x = m.scale.y = d * 0.8
            m.scale.z = hgt
            m.color = ColorRGBA(r=rgba[0], g=rgba[1], b=rgba[2], a=rgba[3])
            markers.append(m)
        self.cones_pub.publish(msg)
        self._cone_markers = markers
        self._publish_markers()

    def _ev_track(self, ev):
        h = self._header(ev, self.map_frame)
        path = Path(header=h)
        for x, y in ev["centerline"]:
            ps = PoseStamped(header=h)
            ps.pose.position.x, ps.pose.position.y = x, y
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)
        self.path_pub.publish(path)
        lines = Marker(header=h, ns="timing", id=0, type=Marker.LINE_LIST, action=Marker.ADD)
        lines.scale.x = 0.12
        lines.color = ColorRGBA(r=1.0, g=0.1, b=0.3, a=1.0)
        lines.pose.orientation.w = 1.0
        for key in ("timing_line", "finish_line"):
            seg = ev.get(key)
            if seg:
                lines.points += [Point(x=seg[0][0], y=seg[0][1], z=0.01), Point(x=seg[1][0], y=seg[1][1], z=0.01)]
        car = Marker(header=Header(stamp=h.stamp, frame_id=self.base_frame), ns="vehicle", id=0, type=Marker.CUBE,
                     action=Marker.ADD)
        p = self.vp
        length = p.front_bumper + p.rear_bumper
        car.pose.position.x = 0.5 * (p.front_bumper - p.rear_bumper)
        car.pose.position.z = 0.3
        car.pose.orientation.w = 1.0
        car.scale.x, car.scale.y, car.scale.z = length, p.footprint.width, 0.6
        car.color = ColorRGBA(r=0.85, g=0.1, b=0.1, a=0.6)
        car.frame_locked = True
        self._track_markers = [lines, car]
        self._publish_markers()

    def _publish_markers(self):
        self.markers_pub.publish(MarkerArray(markers=self._track_markers + self._cone_markers))
