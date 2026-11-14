"""Reference driver node: closes the loop so the simulator can be tested before
your own stack exists. Replace it with your perception / SLAM / planning /
control nodes - they only need the same topics.

Inputs : /ground_truth/odom, /ground_truth/cones (or your perception output),
         /ground_truth/centerline (map mode), /as/status, /imu/data (RL driver),
         /sim/vehicle (active vehicle incl. LapSim parameters: speed targets in map mode),
         /sim/driver_set (std_msgs/String JSON {"longitudinal": "rl"|"reference", "policy": "auto"|"universal"|name})
Output : /control/command (ackermann_msgs/AckermannDriveStamped),
         /sim/driver (latched JSON: which driver / policy is active)

longitudinal:=rl uses a learned throttle / brake policy (fsdsim.rl, tools/train_rl.py) with this
node's steering; rl_policy:=auto picks the policy trained for the running vehicle spec, else a
universal policy (tools/train_rl_universal.py) whose training range contains it; rl_policy:=universal
always takes the universal one.

With a performance (LapSim) vehicle the map-mode speed targets come from the LapSim QSS model
of exactly the car being simulated (incl. run-time parameter changes), at `grip_util` of the
tyre limit (`long_util` extra margin for braking).
"""

from __future__ import annotations

import json
import math
import time

import numpy as np
import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.executors import ExternalShutdownException
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import String

from fsd_sim_msgs.msg import ASStatus, ConeArray, VehicleSensors
from fsdsim.control import DemoDriver, DriverParams, RLDriver, make_vehicle_driver, reference_path
from fsdsim.control.demo_driver import predict_pose
from fsdsim.perception.fresh_pose import FreshPose
from fsdsim.mission import NAME_BY_MISSION

from .common import vehicle_from_summary
from .ros_interface import LATCHED, RELIABLE
from .common import shutdown_quietly

AUTO_MODE = {
    ASStatus.MISSION_ACCELERATION: "map",
    ASStatus.MISSION_EBS_TEST: "map",
    ASStatus.MISSION_SKIDPAD: "map",
    ASStatus.MISSION_AUTOCROSS: "cones",
    ASStatus.MISSION_TRACKDRIVE: "cones",
}


class DemoDriverNode(Node):
    def __init__(self):
        super().__init__("fsd_demo_driver")
        p = DriverParams()
        self.declare_parameter("mode", "auto")  # auto | map | cones
        self.declare_parameter("rate_hz", 50.0)
        # sim: the tick follows /clock (lite, faster than real time); steady: wall clock (Isaac, where a late /clock
        # held the tick back for up to 0.8 s on the team PC, 11 Nov 2026)
        self.declare_parameter("tick_clock", "sim")
        self.declare_parameter("v_max", p.v_max)
        self.declare_parameter("v_max_cones", p.v_max_cones)
        self.declare_parameter("a_lat_max", p.a_lat_max)
        self.declare_parameter("cone_range", p.cone_range)
        self.declare_parameter("track_width", p.track_width_guess)
        self.declare_parameter("wheelbase", p.wheelbase)
        self.declare_parameter("lr", p.lr)
        self.declare_parameter("max_steer", p.max_steer)
        self.declare_parameter("cones_topic", "/ground_truth/cones")
        self.declare_parameter("odom_topic", "/ground_truth/odom")   # /slam/odom: pose from fsd_perception (no ground truth)
        self.declare_parameter("line", "center")              # center | racing (map mode on circuits: perf.raceline)
        self.declare_parameter("race_after", 0)               # cones mode on a circuit: after N laps drive the mapped track
        self.declare_parameter("grip_util", p.grip_util)
        self.declare_parameter("long_util", p.long_util)
        self.declare_parameter("latency", 0.0)    # s: pose age + time until a command acts (isaac.launch.py: 2 frames)
        self.declare_parameter("longitudinal", "reference")   # reference | rl
        self.declare_parameter("rl_policy", "auto")           # auto | universal | <name in rl/policies> | <folder>
        self.declare_parameter("imu_topic", "/imu/data")
        # auto: on with the SLAM pose (/slam/odom): integrate wheel speeds + gyro here and take only the SLAM
        # correction from the perception node, whose pose arrived 0.15-0.27 s late in Isaac (fsdsim.perception.fresh_pose)
        self.declare_parameter("fresh_pose", "auto")
        gp = self.get_parameter
        self.driver_params = DriverParams(
            v_max=float(gp("v_max").value), v_max_cones=float(gp("v_max_cones").value),
            a_lat_max=float(gp("a_lat_max").value), cone_range=float(gp("cone_range").value),
            track_width_guess=float(gp("track_width").value), wheelbase=float(gp("wheelbase").value),
            lr=float(gp("lr").value), max_steer=float(gp("max_steer").value), latency=float(gp("latency").value),
        )
        self.mode_param = gp("mode").value
        self.driver: DemoDriver | None = None
        self.odom: Odometry | None = None
        self.cones: ConeArray | None = None
        self.path: Path | None = None
        self.status: ASStatus | None = None
        self.vehicle = None  # VehicleParams from /sim/vehicle
        self.longitudinal = str(gp("longitudinal").value)
        self.rl_policy = str(gp("rl_policy").value)
        self.imu: Imu | None = None
        self.driver_desc = ""

        # depth 1: the driver only wants the newest pose. With depth 20 a busy executor (one message per subscription
        # per wake-up, IMU at 200 Hz) worked through a full queue and steered on 0.4 s old poses (11 Nov 2026)
        self.create_subscription(Odometry, gp("odom_topic").value, self._on_odom, 1)
        self.line = str(gp("line").value)
        self.race_after = int(gp("race_after").value)
        self._raced = False
        self._traj = []
        self.create_subscription(ConeArray, gp("cones_topic").value, lambda m: setattr(self, "cones", m), LATCHED)
        self.create_subscription(Path, "/ground_truth/centerline", self._on_path, LATCHED)
        self.create_subscription(ASStatus, "/as/status", self._on_status, RELIABLE)
        self.create_subscription(String, "/sim/vehicle", self._on_vehicle, LATCHED)
        self.create_subscription(Imu, gp("imu_topic").value, self._on_imu, 1)
        fp = str(gp("fresh_pose").value).lower()
        self._fresh_on = fp == "true" or (fp == "auto" and gp("odom_topic").value == "/slam/odom")
        self.fresh: FreshPose | None = None
        if self._fresh_on:
            self.create_subscription(VehicleSensors, "/vehicle/sensors", self._on_wheels, 20)
        self.create_subscription(String, "/sim/driver_set", self._on_driver_set, RELIABLE)
        self.driver_pub = self.create_publisher(String, "/sim/driver", LATCHED)
        self.cmd_pub = self.create_publisher(AckermannDriveStamped, "/control/command", 10)
        steady = str(gp("tick_clock").value).lower() == "steady"
        self._steady = steady
        self._odom_wall, self._last_steer, self._rtf = None, 0.0, 1.0
        self.create_timer(1.0 / float(gp("rate_hz").value), self._tick,
                          clock=Clock(clock_type=ClockType.STEADY_TIME) if steady else None)

    def _on_imu(self, msg: Imu):
        self.imu = msg
        if self.fresh is not None:
            self.fresh.imu(msg.angular_velocity.z)

    def _on_wheels(self, msg: VehicleSensors):
        if self.fresh is None:
            if self.vehicle is None:
                return
            self.fresh = FreshPose(self.vehicle.wheel_radius, self.vehicle.lr)
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.fresh.t is not None and t < self.fresh.t - 0.5:
            self.fresh.reset()                              # a new run: the simulator clock restarted
        gs = (msg.ground_speed_x, msg.ground_speed_y) if math.isfinite(msg.ground_speed_x) else None
        self.fresh.wheels(t, [msg.wheel_rpm_fl, msg.wheel_rpm_fr, msg.wheel_rpm_rl, msg.wheel_rpm_rr],
                          msg.steering_angle, gs)

    def _fresh_pose(self, now_sim: float):
        """(pose, v) from the driver's own dead reckoning on the SLAM anchor, or None (not on / no recent anchor)."""
        f = self.fresh
        if f is None or f.anchor_t is None or now_sim - f.anchor_t > 1.0:
            return None
        p = f.pose()
        return None if p is None else (p, f.v)

    def _on_odom(self, msg: Odometry):
        self.odom, self._odom_wall = msg, time.monotonic()
        if self.fresh is not None:
            o = msg.pose.pose
            q = o.orientation
            yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            self.fresh.anchor(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9, o.position.x, o.position.y, yaw)
        age = self.get_clock().now().nanoseconds * 1e-9 - (msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
        self._odom_rx_age = max(getattr(self, "_odom_rx_age", 0.0), age)       # how old odometry is on arrival

    STALE_AFTER = 0.08   # s of wall time without new odometry before the pose is carried forward (Isaac: ~45 ms/frame)

    def _bridge_stale_pose(self, pose, v):
        """Wall-clock ticks only (Isaac): when odometry and /clock stop arriving for a while (seen up to 0.46 s on the
        team PC while Isaac kept driving), carry the pose forward on the last steering arc for the missing time
        (wall time x the measured real-time factor), so the car does not steer on an old position.
        Not for a pose that arrives regularly but old (/slam/odom): carried 0.15-0.25 s forward on the steering arc
        the racing line went off the track (Isaac, and tools/closed_loop_eval.py --cmd-latency 0.2, 11 Nov 2026);
        that pose has to arrive fresh instead."""
        if not self._steady or self._odom_wall is None:
            return pose, 0.0
        stale = time.monotonic() - self._odom_wall
        if stale < self.STALE_AFTER:
            return pose, 0.0
        dt = min(stale - 0.04, 0.5) * self._rtf
        return predict_pose(pose, v, self._last_steer, self.driver_params.wheelbase, dt), dt

    def _on_status(self, msg: ASStatus):
        """A new run (dashboard "Reset simulation", or Ready again after Finished / an emergency) starts with a fresh
        driver: the same track sends the same centre line, so before 13 Nov 2026 the driver kept its path memory,
        its racing state and its stop state from the run before."""
        prev = self.status
        ready = msg.state in (ASStatus.AS_OFF, ASStatus.AS_READY)
        if prev is not None and ready and (prev.state not in (ASStatus.AS_OFF, ASStatus.AS_READY)
                                           or prev.mission != msg.mission):
            self.driver = None
            self._raced = False
            self._race_ready = None
            self._traj = []
            if self.fresh is not None:
                self.fresh.reset()
            self.get_logger().info("new run: driver restarted")
        self.status = msg

    def _on_path(self, msg: Path):
        def key(m):
            if not m.poses:
                return (0,)
            a, b = m.poses[0].pose.position, m.poses[-1].pose.position
            return (len(m.poses), round(a.x, 3), round(a.y, 3), round(b.x, 3), round(b.y, 3))

        if self.path is None or key(msg) != key(self.path):
            self.driver = None  # new track (or sim reset): rebuild with the new reference
            self._raced = False
            self._traj = []
        self.path = msg

    def _on_vehicle(self, msg: String):
        try:
            veh = vehicle_from_summary(msg.data)
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"/sim/vehicle not understood: {e}")
            return
        self.vehicle = veh
        self.driver = None  # rebuild with the new vehicle
        self.fresh = None   # and the dead reckoning with its wheel radius
        extra = f" ({' '.join(veh.performance_changes)})" if veh.performance_changes else ""
        self.get_logger().info(f"vehicle: {veh.name}, model {veh.model}{extra}")

    def _on_driver_set(self, msg: String):
        try:
            req = json.loads(msg.data or "{}")
        except ValueError:
            self.get_logger().warn("/sim/driver_set: invalid JSON")
            return
        lon = str(req.get("longitudinal", self.longitudinal))
        if lon not in ("reference", "rl"):
            self.get_logger().warn(f"/sim/driver_set: unknown longitudinal '{lon}'")
            return
        self.longitudinal = lon
        self.rl_policy = str(req.get("policy", self.rl_policy) or "auto")
        self.driver = None  # rebuilt on the next tick (the run continues)
        self.get_logger().info(f"driver set: longitudinal {lon}, policy {self.rl_policy}")

    def _publish_driver(self):
        self.driver_pub.publish(String(data=json.dumps({
            "longitudinal": "rl" if isinstance(self.driver, RLDriver) else "reference",
            "requested": self.longitudinal, "policy": self.rl_policy, "description": self.driver_desc})))

    def _ensure_driver(self):
        if self.driver is not None or self.status is None:
            return self.driver
        mode = self.mode_param if self.mode_param != "auto" else AUTO_MODE.get(self.status.mission, "cones")
        if self.longitudinal == "rl" and self.mode_param == "auto":
            mode = "map"   # the RL driver follows the known path
        ground_truth = self.get_parameter("cones_topic").value == "/ground_truth/cones"
        if (self.mode_param == "auto" and self.line == "racing" and ground_truth and not self.race_after
                and self.status.mission in (ASStatus.MISSION_AUTOCROSS, ASStatus.MISSION_TRACKDRIVE)):
            mode = "map"   # a racing line needs the known track: drive it from the start (was silently ignored)
        elif self.line == "racing" and mode == "cones" and not self.race_after:
            self.get_logger().warn("line:=racing is used on the mapped track only: add race_after:=1 "
                                   "(with perception the track is not known before the first lap)")
        params = self.driver_params
        if self.status.mission == ASStatus.MISSION_SKIDPAD:
            params.stop_mode = "path_end"
        if self.status.mission in (ASStatus.MISSION_ACCELERATION, ASStatus.MISSION_EBS_TEST):
            params.v_max = max(params.v_max, 26.0)
        if self.vehicle is not None:
            gp = self.get_parameter
            mission = NAME_BY_MISSION.get(self.status.mission, "")
            drv, self.driver_desc = make_vehicle_driver(
                self.vehicle, mission, mode, params.track_width_guess, self.longitudinal, self.rl_policy,
                grip_util=None if (self.line == "racing" and mode == "map") else float(gp("grip_util").value),
                long_util=float(gp("long_util").value), line=self.line if mode == "map" else "center",
                v_max_cones=params.v_max_cones, a_lat_max=params.a_lat_max, cone_range=params.cone_range,
                latency=params.latency)
        else:
            drv = DemoDriver(params, mode)
            self.driver_desc = "reference driver" + (" (RL needs the vehicle from /sim/vehicle)"
                                                     if self.longitudinal == "rl" else "")
        if mode == "map":
            if self.path is None or not self.path.poses:
                return None
            pts = [(ps.pose.position.x, ps.pose.position.y) for ps in self.path.poses]
            closed = self.status.mission in (ASStatus.MISSION_TRACKDRIVE, ASStatus.MISSION_AUTOCROSS)
            if self.vehicle is not None and self.line == "racing" and closed:
                pts, info = reference_path(np.asarray(pts, float), params.track_width_guess, True, self.vehicle, "racing")
                if info:
                    self.get_logger().info(f"racing line: QSS {info['t_center']:.2f} s on the centre line -> "
                                           f"{info['t_line']:.2f} s, up to {info['bound']:.2f} m off centre")
            drv.set_reference(pts, closed)
        self.get_logger().info(f"demo driver mode: {mode} | {self.driver_desc}")
        self.driver = drv
        self._publish_driver()
        return drv

    def _maybe_race(self, drv, cones, colors):
        return _race_on_map(self, drv, cones, colors)

    def _timing(self, now_sim: float) -> None:
        """Every 10 s: the longest wall-clock gap between two ticks and the oldest odometry used (sim time).
        Tells driver-side stalls apart from transport delays (Isaac logs the command age on its side)."""
        w = time.monotonic()
        if self.status is None or self.status.state != ASStatus.AS_DRIVING:     # only while driving
            self._tick_wall = self._tick_sim = self._clock_sim = self._rtf_ref = None
            self._gap_max = self._gap_sim = self._odom_age_max = self._stall_max = 0.0
            return
        last, last_sim = getattr(self, "_tick_wall", None), getattr(self, "_tick_sim", None)
        self._tick_wall, self._tick_sim = w, now_sim
        if last is not None and w - last > getattr(self, "_gap_max", 0.0):
            # sim time that passed in that gap: about the wall gap x real-time factor -> this node was held up
            # (/clock arrived but the tick did not run); much less -> /clock itself stopped (Isaac, relay or transport)
            self._gap_max, self._gap_sim = w - last, now_sim - (last_sim if last_sim is not None else now_sim)
        r0 = getattr(self, "_rtf_ref", None)                 # real-time factor over windows of >= 2 s
        if r0 is None:
            self._rtf_ref = (w, now_sim)
        elif w - r0[0] >= 2.0:
            self._rtf = float(np.clip((now_sim - r0[1]) / (w - r0[0]), 0.2, 1.5))
            self._rtf_ref = (w, now_sim)
        if now_sim != getattr(self, "_clock_sim", None):      # longest wall time without a new /clock
            self._clock_sim, self._clock_wall = now_sim, w
        self._stall_max = max(getattr(self, "_stall_max", 0.0), w - getattr(self, "_clock_wall", w))
        if self.odom is not None:
            st = self.odom.header.stamp
            self._odom_age_max = max(getattr(self, "_odom_age_max", 0.0), now_sim - (st.sec + st.nanosec * 1e-9))
        if w - getattr(self, "_timing_log", w - 11.0) >= 10.0 and self.status is not None and \
                self.status.state == ASStatus.AS_DRIVING:
            self.get_logger().info(f"timing: longest gap between ticks {1000 * getattr(self, '_gap_max', 0.0):.0f} ms "
                                   f"(sim time in it {1000 * getattr(self, '_gap_sim', 0.0):.0f} ms), "
                                   f"oldest odometry used {1000 * getattr(self, '_odom_age_max', 0.0):.0f} ms "
                                   f"(on arrival {1000 * getattr(self, '_odom_rx_age', 0.0):.0f} ms), "
                                   f"longest wait for /clock {1000 * getattr(self, '_stall_max', 0.0):.0f} ms, "
                                   f"pose carried forward up to {1000 * getattr(self, '_bridged_max', 0.0):.0f} ms "
                                   f"(real-time factor {self._rtf:.2f})")
            self._bridged_max = self._odom_rx_age = 0.0
            self._timing_log, self._gap_max, self._gap_sim, self._odom_age_max, self._stall_max = w, 0.0, 0.0, 0.0, 0.0

    def _tick(self):
        cmd = AckermannDriveStamped()
        now = self.get_clock().now()
        self._timing(now.nanoseconds * 1e-9)
        cmd.header.stamp = now.to_msg()
        cmd.header.frame_id = "base_link"
        drv = self._ensure_driver()
        if drv is not None and self.odom is not None and self.status is not None:
            o = self.odom.pose.pose
            q = o.orientation      # full yaw formula: the body also rolls and pitches (fsdsim.chassis)
            yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            pose = (o.position.x, o.position.y, yaw)
            v = self.odom.twist.twist.linear.x
            fresh = self._fresh_pose(now.nanoseconds * 1e-9)
            if fresh is not None:
                pose, v = fresh
                bridged = 0.0
                if not getattr(self, "_fresh_logged", False):
                    self._fresh_logged = True
                    self.get_logger().info("pose: own wheel-speed + gyro dead reckoning, corrected by /slam/odom")
            else:
                pose, bridged = self._bridge_stale_pose(pose, v)
            if bridged:
                self._bridged_max = max(getattr(self, "_bridged_max", 0.0), bridged)
            if not self._traj or math.hypot(pose[0] - self._traj[-1][0], pose[1] - self._traj[-1][1]) >= 0.5:
                self._traj.append(pose[:2])              # the driven path (for racing on the mapped track)
            cones, colors = [], []
            if self.cones is not None:
                for c in self.cones.cones:
                    if not c.knocked:
                        cones.append((c.position.x, c.position.y))
                        colors.append(c.color)
            drv = self._maybe_race(drv, cones, colors) or drv
            if isinstance(drv, RLDriver):
                tw = self.odom.twist.twist
                st = {"vy": tw.linear.y, "r": tw.angular.z}
                if self.imu is not None:
                    st["ax"], st["ay"] = self.imu.linear_acceleration.x, self.imu.linear_acceleration.y
                if self.status.state != ASStatus.AS_DRIVING:
                    drv.idle()
                steer, accel, _ = drv.update(pose, v, cones, colors, stop=self.status.mission_complete, state=st)
            else:
                steer, accel, _ = drv.update(pose, v, cones, colors, stop=self.status.mission_complete)
            if self.status.state == ASStatus.AS_DRIVING:
                cmd.drive.steering_angle = float(steer)
                self._last_steer = float(steer)
                cmd.drive.acceleration = float(accel)
        self.cmd_pub.publish(cmd)


def _race_on_map(node, drv, cones, colors):
    """Explore, then race: after `race_after` laps of following cones on a circuit, build the centre line from the
    cone map the driver was given (ground truth or /perception/cones), plan the line and drive it in map mode.
    Planning takes a second or two, so it runs in a thread while the car keeps following cones (a blocked control
    loop would trip the command watchdog); the finished driver is swapped in on a later tick."""
    st = node.status
    ready = getattr(node, "_race_ready", None)
    if ready is not None:
        node._race_ready = None
        new, desc, text = ready
        if new is None:
            node.get_logger().warn(text)
            return None
        new.follower.initialise((node.odom.pose.pose.position.x, node.odom.pose.pose.position.y))
        new._initialised = True
        node.driver, node.driver_desc = new, desc
        node._publish_driver()
        node.get_logger().info(text)
        return new
    if (not node.race_after or node._raced or drv.mode != "cones" or node.vehicle is None or st.lap_count < node.race_after
            or st.mission not in (ASStatus.MISSION_TRACKDRIVE, ASStatus.MISSION_AUTOCROSS) or st.mission_complete):
        return None
    node._raced = True
    import threading

    from fsdsim.control.demo_driver import RACING_MARGIN
    from fsdsim.perception.pipeline import track_from_map

    blue = np.array([c for c, k in zip(cones, colors) if k == 0], float).reshape(-1, 2)
    yellow = np.array([c for c, k in zip(cones, colors) if k == 1], float).reshape(-1, 2)
    traj = np.array(node._traj)
    gp = node.get_parameter
    grip, long_util = float(gp("grip_util").value), float(gp("long_util").value)
    veh, line, lap, mission = node.vehicle, node.line, st.lap_count, NAME_BY_MISSION.get(st.mission, "")
    longitudinal, policy, width_guess = node.longitudinal, node.rl_policy, node.driver_params.track_width_guess

    def plan():
        try:
            cl, width = track_from_map(traj, blue, yellow)
            path, info = reference_path(cl, width, True, veh, line, margin=RACING_MARGIN + 0.25)
            new, desc = make_vehicle_driver(veh, mission, "map", width, longitudinal, policy,
                                            grip_util=None if line == "racing" else grip, long_util=long_util, line=line,
                                            latency=float(node.get_parameter("latency").value))
            new.set_reference(path, True)
            node._race_ready = (new, desc + " on the mapped track" + (" (racing line)" if info else ""),
                                f"lap {lap} done: driving the mapped track ({len(blue)} blue, {len(yellow)} yellow cones, "
                                f"width {width:.2f} m)" + (f", racing line QSS {info['t_line']:.2f} s" if info else ""))
        except Exception as e:  # noqa: BLE001
            node._race_ready = (None, "", f"race on the map not possible ({e}): keep following cones")

    threading.Thread(target=plan, daemon=True).start()
    return None


def main(args=None):
    rclpy.init(args=args)
    node = DemoDriverNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass  # Ctrl+C / launch shutdown
    finally:
        node.destroy_node()
        shutdown_quietly()  # the SIGINT handler may already have shut the context down


if __name__ == "__main__":
    main()
