"""Relay between the Isaac Sim process and the ROS 2 graph (runs in WSL / Linux).

Isaac Sim (Windows or Linux) connects to this node over TCP (fsdsim.link) and
streams simulator events: AS state, IMU / GNSS / wheel sensors, ground truth,
and - with sensor transport "tcp" - the RTX LiDAR cloud and the ZED images.
This node publishes them as the same typed topics as the lite backend, and
forwards /control/command and the /as/* services back to Isaac.

No DDS traffic has to cross the Windows <-> WSL boundary.

    ros2 launch fsd_sim_bringup isaac.launch.py            # listens on 0.0.0.0:47800
"""

from __future__ import annotations

import queue
import time

import rclpy
from rclpy.executors import ExternalShutdownException
import rclpy.time
from rclpy.clock import Clock, ClockType
from rclpy.node import Node

from tf2_ros import Buffer, TransformListener

from fsdsim import __version__ as FSDSIM_VERSION
from fsdsim.config import config_fingerprint, sensors_from_dict, sim_from_dict, vehicle_from_dict
from fsdsim.link import DEFAULT_PORT, PROTOCOL_VERSION, LinkServer
from fsdsim.mission import ASState

from .ros_interface import SimRosInterface
from .common import shutdown_quietly


class IsaacRelayNode(Node):
    def __init__(self):
        super().__init__("fsd_isaac_relay")
        self.declare_parameter("bind_host", "0.0.0.0")
        self.declare_parameter("port", DEFAULT_PORT)
        self.declare_parameter("publish_clock", "auto")  # auto | true | false
        self.declare_parameter("drain_hz", 500.0)
        self.declare_parameter("expect_rviz", False)  # set by isaac.launch.py when it starts RViz
        self.clock_param = str(self.get_parameter("publish_clock").value).lower()
        host = self.get_parameter("bind_host").value
        port = int(self.get_parameter("port").value)
        self.link = LinkServer(host, port, log=lambda m: self.get_logger().info(m))
        self.get_logger().info(f"fsg-dv-sim {FSDSIM_VERSION} relay - waiting for Isaac Sim on tcp://{host}:{self.link.port}")
        self.io: SimRosInterface | None = None
        self.fingerprint = None
        self.publish_clock = False
        self._last_state = None
        self._last_msg_wall = None
        self._session = False  # a connection that sent "hello" (launcher port probes don't)
        self._kit_log = ""
        # self-check: does map -> base_link actually arrive over ROS 2? (tells relay vs. RViz problems apart)
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._hello_wall = None
        self._cfg = None
        self._tf_checked = False
        self._graph_checked = False
        self._stats = {"msgs": 0, "lidar": 0, "image": 0, "skipped": 0}
        self._last_drain, self._max_gap, self._max_wait = None, 0.0, 0.0
        self._transit_min, self._transit_max = float("inf"), float("-inf")
        self._state_t = None          # newest sim time of a state message (frames much older are skipped)
        self._stats_t = time.monotonic()
        self._bytes0 = 0
        # Steady-clock timers: this node publishes /clock, so its own timers must never wait
        # for sim time (with use_sim_time:=true they would never fire -> deadlock).
        steady = Clock(clock_type=ClockType.STEADY_TIME)
        self.create_timer(1.0 / float(self.get_parameter("drain_hz").value), self._drain, clock=steady)
        self.create_timer(5.0, self._watchdog, clock=steady)
        self.create_timer(1.0, self._sensor_demand, clock=steady)
        self._demand = None

    # ------------------------------------------------------------------ #
    BACKLOG_FRAMES = 4   # frames waiting beyond this: publish only the light events of the older ones

    def _drain(self) -> None:
        now = time.monotonic()
        if self._last_drain is not None:
            self._max_gap = max(self._max_gap, now - self._last_drain)   # executor busy elsewhere this long
        self._last_drain = now
        budget = 200
        # at most ~15 ms per call: the driver's commands are handled by the same executor between calls
        # (a long backlog of images used to hold them back for 0.4-0.7 s, 11 Nov 2026)
        while budget and time.monotonic() - now < 0.015:
            try:
                msg = self.link.incoming.get_nowait()
            except queue.Empty:
                return
            budget -= 1
            rx = msg.pop("_rx", None)
            if rx is not None:
                self._max_wait = max(self._max_wait, time.monotonic() - rx)
            sent = msg.pop("wall", None)
            if sent is not None and rx is not None:
                # Windows and WSL wall clocks differ by a fixed offset: the fastest message is the reference
                transit = time.time() - (time.monotonic() - rx) - float(sent)
                self._transit_min = min(self._transit_min, transit)
                self._transit_max = max(self._transit_max, transit)
            kind = msg.get("kind")
            if kind == "events":
                # When the relay falls behind (CPU busy: YOLO, perception, the dashboard), images and point clouds
                # of frames that are already old are skipped, so pose, IMU, wheels and AS state stay current.
                # Before 11 Nov 2026 the queue grew without bound: the driver steered on 1-2 s old poses and the
                # car left the track even on ground truth.
                behind = self.link.incoming.qsize() > self.BACKLOG_FRAMES
                t_msg = float(msg.get("t", 0.0))
                if msg.get("sensors"):
                    # a frame of only images / scans: drop it whole when the relay is behind, or when it is much older
                    # than the state already published (state has its own connection: frames can arrive late, and
                    # a frame 0.5 s back would look like a simulator restart to the ROS timeline)
                    if behind or (self._state_t is not None and t_msg < self._state_t - 0.25):
                        self._stats["skipped"] += 1
                        continue
                else:
                    self._state_t = t_msg      # state arrives in order on its connection (and restarts at 0 on reset)
                self._on_events(msg, light=behind)
            elif kind == "hello":
                self._on_hello(msg)
            elif kind == "vehicle":
                self._on_vehicle(msg)
            elif kind == "log":
                self.get_logger().info(f"[isaac] {msg.get('text', '')}")
            elif kind == "_connected":
                self._session = False
                self._state_t = None
                self.get_logger().debug(f"TCP connection from {msg['peer']}")
            elif kind == "_disconnected":
                if self._session:
                    hint = f" If it crashed, see the end of the Kit log on Windows: {self._kit_log}" if self._kit_log else ""
                    self.get_logger().warn(f"Isaac Sim disconnected ({msg.get('reason', '')}).{hint}")
                self._session = False

    def _on_hello(self, msg: dict) -> None:
        if msg.get("protocol") != PROTOCOL_VERSION:
            self.get_logger().error(f"protocol mismatch: Isaac {msg.get('protocol')} vs relay {PROTOCOL_VERSION}; "
                                    "update both sides to the same fsg-dv-sim version")
            return
        veh = vehicle_from_dict(msg["vehicle"])
        sens = sensors_from_dict(msg["sensors"])
        sim = sim_from_dict(msg["sim"])
        fp = config_fingerprint(veh, sens, sim)
        transport = msg.get("sensor_transport", "tcp")
        self._session = True
        self._cfg = (sens, sim)
        self._kit_log = msg.get("kit_log", "")
        self._hello_wall = time.monotonic()
        self._demand = None                    # tell the new Isaac session what to send (next _sensor_demand)
        self._tf_checked = False
        self._graph_checked = False
        self.get_logger().info(f"Isaac Sim connected from {self.link.peer}")
        self.get_logger().info(
            f"hello from Isaac Sim {msg.get('isaac_version', '?')} | mission {sim.sim.mission} | "
            f"sensor transport {transport} | config {fp}")
        if self.io is None:
            if self.clock_param == "auto":
                self.publish_clock = transport == "tcp"  # with "ros2", Isaac's OmniGraph publishes /clock
            else:
                self.publish_clock = self.clock_param == "true"
            self.io = SimRosInterface(
                self, sens, veh, hooks=_Hooks(self), publish_clock=self.publish_clock,
                publish_gt=sim.sim.publish_ground_truth, publish_gt_tf=sim.sim.publish_gt_tf,
            )
            self.fingerprint = fp
        else:
            self.io.new_session()
            self.io.vp = veh
            self.io.publish_vehicle(veh)   # Isaac may have been started with other -Set changes
        if veh.model == "performance" and veh.performance_changes:
            self.get_logger().info(f"vehicle changes from Isaac: {' '.join(veh.performance_changes)}")
        if fp != self.fingerprint:
            self.get_logger().warn("Isaac Sim reconnected with a different config (topics / frames may be stale) - "
                                   "restart this relay to apply it")

    def _on_vehicle(self, msg: dict) -> None:
        """Isaac swapped the vehicle (dashboard Vehicle setup): publish it on /sim/vehicle."""
        veh = vehicle_from_dict(msg["vehicle"])
        if self.io is not None:
            self.io.vp = veh
            self.io.publish_vehicle(veh)
        if self._cfg is not None:
            self.fingerprint = config_fingerprint(veh, *self._cfg)
        ch = " ".join(veh.performance_changes) or "LapSim baseline"
        self.get_logger().info(f"Isaac Sim switched vehicle ({ch}); run restarted")

    HEAVY = ("image", "lidar")

    def _on_events(self, msg: dict, light: bool = False) -> None:
        if self.io is None:
            return
        self._last_msg_wall = time.monotonic()
        self._stats["msgs"] += 1
        self.io.begin_frame(float(msg["t"]))
        for ev in msg.get("events", []):
            t = ev.get("type")
            if light and t in self.HEAVY:
                self._stats["skipped"] += 1
                continue
            if t in self._stats:
                self._stats[t] += 1
            self.io.handle(ev)
            if t == "as_status" and ev["state"] != self._last_state:
                self._last_state = ev["state"]
                self.get_logger().info(f"[t={ev['stamp']:7.2f}] {ev['message']}")

    def _check_tf(self) -> None:
        if self._tf_checked or not self._session or self._hello_wall is None or self.io is None:
            return
        if time.monotonic() - self._hello_wall < 4.0:
            return
        self._tf_checked = True
        sc = self.io.sc
        try:
            t = self._tf_buffer.lookup_transform(sc.map_frame, sc.base_frame, rclpy.time.Time())
            tr = t.transform.translation
            self.get_logger().info(
                f"TF self-check OK: {sc.map_frame} -> {sc.base_frame} received over ROS 2 "
                f"(x={tr.x:.2f}, y={tr.y:.2f}, stamp {t.header.stamp.sec}.{t.header.stamp.nanosec:09d}). "
                "If RViz still shows no frames, the problem is on the RViz side (see docs/WINDOWS_WSL.md).")
        except Exception as e:  # noqa: BLE001
            frames = self._tf_buffer.all_frames_as_string().strip() or "(none)"
            self.get_logger().error(f"TF self-check FAILED: {sc.map_frame} -> {sc.base_frame}: {e}. Frames seen: {frames}")
        self._log_graph(warn_rviz=False)

    def _check_graph_late(self) -> None:
        """RViz (rviz_wait) starts after the TF self-check; look again once it should be up."""
        if self._graph_checked or not self._session or self._hello_wall is None:
            return
        if time.monotonic() - self._hello_wall < 15.0:
            return
        self._graph_checked = True
        self._log_graph(warn_rviz=bool(self.get_parameter("expect_rviz").value))

    def _log_graph(self, warn_rviz: bool) -> None:
        """What DDS discovery shows from here: tells 'RViz never discovered' apart from RViz-side problems."""
        try:
            nodes = sorted(f"{ns.rstrip('/')}/{n}" for n, ns in self.get_node_names_and_namespaces())
            tf_subs = sorted({f"{i.node_namespace.rstrip('/')}/{i.node_name}"
                              for i in self.get_subscriptions_info_by_topic("/tf")})
        except Exception:  # noqa: BLE001
            return
        self.get_logger().info(f"ROS graph seen by the relay: nodes {', '.join(nodes) or '(none)'}; "
                               f"/tf subscribers {', '.join(tf_subs) or '(none)'}")
        if warn_rviz and not any("rviz" in n for n in nodes):
            import os  # noqa: PLC0415

            if os.environ.get("RMW_IMPLEMENTATION") == "rmw_zenoh_cpp":
                hint = ("all terminals must use `fsgsim` (same ROS_DOMAIN_ID) and the Zenoh router "
                        "(`fsgrouter`) must keep running")
            else:
                hint = ("DDS discovery may be failing between processes (common in WSL mirrored networking); "
                        "the Zenoh setup avoids this: bash tools/wsl/setup_shell.sh")
            self.get_logger().warn(f"RViz is not visible in the ROS graph yet. If it stays empty: {hint} "
                                   "(docs/WINDOWS_WSL.md).")

    def _check_clock_publishers(self) -> None:
        """Two /clock sources make sim-time nodes (RViz) jump back and forth in time."""
        if not self.publish_clock:
            return
        try:
            infos = self.get_publishers_info_by_topic("/clock")
        except Exception:  # noqa: BLE001
            return
        others = sorted({f"{i.node_namespace.rstrip('/')}/{i.node_name}" for i in infos if i.node_name != self.get_name()})
        if others and others != getattr(self, "_clock_others", None):
            self.get_logger().error(
                f"another node also publishes /clock: {', '.join(others)}. Sim-time nodes such as RViz will see time "
                "jump back and forth and drop all TF ('Frame [map] does not exist'). Stop it (e.g. an old lite_sim, "
                "rosbag play, or Isaac's own ROS 2 bridge) or run `ros2 node list` to find it.")
        self._clock_others = others

    def _watchdog(self) -> None:
        self._check_tf()
        self._check_graph_late()
        self._check_clock_publishers()
        now = time.monotonic()
        dt = now - self._stats_t
        if self.link.connected and self._stats["msgs"]:
            self.get_logger().info(
                f"link {self.link.peer}: {self._stats['msgs'] / dt:5.1f} msg/s, LiDAR {self._stats['lidar'] / dt:4.1f} Hz, "
                f"images {self._stats['image'] / dt:5.1f}/s, {(self.link.recv_bytes - self._bytes0) / 1e6 / dt:6.1f} MB/s"
                + (f", {self._stats['skipped']} old images / scans skipped (relay behind, queue "
                   f"{self.link.incoming.qsize()})" if self._stats["skipped"] else "")
                + f"; longest wait in the relay {1000 * self._max_wait:.0f} ms, longest pause {1000 * self._max_gap:.0f} ms"
                + (f", state on the way from Isaac up to {1000 * (self._transit_max - self._transit_min):.0f} ms "
                   "longer than the fastest" if self._transit_max > float("-inf") else ""),
                throttle_duration_sec=10.0,
            )
        elif not self.link.connected or not self._session:
            self.get_logger().warn(
                f"no Isaac Sim connected on port {self.link.port} - start isaac/run_isaac.ps1 on Windows "
                "(see docs/WINDOWS_WSL.md)", throttle_duration_sec=30.0)
        self._stats = {k: 0 for k in self._stats}
        self._max_gap, self._max_wait, self._transit_max = 0.0, 0.0, float("-inf")
        self._stats_t = now
        self._bytes0 = self.link.recv_bytes

    def send(self, cmd: dict) -> bool:
        return self.link.send({"kind": "cmd", **cmd})

    def _sensor_demand(self) -> None:
        """Full 1280 x 720 images and full LiDAR clouds only while a ROS node subscribes to them (YOLO, perception,
        RViz, a recorder); else Isaac sends small ones that are enough for the dashboard previews. On the team PC
        (11 Nov 2026) ~23 MB/s of images nobody read held the car's state back in the Windows -> WSL link for up
        to 0.4 s (Isaac's log: "state queued before sending")."""
        if not self._session or self.io is None:
            return
        sc = self.io.sc
        ns = "/" + sc.camera.namespace.strip("/")
        cams = (f"{ns}/left/image_rect_color", f"{ns}/right/image_rect_color", f"{ns}/depth/depth_registered")
        demand = {"images": any(self.count_subscribers(t) > 0 for t in cams),
                  "lidar": self.count_subscribers(sc.lidar.topic) > 0}
        if demand != self._demand and self.send({"cmd": "sensor_demand", **demand}):
            self._demand = demand
            self.get_logger().info(f"Isaac sends {'full' if demand['images'] else 'small (no subscriber)'} images and "
                                   f"{'full' if demand['lidar'] else 'thinned (no subscriber)'} LiDAR clouds")


class _Hooks:
    NOT_CONNECTED = (False, "Isaac Sim is not connected")

    def __init__(self, node: IsaacRelayNode):
        self.n = node

    def _status(self):
        return self.n.io.last_status if self.n.io else {}

    def _fwd(self, cmd: dict, ok_text: str):
        return (True, ok_text) if self.n.send(cmd) else self.NOT_CONNECTED

    def on_control(self, steer, accel, stamp):
        self.n.send({"cmd": "control", "steer": steer, "accel": accel, "stamp": stamp})

    def select_mission(self, mission):
        if self._status().get("state", ASState.OFF) != ASState.OFF:
            return False, "mission can only be changed in AS Off"
        return self._fwd({"cmd": "select_mission", "mission": int(mission)}, "forwarded (check /as/status)")

    def activate(self):
        return self._fwd({"cmd": "activate"}, "forwarded (check /as/status)")

    def go(self):
        return self._fwd({"cmd": "go"}, "forwarded (check /as/status)")

    def emergency(self):
        return self._fwd({"cmd": "emergency"}, "forwarded")

    def reset(self):
        return self._fwd({"cmd": "reset"}, "forwarded")

    def set_vehicle(self, changes):
        return self._fwd({"cmd": "vehicle", "changes": list(changes)}, "forwarded to Isaac Sim (it restarts the run)")


def main(args=None):
    rclpy.init(args=args)
    node = IsaacRelayNode()
    # two threads: commands (own callback group in SimRosInterface) are forwarded while the other thread publishes
    from rclpy.executors import MultiThreadedExecutor  # noqa: PLC0415

    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass  # Ctrl+C / launch shutdown
    finally:
        node.link.close()
        node.destroy_node()
        shutdown_quietly()  # the SIGINT handler may already have shut the context down


if __name__ == "__main__":
    main()
