"""Lightweight simulator backend (no GPU): fsdsim core + analytic LiDAR / camera.

    ros2 run fsd_sim_bridge lite_sim --ros-args -p mission:=skidpad

This node owns the simulation clock and publishes /clock; every other node
should run with use_sim_time:=true.
"""

from __future__ import annotations

import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.clock import Clock, ClockType
from rclpy.node import Node

from fsdsim.dv_can import CandumpWriter, DvLogger
from fsdsim.mission import MISSION_BY_NAME
from fsdsim.sim import Simulation

from .common import load_configs_from_node
from .ros_interface import SimRosInterface
from .common import shutdown_quietly


class LiteSimNode(Node):
    def __init__(self):
        super().__init__("fsd_lite_sim")
        veh, sens, simcfg = load_configs_from_node(self)
        self.declare_parameter("realtime_factor", simcfg.sim.realtime_factor)
        self.declare_parameter("step_hz", 100.0)
        self.declare_parameter("control_mode", "acceleration")  # acceleration | speed
        self.declare_parameter("speed_kp", 2.0)
        self.rtf = float(self.get_parameter("realtime_factor").value)
        self.step_hz = float(self.get_parameter("step_hz").value)
        self.control_mode = self.get_parameter("control_mode").value
        self.speed_kp = float(self.get_parameter("speed_kp").value)

        self.sim = Simulation(veh, sens, simcfg)
        self.get_logger().info(self.sim.track.summary())
        self.io = SimRosInterface(
            self, sens, veh, hooks=_Hooks(self), publish_clock=True,
            publish_gt=simcfg.sim.publish_ground_truth, publish_gt_tf=simcfg.sim.publish_gt_tf,
        )
        self.dv_logger = None
        if simcfg.sim.candump_log:
            a = veh.actuators
            self.dv_logger = DvLogger(CandumpWriter(simcfg.sim.candump_log), a.max_steer, a.max_accel, a.max_decel)
            self.get_logger().info(f"writing DV CAN log to {simcfg.sim.candump_log}")
        self._cmd = (0.0, 0.0)
        self._last_status = None
        self._last_gt = None
        self._wall0 = time.monotonic()
        self._last_report = self._wall0
        self._last_state = None
        self._debt, self._dropped, self._t_report = 0.0, 0.0, 0.0
        self._last_tick = self._wall0
        self.max_catch_up = 4                    # steps per tick at most when behind
        # steady clock: this node is the /clock source (see isaac_relay_node)
        self.create_timer(1.0 / self.step_hz, self._tick, clock=Clock(clock_type=ClockType.STEADY_TIME))

    # ------------------------------------------------------------------ #
    def _tick(self):
        # Keep up with the wall clock: a late tick (LiDAR scan, GC, a busy CPU) is made up by extra steps in the
        # next ticks instead of being lost (before: every late tick slowed the run, 0.86-0.89x on the team laptop).
        now = time.monotonic()
        self._debt += self.rtf * (now - self._last_tick)
        self._last_tick = now
        h = self.rtf / self.step_hz
        n = 0
        while self._debt >= 0.5 * h and n < self.max_catch_up:
            self._step_once(h)
            self._debt -= h
            n += 1
        if self._debt > 0.25 * max(self.rtf, 1.0):          # more than 0.25 s behind: give up on that part
            self._dropped += self._debt
            self._debt = 0.0
        if now - self._last_report > 10.0:
            ratio = (self.sim.t - self._t_report) / max(now - self._last_report, 1e-6)
            if 0.0 <= ratio < 0.95 * self.rtf:                  # < 0: the run was restarted
                self.get_logger().warn(f"simulation running at {ratio:.2f}x real time (requested {self.rtf:.2f}x): "
                                       "the CPU cannot keep up")
            self._last_report, self._t_report = now, self.sim.t

    def _step_once(self, h: float) -> None:
        events = self.sim.step(h)
        self.io.begin_frame(self.sim.t)
        for ev in events:
            self.io.handle(ev)
            if ev["type"] == "as_status":
                self._last_status = ev
                if ev["state"] != self._last_state:
                    self._last_state = ev["state"]
                    self.get_logger().info(f"[t={self.sim.t:7.2f}] {ev['message']}")
            elif ev["type"] == "ground_truth":
                self._last_gt = ev
        if self.dv_logger and self._last_status and self._last_gt:
            self.dv_logger.maybe_log(self.sim.t, self._last_gt, self._last_status, self._cmd)
        if self.sim.mission.state.name in ("FINISHED",) and not getattr(self, "_reported", False):
            self._reported = True
            self.get_logger().info(f"result: {self.sim.result_dict()}")


class _Hooks:
    """Callbacks used by SimRosInterface (kept off the Node to avoid name clashes)."""

    def __init__(self, node: LiteSimNode):
        self.n = node

    def on_control(self, steer: float, accel: float, stamp: float) -> None:
        n = self.n
        if n.control_mode == "speed":
            accel = n.speed_kp * (accel - n.sim.vehicle.state.vx)  # 'acceleration' field carries speed
        n._cmd = (steer, accel)
        n.sim.set_command(steer, accel)

    def select_mission(self, mission: int):
        return self.n.sim.select_mission(mission)

    def activate(self):
        return self.n.sim.activate()

    def go(self):
        return self.n.sim.go()

    def emergency(self):
        return self.n.sim.emergency("RES (service call)")

    def reset(self):
        self.n.sim.reset()
        self.n._reported = False
        self.n._t_report = 0.0
        return True, "simulation reset"

    def set_vehicle(self, changes):
        """Re-read the vehicle (LapSim MATLAB files + vehicle.yaml), apply `changes`, restart the run."""
        from fsdsim.config import load_all_configs  # noqa: PLC0415

        n = self.n
        try:
            veh = load_all_configs(n.get_parameter("config_dir").value, vehicle_overrides=changes)[0]
        except Exception as e:  # noqa: BLE001
            return False, f"{type(e).__name__}: {e}"
        n.sim.set_vehicle(veh)
        n._reported = False
        n.io.vp = veh
        n.io.publish_vehicle(veh)
        return True, f"vehicle {veh.name}: mass {veh.mass:.1f} kg, run restarted"


def main(args=None):
    rclpy.init(args=args)
    node = LiteSimNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass  # Ctrl+C / launch shutdown
    finally:
        if node.dv_logger:
            node.dv_logger.w.close()
        node.destroy_node()
        shutdown_quietly()  # the SIGINT handler may already have shut the context down


if __name__ == "__main__":
    main()

__all__ = ["LiteSimNode", "MISSION_BY_NAME"]
