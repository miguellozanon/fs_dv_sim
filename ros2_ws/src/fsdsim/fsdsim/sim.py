"""Simulation orchestrator shared by every backend.

`Simulation.step(dt)` advances vehicle, AS state machine and sensors and
returns a list of *events*: plain dicts (JSON-serialisable except LiDAR point
arrays) such as::

    {"type": "imu", "stamp": 12.345, "frame_id": "imu_link", ...}

The lightweight ROS node turns events into ROS messages directly; the Isaac Sim
backend forwards them as JSON to the relay node, which uses the same
converter. That keeps one source of truth for vehicle behaviour, AS logic and
proprioceptive sensors, whichever renderer is used for LiDAR and cameras.
"""

from __future__ import annotations

import math
from dataclasses import asdict

import numpy as np

from .config import SensorsConfig, SimConfig, VehicleParams
from .dynamics import VehicleModel
from .geometry import circle_rect_distance
from .mission import ASState, MissionManager, MISSION_BY_NAME
from .sensors import GnssModel, ImuModel, LidarModel, StereoCameraModel, VehicleSensorModel
from .track import Track, build_track, load_csv


class _Scheduled:
    def __init__(self, name: str, rate_hz: float, model, offset: float = 0.0):
        self.name = name
        self.period = 1.0 / rate_hz
        self.model = model
        self.next_t = offset


class Simulation:
    def __init__(
        self,
        vehicle: VehicleParams,
        sensors: SensorsConfig,
        simcfg: SimConfig,
        track: Track | None = None,
        mission: str | None = None,
        enable_lidar: bool | None = None,
        enable_camera: bool | None = None,
    ):
        self.vp = vehicle
        self.sc = sensors
        self.cfg = simcfg
        s = simcfg.sim
        self.mission_name = (mission or s.mission).lower()
        self.rng = np.random.default_rng(s.seed)
        if track is not None:
            self.track = track
        elif s.track_file:
            self.track = load_csv(s.track_file, self.mission_name, simcfg.events, vehicle)
        else:
            self.track = build_track(self.mission_name, simcfg.events, vehicle)
        self.vehicle = VehicleModel(vehicle)
        self.mission = MissionManager(self.track, simcfg.autonomous_system, simcfg.penalties, simcfg.events)
        self.physics_dt = 1.0 / s.physics_rate_hz

        use_lidar = sensors.lidar.enabled and (s.analytic_lidar if enable_lidar is None else enable_lidar)
        use_cam = sensors.camera.enabled and (s.analytic_camera if enable_camera is None else enable_camera)
        self.sensors: list[_Scheduled] = []
        sub = lambda: np.random.default_rng(self.rng.integers(1 << 31))  # noqa: E731
        if sensors.imu.enabled:
            self.sensors.append(_Scheduled("imu", sensors.imu.rate_hz, ImuModel(sensors.imu, sub())))
        if sensors.gnss.enabled:
            self.sensors.append(_Scheduled("gnss", sensors.gnss.rate_hz, GnssModel(sensors.gnss, sub())))
        if sensors.vehicle_sensors.enabled:
            self.sensors.append(
                _Scheduled("vehicle_sensors", sensors.vehicle_sensors.rate_hz, VehicleSensorModel(sensors.vehicle_sensors, sub()))
            )
        if use_lidar:
            self.sensors.append(_Scheduled("lidar", sensors.lidar.rate_hz, LidarModel(sensors.lidar, sub())))
        if use_cam:
            self.sensors.append(_Scheduled("camera", sensors.camera.rate_hz, StereoCameraModel(sensors.camera, sub())))
        from .faults import FaultInjector  # noqa: PLC0415

        self.faults = FaultInjector(s.faults, s.seed)
        self.status_period = 0.1
        self.gt_period = 0.02
        self._cone_home = [(c.x, c.y) for c in self.track.cones]   # knocked cones are moved; reset puts them back
        self.reset()

    # ------------------------------------------------------------------ #
    def enable_analytic_lidar(self) -> None:
        """Turn on the analytic LiDAR model at runtime (fallback when RTX is unavailable)."""
        if any(s.name == "lidar" for s in self.sensors):
            return
        rng = np.random.default_rng(self.rng.integers(1 << 31))
        sch = _Scheduled("lidar", self.sc.lidar.rate_hz, LidarModel(self.sc.lidar, rng))
        sch.next_t = self.t
        self._scan_start_pose = self.vehicle.state.pose
        self.sensors.append(sch)

    def set_vehicle(self, vehicle: VehicleParams) -> None:
        """Swap the vehicle (e.g. new LapSim parameters from the dashboard) and restart the mission.
        The track and the sensor models stay; the car starts again from the start pose."""
        self.vp = vehicle
        self.vehicle = VehicleModel(vehicle)
        self.reset()

    def reset(self) -> None:
        x, y, yaw = self.track.start_pose
        self.vehicle.reset(x, y, yaw)
        self.vehicle.brake_override = self.cfg.autonomous_system.hold_decel
        self.mission.reset()
        for c, (hx, hy) in zip(self.track.cones, self._cone_home):
            c.knocked = False
            c.x, c.y = hx, hy
        self._cones_xy = self.track.cone_xy()
        self._cone_r = np.array([c.hit_radius for c in self.track.cones])
        self.t = 0.0
        self._last_cmd_t: float | None = None
        self._next_status = 0.0
        self._next_gt = 0.0
        self._scan_start_pose = self.vehicle.state.pose
        self._cones_dirty = True
        for i, sch in enumerate(self.sensors):
            sch.next_t = 0.0
        self._auto = {"selected": False, "activated": False, "go": False}
        self.faults.reset()
        self.pending_events: list[dict] = [self.track_event()]

    # ------------------------------------------------------------------ #
    # Inputs
    # ------------------------------------------------------------------ #
    def set_command(self, steer: float, accel: float) -> None:
        self.vehicle.set_command(steer, accel)
        self._last_cmd_t = self.t

    def select_mission(self, mission=None):
        return self.mission.select_mission(mission if mission is not None else self.mission_name)

    def activate(self):
        return self.mission.activate(self.t)

    def go(self):
        ok, msg = self.mission.go(self.t)
        if ok:
            self._last_cmd_t = self.t
        return ok, msg

    def emergency(self, reason: str = "RES"):
        return self.mission.emergency(self.t, reason)

    # ------------------------------------------------------------------ #
    def _auto_sequence(self) -> None:
        s = self.cfg.sim
        if s.auto_activate and not self._auto["selected"] and self.mission.state == ASState.OFF:
            self.select_mission()
            self._auto["selected"] = True
            self.activate()
            self._auto["activated"] = True
        if s.auto_go and not self._auto["go"] and self.mission.state == ASState.READY:
            stack_alive = self._last_cmd_t is not None and self.t - self._last_cmd_t < 0.5
            if s.auto_go_wait_for_stack and not stack_alive:
                return
            if self.t - self.mission.t_state >= s.go_delay_s:
                self.go()
                self._auto["go"] = True

    def step(self, dt: float) -> list[dict]:
        """Advance the simulation by dt seconds and return emitted events."""
        events = self.pending_events
        self.pending_events = []
        t_end = self.t + dt
        as_cfg = self.cfg.autonomous_system
        while self.t < t_end - 1e-9:
            h = min(self.physics_dt, t_end - self.t)
            self._auto_sequence()
            st = self.mission.state
            if st == ASState.DRIVING:
                if self._last_cmd_t is not None and self.t - self._last_cmd_t > as_cfg.command_timeout_s:
                    self.mission.emergency(self.t, "control command timeout")
                    st = self.mission.state
            if st == ASState.DRIVING:
                self.vehicle.brake_override = None
            elif st in (ASState.EMERGENCY,):
                self.vehicle.brake_override = as_cfg.ebs_decel
            else:  # Off / Ready / Finished: brakes held
                self.vehicle.brake_override = as_cfg.hold_decel
            prev, prev_front = (self.vehicle.state.x, self.vehicle.state.y), self._front_point()
            self.vehicle.step(h)
            self.t += h
            cur = (self.vehicle.state.x, self.vehicle.state.y)
            self._check_cones()
            self.mission.update(self.t, h, prev, cur, self.vehicle.state.speed, prev_front, self._front_point())
            events.extend(self._sample_sensors())
        if self.t + 1e-9 >= self._next_status:
            self._next_status = max(self._next_status + self.status_period, self.t)
            events.append(self.status_event())
        if self.cfg.sim.publish_ground_truth and self.t + 1e-9 >= self._next_gt:
            self._next_gt = max(self._next_gt + self.gt_period, self.t)
            events.append(self.gt_event())
        if self._cones_dirty:
            events.append(self.cones_event())
            self._cones_dirty = False
        return self.faults.apply(events, self.t) if self.faults else events

    def _front_point(self) -> tuple[float, float]:
        s, fb = self.vehicle.state, self.vp.front_bumper
        return (s.x + fb * math.cos(s.yaw), s.y + fb * math.sin(s.yaw))

    # ------------------------------------------------------------------ #
    def _check_cones(self) -> None:
        s = self.vehicle.state
        p = self.vp
        if len(self._cones_xy) == 0:
            return
        dx = self._cones_xy[:, 0] - s.x
        dy = self._cones_xy[:, 1] - s.y
        near = np.nonzero(dx * dx + dy * dy < 9.0)[0]
        if near.size == 0:
            return
        c, sn = math.cos(s.yaw), math.sin(s.yaw)
        lx = c * dx[near] + sn * dy[near]
        ly = -sn * dx[near] + c * dy[near]
        center = 0.5 * (p.front_bumper - p.rear_bumper)
        d = circle_rect_distance(lx - center, ly, 0.5 * (p.front_bumper + p.rear_bumper), p.footprint.width / 2)
        for k in np.nonzero(d < self._cone_r[near])[0]:
            cone = self.track.cones[near[k]]
            if cone.knocked:
                continue
            cone.knocked = True
            # push the cone away from the car and mark it as down
            push = 0.6
            norm = math.hypot(dx[near[k]], dy[near[k]]) or 1.0
            cone.x += push * dx[near[k]] / norm + 0.3 * c
            cone.y += push * dy[near[k]] / norm + 0.3 * sn
            self._cones_xy[near[k]] = (cone.x, cone.y)
            if self.mission.state == ASState.DRIVING:
                self.mission.on_cone_hit()
            self._cones_dirty = True

    def _sample_sensors(self) -> list[dict]:
        out = []
        for sch in self.sensors:
            if self.t + 1e-9 < sch.next_t:
                continue
            sch.next_t += sch.period
            if sch.next_t < self.t:  # fell behind (large dt): resync
                sch.next_t = self.t + sch.period
            out.append(self._measure(sch))
        return out

    def _measure(self, sch: _Scheduled) -> dict:
        sc, t, vm = self.sc, self.t, self.vehicle
        if sch.name == "imu":
            ev = sch.model.measure(vm)
            ev.update(type="imu", stamp=t, frame_id=sc.imu.frame_id)
        elif sch.name == "gnss":
            ev = sch.model.measure(vm)
            ev.update(type="gnss", stamp=t, frame_id=sc.gnss.frame_id)
        elif sch.name == "vehicle_sensors":
            ev = sch.model.measure(vm)
            ev.update(type="vehicle_sensors", stamp=t, frame_id=sc.base_frame)
        elif sch.name == "lidar":
            start = self._scan_start_pose
            end = vm.state.pose
            cloud = sch.model.scan(start, end, self.track.cones, vm.chassis.state.attitude)
            self._scan_start_pose = end
            ev = {"type": "lidar", "stamp": t - sch.period, "frame_id": sc.lidar.frame_id, "cloud": cloud}
        elif sch.name == "camera":
            dets = sch.model.detect(vm.state.pose, self.track.cones)
            ev = {
                "type": "camera_detections",
                "stamp": t,
                "frame_id": f"{sc.camera.frame_prefix}_left_camera_optical_frame",
                "detections": dets,
            }
        else:  # pragma: no cover
            raise KeyError(sch.name)
        return ev

    # ------------------------------------------------------------------ #
    def status_event(self) -> dict:
        ev = self.mission.status(self.t)
        ev.update(type="as_status", stamp=self.t)
        return ev

    def gt_event(self) -> dict:
        s = self.vehicle.state
        return {
            "type": "ground_truth",
            "stamp": self.t,
            "pose": [s.x, s.y, s.yaw],
            "twist": [s.vx, s.vy, s.r],
            "steer": s.steer,
            "accel": [self.vehicle.ax_body, self.vehicle.ay_body],
            "wheel_omega": list(self.vehicle.wheel_speeds()),
            "attitude": list(self.vehicle.chassis.state.attitude),     # roll, pitch [rad], heave [m] of the body
            "energy_kwh": self.vehicle.energy_j / 3.6e6,
            "power_kw": self.vehicle.power_w / 1e3,
        }

    def cones_event(self) -> dict:
        return {
            "type": "cones",
            "stamp": self.t,
            "cones": [[c.id, int(c.color), c.x, c.y, bool(c.knocked)] for c in self.track.cones],
        }

    def track_event(self) -> dict:
        tl = self.track.timing_line
        return {
            "type": "track",
            "stamp": self.t,
            "name": self.track.name,
            "mission": self.track.mission,
            "closed": self.track.closed,
            "width": self.track.width,
            "laps": self.track.laps,
            "centerline": self.track.centerline.tolist(),
            "start_pose": list(self.track.start_pose),
            "timing_line": [tl.p1.tolist(), tl.p2.tolist()] if tl is not None else None,
            "finish_line": (
                [self.track.finish_line.p1.tolist(), self.track.finish_line.p2.tolist()]
                if self.track.finish_line is not None
                else None
            ),
        }

    def result(self):
        return self.mission.result(self.t)

    def result_dict(self) -> dict:
        return asdict(self.result())


__all__ = ["Simulation", "MISSION_BY_NAME"]
