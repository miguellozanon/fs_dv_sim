"""Offline dashboard demo: runs fsdsim + the reference driver in-process, no ROS needed.

    python3 -m fsd_dashboard.demo --mission trackdrive          # from ros2_ws/src/fsd_dashboard
    ros2 run fsd_dashboard dashboard_demo --mission skidpad     # after colcon build

Then open http://localhost:8080. The buttons drive the simulated AS state machine
(mission select, Activate, Go, Emergency stop, Reset) exactly like the ROS services.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

from fsdsim.config import load_all_configs
from fsdsim.control import RLDriver, make_vehicle_driver, reference_path
from fsdsim.mission import ASState, MISSION_BY_NAME
from fsdsim.sim import Simulation
from fsdsim.track import MISSIONS as TRACK_MISSIONS

from .feed import FsdsimFeed
from .model import MISSION_KEYS, DashboardModel, default_streams
from .predict import predict
from .server import DashboardServer
from .vehicle_setup import VehicleSetup, decode_start_setup


def _config_dir() -> str:
    here = Path(__file__).resolve()
    for p in [*here.parents]:
        cand = p / "fsd_sim_bringup" / "config"
        if cand.is_dir():
            return str(cand)
        cand = p / "src" / "fsd_sim_bringup" / "config"
        if cand.is_dir():
            return str(cand)
    try:
        from ament_index_python.packages import get_package_share_directory

        return os.path.join(get_package_share_directory("fsd_sim_bringup"), "config")
    except Exception:  # noqa: BLE001
        sys.exit("cannot find fsd_sim_bringup/config; pass --config-dir")


class DemoRunner:
    def __init__(self, config_dir: str, mission: str, rtf: float = 1.0, step_hz: float = 50.0, auto: bool = True,
                 changes=(), line: str = "center", faults: str = ""):
        self.veh, self.sens, self.simcfg = load_all_configs(config_dir, vehicle_overrides=list(changes))
        self.line = "racing" if str(line).lower() == "racing" else "center"
        self.simcfg.sim.faults = str(faults or "")
        self.simcfg.sim.mission = mission
        self.simcfg.sim.auto_activate = auto
        self.simcfg.sim.auto_go = auto
        self.rtf, self.step_hz = rtf, step_hz
        sc = self.sens
        self.model = DashboardModel(
            source="Offline demo (fsdsim, no ROS)",
            streams=default_streams(sc.camera.rate_hz, sc.lidar.rate_hz, sc.imu.rate_hz, sc.gnss.rate_hz,
                                    sc.vehicle_sensors.rate_hz))
        self.model.command_handler = self.command
        fp = self.veh.footprint
        self.model.set_car(self.veh.front_bumper, self.veh.rear_bumper, fp.width)
        self._lock = threading.Lock()
        self.config_dir = config_dir
        self._build(mission)
        self._stop = threading.Event()
        try:
            self.setup = VehicleSetup(config_dir, send_to_sim=self.set_vehicle, on_change=self.model.bump_setup,
                                      log=self.model.log, send_driver_set=self.set_driver)
            self.setup.rl.on_driver({"longitudinal": "reference", "requested": "reference",
                                     "description": self.driver_desc})
            self.model.setup = self.setup
            self.model.on_mission_complete = self.setup.on_mission_complete
        except Exception as e:  # noqa: BLE001
            self.setup = None
            self.model.log(f"Vehicle setup unavailable: {e}", "warning")

    def _make_driver(self) -> None:
        self.driver, self.driver_desc = make_vehicle_driver(self.veh, self.sim.mission_name, "map", self.sim.track.width,
                                                            getattr(self, "longitudinal", "reference"),
                                                            getattr(self, "rl_policy", "auto"),
                                                            **({"line": "racing"} if self._racing() else {}))
        path = self.sim.track.centerline
        if self._racing():
            # racing line inside the cones (fsdsim.perf.raceline); the reference driver only (the RL policy was
            # trained on the centre line)
            try:
                path, info = reference_path(path, self.sim.track.width, True, self.veh, "racing")
                self.driver_desc += f", racing line ({info['t_center'] - info['t_line']:.2f} s per lap faster in LapSim)"
            except Exception as e:  # noqa: BLE001
                self.model.log(f"Racing line not available, centre line used: {e}", "warning")
        self.driver.set_reference(path, self.sim.track.closed)
        if getattr(self, "setup", None) is not None:
            self.setup.rl.on_driver({"longitudinal": "rl" if isinstance(self.driver, RLDriver) else "reference",
                                     "requested": getattr(self, "longitudinal", "reference"),
                                     "policy": getattr(self, "rl_policy", "auto"),
                                     "description": self.driver_desc})

    def _racing(self) -> bool:
        return (getattr(self, "line", "center") == "racing" and self.sim.track.closed
                and self.sim.mission_name in ("autocross", "trackdrive")
                and getattr(self, "longitudinal", "reference") != "rl" and self.veh.model == "performance")

    def set_driver(self, req: dict):
        """Vehicle setup -> reference or RL driver (the run continues)."""
        lon = str(req.get("longitudinal", "reference"))
        with self._lock:
            self.longitudinal, self.rl_policy = lon, str(req.get("policy") or "auto")
            self._make_driver()
        self.model.log(f"Driver: {self.driver_desc}", "info")
        return True, self.driver_desc

    def set_vehicle(self, changes):
        """Vehicle setup -> swap the vehicle in this in-process simulation and restart the run."""
        try:
            veh = load_all_configs(self.config_dir, vehicle_overrides=list(changes))[0]
        except Exception as e:  # noqa: BLE001
            return False, f"{type(e).__name__}: {e}"
        with self._lock:
            self.veh = veh
            self.sim.set_vehicle(veh)
            mission = self.sim.mission_name
            self._make_driver()
            info = {"name": veh.name, "model": veh.model, "changes": list(veh.performance_changes)}
            self.model.set_vehicle_info(info)
            self.model.set_car(veh.front_bumper, veh.rear_bumper, veh.footprint.width)
            perf = veh.perf() if veh.model == "performance" else None
            tr = self.sim.track
        if perf is not None:
            self.model.set_prediction(predict(perf, mission, tr.centerline, tr.closed, tr.laps))
        if getattr(self, "setup", None) is not None:
            self.setup.on_vehicle(info)
        return True, f"vehicle swapped (mass {veh.mass:.1f} kg), run restarted"

    def _build(self, mission: str) -> None:
        self.simcfg.sim.mission = mission
        self.sim = Simulation(self.veh, self.sens, self.simcfg)
        self.sim.enable_analytic_lidar()
        self.feed = FsdsimFeed(self.model, self.sens)
        self._make_driver()
        self.model.log(f"Track loaded: {self.sim.track.name} ({mission})", "info")
        self.model.set_vehicle_info({"name": self.veh.name, "model": self.veh.model,
                                     "changes": list(self.veh.performance_changes)})
        perf = self.veh.perf() if self.veh.model == "performance" else None
        if perf is not None:
            tr = self.sim.track
            self.model.set_prediction(predict(perf, mission, tr.centerline, tr.closed, tr.laps))

    def command(self, cmd: dict):
        with self._lock:
            c = cmd["cmd"]
            if c == "select_mission":
                name = str(cmd.get("mission", "")).lower()
                if name not in MISSION_KEYS:
                    return False, f"unknown mission '{name}'"
                if name not in TRACK_MISSIONS:
                    return False, f"no {name} layout in the simulator (the car stands on jacks for DV inspection)"
                if self.sim.mission.state != ASState.OFF:
                    return False, "mission can only be changed in AS Off"
                if name != self.sim.mission_name and name in MISSION_BY_NAME:
                    self._build(name)          # new track for the new mission
                return self.sim.select_mission(name)
            if c == "activate":
                return self.sim.activate()
            if c == "go":
                return self.sim.go()
            if c == "emergency":
                return self.sim.emergency("RES (dashboard)")
            if c == "reset":
                self.sim.reset()
                return True, "simulation reset"
        return False, "unknown command"

    def step(self, dt: float) -> None:
        """Advance the simulation by dt (sim seconds) and feed the dashboard."""
        with self._lock:
            events = self.sim.step(dt)
            s = self.sim.vehicle.state
            status = self.sim.mission.status(self.sim.t)
            if isinstance(self.driver, RLDriver):
                if self.sim.mission.state != ASState.DRIVING:
                    self.driver.idle()
                st = {"vy": s.vy, "r": s.r, "ax": self.sim.vehicle.ax_body, "ay": self.sim.vehicle.ay_body}
                steer, accel, _ = self.driver.update(s.pose, s.vx, stop=status.get("mission_complete", False), state=st)
            else:
                steer, accel, _ = self.driver.update(s.pose, s.vx, stop=status.get("mission_complete", False))
            if self.sim.mission.state in (ASState.READY, ASState.DRIVING):
                self.sim.set_command(steer, accel)
                self.model.tick("control", self.sim.t)
                self.model.set_command(self.sim.t, steer, accel)
            for ev in events:
                self.feed.handle(ev)

    def run(self) -> None:
        dt = 1.0 / self.step_hz
        next_wall = time.monotonic()
        n_sub = max(1, int(round(dt * self.rtf / 0.02)))     # the driver runs at >= 50 Hz of sim time
        while not self._stop.is_set():
            for _ in range(n_sub):
                self.step(dt * self.rtf / n_sub)
            next_wall += dt
            time.sleep(max(0.0, next_wall - time.monotonic()))

    def stop(self) -> None:
        self._stop.set()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mission", default="trackdrive", choices=sorted(set(MISSION_KEYS) - {"inspection"}))   # no inspection track
    ap.add_argument("--config-dir", default=None)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--rtf", type=float, default=1.0, help="real-time factor")
    ap.add_argument("--manual", action="store_true", help="do not auto-activate / Go: use the buttons")
    ap.add_argument("--set", nargs="*", default=[], help="vehicle changes, e.g. m=300 mux*1.05 (see vehicle.yaml)")
    ap.add_argument("--start-setup", default="", help="the launcher's setup (base64 JSON name + changes)")
    ap.add_argument("--line", default="center", choices=["center", "racing"],
                    help="circuits: follow the centre line or the racing line (reference driver)")
    ap.add_argument("--faults", default="", help="sensor faults, e.g. lidar:dropout@20+3 (fsdsim.faults)")
    ap.add_argument("--longitudinal", default="reference", choices=["reference", "rl"],
                    help="throttle / brake: reference driver or the RL policy for this vehicle")
    args = ap.parse_args(argv)
    runner = DemoRunner(args.config_dir or _config_dir(), args.mission, args.rtf, auto=not args.manual, changes=args.set,
                        line=args.line, faults=args.faults)
    start = decode_start_setup(args.start_setup)
    if start is not None and runner.setup is not None:
        runner.setup.set_start(start["name"], start["changes"])
    if args.longitudinal == "rl":
        print("[fsd-dashboard]", runner.set_driver({"longitudinal": "rl"})[1])
    server = DashboardServer(runner.model, args.host, args.port).start()
    print(f"[fsd-dashboard] offline demo ({args.mission}) on http://localhost:{server.port}  (Ctrl+C to stop)")
    try:
        runner.run()
    except KeyboardInterrupt:
        pass
    finally:
        runner.stop()
        server.stop()


if __name__ == "__main__":
    main()
