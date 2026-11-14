"""Helpers shared by the bridge nodes."""

from __future__ import annotations

import os

from fsdsim.config import load_all_configs


def shutdown_quietly() -> None:
    """End rclpy after Ctrl+C / launch shutdown. The SIGINT handler may shut the context down at the same moment;
    rclpy.try_shutdown() then raised "rcl_shutdown already called" and every node printed a traceback at the end of
    each run (11 Nov 2026)."""
    import rclpy

    try:
        rclpy.try_shutdown()
    except Exception:  # noqa: BLE001
        pass


def default_config_dir() -> str:
    try:
        from ament_index_python.packages import get_package_share_directory

        return os.path.join(get_package_share_directory("fsd_sim_bringup"), "config")
    except Exception:  # pragma: no cover
        return ""


def split_changes(text) -> list[str]:
    """'m=300, mux*1.05' -> ['m=300', 'mux*1.05']"""
    import re

    return [c for c in re.split(r"[,\s]+", str(text or "")) if c]


def vehicle_from_summary(text: str):
    """VehicleParams from a /sim/vehicle message (the JSON of vehicle_summary): the car is under "vehicle".
    (The perception node read the whole message until 11 Nov 2026 and got the default car: 0.235 m wheels,
    1.40 m nose instead of 0.250 m / 1.81 m, so the own-car LiDAR filter missed the CAD nose.)"""
    import json

    from fsdsim.config import vehicle_from_dict

    d = json.loads(text)
    if not isinstance(d, dict) or not isinstance(d.get("vehicle"), dict):
        raise ValueError("no 'vehicle' in the message")
    return vehicle_from_dict(d["vehicle"])


def vehicle_summary(veh) -> dict:
    """Small JSON-able description of the active vehicle (published on /sim/vehicle)."""
    from fsdsim.config import to_dict

    d = {"name": veh.name, "model": veh.model, "changes": list(veh.performance_changes), "vehicle": to_dict(veh)}
    p = veh.perf() if veh.model == "performance" else None
    if p is not None:
        d["performance"] = p.to_dict()
        try:
            from fsdsim.perf import qss

            d["lapsim"] = {"acceleration": qss.acceleration(p).time, "skidpad": qss.skidpad(p).time}
        except Exception:  # noqa: BLE001
            pass
    return d


def load_configs_from_node(node):
    """Declare the common parameters and load (vehicle, sensors, sim) configs."""
    node.declare_parameter("config_dir", default_config_dir())
    node.declare_parameter("mission", "")
    node.declare_parameter("track_file", "")
    node.declare_parameter("seed", -1)
    node.declare_parameter("vehicle_set", "")   # e.g. "m=300 mux*1.05" (see vehicle.yaml)
    node.declare_parameter("faults", "")        # sensor faults, e.g. "lidar:dropout@20+3" (fsdsim.faults)
    cfg_dir = node.get_parameter("config_dir").value
    overrides = split_changes(node.get_parameter("vehicle_set").value)
    veh, sens, sim = load_all_configs(cfg_dir, vehicle_overrides=overrides)
    if overrides:
        node.get_logger().info(f"vehicle changes: {' '.join(overrides)}")
    mission = node.get_parameter("mission").value
    if mission:
        sim.sim.mission = mission
    track_file = node.get_parameter("track_file").value
    if track_file:
        sim.sim.track_file = track_file
    faults = str(node.get_parameter("faults").value or "")
    if faults:
        from fsdsim.faults import parse

        parse(faults)                              # a typo stops the start with a clear message
        sim.sim.faults = faults
        node.get_logger().info(f"sensor faults: {faults}")
    seed = int(node.get_parameter("seed").value)
    if seed >= 0:
        sim.events.track_generator.seed = seed
    node.get_logger().info(f"config dir: {cfg_dir or '<built-in defaults>'} | mission: {sim.sim.mission}")
    return veh, sens, sim


def zenoh_router_problem(host: str = "127.0.0.1", port: int = 7447, timeout: float = 0.5) -> str | None:
    """With rmw_zenoh, every node waits for the Zenoh router. Return a message if it is not running."""
    import socket

    if os.environ.get("RMW_IMPLEMENTATION", "") != "rmw_zenoh_cpp":
        return None
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return None
    except OSError:
        return (f"RMW_IMPLEMENTATION=rmw_zenoh_cpp but no Zenoh router on tcp/{host}:{port}. "
                "Start it first in its own terminal:  fsgsim && ros2 run rmw_zenoh_cpp rmw_zenohd  (alias: fsgrouter)")
