#!/usr/bin/env python3
"""Isaac Sim backend for the FSG driverless simulator.

Target setup: Isaac Sim 6.1 (or an Isaac Lab 3.0 Python env) on Windows 11, ROS 2 Jazzy
in WSL2 Ubuntu 24.04. Also works with Isaac Sim on native Linux.

Windows (PowerShell), after starting the relay in WSL:
    .\\isaac\\run_isaac.ps1 -Mission trackdrive
or directly with any Python that has Isaac Sim installed:
    C:\\isaacsim\\python.bat isaac\\run_isaac.py --mission trackdrive --relay-host <WSL IP>

WSL:
    ros2 launch fsd_sim_bringup isaac.launch.py

Division of labour
    fsdsim (this process)     vehicle dynamics, AS state machine, timing,
                              IMU / GNSS / wheel speed / steering models
    Isaac Sim RTX             Ouster OS1 LiDAR point cloud, ZED images
    TCP link -> isaac_relay   everything above goes to WSL as ROS 2 topics
                              (sensor transport "tcp", default). With "ros2",
                              LiDAR / cameras / clock are published by Isaac's
                              own ROS 2 bridge over DDS instead (native Linux /
                              Pixi setups).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "ros2_ws", "src", "fsdsim"))


PREVIEW_WIDTH = 320      # images when no ROS node reads the full ones (relay: sensor_demand)
PREVIEW_POINTS = 6000    # LiDAR points per scan when no ROS node reads the full cloud


def thin_cloud(cloud: dict, n: int) -> dict:
    """Every k-th point of a scan (all per-point arrays), at most n points."""
    import numpy as np  # noqa: PLC0415

    m = len(cloud["xyz"])
    if m <= n:
        return cloud
    idx = np.linspace(0, m - 1, n).astype(np.int64)
    return {key: (v[idx] if hasattr(v, "__len__") and len(v) == m else v) for key, v in cloud.items()}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mission", default="", help="acceleration | skidpad | autocross | trackdrive | ebs_test")
    ap.add_argument("--config", default=os.path.join(ROOT, "ros2_ws", "src", "fsd_sim_bringup", "config"))
    ap.add_argument("--track-file", default="")
    ap.add_argument("--seed", type=int, default=-1)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--render-hz", type=float, default=30.0, help="render / sensor update rate")
    ap.add_argument("--relay-host", default="127.0.0.1", help="host running the ROS 2 relay (WSL IP on Windows)")
    ap.add_argument("--relay-port", type=int, default=47800)
    ap.add_argument("--sensor-transport", choices=["tcp", "ros2"], default="tcp",
                    help="tcp: stream LiDAR/cameras to the relay (Windows+WSL). ros2: Isaac's DDS bridge")
    ap.add_argument("--lidar-source", choices=["rtx", "analytic"], default="rtx",
                    help="rtx: Isaac RTX LiDAR (falls back to analytic if unavailable)")
    ap.add_argument("--camera-streams", default="left,depth",
                    help="tcp transport: comma list of left,right,depth (more = more bandwidth)")
    ap.add_argument("--camera-hz", type=float, default=0.0, help="tcp transport: camera rate (0 = sensors.yaml)")
    ap.add_argument("--no-lidar", action="store_true")
    ap.add_argument("--no-camera", action="store_true")
    ap.add_argument("--motion-bvh", action="store_true",
                    help="enable RTX motion BVH at startup (LiDAR motion compensation; costs VRAM)")
    ap.add_argument("--max-time", type=float, default=0.0, help="stop after this many sim seconds (0 = forever)")
    ap.add_argument("--set", action="append", default=[], metavar="CHANGE",
                    help="vehicle parameter change, e.g. --set m=300 --set mux*1.05 (LapSim names, see vehicle.yaml)")
    ap.add_argument("--faults", default="", help='sensor faults, e.g. "lidar:dropout@20+3,camera:latency=0.2@0" (fsdsim.faults)')
    args, _ = ap.parse_known_args()  # Kit passes its own arguments through
    return args


def main():
    args = parse_args()

    import isaac_compat as ic

    app = ic.create_app(headless=args.headless, motion_bvh=args.motion_bvh)
    isaac_version = ic.isaac_version()
    ic.log(f"Isaac Sim {isaac_version} | Python {sys.version.split()[0]} | sensor transport {args.sensor_transport}")
    exts = ["isaacsim.sensors.experimental.rtx|isaacsim.sensors.rtx|omni.isaac.sensor", "omni.replicator.core"]
    if args.sensor_transport == "ros2":
        exts += ["isaacsim.ros2.bridge|omni.isaac.ros2_bridge", "isaacsim.ros2.nodes",
                 "isaacsim.core.nodes|omni.isaac.core_nodes"]
    ic.enable_extensions(exts)
    app.update()

    import numpy as np
    from fsdsim.config import config_fingerprint, load_all_configs, to_dict
    from fsdsim.link import PROTOCOL_VERSION, LinkClient
    from fsdsim.sim import Simulation
    from scene import Scene
    from sensor_streams import CameraStream, RtxLidarStream, zed_intrinsics

    veh, sens, simcfg = load_all_configs(args.config, vehicle_overrides=args.set)
    if args.set:
        ic.log(f"vehicle changes: {' '.join(args.set)}")
    if args.mission:
        simcfg.sim.mission = args.mission
    if args.track_file:
        simcfg.sim.track_file = args.track_file
    if args.seed >= 0:
        simcfg.events.track_generator.seed = args.seed
    if args.faults:
        simcfg.sim.faults = args.faults
        ic.log(f"sensor faults: {args.faults}")
    want_lidar = sens.lidar.enabled and not args.no_lidar
    want_cam = sens.camera.enabled and not args.no_camera
    sim = Simulation(veh, sens, simcfg, enable_lidar=want_lidar and args.lidar_source == "analytic",
                     enable_camera=False)
    ic.log(sim.track.summary())

    stage = ic.get_stage()
    scene = Scene(stage, sim.track, veh, sens)
    scene.build()
    app.update()

    # ---------------- sensors ----------------
    lidar_stream, cam_streams, lidar_rp, ros_cams = None, [], None, []
    if want_lidar and args.lidar_source == "rtx":
        try:
            lidar_prim, lidar_obj, variant = ic.create_rtx_lidar("/World/Car/os_lidar_mount", sens.lidar)
            if args.sensor_transport == "tcp":
                prof_ch = ic.profile_channels(variant, sens.lidar.channels)
                if prof_ch > sens.lidar.channels and prof_ch % sens.lidar.channels == 0:
                    ic.log(f"emulating a {sens.lidar.channels}-channel LiDAR: keeping 1 of every "
                           f"{prof_ch // sens.lidar.channels} beams of the {prof_ch}-channel profile")
                lidar_stream = RtxLidarStream(lidar_prim, sens.lidar, ic.log, lidar_obj, prof_ch)
            else:
                lidar_rp = ic.create_render_product(lidar_prim, (1, 1), "os_lidar")
        except Exception as e:  # noqa: BLE001
            ic.log(f"WARNING: RTX LiDAR unavailable ({e}); using the analytic OS1 model instead")
            sim.enable_analytic_lidar()

    cc = sens.camera
    cam_period = 1.0 / (args.camera_hz or cc.rate_hz)
    if want_cam:
        paths = scene.create_zed_cameras()
        ns = "/" + cc.namespace.strip("/")
        p = cc.frame_prefix
        streams = {s.strip() for s in args.camera_streams.split(",") if s.strip()}
        if args.sensor_transport == "tcp":
            for side in ("left", "right"):
                if side not in streams and not (side == "left" and "depth" in streams):
                    continue
                try:
                    cs = CameraStream(paths[side], tuple(cc.resolution), side == "left" and "depth" in streams, ic.log)
                    cam_streams.append((side, cs, side in streams))
                except Exception as e:  # noqa: BLE001
                    ic.log(f"WARNING: camera {side} unavailable ({e})")
        else:
            for side in ("left", "right"):
                rp = ic.create_render_product(paths[side], tuple(cc.resolution), f"zed_{side}")
                frame = f"{p}_{side}_camera_optical_frame"
                ros_cams.append({"rp": rp, "type": "rgb", "topic": f"{side}/image_rect_color", "frame": frame, "ns": ns.strip("/")})
                ros_cams.append({"rp": rp, "type": "info", "topic": f"{side}/camera_info", "frame": frame, "ns": ns.strip("/")})
                if side == "left":
                    ros_cams.append({"rp": rp, "type": "depth", "topic": "depth/depth_registered", "frame": frame, "ns": ns.strip("/")})
    if args.sensor_transport == "ros2":
        ic.build_ros_graph(lidar_rp, sens.lidar, ros_cams)

    # ---------------- link to the ROS 2 relay ----------------
    fingerprint = config_fingerprint(veh, sens, simcfg)

    cur = {"veh": veh}       # the vehicle can be swapped at run time (dashboard "Vehicle setup")

    def on_connect():
        veh = cur["veh"]
        return [
            {
                "kind": "hello",
                "protocol": PROTOCOL_VERSION,
                "isaac_version": isaac_version,
                "kit_log": ic.kit_log_path(),
                "sensor_transport": args.sensor_transport,
                "vehicle": to_dict(veh),
                "sensors": to_dict(sens),
                "sim": to_dict(simcfg),
                "fingerprint": fingerprint,
            },
            {"kind": "events", "t": sim.t, "events": [sim.track_event(), sim.cones_event()]},
        ]

    link = LinkClient(args.relay_host, args.relay_port, on_connect=on_connect, log=ic.log)
    cmd_age = []         # command ages since the last log line [s]
    cmd_wait = []        # wall time commands waited in Isaac before use [s]

    # what the ROS side reads (the relay's sensor_demand): full images / clouds only when a node subscribes to them
    demand = {"images": True, "lidar": True}

    restarted = [False]   # sim.reset() / vehicle swap: the simulation clock starts at 0 again

    def handle_cmd(cmd: dict):
        kind = cmd.get("cmd")
        if kind == "sensor_demand":
            demand.update(images=bool(cmd.get("images", True)), lidar=bool(cmd.get("lidar", True)))
            ic.log(f"relay reads: full images {'yes' if demand['images'] else 'no (small previews)'}, "
                   f"full LiDAR clouds {'yes' if demand['lidar'] else 'no (thinned)'}")
        elif kind == "control":
            sim.set_command(float(cmd["steer"]), float(cmd["accel"]))
            if cmd.get("stamp") is not None:     # how old the driver's command is when it reaches the car (sim time)
                cmd_age.append(sim.t - float(cmd["stamp"]))
        elif kind == "select_mission":
            ic.log(f"select_mission -> {sim.select_mission(int(cmd['mission']))}")
        elif kind == "activate":
            ic.log(f"activate -> {sim.activate()}")
        elif kind == "go":
            ic.log(f"go -> {sim.go()}")
        elif kind == "emergency":
            ic.log(f"emergency -> {sim.emergency('RES (service call)')}")
        elif kind == "reset":
            sim.reset()
            restarted[0] = True
            ic.log("simulation reset")
        elif kind == "vehicle":
            changes = [str(c) for c in cmd.get("changes") or []]
            try:
                new = load_all_configs(args.config, vehicle_overrides=changes)[0]
            except Exception as e:  # noqa: BLE001
                ic.log(f"vehicle change rejected ({' '.join(changes)}): {type(e).__name__}: {e}")
                link.send({"kind": "log", "text": f"vehicle change rejected: {type(e).__name__}: {e}"}, droppable=False)
                return
            sim.set_vehicle(new)
            restarted[0] = True
            cur["veh"] = new
            link.send({"kind": "vehicle", "vehicle": to_dict(new)}, droppable=False)
            ic.log(f"vehicle swapped ({' '.join(changes) or 'LapSim baseline'}), mass {new.mass:.1f} kg; run restarted")

    fx, fy, cxp, cyp = zed_intrinsics(cc)

    # ---------------- main loop ----------------
    dt = 1.0 / args.render_hz
    stepper = ic.Stepper(app, dt)
    stepper.start()
    ic.log(f"running: render {args.render_hz:.0f} Hz, fsdsim physics {simcfg.sim.physics_rate_hz:.0f} Hz, "
           f"relay tcp://{args.relay_host}:{args.relay_port}")
    wall0, last_log, last_state, next_cam = time.time(), time.time(), None, 0.0
    log_state = {"t": sim.t, "bytes": 0}
    last_isaac_t, time_offset, stalled = None, 0.0, 0
    # sim time of the recent rendered frames: a camera image read now shows the frame isaac_frame_lag renders ago
    from collections import deque
    cam_lag = max(0, int(getattr(cc, "isaac_frame_lag", 0)))
    lidar_lag = max(0, int(getattr(sens.lidar.isaac, "frame_lag", 0)))
    rendered_t = deque(maxlen=cam_lag + 1)
    try:
        while app.is_running():
            for cmd in link.poll():
                rx = cmd.pop("_rx", None) if isinstance(cmd, dict) else None
                if rx is not None:
                    cmd_wait.append(time.monotonic() - rx)   # wall time between arrival and use in this loop
                if cmd.get("kind") == "cmd":
                    handle_cmd(cmd)
            if restarted[0]:
                # the clock restarted at 0: the next camera frame was still due at the old time (e.g. t = 120 s), so
                # after "Reset simulation" no ZED image came for that long and YOLO drove blind (13 Nov 2026)
                restarted[0] = False
                next_cam = 0.0
                rendered_t.clear()
                last_isaac_t = None              # re-anchor Isaac's timeline to the new sim time
                log_state.update(t=sim.t, bytes=link.sent_bytes)
            # fsdsim advances exactly one render step per frame. Isaac's timeline time is only
            # monitored: Stop/Play in the Isaac window resets it, and the simulation must go on.
            isaac_t = stepper.time()
            if last_isaac_t is None:
                # Isaac's timeline already ran a few warm-up frames; measure drift relative to that.
                time_offset = sim.t - isaac_t
            else:
                if isaac_t + 1e-6 < last_isaac_t:
                    ic.log(f"Isaac timeline jumped back ({last_isaac_t:.2f} s -> {isaac_t:.2f} s; Stop/Play in the "
                           "Isaac window?) - simulation time continues")
                    time_offset = sim.t - isaac_t
                stalled = stalled + 1 if abs(isaac_t - last_isaac_t) < 1e-9 else 0
                if stalled == 30:
                    ic.log("Isaac timeline is not playing (Stop pressed?) - restarting playback so RTX sensors keep running")
                    getattr(stepper, "ensure_playing", lambda: None)()
            last_isaac_t = isaac_t
            step = dt
            events = sim.step(step)
            scene.update(sim, step)
            stepper.step()  # render: RTX sensors now see the new pose
            rendered_t.append(sim.t)

            sensor_events = []
            if lidar_stream is not None:
                cloud = lidar_stream.poll()
                if cloud is not None and len(cloud["xyz"]):
                    if lidar_stream.scans == 1:
                        ring = cloud["ring"]
                        ic.log(f"first RTX LiDAR scan: {len(cloud['xyz'])} points, rings {int(ring.min())}-"
                               f"{int(ring.max())}, max range {float(cloud['range'].max()) / 1000:.1f} m")
                    # the scan ends frame_lag renders ago; without per-point times the points are stamped
                    # mid-scan (mean error 0 instead of half a scan; 0.3 m cone offsets while driving, 11 Nov 2026)
                    t_end = sim.t - lidar_lag * dt
                    has_t = bool(len(cloud["t"])) and int(cloud["t"].max()) > 0
                    if lidar_stream.scans == 1:
                        ic.log(f"RTX LiDAR timing: per-point times {'yes, span ' + str(round(int(cloud['t'].max()) * 1e-6, 1)) + ' ms' if has_t else 'no (stamped mid-scan)'}, "
                               f"frame lag {lidar_lag} ({lidar_lag * dt * 1000:.0f} ms)")
                    period = 1.0 / sens.lidar.rate_hz
                    if not demand["lidar"]:
                        cloud = thin_cloud(cloud, PREVIEW_POINTS)
                    sensor_events.append({"type": "lidar", "stamp": max(0.0, t_end - (period if has_t else 0.5 * period)),
                                          "frame_id": sens.lidar.frame_id, "cloud": cloud})
                elif lidar_stream.scans == 0 and sim.t > 3.0:
                    ic.log(f"WARNING: no RTX LiDAR scan after {sim.t:.1f} s ({lidar_stream.diagnostics()}); "
                           "switching to the analytic OS1 model. Please report this log.")
                    lidar_stream = None
                    sim.enable_analytic_lidar()
            if cam_streams and sim.t + 1e-9 >= next_cam:
                next_cam = max(next_cam + cam_period, sim.t)
                for side, cs, send_rgb in cam_streams:
                    rgb, depth = cs.read()
                    frame = f"{cc.frame_prefix}_{side}_camera_optical_frame"
                    ns = "/" + cc.namespace.strip("/")
                    shown_t = rendered_t[0] if len(rendered_t) > cam_lag else sim.t
                    img = rgb if rgb is not None else depth
                    k = 1 if demand["images"] or img is None else max(1, -(-img.shape[1] // PREVIEW_WIDTH))
                    if k > 1:                           # nobody reads full images: a small one for the previews
                        rgb = None if rgb is None else np.ascontiguousarray(rgb[::k, ::k])
                        depth = None if depth is None else np.ascontiguousarray(depth[::k, ::k])
                    base = {"type": "image", "stamp": shown_t, "frame_id": frame,
                            "K": [fx / k, fy / k, cxp / k, cyp / k]}
                    if rgb is not None and send_rgb:
                        sensor_events.append({**base, "topic": f"{ns}/{side}/image_rect_color",
                                              "info_topic": f"{ns}/{side}/camera_info", "encoding": "rgb8",
                                              "data": rgb, "baseline": cc.baseline if side == "right" else 0.0})
                    if depth is not None:
                        # over the link as uint16 millimetres (half the bytes; the relay publishes 32FC1 metres
                        # again). The Windows -> WSL link carries ~80 MB/s at best; full float depth at 15 Hz
                        # filled it and every message, the car's pose too, arrived ~1 s late (11 Nov 2026)
                        mm = np.where(np.isfinite(depth) & (depth > 0) & (depth < 65.0), depth * 1000.0, 0.0)
                        sensor_events.append({**base, "topic": f"{ns}/depth/depth_registered",
                                              "info_topic": f"{ns}/depth/camera_info", "encoding": "32FC1",
                                              "data": mm.astype(np.uint16), "depth_mm": True})
            if sim.faults:                      # fault injection also on the rendered LiDAR / camera (fsdsim.faults)
                sensor_events = sim.faults.apply(sensor_events, sim.t)
            if events:
                link.send({"kind": "events", "t": sim.t, "events": events, "wall": time.time()}, droppable=False)
            if sensor_events:
                link.send({"kind": "events", "t": sim.t, "events": sensor_events, "sensors": True}, droppable=True)

            for ev in events:
                if ev["type"] == "as_status" and ev["state"] != last_state:
                    last_state = ev["state"]
                    ic.log(f"[t={sim.t:7.2f}] {ev['message']}")
            now = time.time()
            if now - last_log > 10.0:
                drift = stepper.time() + time_offset - sim.t
                # rates over the last interval (averages since start hide the cost of streaming)
                dt_wall = max(now - last_log, 1e-6)
                rtf = (sim.t - log_state["t"]) / dt_wall
                rate = (link.sent_bytes - log_state["bytes"]) / 1e6 / dt_wall
                ic.log(f"sim t={sim.t:7.2f}s  real-time factor {rtf:.2f}  "
                       f"drift {drift * 1000:+.1f} ms  link {'up' if link.connected.is_set() else 'DOWN'} "
                       f"({rate:.1f} MB/s, {link.dropped} frames dropped)"
                       + (f"  RTX LiDAR scans {lidar_stream.scans}" if lidar_stream is not None else "")
                       + (f"  command age median {1000 * sorted(cmd_age)[len(cmd_age) // 2]:.0f} ms, max "
                          f"{1000 * max(cmd_age):.0f} ms" if cmd_age else "")
                       + (f" (waited in Isaac max {1000 * max(cmd_wait):.0f} ms)" if cmd_wait else "")
                       + f"  state queued before sending max {1000 * link.state_wait_max:.0f} ms"
                       + f" (longest write {1000 * link.write_max[0]:.0f} ms, {link.write_max[1] / 1e6:.1f} MB)")
                link.state_wait_max, link.write_max = 0.0, (0.0, 0)
                cmd_age.clear()
                cmd_wait.clear()
                last_log = now
                log_state.update(t=sim.t, bytes=link.sent_bytes)
            if args.max_time and sim.t >= args.max_time:
                break
    except KeyboardInterrupt:
        pass
    finally:
        res = sim.result_dict()
        ic.log(f"result: {res}")
        link.close()
        app.close()


if __name__ == "__main__":
    main()
