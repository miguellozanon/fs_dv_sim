#!/usr/bin/env python3
"""Run the simulator closed-loop with the reference driver, without ROS.

Useful as a regression test (CI) and to tune vehicle / driver parameters:

    python3 tools/closed_loop_eval.py                       # all events
    python3 tools/closed_loop_eval.py --missions skidpad --plot out/
    python3 tools/closed_loop_eval.py --lidar               # also run the OS1-64 model
    python3 tools/closed_loop_eval.py --missions trackdrive --track-file fsa_endurance.csv   # a real layout

Perception and localisation are perfect here (ground truth). Swap them for
your own modules to evaluate those in isolation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "ros2_ws", "src", "fsdsim"))

import numpy as np  # noqa: E402

from fsdsim.config import load_all_configs  # noqa: E402
from fsdsim.control import make_driver, reference_path  # noqa: E402
from fsdsim.control.demo_driver import RACING_MARGIN  # noqa: E402
from fsdsim.mission import ASState  # noqa: E402
from fsdsim.sim import Simulation  # noqa: E402

DEFAULT_MODES = {"acceleration": "map", "ebs_test": "map", "skidpad": "map", "autocross": "cones", "trackdrive": "cones"}
TIMEOUT = {"acceleration": 40, "ebs_test": 40, "skidpad": 80, "autocross": 120, "trackdrive": 700}


def timeout_for(mission, track):
    """Sim-time limit: the fixed value for the generated layouts, longer for long imported circuits
    (e.g. the 1.3 km FSA endurance lap: the exploration-style driver averages about 6 m/s)."""
    if not track.closed:
        return TIMEOUT[mission]
    lap = float(np.sum(np.linalg.norm(np.diff(track.centerline, axis=0), axis=1)))
    return max(TIMEOUT[mission], 30.0 + track.laps * lap / 4.0)


def run(mission, cfg_dir, mode=None, lidar=False, camera=False, seed=None, verbose=True, changes=(), track_file="",
        dt=0.02, line="center", laps=None, perception="", slam=False, faults="", race_after=0, cmd_latency=0.0,
        latency_comp=None):
    """dt: control period of the driver [s] (the ROS demo driver runs at 50 Hz, many stacks at 20-25 Hz).
    line: "center" or "racing" (map mode on closed circuits: perf.raceline). laps: other number of laps.
    perception: "" = ground-truth cones and pose; "lidar" | "fusion" | "camera" = fsdsim.perception (cones mode);
    slam: with perception, the pose also comes from the pipeline (wheel speeds + gyro + cone map).
    faults: sensor fault specification (fsdsim.faults), e.g. "lidar:dropout@20+3".
    race_after: with perception on a circuit, after this many laps the centre line is built from the cone MAP and
    the remaining laps are driven on it (or on its racing line) at the car's limits: explore, then race."""
    veh, sens, simcfg = load_all_configs(cfg_dir, vehicle_overrides=list(changes))
    simcfg.sim.mission = mission
    if track_file:
        simcfg.sim.track_file = track_file
    if seed is not None:
        simcfg.events.track_generator.seed = seed
    if laps:
        simcfg.events.autocross_laps = simcfg.events.trackdrive_laps = int(laps)
    if faults:
        simcfg.sim.faults = faults
    pipe = None
    if perception:
        lidar = lidar or perception in ("lidar", "fusion")
        camera = camera or perception in ("fusion", "camera")
    sim = Simulation(veh, sens, simcfg, enable_lidar=lidar, enable_camera=camera)
    mode = mode or DEFAULT_MODES[mission]
    if perception:
        from fsdsim.perception import PerceptionPipeline  # noqa: PLC0415

        # circuits are unknown: steer from the perceived cones. Acceleration, skidpad and the EBS test have a layout
        # fixed by the rules: follow the known path, with the pose from SLAM (a figure of eight cannot be driven
        # from cone midpoints)
        if DEFAULT_MODES[mission] == "cones":
            mode = "cones"
        pipe = PerceptionPipeline(veh, sens, perception, slam)
        pipe.start(sim.track.start_pose)
        sim.pipeline = pipe
    driver = make_driver(veh, mission, mode, sim.track.width, line=line if mode == "map" else "center")
    if mode == "map":
        path, sim.line_info = reference_path(sim.track.centerline, sim.track.width, sim.track.closed, veh, line)
        driver.set_reference(path, sim.track.closed)

    log = {"t": [], "x": [], "y": [], "v": [], "steer": [], "accel": [], "state": [], "ay": [], "r": [], "energy": [],
           "power": [], "fx": []}
    lidar_stats = {"scans": 0, "points": 0, "time": 0.0}
    log_slam = []
    from collections import deque  # noqa: PLC0415

    n_delay = int(round(cmd_latency / dt))
    # the driver predicts its pose over the latency it knows of (default: the simulated one)
    driver.p.latency = cmd_latency if latency_comp is None else float(latency_comp)
    cmd_q = deque(maxlen=n_delay + 1)
    sim.slam_log = log_slam          # rows (t, position error, heading error) with --slam
    raced = False
    status = {"mission_complete": False}
    wall0 = time.time()
    last_cloud = None
    t_max = timeout_for(mission, sim.track)
    while sim.t < t_max:
        t0 = time.time()
        events = sim.step(dt)
        if pipe is not None:
            pipe.handle_all(events)
        for ev in events:
            if ev["type"] == "as_status":
                status = ev
            elif ev["type"] == "lidar":
                lidar_stats["scans"] += 1
                lidar_stats["points"] += len(ev["cloud"]["xyz"])
                driving_for = sim.t - sim.mission.t_state
                if last_cloud is None and sim.mission.state == ASState.DRIVING and driving_for > 6.0:
                    last_cloud = (ev, sim.vehicle.state.pose)
        if lidar:
            lidar_stats["time"] += time.time() - t0
        s = sim.vehicle.state
        if pipe is not None:              # the driver sees only what the perception pipeline gives it
            pc = pipe.cones()
            cones, colors = [c[:2] for c in pc], [c[2] for c in pc]
            d_pose, d_v = (pipe.pose(), pipe.speed()) if slam else (s.pose, s.vx)
            if slam:
                log_slam.append((sim.t, math.hypot(d_pose[0] - s.x, d_pose[1] - s.y),
                                 abs((d_pose[2] - s.yaw + math.pi) % (2 * math.pi) - math.pi)))
        else:
            cones = [(c.x, c.y) for c in sim.track.cones if not c.knocked]
            colors = [int(c.color) for c in sim.track.cones if not c.knocked]
            d_pose, d_v = s.pose, s.vx
        if pipe is not None and race_after and not raced and status.get("lap_count", 0) >= race_after and sim.track.closed:
            raced = True
            try:
                cl, width = pipe.mapped_track(d_pose, sim.track.width)
                # more margin than with a perfect map and pose: mapped centre line and SLAM pose are each good to ~0.2 m
                path, sim.line_info = reference_path(cl, width, True, veh, line, margin=RACING_MARGIN + 0.25)
                driver = make_driver(veh, mission, "map", width, line=line)
                driver.p.latency = cmd_latency if latency_comp is None else float(latency_comp)
                driver.set_reference(path, True)
                sim.race_info = {"t": sim.t, "width": width, "length": float(np.sum(np.linalg.norm(np.diff(cl, axis=0), axis=1)))}
            except ValueError as e:
                sim.race_info = {"error": str(e)}
        steer, accel, _ = driver.update(d_pose, d_v, cones, colors, stop=status.get("mission_complete", False))
        # commands are ignored by the vehicle unless AS Driving (brakes held), but keep the
        # stream alive like a real stack does - auto Go waits for it
        if cmd_latency > 0:             # the stack's commands reach the car late (transport, a slow control loop)
            cmd_q.append((steer, accel))
            steer, accel = cmd_q[0] if len(cmd_q) > n_delay else (0.0, 0.0)
        sim.set_command(steer, accel) if sim.mission.state == ASState.DRIVING else sim.set_command(0.0, 0.0)
        vm = sim.vehicle
        for k, v in (("t", sim.t), ("x", s.x), ("y", s.y), ("v", s.vx), ("steer", s.steer),
                     ("accel", vm.ax_body), ("state", int(sim.mission.state)), ("ay", vm.ay_body), ("r", s.r),
                     ("energy", vm.energy_j), ("power", vm.power_w),
                     ("fx", vm.last_forces.get("Fx_f", 0.0) + vm.last_forces.get("Fx_r", 0.0))):
            log[k].append(v)
        if sim.mission.state == ASState.FINISHED:
            break
        if sim.mission.state == ASState.EMERGENCY and "ebs_stop_xy" in sim.mission.extra:
            break
    res = sim.result()
    wall = time.time() - wall0
    if verbose:
        lt = ", ".join(f"{x:.2f}" for x in res.lap_times)
        print(f"[{mission:12s}] mode={mode:5s} state={res.state:9s} laps={res.laps:2d} "
              f"raw={_fmt(res.raw_time)} pen={res.penalty_time:.1f}s cones={res.cones_hit} "
              f"OC={res.off_course} sim={sim.t:6.1f}s wall={wall:5.1f}s ({sim.t / max(wall, 1e-6):4.1f}x)")
        if lt:
            print(f"               laps: {lt}")
        if res.extra:
            print(f"               extra: {json.dumps(res.extra)}")
        if sim.vehicle.perf is not None:
            print(f"               energy: {sim.vehicle.energy_j / 3.6e6:.3f} kWh")
        if getattr(sim, "line_info", None):
            li = sim.line_info
            print(f"               racing line: QSS {li['t_center']:.2f} s on the centre line -> {li['t_line']:.2f} s "
                  f"(up to {li['bound']:.2f} m off centre, {li['length_center'] - li['length']:.1f} m shorter)")
        if pipe is not None:
            acc = pipe.accuracy([(c.x, c.y, int(c.color)) for c in sim.track.cones if not c.knocked])
            print(f"               perception ({perception}{', SLAM' if slam else ', ground-truth pose'}): "
                  f"{acc.get('matched', 0)} of {acc['truth']} cones mapped ({acc.get('false', 0)} false), "
                  f"position rms {acc.get('pos_rms', float('nan')):.3f} m, colour right {acc.get('colour_ok', float('nan')):.0%}")
            if log_slam:
                a = np.array(log_slam)
                print(f"               SLAM pose error: mean {a[:, 1].mean():.3f} m, max {a[:, 1].max():.3f} m, "
                      f"heading max {np.degrees(a[:, 2].max()):.2f} deg, {pipe.slam.corrections} corrections")
        ri = getattr(sim, "race_info", None)
        if ri:
            print("               race on the map: " + (ri["error"] if "error" in ri else
                  f"from t = {ri['t']:.1f} s on the mapped circuit ({ri['length']:.1f} m, width {ri['width']:.2f} m)"))
        if lidar and lidar_stats["scans"]:
            print(f"               lidar: {lidar_stats['scans']} scans, "
                  f"{lidar_stats['points'] / lidar_stats['scans']:.0f} pts/scan")
    return sim, res, log, last_cloud


def _fmt(v):
    return f"{v:7.3f}s" if v is not None else "   --   "


def plot_runs(runs, out_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    col = {0: "#1f5fd1", 1: "#f2c200", 2: "#ff8c1a", 3: "#e8430c", 4: "gray"}
    n = len(runs)
    fig, axs = plt.subplots(2, n, figsize=(5.2 * n, 9), gridspec_kw={"height_ratios": [3, 1]}, squeeze=False)
    for i, (name, (sim, res, log, _)) in enumerate(runs.items()):
        ax = axs[0, i]
        tr = sim.track
        for c in tr.cones:
            ax.scatter(c.x, c.y, s=36 if c.color == 3 else 14, c=col[int(c.color)],
                       marker="x" if c.knocked else "o", zorder=3)
        sc = ax.scatter(log["x"], log["y"], c=log["v"], cmap="viridis", s=2, zorder=2)
        tl = tr.timing_line
        ax.plot([tl.p1[0], tl.p2[0]], [tl.p1[1], tl.p2[1]], "r-", lw=2)
        ext = np.ptp(tr.centerline, axis=0)
        ax.set_aspect("equal" if max(ext) / max(min(ext), 1.0) < 6 else "auto")
        ax.set_title(f"{name}: {res.state}\nraw {_fmt(res.raw_time)}  cones {res.cones_hit}  OC {res.off_course}")
        fig.colorbar(sc, ax=ax, fraction=0.04, label="speed [m/s]")
        ax2 = axs[1, i]
        ax2.plot(log["t"], log["v"], label="v [m/s]")
        ax2.plot(log["t"], np.degrees(log["steer"]), label="steer [deg]")
        ax2.set_xlabel("time [s]")
        ax2.legend(loc="upper right", fontsize=8)
        ax2.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "closed_loop_runs.png")
    fig.savefig(path, dpi=90)
    print(f"saved {path}")


def plot_lidar(cloud_ev, out_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ev, _ = cloud_ev
    c = ev["cloud"]
    xyz, inten, ring = c["xyz"], c["intensity"], c["ring"]
    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(1, 2, 1)
    r = np.linalg.norm(xyz[:, :2], axis=1)
    above = xyz[:, 2] > -0.18  # sensor ~0.22 m above ground -> cone returns
    g = (~above) & (r < 25)
    ax.scatter(xyz[g, 0], xyz[g, 1], c="0.8", s=0.3)
    m = above & (r < 25)
    ax.scatter(xyz[m, 0], xyz[m, 1], c=np.clip(inten[m], 0, 220), s=6, cmap="turbo", vmin=0, vmax=220)
    ax.plot(0, 0, marker=">", color="k", ms=10)
    ax.set_xlim(-10, 25)
    ax.set_ylim(-15, 15)
    ax.set_aspect("equal")
    ax.set_title("OS1-64 model, bird's eye view (sensor frame, colour = reflectivity)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax = fig.add_subplot(1, 2, 2)
    cone = xyz[:, 2] > -0.18  # above ground (sensor ~0.22 m high)
    near = cone & (np.linalg.norm(xyz[:, :2], axis=1) < 25)
    ax.scatter(np.degrees(np.arctan2(xyz[near, 1], xyz[near, 0])), ring[near], c=inten[near], s=4, cmap="turbo")
    ax.invert_yaxis()
    ax.set_xlabel("azimuth [deg]")
    ax.set_ylabel("ring (0 = top beam)")
    ax.set_title("cone returns: azimuth vs ring (stripes visible in reflectivity)")
    fig.tight_layout()
    path = os.path.join(out_dir, "lidar_scan.png")
    fig.savefig(path, dpi=90)
    print(f"saved {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--missions", nargs="+", default=["acceleration", "skidpad", "autocross", "trackdrive"])
    ap.add_argument("--config", default=os.path.join(ROOT, "ros2_ws", "src", "fsd_sim_bringup", "config"))
    ap.add_argument("--mode", choices=["map", "cones"], default=None)
    ap.add_argument("--seed", type=int, default=None, help="track generator seed (autocross/trackdrive)")
    ap.add_argument("--lidar", action="store_true", help="also run the analytic OS1-64 model")
    ap.add_argument("--camera", action="store_true", help="also run the analytic ZED detector")
    ap.add_argument("--plot", default="", help="directory for PNG plots")
    ap.add_argument("--strict", action="store_true", help="exit 1 if a mission does not finish cleanly")
    ap.add_argument("--track-file", default="", help="cone layout CSV (EUFS) instead of the generated track, e.g. from tools/track_import.py")
    ap.add_argument("--set", nargs="*", default=[], help="vehicle changes, e.g. m=300 mux*1.05 (see vehicle.yaml)")
    ap.add_argument("--rate", type=float, default=50.0, help="driver control rate [Hz] (your stack's rate, e.g. 20)")
    ap.add_argument("--cmd-latency", type=float, default=0.0,
                    help="seconds until the driver's commands reach the car (Isaac Sim: about two render frames)")
    ap.add_argument("--latency-comp", type=float, default=None,
                    help="latency the driver compensates (default: --cmd-latency; 0 = no compensation)")
    ap.add_argument("--line", choices=["center", "racing"], default="center",
                    help="map mode on closed circuits: follow the centre line or the racing line (fsdsim.perf.raceline)")
    ap.add_argument("--laps", type=int, default=None, help="other number of laps (autocross / trackdrive)")
    ap.add_argument("--perception", choices=["", "lidar", "fusion", "camera"], default="",
                    help="cones for the driver from fsdsim.perception instead of the ground truth: lidar (colours from "
                         "the intensity stripes), fusion (LiDAR + camera colours), camera (stereo detections)")
    ap.add_argument("--slam", action="store_true", help="with --perception: pose from wheel speeds + gyro + cone map")
    ap.add_argument("--race-after", type=int, default=0, metavar="LAPS",
                    help="with --perception on a circuit: after LAPS laps drive on the mapped track (with --line racing: its racing line)")
    ap.add_argument("--faults", default="", help='sensor faults, e.g. "lidar:dropout@20+3,gnss:loss@10+30" (docs/PERCEPTION.md)')
    ap.add_argument("--csv", default="", help="directory for one CSV per mission with the LionsLTS / Marple signal names")
    args = ap.parse_args()

    runs, failed = {}, []
    for m in args.missions:
        sim, res, log, cloud = run(m, args.config, args.mode, args.lidar, args.camera, args.seed, changes=args.set, track_file=args.track_file,
                                    dt=1.0 / args.rate, line=args.line, laps=args.laps, perception=args.perception,
                                    slam=args.slam, faults=args.faults, race_after=args.race_after, cmd_latency=args.cmd_latency,
                                    latency_comp=args.latency_comp)
        if args.csv:
            from fsdsim.perf.export import write_closed_loop_csv  # noqa: PLC0415

            p = os.path.join(args.csv, f"sim_{m}.csv")
            print(f"               csv: {p} ({write_closed_loop_csv(log, sim.vehicle.perf, p)} rows)")
        runs[m] = (sim, res, log, cloud)
        ok_state = "EMERGENCY" if m == "ebs_test" else "FINISHED"
        if res.state != ok_state or res.cones_hit or res.off_course:
            failed.append(m)
    if args.plot:
        plot_runs(runs, args.plot)
        for m in sorted(runs, key=lambda k: k not in ("autocross", "trackdrive")):
            if runs[m][3] is not None:
                plot_lidar(runs[m][3], args.plot)
                break
    if failed:
        print(f"not clean: {failed}")
        if args.strict:
            sys.exit(1)


if __name__ == "__main__":
    main()
