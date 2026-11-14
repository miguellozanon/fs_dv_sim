#!/usr/bin/env python3
"""Racing line inside the cones for a circuit, with the lap-time gain from the car's QSS model.

    python3 tools/racing_line.py --mission trackdrive --seed 7 --png out/line.png
    python3 tools/racing_line.py --track-file tracks_out/fsa_endurance.csv --out out/fsa_line.csv --png out/fsa.png
    python3 tools/racing_line.py --line-file lapsim/LR26_LapSim/tracks/FSA2025_endurance_real.csv --width 3.5

Input: a simulator circuit (generated or a cone CSV), or a centre line x,y file with --width. Output: the line as
x_m,y_m (usable as a LapSim track: lts_track_csv(..., 'method', 'spline')) and a picture. The driverless driver
follows the same line with `--line racing` (closed_loop_eval.py) / `line:=racing` (launch files).
Method and limits: fsdsim/perf/raceline.py (own code; not the TUM optimisation).
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ros2_ws", "src", "fsdsim"))

import numpy as np  # noqa: E402

from fsdsim.config import load_all_configs  # noqa: E402
from fsdsim.perf import raceline  # noqa: E402
from fsdsim.track import build_track, load_csv  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(ROOT, "ros2_ws", "src", "fsd_sim_bringup", "config"))
    ap.add_argument("--mission", default="trackdrive", choices=["autocross", "trackdrive"])
    ap.add_argument("--seed", type=int)
    ap.add_argument("--track-file", help="cone layout CSV (EUFS)")
    ap.add_argument("--line-file", help="centre line CSV: two number columns x,y [m] (closed loop)")
    ap.add_argument("--width", type=float, help="track width for --line-file [m]")
    ap.add_argument("--margin", type=float, default=0.40, help="car side to cone line [m]")
    ap.add_argument("--util", type=float, default=1.0, help="share of the tyre limit for the lap times (driver: 0.75)")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--out", help="write the line as x_m,y_m CSV")
    ap.add_argument("--png")
    args = ap.parse_args()
    veh, _, sim = load_all_configs(args.config, vehicle_overrides=args.set)
    cones = None
    if args.line_file:
        rows = []
        for ln in open(args.line_file, encoding="utf-8"):
            p = ln.replace(";", ",").split(",")
            try:
                rows.append([float(x) for x in p[:4]])
            except ValueError:
                continue
        a = np.array(rows)
        center = a[:, 1:3] if a.shape[1] >= 4 else a[:, :2]      # s,x,y,kappa files or x,y files
        width = args.width or sim.events.track_generator.width
    else:
        if args.seed is not None:
            sim.events.track_generator.seed = args.seed
        tr = load_csv(args.track_file, args.mission, sim.events, veh) if args.track_file else \
            build_track(args.mission, sim.events, veh)
        center, width, cones = tr.centerline, tr.width, tr
    perf = veh.perf()
    q = perf.copy(util_x=args.util, util_y=args.util, events={})
    r = raceline.racing_line(center, width, q, True, car_width=veh.footprint.width, margin=args.margin)
    print(f"track width {width:.2f} m, car {veh.footprint.width:.2f} m, margin {args.margin:.2f} m -> the line may move "
          f"{r['bound']:.2f} m off centre")
    print(f"centre line: {r['length_center']:.1f} m, {r['t_center']:.3f} s   racing line: {r['length']:.1f} m, "
          f"{r['t_line']:.3f} s   ({100 * (1 - r['t_line'] / r['t_center']):.1f} % faster, QSS at {args.util:.0%} of the tyre limit)")
    print("candidates (length weight: lap time): " + ", ".join(f"{w:g}: {t:.3f}" for w, t in r["candidates"]))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        np.savetxt(args.out, r["xy"][:-1], delimiter=",", header="x_m,y_m", comments="", fmt="%.4f")
        print(args.out)
    if args.png:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 7))
        c = np.vstack([center, center[:1]])
        ax.plot(c[:, 0], c[:, 1], color="0.6", lw=0.8, ls="--", label="centre line")
        ax.plot(r["xy"][:, 0], r["xy"][:, 1], color="#c8102e", lw=1.6, label="racing line")
        if cones is not None:
            col = {0: "#1f5fd1", 1: "#e0b000", 2: "#ff8c1a", 3: "#e8430c"}
            xy, cc = cones.cone_xy(), cones.cone_colors()
            ax.scatter(xy[:, 0], xy[:, 1], s=8, c=[col.get(int(i), "gray") for i in cc])
        ax.set_aspect("equal")
        ax.legend()
        ax.set_title(f"{r['t_center']:.2f} s -> {r['t_line']:.2f} s (QSS)")
        os.makedirs(os.path.dirname(os.path.abspath(args.png)), exist_ok=True)
        fig.savefig(args.png, dpi=110, bbox_inches="tight")
        print(args.png)
    return 0


if __name__ == "__main__":
    sys.exit(main())
