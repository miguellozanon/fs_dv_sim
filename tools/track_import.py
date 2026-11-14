#!/usr/bin/env python3
"""Real tracks for LapSim and the driverless simulator: GPS / line files in, centre line and cone layout out.

    python3 tools/track_import.py gps_lap.csv --latlon --closed --line out/line.csv          # lat,lon -> metres
    python3 tools/track_import.py line.csv --closed --cones out/track.csv --width 3.5 --png  # cones around the line
    python3 tools/track_import.py FSA2025_endurance_real.csv --closed --cones out/fsa_endurance.csv

Input: a CSV with two number columns per row, x,y in metres (or lat,lon in degrees with --latlon);
header and '#' lines are ignored. The line is fitted with a cubic spline (the LionsLTS way: spline, then a
moving average of the curvature) and written with 0.5 m spacing.

--line   writes the smoothed line as x_m,y_m (usable as a LapSim track: lts_track_csv(..., 'method', 'spline')).
--cones  writes an EUFS-style cone CSV for the simulator (blue left, yellow right, big orange at the start /
         finish line, car_start): ros2 launch fsd_sim_bringup isaac.launch.py track_file:=<csv> mission:=trackdrive
         The line must be a closed loop (autocross layouts with different start and finish are not supported).
"""

from __future__ import annotations

import argparse
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ros2_ws", "src", "fsdsim"))

import numpy as np  # noqa: E402

from fsdsim.perf.lapsim import _spline_track  # noqa: E402

EARTH_R = 6378137.0


def read_xy(path: str, latlon: bool = False) -> np.ndarray:
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.replace(";", ",").split(",")]
            try:
                rows.append([float(parts[0]), float(parts[1])])
            except (ValueError, IndexError):
                continue                      # header row
    p = np.array(rows, float)
    if len(p) < 4:
        raise ValueError(f"{path}: need at least 4 points, found {len(p)}")
    if latlon:
        lat0, lon0 = p[0]
        y = np.radians(p[:, 0] - lat0) * EARTH_R
        x = np.radians(p[:, 1] - lon0) * EARTH_R * math.cos(math.radians(lat0))
        p = np.stack([x, y], axis=1)
    return p


def fit_line(p: np.ndarray, closed: bool, ds: float = 0.5, curv_filter: float = 5.0):
    """Spline-fitted line at spacing ds (Track with x, y, k)."""
    return _spline_track(p[:, 0], p[:, 1], ds, curv_filter, closed, "line")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file")
    ap.add_argument("--latlon", action="store_true", help="columns are latitude, longitude [deg] (local metres from the first point)")
    ap.add_argument("--closed", action="store_true", help="the line is a closed loop")
    ap.add_argument("--line", help="write the fitted line as CSV (x_m,y_m)")
    ap.add_argument("--cones", help="write a simulator cone layout (EUFS CSV)")
    ap.add_argument("--width", type=float, default=3.5, help="track width between the cone lines [m]")
    ap.add_argument("--laps", type=int, default=10, help="laps for the layout (trackdrive: 10)")
    ap.add_argument("--curv-filter", type=float, default=5.0, help="curvature averaging width [m]")
    ap.add_argument("--png", action="store_true", help="also write a preview next to the cone CSV")
    args = ap.parse_args()
    if not (args.line or args.cones):
        ap.error("give --line and / or --cones")

    p = read_xy(args.file, args.latlon)
    tr = fit_line(p, args.closed, 0.5, args.curv_filter)
    kmax = float(np.max(np.abs(tr.k)))
    print(f"{os.path.basename(args.file)}: {len(p)} points, {tr.L:.0f} m, {'closed' if args.closed else 'open'}, "
          f"tightest radius {1 / max(kmax, 1e-9):.1f} m")
    if args.line:
        os.makedirs(os.path.dirname(os.path.abspath(args.line)), exist_ok=True)
        n = len(tr.x) - 1 if args.closed else len(tr.x)
        np.savetxt(args.line, np.stack([tr.x[:n], tr.y[:n]], axis=1), delimiter=",", header="x_m,y_m", comments="", fmt="%.4f")
        print(f"line: {args.line}")
    if args.cones:
        if not args.closed:
            print("--cones needs --closed (the simulator lays out closed circuits)", file=sys.stderr)
            return 2
        from fsdsim.config import load_all_configs
        from fsdsim.track import build_closed_track, save_csv

        veh, _, sim = load_all_configs(os.path.join(ROOT, "ros2_ws", "src", "fsd_sim_bringup", "config"))
        gen = sim.events.track_generator
        gen.width = args.width
        n = len(tr.x) - 1
        track = build_closed_track(gen, veh, args.laps, "trackdrive", centerline=np.stack([tr.x[:n], tr.y[:n]], axis=1))
        os.makedirs(os.path.dirname(os.path.abspath(args.cones)), exist_ok=True)
        save_csv(track, args.cones)
        print(f"cones: {args.cones}: {track.summary()}")
        if args.png:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            col = {0: "#1f5fd1", 1: "#f2c200", 2: "#ff8c1a", 3: "#e8430c", 4: "gray"}
            fig, ax = plt.subplots(figsize=(8, 6))
            xy, c = track.cone_xy(), track.cone_colors()
            ax.scatter(xy[:, 0], xy[:, 1], c=[col[i] for i in c], s=10)
            ax.plot(track.centerline[:, 0], track.centerline[:, 1], "k-", lw=0.5)
            ax.set_aspect("equal")
            ax.set_title(track.summary(), fontsize=8)
            out = os.path.splitext(args.cones)[0] + ".png"
            fig.savefig(out, dpi=80)
            plt.close(fig)
            print(f"preview: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
