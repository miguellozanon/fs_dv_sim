#!/usr/bin/env python3
"""Generate FSG track layouts as EUFS/FSSIM-style CSV files (+ optional PNG preview).

    python3 tools/generate_tracks.py                       # default set into fsd_sim_bringup/tracks
    python3 tools/generate_tracks.py --seeds 1 2 3 --png   # more random circuits with previews

CSV columns: tag,x,y,direction,x_variance,y_variance,xy_covariance
tags: blue, yellow, orange, big_orange, car_start (direction = start yaw)
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "ros2_ws", "src", "fsdsim"))

from fsdsim.config import load_all_configs  # noqa: E402
from fsdsim.track import build_track, save_csv  # noqa: E402


def preview(track, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    col = {0: "#1f5fd1", 1: "#f2c200", 2: "#ff8c1a", 3: "#e8430c", 4: "gray"}
    fig, ax = plt.subplots(figsize=(8, 6))
    xy, c = track.cone_xy(), track.cone_colors()
    ax.scatter(xy[:, 0], xy[:, 1], c=[col[i] for i in c], s=[30 if i == 3 else 12 for i in c])
    ax.plot(track.centerline[:, 0], track.centerline[:, 1], "k-", lw=0.5)
    ax.set_aspect("equal")
    ax.set_title(track.summary(), fontsize=8)
    fig.savefig(path, dpi=80)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "ros2_ws", "src", "fsd_sim_bringup", "config"))
    ap.add_argument("--out", default=os.path.join(ROOT, "ros2_ws", "src", "fsd_sim_bringup", "tracks"))
    ap.add_argument("--seeds", type=int, nargs="*", default=[7, 11, 23])
    ap.add_argument("--png", action="store_true")
    args = ap.parse_args()
    veh, _, sim = load_all_configs(args.config)
    os.makedirs(args.out, exist_ok=True)
    jobs = [("acceleration", None), ("skidpad", None)] + [("trackdrive", s) for s in args.seeds]
    for mission, seed in jobs:
        tr = build_track(mission, sim.events, veh, seed=seed)
        name = mission if seed is None else f"trackdrive_seed{seed}"
        path = os.path.join(args.out, name + ".csv")
        save_csv(tr, path)
        print(f"{path}: {tr.summary()}")
        if args.png:
            preview(tr, path.replace(".csv", ".png"))


if __name__ == "__main__":
    main()
