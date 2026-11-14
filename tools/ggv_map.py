#!/usr/bin/env python3
"""ggv map of the car from the LapSim model: drive / brake limits against speed and lateral acceleration.

    python3 tools/ggv_map.py --out results/ggv.csv --png
    python3 tools/ggv_map.py --set m=300 "mu*0.95" --speeds 5 10 15 20 25 30 --event autocross

Columns: v [m/s], ay [m/s^2], ax_acc, ax_dec [m/s^2]. --event applies that event's driver utilisation
(e.g. the skidpad's 80 % lateral); default is the tyre limit.
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ros2_ws", "src", "fsdsim"))

import numpy as np  # noqa: E402

from fsdsim.perf import ggv as G  # noqa: E402
from fsdsim.perf import lapsim as L  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--speeds", type=float, nargs="*", help="[m/s] (default 5 .. 30 step 2.5)")
    ap.add_argument("--n-ay", type=int, default=13)
    ap.add_argument("--event", default="", help="apply this LapSim event's driver utilisation")
    ap.add_argument("--out", default=os.path.join("results", "ggv.csv"))
    ap.add_argument("--png", action="store_true")
    ap.add_argument("--lapsim")
    args = ap.parse_args()
    root = args.lapsim or L.find_lapsim_dir()
    veh = L.LapSim.load(root).vehicle(args.set) if root else L.PerfParams().with_changes(*args.set)
    veh = veh.for_event(args.event) if args.event else veh.copy(util_x=1.0, util_y=1.0)
    t = G.ggv(veh, args.speeds, args.n_ay)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    n = G.write_csv(t, args.out)
    print(f"{'v m/s':>6s} {'ay max':>7s} {'ax drive':>9s} {'ax brake':>9s}   (ax at ay = 0)")
    for i, v in enumerate(t["v"]):
        print(f"{v:6.1f} {t['ay'][i, -1]:7.2f} {t['ax_acc'][i, 0]:9.2f} {t['ax_dec'][i, 0]:9.2f}")
    print(f"{args.out}: {n} rows")
    if args.png:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.5, 6))
        cm = plt.get_cmap("viridis")
        for i, v in enumerate(t["v"]):
            c = cm(i / max(len(t["v"]) - 1, 1))
            ay = np.r_[-t["ay"][i][::-1], t["ay"][i]]
            ax.plot(ay, np.r_[t["ax_acc"][i][::-1], t["ax_acc"][i]], color=c, label=f"{v:.0f} m/s" if i % 2 == 0 else None)
            ax.plot(ay, np.r_[t["ax_dec"][i][::-1], t["ax_dec"][i]], color=c)
        ax.set_xlabel("lateral acceleration [m/s²]")
        ax.set_ylabel("longitudinal acceleration [m/s²]")
        ax.set_title("ggv map (LapSim model)")
        ax.grid(lw=0.3)
        ax.legend(fontsize=7, ncol=2)
        p = os.path.splitext(args.out)[0] + ".png"
        fig.savefig(p, dpi=110, bbox_inches="tight")
        print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
