#!/usr/bin/env python3
"""Export LapSim event runs as CSV files with the signal names of the Performance department's LionsLTS.

    python3 tools/export_run_csv.py --out results/lapsim_csv                      # all four events
    python3 tools/export_run_csv.py --events autocross --set m=300 "mu*1.05"
    python3 tools/export_run_csv.py --competition competition_fsa2025_real.m --energy-kwh 7

One CSV per event: time, cumulative distance, raceline x/y, velocity, reached ax, init ay, curvature, forces,
consumed energy, energy storage state, accumulator power. Details: fsdsim/perf/export.py.
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ros2_ws", "src", "fsdsim"))

from fsdsim.perf import lapsim as L  # noqa: E402
from fsdsim.perf.export import write_csv  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "lapsim_csv"), help="output folder")
    ap.add_argument("--events", nargs="*", help="events to run (default: all of the competition)")
    ap.add_argument("--set", nargs="*", default=[], help='vehicle changes, e.g. m=300 "mu*1.05"')
    ap.add_argument("--competition", help="competition file name in LR26_LapSim/config")
    ap.add_argument("--lapsim", help="LR26_LapSim folder")
    ap.add_argument("--energy-kwh", type=float, default=0.0, help="battery energy at the start of the lap [kWh] "
                    "(for 'energy storage state')")
    ap.add_argument("--name", default="lapsim", help="file name prefix")
    ap.add_argument("--eta-map", help="motor efficiency map (JSON from tools/log_correlate.py --save-map) instead of eta_motor")
    args = ap.parse_args()
    if args.eta_map:
        from fsdsim.perf import efficiency

        efficiency.set_active(args.eta_map)
    root = args.lapsim or L.find_lapsim_dir()
    if not root:
        print("LR26_LapSim folder not found (set FSG_LAPSIM_DIR or use --lapsim)", file=sys.stderr)
        return 2
    ls = L.LapSim.load(root, competition=args.competition)
    res = ls.run(args.set, args.events)
    for name, r in res["events"].items():
        path = os.path.join(args.out, f"{args.name}_{name}.csv")
        n = write_csv(r, path, args.energy_kwh * 3.6e6)
        print(f"{path}: {n} rows, {r['T']:.2f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
