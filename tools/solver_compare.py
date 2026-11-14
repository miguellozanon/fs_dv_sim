#!/usr/bin/env python3
"""Compare lap-time solvers on the LapSim events (built-in ones and plug-ins from lapsim/solvers/).

    python3 tools/solver_compare.py                                   # every solver, all four events
    python3 tools/solver_compare.py --solvers qss point_mass --set m=300
    python3 tools/solver_compare.py --competition competition_fsa2025_real.m

Writing a solver: lapsim/solvers/example_solver.py and fsdsim/perf/solvers.py (docs/PERFORMANCE.md).
"""

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ros2_ws", "src", "fsdsim"))

from fsdsim.perf import lapsim as L  # noqa: E402
from fsdsim.perf import solvers  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--solvers", nargs="*", help="names (default: all registered)")
    ap.add_argument("--set", nargs="*", default=[], help='vehicle changes, e.g. m=290 "mu*1.05"')
    ap.add_argument("--competition", help="competition file in LR26_LapSim/config")
    ap.add_argument("--lapsim", help="LR26_LapSim folder")
    args = ap.parse_args()
    root = args.lapsim or L.find_lapsim_dir()
    if not root:
        print("LR26_LapSim folder not found (set FSG_LAPSIM_DIR or use --lapsim)", file=sys.stderr)
        return 2
    ls = L.LapSim.load(root, competition=args.competition)
    known = solvers.names()
    use = args.solvers or list(known)
    ev = list(ls.comp.events)
    print(f"{ls.comp.name}")
    print(f"{'solver':18s}" + "".join(f"{e[:12]:>13s}" for e in ev) + f"{'points':>9s}{'run s':>7s}  description")
    for name in use:
        t0 = time.time()
        res = ls.run(args.set, solver=name)
        cells = "".join(f"{res['events'][e]['T']:13.3f}" for e in ev)
        print(f"{name:18s}{cells}{res['total']:9.2f}{time.time() - t0:7.1f}  {known.get(name, '')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
