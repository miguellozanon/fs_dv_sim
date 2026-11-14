#!/usr/bin/env python3
"""Look into a car log (.mf4, CAN data from the logger) and cut signals out as CSV. No extra packages.

    python3 tools/mf4_summary.py run.mf4                          # every signal: samples, rate, unit, range
    python3 tools/mf4_summary.py run.mf4 --search speed torque    # only names containing all of these words
    python3 tools/mf4_summary.py run.mf4 --bus CAN3 --search DCUA
    python3 tools/mf4_summary.py run.mf4 --csv out.csv --signals act_Speed_R_DCUA act_Torque_R_DCUA PDB_Pedal_Throttle \
            --rate 20 --from 100 --to 160                          # selected signals on one time base

"Not available" values (all bits set in an unsigned signal) are left out. Reader: fsdsim/logdata/mdf4.py.
The signals of a LapSim / simulator run can be exported with the same column style by tools/export_run_csv.py.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ros2_ws", "src", "fsdsim"))

import numpy as np  # noqa: E402

from fsdsim.logdata import MDF4, resample  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file")
    ap.add_argument("--search", nargs="*", default=[], help="words that must all be in the channel name (any case)")
    ap.add_argument("--bus", help="only groups whose name contains this (e.g. CAN1, CAN3)")
    ap.add_argument("--csv", help="write the --signals as CSV on one time base")
    ap.add_argument("--signals", nargs="*", default=[], help="channel names for --csv (a name may be 'NAME@CAN3' to pick a bus)")
    ap.add_argument("--rate", type=float, default=20.0, help="CSV sample rate [Hz]")
    ap.add_argument("--from", dest="t0", type=float, help="start time [s]")
    ap.add_argument("--to", dest="t1", type=float, help="end time [s]")
    args = ap.parse_args()

    f = MDF4(args.file)
    if args.csv:
        if not args.signals:
            print("--csv needs --signals", file=sys.stderr)
            return 2
        series = {}
        for spec in args.signals:
            name, _, bus = spec.partition("@")
            try:
                series[spec] = f.signal(name, bus or args.bus)
            except KeyError as e:
                print(e, file=sys.stderr)
                return 2
        grid, cols = resample(series, args.rate, args.t0, args.t1)
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["time"] + list(cols))
            for i in range(len(grid)):
                w.writerow([f"{grid[i]:.4f}"] + ["" if np.isnan(cols[k][i]) else f"{cols[k][i]:.6g}" for k in cols])
        print(f"written {args.csv}: {len(grid)} rows, {len(cols)} signals, {args.rate:g} Hz")
        return 0

    words = [w.lower() for w in args.search]
    n = 0
    print(f"{os.path.basename(args.file)}: MDF {f.version}, {len(f.groups)} groups")
    print(f"{'group':34s} {'channel':34s} {'unit':6s} {'n':>7s} {'Hz':>6s} {'p1':>10s} {'median':>10s} {'p99':>10s} {'invalid':>7s}")
    for g in f.groups:
        if args.bus and args.bus not in g.name:
            continue
        t = f.time(g.index)
        for c in g.channels:
            if c.cn_type == 2 or (words and not all(w in c.name.lower() for w in words)):
                continue
            x = f.read(g.index, c, mask_invalid=True)
            if args.t0 is not None or args.t1 is not None:
                m = np.ones(len(x), bool)
                if t is not None:
                    m &= (t >= (args.t0 if args.t0 is not None else -np.inf)) & (t <= (args.t1 if args.t1 is not None else np.inf))
                x = x[m]
                tt = t[m] if t is not None else None
            else:
                tt = t
            ok = x[~np.isnan(x.astype(np.float64))] if len(x) else x
            hz = len(x) / (tt[-1] - tt[0]) if tt is not None and len(tt) > 1 and tt[-1] > tt[0] else float("nan")
            if len(ok):
                p1, p50, p99 = np.percentile(ok.astype(np.float64), [1, 50, 99])
                stats = f"{p1:10.4g} {p50:10.4g} {p99:10.4g}"
            else:
                stats = f"{'-':>10s} {'-':>10s} {'-':>10s}"
            print(f"{g.name[:34]:34s} {c.name[:34]:34s} {c.unit[:6]:6s} {len(x):7d} {hz:6.1f} {stats} {len(x) - len(ok):7d}")
            n += 1
    print(f"{n} signals")
    return 0


if __name__ == "__main__":
    sys.exit(main())
