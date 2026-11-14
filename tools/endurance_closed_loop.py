#!/usr/bin/env python3
"""Endurance in the closed loop with an energy budget: the power cap the driven car needs.

    python3 tools/endurance_closed_loop.py --budget-kwh 0.45                       # 10 laps of the generated circuit
    python3 tools/endurance_closed_loop.py --budget-kwh 3.4 --laps 10 --track-file tracks_out/fsa_endurance.csv
    python3 tools/endurance_closed_loop.py --budget-kwh 0.45 --line racing --eta-map lapsim/configs/eta_map_D1F58.json

tools/endurance_strategy.py answers the question with LapSim (quasi-steady state, the event's driver utilisation).
This tool drives the laps with the reference driver in the full simulator (dynamic car, actuator lags, the
car's own energy meter) and lowers the motor power (P_max) until the energy of the run fits the budget.
The budget is for the laps driven here: for 10 of the 16 FSA laps use 10/16 of the accumulator budget.
Prints, per power cap: lap time, total time, energy, cones; and LapSim's QSS lap time on the same line.
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, os.path.join(ROOT, "ros2_ws", "src", "fsdsim")]

import numpy as np  # noqa: E402

import closed_loop_eval as C  # noqa: E402
from fsdsim.perf import qss  # noqa: E402


def drive(args, cap_w):
    changes = list(args.set) + ([f"P_max={cap_w:.1f}"] if cap_w else [])
    if args.eta_map:
        changes.append(f"efficiency_map={args.eta_map}")
    sim, res, log, _ = C.run("trackdrive", args.config, mode=args.mode, verbose=False, changes=changes,
                             track_file=args.track_file or "", seed=args.seed, dt=1.0 / args.rate, line=args.line,
                             laps=args.laps)
    perf = sim.vehicle.perf
    t_qss = float("nan")
    if args.mode == "map":
        path = sim.line_info["xy"] if getattr(sim, "line_info", None) else sim.track.centerline
        util = 0.75 if sim.line_info else 0.80
        t_qss = qss.run_path(perf.copy(util_x=util, util_y=util, events={}), path, True, event="",
                             standing_start=False).time
    laps = res.lap_times or [float("nan")]
    return {"cap_kw": perf.P_max / 1e3, "state": res.state, "laps": res.laps, "lap": float(np.median(laps)),
            "total": float(np.sum(laps)), "E": sim.vehicle.energy_j / 3.6e6, "cones": res.cones_hit,
            "oc": res.off_course, "t_qss": t_qss}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budget-kwh", type=float, required=True, help="energy for the laps of this run [kWh]")
    ap.add_argument("--laps", type=int, default=10)
    ap.add_argument("--track-file", help="cone layout CSV (default: the generated circuit)")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--mode", choices=["map", "cones"], default="map")
    ap.add_argument("--line", choices=["center", "racing"], default="center")
    ap.add_argument("--rate", type=float, default=50.0)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--eta-map", help="motor efficiency map for the energy meter")
    ap.add_argument("--tol-kwh", type=float, default=0.005)
    ap.add_argument("--max-runs", type=int, default=8)
    ap.add_argument("--config", default=os.path.join(ROOT, "ros2_ws", "src", "fsd_sim_bringup", "config"))
    args = ap.parse_args()

    rows = []

    def run(cap):
        r = drive(args, cap)
        rows.append(r)
        print(f"  P_max {r['cap_kw']:6.2f} kW: lap {r['lap']:7.3f} s (QSS {r['t_qss']:.2f}), {r['laps']} laps {r['total']:8.2f} s, "
              f"{r['E']:.3f} kWh, cones {r['cones']}, off-course {r['oc']}, {r['state']}", flush=True)
        return r

    print(f"budget {args.budget_kwh:g} kWh for {args.laps} laps, {args.mode} mode, line {args.line}")
    top = run(0.0)
    if top["E"] <= args.budget_kwh:
        print("no power cap needed: the full motor power fits the budget")
        return 0
    lo = run(5000.0)
    if lo["E"] > args.budget_kwh:
        print("not feasible: even 5 kW uses more than the budget")
        return 1
    a, b = lo, top                                        # a fits, b does not
    while len(rows) < args.max_runs and b["cap_kw"] - a["cap_kw"] > 0.1 and args.budget_kwh - a["E"] > args.tol_kwh:
        # energy rises smoothly with the cap: regula falsi, kept inside the bracket
        f = (args.budget_kwh - a["E"]) / max(b["E"] - a["E"], 1e-9)
        cap = a["cap_kw"] + min(max(f, 0.08), 0.92) * (b["cap_kw"] - a["cap_kw"])
        r = run(cap * 1e3)
        if r["E"] <= args.budget_kwh:
            a = r
        else:
            b = r
    print(f"\npower cap {a['cap_kw']:.2f} kW: {a['E']:.3f} kWh of {args.budget_kwh:g} kWh, lap {a['lap']:.3f} s, "
          f"{args.laps} laps in {a['total']:.2f} s ({a['total'] - top['total']:+.2f} s against full power, "
          f"which needs {top['E']:.3f} kWh)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
