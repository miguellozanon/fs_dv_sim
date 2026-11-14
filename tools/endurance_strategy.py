#!/usr/bin/env python3
"""Endurance strategy: how much power can the car use for a given battery energy?

    python3 tools/endurance_strategy.py --budget-kwh 5.9                   # the power cap that fits 5.9 kWh
    python3 tools/endurance_strategy.py --budget-kwh 5.5 --set m=290 i_trans=15
    python3 tools/endurance_strategy.py --caps 12 16 20 24 28 --competition competition_fsa2025_real.m
    python3 tools/endurance_strategy.py --budget-kwh 5.9 --out results/endurance.csv

The documentation's endurance lap runs at a constant ~18 kW. The Performance department's LionsLTS solver
finds such a limit from the battery energy: it lowers the motor power until the energy of the lap fits.
This does the same with LapSim (fsdsim.perf.strategy): a table of lap time, total time, energy, event points
and efficiency factor (T^2 * E) against the power cap, and the cap for the budget.
Same numbers as MATLAB: analysis/lts_endurance_power.m.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ros2_ws", "src", "fsdsim"))

from fsdsim.perf import lapsim as L  # noqa: E402
from fsdsim.perf import strategy as S  # noqa: E402

COLS = [("cap_kw", "cap kW", "{:6.2f}"), ("t_lap", "lap s", "{:7.2f}"), ("T", "total s", "{:8.1f}"),
        ("E_lap_kwh", "E lap kWh", "{:9.4f}"), ("E_total_kwh", "E total kWh", "{:11.3f}"),
        ("points", "points", "{:7.2f}"), ("EF", "EF s2 kWh", "{:12.0f}")]


def table(rows: list[dict]) -> str:
    head = " | ".join(h for _, h, _ in COLS)
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(" | ".join(f.format(r[k]) for k, _, f in COLS))
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budget-kwh", type=float, help="accumulator energy for all endurance laps [kWh]")
    ap.add_argument("--caps", type=float, nargs="*", help="power caps to tabulate [kW] (default 10..40 step 5)")
    ap.add_argument("--set", nargs="*", default=[], help='vehicle changes, e.g. m=290 "mu*1.05"')
    ap.add_argument("--competition", help="competition file name in LR26_LapSim/config (default: the first one)")
    ap.add_argument("--lapsim", help="LR26_LapSim folder")
    ap.add_argument("--out", help="write the table as CSV")
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
    veh = ls.vehicle(args.set)
    ev = ls.competition(args.set).events["endurance"]
    print(f"{ls.comp.name}: {ev.track.name}, {ev.laps} laps of {ev.track.L:.0f} m, P_max {veh.P_max / 1e3:.0f} kW")
    caps = args.caps or [10, 15, 20, 25, 30, 35, 40]
    rows = S.sweep_power(veh, ev, caps)
    full = S.run_with_cap(veh, ev, veh.P_max)
    rows.append({**full, "cap_kw": veh.P_max / 1e3})
    if args.budget_kwh:
        fit = S.power_for_energy(veh, ev, args.budget_kwh)
        rows.append(fit)
    rows.sort(key=lambda r: r["cap_kw"])
    print(table(rows))
    print(f"\nWithout a cap ({veh.P_max / 1e3:.0f} kW) the {ev.laps} laps need {full['E_total_kwh']:.2f} kWh.")
    if args.budget_kwh:
        if not fit["feasible"]:
            print(f"Budget {args.budget_kwh:g} kWh: not even {fit['cap_kw']:.0f} kW fits ({fit['E_total_kwh']:.2f} kWh).")
        elif not fit["limited"]:
            print(f"Budget {args.budget_kwh:g} kWh: no cap needed.")
        else:
            print(f"Budget {args.budget_kwh:g} kWh: power cap {fit['cap_kw']:.2f} kW, lap {fit['t_lap']:.2f} s, "
                  f"total {fit['T']:.1f} s, {fit['points']:.1f} endurance points.")
    print("EF = T^2 * E: only meaningful with the efficiency formula of the rules (FS Rules 2026 D9.4).")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([k for k, _, _ in COLS])
            for r in rows:
                w.writerow([f"{r[k]:.6g}" for k, _, _ in COLS])
        print(f"written {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
