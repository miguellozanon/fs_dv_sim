#!/usr/bin/env python3
"""Compare a car log (.mf4) with the LapSim powertrain: efficiency, loss model, roll-out, effect on endurance.

    python3 tools/log_correlate.py path/to/log.mf4
    python3 tools/log_correlate.py log.mf4 --lapsim-effect --csv out/eff.csv

What it can and cannot tell (details in docs/LIONSLTS.md):
* motor + inverter efficiency (shaft power / DC power) per torque and speed bin, against LapSim's constant
  eta_motor; assumes act_Torque is shaft torque in Nm and act_Power the DC power in kW (to confirm);
* a roll-out check: a log whose zero-torque deceleration is far above a car's was not recorded with the car
  on the road, so speed and acceleration from the motor speed must not be used to fit vehicle parameters;
* --lapsim-effect: what the measured efficiency does to the endurance power cap (energy budget) and points.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ros2_ws", "src", "fsdsim"))

import numpy as np  # noqa: E402

from fsdsim.logdata import powertrain as PT  # noqa: E402
from fsdsim.perf import lapsim as L  # noqa: E402

ROLLOUT_CAR_MAX = 0.6        # m/s^2: above this the wheels were not rolling freely on the road


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file")
    ap.add_argument("--rate", type=float, default=20.0, help="resampling rate [Hz]")
    ap.add_argument("--lapsim", help="LR26_LapSim folder")
    ap.add_argument("--lapsim-effect", action="store_true", help="endurance cap / points with the measured efficiency")
    ap.add_argument("--budget-kwh", type=float, default=5.5)
    ap.add_argument("--csv", help="write the efficiency map as CSV")
    ap.add_argument("--save-map", help="write the fitted efficiency map (JSON) for LapSim and the simulator, "
                                       "e.g. lapsim/configs/eta_map_D1F58.json (see docs/PERFORMANCE.md)")
    ap.add_argument("--motors-per-axle", type=float, default=2.0,
                    help="motors that share LapSim's T_motor of one driven axle (LR26 rear: DCUA + DCUB = 2)")
    args = ap.parse_args()

    root = args.lapsim or L.find_lapsim_dir()
    ls = L.LapSim.load(root, competition="competition_fsa2025_real.m") if root else None
    veh = ls.veh if ls else L.PerfParams()
    d = PT.load(args.file, args.rate)
    s = PT.motor_samples(d)
    if len(s["eff"]) < 50:
        print("too few driving samples (torque > 0.3 Nm and DC power > 0.5 kW) in this log", file=sys.stderr)
        return 2
    print(f"{os.path.basename(args.file)}: {d['t'][-1]:.0f} s, {len(s['eff'])} motor samples, "
          f"torque <= {s['T'].max():.1f} Nm, speed <= {s['n'].max():.0f} rpm")
    eff = s["eff"]
    print(f"efficiency motor + inverter (shaft / DC): median {np.median(eff):.3f}, p10 {np.percentile(eff, 10):.3f}, "
          f"p90 {np.percentile(eff, 90):.3f}; LapSim eta_motor = {veh.eta_motor:.2f}")
    te, ne, tab, cnt = PT.efficiency_map(s)
    print("\nmedian efficiency by torque [Nm] (rows) and motor speed [krpm] (columns):")
    print("            " + "".join(f"{ne[j] / 1e3:>5.0f}-{min(ne[j + 1], 20000) / 1e3:<4.0f}" for j in range(len(ne) - 1)))
    for i in range(len(te) - 1):
        cells = "".join(f"{tab[i, j]:>10.3f}" if np.isfinite(tab[i, j]) else f"{'-':>10}" for j in range(len(ne) - 1))
        print(f"{te[i]:>5.1f}-{min(te[i + 1], 99):<5.1f}" + cells)
    fit = PT.loss_fit(s)
    k = fit["k"]
    print(f"\nloss per motor [kW] = {k[0]:.3f} + {k[1]:.5f} T^2 + {k[2]:.4f} n + {k[3]:.5f} n^2   (n in krpm; rms {fit['rms_kw']:.2f} kW;"
          f" valid up to {fit['T_max']:.0f} Nm and {fit['n_max']:.0f} rpm only)")
    ro = PT.rollout(d, veh.i_trans, veh.circ_ref)
    if np.isfinite(ro["decel_median"]):
        verdict = "consistent with a car on the road" if ro["decel_median"] <= ROLLOUT_CAR_MAX else \
            "far above a rolling car: wheels not freely on the road (stand / dyno / other load); do not fit vehicle dynamics to this log"
        print(f"\nroll-out (zero torque, no brake, {ro['samples']} samples): median deceleration {ro['decel_median']:.2f} m/s^2 -> {verdict}")
    else:
        print("\nroll-out: no zero-torque, no-brake stretch long enough")
    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["torque_from_Nm", "torque_to_Nm"] + [f"eff_{ne[j]:.0f}-{ne[j + 1]:.0f}rpm" for j in range(len(ne) - 1)])
            for i in range(len(te) - 1):
                w.writerow([te[i], te[i + 1]] + ["" if not np.isfinite(tab[i, j]) else f"{tab[i, j]:.4f}" for j in range(len(ne) - 1)])
        print(f"map: {args.csv}")
    if args.save_map:
        from fsdsim.perf.efficiency import EfficiencyMap

        t_max = max(veh.T_motor / args.motors_per_axle, fit["T_max"])
        emap = EfficiencyMap.from_loss_fit(k, t_max=t_max, n_max=veh.n_max, source=os.path.basename(args.file),
                                           valid={"T_max_Nm": fit["T_max"], "n_max_rpm": fit["n_max"], "rms_kw": fit["rms_kw"]},
                                           motors_per_set=args.motors_per_axle)
        emap.save(args.save_map)
        full = emap.eta(veh.T_motor, 0.5 * veh.n_max)
        print(f"efficiency map: {args.save_map} (per motor, {args.motors_per_axle:g} motors per axle; measured up to "
              f"{fit['T_max']:.0f} Nm / {fit['n_max']:.0f} rpm, beyond that the loss model is extrapolated: "
              f"{full:.3f} at full torque and half speed)")
        if ls:
            from fsdsim.perf import efficiency as EF, strategy as S2

            ev = ls.comp.events["endurance"]
            base = S2.power_for_energy(veh, ev, args.budget_kwh)
            EF.set_active(emap)
            new = S2.power_for_energy(veh, ev, args.budget_kwh)
            EF.set_active(None)
            print(f"  endurance, {args.budget_kwh} kWh: constant eta {veh.eta_motor:.2f}: cap {base['cap_kw']:.2f} kW, "
                  f"{base['T']:.0f} s, {base['points']:.1f} points | with the map: cap {new['cap_kw']:.2f} kW, "
                  f"{new['T']:.0f} s, {new['points']:.1f} points")
    if args.lapsim_effect and ls:
        from fsdsim.perf import strategy as S

        ev = ls.comp.events["endurance"]
        med = float(np.median(eff))
        print(f"\nendurance on the real FSA track, budget {args.budget_kwh} kWh (this log's torque range is low: indicative only):")
        for name, e in (("LapSim", veh.eta_motor), ("measured median", med)):
            r = S.power_for_energy(veh.copy(eta_motor=e), ev, args.budget_kwh)
            print(f"  eta_motor {e:.3f} ({name}): power cap {r['cap_kw']:.2f} kW, endurance {r['T']:.0f} s, {r['points']:.1f} points")
    return 0


if __name__ == "__main__":
    sys.exit(main())
