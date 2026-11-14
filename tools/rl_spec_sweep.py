#!/usr/bin/env python3
"""Fast setup sweeps with the universal RL driver: many vehicle specifications x many tracks at once.

Every combination of the --vary values is driven by the universal throttle / brake policy on the
acceleration, the skidpad and N autocross layouts - all cars in parallel in the batched fsdsim model
(NumPy on the CPU, or PyTorch on a GPU). For each setup: event times, cones, failures, top speed,
peak lateral acceleration and how long the car was traction / motor / power / brake-grip limited,
next to LapSim's QSS prediction - so the Performance department sees what a parameter does to the
car driven at its limit, not only to the point-mass model.

    python3 tools/rl_spec_sweep.py --vary m=250:320:8
    python3 tools/rl_spec_sweep.py --vary c_z=3.2:5.0:5 --vary aero_bal_f=0.42,0.48,0.54 --tracks 8
    python3 tools/rl_spec_sweep.py --vary "mu*0.9:1.1:5" --set m-20 --compare      # also the reference law
    python3 tools/rl_spec_sweep.py --vary P_max=50000:80000:4 --backend torch --device cuda:0 --tracks 40

--vary NAME=a:b:n (n values from a to b), NAME=v1,v2,..., or NAME*a:b:n (factors on the baseline).
Names as everywhere: LapSim fields and aliases (m, mu, c_z, P_max, i_trans, aero_bal_f, h_cog, ...).
Outputs results/sweep_<time>/sweep.csv and sweep.md. Times are from the batched model; confirm a
chosen setup with the full simulator: tools/train_rl_universal.py --evaluate <policy> --set ...
"""

from __future__ import annotations

import argparse
import copy
import csv
import datetime as _dt
import itertools
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ros2_ws", "src", "fsdsim"))

CONFIG = os.path.join(ROOT, "ros2_ws", "src", "fsd_sim_bringup", "config")


def parse_vary(text: str) -> tuple[str, list[str], list[float]]:
    """-> (name, changes per value, numeric values)."""
    for op in ("=", "*"):
        if op in text:
            name, rhs = text.split(op, 1)
            break
    else:
        raise SystemExit(f"--vary {text}: use NAME=a:b:n, NAME=v1,v2 or NAME*a:b:n")
    name = name.strip()
    if ":" in rhs:
        a, b, n = rhs.split(":")
        vals = list(np.linspace(float(a), float(b), int(n)))
    else:
        vals = [float(v) for v in rhs.split(",")]
    return name, [f"{name}{op}{v:.6g}" for v in vals], vals


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vary", action="append", default=[], help="parameter values (repeat for a grid)")
    ap.add_argument("--set", nargs="*", default=[], help="base changes for every setup, e.g. m-20 c_w*1.05")
    ap.add_argument("--policy", default="universal", help="universal policy name or folder (default: best covering)")
    ap.add_argument("--tracks", type=int, default=6, help="autocross layouts (seeds 7, 11, 23, 42, ...)")
    ap.add_argument("--events", nargs="+", default=["acceleration", "skidpad", "autocross"])
    ap.add_argument("--compare", action="store_true", help="also drive the reference driver's law (teacher)")
    ap.add_argument("--qss", action="store_true", help="LapSim QSS times for the autocross layouts too (slower)")
    ap.add_argument("--backend", choices=["numpy", "torch"], default="numpy")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from fsdsim.config import apply_vehicle_overrides, load_all_configs  # noqa: PLC0415
    from fsdsim.perf import qss  # noqa: PLC0415
    from fsdsim.rl import registry  # noqa: PLC0415
    from fsdsim.rl.policy import MLPPolicy  # noqa: PLC0415
    from fsdsim.vec.backend import Backend  # noqa: PLC0415
    from fsdsim.vec.env import EnvConfig, VecLonEnv  # noqa: PLC0415
    from fsdsim.vec.spec import SpecRanges, coverage, spec_values  # noqa: PLC0415
    from fsdsim.vec.tracks import TrackBank  # noqa: PLC0415

    t0 = time.time()
    base_veh, _, _ = load_all_configs(args.config, vehicle_overrides=list(args.set))
    # setups: the grid of all --vary combinations (or just the base)
    axes = [parse_vary(v) for v in args.vary]
    combos = list(itertools.product(*[list(zip(ch, vals)) for _, ch, vals in axes])) if axes else [()]
    setups = []
    for combo in combos:
        veh = apply_vehicle_overrides(copy.deepcopy(base_veh), [c for c, _ in combo])
        setups.append({"changes": [c for c, _ in combo], "values": [v for _, v in combo], "veh": veh,
                       "spec": spec_values(veh)})

    # the policy
    if args.policy in ("universal", "auto"):
        folder, why = registry.resolve("universal", setups[0]["veh"])
        if folder is None:
            unis = [p for p in registry.list_policies() if p.get("universal")]
            if not unis:
                raise SystemExit("no universal policy in rl/policies (tools/train_rl_universal.py)")
            folder = unis[0]["folder"]
    else:
        folder, why = registry.resolve(args.policy)
        if folder is None:
            raise SystemExit(why)
    pol, meta = MLPPolicy.load(folder)
    if not meta.get("universal"):
        raise SystemExit(f"{folder} is not a universal (features v2) policy")
    ranges = SpecRanges.from_dict(meta.get("spec_ranges"))
    for s in setups:
        s["outside"] = coverage(s["spec"], meta["base_spec"], ranges)

    # tracks
    seeds = [7, 11, 23, 42, 57, 64, 77, 88, 91, 99][: args.tracks] if args.tracks <= 10 else \
        [7, 11, 23, 42, 57, 64, 77, 88, 91, 99] + list(range(200, 200 + args.tracks - 10))
    items = [(e, 0) for e in args.events if e in ("acceleration", "skidpad")]
    if "autocross" in args.events:
        items += [("autocross", s) for s in seeds]
    bank = TrackBank.build(args.config, items, args.set)
    T = len(items)
    B = Backend(args.backend, device=args.device) if args.backend == "torch" else Backend("numpy")
    rows = [s["spec"] for s in setups for _ in range(T)]
    tids = np.array([t for _ in setups for t in range(T)])
    env = VecLonEnv(B, bank, setups[0]["spec"], EnvConfig(n_envs=len(rows), randomise=False), teacher=args.compare)
    print(f"{len(setups)} setup(s) x {T} track(s) = {len(rows)} cars, policy {meta.get('name')} ({folder})", flush=True)
    if args.backend == "torch":
        import torch  # noqa: PLC0415

        Ws = [(torch.as_tensor(W, device=B.device, dtype=B.float), torch.as_tensor(b, device=B.device, dtype=B.float))
              for W, b in pol.layers]

        act = {"tanh": torch.tanh, "elu": torch.nn.functional.elu, "relu": torch.relu}[pol.activation]

        def policy_fn(o):
            h = o
            for i, (W, b) in enumerate(Ws):
                h = h @ W + b
                if i < len(Ws) - 1:
                    h = act(h)
            return (torch.tanh(h) if pol.output == "tanh" else h.clamp(-1, 1))[:, 0]
    else:
        def policy_fn(o):
            return pol.forward(o)
    res = env.evaluate(policy_fn, tids, rows)
    ref = env.evaluate("teacher", tids, rows) if args.compare else None

    # aggregate per setup
    out_rows = []
    for k, s in enumerate(setups):
        r = res[k * T:(k + 1) * T]
        q = ref[k * T:(k + 1) * T] if ref else None
        perf = s["veh"].perf()
        row = {**{name: float(f"{v:.6g}") for (name, _, _), v in zip(axes, s["values"])}, "changes": " ".join(args.set + s["changes"]),
               "in_range": "yes" if not s["outside"] else "NO: " + "; ".join(s["outside"])}
        for ev in ("acceleration", "skidpad", "autocross"):
            rr = [x for x in r if x["mission"] == ev]
            if not rr:
                continue
            times = [x["time"] for x in rr if x["time"] is not None and not x["failed"]]
            row[f"{ev}_s"] = round(float(np.mean(times)), 3) if len(times) == len(rr) else None
            row[f"{ev}_cones"] = int(sum(x["cones"] for x in rr))
            row[f"{ev}_failed"] = int(sum(x["failed"] for x in rr))
            if q:
                qq = [x for x in q if x["mission"] == ev]
                qt = [x["time"] for x in qq if x["time"] is not None and not x["failed"]]
                row[f"{ev}_ref_s"] = round(float(np.mean(qt)), 3) if len(qt) == len(qq) else None
            run_t = sum(x["t_run"] for x in rr) or 1.0
            if ev == "autocross":
                row["autocross_vmax_kmh"] = round(3.6 * max(x["v_max"] for x in rr), 1)
                row["autocross_aymax"] = round(max(x["ay_max"] for x in rr), 1)
                for lim in ("traction", "motor", "power", "brake_grip"):
                    row[f"autocross_{lim}_%"] = round(100 * sum(x[f"t_{lim}"] for x in rr) / run_t, 1)
            if ev == "acceleration":
                row["acceleration_vmax_kmh"] = round(3.6 * rr[0]["v_max"], 1)
                row["lapsim_acceleration_s"] = round(qss.acceleration(perf).time, 3)
                for lim in ("traction", "motor", "power"):
                    row[f"acceleration_{lim}_%"] = round(100 * sum(x[f"t_{lim}"] for x in rr) / run_t, 1)
            if ev == "skidpad":
                row["lapsim_skidpad_s"] = round(qss.skidpad(perf).time, 3)
                row["skidpad_aymax"] = round(max(x["ay_max"] for x in rr), 1)
            if ev == "autocross" and args.qss:
                tq = []
                for info in bank.infos:
                    if info.mission == "autocross":
                        tq.append(qss.run_path(perf, info.path, True, event="").time)
                row["lapsim_autocross_s"] = round(float(np.mean(tq)), 3)
        out_rows.append(row)

    out_dir = args.out or os.path.join(ROOT, "results", f"sweep_{_dt.datetime.now():%Y%m%d-%H%M%S}")
    os.makedirs(out_dir, exist_ok=True)
    keys = list(dict.fromkeys(k for r in out_rows for k in r))
    with open(os.path.join(out_dir, "sweep.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(out_rows)
    show = [k for k in keys if k not in ("changes",) and not k.endswith("_failed")]
    lines = ["| " + " | ".join(show) + " |", "|" + "---|" * len(show)]
    for r in out_rows:
        lines.append("| " + " | ".join("" if r.get(k) is None else str(r.get(k)) for k in show) + " |")
    sens = sensitivities(axes, out_rows)
    md = [f"# Setup sweep with the universal RL driver ({meta.get('name')})", "",
          f"Base: {' '.join(args.set) or 'LapSim baseline'}; tracks: {', '.join(f'{m} {s}' for m, s in items)}.",
          "Event times in s (autocross: mean over the layouts, one lap from a standing start); cones not added.", "",
          *lines, ""]
    if sens:
        md += ["## Sensitivities (least-squares slope)", "", *sens, ""]
    with open(os.path.join(out_dir, "sweep.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print("\n".join(lines))
    if sens:
        print("\n" + "\n".join(sens))
    print(f"\n{out_dir}  ({time.time() - t0:.0f} s)")


def sensitivities(axes, rows) -> list[str]:
    """d(time)/d(parameter) per event when exactly one parameter varies."""
    if len(axes) != 1:
        return []
    name = axes[0][0]
    x = np.array([r[name] for r in rows], float)
    out = []
    for ev in ("acceleration", "skidpad", "autocross"):
        y = [r.get(f"{ev}_s") for r in rows]
        if len(x) >= 2 and all(v is not None for v in y) and np.ptp(x) > 0:
            k = np.polyfit(x, np.array(y, float), 1)[0]
            out.append(f"- {ev}: {k:+.4g} s per unit of {name} ({k * np.ptp(x) / (len(x) - 1):+.3f} s per step)")
    return out


if __name__ == "__main__":
    main()
