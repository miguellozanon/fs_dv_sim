#!/usr/bin/env python3
"""Train the RL throttle / brake policy for a vehicle specification (no ROS, no GPU, NumPy only).

    python3 tools/train_rl.py                                   # LR26 as in lapsim/LR26_LapSim
    python3 tools/train_rl.py --set m=270 "mu*1.05" --name light_270
    python3 tools/train_rl.py --generations 120 --workers 12    # longer, on more cores
    python3 tools/train_rl.py --init rl/policies/lr26_baseline --name lr26_v2   # continue training
    python3 tools/train_rl.py --evaluate rl/policies/lr26_baseline   # compare a policy with the reference

The policy is saved in rl/policies/<name>/ (policy.npz, policy.json with the vehicle spec, training
curve and the RL-vs-reference evaluation). The simulator uses it with
    ros2 launch fsd_sim_bringup isaac.launch.py longitudinal:=rl
(it picks the policy trained for the running vehicle spec), or from the dashboard's Vehicle setup.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ros2_ws", "src", "fsdsim"))

from fsdsim.rl import registry  # noqa: E402

CONFIG = os.path.join(ROOT, "ros2_ws", "src", "fsd_sim_bringup", "config")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", nargs="*", default=[], help="vehicle changes, e.g. m=270 mu*1.05 (LapSim names)")
    ap.add_argument("--name", default="", help="policy name (default: from the changes)")
    ap.add_argument("--generations", type=int, default=60)
    ap.add_argument("--pairs", type=int, default=8, help="antithetic pairs per generation (2x rollouts)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--sigma", type=float, default=0.02, help="exploration noise on the output layer")
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--tracks", type=int, default=3, help="autocross layouts per generation (more: less noise)")
    ap.add_argument("--missions", nargs="+", default=["acceleration", "skidpad", "autocross"],
                    choices=["acceleration", "skidpad", "autocross"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--out", default="", help="output folder (default rl/policies/<name>)")
    ap.add_argument("--progress", default="", help="write progress JSON here (used by the dashboard)")
    ap.add_argument("--init", default="", help="continue training from this policy folder (skips imitation)")
    ap.add_argument("--evaluate", default="", help="only evaluate this policy folder against the reference driver")
    args = ap.parse_args()

    if args.evaluate:
        evaluate(args.evaluate, args.config, args.set)
        return

    from fsdsim.rl.es import train  # noqa: PLC0415

    name = args.name or registry.default_name(args.set)
    if not re.fullmatch(r"[\w.+-]{1,60}", name):
        sys.exit("--name: letters, digits and . _ + - only")
    out = args.out or os.path.join(registry.policies_dir(), name)
    prog_path = args.progress

    def progress(d):
        if prog_path:
            tmp = prog_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({**d, "name": name, "changes": args.set, "t": time.time()}, f)
            os.replace(tmp, prog_path)

    stop_file = (prog_path + ".stop") if prog_path else ""
    meta = train(args.config, args.set, out, generations=args.generations, pairs=args.pairs, sigma=args.sigma,
                 lr=args.lr, workers=args.workers, missions=tuple(args.missions), seed=args.seed, name=name,
                 tracks_per_gen=args.tracks, init=args.init,
                 log=lambda m: print(m, flush=True), progress=progress,
                 should_stop=(lambda: bool(stop_file) and os.path.exists(stop_file)))
    v = meta["validation"]
    print(f"\nvalidation fitness: reference driver {v['reference']:.1f} | imitation {v['imitation']:.1f} | "
          f"RL {v['rl']:.1f} (generation {v['best_generation']})")
    print(f"policy: {out}")


def evaluate(folder: str, config: str, changes) -> None:
    from fsdsim.rl.env import ScenarioCache, full_run  # noqa: PLC0415
    from fsdsim.rl.policy import MLPPolicy  # noqa: PLC0415

    pol, meta = MLPPolicy.load(folder)
    cache = ScenarioCache(config, changes or meta.get("changes", []))
    print(f"policy {meta.get('name')} (trained {meta.get('trained')}, vehicle changes: "
          f"{' '.join(meta.get('changes', [])) or 'none'})")
    print(f"{'mission':12s} {'seed':>4s}  {'reference':>22s}  {'RL':>22s}")
    for mission, seeds in (("acceleration", (0,)), ("skidpad", (0,)), ("autocross", (7, 11, 23, 42))):
        for sd in seeds:
            ref, rl = full_run(None, cache, mission, sd), full_run(pol, cache, mission, sd)
            print(f"{mission:12s} {sd:4d}  {_fmt(ref):>22s}  {_fmt(rl):>22s}")


def _fmt(r):
    if r["raw_time"] is None:
        return r["state"].lower()
    pen = f" +{r['cones']}c {r['off_course']}oc" if r["cones"] or r["off_course"] else ""
    return f"{r['raw_time']:.3f} s{pen}"


if __name__ == "__main__":
    main()
