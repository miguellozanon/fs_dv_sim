#!/usr/bin/env python3
"""Train ONE throttle / brake policy for every vehicle specification in a range (universal policy).

Reinforcement learning (PPO) with thousands of LR26 variants in parallel: the batched fsdsim car
model (LapSim forces) on a GPU with PyTorch, or on the CPU with NumPy. The policy is told the
specification of the car it drives, so the Performance department can change parameters in the
dashboard and drive the new car at once - no retraining per setup.

    # GPU (PyTorch with CUDA; in WSL: pip install torch)
    python3 tools/train_rl_universal.py --backend torch --device cuda:0 --num-envs 4096 --iterations 1500
    # CPU (NumPy only; slower: about 8000 car-steps/s per core)
    python3 tools/train_rl_universal.py --iterations 2000
    # Isaac Lab + RSL-RL: see isaaclab/fsd_rl/README.md

    # official runs in fsdsim (start, timing, cones) for any setup in the range, RL vs reference driver
    python3 tools/train_rl_universal.py --evaluate rl/policies/universal --set m=260 "c_z*1.1"

Output: rl/policies/<name>/policy.npz + policy.json (features v2, spec ranges, training curve,
validation). The simulator picks it for every setup inside the ranges (longitudinal:=rl).
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
    ap.add_argument("--backend", choices=["numpy", "torch"], default="numpy")
    ap.add_argument("--device", default="cuda:0", help="torch device (cuda:0, cpu)")
    ap.add_argument("--num-envs", type=int, default=0, help="cars in parallel (default: 4096 torch, 256 numpy)")
    ap.add_argument("--iterations", type=int, default=0, help="PPO iterations (default: 1500 torch, 2000 numpy)")
    ap.add_argument("--steps", type=int, default=32, help="steps per car and iteration")
    ap.add_argument("--eval-every", type=int, default=0)
    ap.add_argument("--lr", type=float, default=0.0, help="learning rate cap (default 5e-4; lower to fine-tune)")
    ap.add_argument("--name", default="universal")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--out", default="", help="output folder (default rl/policies/<name>)")
    ap.add_argument("--init", default="", help="continue from a policy folder (agent.npz / agent_torch.pt)")
    ap.add_argument("--ranges", default="", help="JSON file with spec ranges (see fsdsim/vec/spec.py SpecRanges)")
    ap.add_argument("--no-realism", action="store_true", help="no hidden grip / lag / latency / sensor noise")
    ap.add_argument("--physics-hz", type=float, default=500.0, help="car model rate (fsdsim: 500)")
    ap.add_argument("--compile", action="store_true", help="torch.compile the physics step (PyTorch 2)")
    ap.add_argument("--progress", default="", help="write progress JSON here (dashboard); a .stop file stops")
    ap.add_argument("--evaluate", default="", help="only evaluate this policy folder in fsdsim (official runs)")
    ap.add_argument("--set", nargs="*", default=[], help="vehicle changes for --evaluate, e.g. m=260 c_z*1.1")
    args = ap.parse_args()

    if args.evaluate:
        evaluate(args.evaluate, args.config, args.set)
        return

    from fsdsim.vec.env import EnvConfig  # noqa: PLC0415
    from fsdsim.vec.ppo import PPOConfig  # noqa: PLC0415
    from fsdsim.vec.spec import Realism, SpecRanges  # noqa: PLC0415

    if not re.fullmatch(r"[\w.+-]{1,60}", args.name):
        sys.exit("--name: letters, digits and . _ + - only")
    out = args.out or os.path.join(registry.policies_dir(), args.name)
    torch_ = args.backend == "torch"
    ecfg = EnvConfig(physics_hz=args.physics_hz)
    if args.ranges:
        with open(args.ranges, encoding="utf-8") as f:
            ecfg.ranges = SpecRanges.from_dict(json.load(f))
    if args.no_realism:
        ecfg.realism = Realism.off()
    pcfg = PPOConfig(n_steps=args.steps)
    if args.lr > 0:
        pcfg.lr = pcfg.lr_max = args.lr
    if torch_:
        pcfg.minibatches = 8
        pcfg.bc_steps = 32
    prog_path = args.progress
    stop_file = (prog_path + ".stop") if prog_path else ""

    def progress(d):
        if prog_path:
            with open(prog_path + ".tmp", "w", encoding="utf-8") as f:
                json.dump({**d, "name": args.name, "t": time.time()}, f)
            os.replace(prog_path + ".tmp", prog_path)

    common = dict(out_dir=out, name=args.name, env_cfg=ecfg, ppo_cfg=pcfg, seed=args.seed, init=args.init,
                  log=lambda m: print(m, flush=True), progress=progress,
                  should_stop=lambda: bool(stop_file) and os.path.exists(stop_file))
    if torch_:
        from fsdsim.vec.ppo_torch import train_torch  # noqa: PLC0415

        meta = train_torch(args.config, iterations=args.iterations or 1500, n_envs=args.num_envs or 4096,
                           device=args.device, eval_every=args.eval_every or 50, compile_physics=args.compile,
                           **common)
        if meta.get("aborted"):
            print(f"policy (incomplete): {out}")
            sys.exit(3)
    else:
        from fsdsim.vec.train import train  # noqa: PLC0415

        train(args.config, iterations=args.iterations or 2000, n_envs=args.num_envs or 256,
              eval_every=args.eval_every or 25, **common)
    print(f"policy: {out}")


def evaluate(folder: str, config: str, changes) -> None:
    """Complete missions in fsdsim (the simulator itself, not the batched model): RL vs reference."""
    from fsdsim.config import load_all_configs  # noqa: PLC0415
    from fsdsim.rl.env import ScenarioCache, full_run  # noqa: PLC0415
    from fsdsim.rl.policy import MLPPolicy  # noqa: PLC0415
    from fsdsim.vec.spec import SpecRanges, coverage, spec_values  # noqa: PLC0415

    pol, meta = MLPPolicy.load(folder)
    veh, _, _ = load_all_configs(config, vehicle_overrides=list(changes))
    if meta.get("universal"):
        why = coverage(spec_values(veh), meta["base_spec"], SpecRanges.from_dict(meta["spec_ranges"]))
        print(f"policy {meta.get('name')} (universal, trained {meta.get('trained')}); setup: "
              f"{' '.join(changes) or 'LapSim baseline'} - " + ("inside the training range" if not why else
                                                                 "OUTSIDE the training range: " + "; ".join(why)))
    cache = ScenarioCache(config, list(changes))
    print(f"{'mission':12s} {'seed':>4s}  {'reference':>22s}  {'RL':>22s}")
    for mission, seeds in (("acceleration", (0,)), ("skidpad", (0,)), ("autocross", (7, 11, 23, 42))):
        for sd in seeds:
            ref, rl = full_run(None, cache, mission, sd), full_run(pol, cache, mission, sd)
            print(f"{mission:12s} {sd:4d}  {_fmt(ref):>22s}  {_fmt(rl):>22s}", flush=True)


def _fmt(r):
    if r["raw_time"] is None:
        return r["state"].lower()
    pen = f" +{r['cones']}c {r['off_course']}oc" if r["cones"] or r["off_course"] else ""
    return f"{r['raw_time']:.3f} s{pen}"


if __name__ == "__main__":
    main()
