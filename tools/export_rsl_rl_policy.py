#!/usr/bin/env python3
"""Turn an RSL-RL checkpoint (Isaac Lab task FSD-Longitudinal-Universal-v0) into a simulator policy.

    python3 tools/export_rsl_rl_policy.py logs/rsl_rl/fsd_longitudinal_universal/<run>/model_1499.pt \\
        --name universal_isaaclab

Writes rl/policies/<name>/policy.npz + policy.json (features v2, universal) with the observation
normalisation folded into the first layer, checks the NumPy network against the checkpoint's torch
network on random inputs, and runs the validation suite (timed runs, batched fsdsim model, CPU).
Needs PyTorch (Isaac Lab's Python has it: isaaclab.bat -p tools/export_rsl_rl_policy.py ...).
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ros2_ws", "src", "fsdsim"))

from fsdsim.rl import registry  # noqa: E402
from fsdsim.rl.policy import MLPPolicy  # noqa: E402

CONFIG = os.path.join(ROOT, "ros2_ws", "src", "fsd_sim_bringup", "config")


def actor_layers(sd: dict):
    """(normaliser mean, std, eps) or None, and [(W, b)] of the actor's linear layers in order."""
    lin = []
    for k, v in sd.items():
        m = re.search(r"(?:^|\.)(\d+)\.weight$", k)
        if m and v.dim() == 2 and "normalizer" not in k:
            prefix = k[: -len("weight")]
            lin.append((k.rsplit(".", 2)[0], int(m.group(1)), v, sd[prefix + "bias"]))
    lin.sort(key=lambda t: (t[0], t[1]))
    layers = [(w.detach().double().cpu().numpy().T, b.detach().double().cpu().numpy()) for _, _, w, b in lin]
    norm = None
    mk = [k for k in sd if k.endswith("obs_normalizer._mean")]
    if mk:
        p = mk[0][: -len("_mean")]
        norm = (sd[p + "_mean"].double().cpu().numpy().ravel(), sd[p + "_std"].double().cpu().numpy().ravel(), 1e-2)
    return norm, layers


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("--name", default="universal_isaaclab")
    ap.add_argument("--activation", default="tanh", help="hidden activation of the actor (agent cfg)")
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--no-eval", action="store_true")
    args = ap.parse_args()

    import torch  # noqa: PLC0415

    from fsdsim.vec.env import EnvConfig  # noqa: PLC0415
    from fsdsim.vec.features import N_FEATURES  # noqa: PLC0415
    from fsdsim.vec.train import EvalSuite, baseline, make_meta, save, summarize_eval  # noqa: PLC0415

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = ck.get("actor_state_dict") or ck.get("model_state_dict")
    if sd is None:
        sys.exit(f"no actor_state_dict in {args.checkpoint} (keys: {list(ck)})")
    if "model_state_dict" in ck and "actor_state_dict" not in ck:      # rsl-rl < 4: one ActorCritic module
        sd = {k[len("actor."):] if k.startswith("actor.") else k: v for k, v in sd.items()
              if k.startswith("actor") and not k.startswith("actor_obs")} | \
             {k.replace("actor_obs_normalizer", "obs_normalizer"): v for k, v in sd.items() if "actor_obs_normalizer" in k}
    norm, layers = actor_layers(sd)
    if not layers or layers[0][0].shape[0] != N_FEATURES or layers[-1][0].shape[1] != 1:
        sys.exit(f"unexpected actor layout {[w.shape for w, _ in layers]} (expected {N_FEATURES} inputs, 1 output)")
    W0, b0 = layers[0]
    if norm is not None:
        mean, std, eps = norm
        s = std + eps
        W0, b0 = W0 / s[:, None], b0 - (mean / s) @ W0
    pol = MLPPolicy(N_FEATURES, tuple(w.shape[1] for w, _ in layers[:-1]))
    pol.layers = [[W0, b0]] + [[w, b] for w, b in layers[1:]]
    pol.activation, pol.output = args.activation, "clip"
    pol.features_version, pol.universal = 2, True

    # check against torch on random inputs
    x = np.random.default_rng(0).normal(size=(256, N_FEATURES)) * 2.0
    act = {"tanh": torch.tanh, "elu": torch.nn.functional.elu, "relu": torch.relu}[args.activation]
    h = torch.as_tensor(x)
    if norm is not None:
        h = (h - torch.as_tensor(norm[0])) / (torch.as_tensor(norm[1]) + norm[2])
    for i, (w, b) in enumerate(layers):
        h = h @ torch.as_tensor(w) + torch.as_tensor(b)
        if i < len(layers) - 1:
            h = act(h)
    ref = h.clamp(-1, 1).numpy()[:, 0]
    err = float(np.abs(pol.forward(x) - ref).max())
    print(f"export check: max |numpy - torch| = {err:.2e}")
    if err > 1e-6:
        sys.exit("export mismatch")

    # the run's task settings (Isaac Lab writes <run>/params/env.yaml): baseline changes, ranges, realism
    from fsdsim.vec.spec import Realism, SpecRanges  # noqa: PLC0415

    ecfg = EnvConfig()
    changes = []
    env_yaml = os.path.join(os.path.dirname(os.path.abspath(args.checkpoint)), "params", "env.yaml")
    if os.path.isfile(env_yaml):
        try:
            import yaml  # noqa: PLC0415

            with open(env_yaml, encoding="utf-8") as f:
                ycfg = yaml.unsafe_load(f) or {}
            changes = list(ycfg.get("vehicle_changes") or [])
            if ycfg.get("spec_ranges"):
                ecfg.ranges = SpecRanges.from_dict(ycfg["spec_ranges"])
            if ycfg.get("realism") is False:
                ecfg.realism = Realism.off()
            print(f"task settings from {env_yaml}")
        except Exception as e:  # noqa: BLE001
            print(f"note: could not read {env_yaml} ({e}); using the default ranges")
    _, base = baseline(args.config, changes)
    ev = None
    if not args.no_eval:
        suite = EvalSuite(args.config, base, ecfg.ranges)
        ev = suite.run(pol)
        print(f"validation: {ev['score']:.3f} vs reference law {ev['ref_score']:.3f} ({100 * ev['gain']:+.2f} %), "
              f"cones {ev['cones']}/{ev['ref_cones']}, failed {ev['failed']}")
    meta = make_meta(args.name, base, ecfg.ranges, ecfg.realism, ecfg,
                     f"PPO (RSL-RL in Isaac Lab), checkpoint {os.path.basename(args.checkpoint)}", [],
                     summarize_eval(ev) if ev else {}, [w.shape[1] for w, _ in layers[:-1]],
                     {"source_checkpoint": os.path.abspath(args.checkpoint)})
    out = os.path.join(registry.policies_dir(), args.name)
    save(out, pol, meta)
    print(f"policy: {out}")


if __name__ == "__main__":
    main()
