"""Where trained policies live and which one fits a vehicle.

Policies are folders under <repo>/rl/policies/ (or $FSG_RL_POLICIES), each with policy.npz and
policy.json. policy.json records the vehicle specification (spec_hash) it was trained for, or - for
a universal policy (fsdsim.vec, features v2) - the baseline and the ranges of specifications it covers.

'auto' picks, for the running vehicle:
    1. a policy trained for exactly this specification (tools/train_rl.py, the dashboard), else
    2. a universal policy whose training ranges contain this specification (tools/train_rl_universal.py).
"""

from __future__ import annotations

import json
import os
import re


def policies_dir() -> str:
    env = os.environ.get("FSG_RL_POLICIES")
    if env:
        return env
    from ..perf.lapsim import find_lapsim_dir  # noqa: PLC0415

    lap = find_lapsim_dir()
    root = os.path.dirname(os.path.dirname(lap)) if lap else os.getcwd()
    return os.path.join(root, "rl", "policies")


def default_name(changes) -> str:
    if not changes:
        return "lr26_baseline"
    s = "_".join(re.sub(r"[^\w.+-]", "", c.replace("*", "x").replace("=", "")) for c in changes)
    return ("lr26_" + s)[:60]


def list_policies(folder: str | None = None) -> list[dict]:
    folder = folder or policies_dir()
    out = []
    if not os.path.isdir(folder):
        return out
    for name in sorted(os.listdir(folder)):
        meta_p = os.path.join(folder, name, "policy.json")
        if os.path.isfile(meta_p) and os.path.isfile(os.path.join(folder, name, "policy.npz")):
            try:
                with open(meta_p, encoding="utf-8") as f:
                    meta = json.load(f)
            except (OSError, ValueError):
                continue
            meta.pop("history", None)
            out.append({**meta, "folder": os.path.join(folder, name), "dir_name": name})
    return out


def find_for_vehicle(veh, folder: str | None = None) -> dict | None:
    """The policy trained for exactly this vehicle spec (newest first), else None."""
    from .es import spec_hash  # noqa: PLC0415

    h = spec_hash(veh)
    match = [p for p in list_policies(folder) if p.get("spec_hash") == h]
    match.sort(key=lambda p: p.get("trained", ""), reverse=True)
    return match[0] if match else None


def universal_coverage(meta: dict, veh) -> list[str]:
    """[] when the universal policy covers this vehicle, else the reasons it does not."""
    from ..vec.spec import SpecRanges, coverage, spec_values  # noqa: PLC0415

    try:
        vals = spec_values(veh)
    except ValueError as e:
        return [str(e)]
    return coverage(vals, meta.get("base_spec") or {}, SpecRanges.from_dict(meta.get("spec_ranges")))


def find_universal(veh, folder: str | None = None) -> tuple[dict | None, list[str]]:
    """(best universal policy covering this vehicle, reasons of the closest one that does not)."""
    unis = [p for p in list_policies(folder) if p.get("universal")]
    ok, why = [], []
    for p in unis:
        r = universal_coverage(p, veh)
        (ok if not r else why).append((p, r))
    if ok:
        ok.sort(key=lambda pr: (pr[0].get("validation", {}).get("gain", 0.0), pr[0].get("trained", "")))
        return ok[0][0], []
    if why:
        why.sort(key=lambda pr: len(pr[1]))
        return None, [f"universal policy '{why[0][0]['dir_name']}' does not cover it: " + "; ".join(why[0][1])]
    return None, []


def resolve(policy: str, veh=None) -> tuple[str | None, str]:
    """policy: 'auto' (trained for this vehicle, else a universal policy covering it), 'universal',
    a name under rl/policies, or a folder. Returns (folder or None, explanation)."""
    if policy in ("", "auto", "universal"):
        if veh is None:
            return None, "no vehicle to match"
        if policy != "universal":
            m = find_for_vehicle(veh)
            if m:
                return m["folder"], f"policy '{m['dir_name']}' (trained for this vehicle)"
        u, why = find_universal(veh)
        if u:
            return u["folder"], f"universal policy '{u['dir_name']}' (this setup is inside its training range)"
        reason = "no policy trained for this vehicle spec yet (tools/train_rl.py or the dashboard)"
        return None, "; ".join([reason] + why)
    cand = policy if os.path.isdir(policy) else os.path.join(policies_dir(), policy)
    if os.path.isfile(os.path.join(cand, "policy.npz")):
        return cand, f"policy '{os.path.basename(cand.rstrip('/'))}'"
    return None, f"policy '{policy}' not found in {policies_dir()}"
