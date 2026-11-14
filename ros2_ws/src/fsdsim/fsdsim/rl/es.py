"""Train the throttle / brake policy for one vehicle specification.

1. Imitation (warm start): the policy first copies the reference driver's pedal commands, so RL
   starts from a car that already gets round the track.
2. Reinforcement learning with evolution strategies (OpenAI-ES, Salimans et al. 2017): every
   generation, 2N perturbed copies of the policy (antithetic pairs +eps / -eps) drive the same
   scenarios; the fitness (distance in the time window, minus cones, off-course and crashes) of
   each pair gives a gradient estimate, applied with Adam. Rollouts run in parallel processes.
3. Every few generations the policy drives held-out tracks; the best one is kept, then compared
   with the reference driver on complete missions (official times, penalties).

Everything is NumPy + fsdsim; training runs on the CPU (all cores) in WSL or Linux.
"""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing as mp
import os
import time

import numpy as np

from ..control.rl_driver import FEATURES_VERSION, LOOKAHEAD_M, N_FEATURES
from .env import Scenario, ScenarioCache, full_run, make_scenarios, rollout
from .policy import MLPPolicy

# ---------------------------------------------------------------------------- #
# worker processes (each keeps its own scenario cache)
# ---------------------------------------------------------------------------- #
_W: dict = {}


def _winit(config_dir, changes, sizes):
    _W["cache"] = ScenarioCache(config_dir, changes)
    _W["policy"] = MLPPolicy(sizes[0], tuple(sizes[1:-1]))


def _weval(args):
    theta, scenarios = args
    pol = _W["policy"].set_params(theta)
    return [rollout(pol, _W["cache"], sc)["fitness"] for sc in scenarios]


def spec_hash(veh) -> str:
    """Identifies the vehicle a policy was trained for (LapSim parameters + actuator limits)."""
    perf = veh.perf().to_dict() if veh.perf() is not None else {}
    perf.pop("source", None)
    perf.pop("events", None)
    a = veh.actuators
    blob = json.dumps({"perf": perf, "act": [a.max_accel, a.max_decel, a.max_steer], "yaw": veh.yaw_inertia},
                      sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def _ranks(x: np.ndarray) -> np.ndarray:
    r = np.empty(len(x))
    r[np.argsort(x)] = np.arange(len(x))
    return r / (len(x) - 1) - 0.5


def train(config_dir: str, changes=(), out_dir: str = "", generations: int = 60, pairs: int = 12,
          sigma: float = 0.02, lr: float = 0.005, workers: int | None = None, missions=("acceleration", "skidpad", "autocross"),
          train_seeds=tuple(range(100, 140)), val_seeds=(7, 11, 23), tracks_per_gen: int = 3,
          hidden=(32, 32), seed: int = 0, name: str = "", log=print, progress=None, should_stop=None,
          trainable: str = "last", init: str = "") -> dict:
    """Returns the metadata written next to the policy (policy.json)."""
    t_start = time.time()
    rng = np.random.default_rng(seed)
    progress = progress or (lambda d: None)
    should_stop = should_stop or (lambda: False)
    cache = ScenarioCache(config_dir, changes)
    veh = cache.veh
    workers = max(1, workers or (os.cpu_count() or 2) - 1)
    log(f"[rl] vehicle {veh.name} ({' '.join(changes) or 'LapSim baseline'}), spec {spec_hash(veh)}, "
        f"{workers} worker(s)")

    # ---- 1. imitation warm start -------------------------------------------------------------
    # DAgger: round 0 the reference drives; later rounds the policy drives and the reference labels
    # what it would do in the states the policy reaches (so it also learns to recover).
    policy = MLPPolicy(N_FEATURES, hidden, seed)
    X, y = [], []
    progress({"phase": "imitation", "generation": 0, "generations": generations})
    if init:                                    # continue from a trained policy (no imitation)
        policy, init_meta = MLPPolicy.load(init)
        log(f"[rl] continuing from {init} ({init_meta.get('name')}, trained {init_meta.get('trained')})")
    for rnd in range(0 if init else 4):
        demo = make_scenarios(missions, rng.choice(train_seeds, size=min(6, len(train_seeds)), replace=False),
                              rng, random_starts=True)
        for sc in demo:
            rec: list = []
            rollout(policy, cache, sc, driver="reference" if rnd == 0 else "rl", record=rec)
            X += [r[0] for r in rec]
            y += [r[1] for r in rec]
        policy.fit(np.array(X), np.array(y), epochs=40 if rnd < 3 else 80, seed=rnd)
        log(f"[rl] imitation round {rnd + 1}/4: {len(X)} samples")

    val = make_scenarios(missions, val_seeds, rng, random_starts=False)

    def validate(pol, driver="rl"):
        res = [rollout(pol, cache, sc, driver=driver) for sc in val]
        return float(np.mean([r["fitness"] for r in res])), res

    ref_fit, _ = validate(policy, "reference")
    il_fit, _ = validate(policy)
    log(f"[rl] validation fitness: reference driver {ref_fit:.1f}, {'start policy' if init else 'after imitation'} {il_fit:.1f}")

    # ---- 2. evolution strategies ----------------------------------------------------------------
    # trainable "last": the hidden layers keep the features learned by imitation and RL searches the
    # output layer only (33 parameters, like Augmented Random Search on learned features) - far more
    # sample-efficient than perturbing all ~1900 weights. "all": every weight.
    full0 = policy.get_params()
    W_last, b_last = policy.layers[-1]
    n_t = W_last.size + b_last.size if trainable == "last" else full0.size
    theta0 = full0[-n_t:].copy()
    theta = theta0.copy()

    def full(th):
        out = full0.copy()
        out[-n_t:] = th
        return out

    m, v = np.zeros_like(theta), np.zeros_like(theta)
    best = {"fit": il_fit, "theta": full0.copy(), "gen": 0}
    history = [{"generation": 0, "train": None, "val": il_fit, "ref": ref_fit, "time": 0.0}]
    ctx = mp.get_context("fork") if hasattr(os, "fork") else mp.get_context()
    pool = ctx.Pool(workers, initializer=_winit, initargs=(config_dir, list(changes), policy.sizes)) if workers > 1 else None
    if pool is None:
        _winit(config_dir, list(changes), policy.sizes)
    try:
        for gen in range(1, generations + 1):
            if should_stop():
                log("[rl] stopped by the user")
                break
            seeds = rng.choice(train_seeds, size=min(tracks_per_gen, len(train_seeds)), replace=False)
            scen = make_scenarios(missions, seeds, rng, random_starts=True)
            eps = rng.standard_normal((pairs, theta.size))
            cands = [full(theta + sigma * e) for e in eps] + [full(theta - sigma * e) for e in eps]
            jobs = [(c, scen) for c in cands]
            fits = np.array([sum(f) for f in (pool.map(_weval, jobs) if pool else map(_weval, jobs))])
            r = _ranks(fits)
            g = ((r[:pairs] - r[pairs:])[:, None] * eps).sum(0) / (pairs * sigma)
            g -= 0.01 * (theta - theta0)                    # stay near the imitation solution unless RL gains
            m = 0.9 * m + 0.1 * g
            v = 0.999 * v + 0.001 * g * g
            theta = theta + lr * (m / (1 - 0.9 ** gen)) / (np.sqrt(v / (1 - 0.999 ** gen)) + 1e-8)
            entry = {"generation": gen, "train": float(fits.mean()), "train_best": float(fits.max()),
                     "time": round(time.time() - t_start, 1)}
            if gen % 5 == 0 or gen == generations:
                policy.set_params(full(theta))
                vf, _ = validate(policy)
                entry["val"] = vf
                if vf > best["fit"]:
                    best = {"fit": vf, "theta": full(theta), "gen": gen}
                log(f"[rl] gen {gen:3d}/{generations}  train {fits.mean():7.1f} (best {fits.max():7.1f})  "
                    f"validation {vf:7.1f}  (reference {ref_fit:.1f}, best so far {best['fit']:.1f} @ {best['gen']})"
                    f"  {time.time() - t_start:5.0f} s")
            history.append(entry)
            eta = (time.time() - t_start) / gen * (generations - gen)
            progress({"phase": "learning", "generation": gen, "generations": generations, "train": entry["train"],
                      "val": entry.get("val"), "best_val": best["fit"], "ref_val": ref_fit, "eta_s": round(eta)})
    finally:
        if pool:
            pool.close()
            pool.join()

    # ---- 3. evaluation on complete missions --------------------------------------------------------
    policy.set_params(best["theta"])
    progress({"phase": "evaluating", "generation": generations, "generations": generations})
    evals = []
    for mission in missions:
        for sd in ((0,) if mission != "autocross" else val_seeds):
            ref = full_run(None, cache, mission, sd)
            rl = full_run(policy, cache, mission, sd)
            evals.append({"mission": mission, "seed": sd, "reference": ref, "rl": rl})
            log(f"[rl] {mission:12s} seed {sd:3d}: reference {_fmt(ref)} | RL {_fmt(rl)}")
    meta = {
        "name": name or f"spec_{spec_hash(veh)}", "spec_hash": spec_hash(veh), "vehicle": veh.name,
        "changes": list(changes), "perf": veh.perf().to_dict(), "actuators": {"max_accel": veh.actuators.max_accel,
                                                                              "max_decel": veh.actuators.max_decel},
        "features_version": FEATURES_VERSION, "lookahead_m": list(LOOKAHEAD_M), "missions": list(missions),
        "trained": time.strftime("%Y-%m-%d %H:%M"), "train_seconds": round(time.time() - t_start),
        "continued_from": init or None,
        "algorithm": f"imitation (DAgger) + OpenAI-ES on {'the output layer' if trainable == 'last' else 'all weights'} "
                     f"({2 * pairs} rollouts x {generations} generations, sigma {sigma})",
        "validation": {"reference": ref_fit, "imitation": il_fit, "rl": best["fit"], "best_generation": best["gen"]},
        "evaluation": evals, "history": history,
    }
    if out_dir:
        policy.save(out_dir, meta)
        log(f"[rl] saved {out_dir}")
    progress({"phase": "done", "generation": generations, "generations": generations, "best_val": best["fit"],
              "ref_val": ref_fit})
    return meta


def _fmt(r: dict) -> str:
    if r["raw_time"] is None:
        return f"{r['state'].lower()} (no time)"
    pen = f" +{r['cones']}c/{r['off_course']}oc" if r["cones"] or r["off_course"] else ""
    return f"{r['raw_time']:.3f} s{pen}"
