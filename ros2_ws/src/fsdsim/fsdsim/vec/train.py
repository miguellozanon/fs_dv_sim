"""Training loop of the universal throttle / brake policy (NumPy backend) and the shared pieces
(track sets, evaluation suite, saving) used by the PyTorch / GPU trainer as well.

    from fsdsim.vec.train import train
    meta = train(config_dir, out_dir="rl/policies/universal", iterations=600, n_envs=256)
"""

from __future__ import annotations

import math
import os
import time

import numpy as np

from .backend import Backend
from .env import EnvConfig, VecLonEnv
from .features import FEATURES_VERSION, N_FEATURES
from .ppo import PPO, PPOConfig, gae
from .spec import Realism, SpecBatch, SpecRanges, spec_values
from .tracks import LOOKAHEAD, TrackBank

TRAIN_ITEMS = [("acceleration", 0), ("skidpad", 0)] + [("autocross", s) for s in range(100, 160)]
VAL_ITEMS = [("acceleration", 0), ("skidpad", 0), ("autocross", 7), ("autocross", 11), ("autocross", 23)]
CONE_S = 2.0                     # FSG: 2 s per cone


def baseline(config_dir: str, changes=()):
    from ..config import load_all_configs  # noqa: PLC0415

    veh, _, _ = load_all_configs(config_dir, vehicle_overrides=list(changes))
    return veh, spec_values(veh)


def sample_specs(base: dict, ranges: SpecRanges, k: int, seed: int) -> list[dict]:
    """k specifications from the ranges (no hidden variations), deterministic for a seed."""
    B = Backend("numpy", seed=seed)
    sb = SpecBatch(B, base, k)
    sb.randomise(B.arange(k), ranges, Realism.off())
    return sb.rows_numpy()


class EvalSuite:
    """Timed runs on held-out tracks for the baseline car and a few sampled specifications;
    the score compares the policy with the reference driver's law (teacher) on the same runs."""

    def __init__(self, config_dir: str, base: dict, ranges: SpecRanges, items=VAL_ITEMS, n_specs: int = 4,
                 seed: int = 12345, bank: TrackBank | None = None):
        self.B = Backend("numpy")
        self.bank = (bank or TrackBank.build(config_dir, items)).to(self.B)
        self.rows = [dict(base)] + sample_specs(base, ranges, n_specs, seed)
        T = len(self.bank.infos)
        self.tids = np.array([t for _ in self.rows for t in range(T)])
        self.run_rows = [r for r in self.rows for _ in range(T)]
        self.env = VecLonEnv(self.B, self.bank, base, EnvConfig(n_envs=len(self.tids), randomise=False), teacher=True)
        self.ref = self.env.evaluate("teacher", self.tids, self.run_rows)

    @staticmethod
    def score(r: dict, ref: dict) -> float:
        t_ref = ref["time"] if ref["time"] is not None else 60.0
        if r["time"] is None or r["failed"]:
            return 2.0 * t_ref
        return r["time"] + CONE_S * r["cones"]

    def run(self, policy) -> dict:
        res = self.env.evaluate(lambda o: policy.forward(o), self.tids, self.run_rows)
        T = len(self.bank.infos)
        for i, (r, q) in enumerate(zip(res, self.ref)):
            r.update(spec=i // T, ref_time=q["time"], ref_cones=q["cones"])
        s_pol = np.array([self.score(r, q) for r, q in zip(res, self.ref)])
        s_ref = np.array([self.score(q, q) for q in self.ref])
        gain = float(np.mean((s_pol - s_ref) / s_ref))       # < 0: faster than the reference law
        return {"score": float(s_pol.mean()), "ref_score": float(s_ref.mean()), "gain": gain,
                "cones": int(sum(r["cones"] for r in res)), "ref_cones": int(sum(q["cones"] for q in self.ref)),
                "failed": int(sum(r["failed"] for r in res)), "runs": res}


def make_meta(name, base, ranges, realism, env_cfg, algo: str, history, evaluation, hidden, extra=None) -> dict:
    return {
        "name": name, "universal": True, "features_version": FEATURES_VERSION, "n_features": N_FEATURES,
        "lookahead_m": list(LOOKAHEAD), "base_spec": base, "spec_ranges": ranges.to_dict(),
        "realism": realism.to_dict(), "missions": list(env_cfg.missions), "hidden": list(hidden),
        "trained": time.strftime("%Y-%m-%d %H:%M"), "algorithm": algo, "history": history,
        "validation": evaluation, "spec_hash": "universal", "changes": [], **(extra or {}),
    }


def summarize_eval(ev: dict) -> dict:
    """The part of an EvalSuite result that goes into policy.json (spec 0 = the baseline car)."""
    runs = [{k: r.get(k) for k in ("mission", "seed", "time", "cones", "failed", "spin", "off_course", "spec",
                                   "ref_time", "ref_cones")} for r in ev["runs"]]
    return {k: ev[k] for k in ("score", "ref_score", "gain", "cones", "ref_cones", "failed")} | {"runs": runs}


# ---------------------------------------------------------------------------- #
def train(config_dir: str, out_dir: str = "", name: str = "universal", iterations: int = 500, n_envs: int = 256,
          env_cfg: EnvConfig | None = None, ppo_cfg: PPOConfig | None = None, seed: int = 0, train_items=None,
          eval_every: int = 25, log=print, progress=None, should_stop=None, init: str = "",
          bank: TrackBank | None = None, eval_suite: EvalSuite | None = None) -> dict:
    t_start = time.time()
    progress = progress or (lambda d: None)
    should_stop = should_stop or (lambda: False)
    cfg = ppo_cfg or PPOConfig()
    ecfg = env_cfg or EnvConfig()
    ecfg.n_envs = n_envs
    B = Backend("numpy", seed=seed)
    rng = np.random.default_rng(seed)
    _, base = baseline(config_dir)
    bank = bank or TrackBank.build(config_dir, train_items or TRAIN_ITEMS)
    env = VecLonEnv(B, bank, base, ecfg, teacher=True)
    agent = PPO(N_FEATURES, cfg, seed)
    if init:
        st = np.load(os.path.join(init, "agent.npz"), allow_pickle=True)
        agent.load_state_dict(st["state"].item())
        log(f"[ppo] continuing from {init}")
    suite = eval_suite or EvalSuite(config_dir, base, ecfg.ranges)
    log(f"[ppo] {n_envs} cars, {len(bank.infos)} tracks, {N_FEATURES} features, NumPy backend; "
        f"reference law on the validation runs: score {np.mean([suite.score(q, q) for q in suite.ref]):.3f}")
    history = []

    # ---- warm start: imitation of the reference law (DAgger) ----------------------------------
    obs = env.reset()
    if not init and cfg.bc_rounds > 0:
        X, Y = [], []
        for rnd in range(cfg.bc_rounds):
            progress({"phase": "imitation", "iteration": 0, "iterations": iterations})
            for _ in range(cfg.bc_steps):
                X.append(obs.copy())
                Y.append(env.teacher_a.copy())
                if rnd == 0:
                    a = env.teacher_a
                else:
                    mu = agent.actor.forward(agent.norm(obs))[:, 0]
                    a = np.where(rng.random(n_envs) < 0.5 ** rnd, env.teacher_a, mu)
                obs, *_ = env.step(a)
            Xa = np.concatenate(X)
            agent.norm.update(Xa[-n_envs * cfg.bc_steps:])
            mse = agent.fit_mean(agent.norm(Xa), np.concatenate(Y), cfg.bc_epochs, seed=rnd)
            log(f"[ppo] imitation round {rnd + 1}/{cfg.bc_rounds}: {len(Xa)} samples, mse {mse:.4f}")
        agent.log_std[0] = math.log(cfg.std_after_bc)
        ev = suite.run(agent.export_policy())
        log(f"[ppo] after imitation: validation score {ev['score']:.3f} (reference {ev['ref_score']:.3f}, "
            f"gain {100 * ev['gain']:+.2f} %, cones {ev['cones']}, failed {ev['failed']})")
        history.append({"iteration": 0, "val_score": ev["score"], "gain": ev["gain"], "cones": ev["cones"],
                        "failed": ev["failed"]})
    env.teacher = False                     # labels only needed for the warm start
    best = {"gain": history[-1]["gain"] if history else math.inf, "pol": agent.export_policy(), "it": 0,
            "ev": ev if history else None}

    # ---- PPO --------------------------------------------------------------------------------------
    T = cfg.n_steps
    ep_ret, ep_prog, ep_cones, ep_fail, ep_n = [], [], [], [], 0
    for it in range(1, iterations + 1):
        if should_stop():
            log("[ppo] stopped by the user")
            break
        t_it = time.time()
        buf = {k: [] for k in ("obs", "act", "logp", "val", "rew", "term", "trunc", "fval")}
        for _ in range(T):
            x = agent.norm(obs)
            a, mu, logp = agent.act(x, rng)
            v = agent.value(x)
            obs_raw = obs
            obs, rew, term, trunc, info = env.step(a)
            fval = np.where(trunc, agent.value(agent.norm(info["final_obs"])), 0.0) if trunc.any() else np.zeros(n_envs)
            for k, val in (("obs", x), ("act", a), ("logp", logp), ("val", v), ("rew", rew), ("term", term),
                           ("trunc", trunc), ("fval", fval)):
                buf[k].append(val)
            done = term | trunc
            if done.any():
                ep_ret += list(info["ep_return"][done])
                ep_prog += list(info["prog"][done])
                ep_cones += list(info["cones"][done])
                ep_fail += list(term[done].astype(float))
                ep_n += int(done.sum())
            agent.norm.update(obs_raw)
        last_val = agent.value(agent.norm(obs))
        arr = {k: np.stack(v) for k, v in buf.items()}
        adv, ret = gae(arr["rew"], arr["val"], arr["term"], arr["trunc"], last_val, arr["fval"], cfg.gamma, cfg.lam)
        flat = {"obs": arr["obs"].reshape(T * n_envs, -1), "act": arr["act"].ravel(), "logp": arr["logp"].ravel(),
                "adv": adv.ravel(), "ret": ret.ravel(), "val": arr["val"].ravel()}
        st = agent.update(flat, train_actor=it > (0 if init else cfg.critic_warmup))
        entry = {"iteration": it, "steps": it * T * n_envs, "time": round(time.time() - t_start, 1),
                 "ep_return": float(np.mean(ep_ret[-200:])) if ep_ret else None,
                 "ep_progress": float(np.mean(ep_prog[-200:])) if ep_prog else None,
                 "cones_per_ep": float(np.mean(ep_cones[-200:])) if ep_cones else None,
                 "fail_rate": float(np.mean(ep_fail[-200:])) if ep_fail else None, **st}
        if it % eval_every == 0 or it == iterations:
            pol = agent.export_policy()
            ev = suite.run(pol)
            entry.update(val_score=ev["score"], gain=ev["gain"], cones=ev["cones"], failed=ev["failed"])
            if ev["gain"] < best["gain"]:
                best = {"gain": ev["gain"], "pol": pol, "it": it, "ev": ev}
            log(f"[ppo] it {it:4d}/{iterations}  return {entry['ep_return'] or 0:7.2f}  fail {entry['fail_rate'] or 0:.3f}  "
                f"cones/ep {entry['cones_per_ep'] or 0:.2f}  std {st['std']:.3f}  lr {st['lr']:.1e}  kl {st['kl']:.4f} | "
                f"validation {ev['score']:.3f} vs reference {ev['ref_score']:.3f} ({100 * ev['gain']:+.2f} %, "
                f"cones {ev['cones']}/{ev['ref_cones']}, failed {ev['failed']})  best {100 * best['gain']:+.2f} % @ {best['it']}"
                f"  {time.time() - t_start:6.0f} s")
        elif it % 5 == 0:
            log(f"[ppo] it {it:4d}/{iterations}  return {entry['ep_return'] or 0:7.2f}  fail {entry['fail_rate'] or 0:.3f}  "
                f"cones/ep {entry['cones_per_ep'] or 0:.2f}  std {st['std']:.3f}  lr {st['lr']:.1e}  "
                f"{T * n_envs / (time.time() - t_it):6.0f} steps/s")
        history.append(entry)
        eta = (time.time() - t_start) / it * (iterations - it)
        progress({"phase": "learning", "iteration": it, "iterations": iterations, "gain": best["gain"],
                  "eta_s": round(eta)})

    meta = make_meta(name, base, ecfg.ranges, ecfg.realism, ecfg,
                     f"PPO (NumPy, {n_envs} cars x {T} steps per iteration, {len(history)} iterations)"
                     + ("" if init else " after imitation of the reference driver's law"),
                     history, summarize_eval(best["ev"]) | {"best_iteration": best["it"]} if best["ev"] else {},
                     cfg.hidden, {"train_seconds": round(time.time() - t_start), "continued_from": init or None})
    if out_dir:
        save(out_dir, best["pol"], meta, agent)
        log(f"[ppo] saved {out_dir} (best validation at iteration {best['it']}: {100 * best['gain']:+.2f} % vs reference law)")
    progress({"phase": "done", "iteration": iterations, "iterations": iterations, "gain": best["gain"]})
    return meta


def save(out_dir: str, policy, meta: dict, agent=None) -> None:
    policy.save(out_dir, meta)
    if agent is not None and hasattr(agent, "state_dict"):
        np.savez(os.path.join(out_dir, "agent.npz"), state=np.array(agent.state_dict(), dtype=object))
    with open(os.path.join(out_dir, "README.txt"), "w", encoding="utf-8") as f:
        f.write(f"Universal throttle / brake policy '{meta['name']}' (features v{FEATURES_VERSION}).\n"
                "Drives any vehicle specification inside spec_ranges of policy.json (relative to base_spec).\n"
                "Use: ros2 launch fsd_sim_bringup isaac.launch.py longitudinal:=rl  (or the dashboard)\n")
