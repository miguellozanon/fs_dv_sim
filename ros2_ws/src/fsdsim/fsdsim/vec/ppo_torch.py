"""PPO for the universal throttle / brake policy on a GPU (PyTorch): thousands of cars in parallel.

Same environment, rewards, warm start and export as the NumPy trainer (fsdsim.vec.train); the
batched car model, the LapSim limit tables, the track bank and the features all live on the GPU.

    python3 tools/train_rl_universal.py --backend torch --device cuda:0 --num-envs 4096

Needs PyTorch (pip install torch, CUDA build for a GPU). Isaac Lab ships PyTorch as well; with Isaac Lab
the RSL-RL trainer can be used instead (isaaclab/fsd_rl).
"""

from __future__ import annotations

import math
import os
import time

import numpy as np

from .backend import Backend
from .env import EnvConfig, VecLonEnv
from .features import N_FEATURES
from .ppo import PPOConfig
from .train import TRAIN_ITEMS, EvalSuite, baseline, make_meta, save, summarize_eval
from .tracks import TrackBank


def _torch():
    import torch  # noqa: PLC0415

    return torch


class TorchNorm:
    def __init__(self, n, device):
        torch = _torch()
        self.mean = torch.zeros(n, device=device, dtype=torch.float64)
        self.var = torch.ones(n, device=device, dtype=torch.float64)
        self.count = 1e-4

    def update(self, x):
        x = x.double()
        bm, bv, bc = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        d = bm - self.mean
        tot = self.count + bc
        self.mean = self.mean + d * bc / tot
        self.var = (self.var * self.count + bv * bc + d * d * self.count * bc / tot) / tot
        self.count = tot

    def __call__(self, x):
        return ((x.double() - self.mean) / (self.var + 1e-8).sqrt()).to(x.dtype)


def build_model(n_obs: int, hidden, init_std: float):
    torch = _torch()
    nn = torch.nn

    def mlp(out_gain):
        layers, a = [], n_obs
        for h in hidden:
            layers += [nn.Linear(a, h), nn.Tanh()]
            a = h
        last = nn.Linear(a, 1)
        with torch.no_grad():
            last.weight.mul_(out_gain)
            last.bias.zero_()
        return nn.Sequential(*layers, last)

    class ActorCritic(nn.Module):
        def __init__(self):
            super().__init__()
            self.actor = mlp(0.1)
            self.critic = mlp(1.0)
            self.log_std = nn.Parameter(torch.full((1,), math.log(init_std)))

        def mean(self, x):
            return torch.tanh(self.actor(x)).squeeze(-1)

        def value(self, x):
            return self.critic(x).squeeze(-1)

    return ActorCritic()


def export_policy(model, norm):
    """MLPPolicy (NumPy) taking raw features, normalisation folded into the first layer."""
    from ..rl.policy import MLPPolicy  # noqa: PLC0415

    lin = [m for m in model.actor if hasattr(m, "weight")]
    Ws = [m.weight.detach().double().cpu().numpy().T for m in lin]
    bs = [m.bias.detach().double().cpu().numpy() for m in lin]
    mean, var = norm.mean.cpu().numpy(), norm.var.cpu().numpy()
    sd = np.sqrt(var + 1e-8)
    W0 = Ws[0] / sd[:, None]
    b0 = bs[0] - (mean / sd) @ Ws[0]
    pol = MLPPolicy(Ws[0].shape[0], tuple(W.shape[1] for W in Ws[:-1]))
    pol.layers = [[W0, b0]] + [[W, b] for W, b in zip(Ws[1:], bs[1:])]
    return pol


def train_torch(config_dir: str, out_dir: str = "", name: str = "universal", iterations: int = 1500,
                n_envs: int = 4096, device: str = "cuda:0", env_cfg: EnvConfig | None = None,
                ppo_cfg: PPOConfig | None = None, seed: int = 0, train_items=None, eval_every: int = 50, log=print,
                progress=None, should_stop=None, init: str = "", compile_physics: bool = False) -> dict:
    torch = _torch()
    t_start = time.time()
    progress = progress or (lambda d: None)
    should_stop = should_stop or (lambda: False)
    cfg = ppo_cfg or PPOConfig()
    ecfg = env_cfg or EnvConfig()
    ecfg.n_envs = n_envs
    torch.manual_seed(seed)
    B = Backend("torch", device=device, seed=seed)
    dev = B.device
    _, base = baseline(config_dir)
    bank = TrackBank.build(config_dir, train_items or TRAIN_ITEMS)
    env = VecLonEnv(B, bank, base, ecfg, teacher=True)
    if compile_physics:
        try:
            env.car.physics_step = torch.compile(env.car.physics_step)
            log("[ppo-gpu] physics step compiled (torch.compile)")
        except Exception as e:  # noqa: BLE001
            log(f"[ppo-gpu] torch.compile not available ({e}); running eagerly")
    model = build_model(N_FEATURES, cfg.hidden, cfg.init_std).to(dev)
    norm = TorchNorm(N_FEATURES, dev)
    opt_a = torch.optim.Adam(list(model.actor.parameters()) + [model.log_std], lr=cfg.lr)
    opt_c = torch.optim.Adam(model.critic.parameters(), lr=cfg.lr)
    lr = cfg.lr
    if init:
        ck = torch.load(os.path.join(init, "agent_torch.pt"), map_location=dev)
        model.load_state_dict(ck["model"])
        norm.mean, norm.var, norm.count = ck["norm_mean"].to(dev), ck["norm_var"].to(dev), ck["norm_count"]
        log(f"[ppo-gpu] continuing from {init}")
    suite = EvalSuite(config_dir, base, ecfg.ranges)
    log(f"[ppo-gpu] {n_envs} cars on {dev}, {len(bank.infos)} tracks, {N_FEATURES} features; reference law on "
        f"the validation runs: score {np.mean([suite.score(q, q) for q in suite.ref]):.3f}")
    history, ev = [], None

    obs = env.reset()
    # ---- warm start: imitation of the reference law (DAgger) ----------------------------------
    if not init and cfg.bc_rounds > 0:
        X, Y = [], []
        for rnd in range(cfg.bc_rounds):
            progress({"phase": "imitation", "iteration": 0, "iterations": iterations})
            for _ in range(cfg.bc_steps):
                X.append(obs.clone())
                Y.append(env.teacher_a.clone())
                if rnd == 0:
                    a = env.teacher_a
                else:
                    with torch.no_grad():
                        mu = model.mean(norm(obs))
                    a = torch.where(torch.rand(n_envs, device=dev) < 0.5 ** rnd, env.teacher_a, mu)
                obs, *_ = env.step(a)
            Xa, Ya = torch.cat(X), torch.cat(Y).clamp(-0.98, 0.98)
            norm.update(Xa[-n_envs * cfg.bc_steps:])
            opt = torch.optim.Adam(model.actor.parameters(), lr=1e-3)
            Xn = norm(Xa)
            for _ in range(cfg.bc_epochs):
                perm = torch.randperm(Xn.shape[0], device=dev)
                for k in range(0, Xn.shape[0], 8192):
                    j = perm[k:k + 8192]
                    loss = ((model.mean(Xn[j]) - Ya[j]) ** 2).mean()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            with torch.no_grad():
                mse = float(((model.mean(Xn) - Ya) ** 2).mean())
            log(f"[ppo-gpu] imitation round {rnd + 1}/{cfg.bc_rounds}: {Xn.shape[0]} samples, mse {mse:.4f}")
        # the imitation data is not needed any more: give its GPU memory back
        X = Y = Xa = Ya = Xn = opt = perm = j = loss = mu = a = None
        _release(torch, dev)
        with torch.no_grad():
            model.log_std.fill_(math.log(cfg.std_after_bc))
        ev = suite.run(export_policy(model, norm))
        log(f"[ppo-gpu] after imitation: validation {ev['score']:.3f} vs reference {ev['ref_score']:.3f} "
            f"({100 * ev['gain']:+.2f} %){_gpu_mem(torch, dev)}")
        history.append({"iteration": 0, "val_score": ev["score"], "gain": ev["gain"]})
    env.teacher = False
    best = {"gain": ev["gain"] if ev else math.inf, "pol": export_policy(model, norm), "it": 0, "ev": ev}

    def meta_now(complete: bool, reached: int) -> dict:
        extra = {"train_seconds": round(time.time() - t_start), "continued_from": init or None,
                 "complete": complete, "iterations_done": reached, "iterations_planned": iterations}
        return make_meta(name, base, ecfg.ranges, ecfg.realism, ecfg,
                         f"PPO (PyTorch on {dev}, {N} cars x {T} steps per iteration, {reached} iterations)"
                         + ("" if init else " after imitation of the reference driver's law"),
                         history, (summarize_eval(best["ev"]) | {"best_iteration": best["it"]}) if best["ev"] else {},
                         cfg.hidden, extra)

    def save_agent() -> None:
        try:
            torch.save({"model": model.state_dict(), "norm_mean": norm.mean, "norm_var": norm.var,
                        "norm_count": norm.count}, os.path.join(out_dir, "agent_torch.pt"))
        except Exception as e:  # noqa: BLE001 - the policy itself is saved already
            log(f"[ppo-gpu] could not save the training state agent_torch.pt ({e})")

    def checkpoint(reached: int) -> None:
        """Best policy so far + the training state, so a crash or a stop never loses the run."""
        if not out_dir:
            return
        save(out_dir, best["pol"], meta_now(False, reached))
        save_agent()

    # ---- PPO --------------------------------------------------------------------------------------
    T, N = cfg.n_steps, n_envs
    ep = {"ret": [], "fail": [], "cones": []}
    state = {"obs": obs, "lr": cfg.lr, "reset": False}

    def iteration(it: int) -> dict:
        obs, lr = state["obs"], state["lr"]
        if state["reset"]:                       # after running out of memory: start the cars afresh
            obs = env.reset()
            state["reset"] = False
        t_it = time.time()
        bo = torch.zeros((T, N, N_FEATURES), device=dev)
        ba, blp, bv, br = (torch.zeros((T, N), device=dev) for _ in range(4))
        bterm, btrunc = (torch.zeros((T, N), dtype=torch.bool, device=dev) for _ in range(2))
        bfv = torch.zeros((T, N), device=dev)
        with torch.no_grad():
            std = model.log_std.exp()
            for t in range(T):
                x = norm(obs).float()
                mu = model.mean(x)
                a = mu + std * torch.randn(N, device=dev)
                bo[t], ba[t], bv[t] = x, a, model.value(x)
                blp[t] = -0.5 * ((a - mu) / std) ** 2 - model.log_std - 0.5 * math.log(2 * math.pi)
                raw = obs
                obs, rew, term, trunc, info = env.step(a)
                br[t], bterm[t], btrunc[t] = rew, term, trunc
                if bool(trunc.any()):
                    fv = model.value(norm(info["final_obs"]).float())
                    bfv[t] = torch.where(trunc, fv, torch.zeros_like(fv))
                done = term | trunc
                if bool(done.any()):
                    ep["ret"] += info["ep_return"][done].tolist()
                    ep["fail"] += term[done].float().tolist()
                    ep["cones"] += info["cones"][done].tolist()
                norm.update(raw)
            state["obs"] = obs
            last = model.value(norm(obs).float())
            adv = torch.zeros((T, N), device=dev)
            gl = torch.zeros(N, device=dev)
            for t in range(T - 1, -1, -1):
                nv = last if t == T - 1 else bv[t + 1]
                nv = torch.where(btrunc[t], bfv[t], torch.where(bterm[t], torch.zeros_like(nv), nv))
                delta = br[t] + cfg.gamma * nv - bv[t]
                gl = delta + cfg.gamma * cfg.lam * torch.where(bterm[t] | btrunc[t], torch.zeros_like(gl), gl)
                adv[t] = gl
            ret = adv + bv
        for k in ep:                             # only the last episodes are reported
            del ep[k][:-5000]
        x_all, a_all, lp_all = bo.reshape(T * N, -1), ba.reshape(-1), blp.reshape(-1)
        adv_all, ret_all = adv.reshape(-1), ret.reshape(-1)
        n = x_all.shape[0]
        mb = max(1, n // cfg.minibatches)
        train_actor = it > (0 if init else cfg.critic_warmup)
        kls, vls = [], []
        for _ in range(cfg.epochs):
            perm = torch.randperm(n, device=dev)
            for k in range(0, n, mb):
                j = perm[k:k + mb]
                x = x_all[j]
                v_loss = cfg.value_coef * 0.5 * ((model.value(x) - ret_all[j]) ** 2).mean()
                opt_c.zero_grad()
                v_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.critic.parameters(), cfg.max_grad_norm)
                for g in opt_c.param_groups:
                    g["lr"] = lr
                opt_c.step()
                vls.append(v_loss.detach().item())
                if not train_actor:
                    continue
                mu = model.mean(x)
                std = model.log_std.exp()
                logp = -0.5 * ((a_all[j] - mu) / std) ** 2 - model.log_std - 0.5 * math.log(2 * math.pi)
                ratio = (logp - lp_all[j]).clamp(-20, 20).exp()
                A = adv_all[j]
                A = (A - A.mean()) / (A.std() + 1e-8)
                pi_loss = -torch.min(ratio * A, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * A).mean()
                loss = pi_loss - cfg.entropy * model.log_std.sum()
                opt_a.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(model.actor.parameters()) + [model.log_std], cfg.max_grad_norm)
                for g in opt_a.param_groups:
                    g["lr"] = lr
                opt_a.step()
                with torch.no_grad():
                    lr_ = logp - lp_all[j]
                    kls.append(float((lr_.exp() - 1 - lr_).mean()))
            if train_actor and cfg.desired_kl > 0 and kls:
                kl = float(np.mean(kls[-cfg.minibatches:]))
                if kl > 2.0 * cfg.desired_kl:
                    lr = max(cfg.lr_min, lr / 1.5)
                elif kl < 0.5 * cfg.desired_kl:
                    lr = min(cfg.lr_max, lr * 1.5)
        state["lr"] = lr
        with torch.no_grad():
            model.log_std.clamp_(math.log(0.02), 0.0)
        return {"iteration": it, "steps": it * T * N, "time": round(time.time() - t_start, 1),
                "ep_return": float(np.mean(ep["ret"][-2000:])) if ep["ret"] else None,
                "fail_rate": float(np.mean(ep["fail"][-2000:])) if ep["fail"] else None,
                "cones_per_ep": float(np.mean(ep["cones"][-2000:])) if ep["cones"] else None,
                "kl": float(np.mean(kls)) if kls else 0.0, "v_loss": float(np.mean(vls)), "lr": lr,
                "std": float(model.log_std.detach().exp()), "_sps": T * N / (time.time() - t_it)}

    it, reached, oom_tries, aborted = 1, 0, 0, ""
    while it <= iterations:
        if should_stop():
            log("[ppo-gpu] stopped by the user")
            break
        oom = False
        try:
            entry = iteration(it)
        except Exception as e:  # noqa: BLE001
            if not _is_oom(e):
                try:
                    checkpoint(reached)          # keep what was learnt, then report the error
                except Exception:  # noqa: BLE001, S110
                    pass
                raise
            oom = True
        if oom:
            # outside the except block: the traceback (and the tensors it holds) is gone now
            oom_tries += 1
            state["reset"] = True
            _release(torch, dev)
            log(f"[ppo-gpu] it {it}: the GPU ran out of memory (try {oom_tries}/{OOM_TRIES}){_gpu_mem(torch, dev)}."
                " Another program is probably using the GPU (Isaac Sim, a game, a browser with video).")
            if oom_tries >= OOM_TRIES:
                aborted = "out of GPU memory"
                break
            checkpoint(reached)
            time.sleep(OOM_WAIT_S)
            _release(torch, dev)
            continue
        oom_tries = 0
        reached = it
        sps = entry.pop("_sps")
        lr = state["lr"]
        if it % eval_every == 0 or it == iterations:
            pol = export_policy(model, norm)
            ev = suite.run(pol)
            entry.update(val_score=ev["score"], gain=ev["gain"], cones=ev["cones"], failed=ev["failed"])
            improved = ev["gain"] < best["gain"]
            if improved:
                best = {"gain": ev["gain"], "pol": pol, "it": it, "ev": ev}
            log(f"[ppo-gpu] it {it:5d}/{iterations}  return {entry['ep_return'] or 0:7.2f}  fail {entry['fail_rate'] or 0:.3f}"
                f"  std {entry['std']:.3f} | validation {ev['score']:.3f} vs {ev['ref_score']:.3f} ({100 * ev['gain']:+.2f} %,"
                f" cones {ev['cones']}/{ev['ref_cones']}, failed {ev['failed']})  best {100 * best['gain']:+.2f} % @ {best['it']}")
            history.append(entry)
            if improved and it < iterations:
                checkpoint(it)
        else:
            if it % 10 == 0:
                log(f"[ppo-gpu] it {it:5d}/{iterations}  return {entry['ep_return'] or 0:7.2f}  fail {entry['fail_rate'] or 0:.3f}"
                    f"  std {entry['std']:.3f}  lr {lr:.1e}  {sps:8.0f} steps/s{_gpu_mem(torch, dev)}")
            history.append(entry)
        progress({"phase": "learning", "iteration": it, "iterations": iterations, "gain": best["gain"],
                  "eta_s": round((time.time() - t_start) / it * (iterations - it))})
        it += 1

    meta = meta_now(not aborted and reached == iterations, reached)
    if aborted:
        meta["aborted"] = aborted
    if out_dir:
        save(out_dir, best["pol"], meta)
        save_agent()
        log(f"[ppo-gpu] saved {out_dir} (best validation at iteration {best['it']}: {100 * best['gain']:+.2f} %,"
            f" {reached}/{iterations} iterations)")
    if aborted:
        log(f"[ppo-gpu] training stopped early: {aborted}. The best policy so far is saved; close the other GPU"
            f" program and start the training again with 'Continue from' = {name} to go on.")
    progress({"phase": "aborted" if aborted else "done", "iteration": reached, "iterations": iterations,
              "gain": best["gain"]})
    return meta


OOM_TRIES = 3          # out-of-memory errors in a row before the training gives up (and saves)
OOM_WAIT_S = 20.0      # pause after one, so a program that briefly used the GPU can finish


def _is_oom(e: BaseException) -> bool:
    """CUDA out of memory - torch.OutOfMemoryError (allocator) or AcceleratorError (cudaErrorMemoryAllocation)."""
    s = f"{type(e).__name__} {e}".lower()
    return "outofmemory" in s or "out of memory" in s or "memoryallocation" in s


def _release(torch, dev) -> None:
    import gc  # noqa: PLC0415

    gc.collect()
    if getattr(dev, "type", "") == "cuda":
        try:
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001, S110
            pass


def _gpu_mem(torch, dev) -> str:
    """'  GPU 0.6 GB ours, 5.1/8.0 GB free' (free memory counts every program on the GPU)."""
    if getattr(dev, "type", "") != "cuda":
        return ""
    try:
        free, total = torch.cuda.mem_get_info(dev)
        return f"  GPU {torch.cuda.memory_reserved(dev) / 1e9:.1f} GB ours, {free / 1e9:.1f}/{total / 1e9:.1f} GB free"
    except Exception:  # noqa: BLE001
        return ""
