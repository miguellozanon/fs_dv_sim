"""PPO for the universal throttle / brake policy - NumPy version (CPU; exact gradients by hand).

The PyTorch version (fsdsim.vec.ppo_torch) runs the same algorithm on a GPU with thousands of cars;
Isaac Lab users can also train with RSL-RL (isaaclab/fsd_rl). All three export the same policy file
(rl/policies/<name>/policy.npz, an MLPPolicy with the observation normalisation folded in).

Algorithm (Schulman et al. 2017, with the usual details):
    actor  obs -> tanh MLP -> mean a in [-1, 1]; Gaussian exploration with a learned log std
    critic obs -> tanh MLP -> value
    GAE(lambda), clipped surrogate, value loss, entropy bonus, Adam, global gradient clipping,
    learning rate adapted to a target KL (as RSL-RL), running observation normalisation,
    truncated episodes bootstrapped with the critic.
Warm start (optional): imitation of the reference driver's longitudinal law (DAgger) for a few
iterations, so RL starts from a car that already gets round the track.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from ..rl.policy import MLPPolicy


# ---------------------------------------------------------------------------- #
class MLP:
    """tanh hidden layers, output 'tanh' or 'linear'; forward keeps activations for backward()."""

    def __init__(self, sizes, out="linear", seed=0, out_gain=1.0):
        rng = np.random.default_rng(seed)
        self.sizes, self.out = list(sizes), out
        self.params = []
        for a, b in zip(sizes[:-1], sizes[1:]):
            W = rng.standard_normal((a, b)) * math.sqrt(1.0 / a)
            self.params += [W, np.zeros(b)]
        self.params[-2] *= out_gain
        self._hs = None

    @property
    def n_layers(self):
        return len(self.params) // 2

    def forward(self, x, keep=False):
        hs = [x]
        h = x
        for i in range(self.n_layers):
            W, b = self.params[2 * i], self.params[2 * i + 1]
            z = h @ W + b
            last = i == self.n_layers - 1
            h = z if (last and self.out == "linear") else np.tanh(z)
            hs.append(h)
        if keep:
            self._hs = hs
        return h

    def backward(self, g_out):
        """Gradients of the loss w.r.t. the parameters, given dL/d(output) of the last forward(keep=True)."""
        hs = self._hs
        grads = [None] * len(self.params)
        g = g_out
        if self.out == "tanh":
            g = g * (1 - hs[-1] ** 2)
        for i in range(self.n_layers - 1, -1, -1):
            W = self.params[2 * i]
            grads[2 * i] = hs[i].T @ g
            grads[2 * i + 1] = g.sum(0)
            if i:
                g = (g @ W.T) * (1 - hs[i] ** 2)
        return grads


class Adam:
    def __init__(self, params, lr=3e-4, betas=(0.9, 0.999), eps=1e-8):
        self.params, self.lr, self.b1, self.b2, self.eps = params, lr, betas[0], betas[1], eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g * g
            p -= self.lr * (m / (1 - self.b1 ** self.t)) / (np.sqrt(v / (1 - self.b2 ** self.t)) + self.eps)


def clip_grads(grads, max_norm):
    n = math.sqrt(sum(float((g * g).sum()) for g in grads))
    if n > max_norm > 0:
        grads = [g * (max_norm / n) for g in grads]
    return grads, n


class RunningNorm:
    def __init__(self, n):
        self.mean, self.var, self.count = np.zeros(n), np.ones(n), 1e-4

    def update(self, x):
        bm, bv, bc = x.mean(0), x.var(0), len(x)
        d = bm - self.mean
        tot = self.count + bc
        self.mean = self.mean + d * bc / tot
        self.var = (self.var * self.count + bv * bc + d * d * self.count * bc / tot) / tot
        self.count = tot

    def __call__(self, x):
        return (x - self.mean) / np.sqrt(self.var + 1e-8)

    def state(self):
        return {"mean": self.mean.tolist(), "var": self.var.tolist(), "count": self.count}


# ---------------------------------------------------------------------------- #
@dataclass
class PPOConfig:
    hidden: tuple = (64, 64)
    n_steps: int = 32               # per car and iteration (x n_envs = batch)
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    epochs: int = 5
    minibatches: int = 4
    lr: float = 3e-4
    lr_min: float = 1e-5
    lr_max: float = 5e-4
    desired_kl: float = 0.01        # adaptive learning rate (0: fixed)
    entropy: float = 0.0
    value_coef: float = 1.0
    max_grad_norm: float = 1.0
    init_std: float = 0.5
    std_after_bc: float = 0.2
    critic_warmup: int = 5          # iterations that only fit the critic (after the warm start)
    bc_rounds: int = 4              # imitation warm start rounds (0: none)
    bc_steps: int = 64
    bc_epochs: int = 8


class PPO:
    """Actor-critic and the update; the rollout loop is in `train()`."""

    def __init__(self, n_obs: int, cfg: PPOConfig, seed: int = 0):
        self.cfg = cfg
        self.actor = MLP([n_obs, *cfg.hidden, 1], out="tanh", seed=seed, out_gain=0.1)
        self.critic = MLP([n_obs, *cfg.hidden, 1], out="linear", seed=seed + 1, out_gain=1.0)
        self.log_std = np.array([math.log(cfg.init_std)])
        self.norm = RunningNorm(n_obs)
        self.opt_a = Adam(self.actor.params + [self.log_std], cfg.lr)
        self.opt_c = Adam(self.critic.params, cfg.lr)
        self.lr = cfg.lr

    # ------------------------------------------------------------------ acting
    # obs below are NORMALISED observations (self.norm(raw)): stored like that in the rollout buffer,
    # so the update sees exactly what the policy saw when it acted.
    def act(self, x, rng: np.random.Generator, deterministic=False):
        mu = self.actor.forward(x)[:, 0]
        if deterministic:
            return mu, mu, None
        std = math.exp(self.log_std[0])
        a = mu + std * rng.standard_normal(len(mu))
        logp = -0.5 * ((a - mu) / std) ** 2 - self.log_std[0] - 0.5 * math.log(2 * math.pi)
        return a, mu, logp

    def value(self, x):
        return self.critic.forward(x)[:, 0]

    # ------------------------------------------------------------------ imitation
    def fit_mean(self, Xn, y, epochs: int, batch: int = 2048, seed: int = 0) -> float:
        """Imitation: regress the actor's mean on labels y (Xn normalised observations)."""
        rng = np.random.default_rng(seed)
        opt = Adam(self.actor.params, 1e-3)
        y = np.clip(y, -0.98, 0.98)
        for _ in range(epochs):
            perm = rng.permutation(len(Xn))
            for k in range(0, len(Xn), batch):
                j = perm[k:k + batch]
                out = self.actor.forward(Xn[j], keep=True)[:, 0]
                g = (2.0 / len(j)) * (out - y[j])
                opt.step(self.actor.backward(g[:, None]))
        return float(np.mean((self.actor.forward(Xn)[:, 0] - y) ** 2))

    # ------------------------------------------------------------------ PPO update
    def update(self, buf: dict, train_actor: bool = True) -> dict:
        cfg = self.cfg
        obs, act, logp_old, adv, ret, val_old = (buf[k] for k in ("obs", "act", "logp", "adv", "ret", "val"))
        x_all = obs
        n = len(obs)
        mb = max(1, n // cfg.minibatches)
        rng = np.random.default_rng(int(time.time() * 1000) % 2**31)
        stats = {"kl": [], "clipfrac": [], "v_loss": [], "pi_loss": []}
        for _ in range(cfg.epochs):
            perm = rng.permutation(n)
            for k in range(0, n, mb):
                j = perm[k:k + mb]
                x = x_all[j]
                # critic
                v = self.critic.forward(x, keep=True)[:, 0]
                dv = v - ret[j]
                stats["v_loss"].append(float(np.mean(dv ** 2)))
                gc = self.critic.backward((cfg.value_coef * dv / len(j))[:, None])
                gc, _ = clip_grads(gc, cfg.max_grad_norm)
                self.opt_c.lr = self.lr
                self.opt_c.step(gc)
                if not train_actor:
                    continue
                # actor
                mu = self.actor.forward(x, keep=True)[:, 0]
                ls = self.log_std[0]
                std = math.exp(ls)
                z = (act[j] - mu) / std
                logp = -0.5 * z * z - ls - 0.5 * math.log(2 * math.pi)
                ratio = np.exp(np.clip(logp - logp_old[j], -20, 20))
                A = adv[j]
                A = (A - A.mean()) / (A.std() + 1e-8)
                s1, s2 = ratio * A, np.clip(ratio, 1 - cfg.clip, 1 + cfg.clip) * A
                use = s1 <= s2                                   # min picks the unclipped term
                stats["pi_loss"].append(float(-np.mean(np.minimum(s1, s2))))
                stats["clipfrac"].append(float(np.mean(np.abs(ratio - 1) > cfg.clip)))
                dlogp = np.where(use, -A * ratio, 0.0) / len(j)  # dL/dlogp
                g_mu = dlogp * z / std
                g_ls = float(np.sum(dlogp * (z * z - 1.0))) - cfg.entropy
                ga = self.actor.backward(g_mu[:, None]) + [np.array([g_ls])]
                ga, _ = clip_grads(ga, cfg.max_grad_norm)
                self.opt_a.lr = self.lr
                self.opt_a.step(ga)
                # KL(old || new) for diagonal Gaussians with the old std from logp_old is not stored:
                # approximate with the sample estimate (ratio - 1) - log ratio  (Schulman's k3)
                lr_ = logp - logp_old[j]
                stats["kl"].append(float(np.mean(np.exp(lr_) - 1 - lr_)))
            if train_actor and cfg.desired_kl > 0 and stats["kl"]:
                kl = np.mean(stats["kl"][-max(1, n // mb):])
                if kl > 2.0 * cfg.desired_kl:
                    self.lr = max(cfg.lr_min, self.lr / 1.5)
                elif kl < 0.5 * cfg.desired_kl:
                    self.lr = min(cfg.lr_max, self.lr * 1.5)
        self.log_std[0] = float(np.clip(self.log_std[0], math.log(0.02), math.log(1.0)))
        return {k: float(np.mean(v)) if v else 0.0 for k, v in stats.items()} | {"lr": self.lr,
                                                                              "std": math.exp(self.log_std[0])}

    # ------------------------------------------------------------------ export
    def export_policy(self) -> MLPPolicy:
        """MLPPolicy (tanh layers) taking RAW features: the normalisation is folded into layer 1."""
        sizes = self.actor.sizes
        pol = MLPPolicy(sizes[0], tuple(sizes[1:-1]))
        P = self.actor.params
        sd = np.sqrt(self.norm.var + 1e-8)
        W0 = P[0] / sd[:, None]
        b0 = P[1] - (self.norm.mean / sd) @ P[0]
        pol.layers = [[W0, b0]] + [[P[2 * i], P[2 * i + 1]] for i in range(1, len(P) // 2)]
        return pol

    def state_dict(self) -> dict:
        return {"actor": [p.copy() for p in self.actor.params], "critic": [p.copy() for p in self.critic.params],
                "log_std": self.log_std.copy(), "norm": (self.norm.mean.copy(), self.norm.var.copy(), self.norm.count),
                "lr": self.lr}

    def load_state_dict(self, d: dict) -> None:
        for p, q in zip(self.actor.params, d["actor"]):
            p[...] = q
        for p, q in zip(self.critic.params, d["critic"]):
            p[...] = q
        self.log_std[...] = d["log_std"]
        self.norm.mean, self.norm.var, self.norm.count = d["norm"]
        self.lr = min(d.get("lr", self.lr), self.cfg.lr_max)


def gae(rew, val, done_term, done_trunc, last_val, final_val, gamma, lam):
    """Generalised advantage estimation over (T, N) arrays. At truncation the value of the final
    (pre-reset) observation is bootstrapped; at termination nothing is."""
    T = rew.shape[0]
    adv = np.zeros_like(rew)
    last = np.zeros(rew.shape[1])
    for t in range(T - 1, -1, -1):
        nv = last_val if t == T - 1 else val[t + 1]
        end = done_term[t] | done_trunc[t]
        nv = np.where(done_trunc[t], final_val[t], np.where(done_term[t], 0.0, nv))
        delta = rew[t] + gamma * nv - val[t]
        last = delta + gamma * lam * np.where(end, 0.0, last)
        adv[t] = last
    return adv, adv + val
