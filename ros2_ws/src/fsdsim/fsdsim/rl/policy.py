"""Throttle / brake policy: a small multilayer perceptron in plain NumPy.

No PyTorch at run time: Isaac Sim's Python, the ROS nodes and the dashboard only need NumPy.
A policy is saved as <dir>/policy.npz (weights) + <dir>/policy.json (what it was trained for).
"""

from __future__ import annotations

import json
import os

import numpy as np


def _act_fn(name: str):
    if name == "tanh":
        return np.tanh
    if name == "elu":
        return lambda z: np.where(z > 0.0, z, np.expm1(np.minimum(z, 0.0)))
    if name == "relu":
        return lambda z: np.maximum(z, 0.0)
    raise ValueError(f"activation '{name}' not supported (tanh, elu, relu)")


class MLPPolicy:
    """obs (n_in) -> tanh hidden layers -> tanh output in [-1, 1] (+1 full throttle, -1 full brake).

    Policies trained elsewhere (RSL-RL in Isaac Lab) may use another hidden activation (elu, relu)
    and a linear output clipped to [-1, 1] (output 'clip'); both are stored in policy.json."""

    def __init__(self, n_in: int, hidden=(32, 32), seed: int = 0):
        self.sizes = [n_in, *hidden, 1]
        rng = np.random.default_rng(seed)
        self.layers = []
        for a, b in zip(self.sizes[:-1], self.sizes[1:]):
            self.layers.append([rng.standard_normal((a, b)) / np.sqrt(a), np.zeros(b)])
        self.layers[-1][0] *= 0.1      # small first actions
        self.features_version = 1      # what the policy expects (control.rl_driver); set by load()
        self.universal = False
        self.activation = "tanh"       # hidden layers
        self.output = "tanh"           # tanh | clip (linear, clipped to [-1, 1])

    # ------------------------------------------------------------------ #
    @property
    def n_params(self) -> int:
        return sum(W.size + b.size for W, b in self.layers)

    def get_params(self) -> np.ndarray:
        return np.concatenate([np.concatenate([W.ravel(), b]) for W, b in self.layers])

    def set_params(self, theta: np.ndarray) -> "MLPPolicy":
        i = 0
        for layer in self.layers:
            W, b = layer
            layer[0] = theta[i:i + W.size].reshape(W.shape)
            i += W.size
            layer[1] = theta[i:i + b.size].copy()
            i += b.size
        return self

    def copy(self) -> "MLPPolicy":
        p = MLPPolicy(self.sizes[0], tuple(self.sizes[1:-1]))
        return p.set_params(self.get_params())

    # ------------------------------------------------------------------ #
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Batch forward (N x n_in) -> (N,)"""
        h = np.asarray(x, float)
        if self.activation == "tanh" and self.output == "tanh":
            for W, b in self.layers:
                h = np.tanh(h @ W + b)
            return h[..., 0]
        f = _act_fn(self.activation)
        for W, b in self.layers[:-1]:
            h = f(h @ W + b)
        W, b = self.layers[-1]
        z = h @ W + b
        return (np.tanh(z) if self.output == "tanh" else np.clip(z, -1.0, 1.0))[..., 0]

    def act(self, obs: np.ndarray) -> float:
        if self.activation == "tanh" and self.output == "tanh":
            h = np.asarray(obs, float)
            for W, b in self.layers:
                h = np.tanh(h @ W + b)
            return float(h[0])
        return float(self.forward(np.asarray(obs, float)[None, :])[0])

    # ------------------------------------------------------------------ #
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 60, lr: float = 3e-3, batch: int = 256,
            seed: int = 0, log=None) -> float:
        """Supervised regression (MSE, Adam) - used to imitate the reference driver before RL."""
        rng = np.random.default_rng(seed)
        X, y = np.asarray(X, float), np.clip(np.asarray(y, float), -0.98, 0.98)
        m = [np.zeros_like(p) for layer in self.layers for p in layer]
        v = [np.zeros_like(p) for p in m]
        t = 0
        loss = float("nan")
        for ep in range(epochs):
            idx = rng.permutation(len(X))
            for k in range(0, len(X), batch):
                j = idx[k:k + batch]
                hs = [X[j]]
                for W, b in self.layers:
                    hs.append(np.tanh(hs[-1] @ W + b))
                err = hs[-1][:, 0] - y[j]
                g = (2.0 / len(j)) * err[:, None] * (1 - hs[-1] ** 2)
                grads = []
                for li in range(len(self.layers) - 1, -1, -1):
                    W, _ = self.layers[li]
                    grads.append((hs[li].T @ g, g.sum(0)))
                    if li:
                        g = (g @ W.T) * (1 - hs[li] ** 2)
                grads = [p for pair in reversed(grads) for p in pair]
                t += 1
                params = [p for layer in self.layers for p in layer]
                for n, (p, gr) in enumerate(zip(params, grads)):
                    m[n] = 0.9 * m[n] + 0.1 * gr
                    v[n] = 0.999 * v[n] + 0.001 * gr * gr
                    p -= lr * (m[n] / (1 - 0.9 ** t)) / (np.sqrt(v[n] / (1 - 0.999 ** t)) + 1e-8)
            loss = float(np.mean((self.forward(X) - y) ** 2))
            if log and (ep % 10 == 9 or ep == epochs - 1):
                log(f"  imitation epoch {ep + 1}/{epochs}: mse {loss:.4f}")
        return loss

    # ------------------------------------------------------------------ #
    def save(self, folder: str, meta: dict) -> str:
        os.makedirs(folder, exist_ok=True)
        arrays = {f"W{i}": W for i, (W, _) in enumerate(self.layers)}
        arrays.update({f"b{i}": b for i, (_, b) in enumerate(self.layers)})
        np.savez(os.path.join(folder, "policy.npz"), **arrays)
        with open(os.path.join(folder, "policy.json"), "w", encoding="utf-8") as f:
            json.dump({**meta, "sizes": self.sizes, "activation": self.activation, "output": self.output}, f,
                      indent=2, default=str)
        return folder

    @classmethod
    def load(cls, folder: str) -> tuple["MLPPolicy", dict]:
        with open(os.path.join(folder, "policy.json"), encoding="utf-8") as f:
            meta = json.load(f)
        sizes = meta["sizes"]
        p = cls(sizes[0], tuple(sizes[1:-1]))
        with np.load(os.path.join(folder, "policy.npz")) as z:
            p.layers = [[z[f"W{i}"], z[f"b{i}"]] for i in range(len(sizes) - 1)]
        p.features_version = int(meta.get("features_version", 1))
        p.activation = meta.get("activation", "tanh")
        p.output = meta.get("output", "tanh")
        p.universal = bool(meta.get("universal", False))
        return p, meta
