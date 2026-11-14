"""Array backend for the batched (GPU) car model: NumPy on the CPU, or PyTorch on CUDA / CPU.

All of fsdsim.vec is written against this small set of functions, so the same equations run
    * on NumPy (float64) - tests here, CPU training, the RL driver inside fsdsim / ROS / Isaac Sim,
    * on PyTorch (float32 on a CUDA GPU) - thousands of cars in parallel, PPO, Isaac Lab.

    B = Backend("numpy")                     # or Backend("torch", device="cuda:0")
    x = B.zeros(4096); y = B.maximum(x, 0.5)
"""

from __future__ import annotations

import math

import numpy as np


class Backend:
    def __init__(self, name: str = "numpy", device: str = "cpu", dtype: str | None = None, seed: int = 0):
        self.name = name
        if name == "numpy":
            self.np = np
            self.float = np.float64 if (dtype or "float64") == "float64" else np.float32
            self.long = np.int64
            self.device = "cpu"
            self._rng = np.random.default_rng(seed)
        elif name == "torch":
            import torch  # noqa: PLC0415

            self.torch = torch
            self.device = torch.device(device)
            self.float = torch.float64 if dtype == "float64" else torch.float32
            self.long = torch.int64
            self._gen = torch.Generator(device=self.device)
            self._gen.manual_seed(seed)
        else:
            raise ValueError(f"backend '{name}': numpy or torch")
        self.is_torch = name == "torch"

    def __repr__(self):
        return f"Backend({self.name}, {self.device}, {self.float})"

    # ------------------------------------------------------------------ creation / conversion
    def asarray(self, x, dtype=None):
        if self.is_torch:
            t = self.torch
            if isinstance(x, t.Tensor):
                return x.to(device=self.device, dtype=dtype or (x.dtype if not x.is_floating_point() else self.float))
            a = np.asarray(x)
            dt = dtype or (self.long if np.issubdtype(a.dtype, np.integer) else
                           t.bool if a.dtype == bool else self.float)
            return t.as_tensor(a, dtype=dt, device=self.device)
        a = np.asarray(x)
        if dtype is not None:
            return a.astype(dtype)
        if np.issubdtype(a.dtype, np.integer) or a.dtype == bool:
            return a.copy()
        return a.astype(self.float)

    def to_numpy(self, x) -> np.ndarray:
        if self.is_torch and isinstance(x, self.torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def zeros(self, shape, dtype=None):
        if self.is_torch:
            return self.torch.zeros(shape, dtype=dtype or self.float, device=self.device)
        return np.zeros(shape, dtype=dtype or self.float)

    def ones(self, shape, dtype=None):
        if self.is_torch:
            return self.torch.ones(shape, dtype=dtype or self.float, device=self.device)
        return np.ones(shape, dtype=dtype or self.float)

    def full(self, shape, value, dtype=None):
        if self.is_torch:
            return self.torch.full(shape if isinstance(shape, tuple) else (shape,), value,
                                   dtype=dtype or self.float, device=self.device)
        return np.full(shape, value, dtype=dtype or self.float)

    def arange(self, a, b=None, dtype=None):
        lo, hi = (0, a) if b is None else (a, b)
        if self.is_torch:
            return self.torch.arange(lo, hi, dtype=dtype or self.long, device=self.device)
        return np.arange(lo, hi, dtype=dtype or self.long)

    def zeros_bool(self, n):
        if self.is_torch:
            return self.torch.zeros(n, dtype=self.torch.bool, device=self.device)
        return np.zeros(n, dtype=bool)

    def copy(self, x):
        return x.clone() if self.is_torch else x.copy()

    def stack(self, xs, axis=0):
        return self.torch.stack(list(xs), dim=axis) if self.is_torch else np.stack(xs, axis=axis)

    def concat(self, xs, axis=0):
        return self.torch.cat(list(xs), dim=axis) if self.is_torch else np.concatenate(xs, axis=axis)

    def to_long(self, x):
        return x.long() if self.is_torch else x.astype(np.int64)

    def to_float(self, x):
        return x.to(self.float) if self.is_torch else x.astype(self.float)

    # ------------------------------------------------------------------ elementwise
    def sin(self, x):
        return self.torch.sin(x) if self.is_torch else np.sin(x)

    def cos(self, x):
        return self.torch.cos(x) if self.is_torch else np.cos(x)

    def tan(self, x):
        return self.torch.tan(x) if self.is_torch else np.tan(x)

    def atan(self, x):
        return self.torch.atan(x) if self.is_torch else np.arctan(x)

    def atan2(self, y, x):
        return self.torch.atan2(y, x) if self.is_torch else np.arctan2(y, x)

    def sqrt(self, x):
        return self.torch.sqrt(x) if self.is_torch else np.sqrt(x)

    def tanh(self, x):
        return self.torch.tanh(x) if self.is_torch else np.tanh(x)

    def exp(self, x):
        return self.torch.exp(x) if self.is_torch else np.exp(x)

    def log(self, x):
        return self.torch.log(x) if self.is_torch else np.log(x)

    def abs(self, x):
        return self.torch.abs(x) if self.is_torch else np.abs(x)

    def hypot(self, x, y):
        return self.torch.hypot(x, y) if self.is_torch else np.hypot(x, y)

    def floor(self, x):
        return self.torch.floor(x) if self.is_torch else np.floor(x)

    def round(self, x):
        return self.torch.round(x) if self.is_torch else np.round(x)

    def isnan(self, x):
        return self.torch.isnan(x) if self.is_torch else np.isnan(x)

    def where(self, c, a, b):
        if self.is_torch:
            t = self.torch
            if not isinstance(a, t.Tensor):
                a = t.as_tensor(a, dtype=self.float if isinstance(a, float) else None, device=self.device)
            if not isinstance(b, t.Tensor):
                b = t.as_tensor(b, dtype=self.float if isinstance(b, float) else None, device=self.device)
            if a.dtype != b.dtype:
                if a.is_floating_point() or b.is_floating_point():
                    a, b = a.to(self.float), b.to(self.float)
            return t.where(c, a, b)
        return np.where(c, a, b)

    def maximum(self, a, b):
        if self.is_torch:
            t = self.torch
            ta, tb = isinstance(a, t.Tensor), isinstance(b, t.Tensor)
            if ta and tb:
                return t.maximum(a, b)
            if ta:
                return t.clamp(a, min=b)
            if tb:
                return t.clamp(b, min=a)
            return max(a, b)
        return np.maximum(a, b)

    def minimum(self, a, b):
        if self.is_torch:
            t = self.torch
            ta, tb = isinstance(a, t.Tensor), isinstance(b, t.Tensor)
            if ta and tb:
                return t.minimum(a, b)
            if ta:
                return t.clamp(a, max=b)
            if tb:
                return t.clamp(b, max=a)
            return min(a, b)
        return np.minimum(a, b)

    def clip(self, x, lo, hi):
        return self.minimum(self.maximum(x, lo), hi)

    # ------------------------------------------------------------------ reductions
    def max(self, x, axis):
        return self.torch.amax(x, dim=axis) if self.is_torch else np.max(x, axis=axis)

    def min(self, x, axis):
        return self.torch.amin(x, dim=axis) if self.is_torch else np.min(x, axis=axis)

    def sum(self, x, axis=None):
        if self.is_torch:
            return x.sum() if axis is None else x.sum(dim=axis)
        return np.sum(x, axis=axis)

    def mean(self, x, axis=None):
        if self.is_torch:
            return x.mean() if axis is None else x.mean(dim=axis)
        return np.mean(x, axis=axis)

    def argmin(self, x, axis):
        return self.torch.argmin(x, dim=axis) if self.is_torch else np.argmin(x, axis=axis)

    def argmax(self, x, axis):
        return self.torch.argmax(x, dim=axis) if self.is_torch else np.argmax(x, axis=axis)

    def any(self, x, axis=None):
        if self.is_torch:
            return x.any() if axis is None else x.any(dim=axis)
        return np.any(x, axis=axis)

    def count(self, mask) -> int:
        return int(mask.sum().item()) if self.is_torch else int(np.count_nonzero(mask))

    def nonzero(self, mask):
        """Indices where a 1-D mask is true."""
        if self.is_torch:
            return self.torch.nonzero(mask, as_tuple=False).flatten()
        return np.nonzero(mask)[0]

    # ------------------------------------------------------------------ random
    def rand(self, n):
        if self.is_torch:
            return self.torch.rand(n, generator=self._gen, device=self.device, dtype=self.float)
        return self._rng.random(n).astype(self.float)

    def randn(self, n):
        if self.is_torch:
            return self.torch.randn(n, generator=self._gen, device=self.device, dtype=self.float)
        return self._rng.standard_normal(n).astype(self.float)

    def randint(self, lo, hi, n):
        if self.is_torch:
            return self.torch.randint(lo, hi, (n,), generator=self._gen, device=self.device)
        return self._rng.integers(lo, hi, n)

    def uniform(self, lo, hi, n):
        return lo + (hi - lo) * self.rand(n)


PI = math.pi
