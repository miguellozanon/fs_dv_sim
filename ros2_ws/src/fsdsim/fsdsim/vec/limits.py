"""LapSim limits for N cars at once (batched fsdsim.perf.model / qss), as lookup tables.

For every car (its own specification) at the start of an episode:
    corner speed    v_c(k)        highest steady-state speed on curvature k (both axles u <= 1,
                                  drive force >= resistance, n_max)        - qss.corner_speed
    accel / brake   a(v, k)       full traction / braking acceleration at speed v on curvature k,
                                  4-step load-transfer iteration           - model.ax_limits
Tables on a fixed (v, k) grid keep the per-step cost to a few gathers, on the CPU or the GPU.
"""

from __future__ import annotations

import math

import numpy as np

from .backend import Backend
from .spec import SpecBatch

# curvature grid: 0, then log-spaced 1e-3 ... 0.6 1/m (radius 1000 m ... 1.7 m)
K_LOG0, K_LOG1, N_K = -3.0, math.log10(0.6), 40
K_GRID = np.concatenate([[0.0], np.logspace(K_LOG0, K_LOG1, N_K - 1)])
V_GRID = np.linspace(0.0, 40.0, 17)          # 2.5 m/s steps


class _P:
    """Spec arrays reshaped to broadcast against (N, ...) inputs."""

    def __init__(self, spec: SpecBatch, extra_dims: int, util_x: float = 1.0, util_y: float = 1.0):
        self._s, self._e = spec, extra_dims
        self.util_x, self.util_y = util_x, util_y

    def __getattr__(self, k):
        a = getattr(self._s, k)
        return a[(slice(None),) + (None,) * self._e]


def _powertrain(B: Backend, p, v):
    r = p.circ_ref / (2 * math.pi)
    v_max = p.n_max * 2 * math.pi / 60 / p.i_trans * r
    P = B.where(p.P_at_accumulator > 0.5, p.P_max * p.eta_motor, p.P_max)
    vv = B.maximum(v, 0.1)
    T = B.minimum(p.T_motor, P / (vv / r * p.i_trans))
    F_set = B.where(v > v_max * (1 + 1e-9), 0.0, T * p.i_trans * p.eta_g / r)
    return F_set, P * p.eta_g / vv, v_max


def _resistance(B: Backend, p, v):
    DF = 0.5 * p.rho * p.c_z * p.A * p.aero_eff * v * v
    return 0.5 * p.rho * p.c_w * p.A * p.aero_eff * v * v + p.f_roll * (p.m * p.g + DF)


def _axle(B: Backend, p, v, ax, ay):
    """perf.model.axle_state: (Fx_rem_f, Fx_rem_r, u_f, u_r)."""
    l = p.lf + p.lr  # noqa: E741
    DF = 0.5 * p.rho * p.c_z * p.A * p.aero_eff * v * v
    Ff = p.m * p.g * p.lr / l + DF * p.aero_bal_f - p.m * ax * p.h_cog / l
    Fr = p.m * p.g * p.lf / l + DF * (1 - p.aero_bal_f) + p.m * ax * p.h_cog / l
    dyf = p.m * ay * p.h_cog / p.sf * p.lr / l
    dyr = p.m * ay * p.h_cog / p.sr * p.lf / l

    def pots(fz):
        fz = B.maximum(fz, 0.0)
        return (B.maximum(fz * (p.mux + p.dmux_dFz * (fz - p.Fz0)) * p.util_x, 0.0),
                B.maximum(fz * (p.muy + p.dmuy_dFz * (fz - p.Fz0)) * p.util_y, 0.0))

    px1, py1 = pots(Ff / 2 - dyf)
    px2, py2 = pots(Ff / 2 + dyf)
    px3, py3 = pots(Fr / 2 - dyr)
    px4, py4 = pots(Fr / 2 + dyr)
    Fyt = B.abs(p.m * ay)
    u_f = Fyt * p.lr / l / B.maximum(py1 + py2, 1e-9)
    u_r = Fyt * p.lf / l / B.maximum(py3 + py4, 1e-9)
    e = p.tire_exp
    rem_f = (px1 + px2) * B.maximum(1 - B.minimum(u_f, 1.0) ** e, 0.0) ** (1 / e)
    rem_r = (px3 + px4) * B.maximum(1 - B.minimum(u_r, 1.0) ** e, 0.0) ** (1 / e)
    return rem_f, rem_r, u_f, u_r


def _drive_force(B, p, rem_f, rem_r, F_set, F_cap):
    awd = p.drive_awd > 0.5
    F = B.where(awd, B.minimum(F_set, rem_f) + B.minimum(F_set, rem_r), B.minimum(F_set, rem_r))
    return B.minimum(F, F_cap)


def _brake_force(B, p, rem_f, rem_r):
    b = p.brake_bias
    fixed = B.minimum(rem_f / B.maximum(b, 1e-6), rem_r / B.maximum(1 - b, 1e-6))
    return B.where(p.bias_ideal > 0.5, rem_f + rem_r, fixed)


def corner_speed_table(B: Backend, spec: SpecBatch, util_x: float = 1.0, util_y: float = 1.0, iters: int = 26):
    """(N, len(K_GRID)) highest steady-state speed per curvature (qss.corner_speed, bisection)."""
    p = _P(spec, 1, util_x, util_y)
    k = B.asarray(K_GRID)[None, :] + B.zeros((spec.n, 1))
    _, _, v_max = _powertrain(B, p, B.zeros((spec.n, 1)))
    v_max = v_max + B.zeros((spec.n, len(K_GRID)))

    def ok(v):
        ay = v * v * k
        rem_f, rem_r, u_f, u_r = _axle(B, p, v, 0.0 * v, ay)
        F_set, F_cap, vm = _powertrain(B, p, v)
        good = (u_f <= 1.0) & (u_r <= 1.0) & (v <= vm)
        return good & (_drive_force(B, p, rem_f, rem_r, F_set, F_cap) >= _resistance(B, p, v))

    lo = B.zeros((spec.n, len(K_GRID)))
    hi = B.copy(v_max)
    top_ok = ok(hi)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        good = ok(mid)
        lo = B.where(good, mid, lo)
        hi = B.where(good, hi, mid)
    return B.where(top_ok, v_max, lo)


def ax_tables(B: Backend, spec: SpecBatch, util_x: float = 1.0, util_y: float = 1.0):
    """(acc, dec) each (N, len(V_GRID), len(K_GRID)): model.ax_limits at every grid point (dec < 0)."""
    p = _P(spec, 2, util_x, util_y)
    n = spec.n
    v = B.asarray(V_GRID)[None, :, None] + B.zeros((n, len(V_GRID), len(K_GRID)))
    k = B.asarray(K_GRID)[None, None, :] + B.zeros((n, len(V_GRID), len(K_GRID)))
    ay = v * v * k
    meff = p.m * p.e_i
    Fres = _resistance(B, p, v)
    F_set, F_cap, _ = _powertrain(B, p, v)
    acc = B.zeros(v.shape)
    for _ in range(4):
        rem_f, rem_r, _, _ = _axle(B, p, v, acc, ay)
        acc = (_drive_force(B, p, rem_f, rem_r, F_set, F_cap) - Fres) / meff
    dec = B.zeros(v.shape)
    for _ in range(4):
        rem_f, rem_r, _, _ = _axle(B, p, v, dec, ay)
        dec = -(_brake_force(B, p, rem_f, rem_r) + Fres) / meff
    return acc, dec


# ---------------------------------------------------------------------------- #
# table lookups
# ---------------------------------------------------------------------------- #
def _k_index(B: Backend, k):
    """Fractional index of |k| on K_GRID."""
    k = B.abs(k)
    step = (K_LOG1 - K_LOG0) / (N_K - 2)
    lin = k / 10 ** K_LOG0                                     # 0 ... 1 between K_GRID[0] and [1]
    lg = 1.0 + (B.log(B.maximum(k, 1e-12)) / math.log(10.0) - K_LOG0) / step
    return B.clip(B.where(k < 10 ** K_LOG0, lin, lg), 0.0, N_K - 1.0 - 1e-6)


def _v_index(B: Backend, v):
    return B.clip(v / (V_GRID[1] - V_GRID[0]), 0.0, len(V_GRID) - 1.0 - 1e-6)


def _gather_rows(B: Backend, table, j):
    """table (N, G), j (N, K) int -> (N, K)."""
    n = table.shape[0]
    rows = B.arange(n)[:, None] + 0 * j
    return table[rows, j]


def lookup_k(B: Backend, table, k):
    """Linear interpolation of a per-car table (N, len(K_GRID)) at curvatures k (N, K)."""
    f = _k_index(B, k)
    i0 = B.to_long(B.floor(f))
    w = f - B.to_float(i0)
    return (1 - w) * _gather_rows(B, table, i0) + w * _gather_rows(B, table, i0 + 1)


def lookup_vk(B: Backend, table, v, k):
    """Bilinear interpolation of (N, len(V_GRID), len(K_GRID)) at (v, k), both (N, K)."""
    n, nv, nk = table.shape
    flat = table.reshape(n, nv * nk)
    fv, fk = _v_index(B, v), _k_index(B, k)
    iv, ik = B.to_long(B.floor(fv)), B.to_long(B.floor(fk))
    wv, wk = fv - B.to_float(iv), fk - B.to_float(ik)

    def g(a, b):
        return _gather_rows(B, flat, a * nk + b)

    return ((1 - wv) * ((1 - wk) * g(iv, ik) + wk * g(iv, ik + 1)) +
            wv * ((1 - wk) * g(iv + 1, ik) + wk * g(iv + 1, ik + 1)))


class LimitTables:
    """The tables of all cars; rebuild rows when a car gets a new specification (episode reset)."""

    def __init__(self, B: Backend, spec: SpecBatch, teacher: bool = True):
        self.B = B
        self.spec = spec
        self.vc = B.zeros((spec.n, len(K_GRID)))             # corner speed, tyre limit
        self.acc = B.zeros((spec.n, len(V_GRID), len(K_GRID)))
        self.dec = B.zeros((spec.n, len(V_GRID), len(K_GRID)))
        # reference-driver margins (teacher for the imitation warm start): 80 % grip, braking x 0.7
        self.vc_ref = B.zeros((spec.n, len(K_GRID)))
        self.dec_ref = B.zeros((spec.n, len(V_GRID), len(K_GRID)))
        self.update(B.arange(spec.n), teacher=teacher)

    def update(self, idx, teacher: bool = True) -> None:
        B = self.B
        if (int(idx.numel()) if B.is_torch else len(idx)) == 0:
            return
        sub = self.spec.take(idx)
        self.vc[idx] = corner_speed_table(B, sub)
        acc, dec = ax_tables(B, sub)
        self.acc[idx], self.dec[idx] = acc, dec
        if teacher:
            self.vc_ref[idx] = corner_speed_table(B, sub, 0.8, 0.8)
            self.dec_ref[idx] = ax_tables(B, sub, 0.8 * 0.7, 0.8)[1]
