"""Powertrain figures from a car log (.mf4): motor + inverter efficiency, loss model, roll-out check.

Uses the signals of the LR26 inverters (two DCUs, A and B): motor speed [rpm], torque [Nm] and the power
the inverter takes from the DC bus [kW] (it equals the sum seen on `act_DCBusPower_R`), plus the pedals.
Efficiency is shaft power / DC power, with shaft power = torque * speed. That assumes `act_Torque` is the
shaft torque in Nm and `act_Power` the DC power in kW; the log does not say so, so confirm it with the
inverter documentation before using the numbers for the model.

Why the roll-out check: speed derived from the motor speed only means something when the wheels are on
the road. `rollout()` finds the zero-torque, no-brake stretches and gives the deceleration there; a car
rolls out at a few tenths of m/s^2, a motor spinning down on a stand or dyno decelerates several times faster.
"""

from __future__ import annotations

import numpy as np

from .mdf4 import MDF4, resample

SIGNALS = {"nA": "act_Speed_R_DCUA", "nB": "act_Speed_R_DCUB", "TA": "act_Torque_R_DCUA", "TB": "act_Torque_R_DCUB",
           "PA": "act_Power_R_DCUA", "PB": "act_Power_R_DCUB", "thr": "PDB_Pedal_Throttle",
           "bpf": "PDB_Pedal_BPF", "bpr": "PDB_Pedal_BPR"}


def load(path_or_mdf, rate_hz: float = 20.0, signals: dict | None = None) -> dict[str, np.ndarray]:
    """The powertrain signals on one time base (`t` plus the keys of SIGNALS). Missing channels are NaN."""
    f = path_or_mdf if isinstance(path_or_mdf, MDF4) else MDF4(path_or_mdf)
    series = {}
    for key, name in (signals or SIGNALS).items():
        found = f.signals(name)
        if found:
            _, t, x = max(found, key=lambda a: len(a[1]))     # the bus that logs it most often
            series[key] = (t, x)
    if not series:
        raise KeyError("none of the powertrain channels is in the file")
    grid, d = resample(series, rate_hz)
    d["t"] = grid
    for key in (signals or SIGNALS):
        d.setdefault(key, np.full(len(grid), np.nan))
    return d


def _om(n):
    return n * 2.0 * np.pi / 60.0


def motor_samples(d: dict, min_power_kw: float = 0.5, min_torque: float = 0.3) -> dict[str, np.ndarray]:
    """Per-motor driving samples: torque [Nm], speed [rpm], DC power [kW], efficiency."""
    T, n, P = [], [], []
    for k in "AB":
        t, w, p = d["T" + k], d["n" + k], d["P" + k]
        m = np.isfinite(t) & np.isfinite(w) & np.isfinite(p) & (p > min_power_kw) & (t > min_torque)
        T.append(t[m]); n.append(w[m]); P.append(p[m])
    T, n, P = np.concatenate(T), np.concatenate(n), np.concatenate(P)
    return {"T": T, "n": n, "P": P, "eff": T * _om(n) / 1e3 / P}


def efficiency_map(s: dict, t_edges=(0.3, 2, 6, 10, 14.5, 40), n_edges=(0, 3000, 6000, 9000, 12000, 25000), min_n: int = 20):
    """Median efficiency per torque x speed bin: (t_edges, n_edges, table[len(t)-1][len(n)-1] with NaN for sparse bins, counts)."""
    nt, nn = len(t_edges) - 1, len(n_edges) - 1
    tab, cnt = np.full((nt, nn), np.nan), np.zeros((nt, nn), int)
    for i in range(nt):
        for j in range(nn):
            m = (s["T"] >= t_edges[i]) & (s["T"] < t_edges[i + 1]) & (s["n"] >= n_edges[j]) & (s["n"] < n_edges[j + 1])
            cnt[i, j] = int(m.sum())
            if cnt[i, j] >= min_n:
                tab[i, j] = float(np.median(s["eff"][m]))
    return t_edges, n_edges, tab, cnt


def loss_fit(s: dict) -> dict:
    """Least squares loss [kW] = k0 + k1 T^2 + k2 n + k3 n^2 (n in krpm, T in Nm), per motor.

    Only valid inside the torque and speed range of the data (`T_max`, `n_max`); the T^2 term must not be
    extrapolated to the motor's full torque from a low-torque test."""
    loss = s["P"] - s["T"] * _om(s["n"]) / 1e3
    nk = s["n"] / 1e3
    X = np.stack([np.ones_like(nk), s["T"] ** 2, nk, nk ** 2], axis=1)
    c, *_ = np.linalg.lstsq(X, loss, rcond=None)
    return {"k": [float(x) for x in c], "rms_kw": float(np.sqrt(np.mean((X @ c - loss) ** 2))),
            "T_max": float(s["T"].max()), "n_max": float(s["n"].max()), "samples": int(len(loss))}


def rollout(d: dict, i_trans: float, circ_ref: float, brake_max: float = 0.5, speed_min: float = 4.0) -> dict:
    """Deceleration at zero torque, throttle and brake, from the motor speed through the gear ratio and wheel size."""
    n = 0.5 * (np.nan_to_num(d["nA"]) + np.nan_to_num(d["nB"]))
    v = n / i_trans * circ_ref / 60.0
    dt = float(d["t"][1] - d["t"][0])
    k = 11
    v = np.convolve(np.pad(v, (k // 2, k // 2), mode="edge"), np.ones(k) / k, mode="valid")
    a = np.gradient(v, dt)
    tq = np.nan_to_num(d["TA"]) + np.nan_to_num(d["TB"])
    free = (v > speed_min) & (tq < 0.3) & (np.nan_to_num(d["thr"]) < 2) & (np.nan_to_num(d["bpf"]) < brake_max) \
        & (np.nan_to_num(d["bpr"]) < brake_max)
    out = {"samples": int(free.sum()), "decel_median": float("nan"), "v_max": float(v.max())}
    if free.sum() >= 20:
        out["decel_median"] = float(-np.median(a[free]))
    return out
