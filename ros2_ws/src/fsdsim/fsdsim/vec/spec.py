"""Vehicle specifications as per-car arrays, and their randomisation for the universal policy.

Every car of the batch can be a different LR26 variant: the parameters the Performance department
sets in LapSim (m, mu, c_z, P_max, ...) are arrays of length N. The universal throttle / brake
policy is trained over a *range* of specifications (domain randomisation, `SpecRanges`) and is told
the specification of the car it drives (`descriptor`), so one policy drives any setup in the range.

Hidden variations (`Realism`) are applied to the car but not told to the policy: track grip,
actuator lags, sensor noise and control latency. The policy has to keep a margin for them, as a
real driver does.
"""

from __future__ import annotations

import copy
import dataclasses
import math
from dataclasses import dataclass, field

import numpy as np

from .backend import Backend

# LapSim fields the batched car uses (fsdsim.perf.PerfParams names)
PERF_FIELDS = ("m", "g", "lf", "lr", "h_cog", "sf", "sr", "e_i", "mux", "muy", "dmux_dFz", "dmuy_dFz", "Fz0",
               "tire_exp", "rho", "A", "c_w", "c_z", "aero_eff", "aero_bal_f", "f_roll", "circ_ref", "i_trans",
               "T_motor", "n_max", "P_max", "P_at_accumulator", "eta_g", "eta_motor", "drive_awd")
# fsdsim VehicleParams fields (not in LapSim) the dynamic model needs
VEH_FIELDS = ("yaw_inertia", "B", "C", "E", "max_steer", "max_steer_rate", "steer_tc", "accel_tc", "max_accel",
              "max_decel", "v_lo", "v_hi", "car_width", "car_half_len", "front_oh", "rear_oh")
# derived / special
EXTRA_FIELDS = ("brake_bias", "bias_ideal", "alpha_peak", "grip_scale")
ALL_FIELDS = PERF_FIELDS + VEH_FIELDS + EXTRA_FIELDS

N_DESCRIPTOR = 15


def alpha_peak(B: float, C: float, E: float) -> float:
    """Slip angle of the Pacejka shape's peak (as fsdsim.dynamics.VehicleModel._find_alpha_peak)."""
    a = np.linspace(0.0, 0.6, 1201)
    ba = B * a
    y = np.sin(C * np.arctan(ba - E * (ba - np.arctan(ba))))
    return float(a[int(np.argmax(y))])


def spec_values(veh) -> dict:
    """Scalar spec of one fsdsim VehicleParams (model: performance) as {field: float}."""
    perf = veh.perf()
    if perf is None:
        raise ValueError("the batched car model needs a performance (LapSim) vehicle")
    d = {k: float(getattr(perf, k)) for k in PERF_FIELDS}
    a, t = veh.actuators, veh.tire
    d.update(yaw_inertia=float(veh.yaw_inertia), B=float(t.B), C=float(t.C), E=float(t.E),
             max_steer=float(a.max_steer), max_steer_rate=float(a.max_steer_rate),
             steer_tc=float(a.steer_time_constant), accel_tc=float(a.accel_time_constant),
             max_accel=float(a.max_accel), max_decel=float(a.max_decel),
             v_lo=float(veh.low_speed_blend[0]), v_hi=float(veh.low_speed_blend[1]),
             car_width=float(veh.footprint.width), car_half_len=0.5 * float(veh.footprint.length),
             front_oh=float(veh.footprint.front_overhang), rear_oh=float(veh.footprint.rear_overhang))
    bb = float(perf.brake_bias)
    d["bias_ideal"] = 1.0 if math.isnan(bb) else 0.0
    d["brake_bias"] = 0.6 if math.isnan(bb) else bb
    d["alpha_peak"] = alpha_peak(t.B, t.C, t.E)
    d["grip_scale"] = 1.0
    return d


# ---------------------------------------------------------------------------- #
@dataclass
class SpecRanges:
    """What the universal policy is trained on, relative to the baseline car (lr26_params.m).

    ("scale", lo, hi): baseline x U(lo, hi);  ("abs", lo, hi): U(lo, hi)
    mu scales mux and muy together; mux_ratio then moves mux against muy;
    weight_front is the static front axle load share lr / (lf + lr) (wheelbase kept);
    drive_awd: probability of an all-wheel-drive car; brake_bias: (probability of a fixed bias, lo, hi),
    otherwise ideal brake balance (LapSim's NaN)."""

    ranges: dict = field(default_factory=lambda: {
        "m": ("scale", 0.85, 1.15),
        "mu": ("scale", 0.85, 1.12),
        "mux_ratio": ("scale", 0.93, 1.07),
        "c_z": ("scale", 0.70, 1.30),
        "c_w": ("scale", 0.75, 1.25),
        "aero_bal_f": ("abs", 0.40, 0.58),
        "h_cog": ("scale", 0.85, 1.20),
        "weight_front": ("abs", 0.42, 0.56),
        "P_max": ("abs", 40000.0, 80000.0),
        "T_motor": ("scale", 0.80, 1.20),
        "i_trans": ("scale", 0.85, 1.15),
        "yaw_inertia": ("scale", 0.80, 1.25),
        "drive_awd": ("prob", 0.25),
        "brake_bias": ("prob_abs", 0.35, 0.50, 0.72),
    })

    def to_dict(self) -> dict:
        return {k: list(v) for k, v in self.ranges.items()}

    @classmethod
    def from_dict(cls, d: dict | None) -> "SpecRanges":
        return cls({k: tuple(v) for k, v in d.items()}) if d else cls()


@dataclass
class Realism:
    """Hidden variations: applied to the car, not told to the policy."""

    grip: tuple = (0.95, 1.03)            # track / tyre condition x mux, muy
    accel_tc: tuple = (0.8, 1.4)          # x actuator lag of the longitudinal command
    steer_tc: tuple = (0.8, 1.3)
    delay_steps: tuple = (0, 1)           # control latency [policy steps of 40 ms], inclusive
    obs_noise: float = 1.0                # x the sensor noise of `features.NOISE`

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def off(cls) -> "Realism":
        return cls(grip=(1.0, 1.0), accel_tc=(1.0, 1.0), steer_tc=(1.0, 1.0), delay_steps=(0, 0), obs_noise=0.0)

    @classmethod
    def from_dict(cls, d: dict | None) -> "Realism":
        return cls(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items()}) if d else cls()


# ---------------------------------------------------------------------------- #
class SpecBatch:
    """Per-car parameter arrays (attributes named like the LapSim / vehicle fields)."""

    def __init__(self, B: Backend, base: dict, n: int):
        self.xb = B                     # backend (self.B is the Pacejka B factor)
        self.n = n
        self.base = dict(base)
        for k in ALL_FIELDS:
            setattr(self, k, B.full(n, float(base[k])))

    @classmethod
    def from_vehicle(cls, B: Backend, veh, n: int = 1) -> "SpecBatch":
        return cls(B, spec_values(veh), n)

    @classmethod
    def from_values(cls, B: Backend, rows: list[dict]) -> "SpecBatch":
        """One car per dict (e.g. a parameter sweep)."""
        sb = cls(B, rows[0], len(rows))
        for k in ALL_FIELDS:
            setattr(sb, k, B.asarray(np.array([float(r[k]) for r in rows])))
        return sb

    def rows_numpy(self) -> list[dict]:
        cols = {k: self.xb.to_numpy(getattr(self, k)) for k in ALL_FIELDS}
        return [{k: float(cols[k][i]) for k in ALL_FIELDS} for i in range(self.n)]

    # ------------------------------------------------------------------ randomisation
    def randomise(self, idx, ranges: SpecRanges, realism: Realism | None = None) -> None:
        """New specs for the cars `idx` (index array), sampled around the baseline."""
        B, base = self.xb, self.base
        k = len(idx) if not B.is_torch else int(idx.numel())
        if k == 0:
            return
        r = ranges.ranges

        def u(lo, hi):
            return B.uniform(lo, hi, k)

        def put(name, val):
            getattr(self, name)[idx] = val

        for name in ("m", "c_z", "c_w", "h_cog", "T_motor", "i_trans", "yaw_inertia"):
            if name in r:
                kind, lo, hi = r[name]
                put(name, base[name] * u(lo, hi) if kind == "scale" else u(lo, hi))
        if "P_max" in r:
            kind, lo, hi = r["P_max"]
            put("P_max", u(lo, hi) if kind == "abs" else base["P_max"] * u(lo, hi))
        if "aero_bal_f" in r:
            _, lo, hi = r["aero_bal_f"]
            put("aero_bal_f", u(lo, hi))
        mu = u(*r["mu"][1:]) if "mu" in r else B.ones(k)
        ratio = u(*r["mux_ratio"][1:]) if "mux_ratio" in r else B.ones(k)
        put("muy", base["muy"] * mu)
        put("mux", base["mux"] * mu * ratio)
        if "weight_front" in r:
            _, lo, hi = r["weight_front"]
            wb = base["lf"] + base["lr"]
            lr = wb * u(lo, hi)
            put("lr", lr)
            put("lf", wb - lr)
        if "drive_awd" in r:
            put("drive_awd", B.to_float(B.rand(k) < r["drive_awd"][1]))
        if "brake_bias" in r:
            _, p, lo, hi = r["brake_bias"]
            fixed = B.rand(k) < p
            put("bias_ideal", B.to_float(~fixed))
            put("brake_bias", u(lo, hi))
        rz = realism or Realism.off()
        put("grip_scale", u(*rz.grip))
        put("accel_tc", base["accel_tc"] * u(*rz.accel_tc))
        put("steer_tc", base["steer_tc"] * u(*rz.steer_tc))

    # ------------------------------------------------------------------ what the policy is told
    def descriptor(self):
        """(N, N_DESCRIPTOR) normalised specification (the nominal car: grip_scale and lags are hidden)."""
        B = self.xb
        l = self.lf + self.lr  # noqa: E741
        aero = self.A * self.aero_eff
        r = self.circ_ref / (2 * math.pi)
        P = B.where(self.P_at_accumulator > 0.5, self.P_max * self.eta_motor, self.P_max)
        F0 = self.T_motor * self.i_trans * self.eta_g / r * (1.0 + self.drive_awd)
        v_top = self.n_max * 2 * math.pi / 60 / self.i_trans * r
        cols = [
            (self.m / 284.0 - 1.0) * 4.0,
            (self.mux / 1.43 - 1.0) * 5.0,
            (self.muy / 1.64 - 1.0) * 5.0,
            (self.c_z * aero / 3.466 - 1.0) * 2.0,
            (self.c_w * aero / 1.153 - 1.0) * 2.0,
            (self.aero_bal_f - 0.485) * 10.0,
            (self.h_cog / 0.265 - 1.0) * 4.0,
            (self.lr / l - 0.47) * 10.0,
            (P / 80000.0 - 1.0) * 2.0,
            (F0 / (self.m * self.g) - 1.0) * 2.0,
            (v_top / 30.0 - 1.0) * 3.0,
            self.drive_awd * 2.0 - 1.0,
            (1.0 - self.bias_ideal) * 2.0 - 1.0,
            (1.0 - self.bias_ideal) * (self.brake_bias - 0.6) * 10.0,
            (self.yaw_inertia / 120.0 - 1.0) * 3.0,
        ]
        return B.stack(cols, axis=1)

    def take(self, idx) -> "SpecBatch":
        """Sub-batch (rows idx)."""
        out = copy.copy(self)
        out.n = len(idx) if not self.xb.is_torch else int(idx.numel())
        for k in ALL_FIELDS:
            setattr(out, k, getattr(self, k)[idx])
        return out


def coverage(veh_values: dict, base: dict, ranges: SpecRanges, tol: float = 0.02) -> list[str]:
    """Why a car is outside what a universal policy was trained on ([] = inside).
    Randomised fields must lie in their range; the others must equal the training baseline."""
    out = []
    r = ranges.ranges
    v, b = veh_values, base

    def chk(name, val, lo, hi):
        if not (lo * (1 - 1e-6) - 1e-9 <= val <= hi * (1 + 1e-6) + 1e-9):
            out.append(f"{name} {val:.4g} outside {lo:.4g} ... {hi:.4g}")

    done = set()
    for name in ("m", "c_z", "c_w", "h_cog", "T_motor", "i_trans", "yaw_inertia", "P_max", "aero_bal_f"):
        if name in r:
            kind, lo, hi = r[name]
            if kind == "scale":
                chk(name, v[name], b[name] * lo, b[name] * hi)
            else:
                chk(name, v[name], lo, hi)
            done.add(name)
    if "mu" in r:
        chk("muy", v["muy"], b["muy"] * r["mu"][1], b["muy"] * r["mu"][2])
        rx = r.get("mux_ratio", ("scale", 1.0, 1.0))
        chk("mux", v["mux"] / v["muy"] * b["muy"], b["mux"] * rx[1], b["mux"] * rx[2])
        done |= {"mux", "muy"}
    if "weight_front" in r:
        wb_v, wb_b = v["lf"] + v["lr"], b["lf"] + b["lr"]
        chk("weight front", v["lr"] / wb_v, r["weight_front"][1], r["weight_front"][2])
        if abs(wb_v - wb_b) > tol * wb_b:
            out.append(f"wheelbase {wb_v:.3f} m (trained {wb_b:.3f} m)")
        done |= {"lf", "lr"}
    if "drive_awd" in r:
        done.add("drive_awd")
    elif abs(v["drive_awd"] - b["drive_awd"]) > 0.5:
        out.append("drive layout (AWD / RWD)")
    if "brake_bias" in r:
        if not v["bias_ideal"]:
            chk("brake_bias", v["brake_bias"], r["brake_bias"][2], r["brake_bias"][3])
        done |= {"brake_bias", "bias_ideal"}
    elif v["bias_ideal"] != b["bias_ideal"] or (not v["bias_ideal"] and abs(v["brake_bias"] - b["brake_bias"]) > 0.02):
        out.append("brake_bias")
    skip = done | {"grip_scale", "accel_tc", "steer_tc", "alpha_peak", "brake_bias", "bias_ideal", "g",
                   "car_width", "car_half_len", "front_oh", "rear_oh"}
    for name in PERF_FIELDS + VEH_FIELDS:
        if name in skip:
            continue
        bv, vv = b[name], v[name]
        if abs(vv - bv) > tol * max(abs(bv), 1e-9):
            out.append(f"{name} {vv:.4g} (trained with {bv:.4g})")
    return out
