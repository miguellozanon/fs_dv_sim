"""Motor + inverter efficiency as a map over torque and speed (instead of the constant eta_motor).

    m = EfficiencyMap.from_loss_fit(k, t_max=38, n_max=21000)      # k from logdata.powertrain.loss_fit
    m = EfficiencyMap.load("lapsim/configs/eta_map_D1F58.json")
    m.eta(T_Nm, n_rpm)

T is the torque of one motor set (LapSim: per driven axle) and n its speed. Outside the measured range the map
is clamped to its edge, and `valid` says how far the data went: the D1F58 log only reaches 14 Nm, so above that
the map extrapolates the fitted loss model and is indicative only.

The map is used for the ENERGY (accumulator power = wheel power / (eta_g * eta(T, n))): in LapSim's signals
(lapsim._signals) when a map is active, and in the dynamic car's energy meter. The power limit keeps the
constant eta_motor (P_at_accumulator), as the MATLAB LapSim does. MATLAB itself does not read the map.

Activate: vehicle.yaml `efficiency_map: <file>` (simulator), `efficiency.set_active(...)`, tools' --eta-map, or
the environment variable FSG_ETA_MAP.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np

_active = None
_env_loaded = False


class EfficiencyMap:
    def __init__(self, torque, rpm, eta, valid: dict | None = None, source: str = "", motors_per_set: float = 1.0):
        self.motors_per_set = float(motors_per_set)    # eta() takes the torque of a LapSim motor set (one driven axle);
        # the table is per motor: a set of 2 motors (LR26 rear: DCUA + DCUB) splits the torque in two
        self.torque = np.asarray(torque, float)        # [Nm] ascending, >= 0 (per motor)
        self.rpm = np.asarray(rpm, float)              # [1/min] ascending
        self.table = np.clip(np.asarray(eta, float), 0.05, 0.999)     # (len(torque), len(rpm))
        if self.table.shape != (len(self.torque), len(self.rpm)):
            raise ValueError("efficiency table shape does not match the torque / rpm axes")
        self.valid = dict(valid or {})
        self.source = source

    # ------------------------------------------------------------------ #
    def eta(self, torque, rpm):
        """Bilinear interpolation, clamped at the edges. Scalars or arrays."""
        T = np.clip(np.abs(np.asarray(torque, float)) / self.motors_per_set, self.torque[0], self.torque[-1])
        n = np.clip(np.abs(np.asarray(rpm, float)), self.rpm[0], self.rpm[-1])
        i = np.clip(np.searchsorted(self.torque, T) - 1, 0, len(self.torque) - 2)
        j = np.clip(np.searchsorted(self.rpm, n) - 1, 0, len(self.rpm) - 2)
        ft = (T - self.torque[i]) / (self.torque[i + 1] - self.torque[i])
        fn = (n - self.rpm[j]) / (self.rpm[j + 1] - self.rpm[j])
        tb = self.table
        out = (tb[i, j] * (1 - ft) * (1 - fn) + tb[i + 1, j] * ft * (1 - fn)
               + tb[i, j + 1] * (1 - ft) * fn + tb[i + 1, j + 1] * ft * fn)
        return float(out) if np.ndim(out) == 0 else out

    # ------------------------------------------------------------------ #
    @classmethod
    def constant(cls, eta: float) -> "EfficiencyMap":
        return cls([0.0, 1000.0], [0.0, 1e5], [[eta, eta], [eta, eta]], source=f"constant {eta}")

    @classmethod
    def from_loss_fit(cls, k, t_max: float = 38.0, n_max: float = 21000.0, valid: dict | None = None,
                      source: str = "", nt: int = 20, nn: int = 22, motors_per_set: float = 1.0) -> "EfficiencyMap":
        """From loss [kW] = k0 + k1 T^2 + k2 n + k3 n^2 (n in krpm), the model of logdata.powertrain.loss_fit."""
        T = np.linspace(0.0, t_max, nt)
        n = np.linspace(0.0, n_max, nn)
        TT, NN = np.meshgrid(T, n, indexing="ij")
        p_mech = TT * NN * 2 * math.pi / 60 / 1e3                         # kW
        loss = np.maximum(k[0] + k[1] * TT ** 2 + k[2] * NN / 1e3 + k[3] * (NN / 1e3) ** 2, 0.0)
        eta = np.where(p_mech > 1e-6, p_mech / np.maximum(p_mech + loss, 1e-9), 0.05)
        eta[0, :] = eta[1, :]                                             # zero torque / speed: no power, use the neighbour
        eta[:, 0] = eta[:, 1]
        return cls(T, n, eta, valid, source, motors_per_set)

    def to_dict(self) -> dict:
        return {"torque_Nm": self.torque.tolist(), "rpm": self.rpm.tolist(), "eta": np.round(self.table, 4).tolist(),
                "valid": self.valid, "source": self.source, "motors_per_set": self.motors_per_set,
                "note": "motor + inverter efficiency, torque per motor; clamped outside the axes"}

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=1)

    @classmethod
    def load(cls, path: str) -> "EfficiencyMap":
        with open(resolve(path), encoding="utf-8") as f:
            d = json.load(f)
        return cls(d["torque_Nm"], d["rpm"], d["eta"], d.get("valid"), d.get("source", os.path.basename(path)),
                   d.get("motors_per_set", 1.0))


def resolve(path: str) -> str:
    """A map file: absolute, relative to the working directory, or relative to the repository."""
    if os.path.isabs(path) or os.path.exists(path):
        return path
    d = os.path.dirname(os.path.realpath(__file__))
    for _ in range(8):
        cand = os.path.join(d, path)
        if os.path.exists(cand):
            return cand
        d = os.path.dirname(d)
    raise FileNotFoundError(f"efficiency map '{path}' not found (absolute path, or relative to the repository)")


def set_active(m) -> None:
    """Use this map (EfficiencyMap, a file name, or None for the constant eta_motor) in LapSim's energy."""
    global _active, _env_loaded
    _active = EfficiencyMap.load(m) if isinstance(m, str) and m else (m or None)
    _env_loaded = True


def active():
    global _active, _env_loaded
    if not _env_loaded:
        _env_loaded = True
        p = os.environ.get("FSG_ETA_MAP")
        if p:
            _active = EfficiencyMap.load(p)
    return _active
