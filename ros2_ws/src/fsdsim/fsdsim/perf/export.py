"""Write the speed trace of a LapSim event as a CSV with the signal names of LionsLTS.

The Performance department's LionsLTS exports a CSV per run (`framework/result_exporter.py`) that goes
into their telemetry tools (Marple). Writing the same column names lets a LapSim / simulator run be loaded
there next to LionsLTS runs and next to the car's logs. Only the signals this model has are written:

    time, cumulative distance, raceline x, raceline y, velocity, reached ax, init ay, curvature,
    powertrain force, total resistance force, aerodynamic drag force, rolling resistance force,
    consumed energy [J], energy storage state [J] (from `energy_start_j`), accumulator power [W]

(`reached ax` is the longitudinal acceleration, `init ay` the lateral one, as named by LionsLTS.)
"""

from __future__ import annotations

import csv
import os

import numpy as np

from .lapsim import _resistance

COLUMNS = ["time", "cumulative distance", "raceline x", "raceline y", "velocity", "reached ax", "init ay", "curvature",
           "powertrain force", "total resistance force", "aerodynamic drag force", "rolling resistance force",
           "consumed energy", "energy storage state", "accumulator power"]


def run_columns(res: dict, energy_start_j: float = 0.0) -> dict[str, np.ndarray]:
    """The LionsLTS-named columns of one `run_event` result (one lap)."""
    veh, v = res["veh"], res["v"]
    resist = _resistance(veh, v)
    q = 0.5 * veh.rho * v * v
    drag = q * veh.c_w * veh.A * veh.aero_eff
    roll = veh.f_roll * (veh.m * veh.g + q * veh.c_z * veh.A * veh.aero_eff)
    f_tot = veh.m * veh.e_i * res["ax"] + resist
    k = np.asarray(res["k"], float)
    k = np.concatenate([k, k[-1:]]) if len(k) == len(v) - 1 else k
    return {"time": res["t"], "cumulative distance": res["s"], "raceline x": res["x"], "raceline y": res["y"],
            "velocity": v, "reached ax": res["ax"], "init ay": res["ay"], "curvature": k,
            "powertrain force": f_tot, "total resistance force": resist, "aerodynamic drag force": drag,
            "rolling resistance force": roll, "consumed energy": res["E"],
            "energy storage state": energy_start_j - res["E"], "accumulator power": res["P_batt"]}


def write_csv(res: dict, path: str, energy_start_j: float = 0.0, float_format: str = "{:.3f}") -> int:
    cols = run_columns(res, energy_start_j)
    n = len(cols["time"])
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for i in range(n):
            w.writerow([float_format.format(float(cols[c][i])) for c in COLUMNS])
    return n


def closed_loop_columns(log: dict, veh, energy_start_j: float = 0.0) -> dict[str, np.ndarray]:
    """The same LionsLTS-named columns from a closed-loop simulator run (tools/closed_loop_eval.py log), from
    the moment the car moves. `velocity` is the driven speed, `curvature` yaw rate / speed, forces are the tyre
    forces the dynamic model applied, energy is the car's energy meter (accumulator side)."""
    a = {k: np.asarray(v, float) for k, v in log.items()}
    i0 = int(np.argmax(a["v"] > 0.05))
    a = {k: v[i0:] for k, v in a.items()}
    v = a["v"]
    t = a["t"] - a["t"][0]
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(a["x"]), np.diff(a["y"])))])
    q = 0.5 * veh.rho * v * v
    drag = q * veh.c_w * veh.A * veh.aero_eff
    roll = veh.f_roll * (veh.m * veh.g + q * veh.c_z * veh.A * veh.aero_eff) * np.tanh(v / 0.2)
    e = a["energy"] - a["energy"][0]
    return {"time": t, "cumulative distance": s, "raceline x": a["x"], "raceline y": a["y"], "velocity": v,
            "reached ax": a["accel"], "init ay": a["ay"], "curvature": a["r"] / np.maximum(v, 0.5),
            "powertrain force": a["fx"], "total resistance force": drag + roll, "aerodynamic drag force": drag,
            "rolling resistance force": roll, "consumed energy": e, "energy storage state": energy_start_j - e,
            "accumulator power": a["power"]}


def write_closed_loop_csv(log: dict, veh, path: str, energy_start_j: float = 0.0, float_format: str = "{:.3f}") -> int:
    cols = closed_loop_columns(log, veh, energy_start_j)
    n = len(cols["time"])
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for i in range(n):
            w.writerow([float_format.format(float(cols[c][i])) for c in COLUMNS])
    return n
