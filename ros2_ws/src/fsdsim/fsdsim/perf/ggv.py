"""ggv map of the car: the longitudinal acceleration limits against speed and lateral acceleration.

    table = ggv(veh, speeds=np.arange(5, 31, 2.5), n_ay=13)
    table["ax_acc"][i, j]   drive limit [m/s^2] at speeds[i], ay[i, j]
    table["ax_dec"][i, j]   brake limit (negative)

Each row runs from ay = 0 to the cornering limit of that speed (LapSim's lts_ax_limits at v, k = ay / v^2), so
the map is exactly what the QSS solver uses. It is the input other tools ask for (minimum-time optimal control,
MPC speed planners, the Performance department's ggv-based solvers)."""

from __future__ import annotations

import numpy as np

from . import model, qss


def ay_limit(veh, v: float) -> float:
    """Largest lateral acceleration the car holds at speed v (steady state)."""
    lo, hi = 0.0, 60.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if qss.steady_ok(veh, v, mid / max(v * v, 1e-6)):
            lo = mid
        else:
            hi = mid
    return lo


def ggv(veh, speeds=None, n_ay: int = 13) -> dict:
    speeds = np.asarray(speeds if speeds is not None else np.arange(5.0, 32.5, 2.5), float)
    ay = np.zeros((len(speeds), n_ay))
    acc, dec = np.zeros_like(ay), np.zeros_like(ay)
    for i, v in enumerate(speeds):
        ay[i] = np.linspace(0.0, ay_limit(veh, float(v)), n_ay)
        for j, a in enumerate(ay[i]):
            a_acc, a_dec, _, _ = model.ax_limits(veh, float(v), float(a) / max(v * v, 1e-6))
            acc[i, j], dec[i, j] = a_acc, a_dec
        acc[i, -1] = min(acc[i, -1], acc[i, -2])
    return {"v": speeds, "ay": ay, "ax_acc": acc, "ax_dec": dec}


def write_csv(table: dict, path: str) -> int:
    rows = 0
    with open(path, "w", encoding="utf-8") as f:
        f.write("v_mps,ay_mps2,ax_acc_mps2,ax_dec_mps2\n")
        for i, v in enumerate(table["v"]):
            for j in range(table["ay"].shape[1]):
                f.write(f"{v:.3f},{table['ay'][i, j]:.4f},{table['ax_acc'][i, j]:.4f},{table['ax_dec'][i, j]:.4f}\n")
                rows += 1
    return rows
