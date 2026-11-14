"""LR26 vehicle model: line-by-line port of the LapSim core (MATLAB, Performance dept.).

    lts_resistance  -> resistance(veh, v)
    lts_powertrain  -> powertrain(veh, v)
    lts_wheel_loads -> wheel_loads(veh, v, ax, ay)
    lts_axle_state  -> axle_state(veh, v, ax, ay)
    lts_ax_limits   -> ax_limits(veh, v, k, ax_acc0, ax_dec0, mode)

Equations, sign conventions and the 4-step load-transfer iteration are kept exactly as
in the MATLAB functions (extracted from LR26_LapSim's code generation report), so both
programs give the same numbers for the same parameters. Scalars in, floats out.
"""

from __future__ import annotations

import math

import numpy as np

from .params import PerfParams


def downforce(veh: PerfParams, v: float) -> float:
    return 0.5 * veh.rho * veh.c_z * veh.A * veh.aero_eff * v * v


def drag(veh: PerfParams, v: float) -> float:
    return 0.5 * veh.rho * veh.c_w * veh.A * veh.aero_eff * v * v


def resistance(veh: PerfParams, v: float) -> float:
    """Aerodynamic drag + rolling resistance [N]."""
    return drag(veh, v) + veh.f_roll * (veh.m * veh.g + downforce(veh, v))


def powertrain(veh: PerfParams, v: float) -> tuple[float, float, float]:
    """(F_set, F_cap, v_max): force of ONE driven axle, power-limited total force, top speed."""
    r = veh.circ_ref / (2 * math.pi)
    v_max = veh.n_max * 2 * math.pi / 60 / veh.i_trans * r
    P = veh.P_max * veh.eta_motor if veh.P_at_accumulator > 0.5 else veh.P_max
    vv = max(v, 0.1)
    w_mot = vv / r * veh.i_trans
    T = min(veh.T_motor, P / w_mot)
    F_set = T * veh.i_trans * veh.eta_g / r
    if v > v_max * (1 + 1e-9):
        F_set = 0.0
    F_cap = P * veh.eta_g / vv
    return F_set, F_cap, v_max


def wheel_loads(veh: PerfParams, v: float, ax: float, ay: float) -> np.ndarray:
    """Quasi-static tyre loads [fl, fr, rl, rr] in N (ay > 0: left-hand corner)."""
    l = veh.lf + veh.lr  # noqa: E741
    m = veh.m
    DF = downforce(veh, v)
    Ff = m * veh.g * veh.lr / l + DF * veh.aero_bal_f - m * ax * veh.h_cog / l
    Fr = m * veh.g * veh.lf / l + DF * (1 - veh.aero_bal_f) + m * ax * veh.h_cog / l
    dyf = m * ay * veh.h_cog / veh.sf * veh.lr / l
    dyr = m * ay * veh.h_cog / veh.sr * veh.lf / l
    return np.maximum(np.array([Ff / 2 - dyf, Ff / 2 + dyf, Fr / 2 - dyr, Fr / 2 + dyr]), 0.0)


def axle_state(veh: PerfParams, v: float, ax: float, ay: float):
    """(Fx_rem[2], u[2], Fz[4], pxAx[2], pyAx[2], Fy[2]) for [front, rear]."""
    l = veh.lf + veh.lr  # noqa: E741
    Fz = wheel_loads(veh, v, ax, ay)
    px = np.maximum(Fz * (veh.mux + veh.dmux_dFz * (Fz - veh.Fz0)) * veh.util_x, 0.0)
    py = np.maximum(Fz * (veh.muy + veh.dmuy_dFz * (Fz - veh.Fz0)) * veh.util_y, 0.0)
    pxAx = np.array([px[0] + px[1], px[2] + px[3]])
    pyAx = np.array([py[0] + py[1], py[2] + py[3]])
    Fyt = abs(veh.m * ay)
    Fy = np.array([Fyt * veh.lr / l, Fyt * veh.lf / l])
    u = Fy / np.maximum(pyAx, 1e-9)
    e = veh.tire_exp
    Fx_rem = pxAx * np.maximum(1 - np.minimum(u, 1) ** e, 0) ** (1 / e)
    return Fx_rem, u, Fz, pxAx, pyAx, Fy


def axle_potentials(veh: PerfParams, v: float, ax: float, ay: float):
    """Scalar fast path of axle_state for the dynamic model (same equations, plain floats):
    ((Fz_fl, Fz_fr, Fz_rl, Fz_rr), pxAx_f, pxAx_r, pyAx_f, pyAx_r)."""
    l = veh.lf + veh.lr  # noqa: E741
    m = veh.m
    DF = 0.5 * veh.rho * veh.c_z * veh.A * veh.aero_eff * v * v
    Ff = m * veh.g * veh.lr / l + DF * veh.aero_bal_f - m * ax * veh.h_cog / l
    Fr = m * veh.g * veh.lf / l + DF * (1 - veh.aero_bal_f) + m * ax * veh.h_cog / l
    dyf = m * ay * veh.h_cog / veh.sf * veh.lr / l
    dyr = m * ay * veh.h_cog / veh.sr * veh.lf / l
    Fz = (max(Ff / 2 - dyf, 0.0), max(Ff / 2 + dyf, 0.0), max(Fr / 2 - dyr, 0.0), max(Fr / 2 + dyr, 0.0))
    mux, muy, dx, dy, F0, ux, uy = veh.mux, veh.muy, veh.dmux_dFz, veh.dmuy_dFz, veh.Fz0, veh.util_x, veh.util_y
    px = [max(f * (mux + dx * (f - F0)) * ux, 0.0) for f in Fz]
    py = [max(f * (muy + dy * (f - F0)) * uy, 0.0) for f in Fz]
    return Fz, px[0] + px[1], px[2] + px[3], py[0] + py[1], py[2] + py[3]


def ax_limits(veh: PerfParams, v: float, k: float, ax_acc0: float = 0.0, ax_dec0: float = 0.0, mode: int = 0):
    """(ax_acc, ax_dec, F_drive, F_brake) at speed v on curvature k (mode 1: drive only, 2: brake only)."""
    ay = v * v * k
    meff = veh.m * veh.e_i
    Fres = resistance(veh, v)
    F_set, F_cap, _ = powertrain(veh, v)

    ax_acc, F_drive = ax_acc0, 0.0
    if mode != 2:
        for _ in range(4):
            Fx_rem = axle_state(veh, v, ax_acc, ay)[0]
            if veh.drive_awd > 0.5:
                F_drive = min(F_set, Fx_rem[0]) + min(F_set, Fx_rem[1])
            else:
                F_drive = min(F_set, Fx_rem[1])
            F_drive = min(F_drive, F_cap)
            ax_acc = (F_drive - Fres) / meff

    ax_dec, F_brake = ax_dec0, 0.0
    if mode != 1:
        for _ in range(4):
            Fx_rem = axle_state(veh, v, ax_dec, ay)[0]
            if math.isnan(veh.brake_bias):
                F_brake = Fx_rem[0] + Fx_rem[1]
            else:
                b = veh.brake_bias
                F_brake = min(Fx_rem[0] / max(b, 1e-6), Fx_rem[1] / max(1 - b, 1e-6))
            ax_dec = -(F_brake + Fres) / meff
    return ax_acc, ax_dec, F_drive, F_brake
