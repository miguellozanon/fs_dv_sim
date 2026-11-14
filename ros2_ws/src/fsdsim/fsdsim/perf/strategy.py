"""Endurance strategy: the power the car can use for a given battery energy.

The documentation's endurance trace is power limited (a constant ~18 kW at the wheels) but does not say
where the limit comes from. The Performance department's LionsLTS solver derives it: it lowers the motor
power in steps until the energy of the lap fits the energy budget (`optimize_power`). This module does the
same on the LapSim port: `power_for_energy` finds, by bisection, the endurance power cap whose energy per lap
equals the budget, and `sweep_power` tabulates time, energy, points and the efficiency factor against the cap.
Both go through `run_event`, so they use the competition's tracks, driver utilisation and laps.

Energy is accumulator energy as in `run_event` (wheel power / (eta_g * eta_motor), recuperation as set in
the vehicle). The efficiency factor EF = T^2 * E [s^2 kWh] is the product the efficiency score of the
dynamic-events rules is built from (smaller is better, but it always falls with the cap, so it says nothing alone); this module does not turn it into points, because the
FS Rules 2026 D9.4 formula was not available (LionsLTS uses an older one: 100 * ((EFmax - EF)/(EFmax - EFmin))^2
with EFmin = 7 064 138 s^2 kWh and EFmax = 2 EFmin).
"""

from __future__ import annotations

import math

from .lapsim import Event, run_event, score
from .params import PerfParams

J_PER_KWH = 3.6e6
LIONSLTS_EF_MIN = 1172.211 ** 2 * 5.141        # s^2 kWh, FSA 2025 reference of LionsLTS (7 064 138)


def run_with_cap(veh: PerfParams, ev: Event, cap_w: float) -> dict:
    """The endurance event with the power cap `cap_w` [W] (at most the vehicle's P_max)."""
    r = run_event(veh.copy(P_max_endurance=float(cap_w)), ev)
    e_kwh = r["E_total"] / J_PER_KWH
    out = {"cap_kw": cap_w / 1e3, "t_lap": r["t_lap"], "T": r["T"], "v_max": r["v_max"],
           "E_lap_kwh": r["E_lap"] / J_PER_KWH, "E_total_kwh": e_kwh, "EF": r["T"] ** 2 * e_kwh}
    out["points"] = score(r["T"], ev.score)
    return out


def power_for_energy(veh: PerfParams, ev: Event, budget_kwh: float, lo_kw: float = 5.0, tol_w: float = 5.0) -> dict:
    """The largest endurance power cap whose total energy (all laps) fits `budget_kwh`.

    Returns the run at that cap plus `limited` (False: even the full motor power fits the budget) and
    `feasible` (False: not even `lo_kw` fits; the run at `lo_kw` is returned)."""
    hi = float(veh.P_max)
    top = run_with_cap(veh, ev, hi)
    if top["E_total_kwh"] <= budget_kwh:
        return {**top, "limited": False, "feasible": True, "budget_kwh": budget_kwh}
    lo = lo_kw * 1e3
    low = run_with_cap(veh, ev, lo)
    if low["E_total_kwh"] > budget_kwh:
        return {**low, "limited": True, "feasible": False, "budget_kwh": budget_kwh}
    best = low
    while hi - lo > tol_w:
        mid = 0.5 * (lo + hi)
        r = run_with_cap(veh, ev, mid)
        if r["E_total_kwh"] <= budget_kwh:
            lo, best = mid, r
        else:
            hi = mid
    return {**best, "limited": True, "feasible": True, "budget_kwh": budget_kwh}


def sweep_power(veh: PerfParams, ev: Event, caps_kw) -> list[dict]:
    """One run per power cap [kW]."""
    return [run_with_cap(veh, ev, float(c) * 1e3) for c in caps_kw]
