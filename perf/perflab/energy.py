"""Energy and endurance: where the energy of an endurance lap goes, the power cap for a battery budget, and
time / energy / points against the power cap (fsdsim.perf.strategy, the LionsLTS optimize_power idea).

Energy is accumulator energy as in LapSim (wheel power / (eta_g * eta_motor), recuperation limited to P_regen
times eta_regen); with an efficiency map selected in the lab, eta_motor comes from the map.
"""

from __future__ import annotations

import numpy as np

from fsdsim.perf.lapsim import _resistance
from fsdsim.perf.strategy import J_PER_KWH, LIONSLTS_EF_MIN, power_for_energy, sweep_power

from .lab import Lab, clean, decimate


def _integral(p, t) -> float:
    return float(np.sum(0.5 * (p[:-1] + p[1:]) * np.diff(t))) if len(t) > 1 else 0.0


def endurance_event(lab: Lab, competition=None):
    ls = lab.lapsim(competition)
    for name, ev in ls.comp.events.items():
        if ev.type.lower() == "endurance":
            return name
    raise ValueError("this competition has no endurance event")


def breakdown(e: dict) -> dict:
    """Energy of one lap of an event result [kWh]: battery out / back, wheel work, drivetrain losses,
    drag, rolling resistance, friction brakes."""
    veh, v, t, Pw, Pb = e["veh"], e["v"], e["t"], e["P_wheel"], e["P_batt"]
    drag_f = 0.5 * veh.rho * veh.c_w * veh.A * veh.aero_eff * v * v
    roll_f = _resistance(veh, v) - drag_f
    pos, neg = np.maximum(Pw, 0.0), np.maximum(-Pw, 0.0)
    regen_wheel = np.minimum(neg, veh.P_regen)
    out = {"battery_out": _integral(np.maximum(Pb, 0.0), t), "battery_back": _integral(np.maximum(-Pb, 0.0), t),
           "wheel_drive": _integral(pos, t), "drag": _integral(drag_f * v, t), "rolling": _integral(roll_f * v, t),
           "friction_brakes": _integral(neg - regen_wheel, t), "regen_at_wheel": _integral(regen_wheel, t)}
    out["drivetrain_losses"] = out["battery_out"] - out["wheel_drive"]
    out["net"] = out["battery_out"] - out["battery_back"]
    return {k: v / J_PER_KWH for k, v in out.items()}


def analyse(lab: Lab, changes=(), budget_kwh: float | None = None, caps_kw=None, competition=None, job=None) -> dict:
    changes = lab.validate(changes, competition)
    ls = lab.lapsim(competition)
    name = endurance_event(lab, competition)
    veh, comp = ls.vehicle(changes), ls.competition(changes)
    ev = comp.events[name]
    caps = list(caps_kw) if caps_kw else [float(c) for c in np.arange(8.0, min(veh.P_max / 1e3, 80.0) + 0.1, 2.0)]
    if len(caps) > 60:
        raise ValueError("at most 60 power caps")
    with lab.lock:
        lab._use_eta_map()
        if job:
            job.step(0, 3, "baseline")
        r = lab.raw(changes, competition)
        e = r["events"][name]
        out = {"event": name, "laps": ev.laps, "P_max_kw": veh.P_max / 1e3, "cap_kw": e["veh"].P_max / 1e3,
               "t_lap": e["t_lap"], "T": e["T"], "points": e["points"], "E_lap_kwh": e["E_lap"] / J_PER_KWH,
               "E_total_kwh": e["E_total"] / J_PER_KWH, "EF": e["T"] ** 2 * e["E_total"] / J_PER_KWH,
               "EF_lionslts_min": LIONSLTS_EF_MIN, "eta_map": lab.eta_map or f"constant eta_motor = {veh.eta_motor:g}",
               "breakdown": breakdown(e), "per_event": {n: x["E_total"] / J_PER_KWH for n, x in r["events"].items()}}
        sl = decimate(len(e["s"]), 1200)
        out["trace"] = {"s": e["s"][sl], "v": e["v"][sl] * 3.6, "P_batt": e["P_batt"][sl] / 1e3,
                        "E": e["E"][sl] / J_PER_KWH}
        if job:
            job.step(1, 3, f"power sweep ({len(caps)} caps)")
        out["sweep"] = sweep_power(veh, ev, caps)
        if budget_kwh:
            if job:
                job.step(2, 3, f"power for {budget_kwh:g} kWh")
            out["budget"] = power_for_energy(veh, ev, float(budget_kwh))
    if job:
        job.step(3, 3, "done")
    return clean(out)


def ggv_map(lab: Lab, changes=(), event: str = "", competition=None, speeds=None, n_ay: int = 13) -> dict:
    """ggv of the car (at the tyre limit, or with the event's driver utilisation and endurance limits)."""
    from fsdsim.perf.ggv import ggv  # noqa: PLC0415
    from fsdsim.perf.lapsim import event_vehicle  # noqa: PLC0415

    changes = lab.validate(changes, competition)
    ls = lab.lapsim(competition)
    veh = ls.vehicle(changes)
    if event:
        veh = event_vehicle(veh, ls.competition(changes).events[event])
    else:
        veh = veh.copy(util_x=1.0, util_y=1.0)
    with lab.lock:
        t = ggv(veh, speeds if speeds is not None else np.arange(5.0, 32.6, 2.5), n_ay)
    g = 9.81
    return clean({"event": event or "tyre limit", "v_kmh": t["v"] * 3.6, "ay_g": t["ay"] / g,
                  "ax_acc_g": t["ax_acc"] / g, "ax_dec_g": t["ax_dec"] / g})
