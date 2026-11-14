"""Vehicle files of the Performance department's LionsLTS simulator -> LapSim parameter changes.

LionsLTS (Python, TUM quasi-steady-state solvers) describes a car in a YAML file (`inputs/vehicles/*.yml`:
general / engine / gearbox / tires). `to_changes` translates one into the `name=value` changes of this
project's vehicle setup (launcher, dashboard, LapSim, simulator), relative to a baseline, and says what
could not be mapped one to one.

Differences between the two models that the notes report:
* aerodynamics: LionsLTS files give either c_w, c_z, a, aero_eff, aero_dristr_front, or the areas c_w_a,
  c_z_a_f, c_z_a_r. In LionsLTS the efficiency factor is only inside the downforce areas; the LapSim model
  scales drag and downforce by `aero_eff`, so the drag area is divided by it here to keep the same drag.
* tyres: LapSim has one tyre set for all four wheels; LionsLTS has front and rear.
* powertrain: LionsLTS `pow_e_motor` is the motor power limit (after efficiency, 72 kW = 80 kW x 0.9 in
  some files); LapSim P_max limits the motor output the same way. `torque_e_motor_max` is the total motor
  torque on the driven axle(s) for RWD; for AWD files the meaning of the number is not clear.
"""

from __future__ import annotations

import math

from .params import PerfParams

_TOL = 1e-9
_REL_TOL = {"circ_ref": 5e-3}      # 1.5651 m (LionsLTS) vs 1.57 m: not worth a change


def _f(d: dict, *path, default=None):
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return default
        d = d[p]
    return d


def _first(x):
    return x[0] if isinstance(x, (list, tuple)) and x else x


def to_changes(vehicle: dict, base: PerfParams | None = None) -> tuple[list[str], list[str]]:
    """(changes, notes) for the LionsLTS vehicle dict `vehicle` (parsed YAML), relative to `base`."""
    base = base or PerfParams()
    new: dict[str, float] = {}
    notes: list[str] = []
    gen, eng, gbx, tires = (vehicle.get(k) or {} for k in ("general", "engine", "gearbox", "tires"))

    for src, dst in (("lf", "lf"), ("lr", "lr"), ("h_cog", "h_cog"), ("sf", "sf"), ("sr", "sr"), ("m", "m"),
                     ("f_roll", "f_roll"), ("g", "g"), ("rho_air", "rho")):
        if src in gen:
            new[dst] = float(gen[src])

    # aerodynamics
    eff = float(gen.get("aero_eff", base.aero_eff))
    if "c_w_a" in gen:
        a = float(gen.get("a", base.A))
        cz_f, cz_r = float(gen.get("c_z_a_f", 0.0)), float(gen.get("c_z_a_r", 0.0))
        new["A"] = a
        new["c_w"] = float(gen["c_w_a"]) / (a * base.aero_eff)
        total = cz_f + cz_r
        new["c_z"] = total / (a * base.aero_eff)
        if total > 0:
            new["aero_bal_f"] = cz_f / total
        notes.append("aero areas c_w_a / c_z_a_f / c_z_a_r converted with the LapSim aero_eff "
                     f"{base.aero_eff:g} (drag area kept; LionsLTS applies the efficiency factor to downforce only)")
    else:
        for src, dst in (("c_w", "c_w"), ("c_z", "c_z"), ("a", "A"), ("aero_dristr_front", "aero_bal_f"), ("aero_eff", "aero_eff")):
            if src in gen:
                new[dst] = float(gen[src])
        if "aero_eff" in gen:
            notes.append(f"aero_eff {eff:g} scales drag as well as downforce in LapSim, only downforce in LionsLTS")
    if "drs_factor" in gen and float(gen["drs_factor"]) != 0.0:
        notes.append("drs_factor is not modelled")

    # powertrain
    if eng:
        if str(eng.get("powertrain_type", "electric")).lower() != "electric":
            notes.append(f"powertrain_type '{eng.get('powertrain_type')}' is not supported (electric only)")
        top = str(eng.get("topology", "RWD")).upper()
        if top == "FWD":
            notes.append("FWD is not supported (LapSim: RWD or AWD); RWD used")
        new["drive_awd"] = 1.0 if top == "AWD" else 0.0
        if top == "AWD":
            notes.append("AWD: LapSim puts an identical motor set on the front axle; check that torque_e_motor_max "
                         "means the same in the LionsLTS file")
        for src, dst in (("pow_e_motor", "P_max"), ("eta_e_motor", "eta_motor"), ("eta_e_motor_re", "eta_regen"),
                         ("torque_e_motor_max", "T_motor")):
            if src in eng:
                new[dst] = float(eng[src])
    if gbx:
        for src, dst in (("i_trans", "i_trans"), ("n_shift", "n_max"), ("e_i", "e_i"), ("eta_g", "eta_g")):
            if src in gbx:
                new[dst] = float(_first(gbx[src]))
        if isinstance(gbx.get("i_trans"), (list, tuple)) and len(gbx["i_trans"]) > 1:
            notes.append("several gears: only the first ratio is used (LapSim has one gear)")

    # tyres
    front, rear = tires.get("f") or {}, tires.get("r") or {}
    if front:
        for src, dst in (("circ_ref", "circ_ref"), ("fz_0", "Fz0"), ("mux", "mux"), ("muy", "muy"),
                         ("dmux_dfz", "dmux_dFz"), ("dmuy_dfz", "dmuy_dFz")):
            if src in front:
                new[dst] = float(front[src])
        if rear and any(abs(float(front.get(k, 0)) - float(rear.get(k, 0))) > _TOL for k in front):
            notes.append("front and rear tyres differ: only the front tyre data is used")
    if "tire_model_exp" in tires:
        new["tire_exp"] = float(tires["tire_model_exp"])

    changes = []
    for k, v in new.items():
        cur = getattr(base, k)
        if isinstance(cur, float) and not math.isnan(cur) and abs(cur - v) <= max(_TOL, _REL_TOL.get(k, 1e-9) * abs(cur)):
            continue
        changes.append(f"{k}={_num(v)}")
    return changes, notes


def _num(x: float) -> str:
    return f"{x:.10g}"
