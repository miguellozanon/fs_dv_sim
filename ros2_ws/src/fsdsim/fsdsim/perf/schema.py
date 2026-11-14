"""The LapSim parameters with label, unit and description (documentation v2.0, Table 3.1, plus LapSim extras).
Shared by the dashboard's Vehicle setup page and the Performance Lab (perf/), so both show the same list."""

from __future__ import annotations

from .lapsim import EVENT_ORDER

# (key, label, unit, description) - documentation v2.0, Table 3.1 (+ LapSim / simulator extras)
LAPSIM_SCHEMA = [
    ("Vehicle", "lapsim", [
        ("m", "Vehicle mass", "kg", "Mass including driver (LR26 goal)"),
        ("lf", "CG to front axle", "m", "Longitudinal distance CG - front axle"),
        ("lr", "CG to rear axle", "m", "Longitudinal distance CG - rear axle"),
        ("h_cog", "CG height", "m", "Height of the centre of gravity"),
        ("sf", "Front track width", "m", "Distance between the front wheel centres"),
        ("sr", "Rear track width", "m", "Distance between the rear wheel centres"),
        ("f_roll", "Rolling resistance coeff.", "-", "Rolling resistance (assumption)"),
    ]),
    ("Aerodynamics", "lapsim", [
        ("c_w", "Drag coefficient c_w", "-", "LR26 goal"),
        ("c_z", "Lift (downforce) coefficient c_z", "-", "LR26 goal"),
        ("A", "Frontal area", "m²", "Projected frontal area"),
        ("aero_eff", "Aero efficiency factor", "-", "Scales drag and downforce (assumption)"),
        ("aero_bal_f", "Front aero load distribution", "-", "Front share of downforce (CFD)"),
        ("rho", "Air density", "kg/m³", "25 °C, 1013 hPa"),
    ]),
    ("Powertrain", "lapsim", [
        ("P_max", "Maximum motor power", "W", "Rule limit 80 kW"),
        ("T_motor", "Maximum motor torque", "Nm", "Peak torque at the motor shaft (per driven axle)"),
        ("eta_motor", "Motor efficiency", "-", "Incl. battery (energy only)"),
        ("drive_awd", "Drivetrain (0 = RWD, 1 = AWD)", "-", "AWD adds a motor set on the front axle"),
        ("P_at_accumulator", "P_max at accumulator (0/1)", "-", "1: limit applies to accumulator power"),
    ]),
    ("Transmission", "lapsim", [
        ("i_trans", "Gear ratio", "-", "Wheel to motor"),
        ("n_max", "Maximum motor speed", "rpm", ""),
        ("e_i", "Rotational mass factor", "-", "m_eff = e_i · m"),
        ("eta_g", "Transmission efficiency", "-", "Gearbox and drivetrain"),
    ]),
    ("Tyres", "lapsim", [
        ("mux", "Longitudinal friction μx", "-", "At nominal load"),
        ("muy", "Lateral friction μy", "-", "At nominal load"),
        ("dmux_dFz", "Load sensitivity longitudinal", "1/N", ""),
        ("dmuy_dFz", "Load sensitivity lateral", "1/N", ""),
        ("Fz0", "Nominal tyre load", "N", ""),
        ("circ_ref", "Loaded tyre circumference", "m", "r = circ / 2π"),
        ("tire_exp", "Friction ellipse exponent", "-", ""),
    ]),
    ("Brakes, energy and endurance", "lapsim", [
        ("brake_bias", "Brake bias (front share)", "-", "Empty / NaN = ideal balance"),
        ("P_regen", "Recuperation power", "W", "0 = off (documentation)"),
        ("eta_regen", "Recuperation efficiency", "-", ""),
        ("P_max_endurance", "Endurance power cap", "W", "LapSim endurance strategy"),
        ("endurance_torque_scale", "Endurance torque scale", "-", "LapSim endurance strategy"),
    ]),
    ("Driver model friction utilisation (LapSim)", "event", [
        (f"{ev}.{f}", f"{ev.capitalize()} {'longitudinal' if f == 'util_x' else 'lateral'}", "-",
         f"Share of μ{'x' if f == 'util_x' else 'y'} LapSim's driver uses")
        for ev in EVENT_ORDER for f in ("util_x", "util_y")
    ]),
]
