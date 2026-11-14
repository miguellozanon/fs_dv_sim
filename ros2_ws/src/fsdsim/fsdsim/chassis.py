"""Body motion on the suspension and wheel slip, on top of the planar vehicle model.

    roll, pitch, heave   second-order responses of the sprung mass:
                         roll  -> roll_gradient * ay / g          (left corner, ay > 0: right side down, roll > 0)
                         pitch -> -pitch_gradient * ax / g + aero (braking: nose down, pitch > 0)
                         heave -> -downforce / total ride rate    (car sinks with speed, heave < 0)
    slip[4]              longitudinal slip ratio of each wheel from its drive / brake force and tyre potential
                         (inverse of F = potential * sin(C atan(B slip))), so wheel-speed sensors over-read under
                         traction and under-read under braking, as on the car

ROS / REP-103 angles: roll about x (forward), pitch about y (left). The planar motion is not changed by this
module: the load transfer that matters for grip is already in the LapSim wheel loads.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

G = 9.81


@dataclass
class ChassisState:
    roll: float = 0.0
    pitch: float = 0.0
    heave: float = 0.0
    roll_rate: float = 0.0
    pitch_rate: float = 0.0
    heave_rate: float = 0.0
    slip: list = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])   # fl, fr, rl, rr

    @property
    def attitude(self) -> tuple[float, float, float]:
        return (self.roll, self.pitch, self.heave)


class Chassis:
    def __init__(self, cfg, wheelbase: float):
        self.cfg = cfg
        self.L = wheelbase
        self.state = ChassisState()

    def reset(self) -> None:
        self.state = ChassisState()

    # ------------------------------------------------------------------ #
    def targets(self, ax: float, ay: float, downforce: float, aero_bal_f: float) -> tuple[float, float, float]:
        c = self.cfg
        roll = math.radians(c.roll_gradient_deg_g) * ay / G
        kf, kr = 2.0 * c.wheel_rate_front, 2.0 * c.wheel_rate_rear        # per axle
        zf, zr = -downforce * aero_bal_f / kf, -downforce * (1.0 - aero_bal_f) / kr
        pitch = -math.radians(c.pitch_gradient_deg_g) * ax / G + (zr - zf) / self.L   # front lower: nose down (> 0)
        return roll, pitch, 0.5 * (zf + zr)

    def step(self, dt: float, ax: float, ay: float, downforce: float = 0.0, aero_bal_f: float = 0.5) -> None:
        c, s = self.cfg, self.state
        if not c.attitude:
            s.roll = s.pitch = s.heave = s.roll_rate = s.pitch_rate = s.heave_rate = 0.0
            return
        tr, tp, th = self.targets(ax, ay, downforce, aero_bal_f)
        z = c.damping_ratio
        n = max(1, int(math.ceil(dt / 0.002)))
        h = dt / n
        for _ in range(n):       # semi-implicit Euler: stable for these frequencies at 2 ms
            for name, tgt, f in (("roll", tr, c.roll_freq_hz), ("pitch", tp, c.pitch_freq_hz), ("heave", th, c.heave_freq_hz)):
                w = 2.0 * math.pi * f
                x, v = getattr(s, name), getattr(s, name + "_rate")
                v += h * (w * w * (tgt - x) - 2.0 * z * w * v)
                setattr(s, name + "_rate", v)
                setattr(s, name, x + h * v)

    # ------------------------------------------------------------------ #
    def slip_ratio(self, fx: float, potential: float) -> float:
        """Slip ratio that gives the longitudinal force fx on a wheel with this force potential."""
        c = self.cfg
        if not c.wheel_slip or potential <= 1.0:
            return 0.0
        u = max(-0.999, min(0.999, fx / potential))
        return math.tan(math.asin(u) / c.slip_C) / c.slip_B

    def set_slip(self, fx_wheels, potentials) -> None:
        self.state.slip = [self.slip_ratio(f, p) for f, p in zip(fx_wheels, potentials)]
