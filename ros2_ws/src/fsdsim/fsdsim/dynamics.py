"""Vehicle dynamics: dynamic single-track model blended with a kinematic model.

State (body frame at the CoG, map frame for pose):
    x, y, yaw      pose in map
    vx, vy, r      longitudinal / lateral velocity, yaw rate
    steer          actual road-wheel steering angle (after actuator dynamics)
    accel          actual longitudinal acceleration request (after lag)

Inputs:
    steer_cmd      road-wheel angle request [rad]
    accel_cmd      signed longitudinal acceleration request [m/s^2]

Model features: simplified Pacejka lateral tyres with friction ellipse,
longitudinal load transfer, aero drag and downforce, rolling resistance,
power limit, first-order actuator lag and steering rate limit, kinematic model
at low speed (the dynamic model is singular at vx -> 0), RK4 integration.

This is the same structure used by most FS driverless simulators (FSSIM, EUFS
sim, AMZ). Replace the parameters in vehicle.yaml with identified values from
your car to turn it into a digital twin.

With `model: performance` in vehicle.yaml the force limits come from the
Performance department's LapSim model (fsdsim.perf, LR26 parameters), so the
closed-loop simulator and the lap time simulation share one vehicle:
    * wheel loads: static + aero (aero_bal_f) + longitudinal / lateral load transfer
    * tyre potential per wheel Fz * (mu + dmu/dFz (Fz - Fz0)), summed per axle
    * lateral force = axle potential x Pacejka shape (B, C, E from `tire`, peak 1),
      reduced by the friction ellipse with exponent tire_exp
    * drive: min(T_motor, P/omega) through i_trans and eta_g per driven axle, F_cap = P eta_g / v,
      zero above n_max; RWD or AWD (drive_awd)
    * brakes: brake_bias or ideal balance; aero drag/downforce with aero_eff; m_eff = m e_i
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .chassis import Chassis
from .config import VehicleParams
from .perf import model as perf_model

G = 9.81


@dataclass
class VehicleState:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    r: float = 0.0
    steer: float = 0.0
    accel: float = 0.0

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.yaw, self.vx, self.vy, self.r, self.steer, self.accel])

    @staticmethod
    def from_array(a) -> "VehicleState":
        return VehicleState(*[float(v) for v in a])

    @property
    def pose(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.yaw)

    @property
    def speed(self) -> float:
        return math.hypot(self.vx, self.vy)


class VehicleModel:
    def __init__(self, params: VehicleParams, max_substep: float = 0.002):
        self.p = params
        self.max_substep = max_substep
        self.state = VehicleState()
        self.steer_cmd = 0.0
        self.accel_cmd = 0.0
        self.brake_override: float | None = None  # forced decel (hold / EBS), m/s^2 > 0
        # Last evaluated body-frame accelerations at the CoG (for the IMU)
        self.ax_body = 0.0
        self.ay_body = 0.0
        self.r_dot = 0.0
        # LapSim parameters at the tyre limit (driver utilisation is a driver property, not the car's)
        perf = params.perf() if params.model == "performance" else None
        self.perf = perf.copy(util_x=1.0, util_y=1.0, events={}) if perf is not None else None
        self._alpha_peak = self._find_alpha_peak()
        self.last_forces: dict = {}
        self.chassis = Chassis(params.chassis, params.wheelbase)
        self.double_track = self.perf is not None and str(params.chassis.handling).lower() == "double_track"
        self.eta_map = None             # optional perf.efficiency.EfficiencyMap (else the constant eta_motor)
        if self.perf is not None and params.efficiency_map:
            from .perf.efficiency import EfficiencyMap  # noqa: PLC0415

            self.eta_map = EfficiencyMap.load(params.efficiency_map)
        self.energy_j = 0.0             # accumulator energy used since reset [J] (as LapSim: wheel power / efficiencies)
        self.power_w = 0.0              # accumulator power now [W]

    # ------------------------------------------------------------------ #
    def reset(self, x: float, y: float, yaw: float) -> None:
        self.state = VehicleState(x=x, y=y, yaw=yaw)
        self.steer_cmd = 0.0
        self.accel_cmd = 0.0
        self.ax_body = self.ay_body = self.r_dot = 0.0
        self.chassis.reset()
        self.energy_j = self.power_w = 0.0

    def set_command(self, steer_cmd: float, accel_cmd: float) -> None:
        a = self.p.actuators
        self.steer_cmd = float(np.clip(steer_cmd, -a.max_steer, a.max_steer))
        self.accel_cmd = float(np.clip(accel_cmd, -a.max_decel, a.max_accel))

    # ------------------------------------------------------------------ #
    def _pacejka(self, alpha: float, fz: float) -> float:
        t = self.p.tire
        ba = t.B * alpha
        return t.mu * fz * math.sin(t.C * math.atan(ba - t.E * (ba - math.atan(ba))))

    def _derivatives(self, s: np.ndarray, steer_cmd: float, accel_cmd: float):
        p = self.p
        act = p.actuators
        _, _, yaw, vx, vy, r, delta, a = s
        m, L, lf, lr, h = p.mass, p.wheelbase, p.lf, p.lr, p.cog_height

        # Actuators
        ddelta = min(max((steer_cmd - delta) / act.steer_time_constant, -act.max_steer_rate), act.max_steer_rate)
        da = (accel_cmd - a) / act.accel_time_constant

        # Dynamic single-track model
        vx_s = max(vx, 0.5)
        alpha_f = delta - math.atan2(vy + lf * r, vx_s)
        alpha_r = -math.atan2(vy - lr * r, vx_s)
        cd, sd = math.cos(delta), math.sin(delta)
        wheels = None
        if self.double_track:
            (Fx_f, Fx_r, Fy_f, Fy_r, F_drag, F_roll, m_x, lim, FX, FY, MZ, wheels) = \
                self._perf_forces_wheels(vx, vy, r, a, delta)
            dvx_dyn = (FX - F_drag - F_roll) / m_x + vy * r
            dvy_dyn = FY / m - vx * r
            dr_dyn = MZ / p.yaw_inertia
        else:
            if self.perf is not None:
                Fx_f, Fx_r, Fy_f, Fy_r, F_drag, F_roll, m_x, lim = self._perf_forces(vx, a, alpha_f, alpha_r)
            else:
                Fx_f, Fx_r, Fy_shape, F_drag, F_roll, m_x, lim = self._st_forces(vx, a)
                Fy_f, Fy_r = Fy_shape(alpha_f, alpha_r)
            dvx_dyn = (Fx_r + Fx_f * cd - Fy_f * sd - F_drag - F_roll) / m_x + vy * r
            dvy_dyn = (Fy_r + Fx_f * sd + Fy_f * cd) / m - vx * r
            dr_dyn = (lf * (Fy_f * cd + Fx_f * sd) - lr * Fy_r) / p.yaw_inertia

        # Kinematic model (low speed)
        dvx_kin = (Fx_f + Fx_r - F_drag - F_roll) / m_x
        dvy_kin = (ddelta * vx + delta * dvx_kin) * lr / L
        dr_kin = (ddelta * vx + delta * dvx_kin) / L

        v_lo, v_hi = p.low_speed_blend
        lam = min(max((math.hypot(vx, vy) - v_lo) / max(v_hi - v_lo, 1e-3), 0.0), 1.0)
        dvx = lam * dvx_dyn + (1.0 - lam) * dvx_kin
        dvy = lam * dvy_dyn + (1.0 - lam) * dvy_kin
        dr = lam * dr_dyn + (1.0 - lam) * dr_kin

        if vx <= 0.0 and dvx < 0.0:  # no reversing
            dvx = 0.0

        cy, sy = math.cos(yaw), math.sin(yaw)
        ds = np.array([vx * cy - vy * sy, vx * sy + vy * cy, r, dvx, dvy, dr, ddelta, da])
        aux = (dvx - vy * r, dvy + vx * r, dr, lam, (Fx_f, Fx_r, Fy_f, Fy_r, lim, wheels))
        return ds, aux

    # ------------------------------------------------------------------ #
    def _st_forces(self, vx: float, a: float):
        """Generic single-track model forces (model: single_track)."""
        p = self.p
        act = p.actuators
        m, L, lf, lr, h = p.mass, p.wheelbase, p.lf, p.lr, p.cog_height
        F_req = m * a
        if F_req > 0.0 and vx > 0.5:
            F_req = min(F_req, p.drivetrain.max_power / vx)
        if F_req > 0.0 and vx >= act.max_speed:
            F_req = 0.0
        aero = p.aero
        F_drag = 0.5 * aero.rho * aero.cd_a * vx * abs(vx)
        F_down = 0.5 * aero.rho * aero.cl_a * vx * vx
        F_roll = p.rolling_resistance * (m * G + F_down) * math.tanh(vx / 0.2)
        Fz_f = max((m * G * lr - m * a * h) / L + F_down * aero.balance_front, 50.0)
        Fz_r = max((m * G * lf + m * a * h) / L + F_down * (1.0 - aero.balance_front), 50.0)
        mu = p.tire.mu
        split = p.drivetrain.drive_split_front if F_req >= 0.0 else p.drivetrain.brake_bias_front
        Fx_f = float(np.clip(F_req * split, -mu * Fz_f, mu * Fz_f))
        Fx_r = float(np.clip(F_req * (1.0 - split), -mu * Fz_r, mu * Fz_r))
        ellipse_f = math.sqrt(max(0.0, 1.0 - (Fx_f / (mu * Fz_f)) ** 2))
        ellipse_r = math.sqrt(max(0.0, 1.0 - (Fx_r / (mu * Fz_r)) ** 2))

        def fy(alpha_f, alpha_r):
            return self._pacejka(alpha_f, Fz_f) * ellipse_f, self._pacejka(alpha_r, Fz_r) * ellipse_r

        return Fx_f, Fx_r, fy, F_drag, F_roll, m, ""

    def _find_alpha_peak(self) -> float:
        a = np.linspace(0.0, 0.6, 1201)
        t = self.p.tire
        ba = t.B * a
        y = np.sin(t.C * np.arctan(ba - t.E * (ba - np.arctan(ba))))
        return float(a[int(np.argmax(y))])

    def _shape(self, alpha: float) -> float:
        """Pacejka lateral shape with peak 1 (B, C, E from vehicle.yaml `tire`)."""
        t = self.p.tire
        ba = t.B * alpha
        return math.sin(t.C * math.atan(ba - t.E * (ba - math.atan(ba))))

    def _perf_forces(self, vx: float, a: float, alpha_f: float, alpha_r: float):
        """Forces from the LapSim model (model: performance).

        As in LapSim, cornering has priority: the lateral force follows the slip angle up to
        the axle potential, and drive / brake forces only use what the friction ellipse leaves
        (Fx_rem = pxAx (1 - u^e)^(1/e)). On the car this is what traction control and ABS do."""
        pp = self.perf
        v = max(vx, 0.0)
        m_x = pp.m * pp.e_i
        # quasi-static loads from the accelerations of the previous evaluation (LapSim: steady state)
        Fz, px_f, px_r, py_f, py_r = perf_model.axle_potentials(pp, v, self.ax_body, self.ay_body)
        pxAx, pyAx = (px_f, px_r), (py_f, py_r)
        F_drag = perf_model.drag(pp, v)
        F_roll = pp.f_roll * (pp.m * pp.g + perf_model.downforce(pp, v)) * math.tanh(vx / 0.2)
        # lateral: potential x Pacejka shape (|shape| <= 1)
        sh_f, sh_r = self._shape(alpha_f), self._shape(alpha_r)
        Fy_f, Fy_r = pyAx[0] * sh_f, pyAx[1] * sh_r
        e = pp.tire_exp
        # past the peak slip angle the tyre is fully used: no drive / brake force is left (what
        # traction control and ABS keep; otherwise the falling Pacejka curve would free grip and spin the car)
        u_f = 1.0 if abs(alpha_f) >= self._alpha_peak else min(abs(sh_f), 1.0)
        u_r = 1.0 if abs(alpha_r) >= self._alpha_peak else min(abs(sh_r), 1.0)
        rem_f = pxAx[0] * max(1.0 - u_f ** e, 0.0) ** (1.0 / e)
        rem_r = pxAx[1] * max(1.0 - u_r ** e, 0.0) ** (1.0 / e)
        F_req = m_x * a
        lim = ""
        if F_req >= 0.0:
            F_set, F_cap, _v_max = perf_model.powertrain(pp, v)
            if pp.drive_awd > 0.5:
                want = 0.5 * F_req
                Fx_f, Fx_r = min(want, F_set, rem_f), min(want, F_set, rem_r)
            else:
                Fx_f, Fx_r = 0.0, min(F_req, F_set, rem_r)
            tot = Fx_f + Fx_r
            if tot > F_cap > 0.0:
                Fx_f, Fx_r = Fx_f * F_cap / tot, Fx_r * F_cap / tot
                lim = "power"
            elif F_req > 0.0 and Fx_r >= rem_r - 1e-6:
                lim = "traction"
            elif F_req > 0.0 and Fx_r >= F_set - 1e-6:
                lim = "motor"
        else:
            B = -F_req
            share_f = rem_f / max(rem_f + rem_r, 1e-6) if math.isnan(pp.brake_bias) else pp.brake_bias
            Fx_f, Fx_r = -min(B * share_f, rem_f), -min(B * (1.0 - share_f), rem_r)
            if B * share_f > rem_f or B * (1.0 - share_f) > rem_r:
                lim = "brake grip"
        self._last_perf = (Fz, pxAx, pyAx)
        return Fx_f, Fx_r, Fy_f, Fy_r, F_drag, F_roll, m_x, lim

    def _perf_forces_wheels(self, vx: float, vy: float, r: float, a: float, delta: float):
        """Double-track version of _perf_forces: slip angle, load, potential and force per wheel.

        Same tyre law (LapSim potential x Pacejka shape, cornering first, friction ellipse for drive / brake), but
        the inner and outer wheel differ: the unloaded inner wheel saturates first, and unequal left / right
        longitudinal forces turn the car. Drive: each driven wheel gets half the axle request, limited by half the
        axle's motor force and by its own remaining grip (per-wheel traction control; the LR26 has one inverter
        per side). Brakes: equal pressure left / right, limited per wheel (ABS)."""
        pp, p = self.perf, self.p
        v = max(vx, 0.0)
        m_x = pp.m * pp.e_i
        Fz, _, _, _, _ = perf_model.axle_potentials(pp, v, self.ax_body, self.ay_body)
        F_drag = perf_model.drag(pp, v)
        F_roll = pp.f_roll * (pp.m * pp.g + perf_model.downforce(pp, v)) * math.tanh(vx / 0.2)
        tf, tr = p.track_front / 2, p.track_rear / 2
        pos = ((p.lf, tf), (p.lf, -tf), (-p.lr, tr), (-p.lr, -tr))          # fl, fr, rl, rr (x fwd, y left)
        e = pp.tire_exp
        Fy_w, rem, px_w = [0.0] * 4, [0.0] * 4, [0.0] * 4
        for i, (xw, yw) in enumerate(pos):
            f = Fz[i]
            px = max(f * (pp.mux + pp.dmux_dFz * (f - pp.Fz0)), 0.0)
            py = max(f * (pp.muy + pp.dmuy_dFz * (f - pp.Fz0)), 0.0)
            alpha = (delta if i < 2 else 0.0) - math.atan2(vy + r * xw, max(vx - r * yw, 0.5))
            sh = self._shape(alpha)
            u = 1.0 if abs(alpha) >= self._alpha_peak else min(abs(sh), 1.0)
            Fy_w[i], px_w[i] = py * sh, px
            rem[i] = px * max(1.0 - u ** e, 0.0) ** (1.0 / e)
        F_req = m_x * a
        Fx_w = [0.0] * 4
        lim = ""
        if F_req >= 0.0:
            F_set, F_cap, _v_max = perf_model.powertrain(pp, v)
            driven = (0, 1, 2, 3) if pp.drive_awd > 0.5 else (2, 3)
            want = F_req / len(driven)
            for i in driven:
                Fx_w[i] = min(want, 0.5 * F_set, rem[i])
            tot = sum(Fx_w)
            if tot > F_cap > 0.0:
                Fx_w = [f * F_cap / tot for f in Fx_w]
                lim = "power"
            elif F_req > 0.0 and any(Fx_w[i] >= rem[i] - 1e-6 for i in (2, 3)):
                lim = "traction"
            elif F_req > 0.0 and Fx_w[2] >= 0.5 * F_set - 1e-6:
                lim = "motor"
        else:
            B = -F_req
            rem_f, rem_r = rem[0] + rem[1], rem[2] + rem[3]
            share_f = rem_f / max(rem_f + rem_r, 1e-6) if math.isnan(pp.brake_bias) else pp.brake_bias
            for i in range(4):
                want = 0.5 * B * (share_f if i < 2 else 1.0 - share_f)
                Fx_w[i] = -min(want, rem[i])
                if want > rem[i]:
                    lim = "brake grip"
        cd, sd = math.cos(delta), math.sin(delta)
        FX = FY = MZ = 0.0
        for i, (xw, yw) in enumerate(pos):
            if i < 2:
                fxb, fyb = Fx_w[i] * cd - Fy_w[i] * sd, Fx_w[i] * sd + Fy_w[i] * cd
            else:
                fxb, fyb = Fx_w[i], Fy_w[i]
            FX += fxb
            FY += fyb
            MZ += xw * fyb - yw * fxb
        wheels = (tuple(Fx_w), tuple(px_w))
        return (Fx_w[0] + Fx_w[1], Fx_w[2] + Fx_w[3], Fy_w[0] + Fy_w[1], Fy_w[2] + Fy_w[3], F_drag, F_roll, m_x, lim,
                FX, FY, MZ, wheels)

    # ------------------------------------------------------------------ #
    def step(self, dt: float) -> None:
        """Advance the model by dt seconds (internally sub-stepped, RK4)."""
        n = max(1, int(math.ceil(dt / self.max_substep)))
        h = dt / n
        if self.brake_override is not None:
            steer_cmd = self.state.steer  # hold wheel position
            accel_cmd = -abs(self.brake_override)
        else:
            steer_cmd, accel_cmd = self.steer_cmd, self.accel_cmd
        s = self.state.to_array()
        lr, L = self.p.lr, self.p.wheelbase
        for _ in range(n):
            k1, aux = self._derivatives(s, steer_cmd, accel_cmd)
            k2, _ = self._derivatives(s + 0.5 * h * k1, steer_cmd, accel_cmd)
            k3, _ = self._derivatives(s + 0.5 * h * k2, steer_cmd, accel_cmd)
            k4, _ = self._derivatives(s + h * k3, steer_cmd, accel_cmd)
            s = s + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            if s[3] < 0.0:
                s[3] = 0.0
            # Keep low-speed lateral states consistent with the kinematic model
            lam = aux[3]
            if lam < 1.0:
                vy_k = s[3] * math.tan(s[6]) * lr / L
                r_k = s[3] * math.tan(s[6]) / L
                s[4] = lam * s[4] + (1 - lam) * vy_k
                s[5] = lam * s[5] + (1 - lam) * r_k
            s[2] = (s[2] + math.pi) % (2 * math.pi) - math.pi
        self.state = VehicleState.from_array(s)
        _, (self.ax_body, self.ay_body, self.r_dot, _, forces) = self._derivatives(s, steer_cmd, accel_cmd)
        Fx_f, Fx_r, Fy_f, Fy_r, lim, wheels = forces
        self.last_forces = {"Fx_f": Fx_f, "Fx_r": Fx_r, "Fy_f": Fy_f, "Fy_r": Fy_r, "limit": lim}
        self._after_step(dt, Fx_f, Fx_r, wheels)

    def _after_step(self, dt: float, Fx_f: float, Fx_r: float, wheels) -> None:
        """Body attitude, wheel slip and the energy meter (none of them feeds back into the planar motion)."""
        pp, v = self.perf, max(self.state.vx, 0.0)
        if pp is not None:
            df, bal = perf_model.downforce(pp, v), pp.aero_bal_f
            if wheels is None:           # axle forces: half per wheel, potentials from the wheel loads
                Fz = getattr(self, "_last_perf", ((0.0,) * 4, None, None))[0]
                px = tuple(max(f * (pp.mux + pp.dmux_dFz * (f - pp.Fz0)), 0.0) for f in Fz)
                wheels = ((0.5 * Fx_f, 0.5 * Fx_f, 0.5 * Fx_r, 0.5 * Fx_r), px)
            self.chassis.set_slip(*wheels)
            # accumulator power as in LapSim (lapsim.py: P_wheel / (eta_g * eta_motor), recuperation limited)
            P_wheel = (Fx_f + Fx_r) * v
            if P_wheel >= 0.0:
                eta_m = pp.eta_motor
                if self.eta_map is not None:
                    n_drv = 2 if pp.drive_awd > 0.5 else 1
                    T_mot = (Fx_f + Fx_r) * pp.r / (pp.i_trans * pp.eta_g) / n_drv        # per motor set
                    eta_m = self.eta_map.eta(T_mot, v / pp.r * pp.i_trans * 60.0 / (2 * math.pi))
                self.power_w = P_wheel / (pp.eta_g * eta_m)
            else:
                self.power_w = -min(-P_wheel, pp.P_regen) * pp.eta_regen
            self.energy_j += self.power_w * dt
        else:
            aero = self.p.aero
            df, bal = 0.5 * aero.rho * aero.cl_a * v * v, aero.balance_front
        self.chassis.step(dt, self.ax_body, self.ay_body, df, bal)

    # ------------------------------------------------------------------ #
    def wheel_speeds(self) -> tuple[float, float, float, float]:
        """Wheel angular speeds [rad/s] (FL, FR, RL, RR), with longitudinal slip (chassis.wheel_slip)."""
        s, p = self.state, self.p
        tf, tr = p.track_front / 2, p.track_rear / 2
        cd, sd = math.cos(s.steer), math.sin(s.steer)
        out = []
        for side, track_half in ((+1, tf), (-1, tf)):
            vxw = s.vx - s.r * track_half * side
            vyw = s.vy + s.r * p.lf
            out.append((vxw * cd + vyw * sd) / p.wheel_radius)
        for side in (+1, -1):
            out.append((s.vx - s.r * tr * side) / p.wheel_radius)
        return tuple(w * (1.0 + k) for w, k in zip(out, self.chassis.state.slip))

    def footprint_corners(self) -> np.ndarray:
        """Car outline corners in the map frame (4, 2)."""
        p, s = self.p, self.state
        f, r, w = p.front_bumper, -p.rear_bumper, p.footprint.width / 2
        local = np.array([[f, w], [f, -w], [r, -w], [r, w]])
        c, sn = math.cos(s.yaw), math.sin(s.yaw)
        return np.stack([s.x + c * local[:, 0] - sn * local[:, 1], s.y + sn * local[:, 0] + c * local[:, 1]], axis=1)
