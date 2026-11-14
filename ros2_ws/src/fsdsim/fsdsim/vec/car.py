"""Batched LR26 car: fsdsim's dynamic single-track model with LapSim forces, N cars at once.

Line-by-line the equations of fsdsim.dynamics.VehicleModel in `performance` mode (and
fsdsim.perf.model for loads, potentials, powertrain and aero), written with array operations so
that thousands of cars - each with its own specification - step together on a GPU:

    * quasi-static wheel loads from the previous accelerations (static + aero + load transfer)
    * load-sensitive tyre potentials per axle, Pacejka lateral shape, friction ellipse (tire_exp),
      cornering priority (traction control / ABS: drive and brake use what the ellipse leaves)
    * powertrain min(T_motor, P / omega) x i_trans x eta_g, power cap, n_max; RWD or AWD
    * brakes with a fixed bias or ideal balance; drag, downforce, rolling resistance; m_eff = m e_i
    * actuator lags and steering rate limit, kinematic blend at low speed, RK4 at 500 Hz

The same physics step as fsdsim.sim (500 Hz, one RK4 step per physics step, loads from the
accelerations at the end of the previous one), verified against VehicleModel in the tests.
"""

from __future__ import annotations

import math

from .backend import Backend
from .spec import SpecBatch


class BatchCar:
    STATE = ("x", "y", "yaw", "vx", "vy", "r", "delta", "a")

    def __init__(self, B: Backend, spec: SpecBatch, physics_hz: float = 500.0):
        self.B = B
        self.spec = spec
        self.n = spec.n
        self.h = 1.0 / physics_hz
        for k in self.STATE:
            setattr(self, k, B.zeros(self.n))
        self.ax_b = B.zeros(self.n)        # body-frame accelerations at the CoG (IMU, load transfer)
        self.ay_b = B.zeros(self.n)
        self.limit = None                  # last force-limit flags: (power, traction, brake grip, motor torque)

    # ------------------------------------------------------------------ #
    def reset(self, idx, x, y, yaw, vx=None, a0=None) -> None:
        """Place cars idx at (x, y, yaw), moving straight at vx (default standstill). a0: actuator state
        (a standing car in AS Ready holds the brakes: fsdsim releases them at Go, a0 = -hold_decel)."""
        self.x[idx], self.y[idx], self.yaw[idx] = x, y, yaw
        self.vx[idx] = 0.0 if vx is None else vx
        for k in ("vy", "r", "delta"):
            getattr(self, k)[idx] = 0.0
        self.a[idx] = 0.0 if a0 is None else a0
        self.ax_b[idx] = 0.0
        self.ay_b[idx] = 0.0

    def state(self):
        return [getattr(self, k) for k in self.STATE]

    # ------------------------------------------------------------------ #
    def _forces(self, vx, a, alpha_f, alpha_r):
        """LapSim forces (dynamics.VehicleModel._perf_forces) for all cars."""
        B, p = self.B, self.spec
        v = B.maximum(vx, 0.0)
        m_x = p.m * p.e_i
        l = p.lf + p.lr  # noqa: E741
        # wheel loads (perf.model.axle_potentials) from the previous accelerations
        DF = 0.5 * p.rho * p.c_z * p.A * p.aero_eff * v * v
        Ff = p.m * p.g * p.lr / l + DF * p.aero_bal_f - p.m * self.ax_b * p.h_cog / l
        Fr = p.m * p.g * p.lf / l + DF * (1 - p.aero_bal_f) + p.m * self.ax_b * p.h_cog / l
        dyf = p.m * self.ay_b * p.h_cog / p.sf * p.lr / l
        dyr = p.m * self.ay_b * p.h_cog / p.sr * p.lf / l
        mux, muy = p.mux * p.grip_scale, p.muy * p.grip_scale

        def pots(fz):
            fz = B.maximum(fz, 0.0)
            return (B.maximum(fz * (mux + p.dmux_dFz * (fz - p.Fz0)), 0.0),
                    B.maximum(fz * (muy + p.dmuy_dFz * (fz - p.Fz0)), 0.0))

        px1, py1 = pots(Ff / 2 - dyf)
        px2, py2 = pots(Ff / 2 + dyf)
        px3, py3 = pots(Fr / 2 - dyr)
        px4, py4 = pots(Fr / 2 + dyr)
        px_f, px_r, py_f, py_r = px1 + px2, px3 + px4, py1 + py2, py3 + py4
        F_drag = 0.5 * p.rho * p.c_w * p.A * p.aero_eff * v * v
        F_roll = p.f_roll * (p.m * p.g + DF) * B.tanh(vx / 0.2)
        # lateral: potential x Pacejka shape
        sh_f, sh_r = self._shape(alpha_f), self._shape(alpha_r)
        Fy_f, Fy_r = py_f * sh_f, py_r * sh_r
        e = p.tire_exp
        u_f = B.where(B.abs(alpha_f) >= p.alpha_peak, 1.0, B.minimum(B.abs(sh_f), 1.0))
        u_r = B.where(B.abs(alpha_r) >= p.alpha_peak, 1.0, B.minimum(B.abs(sh_r), 1.0))
        rem_f = px_f * B.maximum(1.0 - u_f ** e, 0.0) ** (1.0 / e)
        rem_r = px_r * B.maximum(1.0 - u_r ** e, 0.0) ** (1.0 / e)
        F_req = m_x * a
        # drive (perf.model.powertrain)
        r = p.circ_ref / (2 * math.pi)
        v_max = p.n_max * 2 * math.pi / 60 / p.i_trans * r
        P = B.where(p.P_at_accumulator > 0.5, p.P_max * p.eta_motor, p.P_max)
        vv = B.maximum(v, 0.1)
        T = B.minimum(p.T_motor, P / (vv / r * p.i_trans))
        F_set = B.where(v > v_max * (1 + 1e-9), 0.0, T * p.i_trans * p.eta_g / r)
        F_cap = P * p.eta_g / vv
        awd = p.drive_awd > 0.5
        want = B.where(awd, 0.5 * F_req, F_req)
        dF_f = B.where(awd, B.minimum(B.minimum(want, F_set), rem_f), 0.0)
        dF_r = B.minimum(B.minimum(want, F_set), rem_r)
        tot = dF_f + dF_r
        capped = (tot > F_cap) & (F_cap > 0.0)
        scale = B.where(capped, F_cap / B.maximum(tot, 1e-9), 1.0)
        dF_f, dF_r = dF_f * scale, dF_r * scale
        # brake
        Bk = -F_req
        share_f = B.where(p.bias_ideal > 0.5, rem_f / B.maximum(rem_f + rem_r, 1e-6), p.brake_bias)
        bF_f = -B.minimum(Bk * share_f, rem_f)
        bF_r = -B.minimum(Bk * (1.0 - share_f), rem_r)
        drive = F_req >= 0.0
        Fx_f = B.where(drive, dF_f, bF_f)
        Fx_r = B.where(drive, dF_r, bF_r)
        traction = drive & (F_req > 0.0) & (dF_r >= rem_r - 1e-6) & ~capped
        lim = (capped & drive, traction,
               ~drive & ((Bk * share_f > rem_f) | (Bk * (1.0 - share_f) > rem_r)),
               drive & (F_req > 0.0) & ~capped & ~traction & (dF_r >= F_set - 1e-6))
        return Fx_f, Fx_r, Fy_f, Fy_r, F_drag, F_roll, m_x, lim

    def _shape(self, alpha):
        B, p = self.B, self.spec
        ba = p.B * alpha
        return B.sin(p.C * B.atan(ba - p.E * (ba - B.atan(ba))))

    def _derivatives(self, s, steer_cmd, accel_cmd):
        B, p = self.B, self.spec
        _, _, yaw, vx, vy, r, delta, a = s
        m, lf, lr = p.m, p.lf, p.lr
        L = lf + lr
        ddelta = B.clip((steer_cmd - delta) / p.steer_tc, -p.max_steer_rate, p.max_steer_rate)
        da = (accel_cmd - a) / p.accel_tc
        vx_s = B.maximum(vx, 0.5)
        alpha_f = delta - B.atan2(vy + lf * r, vx_s)
        alpha_r = -B.atan2(vy - lr * r, vx_s)
        Fx_f, Fx_r, Fy_f, Fy_r, F_drag, F_roll, m_x, lim = self._forces(vx, a, alpha_f, alpha_r)
        cd, sd = B.cos(delta), B.sin(delta)
        dvx_dyn = (Fx_r + Fx_f * cd - Fy_f * sd - F_drag - F_roll) / m_x + vy * r
        dvy_dyn = (Fy_r + Fx_f * sd + Fy_f * cd) / m - vx * r
        dr_dyn = (lf * (Fy_f * cd + Fx_f * sd) - lr * Fy_r) / p.yaw_inertia
        dvx_kin = (Fx_f + Fx_r - F_drag - F_roll) / m_x
        dvy_kin = (ddelta * vx + delta * dvx_kin) * lr / L
        dr_kin = (ddelta * vx + delta * dvx_kin) / L
        lam = B.clip((B.hypot(vx, vy) - p.v_lo) / B.maximum(p.v_hi - p.v_lo, 1e-3), 0.0, 1.0)
        dvx = lam * dvx_dyn + (1.0 - lam) * dvx_kin
        dvy = lam * dvy_dyn + (1.0 - lam) * dvy_kin
        dr = lam * dr_dyn + (1.0 - lam) * dr_kin
        dvx = B.where((vx <= 0.0) & (dvx < 0.0), 0.0, dvx)
        cy, sy = B.cos(yaw), B.sin(yaw)
        ds = (vx * cy - vy * sy, vx * sy + vy * cy, r, dvx, dvy, dr, ddelta, da)
        aux = (dvx - vy * r, dvy + vx * r, lam, (Fx_f, Fx_r, Fy_f, Fy_r, lim))
        return ds, aux

    # ------------------------------------------------------------------ #
    def set_command(self, steer_cmd, accel_cmd):
        """Clip to the actuator limits (VehicleModel.set_command)."""
        B, p = self.B, self.spec
        return B.clip(steer_cmd, -p.max_steer, p.max_steer), B.clip(accel_cmd, -p.max_decel, p.max_accel)

    def physics_step(self, steer_cmd, accel_cmd) -> None:
        """One physics step h (VehicleModel.step(h) with h <= max_substep: a single RK4 step)."""
        B, h = self.B, self.h
        s = self.state()
        k1, aux = self._derivatives(s, steer_cmd, accel_cmd)
        s2 = [x + 0.5 * h * k for x, k in zip(s, k1)]
        k2, _ = self._derivatives(s2, steer_cmd, accel_cmd)
        s3 = [x + 0.5 * h * k for x, k in zip(s, k2)]
        k3, _ = self._derivatives(s3, steer_cmd, accel_cmd)
        s4 = [x + h * k for x, k in zip(s, k3)]
        k4, _ = self._derivatives(s4, steer_cmd, accel_cmd)
        new = [x + (h / 6.0) * (a + 2 * b + 2 * c + d) for x, a, b, c, d in zip(s, k1, k2, k3, k4)]
        x, y, yaw, vx, vy, r, delta, a = new
        vx = B.maximum(vx, 0.0)
        lam = aux[2]
        L = self.spec.lf + self.spec.lr
        tn = B.tan(delta)
        vy = lam * vy + (1 - lam) * vx * tn * self.spec.lr / L
        r = lam * r + (1 - lam) * vx * tn / L
        yaw = (yaw + math.pi) % (2 * math.pi) - math.pi
        self.x, self.y, self.yaw, self.vx, self.vy, self.r, self.delta, self.a = x, y, yaw, vx, vy, r, delta, a
        _, (self.ax_b, self.ay_b, _, forces) = self._derivatives(self.state(), steer_cmd, accel_cmd)
        self.limit = forces[4]

    def step(self, steer_cmd, accel_cmd, dt: float) -> None:
        """Hold the (clipped) commands for dt seconds of physics steps."""
        sc, ac = self.set_command(steer_cmd, accel_cmd)
        n = max(1, int(round(dt / self.h)))
        for _ in range(n):
            self.physics_step(sc, ac)
