"""Minimum-lap-time optimal control (MLTOC) for the LR26: the fastest way around the track that the vehicle model
allows, with the racing line, speed and inputs optimised together.

Three models (docs/PERF_METHODS.md has the equations, hypotheses and limits):

    point_mass     quasi-steady: a point on its ggV envelope (fitted to the LapSim axle model) plus the
                   powertrain limits; the free line within the track. "QSS with a free trajectory".
    single_track   dynamic bicycle model: yaw inertia, steering angle and rate limits, slip angles with the
                   Magic-Formula shape, LapSim's load-sensitive axle potentials with quasi-static load transfer,
                   friction ellipse per axle, motor torque / power limits.
    double_track   as single_track per wheel: lateral load transfer per side, slip angle per wheel, yaw moment
                   from left/right force differences; drive with or without torque vectoring (one motor per rear
                   wheel on the LR26).

All three are written in curvilinear coordinates along the track's centre line (s: arc length, n: lateral offset,
xi: heading relative to the centre line), discretised in s with trapezoidal collocation, and solved as one
nonlinear programme with CasADi + IPOPT. The point-mass solution is the initial guess of the dynamic models.

Own implementation from the literature (Casanova 2000; Perantoni & Limebeer 2014; Lot & Biral 2014); no code was
copied from other projects. Needs the Python package `casadi` (includes IPOPT).
"""

from __future__ import annotations

import math
import time

import numpy as np

from fsdsim.perf import ggv as _ggv
from fsdsim.perf.lapsim import event_vehicle, score

from .lab import Lab, clean, decimate

G = 9.81
MODELS = ("point_mass", "single_track", "double_track")


def available() -> tuple[bool, str]:
    try:
        import casadi  # noqa: F401, PLC0415

        return True, casadi.__version__
    except ImportError:
        return False, "the Python package 'casadi' is not installed (launcher window: Repair)"


# =============================================================================================== #
# track in curvilinear form
# =============================================================================================== #
class Track:
    """Centre line resampled to a uniform step ds: s, curvature kappa, heading psi and x, y for drawing."""

    def __init__(self, s, kappa, closed: bool, ds: float, x=None, y=None, name: str = ""):
        s, kappa = np.asarray(s, float), np.asarray(kappa, float)
        L = float(s[-1])
        n = max(8, int(round(L / ds)))
        self.closed, self.L, self.name = bool(closed), L, name
        self.ds = L / n
        self.s = np.arange(n + (0 if closed else 1)) * self.ds          # closed: the last node wraps to the first
        h0 = float(np.median(np.diff(s)))
        if abs(self.ds - h0) < 1e-6 * max(h0, 1.0):
            self.kappa = np.interp(self.s, s, kappa)            # the table's own samples
        else:                                                 # mean curvature over each step (heading preserved)
            Kc = np.concatenate([[0.0], np.cumsum(0.5 * (kappa[:-1] + kappa[1:]) * np.diff(s))])
            a, b = self.s - self.ds / 2, self.s + self.ds / 2
            if closed:
                def Kint(x):
                    q, rem = np.divmod(x, L)
                    return q * Kc[-1] + np.interp(rem, s, Kc)
            else:
                def Kint(x):
                    return np.interp(np.clip(x, 0, L), s, Kc)
            width = (np.minimum(b, L) - np.maximum(a, 0)) if not closed else np.full_like(a, self.ds)
            self.kappa = (Kint(b) - Kint(a)) / np.maximum(width, 1e-9)
        # heading and position from the curvature (only used for drawing the line)
        psi = np.concatenate([[0.0], np.cumsum(0.5 * (self.kappa[:-1] + self.kappa[1:]) * self.ds)])
        if x is not None and y is not None and len(x) == len(s):
            # keep the event's own picture: heading from its x, y at the start
            x0, y0 = float(x[0]), float(y[0])
            k1 = min(3, len(x) - 1)
            psi0 = math.atan2(y[k1] - y[0], x[k1] - x[0])
        else:
            x0 = y0 = psi0 = 0.0
        self.psi = psi + psi0
        self.x = x0 + np.concatenate([[0.0], np.cumsum(self.ds * np.cos(0.5 * (self.psi[:-1] + self.psi[1:])))])
        self.y = y0 + np.concatenate([[0.0], np.cumsum(self.ds * np.sin(0.5 * (self.psi[:-1] + self.psi[1:])))])

    @property
    def n(self) -> int:
        return len(self.s)

    def nxt(self, k: int) -> int:
        return (k + 1) % self.n if self.closed else k + 1

    def intervals(self) -> range:
        return range(self.n if self.closed else self.n - 1)

    def line_xy(self, offset):
        """x, y of a line at lateral offset n (left positive) from the centre line."""
        off = np.asarray(offset, float)
        return self.x - off * np.sin(self.psi), self.y + off * np.cos(self.psi)


# =============================================================================================== #
# ggV envelope of the point-mass model
# =============================================================================================== #
def _quad_fit(v, y):
    A = np.stack([np.ones_like(v), v, v * v], axis=1)
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    return c


class Envelope:
    """Smooth ggV envelope fitted to the LapSim axle model.

    ay_max(v) = a0 + a1 v + a2 v^2          lateral limit (downforce makes it grow ~ v^2)
    ab(v), at(v)                           braking / traction-limited drive deceleration / acceleration at ay = 0
    (|ay|/ay_max)^p + (ax/a_lim)^p <= 1    combined, exponent p fitted separately for braking and traction
    The motor torque, power and speed limits are separate, exact constraints (see powertrain)."""

    def __init__(self, veh, v_top: float):
        speeds = np.linspace(2.0, max(v_top, 5.0), 18)
        probe = np.array([_ggv.ay_limit(veh, float(v)) for v in speeds])
        # only speeds the car can hold in a corner (a power cap, e.g. endurance, ends the steady states below the
        # rpm limit: the steady-state lateral limit drops to 0 there and would spoil the fit)
        ok = probe > 1.0
        last = int(np.argmax(~ok)) if (~ok).any() else len(speeds)
        speeds = speeds[:max(last, 4)]
        tab = _ggv.ggv(veh, speeds, n_ay=11)
        free = _ggv.ggv(veh.copy(T_motor=1e5, P_max=1e9), speeds, n_ay=11)    # traction only (no motor limit)
        ay_max = tab["ay"][:, -1]
        self.c_ay = _quad_fit(speeds, ay_max)
        self.c_b = _quad_fit(speeds, -tab["ax_dec"][:, 0])
        self.c_t = _quad_fit(speeds, np.maximum(free["ax_acc"][:, 0], 0.1))
        self.p_b, self.err_b = self._fit_p(speeds, tab["ay"], -tab["ax_dec"], self.c_ay, self.c_b)
        self.p_t, self.err_t = self._fit_p(speeds, free["ay"], np.maximum(free["ax_acc"], 0.0), self.c_ay, self.c_t)
        res = ay_max - self.poly(self.c_ay, speeds)
        self.err_ay = float(np.sqrt(np.mean(res ** 2)))
        self.table = {"v": speeds, "ay_max": ay_max, "ab": -tab["ax_dec"][:, 0], "at": free["ax_acc"][:, 0]}

    @staticmethod
    def poly(c, v):
        return c[0] + c[1] * v + c[2] * v * v

    def _fit_p(self, speeds, ay, ax, c_ay, c_ax):
        best = (math.inf, 2.0)
        for p in np.arange(1.2, 3.61, 0.05):
            err = []
            for i, v in enumerate(speeds):
                am, xm = self.poly(c_ay, v), self.poly(c_ax, v)
                for j in range(ay.shape[1] - 1):                      # the last point is the pure lateral limit
                    if not (np.isfinite(ay[i, j]) and np.isfinite(ax[i, j])):
                        continue
                    pred = xm * max(1 - min(max(ay[i, j] / am, 0.0), 1.0) ** p, 0.0) ** (1 / p)
                    err.append(pred - ax[i, j])
            r = float(np.sqrt(np.mean(np.square(err))))
            if r < best[0]:
                best = (r, float(p))
        return best[1], best[0]

    def info(self) -> dict:
        return {"ay_max_coef": self.c_ay.tolist(), "brake_coef": self.c_b.tolist(), "traction_coef": self.c_t.tolist(),
                "p_brake": self.p_b, "p_traction": self.p_t, "rms_ay": self.err_ay, "rms_brake": self.err_b,
                "rms_traction": self.err_t}


# =============================================================================================== #
# shared pieces
# =============================================================================================== #
class Car:
    """The numbers the OCPs need, from the LapSim set (PerfParams) and the simulator's vehicle.yaml."""

    def __init__(self, veh, sim=None):
        self.p = veh
        self.m = veh.m
        self.meff = veh.m * veh.e_i
        self.lf, self.lr = veh.lf, veh.lr
        self.L = veh.lf + veh.lr
        self.h = veh.h_cog
        self.sf, self.sr = veh.sf, veh.sr
        self.rw = veh.circ_ref / (2 * math.pi)
        self.kDF = 0.5 * veh.rho * veh.c_z * veh.A * veh.aero_eff
        self.kD = 0.5 * veh.rho * veh.c_w * veh.A * veh.aero_eff
        self.P = veh.P_max * veh.eta_motor if veh.P_at_accumulator > 0.5 else veh.P_max
        self.F_T = veh.T_motor * veh.i_trans * veh.eta_g / self.rw          # per driven axle, torque limited
        self.v_max = min(veh.n_max * 2 * math.pi / 60 / veh.i_trans * self.rw, veh.v_lim)
        self.awd = veh.drive_awd > 0.5
        self.Iz = getattr(sim, "yaw_inertia", 120.0) if sim is not None else 120.0
        act = getattr(sim, "actuators", None)
        self.max_steer = getattr(act, "max_steer", 0.42) if act else 0.42
        self.max_steer_rate = getattr(act, "max_steer_rate", 1.5) if act else 1.5
        tire = getattr(sim, "tire", None)
        self.B, self.C, self.E = (tire.B, tire.C, tire.E) if tire else (11.0, 1.45, 0.3)
        self.width = getattr(getattr(sim, "footprint", None), "width", 1.48) if sim is not None else 1.48

    def resistance(self, v):
        return self.kD * v * v + self.p.f_roll * (self.m * G + self.kDF * v * v)

    def shape(self, ca, alpha):
        Ba = self.B * alpha
        return ca.sin(self.C * ca.atan(Ba - self.E * (Ba - ca.atan(Ba))))

    def loads(self, v, ax, ay):
        """Quasi-static wheel loads fl, fr, rl, rr (ay > 0: left-hand corner, the left wheels unload)."""
        p, m, L = self.p, self.m, self.L
        DF = self.kDF * v * v
        Ff = m * G * p.lr / L + DF * p.aero_bal_f - m * ax * p.h_cog / L
        Fr = m * G * p.lf / L + DF * (1 - p.aero_bal_f) + m * ax * p.h_cog / L
        dyf = m * ay * p.h_cog / p.sf * p.lr / L
        dyr = m * ay * p.h_cog / p.sr * p.lf / L
        return Ff / 2 - dyf, Ff / 2 + dyf, Fr / 2 - dyr, Fr / 2 + dyr

    def potentials(self, fz, util_x=1.0, util_y=1.0):
        p = self.p
        px = fz * (p.mux + p.dmux_dFz * (fz - p.Fz0)) * util_x
        py = fz * (p.muy + p.dmuy_dFz * (fz - p.Fz0)) * util_y
        return px, py


def _ipopt_opts(max_iter: int, verbose: bool) -> dict:
    return {"ipopt.max_iter": max_iter, "ipopt.tol": 1e-6, "ipopt.acceptable_tol": 1e-4,
            "ipopt.acceptable_iter": 10, "ipopt.print_level": 5 if verbose else 0, "print_time": False}


class _Progress:
    def __init__(self, job, label, max_iter):
        self.job, self.label, self.max_iter = job, label, max_iter

    def __call__(self, i):
        if self.job is not None:
            self.job.step(min(i, self.max_iter), self.max_iter, f"{self.label}: IPOPT iteration {i}")


# =============================================================================================== #
# point mass on the ggV envelope (quasi-steady, free trajectory)
# =============================================================================================== #
def _idx(tr: Track):
    """Index arrays of interval start and end nodes."""
    k = np.array(list(tr.intervals()))
    return k, np.array([tr.nxt(int(i)) for i in k])


def _solve(opti, job, label, max_iter, verbose):
    opti.solver("ipopt", _ipopt_opts(max_iter, verbose))
    opti.callback(_Progress(job, label, max_iter))
    t0 = time.time()
    try:
        sol = opti.solve()
        ok, val = True, sol.value
    except RuntimeError:
        ok, val = False, opti.debug.value
    st = opti.stats()
    return val, {"ok": ok, "status": str(st.get("return_status")), "iterations": int(st.get("iter_count", 0)),
                 "solve_s": time.time() - t0}


def solve_point_mass(car: Car, env: Envelope, tr: Track, half_width: float, v0: float | None,
                     fixed_line: bool = False, max_iter: int = 3000, job=None, verbose: bool = False,
                     init: dict | None = None) -> dict:
    """States n, xi, v at the nodes; the inputs ax (net acceleration) and kp (curvature of the driven path) are
    constant over each interval, and the ggV / powertrain limits hold at both ends of it. (Inputs at the nodes with
    trapezoidal integration let kp zig-zag around the track curvature and beat the envelope.)"""
    import casadi as ca  # noqa: PLC0415

    N, ds = tr.n, tr.ds
    ik, ij = _idx(tr)
    M = len(ik)
    Kk, Kj = tr.kappa[ik], tr.kappa[ij]
    opti = ca.Opti()
    n, xi, v = opti.variable(N), opti.variable(N), opti.variable(N)
    ax, kp = opti.variable(M), opti.variable(M)
    eps = 1e-6

    def spow(z2, p):                            # (z^2)^(p/2), smooth at 0 (no infinite curvature for p < 2)
        return (z2 + eps) ** (p / 2)

    def pos(x):                                 # smooth max(x, 0)
        return 0.5 * (x + ca.sqrt(x * x + eps))

    def limits(vv, a, k_path):
        ay = vv * vv * k_path
        lat2 = ay * ay / env.poly(env.c_ay, vv) ** 2
        Fw = car.meff * a + car.resistance(vv)
        return [spow(lat2, env.p_b) + spow((pos(-a) / env.poly(env.c_b, vv)) ** 2, env.p_b) - 1,
                spow(lat2, env.p_t) + spow((pos(a) / env.poly(env.c_t, vv)) ** 2, env.p_t) - 1,
                (Fw - car.F_T * (2 if car.awd else 1)) / (car.m * G),
                (Fw * vv - car.P * car.p.eta_g) / 1e4]

    nk, nj, xk, xj, vk, vj = n[ik], n[ij], xi[ik], xi[ij], v[ik], v[ij]
    hk = (1 - nk * Kk) / ca.cos(xk)
    hj = (1 - nj * Kj) / ca.cos(xj)
    hm = 0.5 * (hk + hj)
    opti.subject_to(nj - nk == ds * 0.5 * ((1 - nk * Kk) * ca.tan(xk) + (1 - nj * Kj) * ca.tan(xj)))
    opti.subject_to(xj - xk == ds * (kp * hm - 0.5 * (Kk + Kj)))
    opti.subject_to(vj ** 2 - vk ** 2 == 2 * ds * ax * hm)
    for c in limits(vk, ax, kp) + limits(vj, ax, kp):
        opti.subject_to(c <= 0)
    T = ca.sum1(ds * hm * 2 / (vk + vj + 1e-3))
    reg = 50 * ca.sumsqr(kp[1:] - kp[:-1]) + 1e-2 * ca.sumsqr((ax[1:] - ax[:-1]) / G)
    opti.minimize(T + 1e-3 * reg)
    opti.subject_to(opti.bounded(0.0, v, car.v_max))
    opti.subject_to(opti.bounded(-1 / 3.0, kp, 1 / 3.0))
    opti.subject_to(opti.bounded(-1.2, xi, 1.2))
    if fixed_line:
        opti.subject_to(n == 0)
        opti.subject_to(xi == 0)
    else:
        opti.subject_to(opti.bounded(-half_width, n, half_width))
    if not tr.closed:
        if v0 is not None:
            opti.subject_to(v[0] == v0)
        if not fixed_line:
            opti.subject_to(n[0] == 0)
            opti.subject_to(xi[0] == 0)

    g = init or {}
    opti.set_initial(v, np.maximum(g.get("v", np.full(N, 10.0)), 1.0))
    opti.set_initial(n, g.get("n", np.zeros(N)))
    opti.set_initial(xi, g.get("xi", np.zeros(N)))
    opti.set_initial(kp, g.get("kp_int", 0.5 * (Kk + Kj)))
    opti.set_initial(ax, g.get("ax_int", np.zeros(M)))
    val, stats = _solve(opti, job, "point mass", max_iter, verbose)
    out = {k: np.asarray(val(x)).ravel() for k, x in (("n", n), ("xi", xi), ("v", v))}
    a_int, k_int = np.asarray(val(ax)).ravel(), np.asarray(val(kp)).ravel()
    # node values (plots, first guess of the dynamic models): mean of the neighbouring intervals
    prev = np.array([(k - 1) % M if tr.closed else max(k - 1, 0) for k in range(N)])
    cur = np.array([k % M if tr.closed else min(k, M - 1) for k in range(N)])
    out["ax"], out["kp"] = 0.5 * (a_int[prev] + a_int[cur]), 0.5 * (k_int[prev] + k_int[cur])
    out["ay"] = out["v"] ** 2 * out["kp"]
    out.update(stats)
    return out


# =============================================================================================== #
# dynamic single- / double-track
# =============================================================================================== #
def steady_guess(car: Car, pm: dict, util) -> dict:
    """First guess of the dynamic states from the point-mass solution: at every node the steady-state (constant
    speed, constant curvature) single-track solution: axle lateral forces from the moment balance, slip angles by
    inverting the Magic-Formula shape on the rising branch, then side slip and steering angle."""
    v = np.maximum(pm["v"], 1.0)
    kp = pm["kp"]
    ax, ay = pm["ax"], v * v * kp
    r = v * kp
    a = np.linspace(0.0, 0.6, 600)
    Ba = car.B * a
    shp = np.sin(car.C * np.arctan(Ba - car.E * (Ba - np.arctan(Ba))))
    ipk = int(np.argmax(shp))
    a_up, s_up = a[: ipk + 1], shp[: ipk + 1]
    alpha = {}
    for ax_name, share, idx in (("f", car.lr / car.L, (0, 1)), ("r", car.lf / car.L, (2, 3))):
        fz = car.loads(v, ax, ay)
        py = sum(car.potentials(np.maximum(fz[i], 0.0), *util)[1] for i in idx)
        need = np.clip(np.abs(car.m * ay * share) / np.maximum(py, 1.0), 0.0, 0.98 * s_up[-1])
        alpha[ax_name] = np.sign(ay) * np.interp(need, s_up, a_up)
    vy = car.lr * r - v * np.tan(alpha["r"])
    delta = alpha["f"] + np.arctan2(vy + car.lf * r, v)
    return {"n": pm["n"], "xi": pm["xi"], "vx": v, "vy": vy, "r": r,
            "delta": np.clip(delta, -car.max_steer, car.max_steer), "axg": ax / G, "ayg": ay / G}


# =============================================================================================== #
# =============================================================================================== #
def solve_dynamic(car: Car, tr: Track, half_width: float, v0: float | None, init: dict, model: str = "single_track",
                  torque_vectoring: bool = False, util=(1.0, 1.0), max_iter: int = 3000, job=None,
                  verbose: bool = False, fixed_line: bool = False) -> dict:
    """States n, xi, vx, vy, r, delta at the nodes; inputs: steering rate and the longitudinal tyre forces (per axle
    or per wheel, / m g) at the nodes; algebraic: the body accelerations ax, ay (/ g) of the quasi-static load
    transfer. Trapezoidal collocation in s. Vectorised over all nodes."""
    import casadi as ca  # noqa: PLC0415

    double = model == "double_track"
    N, ds, K = tr.n, tr.ds, tr.kappa
    mg = car.m * G
    opti = ca.Opti()
    n, xi, vx, vy, r, de = (opti.variable(N) for _ in range(6))
    dde = opti.variable(N)
    nw = 4 if double else 2
    fx = opti.variable(N, nw)
    axa, aya = opti.variable(N), opti.variable(N)
    ux, uy = util
    tf, trr = car.sf / 2, car.sr / 2

    V, Vy, R, D = vx, vy, r, de
    vv = ca.fmax(V, 0.5)
    fz = car.loads(V, axa * G, aya * G)
    pw = [car.potentials(f, ux, uy) for f in fz]
    ell = []
    if double:
        pos = ((car.lf, tf), (car.lf, -tf), (-car.lr, trr), (-car.lr, -trr))
        FXb = FYb = MZ = 0
        for i, (xw, yw) in enumerate(pos):
            Fx = fx[:, i] * mg
            d = D if i < 2 else 0
            alpha = d - ca.atan2(Vy + R * xw, ca.fmax(V - R * yw, 0.5))
            S = car.shape(ca, alpha)
            px, py = pw[i]
            fy = py * S
            ell.append((Fx / (px + 1.0)) ** 2 + S ** 2)
            if i < 2:
                fxb, fyb = Fx * ca.cos(D) - fy * ca.sin(D), Fx * ca.sin(D) + fy * ca.cos(D)
            else:
                fxb, fyb = Fx, fy
            FXb = FXb + fxb
            FYb = FYb + fyb
            MZ = MZ + xw * fyb - yw * fxb
    else:
        af = D - ca.atan2(Vy + car.lf * R, vv)
        ar = -ca.atan2(Vy - car.lr * R, vv)
        Sf, Sr = car.shape(ca, af), car.shape(ca, ar)
        pxf, pyf = pw[0][0] + pw[1][0], pw[0][1] + pw[1][1]
        pxr, pyr = pw[2][0] + pw[3][0], pw[2][1] + pw[3][1]
        Fxf, Fxr = fx[:, 0] * mg, fx[:, 1] * mg
        Fyf, Fyr = pyf * Sf, pyr * Sr
        ell += [(Fxf / (pxf + 1.0)) ** 2 + Sf ** 2, (Fxr / (pxr + 1.0)) ** 2 + Sr ** 2]
        FXb = Fxr + Fxf * ca.cos(D) - Fyf * ca.sin(D)
        FYb = Fyr + Fxf * ca.sin(D) + Fyf * ca.cos(D)
        MZ = car.lf * (Fyf * ca.cos(D) + Fxf * ca.sin(D)) - car.lr * Fyr
    Fres = car.resistance(V)
    sdot = (V * ca.cos(xi) - Vy * ca.sin(xi)) / (1 - n * K)
    f = [V * ca.sin(xi) + Vy * ca.cos(xi),                    # dn/dt
         R - K * sdot,                                        # dxi/dt
         (FXb - Fres) / car.meff + Vy * R,                    # dvx/dt
         FYb / car.m - V * R,                                 # dvy/dt
         MZ / car.Iz,                                         # dr/dt
         dde]                                                 # ddelta/dt
    X = [n, xi, vx, vy, r, de]
    ik, ij = _idx(tr)
    inv = 1 / sdot
    for x, fi in zip(X, f):
        g = fi * inv                                          # d/ds
        opti.subject_to(x[ij] - x[ik] == ds * 0.5 * (g[ik] + g[ij]))
    T = ca.sum1(ds * 0.5 * (inv[ik] + inv[ij]))
    reg = ca.sumsqr(dde[ij] - dde[ik]) + ca.sumsqr(fx[ij, :] - fx[ik, :])
    opti.minimize(T + 1e-3 * reg + 1e-4 * ca.sumsqr(dde))
    for e in ell:
        opti.subject_to(e <= 1)
    opti.subject_to(axa == (FXb - Fres) / mg)
    opti.subject_to(aya == FYb / mg)
    opti.subject_to(sdot >= 0.3)

    rear = (2, 3) if double else (1,)
    front = (0, 1) if double else (0,)
    per = 0.5 if double else 1.0
    drive_cols = rear + (front if car.awd else ())
    Fd = 0
    for i in drive_cols:
        Fd = Fd + fx[:, i] * mg
        opti.subject_to(fx[:, i] * mg <= car.F_T * per)
    opti.subject_to(Fd * vx / 1e4 <= car.P * car.p.eta_g / 1e4)
    if not car.awd:
        for i in front:
            opti.subject_to(fx[:, i] <= 0)
    if double and not torque_vectoring:
        opti.subject_to(fx[:, 0] == fx[:, 1])
        opti.subject_to(fx[:, 2] == fx[:, 3])
    opti.subject_to(opti.bounded(-car.max_steer, de, car.max_steer))
    opti.subject_to(opti.bounded(-car.max_steer_rate, dde, car.max_steer_rate))
    opti.subject_to(opti.bounded(0.5, vx, car.v_max))
    opti.subject_to(opti.bounded(-1.0, xi, 1.0))
    if fixed_line:
        opti.subject_to(n == 0)
    else:
        opti.subject_to(opti.bounded(-half_width, n, half_width))
    if not tr.closed:
        if not fixed_line:
            opti.subject_to(n[0] == 0)
        opti.subject_to(xi[0] == 0)
        opti.subject_to(vy[0] == 0)
        opti.subject_to(r[0] == 0)
        if v0 is not None:
            opti.subject_to(vx[0] == v0)

    if init.get("_states"):                          # from a dynamic solution
        g0 = {"n": init["n"], "xi": init["xi"], "vx": init["vx"], "vy": init["vy"], "r": init["r"], "delta": init["delta"],
              "axg": init["ax"] / G, "ayg": init["ay"] / G}
    else:
        g0 = steady_guess(car, init, util)
    for x, key in ((n, "n"), (xi, "xi"), (vx, "vx"), (vy, "vy"), (r, "r"), (de, "delta"), (axa, "axg"), (aya, "ayg")):
        opti.set_initial(x, g0[key])
    opti.set_initial(dde, 0.0)
    v = g0["vx"]
    fx0 = np.zeros((N, nw))
    a = init["ax"] * car.meff / mg
    if init.get("_states") and init["fx"].shape[1] != nw:       # axle forces -> half per wheel
        fx0 = np.repeat(init["fx"] / mg / 2, 2, axis=1)
    elif init.get("_states"):
        fx0 = init["fx"] / mg
    elif double:
        fx0[:, 2] = fx0[:, 3] = np.where(a > 0, a / 2, a * 0.25)
        fx0[:, 0] = fx0[:, 1] = np.where(a > 0, 0.0, a * 0.25)
    else:
        fx0[:, 1] = np.where(a > 0, a, a * 0.5)
        fx0[:, 0] = np.where(a > 0, 0.0, a * 0.5)
    if not car.awd:
        fx0[:, :nw // 2] = np.minimum(fx0[:, :nw // 2], 0.0)
    opti.set_initial(fx, fx0)
    val, stats = _solve(opti, job, model.replace("_", " "), max_iter, verbose)
    out = {k: np.asarray(val(x)).ravel() for k, x in
           (("n", n), ("xi", xi), ("vx", vx), ("vy", vy), ("r", r), ("delta", de), ("ax", axa), ("ay", aya))}
    out["ax"] *= G
    out["ay"] *= G
    out["fx"] = np.asarray(val(fx)).reshape(N, nw) * mg
    out["v"] = np.hypot(out["vx"], out["vy"])
    out["beta"] = np.degrees(np.arctan2(out["vy"], np.maximum(out["vx"], 0.5)))
    out.update(stats)
    return out


# =============================================================================================== #
# driver: one event, one model
# =============================================================================================== #
def _time_of(tr: Track, sol: dict) -> np.ndarray:
    """Time along the lap from the solution (same rule as the objective)."""
    n, xi, K = sol["n"], sol["xi"], tr.kappa
    if "vx" in sol:
        sdot = (sol["vx"] * np.cos(xi) - sol["vy"] * np.sin(xi)) / (1 - n * K)
        inv = 1 / np.maximum(sdot, 1e-3)
        dt = [tr.ds * 0.5 * (inv[k] + inv[tr.nxt(k)]) for k in tr.intervals()]
    else:
        h = (1 - n * K) / np.cos(xi)
        v = sol["v"]
        dt = [tr.ds * 0.5 * (h[k] + h[tr.nxt(k)]) * 2 / (v[k] + v[tr.nxt(k)] + 1e-3) for k in tr.intervals()]
    return np.concatenate([[0.0], np.cumsum(dt)])


def _guess_from_dynamic(sol: dict) -> dict:
    """A dynamic solution as the first guess of the next model (its states are used as they are)."""
    g = dict(sol)
    g["kp"] = sol["r"] / np.maximum(sol["vx"], 1.0)
    g["_states"] = True
    return g


def qss_on_track(veh, ev, tr: Track) -> float:
    """LapSim's QSS lap time on the OCP's own grid (same curvature samples): separates the effect of the grid from
    the effect of the model."""
    import copy  # noqa: PLC0415

    from fsdsim.perf import lapsim as LS  # noqa: PLC0415

    k = np.append(tr.kappa, tr.kappa[0]) if tr.closed else tr.kappa
    s = np.arange(len(k)) * tr.ds
    e2 = copy.copy(ev)
    e2.track = LS.Track(ev.track.name, s, k, np.zeros_like(s), np.zeros_like(s), ev.track.closed)
    return float(LS.run_event(veh, e2)["t_lap"])


def run(lab: Lab, event: str, model: str = "point_mass", changes=(), competition=None, track_width: float = 3.5,
        ds: float | None = None, use_event_util: bool = True, torque_vectoring: bool = False,
        fixed_line: bool = True, max_iter: int = 3000, job=None, verbose: bool = False) -> dict:
    """Minimum-time solution of one event with one model. Returns times, points and the traces."""
    ok, why = available()
    if not ok:
        raise ValueError(why)
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r} (use {', '.join(MODELS)})")
    t_wall = time.time()
    changes = lab.validate(changes, competition)
    ls = lab.lapsim(competition)
    comp = ls.competition(changes)
    if event not in comp.events:
        raise ValueError(f"unknown event {event!r}")
    ev = comp.events[event]
    veh = event_vehicle(ls.vehicle(changes), ev)
    util = (float(ev.util_x), float(ev.util_y)) if use_event_util else (1.0, 1.0)
    vehU = veh.copy(util_x=util[0], util_y=util[1])
    from .transient import build_vehicle  # noqa: PLC0415

    sim, _ = build_vehicle(lab, changes, event, competition)
    car = Car(vehU, sim)
    trk = ev.track
    # 1 m grid, curvature averaged over each metre (heading kept). Finer grids follow the 0.5 m curvature tables
    # more closely but make closed-track problems much slower; qss_same_grid tells the effect of the grid.
    ds = float(ds or 1.0)
    if trk.L / ds > 3000:
        ds = trk.L / 3000
    tr = Track(trk.s, trk.k, trk.closed, ds, trk.x, trk.y, trk.name)
    half = max((float(track_width) - car.width) / 2, 0.0)
    v0 = None if tr.closed else (0.0 if model == "point_mass" else 1.0)
    if job is not None:
        job.step(0, 3, "fitting the ggV envelope")
    env = Envelope(vehU, car.v_max)

    # QSS speeds as the first guess of the point mass
    q = lab.raw(changes, competition, [event])["events"][event]
    v_q = np.interp(tr.s, q["s"], q["v"])
    pm = solve_point_mass(car, env, tr, half, v0, fixed_line, max_iter, job, verbose,
                          init={"v": v_q, "kp": tr.kappa, "ax": np.zeros(tr.n)})
    sol, stages = pm, [{"model": "point_mass", "ok": pm["ok"], "status": pm["status"], "iterations": pm["iterations"],
                        "solve_s": pm["solve_s"]}]
    chain = [] if model == "point_mass" else ["single_track"] if model == "single_track" else ["single_track", model]
    for mdl in chain:                                # each model starts from the solution of the simpler one
        guess = sol if mdl == "single_track" else _guess_from_dynamic(sol)
        sol = solve_dynamic(car, tr, half, v0, guess, mdl, torque_vectoring and mdl == "double_track", util,
                            max_iter, job, verbose, fixed_line)
        stages.append({"model": mdl, "ok": sol["ok"], "status": sol["status"], "iterations": sol["iterations"],
                       "solve_s": sol["solve_s"]})
    q_grid = qss_on_track(veh, ev, tr)
    t = _time_of(tr, sol)
    t_lap = float(t[-1])
    t_start = 0.0
    if not tr.closed and v0:                         # dynamic models start at v0 > 0: add the time to reach it
        a0 = max(env.poly(env.c_t, 0.5), 1.0)
        t_start = v0 / a0
        t_lap += t_start
    if tr.closed:
        T = ev.laps * t_lap if ev.timing == "flying_lap" else t_lap * ev.laps     # periodic = flying laps
    else:
        T = t_lap
    xl, yl = tr.line_xy(sol["n"])
    v = sol["v"]
    # wheel power and accumulator energy of one lap (constant eta_motor; regen as in LapSim)
    p = vehU
    Fw = car.meff * sol["ax"] + car.resistance(v)
    Pw = Fw * v
    Pb = np.where(Pw >= 0, Pw / (p.eta_g * p.eta_motor), -np.minimum(-Pw, p.P_regen) * p.eta_regen)
    E = float(np.sum(0.5 * (Pb[:-1] + Pb[1:]) * np.diff(t[: len(Pb)])) if len(t) >= len(Pb) else 0.0)
    if tr.closed:
        tt = t[:-1]
    else:
        tt = t
    sl = decimate(tr.n, 1500)
    out = {"event": event, "model": model, "torque_vectoring": bool(torque_vectoring), "fixed_line": bool(fixed_line),
           "ok": all(s["ok"] for s in stages), "stages": stages, "t_lap": t_lap, "T": T, "laps": ev.laps,
           "t_start_correction": t_start, "points": score(T, ev.score), "v_max_kmh": float(np.max(v) * 3.6),
           "E_lap_kwh": E / 3.6e6, "half_width": half, "track_width": float(track_width), "ds": tr.ds,
           "nodes": tr.n, "util": list(util), "envelope": env.info(), "wall_s": time.time() - t_wall,
           "qss": {"T": q["T"], "t_lap": q["t_lap"], "points": q["points"], "t_lap_same_grid": q_grid},
           "trace": {"s": tr.s[sl], "t": tt[sl], "v": v[sl] * 3.6, "n": sol["n"][sl], "ax": sol["ax"][sl] / G,
                     "ay": sol["ay"][sl] / G, "x": xl[sl], "y": yl[sl]},
           "centre": {"x": tr.x[sl], "y": tr.y[sl]}}
    warn = []
    if not out["ok"]:
        bad = [st for st in stages if not st["ok"]][0]
        warn.append(f"The {bad['model'].replace('_', ' ')} stage did not converge ({bad['status']}). No time or points "
                    "are given; the traces show the last iterate and must not be used. Try the simpler model, a "
                    "coarser grid (ds) or more iterations.")
        out.update(T=None, t_lap=None, points=None, E_lap_kwh=None)
    if not fixed_line:
        warn.append("Free line: the car may use the whole track width around the given line. The FSA tables are "
                    "already a driven line (documentation curvature tables with small wiggles, or the real tracks' "
                    "driven line), so a free line re-optimises a line that is already optimised and the time is too "
                    "good. Use the free line only on a true centre line with a known width.")
    out["warnings"] = warn
    if model != "point_mass":
        out["trace"].update(delta=np.degrees(sol["delta"][sl]), beta=sol["beta"][sl], r=np.degrees(sol["r"][sl]),
                            fx=(sol["fx"][sl] / 1e3))
    if job is not None:
        job.step(3, 3, "done")
    return clean(out)
