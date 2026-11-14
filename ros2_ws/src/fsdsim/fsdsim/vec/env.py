"""Vectorised RL environment for the universal throttle / brake policy (N cars, NumPy or PyTorch).

Every car drives its own scenario: an event (acceleration, skidpad, a generated autocross layout),
from the start line or - on autocross - from a random point at speed, with its own vehicle
specification sampled from `SpecRanges` and hidden variations (`Realism`). Steering is the reference
driver's pure pursuit; the policy chooses a in [-1, 1] every 40 ms (throttle a x max_accel, brake
a x max_decel). Physics: the batched fsdsim car (500 Hz).

Reward per step (metres, x reward_scale):
    progress along the path  - 30 per cone strike  - 0.05 x |change of action|
    leaving the track (all wheels out) or spinning (side-slip > 0.6 rad): - 100 and the episode ends
so the return is "distance in the time window, without cones, without losing the car". A cone costs
30 m, about the 2 s FSG penalty at autocross speeds.

Cones: fsdsim's contact check (car footprint rectangle vs each cone's base circle, every 8 ms of
physics), each cone counted once. Off course: the car's centre further than half the track width plus
half the car's width from the path (all four wheels out, as fsdsim's mission manager).

API (like Isaac Lab / gymnasium vector envs; all arrays on the backend's device):
    obs = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)     # finished cars restart
    env.evaluate(policy_fn, ...)                                       # official-style timed runs
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .backend import Backend
from .car import BatchCar
from .features import NOISE, N_FEATURES, features, teacher_action
from .limits import LimitTables
from .spec import ALL_FIELDS, Realism, SpecBatch, SpecRanges
from .steer import BatchFollower, unwrap_progress
from .tracks import DS, TrackBank

LAUNCH_SPEED, LAUNCH_MIN = 2.0, 0.5          # as fsdsim.control.rl_driver.RLDriver.act


@dataclass
class EnvConfig:
    n_envs: int = 256
    control_dt: float = 0.04
    physics_hz: float = 500.0
    missions: dict = field(default_factory=lambda: {"acceleration": 0.12, "skidpad": 0.13, "autocross": 0.75})
    horizon: dict = field(default_factory=lambda: {"acceleration": 5.0, "skidpad": 24.0, "autocross": 20.0})
    flying_start: float = 0.5               # autocross: share of episodes starting at a random point at speed
    v0_frac: tuple = (0.3, 0.85)            # ... at this share of the envelope speed there
    cone_penalty: float = 30.0              # [m] ~ 2 s (FSG cone penalty) at 15 m/s
    end_penalty: float = 100.0
    smooth_penalty: float = 0.05
    reward_scale: float = 0.1
    spin_beta: float = 0.6
    hold_decel: float = 5.0                 # standing start: brakes held until Go (fsdsim AS Ready)
    cone_check_every: int = 4               # physics steps between cone contact checks (4 x 2 ms)
    randomise: bool = True
    ranges: SpecRanges = field(default_factory=SpecRanges)
    realism: Realism = field(default_factory=Realism)


class VecLonEnv:
    def __init__(self, B: Backend, bank: TrackBank, base_spec: dict, cfg: EnvConfig | None = None, teacher=False):
        self.B = B
        self.cfg = cfg or EnvConfig()
        self.n = self.cfg.n_envs
        self.bank = bank if getattr(bank, "B", None) is B else bank.to(B)
        self.spec = SpecBatch(B, base_spec, self.n)
        self.car = BatchCar(B, self.spec, self.cfg.physics_hz)
        self.tables = LimitTables(B, self.spec)
        self.follow = BatchFollower(B, self.bank)
        self.teacher = teacher
        n = self.n
        self.tid = B.zeros(n, B.long)
        self.idx = B.zeros(n, B.long)
        self.prog = B.zeros(n)           # metres of progress since the start of the episode
        self.t = B.zeros(n)
        self.horizon = B.zeros(n)
        self.prev_a = B.zeros(n)
        self.a_buf = B.zeros(n)          # action waiting one step (latency)
        self.delay = B.zeros(n, B.long)
        self.knocked = B.zeros((n, self.bank.n_cones_max), dtype=B.np.bool_ if not B.is_torch else B.torch.bool)
        self.cones = B.zeros(n)
        self.ep_return = B.zeros(n)
        self.desc = self.spec.descriptor()
        self.steer_cmd = B.zeros(n)
        self.along = B.zeros(n)
        self.mark_t = None
        # mission categories -> track ids
        mis = bank.missions()
        self.cats = [m for m in self.cfg.missions if any(x == m for x in mis)]
        w = np.array([self.cfg.missions[m] for m in self.cats], float)
        self.cat_p = np.cumsum(w / w.sum())
        self.cat_ids = [np.array([i for i, x in enumerate(mis) if x == m], np.int64) for m in self.cats]
        hz = np.array([self.cfg.horizon.get(m, 20.0) for m in mis], float)
        self.track_h = B.asarray(hz)
        self.obs = None
        self.teacher_a = None
        self.stats = {}
        self.noise_on = True

    # ------------------------------------------------------------------ scenario sampling
    def _sample_tracks(self, k: int):
        B = self.B
        u = B.to_numpy(B.rand(k))
        cat = np.searchsorted(self.cat_p, u, side="right").clip(0, len(self.cats) - 1)
        pick = B.to_numpy(B.rand(k))
        tid = np.array([self.cat_ids[c][min(int(p * len(self.cat_ids[c])), len(self.cat_ids[c]) - 1)]
                        for c, p in zip(cat, pick)], np.int64)
        return B.asarray(tid)

    def _reset_idx(self, idx, tid=None, flying=None) -> None:
        B, cfg = self.B, self.cfg
        k = int(idx.numel()) if B.is_torch else len(idx)
        if k == 0:
            return
        if tid is None:
            tid = self._sample_tracks(k)
        self.tid[idx] = tid
        if cfg.randomise:
            self.spec.randomise(idx, cfg.ranges, cfg.realism)
            self.tables.update(idx, teacher=self.teacher)
            self.desc[idx] = self.spec.take(idx).descriptor()
            d0, d1 = cfg.realism.delay_steps
            self.delay[idx] = B.randint(int(d0), int(d1) + 1, k)
        bk = self.bank
        sp = bk.start_pose[tid]
        x, y, yaw = sp[:, 0], sp[:, 1], sp[:, 2]
        i0 = bk.start_idx[tid]
        v0 = B.zeros(k)
        if flying is None:
            is_ac = ~self._is_open(tid)
            flying = is_ac & (B.rand(k) < cfg.flying_start)
        if B.any(flying):
            n = bk.n[tid]
            frac = B.uniform(0.05, 0.9, k)
            ir = B.to_long(frac * B.to_float(n))
            ir2 = bk.wrap(tid, ir + 4)
            px, py = bk.px[tid, ir], bk.py[tid, ir]
            yr = B.atan2(bk.py[tid, ir2] - py, bk.px[tid, ir2] - px)
            sub = self.tables
            km = bk.kmax[tid, ir]
            from .features import envelope  # noqa: PLC0415
            env = envelope(B, sub.vc[idx], sub.dec[idx], km)
            vf = env[:, 0] * B.uniform(*cfg.v0_frac, k)
            x, y, yaw = B.where(flying, px, x), B.where(flying, py, y), B.where(flying, yr, yaw)
            i0 = B.where(flying, ir, i0)
            v0 = B.where(flying, vf, v0)
        a0 = B.where(v0 > 0.0, 0.0, -cfg.hold_decel)
        self.car.reset(idx, x, y, yaw, v0, a0)
        self.idx[idx] = i0
        self.prog[idx] = 0.0
        self.t[idx] = 0.0
        self.horizon[idx] = self.track_h[tid]
        self.prev_a[idx] = 0.0
        self.a_buf[idx] = 0.0
        self.knocked[idx] = False
        self.cones[idx] = 0.0
        self.ep_return[idx] = 0.0

    def _is_open(self, tid):
        return ~self.bank.closed[tid]

    # ------------------------------------------------------------------ observation
    def _observe(self, noise: bool | None = None):
        B, c, s = self.B, self.car, self.spec
        noise = self.noise_on if noise is None else noise
        self.idx = self.follow.nearest(self.tid, self.idx, c.x, c.y)
        self.steer_cmd = self.follow.steer(self.tid, self.idx, c.x, c.y, c.yaw, c.vx, s.lf + s.lr, s.lr, s.max_steer)
        e_lat, e_head, along = self.follow.tracking_error(self.tid, self.idx, c.x, c.y, c.yaw)
        self.e_lat, self.e_head, self.along = e_lat, e_head, B.clip(along, -DS, DS)
        v, vy, r, ax, ay = c.vx, c.vy, c.r, c.ax_b, c.ay_b
        el, eh = e_lat, e_head
        ns = self.cfg.realism.obs_noise if noise else 0.0
        if ns > 0:
            n = self.n
            v = v + ns * NOISE["v"] * B.randn(n)
            vy = vy + ns * NOISE["vy"] * B.randn(n)
            r = r + ns * NOISE["r"] * B.randn(n)
            ax = ax + ns * NOISE["ax"] * B.randn(n)
            ay = ay + ns * NOISE["ay"] * B.randn(n)
            el = el + ns * NOISE["e_lat"] * B.randn(n)
            eh = eh + ns * NOISE["e_head"] * B.randn(n)
        km = self.bank.kmax[self.tid, self.idx]
        obs, env = features(B, self.bank, self.tables, self.desc, self.tid, self.idx, v, vy, r, ax, ay, el, eh,
                            self.prev_a, km=km)
        self.env0 = env[:, 0]
        if self.teacher:
            self.teacher_a = teacher_action(B, self.tables, km, c.vx, s.max_accel, s.max_decel)
        self.obs = obs
        return obs

    # ------------------------------------------------------------------ API
    def reset(self, tid=None, flying=None):
        B = self.B
        all_ = B.arange(self.n)
        if tid is not None:
            tid = B.asarray(np.asarray(tid, np.int64))
        self._reset_idx(all_, tid=tid, flying=flying)
        return self._observe()

    def step(self, action, auto_reset: bool = True):
        B, c, s, cfg = self.B, self.car, self.spec, self.cfg
        a = B.clip(action, -1.0, 1.0)
        a = B.where(c.vx < LAUNCH_SPEED, B.maximum(a, LAUNCH_MIN), a)
        a_exec = B.where(self.delay > 0, self.a_buf, a)
        self.a_buf = a
        accel = B.where(a_exec >= 0.0, a_exec * s.max_accel, a_exec * s.max_decel)
        idx_prev, along_prev = self.idx, self.along
        hits = self._drive(accel)
        self.t = self.t + cfg.control_dt
        smooth = B.abs(a - self.prev_a)
        self.prev_a = a
        obs_next = self._observe()
        bk = self.bank
        d = unwrap_progress(B, self.idx - idx_prev, bk.n[self.tid], bk.closed[self.tid])
        dprog = B.to_float(d) * DS + (self.along - along_prev)
        self.prog = self.prog + dprog
        half_w = 0.5 * bk.width[self.tid]
        strike = hits
        self.cones = self.cones + strike
        off = (B.abs(self.e_lat) - 0.5 * s.car_width) > half_w
        spin = (c.vx > 3.0) & (B.abs(B.atan2(c.vy, B.maximum(c.vx, 1e-3))) > cfg.spin_beta)
        terminated = off | spin
        n_pts = bk.n[self.tid]
        end_of_path = self._is_open(self.tid) & (self.idx >= n_pts - 8)
        truncated = ((self.t >= self.horizon - 1e-6) | end_of_path) & ~terminated
        rew = (dprog - cfg.cone_penalty * strike - cfg.smooth_penalty * smooth
               - (cfg.end_penalty + cfg.cone_penalty * B.to_float(off)) * B.to_float(terminated))
        rew = rew * cfg.reward_scale
        self.ep_return = self.ep_return + rew
        done = terminated | truncated
        info = {"final_obs": obs_next, "prog": self.prog, "cones": self.cones, "spin": spin, "off": off,
                "t": self.t, "ep_return": self.ep_return}
        if auto_reset and B.any(done):
            info = {k: B.copy(v) for k, v in info.items()}
            ids = B.nonzero(done)
            self._reset_idx(ids)
            obs_next = self._observe()
        return obs_next, rew, terminated, truncated, info

    def _drive(self, accel):
        """Physics for one control step with cone contact checks; returns cones knocked per car."""
        B, c, s, bk = self.B, self.car, self.spec, self.bank
        sc, ac = c.set_command(self.steer_cmd, accel)
        n_sub = max(1, int(round(self.cfg.control_dt / c.h)))
        near = bk.cone_near[self.tid, self.idx]                       # (N, M) cones near the car (-1: none)
        valid = near >= 0
        ci = B.maximum(near, 0)
        tt = self.tid[:, None] + 0 * ci
        cx, cy, cr = bk.cone[tt, ci, 0], bk.cone[tt, ci, 1], bk.cone[tt, ci, 2]
        rows = B.arange(self.n)[:, None] + 0 * ci
        hits = B.zeros(self.n)
        fb, rb = s.lf + s.front_oh, s.lr + s.rear_oh
        centre, half_l, half_w = (0.5 * (fb - rb))[:, None], (0.5 * (fb + rb))[:, None], (0.5 * s.car_width)[:, None]
        every = max(1, int(self.cfg.cone_check_every))
        for k in range(n_sub):
            c.physics_step(sc, ac)
            if (k + 1) % every and k != n_sub - 1:
                continue
            dx, dy = cx - c.x[:, None], cy - c.y[:, None]
            co, sn = B.cos(c.yaw)[:, None], B.sin(c.yaw)[:, None]
            lx, ly = co * dx + sn * dy - centre, -sn * dx + co * dy
            ex = B.maximum(B.abs(lx) - half_l, 0.0)
            ey = B.maximum(B.abs(ly) - half_w, 0.0)
            hit = valid & (B.hypot(ex, ey) < cr) & ~self.knocked[rows, ci]
            if B.any(hit):
                self.knocked[rows[hit], ci[hit]] = True
                hits = hits + B.sum(B.to_float(hit), 1)
        return hits

    # ------------------------------------------------------------------ evaluation
    def evaluate(self, policy_fn, tids, rows=None, t_max: float = 120.0, record=None):
        """Timed runs like the competition: car i drives track tids[i] from the start line at standstill
        with spec rows[i] (dicts; default: the baseline), no randomisation, no noise, no latency.
        policy_fn(obs) -> actions (N,), or "teacher" for the reference driver's law.
        Returns per car: crossing times of the track's timing marks, cones, off-course, spin, stats."""
        B, cfg = self.B, self.cfg
        n = self.n
        tids = np.asarray(tids, np.int64)
        assert len(tids) == n, "one track id per car"
        if rows is not None:
            sb = SpecBatch.from_values(B, rows)
            for k in ALL_FIELDS:
                setattr(self.spec, k, getattr(sb, k))
        self.tables.update(B.arange(n), teacher=True)
        self.desc = self.spec.descriptor()
        was = cfg.randomise, self.teacher, self.noise_on
        cfg.randomise, self.teacher, self.noise_on = False, True, False
        self.delay = B.zeros(n, B.long)
        try:
            self._reset_idx(B.arange(n), tid=B.asarray(tids), flying=B.zeros_bool(n))
            self.horizon = B.full(n, t_max + 1.0)
            self._observe(noise=False)
            marks = self.bank.marks[B.asarray(tids)]
            nm = marks.shape[1]
            mark_t = B.full((n, nm), float("nan"))
            active = B.ones(n, dtype=B.long) > 0
            stat = {k: B.zeros(n) for k in ("v_max", "ay_max", "t_power", "t_traction", "t_brake_grip", "t_motor",
                                             "t_run", "a_sum")}
            failed = B.zeros_bool(n)
            spin_any, off_any = B.zeros_bool(n), B.zeros_bool(n)
            steps = int(t_max / cfg.control_dt)
            for _ in range(steps):
                obs = self.obs
                if policy_fn == "teacher":
                    act = self.teacher_a
                else:
                    act = policy_fn(obs)
                prog0 = self.prog
                obs, rew, term, trunc, info = self.step(act, auto_reset=False)
                # timing marks crossed in this step (linear interpolation in progress)
                p1 = self.prog
                cross = (prog0[:, None] < marks) & (p1[:, None] >= marks) & B.isnan(mark_t) & active[:, None]
                frac = (marks - prog0[:, None]) / B.maximum(p1 - prog0, 1e-9)[:, None]
                tc = (self.t - cfg.control_dt)[:, None] + B.clip(frac, 0.0, 1.0) * cfg.control_dt
                mark_t = B.where(cross, tc, mark_t)
                # stats while timing runs (between the first and the last mark)
                timing = active & (~B.isnan(mark_t[:, 0])) & B.isnan(mark_t[:, -1])
                tf = B.to_float(timing)
                stat["v_max"] = B.where(timing, B.maximum(stat["v_max"], self.car.vx), stat["v_max"])
                stat["ay_max"] = B.where(timing, B.maximum(stat["ay_max"], B.abs(self.car.ay_b)), stat["ay_max"])
                lim = self.car.limit
                stat["t_power"] = stat["t_power"] + tf * B.to_float(lim[0]) * cfg.control_dt
                stat["t_traction"] = stat["t_traction"] + tf * B.to_float(lim[1]) * cfg.control_dt
                stat["t_brake_grip"] = stat["t_brake_grip"] + tf * B.to_float(lim[2]) * cfg.control_dt
                stat["t_motor"] = stat["t_motor"] + tf * B.to_float(lim[3]) * cfg.control_dt
                stat["t_run"] = stat["t_run"] + tf * cfg.control_dt
                stat["a_sum"] = stat["a_sum"] + tf * self.prev_a
                spin_any = spin_any | (info["spin"] & active)
                off_any = off_any | (info["off"] & active)
                failed = failed | (term & active)
                # tracks with fewer marks: the remaining columns are +inf -> done when the real ones are set
                real = marks < float("inf")
                done_m = B.where(real, ~B.isnan(mark_t), True)
                finished = B.sum(B.to_float(done_m), 1) >= nm
                active = active & ~finished & ~term & ~trunc
                if not B.any(active):
                    break
            out = []
            mt = B.to_numpy(mark_t)
            st = {k: B.to_numpy(v) for k, v in stat.items()}
            cones = B.to_numpy(self.cones)
            fl, sp_, of_ = B.to_numpy(failed), B.to_numpy(spin_any), B.to_numpy(off_any)
            mk = B.to_numpy(marks)
            for i in range(n):
                m = self.bank.infos[int(tids[i])].mission
                times = [float(x) for x, mm in zip(mt[i], mk[i]) if np.isfinite(mm)]
                out.append({"track": int(tids[i]), "mission": m, "seed": self.bank.infos[int(tids[i])].seed,
                            "marks": times, "time": event_time(m, times), "cones": int(cones[i]),
                            "failed": bool(fl[i]), "spin": bool(sp_[i]), "off_course": bool(of_[i]),
                            **{k: float(st[k][i]) for k in st}})
            return out
        finally:
            cfg.randomise, self.teacher, self.noise_on = was


def event_time(mission: str, marks: list) -> float | None:
    """Official-style time from the timing mark crossings (None when not finished)."""
    if not marks or any(math.isnan(x) for x in marks):
        return None
    if mission == "skidpad":
        return 0.5 * ((marks[2] - marks[1]) + (marks[4] - marks[3]))
    return marks[-1] - marks[0]


__all__ = ["EnvConfig", "VecLonEnv", "event_time", "N_FEATURES"]
