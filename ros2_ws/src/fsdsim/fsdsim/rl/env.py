"""Training scenarios for the throttle / brake policy: headless fsdsim runs (same physics as Isaac).

A scenario is one mission on one track for a fixed time window:
  acceleration  from standstill, 4 s           (traction / launch)
  skidpad       the whole figure of eight, 24 s (steady-state cornering at the limit, both directions)
  autocross     a generated layout (seed), 18 s, from the start or a random point at speed

Fitness of one run (metres):
    distance driven along the path in the window
    - 10 per cone hit or off-course excursion  - 80 if the run ends early (left the track, spin)
    - 0.05 per unit of action change (smooth pedal work)
so maximising it means: as fast as possible, without cones, without losing the car.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import numpy as np

from ..config import load_all_configs
from ..control.rl_driver import RLDriver
from ..mission import ASState
from ..sim import Simulation

CONTROL_DT = 0.04          # 25 Hz policy
CONE_PENALTY = 10.0
END_PENALTY = 80.0
SMOOTH_PENALTY = 0.05
SPIN_BETA = 0.6            # rad side-slip -> spun


@dataclass(frozen=True)
class Scenario:
    mission: str
    seed: int = 0
    horizon: float = 14.0
    start_frac: float = 0.0    # autocross: start at this fraction of the lap (0 = start line)
    v0_frac: float = 0.0       # ... at this fraction of the tyre-limit speed

    @property
    def key(self):
        return (self.mission, self.seed)


DEFAULT_SCENARIOS = {
    "acceleration": dict(horizon=4.0),
    "skidpad": dict(horizon=24.0),
    "autocross": dict(horizon=18.0),
}


class ScenarioCache:
    """Builds Simulation + reference driver per (mission, seed) once and resets them per run."""

    def __init__(self, config_dir: str, changes=()):
        self.config_dir = config_dir
        self.changes = list(changes)
        self.veh, self.sens, self.simcfg = load_all_configs(config_dir, vehicle_overrides=self.changes)
        for s in (self.sens.imu, self.sens.gnss, self.sens.vehicle_sensors, self.sens.lidar, self.sens.camera):
            s.enabled = False
        c = self.simcfg
        c.sim.auto_activate = c.sim.auto_go = True
        c.sim.auto_go_wait_for_stack = False
        c.sim.go_delay_s = 0.05
        c.autonomous_system.ready_min_time_s = 0.0
        self._cache: dict = {}

    def get(self, sc: Scenario):
        if sc.key not in self._cache:
            cfg = copy.deepcopy(self.simcfg)
            cfg.events.track_generator.seed = sc.seed
            sim = Simulation(self.veh, self.sens, cfg, mission=sc.mission, enable_lidar=False, enable_camera=False)
            drv = RLDriver(None, self.veh, sc.mission, sim.track.width)
            drv.set_reference(sim.track.centerline, sim.track.closed)
            self._cache[sc.key] = (sim, drv)
        return self._cache[sc.key]


def _state(sim) -> dict:
    v = sim.vehicle
    s = v.state
    return {"vy": s.vy, "r": s.r, "ax": v.ax_body, "ay": v.ay_body}


def _place(sim, drv, sc: Scenario) -> None:
    """Autocross runs from a random point of the lap at speed (more varied corners per second)."""
    if sc.start_frac <= 0.0:
        return
    f = drv.ref.follower
    i = int(sc.start_frac * (drv.n - 1))
    p0, p1 = f.path[i], f.path[(i + 4) % drv.n]
    yaw = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    st = sim.vehicle.state
    st.x, st.y, st.yaw = float(p0[0]), float(p0[1]), yaw
    st.vx = float(sc.v0_frac * drv.v_lim[i])
    st.vy = st.r = 0.0
    f.idx = i
    drv._init = True
    drv.ref._initialised = True
    cl = sim.track.centerline                       # the mission's track-limit check follows progress too
    sim.mission._progress_idx = int(np.argmin(np.hypot(cl[:, 0] - p0[0], cl[:, 1] - p0[1])))


def rollout(policy, cache: ScenarioCache, sc: Scenario, driver: str = "rl", record=None) -> dict:
    """One run. driver 'rl' (policy) or 'reference' (for comparison / imitation data).
    record: list to append (features, reference action) pairs to (imitation learning)."""
    sim, drv = cache.get(sc)
    sim.reset()
    drv.policy = policy
    drv.prev_a = 0.0
    drv._init = False
    drv.ref._initialised = False
    f = drv.ref.follower
    # run the AS sequence up to Driving
    while sim.mission.state != ASState.DRIVING and sim.t < 2.0:
        sim.step(CONTROL_DT)
    _place(sim, drv, sc)
    if not drv.ref._initialised:
        f.initialise((sim.vehicle.state.x, sim.vehicle.state.y))
        drv.ref._initialised = True
    t0 = sim.t
    idx0 = last_idx = f.idx
    progress = 0
    prev_a, smooth, ended = 0.0, 0.0, ""
    st0 = sim.mission.status(sim.t)
    cones0, oc0 = int(st0.get("cones_hit", 0)), int(st0.get("off_course", 0))
    a_hist = []
    while sim.t - t0 < sc.horizon:
        s = sim.vehicle.state
        pose, v, st = (s.x, s.y, s.yaw), s.vx, _state(sim)
        steer, accel_ref, _ = drv.ref.update(pose, v, stop=False)
        idx = f.idx
        feats = None
        if driver == "rl" or record is not None:
            feats = drv.features(pose, v, idx, st)
        if record is not None:
            a_ref = accel_ref / drv.a_max if accel_ref >= 0 else accel_ref / drv.d_max
            record.append((feats, float(np.clip(a_ref, -1, 1))))
        if driver == "rl":
            a = drv.act(feats, v)
            drv.prev_a = a
        else:
            a = float(np.clip(accel_ref / drv.a_max if accel_ref >= 0 else accel_ref / drv.d_max, -1, 1))
            drv.prev_a = a
        smooth += abs(a - prev_a)
        prev_a = a
        a_hist.append(a)
        sim.set_command(steer, a * drv.a_max if a >= 0 else a * drv.d_max)
        sim.step(CONTROL_DT)
        # progress along the path (unwrapped on closed tracks)
        d = f.idx - last_idx
        if drv.closed:
            if d < -drv.n / 2:
                d += drv.n
            elif d > drv.n / 2:
                d -= drv.n
        progress += d
        last_idx = f.idx
        s = sim.vehicle.state
        if sim.mission.state == ASState.EMERGENCY:
            ended = "emergency (" + str(sim.mission.status(sim.t).get("message", "")) + ")"
            break
        if s.vx > 3.0 and abs(math.atan2(s.vy, s.vx)) > SPIN_BETA:
            ended = "spin"
            break
        if sim.mission.status(sim.t).get("mission_complete"):
            break
    st = sim.mission.status(sim.t)
    cones = int(st.get("cones_hit", 0)) - cones0
    off = int(st.get("off_course", 0)) - oc0
    dist = progress * f.ds
    fitness = dist - CONE_PENALTY * (cones + off) - (END_PENALTY if ended else 0.0) - SMOOTH_PENALTY * smooth
    # finishing early (acceleration) is good: credit the remaining window at the current speed
    if not ended and st.get("mission_complete") and sim.t - t0 < sc.horizon:
        fitness += (sc.horizon - (sim.t - t0)) * max(sim.vehicle.state.vx, 0.0)
    return {"fitness": float(fitness), "distance": float(dist), "cones": cones, "off_course": off, "ended": ended,
            "time": float(sim.t - t0), "laps": [float(x) for x in st.get("lap_times", [])] if st.get("lap_times") else [],
            "v_end": float(sim.vehicle.state.vx), "mean_action": float(np.mean(a_hist)) if a_hist else 0.0}


def make_scenarios(missions, seeds, rng: np.random.Generator, random_starts: bool = True) -> list[Scenario]:
    out = []
    for m in missions:
        base = DEFAULT_SCENARIOS.get(m, dict(horizon=18.0))
        if m in ("autocross", "trackdrive"):
            for sd in seeds:
                if random_starts and rng.random() < 0.5:
                    out.append(Scenario("autocross", int(sd), base["horizon"], float(rng.uniform(0.05, 0.9)),
                                        float(rng.uniform(0.4, 0.8))))
                else:
                    out.append(Scenario("autocross", int(sd), base["horizon"]))
        else:
            out.append(Scenario(m, 0, base["horizon"]))
    return out


def full_run(policy, cache: ScenarioCache, mission: str, seed: int = 0, tmax: float = 120.0) -> dict:
    """A complete mission as in the competition (start, timed run, stop): official time and penalties.
    policy None = the reference driver."""
    sim, drv = cache.get(Scenario(mission, seed))
    sim.reset()
    drv.policy = policy
    drv.prev_a = 0.0
    drv.ref._initialised = False
    while sim.t < tmax:
        s = sim.vehicle.state
        st = sim.mission.status(sim.t)
        stop = bool(st.get("mission_complete"))
        if sim.mission.state != ASState.DRIVING:
            drv.idle()
        if sim.mission.state in (ASState.READY, ASState.DRIVING):
            if policy is None:
                steer, accel, _ = drv.ref.update((s.x, s.y, s.yaw), s.vx, stop=stop)
            else:
                steer, accel, _ = drv.update((s.x, s.y, s.yaw), s.vx, stop=stop, state=_state(sim))
            sim.set_command(steer, accel)
        sim.step(CONTROL_DT)
        if sim.mission.state in (ASState.FINISHED, ASState.EMERGENCY):
            break
    r = sim.result()
    return {"mission": mission, "seed": seed, "state": r.state, "raw_time": r.raw_time, "final_time": r.final_time,
            "cones": r.cones_hit, "off_course": r.off_course, "lap_times": list(r.lap_times),
            "stopped_in_zone": r.extra.get("stopped_in_zone")}
