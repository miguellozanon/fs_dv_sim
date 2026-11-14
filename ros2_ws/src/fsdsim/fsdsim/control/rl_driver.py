"""Driver with a learned throttle / brake policy (reinforcement learning, fsdsim.rl).

Steering stays with the reference path follower (pure pursuit) and so does the stop after the
mission is complete (a rules / safety function). While the car is racing, the longitudinal command
comes from the policy:

    a in [-1, 1]  ->  accel = a * max_accel (a >= 0, throttle)  or  a * max_decel (a < 0, brake)

The policy sees (features()):
  * its speed, and the speed the LapSim model of *this* car allows at the tyre limit at points
    0 ... 60 m ahead on the path (QSS profile at full grip: the vehicle specification enters here),
  * path curvature at the same points,
  * body motion: lateral velocity, yaw rate, longitudinal / lateral acceleration, side-slip angle,
  * path tracking error and heading error, and its previous action.

Policies with features version 2 (the *universal* policy, fsdsim.vec, trained with PPO on many
vehicle specifications at once) see the speed envelope of this car 0 ... 60 m ahead, its current
traction / braking capability and the vehicle specification itself; the inputs are computed by
fsdsim.vec.features - the same code the (GPU) training used.
"""

from __future__ import annotations

import math

import numpy as np

LOOKAHEAD_M = (0.0, 4.0, 8.0, 14.0, 22.0, 32.0, 45.0, 60.0)
FEATURES_VERSION = 1
N_FEATURES = 1 + 2 * len(LOOKAHEAD_M) + 4 + 3 + 1 + 1


class RLDriver:
    """Same interface as DemoDriver: set_reference(path, closed), update(...) -> (steer, accel, info)."""

    def __init__(self, policy, veh, mission: str = "", track_width: float = 3.5, ref_driver=None):
        from .demo_driver import make_driver  # noqa: PLC0415

        self.policy = policy
        self.veh = veh
        self.mission = mission
        self.ref = ref_driver or make_driver(veh, mission, "map", track_width)
        self.mode = "map"
        self.a_max = float(veh.actuators.max_accel)
        self.d_max = float(veh.actuators.max_decel)
        self.prev_a = 0.0
        self.v_lim = None
        self._init = False
        self._v2 = None            # features v2 context (track bank, limit tables), built on demand

    # ------------------------------------------------------------------ #
    def set_reference(self, path, closed):
        from ..perf import qss  # noqa: PLC0415

        self.ref.set_reference(path, closed)
        f = self.ref.follower
        perf = self.veh.perf()
        if perf is None:
            raise ValueError("the RL driver needs a performance (LapSim) vehicle (vehicle.yaml: model: performance)")
        self.closed = closed
        self._v2 = None
        if self.version == 2:
            self._setup_v2()
            self.v_lim = None
        else:
            # tyre-limit speed of this car along the path (LapSim QSS, full grip, free end speed)
            self.v_lim = qss.reference_speed(perf, f.path, closed, 1.0, v_end=None, util_long=1.0)
        self.n = len(f.path)
        self._s_steps = [int(round(d / f.ds)) for d in LOOKAHEAD_M]
        self.prev_a = 0.0
        self._init = False

    @property
    def follower(self):
        """The reference driver's path follower (same path and index as the RL driver uses)."""
        return self.ref.follower

    @property
    def version(self) -> int:
        return int(getattr(self.policy, "features_version", 1)) if self.policy is not None else self._version

    _version = 1

    def _setup_v2(self) -> None:
        from ..vec.backend import Backend  # noqa: PLC0415
        from ..vec.limits import LimitTables  # noqa: PLC0415
        from ..vec.spec import SpecBatch  # noqa: PLC0415
        from ..vec.steer import BatchFollower  # noqa: PLC0415
        from ..vec.tracks import TrackBank  # noqa: PLC0415

        f = self.ref.follower
        B = Backend("numpy")
        bank = TrackBank.from_path(f.path, f.kappa, self.closed).to(B)
        spec = SpecBatch.from_vehicle(B, self.veh, 1)
        self._v2 = {"B": B, "bank": bank, "tables": LimitTables(B, spec, teacher=False),
                    "desc": spec.descriptor(), "follow": BatchFollower(B, bank), "zero": np.zeros(1, np.int64)}

    def _features_v2(self, pose, v: float, idx: int, state: dict | None):
        from ..vec.features import features  # noqa: PLC0415

        if self._v2 is None:
            self._setup_v2()
        c = self._v2
        st = state or {}
        tid, i = c["zero"], np.array([idx], np.int64)
        x, y, yaw = (np.array([float(q)]) for q in pose)
        e_lat, e_head, _ = c["follow"].tracking_error(tid, i, x, y, yaw)
        arr = lambda k: np.array([float(st.get(k, 0.0))])  # noqa: E731
        f, env = features(c["B"], c["bank"], c["tables"], c["desc"], tid, i, np.array([float(v)]), arr("vy"), arr("r"),
                          arr("ax"), arr("ay"), e_lat, e_head, np.array([self.prev_a]))
        self._env0 = float(env[0, 0])
        return f[0]

    def _at(self, i: int, k: int) -> int:
        j = i + k
        return j % self.n if self.closed else min(j, self.n - 1)

    def features(self, pose, v: float, idx: int, state: dict | None = None) -> np.ndarray:
        if self.version == 2:
            return self._features_v2(pose, v, idx, state)
        f = self.ref.follower
        st = state or {}
        x, y, yaw = pose
        lim = [self.v_lim[self._at(idx, k)] / 30.0 for k in self._s_steps]
        kap = [f.kappa[self._at(idx, k)] * 5.0 for k in self._s_steps]
        vy, r = float(st.get("vy", 0.0)), float(st.get("r", 0.0))
        ax, ay = float(st.get("ax", 0.0)), float(st.get("ay", 0.0))
        beta = math.atan2(vy, max(abs(v), 1.0))
        # tracking error w.r.t. the path point and its tangent
        p0 = f.path[idx]
        p1 = f.path[self._at(idx, 4)] if self._at(idx, 4) != idx else f.path[max(idx - 4, 0)]
        tx, ty = p1[0] - p0[0], p1[1] - p0[1]
        tn = math.hypot(tx, ty) or 1.0
        tx, ty = tx / tn, ty / tn
        e_lat = (x - p0[0]) * -ty + (y - p0[1]) * tx
        e_head = math.atan2(math.sin(yaw - math.atan2(ty, tx)), math.cos(yaw - math.atan2(ty, tx)))
        out = [v / 30.0, *lim, *kap, vy / 3.0, r / 1.5, ax / 15.0, ay / 15.0, beta * 3.0, e_lat / 1.5, e_head * 2.0,
               (v - self.v_lim[idx]) / 5.0, self.prev_a]
        return np.clip(np.array(out, float), -5.0, 5.0)

    # ------------------------------------------------------------------ #
    LAUNCH_SPEED = 2.0     # [m/s] below this (while racing) the policy cannot brake: the car always pulls away
    LAUNCH_MIN = 0.5

    def act(self, feats: np.ndarray, v: float) -> float:
        """Policy action in [-1, 1] with the launch rule (standstill must not be a trap)."""
        a = self.policy.act(feats)
        if v < self.LAUNCH_SPEED:
            a = max(a, self.LAUNCH_MIN)
        return a

    def idle(self) -> None:
        """Call while not driving (AS Off / Ready / Finished): the previous action starts at 0."""
        self.prev_a = 0.0

    def update(self, pose, v, cones_world=None, cone_colors=None, stop=False, state=None):
        steer, accel_ref, info = self.ref.update(pose, v, cones_world, cone_colors, stop=stop)
        if stop:
            self.prev_a = 0.0
            return steer, accel_ref, {**info, "driver": "rl (stop)"}
        idx = self.ref.follower.idx
        a = self.act(self.features(pose, v, idx, state), v)
        self.prev_a = a
        accel = a * self.a_max if a >= 0.0 else a * self.d_max
        v_lim = float(self.v_lim[idx]) if self.v_lim is not None else getattr(self, "_env0", float("nan"))
        return steer, accel, {**info, "driver": "rl", "action": a, "v_limit": v_lim}
