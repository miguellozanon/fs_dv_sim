"""Track bank: the reference paths the batched cars drive (FSG events + generated autocross layouts).

Each track is what the reference driver follows in fsdsim (PathFollower: path resampled at 0.25 m,
curvature smoothed over +-2 m), padded into (T, P) arrays so N cars on different tracks step together.
Timing marks (metres of progress from the start position) reproduce the official timing:
    acceleration  front of the car crosses the start line -> the finish line at 75 m
    skidpad       timed laps 2 (right) and 4 (left): ((m2 - m1) + (m4 - m3)) / 2
    autocross     standing start behind the line, one lap line -> line
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import numpy as np

from .backend import Backend

DS = 0.25
# lookahead intervals [m] for the speed envelope (features v2)
LOOKAHEAD = (0.0, 2.5, 5.0, 7.5, 10.0, 14.0, 18.0, 23.0, 29.0, 36.0, 44.0, 52.0, 60.0)
N_INTERVALS = len(LOOKAHEAD) - 1
WINDOW = int(round(LOOKAHEAD[-1] / DS)) + 1
CONE_NEAR = 5.5           # [m] cones listed per path point (car within ~1.5 m of the path + 40 ms of travel)


def near_cones(path: np.ndarray, cones: np.ndarray) -> list[np.ndarray]:
    """For every path point: indices of the cones within CONE_NEAR."""
    if len(cones) == 0:
        return [np.zeros(0, np.int64)] * len(path)
    d2 = (path[:, None, 0] - cones[None, :, 0]) ** 2 + (path[:, None, 1] - cones[None, :, 1]) ** 2
    return [np.nonzero(row < CONE_NEAR ** 2)[0] for row in d2]


@dataclass
class TrackInfo:
    mission: str
    seed: int
    path: np.ndarray          # (n, 2)
    kappa: np.ndarray         # (n,) smoothed |curvature| as the reference driver uses it
    closed: bool
    width: float
    start_pose: tuple
    start_idx: int
    marks: list               # progress [m] from the start at which timing events happen
    cones: np.ndarray = None  # (n_c, 3): x, y, hit radius (fsdsim Cone.hit_radius)


def _marks(track, path, start_idx: int, closed: bool, front_bumper: float) -> list:
    n = len(path)
    if track.mission in ("acceleration", "ebs_test"):
        s0 = -track.start_pose[0]                       # start line at x = 0, car starts at x0 < 0
        return [s0 - front_bumper, s0 + track.meta["length"] - front_bumper]
    if track.mission == "skidpad":
        # path: entry (y < 0) -> laps right -> laps left -> exit; the timekeeping line is y = 0
        i_e = int(np.argmax(path[:, 1] >= 0.0))
        n_lap = int(round(2 * math.pi * 0.5 * (track.meta["r_in"] + track.meta["r_out"]) / DS))
        base = (i_e - start_idx) * DS - front_bumper
        return [base + k * n_lap * DS for k in range(5)]
    if closed:
        first = ((n - start_idx) % n) * DS - front_bumper
        return [first + k * n * DS for k in range(int(track.laps) + 1)]
    return []


def build_track_info(mission: str, seed: int, veh, events) -> TrackInfo:
    from ..control.demo_driver import DriverParams, PathFollower  # noqa: PLC0415
    from ..track import build_track  # noqa: PLC0415

    ev = copy.deepcopy(events)
    ev.track_generator.seed = seed
    track = build_track(mission, ev, veh, seed if mission in ("autocross", "trackdrive") else None)
    f = PathFollower(DriverParams(wheelbase=veh.wheelbase, lr=veh.lr, max_steer=veh.actuators.max_steer))
    f.set_path(track.centerline, track.closed, v_profile=np.zeros(len(track.centerline)), ds=DS)
    sx, sy, _ = track.start_pose
    start_idx = int(np.argmin(np.hypot(f.path[:, 0] - sx, f.path[:, 1] - sy)))
    cones = np.array([[c.x, c.y, c.hit_radius] for c in track.cones], float).reshape(-1, 3)
    return TrackInfo(mission, seed, f.path, f.kappa, track.closed, float(track.width), tuple(track.start_pose),
                     start_idx, _marks(track, f.path, start_idx, track.closed, veh.front_bumper), cones)


def interval_kmax(kappa: np.ndarray, closed: bool) -> np.ndarray:
    """(n, N_INTERVALS): max |curvature| in [i + LOOKAHEAD[j], i + LOOKAHEAD[j+1]] for every i."""
    n = len(kappa)
    k = np.abs(kappa)
    ext = np.concatenate([k, k[:WINDOW]]) if closed else np.concatenate([k, np.full(WINDOW, k[-1])])
    out = np.zeros((n, N_INTERVALS))
    for j in range(N_INTERVALS):
        a, b = int(round(LOOKAHEAD[j] / DS)), int(round(LOOKAHEAD[j + 1] / DS))
        win = np.lib.stride_tricks.sliding_window_view(ext, b - a + 1)      # win[i] = ext[i : i + b - a + 1]
        out[:, j] = win[a:a + n].max(axis=1)
    return out


class TrackBank:
    """All tracks as padded arrays on the backend's device."""

    def __init__(self, infos: list[TrackInfo]):
        self.infos = list(infos)

    # ------------------------------------------------------------------ #
    @classmethod
    def build(cls, config_dir: str, items, changes=()) -> "TrackBank":
        """items: [(mission, seed), ...]; layouts from fsdsim's generator (sim.yaml)."""
        from ..config import load_all_configs  # noqa: PLC0415

        veh, _, simcfg = load_all_configs(config_dir, vehicle_overrides=list(changes))
        return cls([build_track_info(m, int(s), veh, simcfg.events) for m, s in items])

    @classmethod
    def from_path(cls, path, kappa, closed: bool, width: float = 3.0) -> "TrackBank":
        """One reference path as the RL driver has it (fsdsim's PathFollower)."""
        p = np.asarray(path, float)
        return cls([TrackInfo("path", 0, p, np.asarray(kappa, float), bool(closed), float(width),
                              (float(p[0, 0]), float(p[0, 1]), 0.0), 0, [])])

    def save(self, file: str) -> None:
        d = {}
        for i, t in enumerate(self.infos):
            d[f"t{i}_path"], d[f"t{i}_kappa"] = t.path, t.kappa
            d[f"t{i}_meta"] = np.array([t.closed, t.width, *t.start_pose, t.start_idx, t.seed], float)
            d[f"t{i}_marks"] = np.array(t.marks, float)
            d[f"t{i}_mission"] = np.array(t.mission)
            d[f"t{i}_cones"] = t.cones if t.cones is not None else np.zeros((0, 3))
        np.savez_compressed(file, n=len(self.infos), **d)

    @classmethod
    def load(cls, file: str) -> "TrackBank":
        with np.load(file) as z:
            out = []
            for i in range(int(z["n"])):
                m = z[f"t{i}_meta"]
                out.append(TrackInfo(str(z[f"t{i}_mission"]), int(m[6]), z[f"t{i}_path"], z[f"t{i}_kappa"],
                                     bool(m[0]), float(m[1]), (float(m[2]), float(m[3]), float(m[4])), int(m[5]),
                                     list(z[f"t{i}_marks"]), z[f"t{i}_cones"] if f"t{i}_cones" in z else None))
        return cls(out)

    def missions(self) -> list[str]:
        return [t.mission for t in self.infos]

    # ------------------------------------------------------------------ #
    def to(self, B: Backend) -> "TrackBank":
        """Padded device arrays: px, py, kap (T, P), kmax (T, P, N_INTERVALS), n, closed, width, start."""
        T = len(self.infos)
        P = max(len(t.path) for t in self.infos)
        px, py, kap = np.zeros((T, P)), np.zeros((T, P)), np.zeros((T, P))
        kmax = np.zeros((T, P, N_INTERVALS))
        n = np.zeros(T, np.int64)
        for i, t in enumerate(self.infos):
            m = len(t.path)
            n[i] = m
            px[i, :m], py[i, :m], kap[i, :m] = t.path[:, 0], t.path[:, 1], t.kappa
            px[i, m:], py[i, m:], kap[i, m:] = t.path[-1, 0], t.path[-1, 1], t.kappa[-1]
            km = interval_kmax(t.kappa, t.closed)
            kmax[i, :m] = km
            kmax[i, m:] = km[-1]
        self.B = B
        self.T, self.P = T, P
        self.px, self.py, self.kap = B.asarray(px), B.asarray(py), B.asarray(kap)
        self.kmax = B.asarray(kmax.reshape(T, P * N_INTERVALS)).reshape(T, P, N_INTERVALS)
        self.n = B.asarray(n)
        self.closed = B.asarray(np.array([t.closed for t in self.infos]))
        self.width = B.asarray(np.array([t.width for t in self.infos]))
        self.start_idx = B.asarray(np.array([t.start_idx for t in self.infos], np.int64))
        self.start_pose = B.asarray(np.array([t.start_pose for t in self.infos], float))
        nm = max([len(t.marks) for t in self.infos] + [1])
        marks = np.full((T, nm), np.inf)
        for i, t in enumerate(self.infos):
            marks[i, :len(t.marks)] = t.marks
        self.marks = B.asarray(marks)
        self.n_marks = nm
        # cones (x, y, radius) padded, and per path point the nearby cones (-1: none)
        cs = [t.cones if t.cones is not None else np.zeros((0, 3)) for t in self.infos]
        C = max([len(c) for c in cs] + [1])
        cone = np.zeros((T, C, 3))
        cone[:, :, :2] = 1e6
        near = [near_cones(t.path, c) for t, c in zip(self.infos, cs)]
        M = max([len(r) for nr in near for r in nr] + [1])
        tab = np.full((T, P, M), -1, np.int64)
        for i, (c, nr) in enumerate(zip(cs, near)):
            cone[i, :len(c)] = c
            for j, r in enumerate(nr):
                tab[i, j, :len(r)] = r
            tab[i, len(nr):] = tab[i, len(nr) - 1]
        self.n_cones_max, self.cone_m = C, M
        self.cone = B.asarray(cone.reshape(T * C, 3)).reshape(T, C, 3)
        self.cone_near = B.asarray(tab.reshape(T, P * M)).reshape(T, P, M)
        return self

    def wrap(self, tid, idx):
        """Index along track tid (closed: modulo n; open: clamped to the end)."""
        B = self.B
        n = self.n[tid]
        if idx.ndim == 2:
            n = n[:, None]
            closed = self.closed[tid][:, None]
        else:
            closed = self.closed[tid]
        return B.where(closed, idx % n, B.clip(idx, 0, n - 1))
