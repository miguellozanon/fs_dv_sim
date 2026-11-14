"""Autonomous System (AS) state machine, lap timing and penalties.

State and mission numbering match the FS Driverless Specification CAN
interface (DV system status 0x502), so values can be logged 1:1:

    AS_status:  Off=1, Ready=2, Driving=3, Emergency=4, Finished=5
    AMI_state:  Acceleration=1, Skidpad=2, Trackdrive=3, Brake test=4,
                Inspection=5, Autocross=6

Event flow (simplified from the rules):
    AS Off --select mission + activate--> AS Ready
    AS Ready --Go (after >= ready_min_time_s)--> AS Driving
    AS Driving --mission complete + standstill--> AS Finished
    any active state --RES / watchdog / left track--> AS Emergency (EBS)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from .config import ASConfig, EventsConfig, PenaltyConfig
from .geometry import segment_intersection
from .track import Track


class ASState(IntEnum):
    OFF = 1
    READY = 2
    DRIVING = 3
    EMERGENCY = 4
    FINISHED = 5


class Mission(IntEnum):
    NONE = 0
    ACCELERATION = 1
    SKIDPAD = 2
    TRACKDRIVE = 3
    EBS_TEST = 4
    INSPECTION = 5
    AUTOCROSS = 6


MISSION_BY_NAME = {
    "acceleration": Mission.ACCELERATION,
    "skidpad": Mission.SKIDPAD,
    "trackdrive": Mission.TRACKDRIVE,
    "ebs_test": Mission.EBS_TEST,
    "inspection": Mission.INSPECTION,
    "autocross": Mission.AUTOCROSS,
}
NAME_BY_MISSION = {v: k for k, v in MISSION_BY_NAME.items()}


@dataclass
class RunResult:
    mission: str
    state: str
    laps: int
    lap_times: list = field(default_factory=list)
    raw_time: float | None = None
    cones_hit: int = 0
    off_course: int = 0
    penalty_time: float = 0.0
    final_time: float | None = None
    message: str = ""
    extra: dict = field(default_factory=dict)


class MissionManager:
    def __init__(self, track: Track, as_cfg: ASConfig, penalties: PenaltyConfig, events: EventsConfig):
        self.track = track
        self.cfg = as_cfg
        self.penalties = penalties
        self.events = events
        self.reset()

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        self.state = ASState.OFF
        self.mission = Mission.NONE
        self.t_state = 0.0
        self.crossings: list[float] = []
        self.finish_time: float | None = None
        self.start_time: float | None = None
        self.mission_complete = False
        self.cones_hit = 0
        self.off_course = 0
        self._outside = False
        self._still_since: float | None = None
        self.message = "AS Off"
        self.extra: dict = {}
        self._progress_idx = 0

    @property
    def laps_target(self) -> int:
        return self.track.laps

    @property
    def lap_count(self) -> int:
        if self.mission in (Mission.ACCELERATION, Mission.EBS_TEST):
            return 1 if self.finish_time is not None else 0
        return max(0, len(self.crossings) - 1)

    @property
    def lap_times(self) -> list[float]:
        c = self.crossings
        return [float(c[i + 1] - c[i]) for i in range(len(c) - 1)]

    # ------------------------------------------------------------------ #
    # Commands (RES, ASMS, mission selection)
    # ------------------------------------------------------------------ #
    def select_mission(self, mission: Mission | int | str) -> tuple[bool, str]:
        if isinstance(mission, str):
            mission = MISSION_BY_NAME.get(mission.lower(), Mission.NONE)
        try:
            mission = Mission(int(mission))
        except (TypeError, ValueError):   # e.g. /as/select_mission with an unknown code: answer, don't crash the node
            return False, f"unknown mission {mission!r}"
        if self.state != ASState.OFF:
            return False, "mission can only be changed in AS Off"
        expected = MISSION_BY_NAME.get(self.track.mission)
        if mission != expected:
            return False, f"track is set up for '{self.track.mission}'; restart the sim with that mission"
        self.mission = mission
        self.message = f"mission {NAME_BY_MISSION[mission]} selected"
        return True, self.message

    def activate(self, t: float) -> tuple[bool, str]:
        if self.state != ASState.OFF:
            return False, f"cannot activate from {self.state.name}"
        if self.mission == Mission.NONE:
            return False, "select a mission first"
        self._set(ASState.READY, t, "AS Ready - waiting for Go")
        return True, self.message

    def go(self, t: float) -> tuple[bool, str]:
        if self.state != ASState.READY:
            return False, f"Go ignored in {self.state.name}"
        if t - self.t_state < self.cfg.ready_min_time_s:
            return False, f"Go ignored: AS Ready for {t - self.t_state:.1f} s < {self.cfg.ready_min_time_s} s"
        self._set(ASState.DRIVING, t, "AS Driving")
        return True, self.message

    def emergency(self, t: float, reason: str = "RES") -> tuple[bool, str]:
        if self.state in (ASState.READY, ASState.DRIVING):
            self._set(ASState.EMERGENCY, t, f"AS Emergency: {reason}")
            return True, self.message
        return False, f"emergency ignored in {self.state.name}"

    def _set(self, state: ASState, t: float, msg: str) -> None:
        self.state = state
        self.t_state = t
        self.message = msg

    # ------------------------------------------------------------------ #
    # Per-step update
    # ------------------------------------------------------------------ #
    def update(self, t: float, dt: float, prev_xy, xy, speed: float, prev_front=None, front=None) -> None:
        """prev_front / front: front-bumper points. Timing lines are photocells, triggered by the front of
        the car (acceleration starts 0.30 m behind the line); without them the CoG is used."""
        if self.state == ASState.DRIVING:
            if front is not None and prev_front is not None:
                self._timing(t, dt, prev_front, front)
            else:
                self._timing(t, dt, prev_xy, xy)
            self._track_limits(t, xy)
        # Standstill detection for Finished / EBS test completion
        if speed < self.cfg.standstill_speed:
            if self._still_since is None:
                self._still_since = t
        else:
            self._still_since = None
        still = self._still_since is not None and (t - self._still_since) >= self.cfg.standstill_time_s
        if self.state == ASState.DRIVING and self.mission_complete and still:
            self._set(ASState.FINISHED, t, "AS Finished")
            self.extra["stop_xy"] = [float(xy[0]), float(xy[1])]
            if self.track.stop_zone is not None:
                self.extra["stopped_in_zone"] = bool(_point_in_polygon(xy, self.track.stop_zone))
        if self.state == ASState.EMERGENCY and still and "ebs_stop_xy" not in self.extra:
            self.extra["ebs_stop_xy"] = [float(xy[0]), float(xy[1])]
            if "ebs_trigger_xy" in self.extra:
                p = self.extra["ebs_trigger_xy"]
                self.extra["ebs_stop_distance"] = float(math.hypot(xy[0] - p[0], xy[1] - p[1]))
            if self.mission == Mission.EBS_TEST:
                self.mission_complete = True

    def _crossed(self, line, prev_xy, xy):
        if line is None:
            return None
        hit = segment_intersection(prev_xy, xy, line.p1, line.p2)
        if hit is None:
            return None
        motion = np.asarray(xy, float) - np.asarray(prev_xy, float)
        if float(np.dot(motion, line.forward)) <= 0.0:
            return None
        return hit[0]

    def _timing(self, t, dt, prev_xy, xy):
        tr = self.track
        s = self._crossed(tr.timing_line, prev_xy, xy)
        if s is not None:
            tc = float(t - dt + s * dt)
            if self.mission in (Mission.ACCELERATION, Mission.EBS_TEST):
                if self.start_time is None:
                    self.start_time = tc
                    self.crossings = [tc]
            else:
                self.crossings.append(tc)
                if self.start_time is None:
                    self.start_time = tc
                if self.lap_count >= tr.laps and not self.mission_complete:
                    self.mission_complete = True
                    self.finish_time = tc
                    self.message = "mission complete - stopping"
        if self.mission in (Mission.ACCELERATION, Mission.EBS_TEST) and self.start_time is not None:
            f = self._crossed(tr.finish_line, prev_xy, xy)
            if f is not None and self.finish_time is None:
                self.finish_time = float(t - dt + f * dt)
                self.crossings.append(self.finish_time)
                if self.mission == Mission.ACCELERATION:
                    self.mission_complete = True
                    self.message = "finish line crossed - stopping"
        if self.mission == Mission.EBS_TEST and self.start_time is not None and "ebs_trigger_xy" not in self.extra:
            p0 = tr.timing_line.p1 * 0.5 + tr.timing_line.p2 * 0.5
            if np.dot(np.asarray(xy) - p0, tr.timing_line.forward) >= self.events.ebs_test_trigger_distance:
                self.extra["ebs_trigger_xy"] = [float(xy[0]), float(xy[1])]
                self.emergency(t, "EBS test - RES pressed by official")

    def _track_limits(self, t, xy):
        cl = self.track.centerline
        n = len(cl)
        # windowed nearest point search keeps overlapping layouts (skidpad) unambiguous
        lo = max(0, self._progress_idx - 40)
        hi = min(n, self._progress_idx + 200)
        if self.track.closed:
            idx = (np.arange(self._progress_idx - 40, self._progress_idx + 200)) % n
        else:
            idx = np.arange(lo, hi)
        d = np.hypot(cl[idx, 0] - xy[0], cl[idx, 1] - xy[1])
        j = int(np.argmin(d))
        self._progress_idx = int(idx[j])
        dist = float(d[j])
        half = self.track.width / 2
        outside = dist > half + 0.5 * 1.45  # all four wheels beyond the border line (approx.)
        if outside and not self._outside:
            self.off_course += 1
        self._outside = outside
        if dist > half + self.cfg.leave_track_distance:
            self.emergency(t, "vehicle left the track")

    def on_cone_hit(self) -> None:
        self.cones_hit += 1

    # ------------------------------------------------------------------ #
    def penalty_time(self) -> float:
        return self.cones_hit * self.penalties.cone_down_s + self.off_course * self.penalties.off_course_s

    def mission_time(self, t: float) -> float:
        if self.start_time is None:
            return 0.0
        end = self.finish_time if self.finish_time is not None else t
        return max(0.0, end - self.start_time)

    def result(self, t: float) -> RunResult:
        laps = self.lap_times
        raw = None
        extra = dict(self.extra)
        if self.mission in (Mission.ACCELERATION,):
            raw = (self.finish_time - self.start_time) if self.finish_time and self.start_time else None
        elif self.mission == Mission.SKIDPAD:
            if len(laps) >= 4:
                raw = 0.5 * (laps[1] + laps[3])  # timed laps: 2nd right, 2nd left
                extra["timed_laps"] = [laps[1], laps[3]]
        elif self.mission in (Mission.AUTOCROSS, Mission.TRACKDRIVE):
            if self.mission_complete:
                raw = sum(laps[: self.track.laps])
        pen = self.penalty_time()
        return RunResult(
            mission=NAME_BY_MISSION.get(self.mission, "none"),
            state=self.state.name,
            laps=self.lap_count,
            lap_times=laps,
            raw_time=raw,
            cones_hit=self.cones_hit,
            off_course=self.off_course,
            penalty_time=pen,
            final_time=(raw + pen) if raw is not None else None,
            message=self.message,
            extra=extra,
        )

    def status(self, t: float) -> dict:
        laps = self.lap_times
        return {
            "state": int(self.state),
            "mission": int(self.mission),
            "lap_count": self.lap_count,
            "laps_target": self.laps_target,
            "mission_complete": self.mission_complete,
            "cones_hit": self.cones_hit,
            "off_course": self.off_course,
            "mission_time": self.mission_time(t),
            "last_lap_time": laps[-1] if laps else 0.0,
            "best_lap_time": min(laps) if laps else 0.0,
            "penalty_time": self.penalty_time(),
            "message": self.message,
        }


def _point_in_polygon(p, poly) -> bool:
    x, y = p
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xin = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < xin:
                inside = not inside
    return inside
