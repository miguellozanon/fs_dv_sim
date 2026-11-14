"""LapSim predictions for the dashboard (ROS-free): what the Performance model expects for the
event that is running, for the vehicle that is running (incl. run-time parameter changes)."""

from __future__ import annotations

from fsdsim.control import DriverParams
from fsdsim.perf import PerfParams, qss


def predict(perf: PerfParams, mission: str, centerline=None, closed: bool = False, laps: int = 1,
            grip_util: float | None = None, long_util: float | None = None) -> dict:
    """{'event', 'lapsim': {...}, 'dv_target': {...}} with 'time' (event time) and 'lap' (per lap, circuits)."""
    d = DriverParams()
    gu = d.grip_util if grip_util is None else grip_util
    lu = d.long_util if long_util is None else long_util
    dv = perf.copy(util_x=gu, util_y=gu, events={})
    dv_brake = dv.copy(util_x=gu * lu)
    ev_key = "endurance" if mission == "trackdrive" and "endurance" in (perf.events or {}) else mission
    out = {"event": mission, "grip_util": gu, "long_util": lu, "lapsim_util": perf.for_event(ev_key).util_y}
    if mission == "acceleration":
        out["lapsim"] = {"time": qss.acceleration(perf).time}
        out["dv_target"] = {"time": qss.acceleration(dv).time}
    elif mission == "skidpad":
        out["lapsim"] = {"time": qss.skidpad(perf).time}
        out["dv_target"] = {"time": qss.skidpad(dv).time}
    elif mission in ("autocross", "trackdrive") and centerline is not None and len(centerline) > 3:
        ev = "endurance" if mission == "trackdrive" and "endurance" in (perf.events or {}) else mission
        for key, p, pb in (("lapsim", perf, None), ("dv_target", dv, dv_brake)):
            r = qss.run_path(p, centerline, closed, event=ev, veh_brake=pb)
            first = r.lap_times[0]
            flying = r.lap_times[1] if len(r.lap_times) > 1 else first
            total = first + max(laps - 1, 0) * flying
            out[key] = {"time": total, "first_lap": first, "flying_lap": flying, "v_max": r.v_max * 3.6}
    return out
