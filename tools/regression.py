#!/usr/bin/env python3
"""Regression run of the whole closed loop: many events, circuits, driver rates, models and perception modes.

    python3 tools/regression.py                 # the standard set, 37 cases (about 3 minutes on 2 cores)
    python3 tools/regression.py --quick         # 12 cases, about 40 s
    python3 tools/regression.py --seeds 1 2 3 4 5 6 --jobs 8 --out results/regression_mine

Every case runs tools/closed_loop_eval.py's `run` in its own process and must finish its mission; cases marked
"clean" must also hit no cone and stay on the track. Output: a table on the screen, report.md and results.json
in --out (default results/regression_<time>), exit code 1 if a case fails. Use it before and after a change to
the vehicle model, the driver or the perception.

Groups: events (reference driver, ground truth, 50 and 20 Hz), circuits (map mode: centre and racing line on
several generated circuits), double-track handling, perception (LiDAR / fusion / camera with SLAM, explore then
race), sensor faults (must still finish; cones are reported, not required to be zero).
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, os.path.join(ROOT, "ros2_ws", "src", "fsdsim")]
CONFIG = os.path.join(ROOT, "ros2_ws", "src", "fsd_sim_bringup", "config")


def cases(seeds, quick: bool) -> list[dict]:
    c = []

    def add(group, name, mission, clean=True, **kw):
        c.append({"group": group, "name": name, "mission": mission, "clean": clean, "kw": kw})

    for rate in ((50,) if quick else (50, 20)):
        for m in ("acceleration", "skidpad", "autocross") + (() if quick else ("trackdrive",)):
            add("events", f"{m} @ {rate} Hz", m, dt=1.0 / rate, **({"laps": 3} if m == "trackdrive" else {}))
    add("events", "ebs_test", "ebs_test", expect="EMERGENCY")
    for s in (seeds[:2] if quick else seeds):
        add("circuits", f"seed {s} centre line", "autocross", mode="map", seed=s)
        add("circuits", f"seed {s} racing line", "autocross", mode="map", seed=s, line="racing")
        if not quick:
            add("circuits", f"seed {s} racing line @ 20 Hz", "autocross", mode="map", seed=s, line="racing", dt=0.05)
    for m in ("skidpad", "autocross"):
        add("double track", m, m, changes=["chassis.handling=double_track"], mode="map")
    add("perception", "fusion + SLAM, autocross", "autocross", perception="fusion", slam=True)
    add("perception", "skidpad on the known path, pose from SLAM (LiDAR)", "skidpad", perception="lidar", slam=True)
    if not quick:
        add("perception", "LiDAR + SLAM, autocross", "autocross", perception="lidar", slam=True)
        add("perception", "camera + SLAM, autocross", "autocross", perception="camera", slam=True)
        add("perception", "fusion, ground-truth pose", "autocross", perception="fusion")
        add("perception", "explore then race (LiDAR, SLAM, racing line)", "trackdrive", perception="lidar", slam=True,
            race_after=1, line="racing", laps=3)
        add("faults", "LiDAR dropout 1.5 s", "autocross", clean=False, perception="fusion", slam=True, faults="lidar:dropout@14+1.5")
        add("faults", "70 % of the LiDAR points lost", "autocross", clean=False, perception="fusion", slam=True, faults="lidar:thin=0.3@0")
        add("faults", "gyro bias 0.02 rad/s", "autocross", clean=False, perception="fusion", slam=True, faults="imu:bias=0.02@10")
        add("faults", "camera 0.2 s late", "autocross", clean=False, perception="fusion", slam=True, faults="camera:latency=0.2@0")
        # found on the team PC, 11 Nov 2026: these must stay clean
        add("faults", "camera 0.4 s late + 70 % LiDAR points lost", "autocross", perception="fusion", slam=True,
            faults="camera:latency=0.4@5,lidar:thin=0.3@0")
        add("faults", "LiDAR frozen 2 s", "autocross", perception="fusion", slam=True, faults="lidar:freeze@15+2")
        add("faults", "camera frozen 3 s", "autocross", perception="fusion", slam=True, faults="camera:freeze@10+3")
        add("circuits", "seed 13, cones mode (finish line)", "autocross", seed=13)
    return c


def run_case(case: dict) -> dict:
    import closed_loop_eval as C

    kw = dict(case["kw"])
    expect = kw.pop("expect", "FINISHED")
    t0 = time.time()
    try:
        sim, res, log, _ = C.run(case["mission"], CONFIG, verbose=False, **kw)
    except Exception as e:  # noqa: BLE001
        return {**case, "ok": False, "state": f"{type(e).__name__}: {e}", "wall": time.time() - t0}
    out = {**case, "state": res.state, "time": res.raw_time, "laps": [round(x, 3) for x in res.lap_times],
           "cones": res.cones_hit, "off_course": res.off_course, "wall": time.time() - t0,
           "energy_kwh": sim.vehicle.energy_j / 3.6e6}
    pipe = getattr(sim, "pipeline", None)
    if pipe is not None:
        acc = pipe.accuracy([(c.x, c.y, int(c.color)) for c in sim.track.cones if not c.knocked])
        out["map"] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in acc.items()}
        log = getattr(sim, "slam_log", None)
        if kw.get("slam") and log:
            err = [row[1] for row in log]
            out["pose_err"] = round(sum(err) / len(err), 3)          # mean over the run [m]
            out["pose_err_max"] = round(max(err), 3)
    out["ok"] = res.state == expect and (not case["clean"] or (res.cones_hit == 0 and res.off_course == 0))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="*", default=[7, 11, 23, 42])
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default="")
    ap.add_argument("--only", default="", help="run only the groups whose name contains this text")
    args = ap.parse_args()
    todo = [c for c in cases(args.seeds, args.quick) if args.only.lower() in c["group"]]
    out_dir = args.out or os.path.join(ROOT, "results", "regression_" + time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    print(f"{len(todo)} cases, {args.jobs} processes -> {out_dir}", flush=True)
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(args.jobs) as ex:
        for r in ex.map(run_case, todo):
            results.append(r)
            extra = ""
            if "map" in r:
                m = r["map"]
                extra = f"  map {m.get('matched', 0)}/{m.get('truth', 0)} rms {m.get('pos_rms', float('nan')):.2f} m"
                extra += f", pose {r['pose_err']:.2f} m (max {r['pose_err_max']:.2f})" if "pose_err" in r else ""
            t = "   --  " if r.get("time") is None else f"{r['time']:7.3f}"
            print(f"{'ok  ' if r['ok'] else 'FAIL'} {r['group']:13s} {r['name']:46s} {r['state'][:22]:10s} {t} s "
                  f"cones {r.get('cones', '-')} oc {r.get('off_course', '-')}{extra}", flush=True)
    bad = [r for r in results if not r["ok"]]
    lines = [f"# Regression run {time.strftime('%Y-%m-%d %H:%M')}", "",
             f"{len(results)} cases, {len(bad)} failed, {time.time() - t0:.0f} s.", "",
             "| | group | case | state | time [s] | cones | off course | map (matched / truth, rms) | pose error mean / max [m] |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        m = r.get("map")
        ms = f"{m.get('matched', 0)} / {m.get('truth', 0)}, {m.get('pos_rms', float('nan')):.2f} m" if m else ""
        t = "" if r.get("time") is None else f"{r['time']:.3f}"
        pe = f"{r['pose_err']:.2f} / {r['pose_err_max']:.2f}" if "pose_err" in r else ""
        lines.append(f"| {'ok' if r['ok'] else '**FAIL**'} | {r['group']} | {r['name']} | {r['state']} | {t} | {r.get('cones', '')} | "
                     f"{r.get('off_course', '')} | {ms} | {pe} |")
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1, default=str)
    print(f"\n{len(results) - len(bad)} of {len(results)} ok in {time.time() - t0:.0f} s; report: {os.path.join(out_dir, 'report.md')}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
