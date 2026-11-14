#!/usr/bin/env python3
"""Stand-in for `matlab -batch "run('<job.m>')"` in tests (no MATLAB here).

Interprets the job scripts written by fsd_dashboard.matlab_runner: lts_set_param (absolute and
'scale'), veh.<f> = veh.<f> <op> <x>, comp.events.<e>.<f> = ..., then runs the Python port of
lts_run_all and writes the results table and the done file like the real job.
"""

import csv
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ros2_ws", "src", "fsdsim"))

from fsdsim.perf.lapsim import LapSim  # noqa: E402


def main() -> None:
    arg = sys.argv[sys.argv.index("-batch") + 1]
    job = re.search(r"run\('(.+)'\)", arg).group(1)
    text = open(job, encoding="utf-8").read()
    lapsim_dir = re.search(r"cd\('(.+?)'\)", text).group(1)
    csv_path = re.search(r"writetable\(tbl, '(.+?)'\)", text).group(1)
    done_path = re.search(r"fid = fopen\('(.+?)', 'w'\)", text).group(1)
    changes = []
    for line in text.splitlines():
        line = line.strip()
        m = re.match(r"veh = lts_set_param\(veh, '(\w+)', ([^,)]+)(, 'scale')?\);", line)
        if m:
            changes.append(f"{m.group(1)}{'*' if m.group(3) else '='}{m.group(2)}")
            continue
        for m in re.finditer(r"veh\.(\w+) = veh\.\1 ([+-]) ([^;]+);", line):
            changes.append(f"{m.group(1)}{m.group(2)}{m.group(3)}")
        m = re.match(r"comp\.events\.(\w+)\.(\w+) = (?:comp\.events\.\1\.\2 ([*+-]) )?([^;]+);", line)
        if m:
            changes.append(f"{m.group(1)}.{m.group(2)}{m.group(3) or '='}{m.group(4)}")
    if os.environ.get("FAKE_MATLAB_FAIL"):
        open(done_path, "w").write("error\nUndefined function 'lr26_params'.\n")
        return
    res = LapSim.load(lapsim_dir).run(changes)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Event", "Time_s", "Points", "Laps", "Lap_s", "Vavg_kmh", "Vmax_kmh", "Energy_kWh"])
        for n, r in res["events"].items():
            w.writerow([n, r["T"], r["points"], r["laps"], r["t_lap"], r["v_avg"] * 3.6, r["v_max"] * 3.6,
                        r["E_total"] / 3.6e6])
    open(done_path, "w").write(f"ok\n{res['total']:.6f}\n")
    print("fake MATLAB:", " ".join(changes) or "baseline")


if __name__ == "__main__":
    main()
