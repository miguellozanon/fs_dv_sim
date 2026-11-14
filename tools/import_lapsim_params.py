#!/usr/bin/env python3
"""Write the fallback copy of the LR26 LapSim (MATLAB) parameters (performance_lr26.yaml).

The simulator reads lapsim/LR26_LapSim/config/lr26_params.m directly (vehicle.yaml:
performance_source: lapsim), so you normally do not need this tool. It refreshes the YAML copy
that is used when the MATLAB folder is not available, or imports another LapSim folder:

    python3 tools/import_lapsim_params.py                     # lapsim/LR26_LapSim of this repo
    python3 tools/import_lapsim_params.py /mnt/c/Users/<you>/.../LR26_LapSim
    python3 tools/import_lapsim_params.py path/to/lr26_params.m --competition path/to/competition_fsa2025.m

Reads `veh.<field> = <expr>;` from config/lr26_params.m and the per-event driver
utilisation (util_x / util_y) from config/competition_*.m, writes
ros2_ws/src/fsd_sim_bringup/config/performance_lr26.yaml, and prints the QSS times of
the imported car next to the LapSim baseline (acceleration 4.662 s, skidpad 5.154 s) so
you can see straight away whether the import and the port agree.
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ros2_ws", "src", "fsdsim"))

import yaml  # noqa: E402

from fsdsim.perf import PerfParams, qss  # noqa: E402
from fsdsim.perf.params import _eval, matlab_statements  # noqa: E402

DEFAULT_OUT = os.path.join(ROOT, "ros2_ws", "src", "fsd_sim_bringup", "config", "performance_lr26.yaml")
EVENT_KEYS = {"acceleration": ("accel", "acceleration"), "skidpad": ("skid", "skidpad"),
              "autocross": ("autocross", "ax"), "endurance": ("endurance", "endu", "end")}
LAPSIM_BASELINE = {"acceleration": 4.662, "skidpad": 5.154}


def find_files(arg: str, competition: str | None):
    if os.path.isdir(arg):
        params = (glob.glob(os.path.join(arg, "config", "lr26_params.m")) or
                  glob.glob(os.path.join(arg, "**", "lr26_params.m"), recursive=True))
        if not params:
            sys.exit(f"no lr26_params.m under {arg}")
        comp = competition or next(iter(sorted(glob.glob(os.path.join(arg, "config", "competition_*.m")))), None)
        return os.path.normpath(params[0]), comp and os.path.normpath(comp)
    return arg, competition


def parse_assignments(path: str) -> dict[str, float]:
    """All `a.b.c = <numeric expr>;` (and `name = expr;`) lines -> {'a.b.c': value}."""
    out, env = {}, {}
    for code in matlab_statements(open(path, encoding="utf-8", errors="replace").read()):
        m = re.match(r"^([A-Za-z_][\w.]*)\s*=\s*(.+?)\s*$", code)
        if not m:
            continue
        name, expr = m.groups()
        expr = expr.strip()
        if expr.startswith("[") and expr.endswith("]"):          # [ux uy] pairs
            parts = [p for p in re.split(r"[\s,;]+", expr[1:-1]) if p]
            try:
                vals = [_eval(p, env) for p in parts]
            except Exception:  # noqa: BLE001
                continue
            for i, v in enumerate(vals):
                out[f"{name}[{i}]"] = v
            continue
        try:
            v = _eval(expr, env)
        except Exception:  # noqa: BLE001
            continue
        out[name] = v
        env[name.split(".")[-1]] = v
    return out


def struct_event_utils(path: str) -> dict[str, dict]:
    """Layout of LR26_LapSim's competition_fsa2025.m: one struct is filled and then stored,

        e.util_x = 0.9;  e.util_y = 0.8;
        comp.events.autocross = e;
    """
    found: dict[str, dict] = {}
    fields: dict[str, dict] = {}
    for code in matlab_statements(open(path, encoding="utf-8", errors="replace").read()):
        m = re.match(r"^([A-Za-z_]\w*)\.(util_[xy])\s*=\s*(.+?)\s*$", code)
        if m:
            try:
                fields.setdefault(m.group(1), {})[m.group(2)] = _eval(m.group(3), {})
            except Exception:  # noqa: BLE001
                pass
            continue
        m = re.match(r"^[A-Za-z_][\w.]*\.(\w+)\s*=\s*([A-Za-z_]\w*)\s*$", code)
        if m and m.group(2) in fields:
            ev = next((e for e, toks in EVENT_KEYS.items() if m.group(1).lower() in toks), None)
            if ev:
                found[ev] = dict(fields[m.group(2)])
            continue
        m = re.match(r"^([A-Za-z_]\w*)\s*=\s*(struct\(\)|\w+\()", code)    # e = struct() / e = new_event(...)
        if m and m.group(1) in fields:
            fields[m.group(1)] = {}
    return found


def event_utils(assign: dict[str, float]) -> dict[str, dict]:
    """Find util_x / util_y per event in the competition file, whatever the struct layout."""
    found: dict[str, dict] = {}
    for key, val in assign.items():
        low = key.lower()
        if "util" not in low:
            continue
        for ev, tokens in EVENT_KEYS.items():
            parts = re.split(r"[._\[\]]+", low)
            if not any(t in parts for t in tokens):
                continue
            d = found.setdefault(ev, {})
            if low.endswith("[0]") or "util_x" in low or low.endswith("ux"):
                d["util_x"] = val
            elif low.endswith("[1]") or "util_y" in low or low.endswith("uy"):
                d["util_y"] = val
            else:
                d["util_x"] = d["util_y"] = val
    return found


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", nargs="?", default=os.path.join(ROOT, "lapsim", "LR26_LapSim"),
                    help="LR26_LapSim folder or lr26_params.m (default: lapsim/LR26_LapSim)")
    ap.add_argument("--competition", default=None, help="competition_*.m (default: config/competition_*.m)")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    params_m, comp_m = find_files(args.source, args.competition)
    base = PerfParams.from_yaml(args.out) if os.path.exists(args.out) else PerfParams()
    p, rep = PerfParams.from_matlab(params_m, base=base)
    print(f"{params_m}: imported {len(rep['imported'])} fields")
    for k, v in sorted(rep["imported"].items()):
        old = getattr(base, k)
        same = (math.isnan(old) and math.isnan(v)) if isinstance(old, float) and math.isnan(old) else abs(old - v) < 1e-12
        print(f"  {k:18s} {v:<14.6g}{'' if same else f'(was {old:.6g})'}")
    if rep["unknown"]:
        print("  not used by the vehicle model:", ", ".join(f"{k}={v:g}" for k, v in sorted(rep["unknown"].items())))
    if rep["failed"]:
        print("  could not evaluate (check by hand):")
        for k, v in rep["failed"].items():
            print(f"    {k}: {v}")
    missing = sorted(set(p.numeric_fields()) - set(rep["imported"]) - {"util_x", "util_y"})
    if missing:
        print("  kept from the previous file (not in lr26_params.m):", ", ".join(missing))

    events = dict(base.events or {})
    if comp_m and os.path.exists(comp_m):
        utils = struct_event_utils(comp_m) or event_utils(parse_assignments(comp_m))
        print(f"{comp_m}: driver utilisation per event: {utils or 'none found'}")
        events.update(utils)
    p = p.copy(events=events, source=f"imported from {os.path.basename(params_m)}"
                                     + (f" + {os.path.basename(comp_m)}" if comp_m else ""))

    a, s = qss.acceleration(p).time, qss.skidpad(p).time
    print("\nQSS check with this port - must equal the Time_s column of MATLAB's run_lapsim:")
    print(f"  acceleration {a:.3f} s   (unchanged LR26: {LAPSIM_BASELINE['acceleration']:.3f} s)")
    print(f"  skidpad      {s:.3f} s   (unchanged LR26: {LAPSIM_BASELINE['skidpad']:.3f} s)")

    if args.dry_run:
        return
    d = p.to_dict()
    header = ("# LR26 Performance parameters (LapSim names), used by fsdsim.perf and by vehicle.yaml\n"
              f"# ({d['source']}). Regenerate with tools/import_lapsim_params.py.\n")
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(header)
        yaml.safe_dump(d, f, sort_keys=False, default_flow_style=None)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
