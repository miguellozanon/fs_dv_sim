#!/usr/bin/env python3
"""Turn vehicle files of the Performance department's LionsLTS simulator into vehicle setups.

    python3 tools/import_lionslts_vehicle.py path/to/LR26_emrax.yml                  # print the changes
    python3 tools/import_lionslts_vehicle.py path/to/inputs/vehicles/*.yml --save    # save each as a configuration
    python3 tools/import_lionslts_vehicle.py LR25.yml --save --name "LR25 (LionsLTS)"

Saved configurations go to lapsim/configs/ (the launcher's and the dashboard's Vehicle setup list them).
The changes are relative to lr26_params.m; what could not be mapped is written into the configuration's
notes. Details: fsdsim/perf/lionslts.py.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ros2_ws", "src", "fsdsim"))

import yaml  # noqa: E402

from fsdsim.perf import lapsim as L  # noqa: E402
from fsdsim.perf.lionslts import to_changes  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="LionsLTS vehicle .yml files")
    ap.add_argument("--save", action="store_true", help="save as configurations in lapsim/configs")
    ap.add_argument("--name", help="configuration name (one file only; default: LionsLTS <file name>)")
    ap.add_argument("--lapsim", help="LR26_LapSim folder (baseline = its lr26_params.m)")
    ap.add_argument("--run", action="store_true", help="also print LapSim's points for each setup")
    args = ap.parse_args()
    if args.name and len(args.files) > 1:
        print("--name needs exactly one file", file=sys.stderr)
        return 2
    root = args.lapsim or L.find_lapsim_dir()
    if not root:
        print("LR26_LapSim folder not found (set FSG_LAPSIM_DIR or use --lapsim)", file=sys.stderr)
        return 2
    ls = L.LapSim.load(root)
    for path in args.files:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        changes, notes = to_changes(data, ls.veh)
        stem = os.path.splitext(os.path.basename(path))[0]
        name = args.name or f"LionsLTS {stem}"
        print(f"== {name}: {len(changes)} changes")
        print("   " + " ".join(changes))
        for n in notes:
            print(f"   note: {n}")
        if args.run:
            r = ls.run(changes)
            print("   " + "  ".join(f"{k} {e['T']:.2f} s ({e['points']:.1f})" for k, e in r["events"].items())
                  + f"  total {r['total']:.1f} points")
        if args.save:
            cfg_dir = os.path.join(os.path.dirname(root), "configs")
            os.makedirs(cfg_dir, exist_ok=True)
            out = os.path.join(cfg_dir, re.sub(r"\s+", "_", name.strip()) + ".json")
            with open(out, "w", encoding="utf-8") as f:
                json.dump({"name": name, "changes": changes, "notes": "From LionsLTS " + os.path.basename(path)
                           + ("; " + "; ".join(notes) if notes else ""), "saved": time.strftime("%Y-%m-%d %H:%M")}, f, indent=2)
            print(f"   saved {os.path.relpath(out, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
