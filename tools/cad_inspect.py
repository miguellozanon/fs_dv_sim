#!/usr/bin/env python3
"""Read STEP files of the car as text (no CAD kernel): solids, wheels, dimensions, and how they compare with the model.

    python3 tools/cad_inspect.py cad/lr26/*.stp                     # report
    python3 tools/cad_inspect.py cad/lr26/*.stp --png out/cad.png   # top / side layout of the parts
    python3 tools/cad_inspect.py cad/lr26/*.stp --json out/cad.json

The CAD frame is found from the wheels: the front axle is x = 0 and x points to the rear (NX default of the
LR26 files), y is lateral, z up, units mm. The simulator frame has x forward, y left, z up and the origin at
the centre of gravity on the ground; `fsdsim.cad.step_info` only reads, `tools/cad_to_usd.py` converts.
The wheels come from the toroidal tyre faces, so they need the monocoque + suspension assembly.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ros2_ws", "src", "fsdsim"))

import numpy as np  # noqa: E402

from fsdsim.cad import step_info as S  # noqa: E402


def model_values(config_dir: str | None = None) -> dict:
    """Geometry the simulator currently uses (performance mode: from lr26_params.m)."""
    from fsdsim.config import load_all_configs

    veh, sens, _ = load_all_configs(config_dir or os.path.join(ROOT, "ros2_ws", "src", "fsd_sim_bringup", "config"))
    return {"wheelbase": veh.wheelbase, "lf": veh.lf, "lr": veh.lr, "track_front": veh.track_front,
            "track_rear": veh.track_rear, "wheel_radius": veh.wheel_radius, "front_overhang": veh.footprint.front_overhang,
            "rear_overhang": veh.footprint.rear_overhang, "width": veh.footprint.width, "length": veh.footprint.length,
            "lidar": sens.lidar.mount, "camera": sens.camera.mount}


def analyse(paths: list[str]) -> dict:
    infos = [S.read(p) for p in paths]
    wheels = [w for i in infos for w in i.wheels]
    merged = S.StepInfo(",".join(paths), 1.0, [s for i in infos for s in i.solids], wheels)
    dims = S.vehicle_dimensions(merged)
    lo, hi = merged.bounds()
    out = {"files": {os.path.basename(i.path): {"solids": len(i.solids), "bounds_lo_m": [round(float(x), 4) for x in i.bounds()[0]],
                                               "bounds_hi_m": [round(float(x), 4) for x in i.bounds()[1]]}
                     for i in infos if i.solids},
           "dims": {k: round(float(v), 4) for k, v in (dims or {}).items()},
           "bounds_lo_m": [round(float(x), 4) for x in lo], "bounds_hi_m": [round(float(x), 4) for x in hi],
           "wheels": [{"centre_m": [round(float(x), 4) for x in w.centre], "outer_radius_m": round(w.outer_radius, 4)} for w in wheels]}
    return {"infos": infos, "merged": merged, "report": out}


def to_sim_frame(p, x_front_axle: float, lf: float):
    """CAD point(s) (x rearward, front axle at x_front_axle) -> simulator frame (x forward, CoG origin)."""
    p = np.asarray(p, float)
    return np.stack([lf - (p[..., 0] - x_front_axle), -p[..., 1], p[..., 2]], axis=-1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--png")
    ap.add_argument("--json")
    ap.add_argument("--config", help="fsd_sim_bringup/config folder")
    args = ap.parse_args()
    a = analyse(args.files)
    rep, merged = a["report"], a["merged"]
    dims = rep["dims"]
    m = model_values(args.config)
    print("files:")
    for n, f in rep["files"].items():
        print(f"  {n}: {f['solids']} solids, x {f['bounds_lo_m'][0]:+.3f}..{f['bounds_hi_m'][0]:+.3f} m, "
              f"y {f['bounds_lo_m'][1]:+.3f}..{f['bounds_hi_m'][1]:+.3f}, z {f['bounds_lo_m'][2]:+.3f}..{f['bounds_hi_m'][2]:+.3f}")
    if not dims:
        print("no four wheels found (give the monocoque + suspension assembly with the tyres); no frame check possible")
        return 0
    xf = dims["x_front_axle"]
    lo, hi = np.array(rep["bounds_lo_m"]), np.array(rep["bounds_hi_m"])
    # overall extents of the aero bodies (not the suspension rod ends): use the solids' boxes
    front_tip = xf - lo[0]
    rear_tip = hi[0] - dims["x_rear_axle"]
    width = max(hi[1], -lo[1]) * 2
    print("\ncompared with the simulator (vehicle.yaml / lr26_params.m):")
    rows = [("wheelbase [m]", dims["wheelbase"], m["wheelbase"]), ("track front [m]", dims["track_front"], m["track_front"]),
            ("track rear [m]", dims["track_rear"], m["track_rear"]),
            ("wheel centre height [m]", dims["wheel_radius_centre_height"], m["wheel_radius"]),
            ("front overhang [m] (axle -> wing tip)", front_tip, m["front_overhang"]),
            ("rear overhang [m] (axle -> wing)", rear_tip, m["rear_overhang"]),
            ("overall width [m]", width, m["width"]), ("overall length [m]", front_tip + dims["wheelbase"] + rear_tip, m["length"])]
    for name, cad, sim in rows:
        flag = "" if abs(cad - sim) < 0.01 else "   <-- differs"
        print(f"  {name:40s} CAD {cad:7.3f}   simulator {sim:7.3f}{flag}")
    print(f"  tyre: outer radius {dims['tyre_outer_radius']:.4f} m, width {dims['tyre_width']:.3f} m")
    lf = m["lf"]
    print(f"\nframe: CoG is {lf:.4f} m behind the front axle -> CAD x = {xf + lf:.4f} m; simulator x = {lf:.4f} - (x_cad - {xf:.3f}), y = -y_cad, z = z_cad")
    for nm, mt in (("LiDAR", m["lidar"]), ("ZED", m["camera"])):
        c_cad = np.array([xf + lf - mt.x, -mt.y, mt.z])
        near = [s.name for s in merged.solids if np.all(c_cad >= s.lo - 0.05) and np.all(c_cad <= s.hi + 0.05)]
        print(f"  {nm} mount at x_cad {c_cad[0]:+.3f}, z {c_cad[2]:.2f}: " + ("inside / within 5 cm of " + ", ".join(near[:4]) if near else "free space in the CAD"))
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({**rep, "x_front_axle_m": xf, "cog_x_cad_m": xf + lf}, f, indent=2)
        print(f"json: {args.json}")
    if args.png:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, Rectangle

        fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 8.5), gridspec_kw={"height_ratios": [1.25, 1]})
        cols = {}
        palette = ["#c0392b", "#2471a3", "#1e8449", "#b9770e", "#7d3c98"]
        for info in a["infos"]:
            c = cols.setdefault(os.path.basename(info.path), palette[len(cols) % len(palette)])
            for s in info.solids:
                p0 = to_sim_frame(s.lo, xf, lf)
                p1 = to_sim_frame(s.hi, xf, lf)
                lo_, hi_ = np.minimum(p0, p1), np.maximum(p0, p1)
                a1.add_patch(Rectangle((lo_[0], lo_[1]), hi_[0] - lo_[0], hi_[1] - lo_[1], fc=c, ec=c, alpha=0.18, lw=0.6))
                a2.add_patch(Rectangle((lo_[0], lo_[2]), hi_[0] - lo_[0], hi_[2] - lo_[2], fc=c, ec=c, alpha=0.18, lw=0.6))
        for w in merged.wheels:
            c = to_sim_frame(w.centre, xf, lf)
            a1.add_patch(Rectangle((c[0] - w.outer_radius, c[1] - 0.1), 2 * w.outer_radius, 0.2, fc="k", alpha=0.5))
            a2.add_patch(Circle((c[0], c[2]), w.outer_radius, fc="none", ec="k", lw=1.2))
        fo, ro, wd = m["front_overhang"], m["rear_overhang"], m["width"]
        a1.add_patch(Rectangle((-(m["lr"] + ro), -wd / 2), (m["lr"] + ro) + (lf + fo), wd, fc="none", ec="#555", ls="--", lw=1.0, label="footprint in the simulator (cone hits)"))
        a1.plot(0, 0, "r+", ms=14, mew=2, label="CoG")
        for nm, mt, mk in (("LiDAR", m["lidar"], "g^"), ("ZED", m["camera"], "bs")):
            a1.plot(mt.x, mt.y, mk, ms=8, label=nm + " mount")
            a2.plot(mt.x, mt.z, mk, ms=8)
        a1.set_title("top view, simulator frame (x forward)")
        a2.set_title("side view")
        for ax in (a1, a2):
            ax.set_aspect("equal")
            ax.grid(alpha=0.25)
            ax.set_xlim(-1.5, 2.1)
        a1.set_ylim(-0.9, 0.9)
        a2.set_ylim(-0.15, 1.3)
        a2.axhline(0, color="k", lw=0.5)
        a1.legend(loc="upper left", fontsize=8)
        fig.text(0.01, 0.005, "boxes = bounding boxes of the CAD solids (" + ", ".join(cols) + ")", fontsize=7)
        fig.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(args.png)), exist_ok=True)
        fig.savefig(args.png, dpi=110)
        print(f"png: {args.png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
