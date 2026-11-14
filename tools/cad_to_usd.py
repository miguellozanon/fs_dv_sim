#!/usr/bin/env python3
"""Car CAD (STL / OBJ / STEP) -> USD files for the Isaac Sim backend: one body and four wheels.

    python3 tools/cad_to_usd.py cad/lr26/*.stp                       # STEP: needs cadquery (python3 -m pip install cadquery)
    python3 tools/cad_to_usd.py cad/lr26/export/*.stl --set-yaml     # STL / OBJ exported from NX: no extra package
    python3 tools/cad_to_usd.py ... --out assets/lr26 --name lr26

Writes <out>/<name>_body.usda and <out>/<name>_wheel_<fl|fr|rl|rr>.usda (wheels centred on their hub, so the
simulator can steer and spin them), and <name>_cad.json. `--set-yaml` puts the paths into vehicle.yaml
(`usd_path`, `usd_wheel_path`). Restart the simulation (the launcher: Start simulation).

Frame: the defaults are the LR26 CAD (millimetres, front axle at x = 0, x to the rear, z up); the simulator
frame is x forward, y left, origin at the centre of gravity on the ground. Change with --unit-scale,
--front-axle-x, --x-forward. Wheels are placed from the simulator's own geometry (lr26_params.m), so they stay
where the dynamic model has them; the tyre envelope is --tyre-radius / --tyre-width (LR26: 0.2565 m, 0.2007 m).
Everything that lies completely inside a wheel's envelope is the wheel; if the whole car is one mesh,
the triangles between the hub radius and the tyre are split off instead.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ros2_ws", "src", "fsdsim"))

import numpy as np  # noqa: E402

from fsdsim.cad import meshes as M  # noqa: E402

OS1_RADIUS, OS1_HALF_HEIGHT = 0.0435, 0.0368   # Ouster OS1: 87 mm diameter, 73.5 mm high [m]
CARBON, RED, TYRE, GREY = (0.06, 0.06, 0.07), (0.75, 0.05, 0.05), (0.02, 0.02, 0.02), (0.55, 0.55, 0.58)


def colour_for(name: str):
    n = name.lower()
    if re.search(r"(^|_)(fw|rw)(_|$)|wing", n):
        return CARBON
    if re.search(r"(^|_)sp(_|$)|side|pod|mono|nose", n):
        return RED
    return None


def is_rod(v) -> bool:
    """A long part with a small, round section (wishbone, push / pull or tie rod). Flat ribbons (nose, wings) are not.
    The same rule as isaac/scene.py Scene.classify_part."""
    v = np.asarray(v, float)
    if len(v) < 3:
        return False
    q = v - v.mean(0)
    ext = np.ptp(q @ np.linalg.svd(q, full_matrices=False)[2].T, axis=0)
    return bool(ext[0] > 0.2 and 0.005 < ext[1] < 0.03 and ext[2] > 0.5 * ext[1])


def paint(part, centre_x_wing=(1.15, -0.35)):
    """Colours for a model without any: wings carbon, big body parts red, the rest metal. Simulator frame."""
    if is_rod(part.v):
        return GREY          # suspension tubes are longer than 0.3 m: they came out red before 13 Nov 2026
    lo, hi = part.bounds
    size = float(np.linalg.norm(hi - lo))
    cx, cz = 0.5 * (lo[0] + hi[0]), 0.5 * (lo[2] + hi[2])
    if (cx > centre_x_wing[0] and hi[2] < 0.35) or (cx < centre_x_wing[1] and cz > 0.45):
        return CARBON
    return RED if size > 0.30 else GREY


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="CAD files (globs allowed): .stp .step .stl .obj")
    ap.add_argument("--out", default=os.path.join(ROOT, "assets", "lr26"))
    ap.add_argument("--name", default="lr26")
    ap.add_argument("--unit-scale", type=float, default=1e-3, help="CAD unit in metres (mm: 0.001)")
    ap.add_argument("--front-axle-x", type=float, default=0.0, help="x of the front axle in the CAD [CAD units]")
    ap.add_argument("--x-forward", action="store_true", help="the CAD x axis points to the front (default: to the rear)")
    ap.add_argument("--tyre-radius", type=float, default=0.2565)
    ap.add_argument("--tyre-width", type=float, default=0.2007)
    ap.add_argument("--tolerance", type=float, default=0.3, help="STEP tessellation tolerance [mm]")
    ap.add_argument("--max-part-tris", type=int, default=12000, help="simplify parts with more triangles than this (springs, threads); 0 = never")
    ap.add_argument("--paint", choices=["auto", "none"], default="auto",
                    help="auto: colours by position and size where the file has none (wings carbon, large parts red, the rest grey)")
    ap.add_argument("--filler", action="store_true", help="the CAD has no monocoque: add the simulator's simple tub (usd_fill_body)")
    ap.add_argument("--hoop", action="store_true", help="the CAD has no main hoop: add the simulator's, under the camera (usd_fill_hoop)")
    ap.add_argument("--move-lidar", action="store_true",
                    help="if the model blocks the LiDAR mount, raise the mount (sensors.yaml) to just above the model there")
    ap.add_argument("--no-normals", action="store_true", help="smaller files; the renderer computes the normals")
    ap.add_argument("--config", help="fsd_sim_bringup/config folder (geometry and sensor mounts)")
    ap.add_argument("--set-yaml", action="store_true", help="write usd_path / usd_wheel_path into vehicle.yaml")
    args = ap.parse_args()

    from fsdsim.config import load_all_configs

    cfg = args.config or os.path.join(ROOT, "ros2_ws", "src", "fsd_sim_bringup", "config")
    veh, sens, _ = load_all_configs(cfg)
    files = [f for pat in args.files for f in (sorted(glob.glob(pat)) or [pat])]
    parts = []
    for f in files:
        got = M.read_any(f, tolerance_mm=args.tolerance) if f.lower().endswith((".stp", ".step")) else M.read_any(f)
        for p in got:
            if p.colour is None:
                p.colour = colour_for(os.path.basename(f) + "_" + p.name)      # None when the name says nothing
        print(f"{os.path.basename(f)}: {len(got)} part(s), {sum(len(p.f) for p in got)} triangles")
        parts += got
    if not parts:
        print("no geometry read", file=sys.stderr)
        return 2
    frame = M.Frame(args.unit_scale, args.front_axle_x, veh.lf, not args.x_forward)
    parts = M.to_sim(parts, frame)
    lo = np.min([p.v.min(axis=0) for p in parts], axis=0)
    hi = np.max([p.v.max(axis=0) for p in parts], axis=0)
    print(f"model in the simulator frame: x {lo[0]:+.3f}..{hi[0]:+.3f}, y {lo[1]:+.3f}..{hi[1]:+.3f}, z {lo[2]:+.3f}..{hi[2]:+.3f} m")
    if lo[2] < -0.02 or lo[2] > 0.1:
        print("  warning: the lowest point is not near the ground; check --unit-scale / the CAD z origin")
    if args.max_part_tris > 0:
        before = sum(len(p.f) for p in parts)
        parts = [M.simplify(p, args.max_part_tris) for p in parts]
        print(f"simplified parts above {args.max_part_tris} triangles: {before} -> {sum(len(p.f) for p in parts)} triangles")
    wheels = M.default_wheels(veh.lf, veh.lr, veh.track_front, veh.track_rear, args.tyre_radius, args.tyre_width, veh.wheel_radius)
    body, wparts = M.split_wheels(parts, wheels)
    if args.paint == "auto":
        for p in body:
            if p.colour is None:
                p.colour = paint(p)
    missing = [k for k, v in wparts.items() if not v]
    if missing:
        print(f"  warning: no tyre geometry found for {', '.join(missing)}: those wheels stay the simulator's cylinders "
              "(give the assembly that contains the wheels, or check the frame)")
    os.makedirs(args.out, exist_ok=True)
    rep = {"frame": {"unit_scale": args.unit_scale, "front_axle_x": args.front_axle_x, "x_rearward": not args.x_forward, "lf": veh.lf},
           "files": [os.path.basename(f) for f in files], "bounds_lo": [round(float(x), 4) for x in lo],
           "bounds_hi": [round(float(x), 4) for x in hi], "outputs": {}}
    doc = f"{args.name}: converted from {', '.join(os.path.basename(f) for f in files)} by tools/cad_to_usd.py"
    b = M.write_usda(os.path.join(args.out, f"{args.name}_body.usda"), "Body", body, normals=not args.no_normals, comment=doc)
    rep["outputs"]["body"] = b
    print(f"body: {b['parts']} parts, {b['triangles']} triangles")
    for k, ps in wparts.items():
        if ps:
            r = M.write_usda(os.path.join(args.out, f"{args.name}_wheel_{k}.usda"), "Wheel", [p.copy(colour=TYRE) for p in ps],   # all four alike (a CAD colour made one wheel grey)
                             default_colour=TYRE, normals=not args.no_normals, comment=doc)
            rep["outputs"][f"wheel_{k}"] = r
            print(f"wheel {k}: {r['parts']} parts, {r['triangles']} triangles")
    mounts = {"lidar": (sens.lidar.mount.x, sens.lidar.mount.y, sens.lidar.mount.z),
              "zed": (sens.camera.mount.x, sens.camera.mount.y, sens.camera.mount.z)}
    near = M.clearance(body, mounts, 0.12)
    rep["sensor_vertices_within_12cm"] = near
    # the OS1 itself (87 mm wide, 73.5 mm high, origin at its centre) + 5 mm: a sensor sitting on top of the
    # bodywork has model vertices within 12 cm but none inside it, and must not be reported (or moved) again
    in_os1 = M.inside_cylinder(body, mounts["lidar"], OS1_RADIUS + 0.005, OS1_HALF_HEIGHT + 0.005)
    rep["lidar_body_vertices"] = in_os1
    near = {**near, "lidar": in_os1}
    if args.move_lidar and in_os1:
        mx, my, mz = mounts["lidar"]
        allv = np.concatenate([p.v for p in body])
        m = np.hypot(allv[:, 0] - mx, allv[:, 1] - my) < 0.12
        new_z = round(float(allv[m, 2].max()) + OS1_HALF_HEIGHT + 0.005, 3)      # OS1: 73.5 mm high, origin at its centre
        sp = os.path.join(cfg, "sensors.yaml")
        with open(sp, encoding="utf-8") as fh:
            st = fh.read()
        i = st.index("lidar:")
        j = st.index("mount:", i)
        k = st.index("\n", j)
        line = re.sub(r"z:\s*[-0-9.]+", f"z: {new_z}", st[j:k], count=1)
        with open(sp, "w", encoding="utf-8") as fh:
            fh.write(st[:j] + line + st[k:])
        print(f"  LiDAR mount z {mz} -> {new_z} m in sensors.yaml (on top of the model at x = {mx:g}); set it back to {mz} to undo")
        rep["lidar_mount_z"] = {"old": mz, "new": new_z}
        mounts["lidar"] = (mx, my, new_z)
        near_after = M.clearance(body, mounts, 0.06)             # the OS1 itself: 75 mm wide, 74 mm high
        print(f"  LiDAR mount after the move: {near_after['lidar']} model vertices inside the sensor's own body (0 = sitting free on top)")
        rep["sensor_vertices_within_6cm_after_move"] = near_after
        near = {**near, "lidar": 0}
    for k, n in near.items():
        where = "model vertices / faces inside the OS1 body" if k == "lidar" else "body vertices within 12 cm"
        print(f"  {k} mount: {n} {where}" + ("  <-- the model may block this sensor (--move-lidar raises the LiDAR)" if n else ""))
    with open(os.path.join(args.out, f"{args.name}_cad.json"), "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=2)
    rel = os.path.relpath(args.out, ROOT).replace("\\", "/")
    yaml_body = f"{rel}/{args.name}_body.usda"
    yaml_wheel = f"{rel}/{args.name}_wheel_{{name}}.usda"
    if args.set_yaml:
        yp = os.path.join(cfg, "vehicle.yaml")
        with open(yp, encoding="utf-8") as fh:
            txt = fh.read()
        for key, val in (("usd_path", f'"{yaml_body}"'), ("usd_wheel_path", f'"{yaml_wheel}"'),
                         ("usd_fill_body", "true" if args.filler else "false"), ("usd_fill_hoop", "true" if args.hoop else "false")):
            if re.search(rf"(?m)^{key}:", txt):
                txt = re.sub(rf"(?m)^{key}:[^#\n]*", f"{key}: {val}      ", txt, count=1)
            else:
                txt = re.sub(r"(?m)^(usd_path:.*)$", rf"\1\n{key}: {val}", txt, count=1)
        with open(yp, "w", encoding="utf-8") as fh:
            fh.write(txt)
        print("vehicle.yaml updated: usd_path, usd_wheel_path, usd_fill_body, usd_fill_hoop")
    else:
        print(f"\nvehicle.yaml:\n  usd_path: \"{yaml_body}\"\n  usd_wheel_path: \"{yaml_wheel}\"   # wheels (optional)\n  usd_fill_body: {'true' if args.filler else 'false'}\n  usd_fill_hoop: {'true' if args.hoop else 'false'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
