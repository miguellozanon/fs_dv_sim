"""CAD import: STEP read as text, mesh readers, CAD -> simulator frame, wheel split, USD output, CLI."""

import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "ros2_ws", "src")
sys.path[:0] = [os.path.join(SRC, "fsdsim"), os.path.join(ROOT, "tools")]

import numpy as np  # noqa: E402

from fsdsim.cad import meshes as M  # noqa: E402
from fsdsim.cad import step_info as S  # noqa: E402
from fsdsim.config import load_all_configs  # noqa: E402

CAD = os.path.join(ROOT, "cad", "lr26")
CONFIG = os.path.join(SRC, "fsd_sim_bringup", "config")


def box(lo, hi, name="box"):
    lo, hi = np.array(lo, float), np.array(hi, float)
    c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    f = []
    for a, b, cc, d in quads:
        f += [(a, b, cc), (a, cc, d)]
    p = M.Part(name, c, np.array(f))
    # make the winding outward
    tri = p.v[p.f]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    cen = tri.mean(axis=1) - p.v.mean(axis=0)
    flip = np.einsum("ij,ij->i", n, cen) < 0
    p.f[flip] = p.f[flip][:, ::-1]
    return p


def wheel_part(centre, radius, half_w, name, n=24):
    """Cylinder with its axis along y."""
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.stack([radius * np.cos(a), np.zeros(n), radius * np.sin(a)], axis=1)
    v = np.concatenate([ring + [0, -half_w, 0], ring + [0, half_w, 0], [[0, -half_w, 0]], [[0, half_w, 0]]]) + np.array(centre)
    f = []
    for i in range(n):
        j = (i + 1) % n
        f += [(i, j, n + j), (i, n + j, n + i), (2 * n, j, i), (2 * n + 1, n + i, n + j)]
    part = M.Part(name, v, np.array(f))
    tri = part.v[part.f]
    out = np.einsum("ij,ij->i", np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), tri.mean(axis=1) - part.v.mean(axis=0)) < 0
    part.f[out] = part.f[out][:, ::-1]                              # outward winding
    return part


def write_obj(path, parts):
    off = 0
    with open(path, "w") as fh:
        for p in parts:
            fh.write(f"o {p.name}\n")
            for v in p.v:
                fh.write("v %.6f %.6f %.6f\n" % tuple(v))
            for t in p.f + off + 1:
                fh.write("f %d %d %d\n" % tuple(t))
            off += len(p.v)


def write_stl(path, parts):
    tri = np.concatenate([p.v[p.f] for p in parts]).astype(np.float32)
    with open(path, "wb") as fh:
        fh.write(b"\0" * 80 + struct.pack("<I", len(tri)))
        for t in tri:
            fh.write(struct.pack("<3f", 0, 0, 0) + t.tobytes() + b"\0\0")


MINI_STEP = """ISO-10303-21;
HEADER;
FILE_NAME('mini','2026',(''),(''),'','','');
ENDSEC;
DATA;
#1=( LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.) );
#10=CARTESIAN_POINT('',(0.,-600.,250.));
#11=DIRECTION('',(0.,1.,0.));
#12=DIRECTION('',(1.,0.,0.));
#13=AXIS2_PLACEMENT_3D('',#10,#11,#12);
#14=TOROIDAL_SURFACE('',#13,230.,25.);
#20=CARTESIAN_POINT('',(0.,-680.,250.));
#23=AXIS2_PLACEMENT_3D('',#20,#11,#12);
#24=TOROIDAL_SURFACE('',#23,230.,25.);
#30=CARTESIAN_POINT('',(0.,600.,250.));
#31=DIRECTION('',(0.,-1.,0.));
#33=AXIS2_PLACEMENT_3D('',#30,#31,#12);
#34=TOROIDAL_SURFACE('',#33,230.,25.);
#40=CARTESIAN_POINT('',(0.,680.,250.));
#43=AXIS2_PLACEMENT_3D('',#40,#31,#12);
#44=TOROIDAL_SURFACE('',#43,230.,25.);
#50=CARTESIAN_POINT('',(1600.,-600.,250.));
#53=AXIS2_PLACEMENT_3D('',#50,#11,#12);
#54=TOROIDAL_SURFACE('',#53,230.,25.);
#60=CARTESIAN_POINT('',(1600.,-680.,250.));
#63=AXIS2_PLACEMENT_3D('',#60,#11,#12);
#64=TOROIDAL_SURFACE('',#63,230.,25.);
#70=CARTESIAN_POINT('',(1600.,600.,250.));
#73=AXIS2_PLACEMENT_3D('',#70,#31,#12);
#74=TOROIDAL_SURFACE('',#73,230.,25.);
#80=CARTESIAN_POINT('',(1600.,680.,250.));
#83=AXIS2_PLACEMENT_3D('',#80,#31,#12);
#84=TOROIDAL_SURFACE('',#83,230.,25.);
#100=ADVANCED_FACE('',(),#14,.T.);
#101=ADVANCED_FACE('',(),#24,.T.);
#102=ADVANCED_FACE('',(),#34,.T.);
#103=ADVANCED_FACE('',(),#44,.T.);
#104=ADVANCED_FACE('',(),#54,.T.);
#105=ADVANCED_FACE('',(),#64,.T.);
#106=ADVANCED_FACE('',(),#74,.T.);
#107=ADVANCED_FACE('',(),#84,.T.);
#110=CLOSED_SHELL('',(#100,#101,#102,#103,#104,#105,#106,#107));
#111=MANIFOLD_SOLID_BREP('Tyres',#110);
ENDSEC;
END-ISO-10303-21;
"""


class TestStepInfo(unittest.TestCase):
    def test_synthetic_wheels_and_dimensions(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "mini.stp")
            with open(p, "w") as fh:
                fh.write(MINI_STEP)
            info = S.read(p)
        self.assertEqual(info.unit_to_m, 1e-3)
        self.assertEqual(len(info.solids), 1)
        self.assertEqual(len(info.wheels), 4)
        dims = S.vehicle_dimensions(info)
        self.assertAlmostEqual(dims["wheelbase"], 1.6, places=4)
        self.assertAlmostEqual(dims["track_front"], 1.28, places=4)     # shoulders at 600 / 680 -> wheel centre 640 mm
        self.assertAlmostEqual(dims["tyre_outer_radius"], 0.255, places=4)
        self.assertAlmostEqual(dims["wheel_radius_centre_height"], 0.25, places=4)

    @unittest.skipUnless(os.path.isfile(os.path.join(CAD, "LR26_CFD_Asm_Monosuspension.stp")), "LR26 CAD not in cad/lr26")
    def test_lr26_cad_matches_lapsim_geometry(self):
        veh, _, _ = load_all_configs(CONFIG)
        info = S.read(os.path.join(CAD, "LR26_CFD_Asm_Monosuspension.stp"))
        dims = S.vehicle_dimensions(info)
        self.assertAlmostEqual(dims["wheelbase"], veh.wheelbase, delta=0.002)
        self.assertAlmostEqual(dims["track_front"], veh.track_front, delta=0.002)
        self.assertAlmostEqual(dims["wheel_radius_centre_height"], veh.wheel_radius, delta=0.002)

    @unittest.skipUnless(os.path.isfile(os.path.join(CAD, "LR26_FW_final.stp")), "LR26 CAD not in cad/lr26")
    def test_footprint_follows_cad(self):
        veh, _, _ = load_all_configs(CONFIG)
        fw = S.read(os.path.join(CAD, "LR26_FW_final.stp")).bounds()
        rw = S.read(os.path.join(CAD, "LR26_RW_final.stp")).bounds()
        self.assertAlmostEqual(veh.footprint.front_overhang, 0.0 - fw[0][0], delta=0.01)
        self.assertAlmostEqual(veh.footprint.rear_overhang, rw[1][0] - veh.wheelbase, delta=0.01)
        self.assertAlmostEqual(veh.footprint.width, 2 * fw[1][1], delta=0.01)


class TestMeshes(unittest.TestCase):
    def test_stl_binary_ascii_obj_roundtrip(self):
        p = box((0, 0, 0), (1, 2, 3))
        with tempfile.TemporaryDirectory() as d:
            write_stl(os.path.join(d, "b.stl"), [p])
            with open(os.path.join(d, "a.stl"), "w") as fh:
                fh.write("solid x\n")
                for t in p.v[p.f]:
                    fh.write("facet normal 0 0 0\nouter loop\n" + "".join("vertex %g %g %g\n" % tuple(v) for v in t) + "endloop\nendfacet\n")
                fh.write("endsolid x\n")
            write_obj(os.path.join(d, "o.obj"), [p, box((5, 5, 5), (6, 6, 6), "other")])
            for name in ("b.stl", "a.stl"):
                got = M.read_stl(os.path.join(d, name))[0]
                self.assertEqual(len(got.v), 8)
                self.assertEqual(len(got.f), 12)
                np.testing.assert_allclose(got.bounds[1], [1, 2, 3])
            parts = M.read_obj(os.path.join(d, "o.obj"))
        self.assertEqual([q.name for q in parts], ["box", "other"])
        np.testing.assert_allclose(parts[1].bounds[0], [5, 5, 5])

    def test_frame_lr26(self):
        fr = M.Frame()                                  # mm, front axle x = 0, x to the rear, lf 0.85065
        v = fr.apply(np.array([[0.0, 0.0, 249.0], [1605.0, 613.2, 0.0], [-956.0, 0.0, 50.0]]))
        np.testing.assert_allclose(v[0], [0.85065, 0.0, 0.249])
        np.testing.assert_allclose(v[1], [0.85065 - 1.605, -0.6132, 0.0], atol=1e-9)    # rear axle at -lr
        np.testing.assert_allclose(v[2], [0.85065 + 0.956, 0.0, 0.05])                  # wing tip ahead of the CoG
        fwd = M.Frame(1.0, 0.0, 0.5, x_rearward=False).apply(np.array([[1.0, 2.0, 3.0]]))
        np.testing.assert_allclose(fwd[0], [1.5, 2.0, 3.0])

    def wheels(self):
        return M.default_wheels(0.85065, 0.75435, 1.2264, 1.2264, 0.2565, 0.2007, 0.249)

    def test_split_wheels_by_part(self):
        ws = self.wheels()
        parts = [box((-0.5, -0.3, 0.1), (0.5, 0.3, 0.5), "tub")]
        parts += [wheel_part(w.centre, 0.25, 0.1, f"tyre_{w.name}") for w in ws]
        body, out = M.split_wheels(parts, ws)
        self.assertEqual([p.name for p in body], ["tub"])
        for w in ws:
            self.assertEqual(len(out[w.name]), 1)
            np.testing.assert_allclose(out[w.name][0].v.mean(axis=0), [0, 0, 0], atol=1e-6)   # centred on the hub

    def test_split_wheels_single_mesh_by_triangle(self):
        ws = self.wheels()
        tub = box((-0.5, -0.3, 0.1), (0.5, 0.3, 0.5))
        parts = [tub] + [wheel_part(w.centre, 0.25, 0.1, "x") for w in ws]
        # one STL: merge to a single part
        v = np.concatenate([p.v for p in parts])
        off = np.cumsum([0] + [len(p.v) for p in parts[:-1]])
        f = np.concatenate([p.f + o for p, o in zip(parts, off)])
        body, out = M.split_wheels([M.Part("car", v, f)], ws)
        self.assertEqual(len(body), 1)
        self.assertEqual(len(body[0].f), 12)                    # the tub only
        for w in ws:
            self.assertGreater(sum(len(p.f) for p in out[w.name]), 90)

    def test_part_straddling_a_wheel_stays_with_the_body(self):
        ws = self.wheels()
        arm = box((0.85, 0.3, 0.2), (0.86, 0.9, 0.3), "arm")    # a wishbone reaching into the front-left wheel
        body, out = M.split_wheels([arm, box((-0.5, -0.3, 0.1), (0.5, 0.3, 0.5), "tub")], ws)
        self.assertEqual({p.name for p in body}, {"arm", "tub"})
        self.assertTrue(all(not v for v in out.values()))

    def test_normals_hard_edges_and_smooth_surface(self):
        n = M.corner_normals(box((0, 0, 0), (1, 1, 1)))
        tri = box((0, 0, 0), (1, 1, 1))
        tn = np.cross(tri.v[tri.f][:, 1] - tri.v[tri.f][:, 0], tri.v[tri.f][:, 2] - tri.v[tri.f][:, 0])
        tn = tn / np.linalg.norm(tn, axis=1)[:, None]
        np.testing.assert_allclose(n, np.repeat(tn, 3, axis=0), atol=1e-9)           # cube edges stay hard
        w = wheel_part((0, 0, 0), 1.0, 0.5, "w", n=48)
        n2 = M.corner_normals(w)
        # on the cylinder wall the normal points away from the axis (no y component)
        wall = (np.abs(n2[:, 1]) < 0.2)
        self.assertGreater(np.count_nonzero(wall), 0)
        radial = np.stack([w.v[w.f.reshape(-1)][:, 0], np.zeros(len(n2)), w.v[w.f.reshape(-1)][:, 2]], axis=1)
        radial = radial / np.maximum(np.linalg.norm(radial, axis=1)[:, None], 1e-9)
        self.assertGreater(float(np.mean(np.einsum("ij,ij->i", n2[wall], radial[wall]))), 0.97)

    def test_write_and_read_usda(self):
        p = box((0, 0, 0), (1, 1, 1), "a b")
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x.usda")
            r = M.write_usda(path, "Body", [p, p.copy(name="a b")])
            txt = open(path).read()
            back = M.read_usda(path)
        self.assertEqual(r, {"parts": 2, "triangles": 24})
        self.assertIn('defaultPrim = "Body"', txt)
        self.assertIn('upAxis = "Z"', txt)
        self.assertEqual(set(back), {"a_b", "a_b_2"})
        np.testing.assert_allclose(back["a_b"][0], p.v, atol=1e-5)

    def test_inside_cylinder(self):
        p = box((-0.1, -0.1, 0.0), (0.1, 0.1, 0.5))
        self.assertGreater(M.inside_cylinder([p], (0.0, 0.0, 0.5), 0.2, 0.05), 0)      # sensor centre on the top face
        self.assertEqual(M.inside_cylinder([p], (0.0, 0.0, 0.6), 0.2, 0.05), 0)        # sitting 5 cm above it
        big = box((-1.0, -1.0, 0.0), (1.0, 1.0, 1.0))                                   # no vertex near the sensor ...
        self.assertGreater(M.inside_cylinder([big], (0.3, 0.0, 0.98), 0.04, 0.04), 0)   # ... but its top face cuts it
        self.assertEqual(M.inside_cylinder([big], (0.3, 0.0, 0.5), 0.04, 0.04), 0)      # enclosed but not touching

    def test_clearance(self):
        p = box((0, 0, 0), (1, 1, 1))
        res = M.clearance([p], {"near": (0.05, 0.0, 0.0), "far": (3, 3, 3)}, 0.1)
        self.assertGreater(res["near"], 0)
        self.assertEqual(res["far"], 0)


class TestCadCli(unittest.TestCase):
    def test_cad_to_usd_with_synthetic_car(self):
        veh, _, _ = load_all_configs(CONFIG)
        # a synthetic car in the LR26 CAD frame (mm, x rearward, front axle at 0)
        parts = [box((-900, -250, 40), (1800, 250, 450), "mono")]
        for x in (0.0, 1605.0):
            for y in (-613.2, 613.2):
                parts.append(wheel_part((x, y, 249.0), 250.0, 100.0, f"tyre_{x:.0f}_{y:.0f}"))
        with tempfile.TemporaryDirectory() as d:
            obj = os.path.join(d, "car.obj")
            write_obj(obj, parts)
            cfg = os.path.join(d, "config")
            shutil.copytree(CONFIG, cfg)
            out = os.path.join(d, "assets")
            r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "cad_to_usd.py"), obj, "--out", out, "--name", "t",
                                "--config", cfg, "--set-yaml"], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            for f in ("t_body.usda", "t_wheel_fl.usda", "t_wheel_fr.usda", "t_wheel_rl.usda", "t_wheel_rr.usda", "t_cad.json"):
                self.assertTrue(os.path.isfile(os.path.join(out, f)), f)
            wheel = M.read_usda(os.path.join(out, "t_wheel_fl.usda"))
            pts = np.concatenate([v for v, _ in wheel.values()])
            np.testing.assert_allclose(pts.mean(axis=0), [0, 0, 0], atol=1e-3)             # centred on the hub
            body = M.read_usda(os.path.join(out, "t_body.usda"))
            allp = np.concatenate([v for v, _ in body.values()])
            self.assertAlmostEqual(allp[:, 0].max(), veh.lf + 0.9, places=3)               # nose ahead of the CoG
            rep = json.load(open(os.path.join(out, "t_cad.json")))
            self.assertEqual(rep["outputs"]["body"]["parts"], 1)
            yml = open(os.path.join(cfg, "vehicle.yaml")).read()
            self.assertIn('usd_wheel_path: "', yml)
            self.assertIn("t_wheel_{name}.usda", yml)
            # the changed yaml still loads and the new fields are read
            v2, _, _ = load_all_configs(cfg)
            self.assertTrue(v2.usd_path.endswith("t_body.usda"))
            self.assertIn("{name}", v2.usd_wheel_path)

    def test_simplify_keeps_small_parts_and_shrinks_big_ones(self):
        small = box((0, 0, 0), (1, 1, 1))
        self.assertIs(M.simplify(small, 100), small)
        self.assertIs(M.simplify(small, 0), small)
        big = wheel_part((0, 0, 0), 0.5, 0.2, "spring", n=400)             # 1600 triangles
        out = M.simplify(big, 400)
        self.assertLessEqual(len(out.f), 400)
        self.assertGreater(len(out.f), 20)
        lo, hi = big.bounds
        lo2, hi2 = out.bounds
        self.assertLess(np.abs(hi2 - hi).max(), 0.08)                      # shape stays within a few cells
        self.assertLess(np.abs(lo2 - lo).max(), 0.08)

    def test_obj_material_colours(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "c.obj")
            with open(p, "w") as fh:
                fh.write("o a\nusemtl Opaque(255,0,51)\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"
                         "o b\nv 0 0 1\nv 1 0 1\nv 0 1 1\nf 4 5 6\n")
            parts = {q.name: q for q in M.read_obj(p)}
            self.assertEqual(len(parts), 2)
            np.testing.assert_allclose(parts["a"].colour, (1.0, 0.0, 0.2))
            self.assertIsNone(parts["b"].colour)

    def test_paint_guess(self):
        sys.path.insert(0, os.path.join(ROOT, "tools"))
        import cad_to_usd as C
        wing = box((1.2, -0.7, 0.05), (1.5, 0.7, 0.2), "w")                # ahead of the nose axle, low
        self.assertEqual(C.paint(wing), C.CARBON)
        rear = box((-1.2, -0.5, 0.6), (-0.9, 0.5, 0.9), "r")
        self.assertEqual(C.paint(rear), C.CARBON)
        body = box((-0.2, -0.3, 0.1), (0.8, 0.3, 0.5), "b")
        self.assertEqual(C.paint(body), C.RED)
        bolt = box((0.0, 0.0, 0.1), (0.02, 0.02, 0.12), "bolt")
        self.assertEqual(C.paint(bolt), C.GREY)

    @staticmethod
    def _lidar_z(cfg, z):
        """Put the LiDAR mount of a config copy at height z (the shipped sensors.yaml has it raised for the LR26 CAD)."""
        import re
        sp = os.path.join(cfg, "sensors.yaml")
        with open(sp, encoding="utf-8") as fh:
            txt = fh.read()
        i = txt.index("mount:", txt.index("lidar:"))
        j = txt.index("\n", i)
        txt = txt[:i] + re.sub(r"z:\s*[-0-9.]+", f"z: {z}", txt[i:j], count=1) + txt[j:]
        with open(sp, "w", encoding="utf-8") as fh:
            fh.write(txt)

    def test_move_lidar_and_hoop_options(self):
        # a car whose nose shell is solid around the default LiDAR mount (x = 1.55, z = 0.22 in the simulator)
        veh, _, _ = load_all_configs(CONFIG)
        z_levels = (40, 150, 250, 350)
        parts = [box((x, y, z), (x + 100, y + 100, z + 100 if z < 350 else 450), f"mono_{x}_{y}_{z}")     # a finely tessellated nose shell
                 for x in range(-900, 1800, 100) for y in range(-250, 250, 100) for z in z_levels]
        for x in (0.0, 1605.0):
            for y in (-613.2, 613.2):
                parts.append(wheel_part((x, y, 249.0), 250.0, 100.0, f"tyre_{x:.0f}_{y:.0f}"))
        with tempfile.TemporaryDirectory() as d:
            obj = os.path.join(d, "car.obj")
            write_obj(obj, parts)
            cfg = os.path.join(d, "config")
            shutil.copytree(CONFIG, cfg)
            self._lidar_z(cfg, 0.22)
            cmd = [sys.executable, os.path.join(ROOT, "tools", "cad_to_usd.py"), obj, "--out", os.path.join(d, "a"), "--name", "t",
                   "--config", cfg, "--set-yaml", "--hoop", "--move-lidar"]
            # lidar mount at sim x = 1.55 is cad x = lf - 1.55 = -0.699 m, inside the box (x from -0.9 m)
            r = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn("LiDAR mount z", r.stdout)
            _, sens, _ = load_all_configs(cfg)
            self.assertAlmostEqual(sens.lidar.mount.z, 0.45 + 0.0368 + 0.005, places=3)
            rep = json.load(open(os.path.join(d, "a", "t_cad.json")))
            self.assertEqual(rep["lidar_mount_z"]["old"], 0.22)
            self.assertEqual(rep["sensor_vertices_within_6cm_after_move"]["lidar"], 0)
            # a second run on the moved mount: the sensor sits on the model, nothing to report or move
            r = subprocess.run(cmd[:-1], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertNotIn("may block this sensor", r.stdout)
            self.assertEqual(json.load(open(os.path.join(d, "a", "t_cad.json")))["lidar_body_vertices"], 0)
            self.assertAlmostEqual(load_all_configs(cfg)[1].lidar.mount.z, 0.45 + 0.0368 + 0.005, places=3)
            v2, _, _ = load_all_configs(cfg)
            self.assertTrue(v2.usd_fill_hoop)
            self.assertFalse(v2.usd_fill_body)
            # without the options the sensors file is untouched and the hoop filler is off
            cfg2 = os.path.join(d, "config2")
            shutil.copytree(CONFIG, cfg2)
            self._lidar_z(cfg2, 0.22)
            r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "cad_to_usd.py"), obj, "--out", os.path.join(d, "b"),
                                "--name", "t", "--config", cfg2, "--set-yaml"], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn("may block this sensor", r.stdout)
            _, s2, _ = load_all_configs(cfg2)
            self.assertEqual(s2.lidar.mount.z, 0.22)
            self.assertFalse(load_all_configs(cfg2)[0].usd_fill_hoop)

    def test_defaults_are_empty(self):
        veh, _, _ = load_all_configs(CONFIG)
        generic = os.path.join(CONFIG, "vehicle_generic.yaml")
        with tempfile.TemporaryDirectory() as d:
            shutil.copytree(CONFIG, os.path.join(d, "c"))
            shutil.copy(generic, os.path.join(d, "c", "vehicle.yaml"))
            g, _, _ = load_all_configs(os.path.join(d, "c"))
        self.assertEqual(g.usd_wheel_path, "")
        self.assertTrue(g.usd_fill_body)
        self.assertTrue(g.usd_fill_hoop)
        self.assertTrue(veh.usd_wheel_path.endswith("{name}.usda"))        # the shipped config uses the LR26 CAD

    def test_inspect_cli_on_synthetic_step(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "mini.stp")
            with open(p, "w") as fh:
                fh.write(MINI_STEP)
            r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "cad_inspect.py"), p, "--json", os.path.join(d, "o.json")],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn("wheelbase", r.stdout)
            self.assertEqual(json.load(open(os.path.join(d, "o.json")))["dims"]["wheelbase"], 1.6)


if __name__ == "__main__":
    unittest.main()
