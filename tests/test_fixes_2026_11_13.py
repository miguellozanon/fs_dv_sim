"""Fixes from the runs on the team PC, 13 Nov 2026: Reset simulation with the ZED camera, the "Newest" cone
detector, and the car's materials in Isaac Sim."""

import importlib.util
import math
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ros2_ws", "src", "fsdsim"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
HAVE_PXR = importlib.util.find_spec("pxr") is not None


def _read(*parts):
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as f:
        return f.read()


class TestResetRestartsTheCamera(unittest.TestCase):
    def test_isaac_camera_schedule_restarts_with_the_clock(self):
        src = _read("isaac", "run_isaac.py")
        # reset and the vehicle swap both restart the clock; the camera schedule and frame history follow
        self.assertEqual(src.count("restarted[0] = True"), 2)
        block = src[src.index("if restarted[0]:"):][:900]
        for line in ("next_cam = 0.0", "rendered_t.clear()", "last_isaac_t = None"):
            self.assertIn(line, block)

    def test_driver_starts_a_new_run_fresh(self):
        src = _read("ros2_ws", "src", "fsd_sim_bridge", "fsd_sim_bridge", "demo_driver_node.py")
        self.assertIn('self.create_subscription(ASStatus, "/as/status", self._on_status', src)
        block = src[src.index("def _on_status"):][:1200]
        self.assertIn("self.driver = None", block)
        self.assertIn("self._raced = False", block)


class TestNewestConeDetector(unittest.TestCase):
    def test_no_empty_launch_argument(self):
        # `ros2 launch ... cone_model:=` is "malformed launch argument": the relay never started (port 47800)
        src = _read("tools", "launcher", "launcher.py")
        self.assertNotIn('if o.get("cone_model") else "")]', src)
        self.assertIn('if o.get("cone_model"):', src)


class TestCarMaterials(unittest.TestCase):
    @staticmethod
    def _tube(p0, p1, r=0.009, n=12):
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        a = (p1 - p0) / np.linalg.norm(p1 - p0)
        u = np.cross(a, [0, 0, 1.0]) if abs(a[2]) < 0.9 else np.cross(a, [1.0, 0, 0])
        u /= np.linalg.norm(u)
        w = np.cross(a, u)
        ring = [r * (math.cos(t) * u + math.sin(t) * w) for t in np.linspace(0, 2 * math.pi, n, endpoint=False)]
        return np.array([p + q for p in (p0, p1) for q in ring])

    def test_rule_in_the_converter(self):
        import cad_to_usd as C

        self.assertTrue(C.is_rod(self._tube((0.7, 0.2, 0.15), (0.85, 0.57, 0.16))))         # a wishbone leg
        ribbon = np.array([[x, y, 0.45 + 0.002 * y] for x in np.linspace(1.39, 1.71, 9) for y in (0.08, 0.095)])
        self.assertFalse(C.is_rod(ribbon))                                                   # a nose panel strip

    @unittest.skipUnless(HAVE_PXR, "OpenUSD (pxr) not installed")
    def test_scene_materials(self):
        sys.path.insert(0, os.path.join(ROOT, "isaac"))
        import scene as SC
        from pxr import Gf, Usd, UsdGeom, UsdShade

        red, carbon, beige = (0.75, 0.05, 0.05), (0.06, 0.06, 0.07), (0.776, 0.659, 0.553)
        tube = self._tube((0.7, 0.2, 0.15), (0.85, 0.57, 0.16))
        panel = np.array([[x, y, 0.3] for x in (0.0, 0.8) for y in (0.2, 0.5)] + [[0.4, 0.35, 0.33]])
        self.assertEqual(SC.Scene.classify_part(red, tube), "metal")
        self.assertEqual(SC.Scene.classify_part(red, panel), "paint")
        self.assertEqual(SC.Scene.classify_part(carbon, panel), "carbon")
        self.assertEqual(SC.Scene.classify_part(beige, panel), "paint")
        self.assertEqual(SC.Scene.classify_part((0.55, 0.55, 0.58), panel), "metal")

        st = Usd.Stage.CreateInMemory()
        sc = SC.Scene.__new__(SC.Scene)
        sc.stage, sc.materials = st, {}
        body = UsdGeom.Xform.Define(st, "/World/Car/body").GetPrim()
        for name, pts, col in (("rod", tube, red), ("side", panel, red)):
            m = UsdGeom.Mesh.Define(st, f"/World/Car/body/{name}")
            m.CreatePointsAttr([Gf.Vec3f(*map(float, p)) for p in pts])
            m.CreateFaceVertexCountsAttr([3])
            m.CreateFaceVertexIndicesAttr([0, 1, 2])
            m.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set([Gf.Vec3f(*col)])
        self.assertEqual(sc._paint_body(body), {"metal": 1, "paint": 1})
        bound = UsdShade.MaterialBindingAPI(st.GetPrimAtPath("/World/Car/body/rod")).ComputeBoundMaterial()[0]
        self.assertEqual(bound.GetPath().name, "car_metal")

        # wheel: tyre faces at the outer radius, a hub face inside -> rubber + a "rim" subset
        wheel = UsdGeom.Xform.Define(st, "/World/Car/wheel_rl/spin/tyre").GetPrim()
        w = UsdGeom.Mesh.Define(st, "/World/Car/wheel_rl/spin/tyre/Body")
        pts = [(0.25, -0.1, 0.0), (0.25, 0.1, 0.0), (0.0, -0.1, 0.25), (0.05, 0.0, 0.0), (0.1, 0.0, 0.05), (0.0, 0.0, 0.1)]
        w.CreatePointsAttr([Gf.Vec3f(*p) for p in pts])
        w.CreateFaceVertexCountsAttr([3, 3])
        w.CreateFaceVertexIndicesAttr([0, 1, 2, 3, 4, 5])
        sc._paint_wheel(wheel, 0.25)
        api = UsdShade.MaterialBindingAPI(w.GetPrim())
        self.assertEqual(api.ComputeBoundMaterial()[0].GetPath().name, "car_rubber")
        subs = api.GetMaterialBindSubsets()
        self.assertEqual([list(s.GetIndicesAttr().Get()) for s in subs], [[1]])
