"""perf/perflab/ocp.py and methods.py: minimum-time optimal control and the method comparison."""

import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "perf"))
LAPSIM = os.path.join(ROOT, "lapsim", "LR26_LapSim")
HAVE_LAPSIM = os.path.isfile(os.path.join(LAPSIM, "config", "lr26_params.m"))
try:
    import casadi  # noqa: F401

    HAVE_CASADI = True
except ImportError:
    HAVE_CASADI = False


class TestTrack(unittest.TestCase):
    def test_circle_and_heading(self):
        from perflab.ocp import Track

        R, L = 10.0, 2 * np.pi * 10.0
        s = np.linspace(0, L, 126)
        tr = Track(s, np.full(len(s), 1 / R), True, L / 120)
        self.assertEqual(tr.n, 120)                                   # closed: no repeated end node
        self.assertAlmostEqual(tr.psi[-1] + tr.ds / R, 2 * np.pi, places=6)
        self.assertLess(np.hypot(tr.x[0] - tr.x[-1], tr.y[0] - tr.y[-1]), 1.0)
        x, y = tr.line_xy(np.full(tr.n, 1.0))                         # 1 m to the left of a left-hand circle
        self.assertAlmostEqual(float(np.mean(np.hypot(x - x.mean(), y - y.mean()))), R - 1, delta=0.05)


@unittest.skipUnless(HAVE_LAPSIM, "lapsim/LR26_LapSim not present (confidential, not in git)")
class LabCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from perflab.lab import Lab

        cls.tmp = tempfile.mkdtemp(prefix="perf-ocp-")
        cls.lab = Lab(LAPSIM, configs_dir=cls.tmp)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)


@unittest.skipUnless(HAVE_CASADI, "casadi not installed (optional package)")
class TestOcp(LabCase):
    def test_point_mass_agrees_with_qss_on_the_same_grid(self):
        from perflab import ocp

        for ev in ("skidpad", "autocross"):
            r = ocp.run(self.lab, ev, "point_mass")
            self.assertTrue(r["ok"], r["stages"])
            q = r["qss"]["t_lap_same_grid"]
            # same physics as QSS solved as one optimisation: within 0.5 %, and not slower than QSS
            self.assertLess(abs(r["t_lap"] / q - 1), 0.005, (ev, r["t_lap"], q))
            self.assertEqual(r["warnings"], [])
            self.assertTrue(r["fixed_line"])
            self.assertAlmostEqual(float(np.max(np.abs(r["trace"]["n"]))), 0.0, places=6)

    def test_dynamic_models_on_skidpad(self):
        from perflab import ocp

        res = {}
        for name, model, tv in (("st", "single_track", False), ("dt", "double_track", False),
                                ("tv", "double_track", True)):
            r = ocp.run(self.lab, "skidpad", model, torque_vectoring=tv)
            self.assertTrue(r["ok"], r["stages"])
            self.assertEqual([s["model"] for s in r["stages"]][-1], model)
            self.assertIn("delta", r["trace"])
            res[name] = r["t_lap"]
        pm = ocp.run(self.lab, "skidpad", "point_mass")["t_lap"]
        self.assertLess(pm, res["st"])                   # more physics, never faster than the point mass
        self.assertLessEqual(res["tv"], res["dt"] + 1e-3)  # torque vectoring can only help
        self.assertLess(res["st"] / pm, 1.05)

    def test_acceleration_and_free_line_warning(self):
        from perflab import ocp

        r = ocp.run(self.lab, "acceleration", "single_track")
        self.assertTrue(r["ok"], r["stages"])
        self.assertAlmostEqual(r["t_lap"], r["qss"]["t_lap"], delta=0.05)
        r = ocp.run(self.lab, "skidpad", "point_mass", fixed_line=False)
        self.assertTrue(any("Free line" in w for w in r["warnings"]))
        self.assertGreater(float(np.max(np.abs(r["trace"]["n"]))), 0.1)

    def test_failed_stage_gives_no_time(self):
        from perflab import ocp

        r = ocp.run(self.lab, "skidpad", "single_track", max_iter=2)
        self.assertFalse(r["ok"])
        self.assertIsNone(r["T"])
        self.assertIsNone(r["points"])
        self.assertTrue(r["warnings"])

    def test_bad_input(self):
        from perflab import ocp

        with self.assertRaises(ValueError):
            ocp.run(self.lab, "skidpad", "full_car")
        with self.assertRaises(ValueError):
            ocp.run(self.lab, "drag_race", "point_mass")


class TestMethods(LabCase):
    def test_compare_quick_methods(self):
        from perflab import methods

        ms = ["qss", "qss_point_mass"] + (["ocp_point_mass"] if HAVE_CASADI else [])
        r = methods.compare(self.lab, "skidpad", "mass-5", ms)
        rows = {m["method"]: m for m in r["methods"]}
        self.assertAlmostEqual(rows["qss"]["base"]["T"], 5.154, places=2)
        self.assertEqual(rows["qss"]["vs_qss"], 0.0)
        for m in rows.values():
            self.assertTrue(m["base"]["ok"])
            self.assertLess(m["dT"], 0)                  # 5 kg lighter is faster with every method
        self.assertTrue(r["agree"])
        with self.assertRaises(ValueError):
            methods.compare(self.lab, "skidpad", "", ["magic"])

    def test_monte_carlo(self):
        from perflab import methods

        r = methods.monte_carlo(self.lab, 12, "mass-5", {"mu": 0.02, "cl": 0.0}, seed=3)
        self.assertEqual(len(r["samples"]), 12)
        self.assertGreater(r["total"]["sd"], 0)
        self.assertLessEqual(r["total"]["p5"], r["total"]["p95"])
        self.assertEqual(r["gain"]["p_gain"], 1.0)        # a lighter car gains on every draw
        self.assertGreater(r["gain"]["nominal"], 0)
        r2 = methods.monte_carlo(self.lab, 12, "", {"mu": 0.02, "cl": 0.0}, seed=3)
        self.assertEqual(r2["samples"], r["samples"])      # same seed, same draws
        self.assertNotIn("gain", r2)
        with self.assertRaises(ValueError):
            methods.monte_carlo(self.lab, 3)
        self.assertTrue(methods.guidance())


if __name__ == "__main__":
    unittest.main()
