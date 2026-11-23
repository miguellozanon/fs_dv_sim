"""Racing line, efficiency map, ggv map, solver plug-ins, closed-loop energy and the Marple export of sim runs."""

import csv
import math
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEEDS_LAPSIM = unittest.skipUnless(os.path.isfile(os.path.join(ROOT, "lapsim", "LR26_LapSim", "config", "lr26_params.m")),
                                   "lapsim/LR26_LapSim not present (confidential, not in git): CI runs without it")
sys.path[:0] = [os.path.join(ROOT, "ros2_ws", "src", "fsdsim"), os.path.join(ROOT, "tools")]
CONFIG = os.path.join(ROOT, "ros2_ws", "src", "fsd_sim_bringup", "config")
LAPSIM = os.path.join(ROOT, "lapsim", "LR26_LapSim")

import numpy as np  # noqa: E402

from fsdsim.config import load_all_configs  # noqa: E402
from fsdsim.perf import efficiency, ggv, lapsim, raceline, solvers  # noqa: E402
from fsdsim.perf.efficiency import EfficiencyMap  # noqa: E402
from fsdsim.track import build_track  # noqa: E402


class TestRacingLine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.veh, _, cls.sim = load_all_configs(CONFIG)
        cls.tr = build_track("trackdrive", cls.sim.events, cls.veh)

    def test_circle_stays_a_circle_and_oval_cuts_corners(self):
        a = np.linspace(0, 2 * np.pi, 200, endpoint=False)
        circ = np.stack([30 * np.cos(a), 30 * np.sin(a)], 1)
        xy, off, b = raceline.offset_line(circ, 3.5, True, 1.5, 0.25)
        self.assertAlmostEqual(b, 0.75)
        self.assertLess(np.ptp(off), 0.01)                            # nothing to gain: constant offset
        oval = np.stack([40 * np.cos(a), 15 * np.sin(a)], 1)
        xy, off, b = raceline.offset_line(oval, 3.5, True, 1.5, 0.25)
        self.assertLessEqual(np.abs(off).max(), b + 1e-9)
        self.assertGreater(np.ptp(off), 0.8)                          # out - in - out

    def test_line_is_inside_the_track_smooth_and_faster(self):
        q = self.veh.perf().copy(util_x=0.8, util_y=0.8, events={})
        r = raceline.racing_line(self.tr.centerline, self.tr.width, q, True, car_width=self.veh.footprint.width, margin=0.4)
        self.assertAlmostEqual(r["bound"], self.tr.width / 2 - self.veh.footprint.width / 2 - 0.4)
        self.assertLessEqual(np.abs(r["offset"]).max(), r["bound"] + 1e-9)
        self.assertLess(np.abs(np.diff(np.r_[r["offset"], r["offset"][0]])).max(), 0.2)     # no kink, also at the seam
        self.assertLess(r["t_line"], r["t_center"] * 0.97)
        self.assertLess(r["length"], r["length_center"])
        seg = np.linalg.norm(np.diff(r["xy"], axis=0), axis=1)
        self.assertGreater(seg.min(), 0.7)

    def test_driver_on_the_racing_line_is_clean_and_faster(self):
        import closed_loop_eval as C
        t = {}
        for line in ("center", "racing"):
            for dt in (0.02, 0.05):
                _, res, _, _ = C.run("autocross", CONFIG, mode="map", verbose=False, line=line, dt=dt)
                self.assertEqual((res.state, res.cones_hit, res.off_course), ("FINISHED", 0, 0), (line, dt))
                t[line, dt] = res.raw_time
        self.assertLess(t["racing", 0.02], t["center", 0.02] - 0.2)

    def test_open_tracks_keep_the_centre_line(self):
        from fsdsim.control import reference_path
        acc = build_track("acceleration", self.sim.events, self.veh)
        path, info = reference_path(acc.centerline, acc.width, acc.closed, self.veh, "racing")
        self.assertIsNone(info)
        self.assertIs(path, acc.centerline)


@NEEDS_LAPSIM
class TestEfficiencyMap(unittest.TestCase):
    def tearDown(self):
        efficiency.set_active(None)

    def test_interpolation_clamp_and_roundtrip(self):
        m = EfficiencyMap([0, 10, 20], [0, 1000, 2000], [[0.5, 0.6, 0.7], [0.6, 0.7, 0.8], [0.7, 0.8, 0.9]])
        self.assertAlmostEqual(m.eta(5, 500), 0.6)
        self.assertAlmostEqual(m.eta(99, 99999), 0.9)
        np.testing.assert_allclose(m.eta(np.array([0.0, 20.0]), np.array([0.0, 2000.0])), [0.5, 0.9])
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.json")
            m.save(p)
            self.assertAlmostEqual(EfficiencyMap.load(p).eta(5, 500), 0.6)

    def test_loss_fit_map_reproduces_the_loss_model(self):
        k = [0.1, 0.004, 0.05, 0.001]                 # kW = k0 + k1 T^2 + k2 n + k3 n^2
        m = EfficiencyMap.from_loss_fit(k, t_max=20, n_max=20000, motors_per_set=2.0, nt=41, nn=41)
        T, n = 12.0, 9000.0                           # per motor
        p = T * n * 2 * math.pi / 60 / 1e3
        loss = k[0] + k[1] * T ** 2 + k[2] * n / 1e3 + k[3] * (n / 1e3) ** 2
        self.assertAlmostEqual(m.eta(2 * T, n), p / (p + loss), delta=0.003)     # eta() takes the torque of the set

    def test_constant_map_equals_constant_eta_in_lapsim_and_a_worse_map_costs_energy(self):
        ls = lapsim.LapSim.load(LAPSIM)
        base = ls.run(events=["endurance"])["events"]["endurance"]["E_total"]
        efficiency.set_active(EfficiencyMap.constant(ls.veh.eta_motor))
        self.assertAlmostEqual(ls.run(events=["endurance"])["events"]["endurance"]["E_total"], base, delta=1e-6 * base)
        efficiency.set_active(EfficiencyMap.constant(0.85))
        self.assertAlmostEqual(ls.run(events=["endurance"])["events"]["endurance"]["E_total"], base * 0.90 / 0.85, delta=1e-6 * base)

    def test_simulator_energy_meter_uses_the_map(self):
        from fsdsim.dynamics import VehicleModel
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.json")
            EfficiencyMap.constant(0.80).save(p)
            E = []
            for ch in ([], [f"efficiency_map={p}"]):
                vm = VehicleModel(load_all_configs(CONFIG, vehicle_overrides=ch)[0])
                vm.reset(0, 0, 0)
                vm.set_command(0.0, 6.0)
                for _ in range(200):
                    vm.step(0.01)
                E.append(vm.energy_j)
        self.assertAlmostEqual(E[1] / E[0], 0.90 / 0.80, places=6)


@NEEDS_LAPSIM
class TestGgvAndSolvers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ls = lapsim.LapSim.load(LAPSIM)

    def test_ggv_shape(self):
        veh = self.ls.veh.copy(util_x=1.0, util_y=1.0)
        t = ggv.ggv(veh, [10.0, 20.0, 30.0], 7)
        self.assertEqual(t["ax_acc"].shape, (3, 7))
        self.assertTrue(np.all(t["ax_dec"] < 0))
        self.assertTrue(np.all(np.diff(t["ay"][:, -1]) > 0))            # downforce: more lateral grip with speed
        self.assertTrue(np.all(t["ax_acc"][:, 0] >= t["ax_acc"][:, -1]))  # cornering leaves less for traction
        self.assertLess(abs(t["ax_dec"][0, -1]), abs(t["ax_dec"][0, 0]))
        from fsdsim.perf import qss
        v = 20.0
        self.assertTrue(qss.steady_ok(veh, v, 0.999 * t["ay"][1, -1] / v ** 2))
        self.assertFalse(qss.steady_ok(veh, v, 1.02 * t["ay"][1, -1] / v ** 2))

    def test_builtin_and_plugin_solvers(self):
        n = solvers.names()
        self.assertIn("qss", n)
        self.assertIn("point_mass", n)
        self.assertIn("example_cautious", n)                              # lapsim/solvers/example_solver.py
        base = self.ls.run()
        self.assertEqual(self.ls.run(solver="qss")["total"], base["total"])
        pm = self.ls.run(solver="point_mass")["events"]
        for e, r in base["events"].items():
            self.assertAlmostEqual(pm[e]["T"], r["T"], delta=0.06 * r["T"], msg=e)
        slow = self.ls.run(solver="example_cautious")["events"]
        self.assertGreater(slow["skidpad"]["T"], base["events"]["skidpad"]["T"] * 1.04)
        with self.assertRaises(KeyError):
            self.ls.run(solver="nope")

    def test_register_own_solver(self):
        def crawl(veh, k, ds, v0, v_end):
            return np.full(len(k), 5.0), np.full(len(k), 5.0), None
        solvers.register("crawl", crawl, "5 m/s everywhere")
        r = self.ls.run(events=["acceleration"], solver="crawl")["events"]["acceleration"]
        self.assertAlmostEqual(r["T"], 75.0 / 5.0, places=6)


class TestClosedLoopExports(unittest.TestCase):
    def test_marple_csv_of_a_simulator_run(self):
        from fsdsim.perf.export import COLUMNS
        with tempfile.TemporaryDirectory() as d:
            r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "closed_loop_eval.py"), "--missions", "acceleration",
                                "--csv", d], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            with open(os.path.join(d, "sim_acceleration.csv")) as fh:
                rows = list(csv.reader(fh))
        self.assertEqual(rows[0], COLUMNS)
        a = np.array(rows[1:], float)
        self.assertGreater(a[:, 4].max(), 25.0)                           # velocity
        self.assertTrue(np.all(np.diff(a[:, 1]) >= 0))                    # distance
        self.assertGreater(a[-1, 12], 1e5)                                # consumed energy [J]

    def test_power_cap_saves_energy_in_the_closed_loop(self):
        import closed_loop_eval as C
        out = []
        for ch in ([], ["P_max=15000"]):
            sim, res, _, _ = C.run("autocross", CONFIG, mode="map", verbose=False, changes=ch)
            self.assertEqual(res.state, "FINISHED")
            out.append((sim.vehicle.energy_j, res.raw_time))
        self.assertLess(out[1][0], out[0][0] * 0.9)
        self.assertGreater(out[1][1], out[0][1])


if __name__ == "__main__":
    unittest.main()
