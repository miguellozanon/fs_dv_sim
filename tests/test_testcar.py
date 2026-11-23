"""The Performance Lab on the fictional CI test car (tests/fixtures/testcar_lapsim).

The LR26 LapSim folder is confidential and not in git, so on GitHub every test that needs it is skipped. These
tests run the same code paths on an invented car and invented tracks, with checks that do not depend on the
LR26's numbers (consistency between methods, identification recovering known parameters, the web API)."""

import copy
import http.client
import json
import math
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "perf"))
sys.path.insert(0, os.path.join(ROOT, "ros2_ws", "src", "fsdsim"))
TESTCAR = os.path.join(ROOT, "tests", "fixtures", "testcar_lapsim")
G = 9.81
try:
    import casadi  # noqa: F401

    HAVE_CASADI = True
except ImportError:
    HAVE_CASADI = False


class TestCarCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from perflab import logs
        from perflab.lab import Lab

        cls.tmp = tempfile.mkdtemp(prefix="testcar-")
        cls.lab = Lab(TESTCAR, configs_dir=os.path.join(cls.tmp, "configs"))
        cls._uploads = logs.UPLOADS
        logs.UPLOADS = os.path.join(cls.tmp, "uploads")
        os.makedirs(logs.UPLOADS)

    @classmethod
    def tearDownClass(cls):
        from perflab import logs

        logs.UPLOADS = cls._uploads
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def write_csv(self, name, cols: dict):
        from perflab import logs

        keys = list(cols)
        with open(os.path.join(logs.UPLOADS, name), "w") as f:
            f.write(",".join(keys) + "\n")
            for row in zip(*[cols[k] for k in keys]):
                f.write(",".join(f"{v:.9g}" for v in row) + "\n")
        return f"upload:{name}"


class TestLapSimOnTheTestCar(TestCarCase):
    def test_events_and_studies(self):
        from perflab import energy, studies

        r = self.lab.run([])
        self.assertEqual(set(r["events"]), {"acceleration", "skidpad", "autocross", "endurance"})
        for e in r["events"].values():
            self.assertTrue(math.isfinite(e["T"]) and e["T"] > 0)
            self.assertTrue(0 < e["points"] <= 250)
        lighter = self.lab.run(["m=280"], traces=False)
        self.assertGreater(lighter["total"], r["total"])                   # 20 kg lighter: more points
        sens = studies.sensitivity(self.lab, ["m", "muy"], 0.05)
        self.assertEqual(len(sens["rows"]), 2)
        en = energy.analyse(self.lab, [], None, [20, 30])
        b = en["breakdown"]
        self.assertGreater(b["battery_out"], 0)
        rg = energy.regen_study(self.lab, [], [0, 10, 20], 1.0)
        e = [w["E_total_kwh"] for w in rg["rows"]]
        self.assertTrue(e[0] > e[1] > e[2])                                # more regen, less energy
        self.assertEqual(len({round(w["T"], 6) for w in rg["rows"]}), 1)   # the lap time does not change
        pts = [w["budget"]["points"] for w in rg["rows"]]
        self.assertGreaterEqual(pts[2], pts[0])                            # within a budget: a higher cap
        with self.assertRaises(ValueError):
            energy.regen_study(self.lab, [], [-5])

    def test_transient_lap(self):
        from perflab import transient

        circle = np.c_[10 * np.cos(np.linspace(0, 2 * np.pi, 200, endpoint=False)),
                       10 * np.sin(np.linspace(0, 2 * np.pi, 200, endpoint=False))]
        noisy = circle + np.random.default_rng(3).normal(0, 0.05, circle.shape)
        sm, lam = transient.smooth_line(noisy, True, 0.1)
        self.assertLessEqual(float(np.max(np.hypot(*(sm - noisy).T))), 0.1 + 1e-9)
        self.assertLess(float(np.std(np.hypot(*sm.T))), float(np.std(np.hypot(*noisy.T))))   # rounder than the points
        self.assertGreater(lam, 0)

        r = transient.run(self.lab, "skidpad")
        self.assertEqual(r["status"], "finished")
        self.assertGreater(r["T"], r["qss"]["T"])                         # a driver on a dynamic car is slower
        self.assertLess(r["T"], 1.15 * r["qss"]["T"])


@unittest.skipUnless(HAVE_CASADI, "casadi not installed (optional package)")
class TestOcpOnTheTestCar(TestCarCase):
    def test_point_mass_matches_qss_and_dynamic_models_converge(self):
        from perflab import ocp

        for ev in ("skidpad", "autocross"):
            r = ocp.run(self.lab, ev, "point_mass")
            self.assertTrue(r["ok"], r["stages"])
            self.assertLess(abs(r["t_lap"] / r["qss"]["t_lap_same_grid"] - 1), 0.01, ev)
        st = ocp.run(self.lab, "skidpad", "single_track")
        tv = ocp.run(self.lab, "skidpad", "double_track", torque_vectoring=True)
        self.assertTrue(st["ok"] and tv["ok"])
        self.assertGreater(st["t_lap"], ocp.run(self.lab, "skidpad", "point_mass")["t_lap"])

    def test_methods_agree_on_a_lighter_car(self):
        from perflab import methods

        r = methods.compare(self.lab, "skidpad", "m-10", ["qss", "qss_point_mass", "ocp_point_mass", "ocp_single_track"])
        self.assertTrue(r["agree"], r["signs"])
        mc = methods.monte_carlo(self.lab, 10, "m-10", seed=2)
        self.assertEqual(mc["gain"]["p_gain"], 1.0)


class TestIdentification(TestCarCase):
    def test_coastdown_recovers_drag_and_rolling(self):
        from perflab import ident

        veh = self.lab.lapsim().vehicle([])
        fr, cw = 0.020, 1.30                               # the "true" car differs from the setup (0.015, 1.20)
        meff, kz = veh.m * veh.e_i, 0.5 * veh.rho * veh.c_z * veh.A * veh.aero_eff
        t, v, h = [0.0], [24.0], 0.01
        while v[-1] > 5:
            a = (fr * (veh.m * G + kz * v[-1] ** 2) + 0.5 * veh.rho * cw * veh.A * veh.aero_eff * v[-1] ** 2) / meff
            v.append(v[-1] - a * h)
            t.append(t[-1] + h)
        rng = np.random.default_rng(0)
        t, v = np.array(t[::2]), np.array(v[::2]) * 3.6 + rng.normal(0, 0.1, len(t[::2]))
        src = self.write_csv("coast.csv", {"time_s": t, "speed_kmh": v, "brake_bar": np.zeros(len(t))})
        r = ident.coastdown(self.lab, src, {"time": "time_s", "speed": {"channel": "speed_kmh", "unit": "kmh"},
                                            "brake": {"channel": "brake_bar"}})
        self.assertAlmostEqual(r["f_roll"], fr, delta=0.002)
        self.assertAlmostEqual(r["c_w"], cw, delta=0.05)
        self.assertEqual(r["warnings"], [])

    def test_steady_cornering_recovers_the_grip_scale(self):
        from fsdsim.perf import ggv

        from perflab import ident

        veh = self.lab.lapsim().vehicle([])
        car = veh.copy(util_x=1, util_y=1, mux=veh.mux * 0.9, muy=veh.muy * 0.9)
        R = 15.0
        vmax = max(x for x in np.linspace(3, 25, 400) if ggv.ay_limit(car, float(x)) >= x * x / R)
        t = np.arange(0, 60, 0.02)
        v = np.minimum(5 + 0.5 * t, vmax)
        src = self.write_csv("circle.csv", {"time_s": t, "speed_kmh": v * 3.6, "yaw_degps": np.degrees(v / R)})
        r = ident.steady(self.lab, src, {"time": "time_s", "speed": {"channel": "speed_kmh", "unit": "kmh"},
                                         "yaw": {"channel": "yaw_degps", "unit": "degps"}})
        self.assertAlmostEqual(r["mu_scale"], 0.9, delta=0.02)
        r = ident.steady(self.lab, src, {"time": "time_s", "speed": {"channel": "speed_kmh", "unit": "kmh"}}, radius=R)
        self.assertAlmostEqual(r["mu_scale"], 0.9, delta=0.02)

    def test_grip_use_of_a_simulated_lap(self):
        from perflab import ident

        e = self.lab.raw([], None, ["autocross"])["events"]["autocross"]      # the QSS lap at util 0.9 / 0.8
        src = self.write_csv("qss_lap.csv", {"time_s": e["t"], "speed_kmh": e["v"] * 3.6, "ax_g": e["ax"] / G,
                                             "ay_g": e["ay"] / G})
        r = ident.grip_use(self.lab, src, {"time": "time_s", "speed": {"channel": "speed_kmh", "unit": "kmh"},
                                           "ax": {"channel": "ax_g", "unit": "g"}, "ay": {"channel": "ay_g", "unit": "g"}})
        self.assertAlmostEqual(r["util_y"], 0.8, delta=0.05)
        self.assertIsNotNone(r["util_x_brake"])
        self.assertAlmostEqual(r["util_x_brake"], 0.9, delta=0.08)

    def test_track_from_gps_gives_the_same_lap(self):
        from fsdsim.perf import lapsim as LS

        from perflab import ident

        ls = self.lab.lapsim()
        ev = ls.competition([]).events["endurance"]
        e = self.lab.raw([], None, ["endurance"])["events"]["endurance"]
        tr, T = ev.track, e["t"][-1]
        tt = np.arange(0, 1.6 * T, 0.1)                                   # 10 Hz GNSS, 1.6 laps, 0.3 m noise
        ss = np.interp(tt % T, e["t"], e["s"])
        rng = np.random.default_rng(1)
        x = np.interp(ss, tr.s, tr.x) + rng.normal(0, 0.3, len(tt))
        y = np.interp(ss, tr.s, tr.y) + rng.normal(0, 0.3, len(tt))
        lat = 49.33 + np.degrees(y / 6371008.8)
        lon = 8.57 + np.degrees(x / (6371008.8 * math.cos(math.radians(49.33))))
        src = self.write_csv("gps.csv", {"time_s": tt, "speed_kmh": np.interp(tt % T, e["t"], e["v"]) * 3.6,
                                         "lat_deg": lat, "lon_deg": lon})
        spec = {"time": "time_s", "speed": {"channel": "speed_kmh", "unit": "kmh"},
                "lat": {"channel": "lat_deg", "unit": "deg"}, "lon": {"channel": "lon_deg", "unit": "deg"}}
        g = ident.track_from_gps(self.lab, src, spec)
        self.assertTrue(g["closed"])
        self.assertEqual(g["turns"], 1.0)
        self.assertAlmostEqual(g["L"], tr.L, delta=0.01 * tr.L)
        path = ident.save_track(self.lab, src, spec, "measured_endurance")
        with self.assertRaises(ValueError):
            ident.save_track(self.lab, src, spec, "measured_endurance")       # no silent overwrite
        with self.assertRaises(ValueError):
            ident.save_track(self.lab, src, spec, "../evil")
        # the true line through the same x, y pipeline against the measured one: same lap time within 1.5 %
        veh = LS.event_vehicle(ls.vehicle([]), ev)
        true_xy = os.path.join(self.tmp, "true_xy.csv")
        np.savetxt(true_xy, np.c_[tr.x[:-1], tr.y[:-1]], delimiter=",", header="x_m,y_m", comments="")
        t_true = LS.run_event(veh, dataclass_replace(ev, LS.track_csv(true_xy, True, 0.5)))["t_lap"]
        t_meas = LS.run_event(veh, dataclass_replace(ev, LS.track_csv(path, True, 0.5)))["t_lap"]
        self.assertLess(abs(t_meas / t_true - 1), 0.015, (t_meas, t_true))


def dataclass_replace(ev, track):
    e = copy.copy(ev)
    e.track = track
    return e


class TestAppOnTheTestCar(TestCarCase):
    def test_http_api(self):
        import app as perf_app

        a = perf_app.App(self.lab)
        srv = perf_app.ThreadingHTTPServer(("127.0.0.1", 0), perf_app.make_handler(a, lambda: None))
        th = threading.Thread(target=srv.serve_forever, daemon=True)
        th.start()
        try:
            def req(method, path, body=None):
                c = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=60)
                c.request(method, path, json.dumps(body) if body is not None else None,
                          {"Content-Type": "application/json", "X-Perf-Lab": "1"})
                r = c.getresponse()
                return r.status, json.loads(r.read() or b"null")

            st, info = req("GET", "/api/info")
            self.assertEqual(st, 200)
            self.assertIn("competition_test.m", info["competitions"])
            st, r = req("POST", "/api/run", {"changes": []})
            self.assertTrue(r["ok"])
            src = self.write_csv("api_coast.csv", {"time_s": np.arange(0, 20, 0.05),
                                                   "speed_kmh": 80 * np.exp(-np.arange(0, 20, 0.05) / 25)})
            st, r = req("POST", "/api/job", {"kind": "ident", "args": {
                "tool": "coastdown", "source": src,
                "spec": {"time": "time_s", "speed": {"channel": "speed_kmh", "unit": "kmh"}}}})
            self.assertEqual(st, 200, r)
            jid = r["job"]["id"]
            for _ in range(200):
                st, j = req("GET", f"/api/job/{jid}")
                if j["state"] not in ("queued", "running"):
                    break
                time.sleep(0.05)
            self.assertEqual(j["state"], "done", j)
            self.assertGreater(j["result"]["c0"], 0)
        finally:
            srv.shutdown()
            srv.server_close()


if __name__ == "__main__":
    unittest.main()
