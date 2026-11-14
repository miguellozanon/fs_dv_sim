"""perf/: the Performance Lab (stand-alone lap time simulation web app)."""

import http.client
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "perf"))
sys.path.insert(0, os.path.join(ROOT, "tests"))
LAPSIM = os.path.join(ROOT, "lapsim", "LR26_LapSim")
HAVE_LAPSIM = os.path.isfile(os.path.join(LAPSIM, "config", "lr26_params.m"))


@unittest.skipUnless(HAVE_LAPSIM, "lapsim/LR26_LapSim not present (confidential, not in git)")
class PerfTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from perflab.lab import Lab

        cls.tmp = tempfile.mkdtemp(prefix="perf-test-")
        cls.cfg = os.path.join(cls.tmp, "configs")
        os.makedirs(cls.cfg)
        cls.lab = Lab(LAPSIM, configs_dir=cls.cfg)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)


class TestLab(PerfTestCase):
    def test_run_matches_the_lapsim_port(self):
        from fsdsim.perf.lapsim import LapSim

        r = self.lab.run(["m=270", "autocross.util_y=0.85"])
        ref = LapSim.load(LAPSIM).run(["m=270", "autocross.util_y=0.85"])
        self.assertAlmostEqual(r["total"], ref["total"], places=9)
        self.assertAlmostEqual(r["events"]["autocross"]["T"], ref["events"]["autocross"]["T"], places=9)
        tr = r["events"]["endurance"]["trace"]
        self.assertLessEqual(len(tr["s"]), 1500)
        self.assertEqual(len(tr["s"]), len(tr["v"]))
        json.dumps(r, allow_nan=False)                         # NaN-free for the browser

    def test_validate(self):
        self.assertEqual(self.lab.validate([" m*1.1 ", "", "yaw_inertia=150"]), ["m*1.1", "yaw_inertia=150"])
        for bad in (["foo=1"], ["m=abc"], ["autocross.util_q=1"], ["m"]):
            with self.assertRaises(ValueError, msg=bad):
                self.lab.validate(bad)

    def test_info_and_competitions(self):
        info = self.lab.info()
        self.assertIn("competition_fsa2025_real.m", info["competitions"])
        keys = [r["key"] for g in info["groups"] for r in g["rows"]]
        self.assertIn("i_trans", keys)
        self.assertIn("endurance.util_y", keys)
        real = self.lab.run([], "competition_fsa2025_real.m", traces=False)
        self.assertGreater(real["total"], 100)
        with self.assertRaises(ValueError):
            self.lab.lapsim("competition_nowhere.m")

    def test_setups_are_shared_files(self):
        where = self.lab.save_setup("Test wing v2", ["c_z=4.4", "m-2"], "note")
        self.assertTrue(where.endswith("Test_wing_v2.json"))
        names = [s["name"] for s in self.lab.setups()]
        self.assertIn("Test wing v2", names)
        with self.assertRaises(FileExistsError):
            self.lab.save_setup("Test wing v2", ["c_z=4.5"])
        self.lab.save_setup("Test wing v2", ["c_z=4.5"], overwrite=True)
        self.assertEqual([s for s in self.lab.setups() if s["name"] == "Test wing v2"][0]["changes"], ["c_z=4.5"])
        for bad in ("../x", "", "a/b"):
            with self.assertRaises(ValueError):
                self.lab.save_setup(bad, [])
        with self.assertRaises(ValueError):
            self.lab.save_setup("ok name", ["nonsense=1"])
        # efficiency maps are not setups
        with open(os.path.join(self.cfg, "eta_map_x.json"), "w") as f:
            json.dump({"torque_Nm": [0, 1], "rpm": [0, 1], "eta": [[0.9, 0.9], [0.9, 0.9]]}, f)
        self.assertNotIn("eta_map_x", [s["name"] for s in self.lab.setups()])
        self.assertIn("eta_map_x.json", self.lab.eta_maps())
        with self.assertRaises(ValueError):
            self.lab.set_eta_map("eta_map_missing.json")


class TestStudies(PerfTestCase):
    def test_sensitivity(self):
        from perflab import studies

        r = studies.sensitivity(self.lab, ["mu", "m", "drive_awd"], 0.1)
        rows = {x["param"]: x for x in r["rows"]}
        self.assertEqual(rows["mu"]["better"], "+")
        self.assertEqual(rows["m"]["better"], "-")
        self.assertGreater(rows["mu"]["points_per_pct"], rows["m"]["points_per_pct"])
        self.assertTrue(rows["drive_awd"]["no_effect"])                   # 0 * 1.1 = 0
        self.assertIn("points_per_kg", rows["m"])
        self.assertGreater(rows["m"]["dT_plus"]["autocross"], 0)          # heavier = slower
        self.assertEqual(r["rows"][0]["param"], "mu")                     # sorted by effect
        with self.assertRaises(ValueError):
            studies.sensitivity(self.lab, ["m"], 1.5)

    def test_sweeps_and_compare(self):
        from perflab import studies

        self.assertEqual(studies.values_from("1, 2;3"), [1.0, 2.0, 3.0])
        self.assertEqual(len(studies.values_from({"from": 1, "to": 2, "n": 5})), 5)
        with self.assertRaises(ValueError):
            studies.values_from({"from": 1, "to": 2, "n": 100})
        r = studies.sweep(self.lab, "i_trans", [12, 14.5, 19], event_traces="autocross")
        self.assertEqual(r["base_value"], 14.5)
        self.assertGreater(r["rows"][2]["total"], r["rows"][0]["total"])  # documentation Table 2.2: 19 beats 14.5
        self.assertIn("trace", r["rows"][0])
        m = studies.sweep2d(self.lab, "c_z", [3.5, 4.5], "m", [260, 300])
        self.assertEqual(m["best"]["v2"], 260)
        self.assertEqual(len(m["total"]), 2)
        c = studies.compare(self.lab, [{"name": "base", "changes": []}, {"name": "g19", "changes": ["i_trans=19"]}])
        self.assertGreater(c["setups"][1]["total"], c["setups"][0]["total"])
        self.assertIn("autocross", c["setups"][0]["traces"])


class TestEnergy(PerfTestCase):
    def test_breakdown_and_budget(self):
        from perflab import energy

        r = energy.analyse(self.lab, [], budget_kwh=5.0, caps_kw=[10, 14, 18, 30])
        b = r["breakdown"]
        # flying lap: wheel work = drag + rolling + braking
        self.assertAlmostEqual(b["wheel_drive"], b["drag"] + b["rolling"] + b["friction_brakes"] + b["regen_at_wheel"], delta=0.01 * b["wheel_drive"])
        self.assertAlmostEqual(b["drivetrain_losses"], b["battery_out"] - b["wheel_drive"])
        self.assertTrue(r["budget"]["limited"])
        self.assertLess(r["budget"]["cap_kw"], r["cap_kw"])
        self.assertAlmostEqual(r["budget"]["E_total_kwh"], 5.0, delta=0.02)
        pts = [x["points"] for x in r["sweep"]]
        self.assertEqual(pts, sorted(pts))                               # more power, more endurance points
        g = energy.ggv_map(self.lab, [], "")
        self.assertGreater(g["ay_g"][-1][-1], g["ay_g"][0][-1])         # downforce: more grip at speed
        ge = energy.ggv_map(self.lab, [], "autocross")
        self.assertLess(ge["ay_g"][0][-1], g["ay_g"][0][-1])            # driver utilisation < 1


class TestTransient(PerfTestCase):
    def test_close_loop(self):
        from perflab.transient import close_loop

        th = np.linspace(0, 2 * np.pi, 201)[:-1]
        circle = np.stack([10 * np.cos(th), 10 * np.sin(th)], 1)
        xy, gap = close_loop(np.vstack([circle, circle[:1]]))
        self.assertEqual(gap, 0.0)
        self.assertEqual(len(xy), 200)
        drift = circle + np.outer(np.arange(200) / 200, [5.0, 0.0])        # does not close: 5 m gap
        xy, gap = close_loop(drift)
        self.assertAlmostEqual(gap, 5.0, delta=0.4)
        step = xy[-1] + (xy[-1] - xy[-2])
        self.assertLess(np.linalg.norm(step - xy[0]), 0.1)

    def test_events(self):
        from perflab import transient

        sk = transient.run(self.lab, "skidpad")
        self.assertEqual(sk["status"], "finished")
        self.assertGreater(sk["t_lap"], sk["qss"]["t_lap"])             # the driver keeps a margin
        self.assertLess(sk["t_lap"], 1.15 * sk["qss"]["t_lap"])
        self.assertEqual(sk["off_track_s"], 0.0)
        acc = transient.run(self.lab, "acceleration", ["yaw_inertia=150"])
        self.assertEqual(acc["status"], "finished")
        self.assertAlmostEqual(acc["T"], acc["qss"]["T"], delta=0.35)
        self.assertEqual(acc["driver"]["yaw_inertia"], 150.0)
        json.dumps(acc, allow_nan=False)
        with self.assertRaises(ValueError):
            transient.run(self.lab, "nowhere")


class TestLogs(PerfTestCase):
    def setUp(self):
        from perflab import logs

        self.logs = logs
        self.up = tempfile.mkdtemp(prefix="perf-up-")
        self.patch = mock.patch.object(logs, "UPLOADS", self.up)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        shutil.rmtree(self.up, ignore_errors=True)

    def test_reference_trace(self):
        src = [s["id"] for s in self.logs.sources(self.lab) if "autocross" in s["id"]][0]
        info = self.logs.inspect(self.lab, src)
        self.assertEqual(info["suggest"]["speed"], "v_kmh")
        r = self.logs.correlate(self.lab, src, {"speed": {"channel": "v_kmh", "unit": "kmh"}, "distance": "s_m"}, "autocross")
        self.assertLess(r["rmse_v_kmh"], 6.0)                           # the documentation's own trace
        self.assertLess(abs(r["t_log"] - r["t_sim"]), 1.5)
        self.assertGreater(r["coverage"], 0.95)

    def test_uploaded_csv_with_time(self):
        # the simulated autocross lap written as a 20 Hz log, shifted by 12 m: the search finds the offset
        e = self.lab.raw([], None, ["autocross"])["events"]["autocross"]
        t = np.arange(0, e["t"][-1], 0.05)
        s = np.interp(t, e["t"], e["s"])
        keep = s >= 12.0
        v = np.interp(s, e["s"], e["v"]) * 3.6
        ax = np.interp(s, e["s"], e["ax"]) / 9.81
        rows = "time_s,v_kmh,ax_g\n" + "".join(f"{a:.3f},{b:.4f},{c:.4f}\n" for a, b, c in zip(t[keep], v[keep], ax[keep]))
        p = self.logs.save_upload("lap.csv", rows.encode())
        self.assertEqual(os.path.dirname(p), self.up)
        spec = {"speed": {"channel": "v_kmh", "unit": "kmh"}, "ax": {"channel": "ax_g", "unit": "g"}, "time": "time_s"}
        r = self.logs.correlate(self.lab, "upload:lap.csv", spec, "autocross")
        self.assertAlmostEqual(r["shift_m"], 12.0, delta=1.0)
        self.assertLess(r["rmse_v_kmh"], 1.0)
        self.assertLess(r["rmse_ax_g"], 0.1)
        w = self.logs.extract(self.lab, "upload:lap.csv", {**spec, "t0": 5, "t1": 10})
        self.assertAlmostEqual(w["t"][-1], 5.0, delta=0.06)
        with self.assertRaises(ValueError):
            self.logs.save_upload("evil.exe", b"x")
        with self.assertRaises(ValueError):
            self.logs.resolve(self.lab, "elsewhere:/etc/passwd")
        with self.assertRaises(FileNotFoundError):
            self.logs.resolve(self.lab, "upload:../../../etc/passwd")

    def test_powertrain_mf4_and_map(self):
        from test_lionslts import TestPowertrainLog, write_mdf

        helper = TestPowertrainLog()
        t, d = helper.synth(decel=0.3)
        ch = {}
        for key, name, fac in (("nA", "act_Speed_R_DCUA", 1.0), ("nB", "act_Speed_R_DCUB", 1.0), ("TA", "act_Torque_R_DCUA", 0.1),
                               ("TB", "act_Torque_R_DCUB", 0.1), ("PA", "act_Power_R_DCUA", 0.01), ("PB", "act_Power_R_DCUB", 0.01),
                               ("thr", "PDB_Pedal_Throttle", 1.0), ("bpf", "PDB_Pedal_BPF", 1.0), ("bpr", "PDB_Pedal_BPR", 1.0)):
            ch[name] = (np.round(d[key] / fac).astype(np.uint16), "", (0.0, fac))
        write_mdf(os.path.join(self.up, "car.mf4"), [("CAN3 message", t, ch)])
        info = self.logs.inspect(self.lab, "upload:car.mf4")
        self.assertEqual(info["powertrain"], 9)
        r = self.logs.powertrain(self.lab, "upload:car.mf4")
        self.assertAlmostEqual(r["eff_median"], helper.EFF, delta=0.01)
        self.assertIn("consistent with a car", r["rollout_verdict"])
        where = self.logs.save_eta_map(self.lab, "upload:car.mf4", "test1")
        self.assertTrue(os.path.isfile(os.path.join(ROOT, where)) or os.path.isfile(os.path.join(self.cfg, "eta_map_test1.json")))
        with self.assertRaises(FileExistsError):
            self.logs.save_eta_map(self.lab, "upload:car.mf4", "test1")
        # the map changes the energy, and the lab can use it
        base = self.lab.run([], traces=False)["events"]["endurance"]["E_total_kwh"]
        self.lab.set_eta_map("eta_map_test1.json")
        try:
            other = self.lab.run([], traces=False)["events"]["endurance"]["E_total_kwh"]
        finally:
            self.lab.set_eta_map("")
            from fsdsim.perf import efficiency
            efficiency.set_active(None)
        self.assertNotAlmostEqual(base, other, places=3)
        # motor speed as vehicle speed
        spec = {"speed": {"channel": "act_Speed_R_DCUA", "unit": "rpm_motor"}}
        x = self.logs.extract(self.lab, "upload:car.mf4", spec)
        veh = self.lab.lapsim().veh
        self.assertAlmostEqual(float(np.nanmax(x["v"])), 12000 / veh.i_trans * veh.circ_ref / 60, delta=0.5)


class TestJobs(unittest.TestCase):
    def test_progress_failure_cancel(self):
        from perflab.jobs import Runner

        r = Runner()

        def work(job, n):
            for i in range(n):
                job.step(i, n, f"{i}")
                time.sleep(0.01)
            return n * 2

        j = r.submit("x", work, {"n": 5})
        self.assertEqual(r.wait(j.id).result, 10)
        bad = r.submit("x", lambda job: (_ for _ in ()).throw(ValueError("no good")), {})
        self.assertEqual(r.wait(bad.id).state, "failed")
        self.assertEqual(bad.error, "no good")
        long = r.submit("x", work, {"n": 500})
        time.sleep(0.05)
        self.assertTrue(r.cancel(long.id))
        self.assertEqual(r.wait(long.id).state, "cancelled")
        self.assertFalse(r.cancel(long.id))


@unittest.skipUnless(HAVE_LAPSIM, "lapsim/LR26_LapSim not present (confidential, not in git)")
class TestHttp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from http.server import ThreadingHTTPServer

        import app as perf_app
        from perflab.lab import Lab

        cls.tmp = tempfile.mkdtemp(prefix="perf-http-")
        cls.app = perf_app.App(Lab(LAPSIM, configs_dir=cls.tmp))
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), perf_app.make_handler(cls.app, lambda: None))
        cls.port = cls.httpd.server_address[1]
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def req(self, method, path, body=None, header=True):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=60)
        h = {"Content-Type": "application/json"}
        if header:
            h["X-Perf-Lab"] = "1"
        c.request(method, path, json.dumps(body) if body is not None else None, h)
        r = c.getresponse()
        data = r.read()
        return r.status, (json.loads(data) if r.getheader("Content-Type", "").startswith("application/json") else data)

    def test_pages_and_api(self):
        st, page = self.req("GET", "/")
        self.assertEqual(st, 200)
        self.assertIn(b"Performance Lab", page)
        for f in ("/static/perf.js", "/static/charts.js", "/static/perf.css", "/brand/style.css"):
            self.assertEqual(self.req("GET", f)[0], 200, f)
        self.assertEqual(self.req("GET", "/static/../app.py")[0], 404)
        st, info = self.req("GET", "/api/info")
        self.assertEqual(st, 200)
        self.assertIn("groups", info)
        self.assertEqual(self.req("POST", "/api/run", {"changes": []}, header=False)[0], 403)
        st, r = self.req("POST", "/api/run", {"changes": ["m=270"]})
        self.assertTrue(r["ok"])
        self.assertGreater(r["result"]["total"], 164.0)
        st, r = self.req("POST", "/api/run", {"changes": ["bogus=1"]})
        self.assertEqual(st, 400)
        self.assertIn("unknown parameter", r["message"])
        st, r = self.req("POST", "/api/job", {"kind": "sweep", "args": {"param": "m", "values": [270, 290]}})
        jid = r["job"]["id"]
        for _ in range(200):
            st, j = self.req("GET", f"/api/job/{jid}")
            if j["state"] not in ("queued", "running"):
                break
            time.sleep(0.05)
        self.assertEqual(j["state"], "done", j)
        self.assertEqual(len(j["result"]["rows"]), 2)
        st, r = self.req("POST", "/api/job", {"kind": "rm -rf", "args": {}})
        self.assertEqual(st, 400)
        st, r = self.req("POST", "/api/ggv", {"changes": []})
        self.assertTrue(r["ok"])
        st, m = self.req("GET", "/api/methods")
        self.assertEqual(st, 200)
        self.assertIn("qss", [x["name"] for x in m["methods"]])
        self.assertIn("available", m["ocp"])
        st, r = self.req("POST", "/api/job", {"kind": "methods", "args": {"event": "skidpad", "change": "mass-5",
                                                                          "methods": ["qss", "qss_point_mass"]}})
        jid = r["job"]["id"]
        for _ in range(200):
            st, j = self.req("GET", f"/api/job/{jid}")
            if j["state"] not in ("queued", "running"):
                break
            time.sleep(0.05)
        self.assertEqual(j["state"], "done", j)
        self.assertTrue(j["result"]["agree"])


class TestFiles(unittest.TestCase):
    def test_bat_and_schema(self):
        with open(os.path.join(ROOT, "perf", "perf.bat"), "rb") as f:
            raw = f.read()
        self.assertIn(b"\r\n", raw)
        self.assertNotIn(b"\n", raw.replace(b"\r\n", b""))              # CRLF only
        self.assertIn(b"app.py", raw)
        sys.path.insert(0, os.path.join(ROOT, "ros2_ws", "src", "fsdsim"))
        from fsdsim.perf.schema import LAPSIM_SCHEMA

        titles = [g[0] for g in LAPSIM_SCHEMA]
        self.assertIn("Tyres", titles)
        with open(os.path.join(ROOT, ".gitignore")) as f:
            self.assertIn("perf/uploads/", f.read())


if __name__ == "__main__":
    unittest.main()
