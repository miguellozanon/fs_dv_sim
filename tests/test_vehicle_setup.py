"""LapSim folder as single source, LapSim event port vs MATLAB, run-time vehicle swap, MATLAB runner,
and the dashboard's Vehicle setup (API + backend)."""

import json
import math
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
import urllib.request

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "ros2_ws", "src")
sys.path[:0] = [os.path.join(SRC, "fsdsim"), os.path.join(SRC, "fsd_dashboard"), os.path.join(SRC, "fsd_sim_bridge")]

from fsdsim.config import load_all_configs  # noqa: E402
from fsdsim.perf import lapsim  # noqa: E402
from fsdsim.sim import Simulation  # noqa: E402

CONFIG = os.path.join(SRC, "fsd_sim_bringup", "config")
LAPSIM = os.path.join(ROOT, "lapsim", "LR26_LapSim")
SWEEPS = json.load(open(os.path.join(ROOT, "tests", "data", "lapsim_lr26_sweeps.json")))["sweeps"]
FAKE_MATLAB = os.path.join(ROOT, "tests", "fake_matlab.py")
try:  # the runner starts it as an executable; a copy from Windows or a zip loses the x bit
    os.chmod(FAKE_MATLAB, os.stat(FAKE_MATLAB).st_mode | 0o111)
except OSError:
    pass


class TestLapSimFolder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ls = lapsim.LapSim.load(LAPSIM)

    def test_competition_file_is_read(self):
        ev = self.ls.comp.events
        self.assertEqual(list(ev), ["acceleration", "skidpad", "autocross", "endurance"])
        self.assertEqual((ev["skidpad"].util_x, ev["skidpad"].util_y, ev["skidpad"].timing), (1.0, 0.8, "flying_lap"))
        self.assertEqual((ev["endurance"].laps, ev["endurance"].track.closed), (16, True))
        self.assertAlmostEqual(ev["autocross"].score["Tmax"], 63.9221)
        self.assertGreater(ev["endurance"].track.L, 1200)

    def test_baseline_matches_matlab_and_documentation(self):
        r = self.ls.run()
        T = {n: x["T"] for n, x in r["events"].items()}
        self.assertAlmostEqual(T["acceleration"], 4.66192, delta=1e-4)
        self.assertAlmostEqual(T["skidpad"], 5.15427, delta=1e-4)
        self.assertAlmostEqual(T["autocross"], 49.94441, delta=1e-4)
        self.assertAlmostEqual(T["endurance"], 1526.96709, delta=1e-3)
        self.assertAlmostEqual(r["total"], 164.35, delta=0.01)       # documentation: 164.46 (fitted scoring)

    def test_all_events_match_matlab_sweeps(self):
        """Every 3rd value of LapSim's own sensitivity sweeps (lts_results_sensitivities.mat), all 4 events."""
        worst = 0.0
        for name, sw in SWEEPS.items():
            for val, T in list(zip(sw["values"], sw["T"]))[::3]:
                res = self.ls.run([f"{name}={val!r}"])["events"]
                for i, e in enumerate(lapsim.EVENT_ORDER):
                    worst = max(worst, abs(res[e]["T"] - T[i]))
        self.assertLess(worst, 1e-4, "max difference to MATLAB [s]")

    def test_event_changes(self):
        base = self.ls.run(events=["autocross"])["events"]["autocross"]["T"]
        r = self.ls.run(["autocross.util_y=0.9"], events=["autocross"])["events"]["autocross"]["T"]
        self.assertLess(r, base - 0.5)
        self.assertEqual(lapsim.split_changes(["m=1", "autocross.util_y*0.9", "actuators.max_steer=0.3"]),
                         (["m=1", "actuators.max_steer=0.3"], ["autocross.util_y*0.9"]))
        self.assertEqual(lapsim.matlab_change("skidpad.util_y=0.85"), "comp.events.skidpad.util_y = 0.85;")
        self.assertTrue(lapsim.matlab_change("m=1\nsystem('x')").startswith("% (not understood)"))
        self.assertTrue(lapsim.matlab_change("yaw_inertia=100").startswith("% (simulator only"))

    def test_simulator_reads_the_matlab_files(self):
        """Editing lr26_params.m changes the simulator's vehicle (single source of truth)."""
        with tempfile.TemporaryDirectory() as d:
            root = os.path.join(d, "lapsim", "LR26_LapSim")
            shutil.copytree(LAPSIM, root)
            cfg = os.path.join(d, "config")
            shutil.copytree(CONFIG, cfg)
            pf = os.path.join(root, "config", "lr26_params.m")
            text = open(pf).read().replace("veh.m         = 284;", "veh.m         = 251;")
            open(pf, "w").write(text)
            old = os.environ.pop("FSG_LAPSIM_DIR", None)
            try:
                veh = load_all_configs(cfg)[0]
            finally:
                if old is not None:
                    os.environ["FSG_LAPSIM_DIR"] = old
        self.assertEqual(veh.mass, 251.0)
        self.assertIn("lr26_params.m", veh.performance["source"])
        self.assertEqual(veh.perf().events["autocross"], {"util_x": 0.9, "util_y": 0.8})

    def test_repo_config_uses_the_lapsim_folder(self):
        veh = load_all_configs(CONFIG)[0]
        self.assertTrue(veh.performance["source"].startswith(os.path.join(ROOT, "lapsim")))
        self.assertAlmostEqual(veh.wheelbase, 0.85065 + 0.75435)


class TestVehicleSwap(unittest.TestCase):
    def test_simulation_set_vehicle_restarts_the_run(self):
        v, s, c = load_all_configs(CONFIG)
        sim = Simulation(v, s, c, mission="acceleration", enable_lidar=False, enable_camera=False)
        for _ in range(30):
            sim.step(0.1)
        v2 = load_all_configs(CONFIG, ["m=320", "yaw_inertia=150"])[0]
        sim.set_vehicle(v2)
        self.assertEqual(sim.t, 0.0)
        self.assertEqual(sim.vp.mass, 320.0)
        self.assertEqual(sim.vehicle.p.yaw_inertia, 150.0)

    def test_relay_and_isaac_handle_vehicle_messages(self):
        relay = open(os.path.join(SRC, "fsd_sim_bridge", "fsd_sim_bridge", "isaac_relay_node.py")).read()
        self.assertIn('elif kind == "vehicle":', relay)
        self.assertIn('"cmd": "vehicle"', relay)
        isaac = open(os.path.join(ROOT, "isaac", "run_isaac.py")).read()
        self.assertIn('elif kind == "vehicle":', isaac)
        self.assertIn("sim.set_vehicle(new)", isaac)
        ri = open(os.path.join(SRC, "fsd_sim_bridge", "fsd_sim_bridge", "ros_interface.py")).read()
        self.assertIn('"/sim/vehicle_set"', ri)


class TestMatlabRunner(unittest.TestCase):
    def test_job_with_fake_matlab(self):
        from fsd_dashboard import matlab_runner as mr

        old = os.environ.get("FSG_MATLAB")
        os.environ["FSG_MATLAB"] = FAKE_MATLAB
        try:
            with tempfile.TemporaryDirectory() as d:
                r = mr.MatlabRunner(LAPSIM)
                r.workdir = d
                self.assertTrue(r.available)
                r.submit(["m=270", "mu*1.05", "autocross.util_y=0.85", "yaw_inertia=100"])
                for _ in range(200):
                    if r.status()["state"] in ("done", "error"):
                        break
                    time.sleep(0.1)
                st = r.status()
                job = open(os.path.join(d, "jobs", "fsg_job_1.m")).read()
        finally:
            if old is None:
                os.environ.pop("FSG_MATLAB")
            else:
                os.environ["FSG_MATLAB"] = old
        self.assertEqual(st["state"], "done", st.get("message"))
        port = lapsim.LapSim.load(LAPSIM).run(["m=270", "mu*1.05", "autocross.util_y=0.85"])
        for e in lapsim.EVENT_ORDER:
            self.assertAlmostEqual(st["results"][e]["T"], port["events"][e]["T"], delta=1e-6)
        self.assertIn("veh = lts_set_param(veh, 'mu', 1.05, 'scale');", job)
        self.assertIn("comp = competition_fsa2025();", job)
        self.assertIn("% (simulator only, not a LapSim parameter) yaw_inertia=100", job)

    def test_job_error_is_reported(self):
        from fsd_dashboard import matlab_runner as mr

        os.environ["FSG_MATLAB"], os.environ["FAKE_MATLAB_FAIL"] = FAKE_MATLAB, "1"
        try:
            with tempfile.TemporaryDirectory() as d:
                r = mr.MatlabRunner(LAPSIM)
                r.workdir = d
                r.submit([])
                for _ in range(100):
                    if r.status()["state"] in ("done", "error"):
                        break
                    time.sleep(0.1)
                st = r.status()
        finally:
            os.environ.pop("FSG_MATLAB")
            os.environ.pop("FAKE_MATLAB_FAIL")
        self.assertEqual(st["state"], "error")
        self.assertIn("Undefined function", st["message"])

    def test_windows_matlab_is_found_by_release(self):
        from fsd_dashboard.matlab_runner import _release_key

        paths = ["/mnt/c/Program Files/MATLAB/R2024a/bin/matlab.exe", "/mnt/c/Program Files/MATLAB/R2025b/bin/matlab.exe",
                 "/mnt/c/Program Files/MATLAB/R2025a/bin/matlab.exe"]
        self.assertEqual(sorted(paths, key=_release_key)[-1], paths[1])


class TestSetupDashboard(unittest.TestCase):
    """Offline demo + HTTP API, as the browser uses it."""

    @classmethod
    def setUpClass(cls):
        os.environ["FSG_MATLAB"] = FAKE_MATLAB
        from fsd_dashboard.demo import DemoRunner
        from fsd_dashboard.server import DashboardServer

        cls.tmp = tempfile.mkdtemp()
        cls.runner = DemoRunner(CONFIG, "acceleration", rtf=8.0)
        cls.runner.setup.configs_dir = os.path.join(cls.tmp, "configs")
        cls.runner.setup.matlab.workdir = os.path.join(cls.tmp, "matlab")
        cls.server = DashboardServer(cls.runner.model, "127.0.0.1", 0).start()
        cls.base = f"http://127.0.0.1:{cls.server.port}"
        cls.thread = threading.Thread(target=cls.runner.run, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.runner.stop()
        cls.server.stop()
        shutil.rmtree(cls.tmp, ignore_errors=True)
        os.environ.pop("FSG_MATLAB", None)

    def get(self, path):
        with urllib.request.urlopen(self.base + path, timeout=10) as r:
            return json.load(r)

    def post(self, body):
        req = urllib.request.Request(self.base + "/api/setup", json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.load(r)

    def test_schema_and_baseline(self):
        st = self.get("/api/setup")
        keys = [it["key"] for g in st["schema"] for it in g["items"]]
        for k in ("m", "lf", "c_z", "mux", "i_trans", "T_motor", "autocross.util_y", "yaw_inertia"):
            self.assertIn(k, keys)
        base = {it["key"]: it["base"] for g in st["schema"] for it in g["items"]}
        self.assertEqual((base["m"], base["muy"], base["skidpad.util_y"]), (284.0, 1.64, 0.8))
        self.assertAlmostEqual(st["baseline"]["total"], 164.35, delta=0.01)
        self.assertIsNotNone(self.get("/api/state")["setup_version"])

    def test_apply_runs_lapsim_matlab_and_simulation(self):
        res = self.post({"action": "apply", "name": "light", "changes": ["m=270", "mu*1.05", "yaw_inertia=110"]})
        self.assertTrue(res["ok"], res["message"])
        st = {}
        for _ in range(300):
            st = self.get("/api/setup")
            if (st["port"]["state"] == "done" and st["matlab"]["state"] in ("done", "error")
                    and st["sim"]["results"].get("acceleration")):
                break
            time.sleep(0.1)
        self.assertEqual(st["matlab"]["state"], "done", st["matlab"].get("message"))
        self.assertAlmostEqual(st["port"]["events"]["acceleration"]["T"], st["matlab"]["results"]["acceleration"]["T"], 5)
        self.assertLess(st["port"]["events"]["acceleration"]["T"], st["baseline"]["events"]["acceleration"]["T"])
        self.assertEqual(st["sim"]["state"], "active")
        self.assertIn("acceleration", st["sim"]["results"])
        self.assertEqual(self.get("/api/state")["vehicle_info"]["changes"], ["m=270", "mu*1.05"])
        self.assertEqual(self.runner.veh.yaw_inertia, 110.0)

    def test_configurations_and_validation(self):
        self.assertTrue(self.post({"action": "save", "name": "Aero+ (c_z 4.6)", "changes": ["c_z=4.6"], "notes": "x"})["ok"])
        names = [c["name"] for c in self.get("/api/setup")["configs"]]
        self.assertIn("Aero+ (c_z 4.6)", names)
        self.assertFalse(self.post({"action": "save", "name": "../../etc", "changes": []})["ok"])
        self.assertFalse(self.post({"action": "apply", "changes": ["nonsense_param=3"]})["ok"])
        self.assertFalse(self.post({"action": "apply", "changes": ["m=abc"]})["ok"])
        self.assertTrue(self.post({"action": "delete", "name": "Aero+ (c_z 4.6)"})["ok"])
        self.assertTrue(self.post({"action": "reload"})["ok"])

    def test_page_has_setup_view(self):
        with urllib.request.urlopen(self.base + "/", timeout=10) as r:
            html = r.read().decode()
        self.assertIn('id="page-setup"', html)
        self.assertIn('id="btn-apply"', html)


if __name__ == "__main__":
    unittest.main()
