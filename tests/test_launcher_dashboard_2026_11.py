"""Launcher and dashboard additions of 10 Nov 2026: perception / SLAM / racing line / faults / handling start
options, regression and installation-check jobs and tools, per-sensor delay, the offline demo's racing line."""

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
import unittest.mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "tools", "launcher"), os.path.join(ROOT, "tools"),
                os.path.join(ROOT, "ros2_ws", "src", "fsdsim"), os.path.join(ROOT, "ros2_ws", "src", "fsd_dashboard")]
os.environ.setdefault("FSG_LAUNCHER_DIR", tempfile.mkdtemp(prefix="fsg-launcher-test-"))
CONFIG = os.path.join(ROOT, "ros2_ws", "src", "fsd_sim_bringup", "config")

import launcher as L  # noqa: E402


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


class TestStartOptions(unittest.TestCase):
    def setUp(self):
        self.app = L.Launcher(probe=False)
        self.app._vehicle_error = "off"
        self.app.status = lambda: {"ros": True, "windows": True}
        self.app.cone_models = lambda: [{"name": "sim_a"}]

    def _start(self, **req):
        got = []
        self.app._run_session = lambda o, port: got.append(o)
        ok, msg = self.app.start_sim({"backend": "lite", "mission": "trackdrive", **req})
        for _ in range(50):
            if got or not ok:
                break
            time.sleep(0.02)
        return ok, msg, (got[0] if got else None)

    def test_perception_slam_line_and_race_after(self):
        ok, msg, o = self._start(perception="fusion", slam=True, line="racing", race_after=1, handling="double_track",
                                 faults="lidar:dropout@20+3, imu:bias=0.02@10")
        self.assertTrue(ok, msg)
        self.assertEqual((o["perception"], o["slam"], o["line"], o["race_after"]), ("fusion", True, "racing", 1))
        self.assertEqual(o["faults"], "lidar:dropout@20+3,imu:bias=0.02@10")
        self.assertEqual(L.sim_changes(o), ["chassis.handling=double_track"])
        self.assertEqual(o["changes"], [])                      # the dashboard's setup shows only the user's changes

    def test_refusals(self):
        for req, text in (({"perception": "radar"}, "unknown cone source"),
                          ({"perception": "none", "race_after": 1}, "mapped track"),
                          ({"perception": "lidar", "race_after": 1, "mission": "autocross"}, "mapped track"),
                          ({"faults": "lidar:explode@3"}, "sensor faults"),
                          ({"faults": "lidar:dropout@3; rm -rf /"}, "sensor faults"),
                          ({"backend": "demo", "perception": "lidar"}, "offline demo"),
                          ({"line": "racing", "longitudinal": "rl"}, "reference driver"),
                          ({"line": "racing", "perception": "lidar"}, "race on the mapped track")):
            ok, msg, _ = self._start(**req)
            self.assertFalse(ok, req)
            self.assertIn(text, msg)

    def test_slam_only_with_own_perception_and_line_only_on_circuits(self):
        _, _, o = self._start(perception="none", slam=True, mission="skidpad", line="racing")
        self.assertEqual((o["slam"], o["line"]), (False, "center"))

    def test_isaac_fusion_needs_a_detector(self):
        self.app.cone_models = lambda: []
        ok, msg, _ = self._start(backend="isaac", perception="fusion")
        self.assertFalse(ok)
        self.assertIn("no cone detector yet", msg)
        ok, msg, _ = self._start(backend="isaac", perception="lidar")
        self.assertTrue(ok, msg)

    def _commands(self, o):
        procs = []
        self.app.ensure_built = lambda: (True, "")
        self.app.start_router = lambda: (True, "router")
        self.app._start = procs.append
        with unittest.mock.patch.object(L, "port_listening", lambda p: True), \
                unittest.mock.patch.object(L, "win_path", lambda p: "C:\\repo\\isaac\\run_isaac.ps1"):
            self.app.session = {"state": "starting", "steps": [], "opts": o}
            self.app._run_session(o, 8080)
        return [" ".join(str(a) for a in p.argv) for p in procs]

    def test_launch_arguments(self):
        run = L.Launcher._run_session
        base = {"mission": "trackdrive", "changes": ["m=270"], "seed": 7, "rtf": 1.0, "longitudinal": "reference",
                "policy": "auto", "rviz": False, "headless": True, "setup_name": "", "lapsim_only": [], "cone_model": "sim_a",
                "record_cones": False, "slam": True, "line": "racing", "race_after": 1, "faults": "lidar:dropout@20+3",
                "handling": "double_track"}
        self.app._run_session = lambda o, port: run(self.app, o, port)
        lite = self._commands({**base, "backend": "lite", "perception": "lidar"})[0]
        for part in ("perception:=lidar", "slam:=true", "line:=racing", "race_after:=1", "faults:=lidar:dropout@20+3",
                     "vehicle_set:=m=270 chassis.handling=double_track"):
            self.assertIn(part, lite)
        self.assertNotIn("cone_model", lite)
        relay, isaac = self._commands({**base, "backend": "isaac", "perception": "fusion"})
        self.assertIn("perception:=fusion", relay)
        self.assertIn(os.path.join("models", "cones", "sim_a"), relay)
        self.assertIn("-Faults 'lidar:dropout@20+3'", isaac)
        self.assertIn("'chassis.handling=double_track'", isaac)
        demo = self._commands({**base, "backend": "demo", "perception": "none"})[0]
        self.assertIn("--line racing", demo)
        self.assertIn("--faults lidar:dropout@20+3", demo)

    def test_jobs(self):
        title, argv, links, _ = self.app._job("regression", {"quick": True, "seeds": "7, 11"}, "x")
        argv = [str(a) for a in argv]
        self.assertTrue(argv[1].endswith("regression.py"))
        self.assertIn("--quick", argv)
        self.assertEqual(argv[-3:], ["--seeds", "7", "11"])
        self.assertTrue(links[0]["path"].endswith("report.md"))
        self.assertTrue(str(self.app._job("check_install", {}, "x")[1][1]).endswith("check_install.py"))

    def test_page_has_the_controls(self):
        html, js = (_read(os.path.join(ROOT, "tools", "launcher", "web", n)) for n in ("index.html", "launcher.js"))
        for p in L.PERCEPTION:
            self.assertIn(f'data-percep="{p}"', html)
        for name in ('name="slam"', 'name="line"', 'name="race_after"', 'name="faults"', 'name="handling"',
                     'data-job-btn="regression"', 'data-job-btn="check_install"'):
            self.assertIn(name, html)
        for key in ("slam:", "line:", "race_after:", "faults:", "handling:"):
            self.assertIn(key, js)


class TestTools(unittest.TestCase):
    def test_check_install_reports_every_group(self):
        import check_install as C
        res = C.checks()
        self.assertEqual({r["group"] for r in res}, {"tools", "ros", "isaac", "optional"})
        self.assertTrue(all(r["fix"] for r in res if not r["ok"] and r["group"] != "tools"))
        need = {r["name"]: r["ok"] for r in res if r["required"]}
        self.assertTrue(need["numpy"] and need["PyYAML"])
        out = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "check_install.py"), "--json"],
                             capture_output=True, text=True, timeout=60)
        self.assertEqual(len(json.loads(out.stdout)), len(res))

    def test_regression_cases_and_one_run(self):
        import regression as R
        full, quick = R.cases([7, 11], False), R.cases([7, 11], True)
        self.assertGreater(len(full), len(quick))
        self.assertEqual({c["group"] for c in full}, {"events", "circuits", "double track", "perception", "faults"})
        self.assertTrue(all(c["clean"] for c in full if c["group"] != "faults"))
        r = R.run_case(next(c for c in quick if c["name"] == "skidpad @ 50 Hz"))
        self.assertTrue(r["ok"], r)
        self.assertGreater(r["energy_kwh"], 0.0)
        bad = R.run_case({"group": "x", "name": "x", "mission": "skidpad", "clean": True, "kw": {"nonsense": 1}})
        self.assertFalse(bad["ok"])                               # a crash is a failed case, not a crashed run


class TestDashboard(unittest.TestCase):
    def test_stream_delay(self):
        from fsd_dashboard.model import Stream
        s = Stream("lidar", "LiDAR", "/lidar/points", 10.0)
        self.assertIsNone(s.view(0.0)["delay_ms"])
        for i in range(20):
            s.tick(i * 0.1, i * 0.1, now=i * 0.1 + 0.06)
        self.assertEqual(s.view(2.0)["delay_ms"], 60)
        s.tick(0.0, 3.0, now=0.02)                                # simulator restarted
        self.assertIsNone(s.delay())

    def test_map_stats_reach_the_page(self):
        from fsd_dashboard.model import ClientCursor, DashboardModel
        m = DashboardModel(source="test")
        m.set_perception("map", {"mode": "fusion", "slam": True, "map": {"mapped": 113, "truth": 113}, "pose_err": 0.05})
        self.assertEqual(m.delta(ClientCursor())["perception"]["map"]["pose_err"], 0.05)
        js = _read(os.path.join(ROOT, "ros2_ws", "src", "fsd_dashboard", "fsd_dashboard", "web", "app.js"))
        self.assertIn("map-meta", js)
        self.assertIn("delay_ms", js)

    def test_demo_racing_line_is_faster_and_clean(self):
        from fsd_dashboard.demo import DemoRunner
        laps = {}
        for line in ("center", "racing"):
            r = DemoRunner(CONFIG, "autocross", line=line)
            for _ in range(3000):
                r.step(0.02)
                if r.sim.mission.state.name in ("FINISHED", "EMERGENCY"):
                    break
            res = r.sim.result()
            self.assertEqual((r.sim.mission.state.name, res.cones_hit, res.off_course), ("FINISHED", 0, 0), line)
            laps[line] = res.lap_times[0]
        self.assertLess(laps["racing"], laps["center"] - 0.3)


if __name__ == "__main__":
    unittest.main()
