"""Launcher (tools/launcher): validation, command building, HTTP API, offline demo start / stop."""

import json
import os
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools", "launcher"))
os.environ.setdefault("FSG_LAUNCHER_DIR", tempfile.mkdtemp(prefix="fsg-launcher-test-"))

import launcher as L  # noqa: E402


class TestValidation(unittest.TestCase):
    def test_changes(self):
        self.assertEqual(L.clean_changes("m=270 c_z*1.05, mu*0.95"), ["m=270", "c_z*1.05", "mu*0.95"])
        self.assertEqual(L.clean_changes(["yaw_inertia=120", "actuators.max_steer=0.4"]),
                         ["yaw_inertia=120", "actuators.max_steer=0.4"])
        self.assertEqual(L.clean_changes(""), [])
        for bad in ("m=270;rm", "m=$(id)", "m=`x`", "x", "m=1 && ls", "m='3'"):
            with self.assertRaises(ValueError, msg=bad):
                L.clean_changes(bad)

    def test_ps_quote(self):
        self.assertEqual(L.ps_quote("a'b"), "'a''b'")

    def test_jobs(self):
        app = L.Launcher(probe=False)
        t, argv, links, _ = app._job("sweep", {"vary": "m=250:320:4\nc_z=3.5,4.15", "tracks": 4, "compare": True}, "X")
        self.assertIn("--compare", argv)
        self.assertEqual(argv.count("--vary"), 2)
        self.assertTrue(links[0]["path"].startswith("results/sweep_X"))
        _, argv, links, _ = app._job("param_study", {"param": "m", "values": "260,300", "driver": "rl",
                                                     "events": ["acceleration", "bogus"]}, "Y")
        self.assertIn("rl", argv)
        self.assertNotIn("bogus", argv)
        app._slow["torch"] = "2.14.1 True"                    # pretend WSL has PyTorch
        app._slow["win_python"] = {}
        _, argv, _, _ = app._job("train_universal", {"backend": "torch", "device": "cuda:1"}, "Z")
        self.assertEqual(argv[argv.index("--device") + 1], "cuda:1")
        _, argv, _, _ = app._job("evaluate", {"policy": "universal", "changes": "m=260"}, "Z")
        self.assertIn("--evaluate", argv)
        for kind, p in (("sweep", {"vary": "m=1;rm -rf /"}), ("param_study", {"param": "m", "values": "1"}),
                        ("train_universal", {"backend": "torch", "device": "cuda;x"}), ("evaluate", {"policy": "../x"}),
                        ("train_setup", {"name": "a b"}), ("nope", {})):
            with self.assertRaises((ValueError, TypeError), msg=kind):
                app._job(kind, p, "S")


class TestPage(unittest.TestCase):
    def test_vehicle_setup_page(self):
        """The Vehicle setup is its own page (tab), like the dashboard's; the start form keeps the changes."""
        with open(os.path.join(ROOT, "tools", "launcher", "web", "index.html"), encoding="utf-8") as f:
            html = f.read()
        js_path = os.path.join(ROOT, "tools", "launcher", "web", "launcher.js")
        with open(js_path, encoding="utf-8") as f:
            js = f.read()
        for frag in ('id="nav-setup"', 'href="#setup"', 'id="page-main"', 'id="page-setup" hidden', 'name="changes" id="f-changes"',
                     'id="vs-groups"', 'id="veh-summary"', 'id="veh-brief"'):
            self.assertIn(frag, html)
        self.assertNotIn("veh-editor", html)
        # the start form, the stop buttons and the router button must stay wired (a rewrite once dropped them)
        for frag in ('form.addEventListener("submit"', 'post("/api/sim/start"', 'post("/api/sim/stop")',
                     '$("btn-stop").addEventListener', '$("hero-stop").addEventListener', '$("btn-router").addEventListener',
                     "function setMsg("):
            self.assertIn(frag, js)
        import re

        for ident in set(re.findall(r'\$\("([\w-]+)"\)', js)):
            self.assertIn(f'id="{ident}"', html, f"launcher.js uses #{ident}, which the page does not have")

    def test_forms_cannot_block_silently(self):
        """A hidden invalid field makes the browser drop a submit without a message (the Start button
        did nothing): every form is novalidate, and no number field's default breaks its own step."""
        import re

        html = open(os.path.join(ROOT, "tools", "launcher", "web", "index.html"), encoding="utf-8").read()
        forms = re.findall(r"<form[^>]*>", html)
        self.assertTrue(forms)
        for f in forms:
            self.assertIn("novalidate", f, f)
        for tag in re.findall(r'<input[^>]*type="number"[^>]*>', html):
            attrs = dict(re.findall(r'(\w+)="([^"]*)"', tag))
            if "value" in attrs and attrs.get("step", "1") != "any":
                lo, step, val = float(attrs.get("min", 0)), float(attrs.get("step", 1)), float(attrs["value"])
                self.assertAlmostEqual(((val - lo) / step) % 1.0, 0.0, msg=tag)


class TestBuild(unittest.TestCase):
    def test_colcon_base_and_stamp(self):
        old = dict(os.environ)
        try:
            os.environ.pop("FSG_COLCON_BASE", None)
            os.environ["FSG_WS"] = "/mnt/c/formula_student/fsg-dv-sim/ros2_ws"
            self.assertEqual(str(L.colcon_base()), os.path.join(os.path.expanduser("~"), ".fsg-dv-sim", "colcon"))
            os.environ["FSG_WS"] = "/home/x/fsg-dv-sim/ros2_ws"
            self.assertEqual(str(L.colcon_base()), "/home/x/fsg-dv-sim/ros2_ws")
            with tempfile.TemporaryDirectory() as d:
                os.environ["FSG_COLCON_BASE"] = d
                self.assertTrue(L.needs_build())                       # nothing installed
                os.makedirs(os.path.join(d, "install"))
                open(os.path.join(d, "install", "setup.bash"), "w").close()
                self.assertTrue(L.needs_build())                       # no stamp
                L.write_build_stamp()
                self.assertFalse(L.needs_build())
                self.assertEqual(len(L.build_signature()), 40)
        finally:
            os.environ.clear()
            os.environ.update(old)


class TestSessions(unittest.TestCase):
    def _app(self):
        app = L.Launcher(probe=False)
        app.started = []
        app._start = lambda p: app.started.append(p) or p
        app._alive = lambda k: True
        return app

    def test_isaac_command(self):
        app = self._app()
        L.win_path = lambda p: r"\\wsl.localhost\Ubuntu-24.04\home\laptop\fsg-dv-sim\isaac\run_isaac.ps1"
        app.settings["isaac_path"] = r"C:\isaac sim's"
        app._start_isaac({"mission": "autocross", "changes": ["m=270", "c_z*1.05"], "headless": True, "seed": 11})
        p = app.started[-1]
        self.assertTrue(p.windows)
        cmd = p.argv[-1]
        for s in ("[fsd-launcher] winpid", "-Mission 'autocross'", "-Set 'm=270','c_z*1.05'", "-Headless",
                  r"-IsaacPath 'C:\isaac sim''s'", "-Extra '--seed','11'", r"& '\\wsl.localhost\Ubuntu-24.04"):
            self.assertIn(s, cmd)

    def test_lite_session(self):
        app = self._app()
        app.start_router = lambda: (True, "router ok")
        orig = L.port_listening
        L.port_listening = lambda port: True
        try:
            app._run_session({"backend": "lite", "mission": "skidpad", "changes": ["m=270"], "seed": 5, "rtf": 2.0,
                              "longitudinal": "rl", "policy": "universal", "rviz": False, "headless": False}, 8080)
        finally:
            L.port_listening = orig
        argv = app.started[-1].argv
        self.assertIn("lite.launch.py", argv)
        for a in ("mission:=skidpad", "longitudinal:=rl", "rl_policy:=universal", "rviz:=false", "seed:=5",
                  "vehicle_set:=m=270", "realtime_factor:=2.0"):
            self.assertIn(a, argv)
        self.assertEqual(app.session["state"], "running")


class TestGpuAndShare(unittest.TestCase):
    def test_gpu_jobs_use_windows_python(self):
        app = L.Launcher(probe=False)
        app._slow["win_python"] = {"path": r"C:\Python314\python.exe", "python": "3.14.0", "torch": "2.14.1+cu132",
                                   "cuda": True, "gpu": "RTX", "missing": []}
        orig = L.win_path
        L.win_path = lambda p: "C:\\formula_student\\fsg-dv-sim" + str(p)[len(str(L.REPO)):].replace("/", "\\")
        try:
            title, argv, _, _ = app._job("train_universal", {"backend": "torch"}, "S")
            self.assertEqual(argv[0], "powershell.exe")
            cmd = argv[-1]
            for frag in (r"& 'C:\Python314\python.exe'", r"'C:\formula_student\fsg-dv-sim\tools\train_rl_universal.py'",
                         "'--backend' 'torch'", "PYTHONUTF8", "[fsd-launcher] winpid", "exit $LASTEXITCODE"):
                self.assertIn(frag, cmd)
            self.assertIn("Windows Python", title)
            _, argv, _, _ = app._job("sweep", {"vary": "m=260,300", "backend": "torch", "tracks": 0}, "S")
            self.assertIn("'--tracks' '0'", argv[-1])
            _, argv, _, _ = app._job("gpu_tests", {}, "S")
            self.assertIn("'tests.test_vec'", argv[-1])
            app._slow["win_python"]["missing"] = ["pyyaml"]
            app._slow["torch"] = ""
            with self.assertRaises(ValueError):
                app._job("train_universal", {"backend": "torch"}, "S")
        finally:
            L.win_path = orig

    def test_report(self):
        app = L.Launcher(probe=False)
        p = L.Proc("jobX", "Fake job", [sys.executable, "-c", "print('hello report')"], kind="job")
        app._start(p)
        while p.alive:
            time.sleep(0.05)
        rep = app.report(n_lines=50)
        self.assertIn("Fake job", rep)
        self.assertIn("hello report", rep)


    def test_gpu_memory_and_shared_gpu_note(self):
        out = "NVIDIA GeForce RTX 5060 Laptop GPU, 6123, 8151, 97\n"
        with mock.patch.object(L.shutil, "which", return_value="/usr/lib/wsl/lib/nvidia-smi"), \
                mock.patch.object(L.subprocess, "run", return_value=mock.Mock(stdout=out, returncode=0)):
            g = L.gpu_query()
        self.assertEqual(g, {"name": "NVIDIA GeForce RTX 5060 Laptop GPU", "used_mb": 6123, "total_mb": 8151, "util": 97})
        with mock.patch.object(L.shutil, "which", return_value=None), mock.patch.object(L.os.path, "exists", return_value=False):
            self.assertEqual(L.gpu_query(), {})
        # a GPU training is marked as such; with Isaac Sim running the start message says they share the GPU
        app = L.Launcher(probe=False)
        app._slow["torch"] = "2.5.0 True"
        app._slow["win_python"] = {}
        app._gpu = g
        started = []
        app._start = lambda proc: started.append(proc)
        with mock.patch.object(app, "_alive", side_effect=lambda k: k == "isaac"):
            ok, msg = app.start_job({"type": "train_universal", "params": {"backend": "torch"}})
        self.assertTrue(ok, msg)
        self.assertTrue(started[0].meta["gpu"])
        self.assertIn("share the GPU (8 GB)", msg)
        ok, msg = app.start_job({"type": "train_universal", "params": {"backend": "torch", "device": "cpu"}})
        self.assertFalse(started[1].meta["gpu"])
        self.assertNotIn("share the GPU", msg)



class TestVehicleSetup(unittest.TestCase):
    """Vehicle setup before the start: LapSim prediction, validation, RL coverage, saved configurations."""

    def setUp(self):
        sys.path.insert(0, os.path.join(ROOT, "tools", "launcher"))
        from vehicle import VehicleEditor

        self.tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp, ignore_errors=True))
        self.ed = VehicleEditor(configs_dir=os.path.join(self.tmp, "configs"))

    def test_info_and_check(self):
        info = self.ed.info()
        keys = [it["key"] for g in info["schema"] for it in g["items"]]
        for k in ("m", "c_z", "mux", "autocross.util_y", "yaw_inertia", "actuators.max_steer"):
            self.assertIn(k, keys)
        self.assertGreater(info["baseline"]["total"], 0)
        r = self.ed.check(["m=260", "c_z*1.1", "autocross.util_y=0.8", "yaw_inertia=120"])
        self.assertTrue(r["ok"], r["errors"])
        self.assertEqual(r["sim_changes"], ["m=260", "c_z*1.1", "yaw_inertia=120"])
        self.assertEqual(r["lapsim_only"], ["autocross.util_y=0.8"])
        self.assertLess(r["lapsim"]["events"]["acceleration"]["T"], info["baseline"]["events"]["acceleration"]["T"])
        self.assertIn("rl", r)
        bad = self.ed.check(["foo=3", "autocross.foo=1", "m=150"])
        self.assertFalse(bad["ok"])
        self.assertEqual(len(bad["errors"]), 2)
        self.assertEqual(bad["changes"], ["m=150"])
        self.assertIsNone(bad["rl"]["universal"])          # 150 kg is outside every training range

    def test_save_and_delete(self):
        ok, msg = self.ed.save("light car", ["m=260"], "test")
        self.assertTrue(ok, msg)
        self.assertEqual([c["name"] for c in self.ed.configs()], ["light car"])
        self.assertFalse(self.ed.save("bad", ["foo=3"])[0])
        self.assertFalse(self.ed.save("../x", ["m=260"])[0])
        self.assertTrue(self.ed.delete("light car")[0])
        self.assertEqual(self.ed.configs(), [])

    def test_start_sends_only_vehicle_changes(self):
        app = L.Launcher(probe=False)
        app._vehicle = self.ed
        got = []
        app._run_session = lambda o, port: got.append(o)
        ok, msg = app.start_sim({"backend": "demo", "mission": "acceleration",
                                 "changes": "m=260 autocross.util_y=0.8", "setup_name": "light car"})
        self.assertTrue(ok, msg)
        for _ in range(50):
            if got:
                break
            time.sleep(0.02)
        self.assertEqual(got[0]["changes"], ["m=260"])
        self.assertEqual(got[0]["lapsim_only"], ["autocross.util_y=0.8"])
        self.assertEqual(got[0]["setup_name"], "light car")
        ok, msg = app.start_sim({"backend": "demo", "mission": "acceleration", "changes": "foo=3"})
        self.assertFalse(ok)
        self.assertIn("unknown parameter", msg)


    def test_dashboard_gets_the_start_setup(self):
        """The setup chosen in the launcher reaches the dashboard: it shows it as the active setup (its form fills
        from it), including the LapSim-only changes the simulator does not get."""
        from fsd_dashboard.vehicle_setup import VehicleSetup, decode_start_setup

        enc = L.encode_start_setup("light car", ["m=260", "autocross.util_y=0.8"])
        self.assertRegex(enc, r"^[A-Za-z0-9_-]+$")                 # a plain string for ROS parameters
        self.assertEqual(decode_start_setup(enc), {"name": "light car", "changes": ["m=260", "autocross.util_y=0.8"]})
        self.assertIsNone(decode_start_setup(""))
        self.assertIsNone(decode_start_setup("not base64 json"))

        class _NoMatlab:
            available = False

            def status(self):
                return {"state": "unavailable"}

        vs = VehicleSetup(os.path.join(ROOT, "ros2_ws", "src", "fsd_sim_bringup", "config"), matlab=_NoMatlab())
        vs.set_start("light car", ["m=260", "autocross.util_y=0.8"])
        a = vs.active
        self.assertEqual((a["name"], a["changes"], a["start"]), ("light car", ["m=260", "autocross.util_y=0.8"], True))
        self.assertIsNotNone(a["applied"])
        self.assertEqual(vs.sim["state"], "active")

        # the launcher passes it to the launch file (lite / isaac) and to the offline demo
        app = L.Launcher(probe=False)
        started = []
        app._start = lambda proc: started.append(proc)
        app.ensure_built = lambda: (True, "built")
        app.start_router = lambda: (True, "router")
        o = {"backend": "lite", "mission": "acceleration", "changes": ["m=260"], "lapsim_only": ["autocross.util_y=0.8"],
             "setup_name": "light car", "seed": -1, "rtf": 1.0, "longitudinal": "reference", "policy": "auto",
             "rviz": False, "headless": False}
        app.session = {"state": "starting", "steps": [], "opts": o}
        app._run_session(o, 8080)
        arg = next(a for a in started[0].argv if str(a).startswith("vehicle_setup:="))
        self.assertEqual(decode_start_setup(arg.split(":=", 1)[1]),
                         {"name": "light car", "changes": ["m=260", "autocross.util_y=0.8"]})
        o["backend"] = "demo"
        app._run_session(o, 8080)
        argv = [str(a) for a in started[-1].argv]
        self.assertEqual(decode_start_setup(argv[argv.index("--start-setup") + 1])["name"], "light car")



FAKE_MATLAB = r"""#!/usr/bin/env python3
# stand-in for "matlab -batch run('job.m')": writes the job's results CSV and done file
import re, sys
src = open(re.search(r"run\('(.+)'\)", sys.argv[-1]).group(1)).read()
csv_p = re.search(r"writetable\(tbl, '(.+?)'\)", src).group(1)
done_p = re.search(r"fid = fopen\('(.+?)', 'w'\)", src).group(1)
open(csv_p, "w").write("Event,Time_s,Points\nacceleration,4.5,40\nskidpad,5.1,45\n")
open(done_p, "w").write("ok\n85.0\n")
print("fake MATLAB done")
"""


class TestMatlab(unittest.TestCase):
    """MATLAB from the launcher and the dashboard: run LR26_LapSim, open its files and results."""

    def setUp(self):
        sys.path.insert(0, os.path.join(ROOT, "tools", "launcher"))
        sys.path.insert(0, os.path.join(ROOT, "ros2_ws", "src", "fsd_dashboard"))
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp, ignore_errors=True))

    def test_open_only_listed_targets(self):
        from fsd_dashboard import matlab_files as mf

        lap = os.path.join(ROOT, "lapsim", "LR26_LapSim")
        tg = mf.targets(lap)
        self.assertTrue(tg["params"]["exists"])
        self.assertEqual(tg["folder"]["kind"], "folder")
        self.assertFalse(mf.open_target("../../etc/passwd", tg, lap)[0])
        calls = []
        with mock.patch.object(mf.shutil, "which", side_effect=lambda n: "/x/" + n if n in ("powershell.exe", "wslpath") else None), \
                mock.patch.object(mf, "win_path", side_effect=lambda p: "C:\\fs\\" + os.path.basename(p)), \
                mock.patch.object(mf.subprocess, "Popen", side_effect=lambda cmd, **kw: calls.append(cmd)):
            self.assertTrue(mf.open_target("params", tg, lap)[0])
            self.assertTrue(mf.open_target("folder", tg, lap)[0])
        self.assertEqual(calls[0][:3], ["powershell.exe", "-NoProfile", "-Command"])
        self.assertIn("Start-Process -FilePath 'C:\\fs\\lr26_params.m'", calls[0][3])
        self.assertEqual(calls[1], ["explorer.exe", "C:\\fs\\LR26_LapSim"])

    def test_launcher_runs_matlab_and_lists_its_results(self):
        fake = os.path.join(self.tmp, "matlab")
        with open(fake, "w") as f:
            f.write(FAKE_MATLAB.replace("#!/usr/bin/env python3", "#!" + sys.executable, 1))
        os.chmod(fake, 0o755)
        from vehicle import VehicleEditor

        with mock.patch.dict(os.environ, {"FSG_MATLAB": fake, "HOME": self.tmp}):
            ed = VehicleEditor(configs_dir=os.path.join(self.tmp, "configs"))
            st = ed.matlab_state()
            self.assertTrue(st["available"])
            self.assertNotIn("results_csv", st["files"])
            self.assertFalse(ed.matlab_run(["foo=3"])[0])            # a setup with mistakes is not run
            ok, msg = ed.matlab_run(["m=260"])
            self.assertTrue(ok, msg)
            for _ in range(300):
                st = ed.matlab_state()
                if st["state"] in ("done", "error"):
                    break
                time.sleep(0.05)
        self.assertEqual(st["state"], "done", st.get("message"))
        self.assertEqual(st["changes"], ["m=260"])
        self.assertEqual(st["results"]["acceleration"]["T"], 4.5)
        self.assertEqual(st["total"], 85.0)
        for k in ("results_csv", "matlab_log", "job_script", "results_dir"):
            self.assertTrue(st["files"][k]["exists"], k)
        self.assertIn(os.path.join("matlab", "launcher", "jobs"), st["files"]["results_csv"]["path"])

    def test_dashboard_open_needs_header_and_this_pc(self):
        from fsd_dashboard import server

        self.assertTrue(server._local("127.0.0.1"))
        self.assertTrue(server._local("::1"))
        self.assertTrue(server._local("::ffff:127.0.0.1"))
        self.assertFalse(server._local("192.168.1.20"))



class TestCameraCones(unittest.TestCase):
    """YOLO camera cones from the launcher: start options, training job, installs."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp, ignore_errors=True))
        self.app = L.Launcher(probe=False)
        self.app._vehicle_error = "off"                 # no LapSim check needed here

    def _start(self, **req):
        got = []
        self.app._run_session = lambda o, port: got.append(o)
        self.app.status = lambda: {"ros": True, "windows": True}
        ok, msg = self.app.start_sim({"backend": "isaac", "mission": "autocross", **req})
        for _ in range(50):
            if got or not ok:
                break
            time.sleep(0.02)
        return ok, msg, (got[0] if got else None)

    def test_start_options(self):
        self.app.cone_models = lambda: []               # independent of the models trained in this repo
        ok, msg, _ = self._start(perception="yolo")
        self.assertFalse(ok)
        self.assertIn("no cone detector yet", msg)
        self.app.cone_models = lambda: [{"name": "sim_a"}]
        ok, msg, o = self._start(perception="yolo", cone_model="sim_a", record_cones=True)
        self.assertTrue(ok, msg)
        self.assertEqual((o["perception"], o["cone_model"], o["record_cones"]), ("yolo", "sim_a", True))
        self.assertFalse(self._start(perception="yolo", cone_model="../x")[0])
        ok, msg = self.app.start_sim({"backend": "lite", "mission": "autocross", "perception": "yolo"})
        self.assertFalse(ok)
        self.assertIn("need Isaac Sim", msg)
        # the launch arguments
        started = []
        self.app._start = lambda proc: started.append(proc)
        self.app.ensure_built = lambda: (True, "built")
        self.app.start_router = lambda: (True, "router")
        self.app.session = {"state": "starting", "steps": [], "opts": o}
        L.Launcher._run_session(self.app, o, 8080)
        argv = [str(a) for a in started[0].argv]
        self.assertIn("perception:=yolo", argv)
        self.assertIn("cone_model:=" + str(L.REPO / "models" / "cones" / "sim_a"), argv)
        self.assertIn("record_cones:=true", argv)
        self.assertTrue(any(a.startswith("record_session:=isaac_autocross_") for a in argv))

    def test_training_job_and_installs(self):
        self.app._slow["win_python"] = {"path": r"C:\\Python314\\python.exe", "python": "3.14", "torch": "2.14",
                                        "cuda": True, "gpu": "RTX", "missing": [], "ultralytics": ""}
        with self.assertRaises(ValueError):                 # Ultralytics missing -> say how to install it
            self.app._job("train_cones", {}, "S")
        self.app._slow["win_python"]["ultralytics"] = "8.4.0"
        self.app._slow["perception"] = {"t": time.time(), "images": 0, "sessions": 0}
        with self.assertRaises(ValueError) as e:               # nothing recorded yet -> say how to record
            self.app._job("train_cones", {}, "S")
        self.assertIn("Record camera images", str(e.exception))
        self.app._slow["perception"] = {"t": time.time(), "images": 800, "sessions": 1}
        orig = L.win_path
        L.win_path = lambda p: "C:\\fs" + str(p)[len(str(L.REPO)):].replace("/", "\\")
        try:
            title, argv, links, _ = self.app._job("train_cones", {"size": "s", "epochs": 80, "name": "cones_x"}, "S")
            cmd = argv[-1]
            for frag in ("train_cone_yolo.py'", "'--model' 'yolo11s.pt'", "'--epochs' '80'", "'--name' 'cones_x'"):
                self.assertIn(frag, cmd)
            self.assertEqual(links[0]["path"], "models/cones/cones_x/results.md")
            _, argv, _, _ = self.app._job("yolo_pip", {}, "S")
            self.assertIn("'ultralytics'", argv[-1])
            self.assertTrue(self.app._job("train_cones", {"size": "n"}, "S")[0].startswith("Cone detector"))
        finally:
            L.win_path = orig
        _, argv, _, _ = self.app._job("ort_pip", {}, "S")
        self.assertIn("onnxruntime", argv[-1])

    def test_perception_info_counts_recorded_images(self):
        with mock.patch.object(L, "REPO", L.Path(self.tmp)):
            for s, n in (("run1", 3), ("run2", 2)):
                os.makedirs(os.path.join(self.tmp, "datasets", "cones", s, "images"))
                for i in range(n):
                    open(os.path.join(self.tmp, "datasets", "cones", s, "images", f"{i}.jpg"), "w").close()
            info = self.app.perception_info()
        self.assertEqual((info["images"], info["sessions"]), (5, 2))


class TestHTTP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = L.Launcher(probe=False)
        cls.app.settings["dashboard_port"] = 18093
        cls.httpd = L.ThreadingHTTPServer(("127.0.0.1", 0), None)
        cls.httpd.RequestHandlerClass = L.make_handler(cls.app, lambda: threading.Thread(target=cls.httpd.shutdown).start())
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.app.stop_all()
        cls.httpd.shutdown()

    def get(self, path):
        with urllib.request.urlopen(self.base + path, timeout=10) as r:
            return r.status, r.read()

    def post(self, path, body, header=True):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json", **({"X-FSG-Launcher": "1"} if header else {})})
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            return json.loads(e.read())

    def test_pages_and_state(self):
        self.assertEqual(self.get("/")[0], 200)
        self.assertEqual(self.get("/brand/lions-logo.png")[0], 200)
        self.assertEqual(self.get("/static/launcher.js")[0], 200)
        st = json.loads(self.get("/api/state")[1])
        self.assertIn("status", st)
        self.assertIn("trackdrive", st["missions"])
        with self.assertRaises(urllib.error.HTTPError):
            self.get("/static/../launcher.py")

    def test_post_needs_header(self):
        r = self.post("/api/sim/start", {"backend": "demo"}, header=False)
        self.assertFalse(r["ok"])
        self.assertIn("header", r["message"])

    def test_demo_start_stop(self):
        r = self.post("/api/sim/start", {"backend": "demo", "mission": "acceleration", "rtf": 5})
        self.assertTrue(r["ok"], r)
        t0 = time.time()
        while time.time() - t0 < 30 and not L.port_listening(18093):
            time.sleep(0.2)
        self.assertTrue(L.port_listening(18093))
        self.assertFalse(self.post("/api/sim/start", {"backend": "demo"})["ok"])     # one at a time
        self.assertTrue(self.post("/api/sim/stop", {})["ok"])
        t0 = time.time()
        while time.time() - t0 < 20 and self.app._alive("sim"):
            time.sleep(0.2)
        self.assertFalse(self.app._alive("sim"))
        self.assertIn("offline demo", self.app.procs["sim"].tail())

    def test_bad_inputs(self):
        self.assertFalse(self.post("/api/sim/start", {"backend": "demo", "changes": "m=1;ls"})["ok"])
        self.assertFalse(self.post("/api/sim/start", {"backend": "x"})["ok"])
        self.assertFalse(self.post("/api/job/start", {"type": "param_study", "params": {"param": "m;"}})["ok"])
        self.assertFalse(self.post("/api/settings", {"isaac_path": "C:\\x\"; rm"})["ok"])


if __name__ == "__main__":
    unittest.main()
