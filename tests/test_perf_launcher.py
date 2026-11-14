"""perf/launcher: the Performance Lab launcher window's logic (core.py), the start scripts, the guide and the DV
launcher's button. The window itself is only smoke-tested where Tk and a display exist."""

import http.client
import json
import os
import shutil
import sys
import tempfile
import threading
import unittest
import zipfile
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "perf", "launcher"))
sys.path.insert(0, os.path.join(ROOT, "perf"))
HAVE_LAPSIM = os.path.isfile(os.path.join(ROOT, "lapsim", "LR26_LapSim", "config", "lr26_params.m"))


def free_port() -> int:
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class LauncherCase(unittest.TestCase):
    def setUp(self):
        import core

        self.core = core
        self.tmp = tempfile.mkdtemp(prefix="perf-launcher-")
        self.env = mock.patch.dict(os.environ, {"PERF_LAB_HOME": os.path.join(self.tmp, "home"),
                                                "PERF_LAB_DOCS": os.path.join(self.tmp, "docs")})
        self.env.start()
        self.cfg = os.path.join(self.tmp, "configs")
        os.makedirs(self.cfg)
        self.patch = mock.patch.object(core, "CONFIGS", core.Path(self.cfg))
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.env.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)


class TestCore(LauncherCase):
    def test_settings(self):
        c = self.core
        self.assertEqual(c.load_settings()["port"], 8090)
        c.save_settings({"port": 8095, "auto_start": False, "stop_on_close": True, "junk": 1})
        s = c.load_settings()
        self.assertEqual((s["port"], s["auto_start"]), (8095, False))
        with open(os.path.join(os.environ["PERF_LAB_HOME"], "settings.json"), "w") as f:
            f.write('{"port": "nonsense"}')
        self.assertEqual(c.load_settings()["port"], 8090)
        c.log("hello")
        self.assertIn("hello", c.log_file().read_text())

    def write(self, name, d):
        with open(os.path.join(self.cfg, name), "w") as f:
            json.dump(d, f)

    def test_backup_and_restore(self):
        c = self.core
        self.write("Wing_v2.json", {"name": "Wing v2", "changes": ["c_z=4.5"]})
        self.write("eta_map_D1.json", {"torque_Nm": [0, 1], "rpm": [0, 1], "eta": [[0.9, 0.9], [0.9, 0.9]]})
        z = c.backup()
        self.assertTrue(str(z).startswith(os.environ["PERF_LAB_DOCS"]))
        with zipfile.ZipFile(z) as f:
            self.assertEqual(sorted(n for n in f.namelist() if n.startswith("configs/")),
                             ["configs/Wing_v2.json", "configs/eta_map_D1.json"])
        # same files: nothing changes; a changed setup is kept and the backup's copy added beside it
        self.assertEqual(sorted(c.restore(z)["same"]), ["Wing_v2.json", "eta_map_D1.json"])
        self.write("Wing_v2.json", {"name": "Wing v2", "changes": ["c_z=4.8"]})
        r = c.restore(z)
        self.assertEqual(r["renamed"], ["Wing_v2 (restored).json"])
        with open(os.path.join(self.cfg, "Wing_v2.json")) as f:
            self.assertEqual(json.load(f)["changes"], ["c_z=4.8"])            # the user's version is untouched
        with open(os.path.join(self.cfg, "Wing_v2 (restored).json")) as f:
            self.assertEqual(json.load(f)["name"], "Wing v2 (restored)")
        os.remove(os.path.join(self.cfg, "eta_map_D1.json"))
        self.assertEqual(c.restore(z)["added"], ["eta_map_D1.json"])
        # odd entries are skipped, nothing leaves the folder
        bad = os.path.join(self.tmp, "bad.zip")
        with zipfile.ZipFile(bad, "w") as f:
            f.writestr("configs/../../evil.json", "{}")
            f.writestr("configs/run.bat", "echo")
            f.writestr("configs/broken.json", "{not json")
        r = c.restore(bad)
        self.assertIn("broken.json", r["skipped"])
        self.assertIn("run.bat", r["skipped"])
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "evil.json")))

    def test_checks_report_and_shortcut(self):
        c = self.core
        with mock.patch.object(c, "missing_packages", return_value=["scipy"]):
            checks = c.health_check(port=free_port(), run_self_test=True)
        pk = [x for x in checks if x.name == "Python packages"][0]
        self.assertFalse(pk.ok)
        self.assertTrue(pk.repairable)
        self.assertEqual(pk.extra["missing"], ["scipy"])
        self.assertNotIn("Test run", [x.name for x in checks])          # no test run without the packages
        z = c.help_report(checks)
        with zipfile.ZipFile(z) as f:
            rep = f.read("report.md").decode()
        self.assertIn("FAILED Python packages", rep)
        opt = [x for x in checks if x.optional]
        self.assertEqual([x.name for x in opt], ["Optional: CasADi (optimal control)"])
        self.assertIn("missing (optional)", rep)
        ps = c.shortcut_script(r"C:\Py\pythonw.exe", r'"C:\x y\perf_launcher.pyw"', r"C:\x y", r"C:\x y\perf_lab.ico")
        self.assertIn("Performance Lab.lnk", ps)
        self.assertIn("'Desktop'", ps)
        self.assertIn("'Programs'", ps)
        self.assertIn(r"'C:\Py\pythonw.exe'", ps)
        self.assertEqual(c._ps_quote("it's"), "'it''s'")
        with mock.patch.object(c.subprocess, "Popen") as popen:
            self.assertTrue(c.pip_install(["numpy; rm -rf ~"]))             # odd names are dropped: nothing is run
            popen.assert_not_called()

    @unittest.skipUnless(HAVE_LAPSIM, "lapsim/LR26_LapSim not present")
    def test_health_start_stop(self):
        c = self.core
        port = free_port()
        checks = c.health_check(port=port)
        self.assertTrue(all(x.ok or x.optional for x in checks), [(x.name, x.detail) for x in checks])
        self.assertIn("points", [x for x in checks if x.name == "Test run"][0].detail)
        self.assertTrue(c.port_free(port))
        ok, msg = c.start_app(port)
        try:
            self.assertTrue(ok, msg)
            self.assertTrue(c.is_running(port))
            self.assertFalse(c.port_free(port))
            self.assertTrue(c.start_app(port)[0])                         # second start: already running
            port_check = [x for x in c.health_check(port=port, run_self_test=False) if x.name == "Web page port"][0]
            self.assertTrue(port_check.ok)
        finally:
            ok2, msg2 = c.stop_app(port)
        self.assertTrue(ok2, msg2)
        self.assertFalse(c.is_running(port))


class TestScripts(unittest.TestCase):
    def read(self, *p):
        with open(os.path.join(ROOT, *p), "rb") as f:
            return f.read()

    def test_bat_files(self):
        for p in (("perf", "perf.bat"), ("Performance Lab.bat",)):
            raw = self.read(*p)
            self.assertNotIn(b"\n", raw.replace(b"\r\n", b""), p)           # CRLF only
        bat = self.read("perf", "perf.bat").decode()
        for needed in ("winget install -e --id Python.Python.3.12 --scope user", "python.org/ftp/python/%PY_VERSION%",
                       "InstallAllUsers=0", "pip install --user", r"launcher\perf_launcher.pyw", "pythonw.exe",
                       "choice /c YN", "--console"):
            self.assertIn(needed, bat)
        self.assertIn(r"perf\perf.bat", self.read("Performance Lab.bat").decode())
        for f in ("perf_launcher.pyw", "core.py", "perf_lab.ico", "logo.png"):
            self.assertTrue(os.path.isfile(os.path.join(ROOT, "perf", "launcher", f)), f)

    def test_dv_launcher_button(self):
        sys.path.insert(0, os.path.join(ROOT, "tools", "launcher"))
        os.environ.setdefault("FSG_LAUNCHER_DIR", tempfile.mkdtemp(prefix="fsg-launcher-test-"))
        import launcher as L

        app = L.Launcher(probe=False)
        with mock.patch.object(L, "win_path", return_value=r"C:\formula_student\fsg-dv-sim\perf\perf.bat"), \
             mock.patch.object(L.shutil, "which", return_value="/mnt/c/Windows/system32/cmd.exe"):
            title, argv, _, _ = app._job("perf_lab", {}, "S")
        self.assertIn("Performance Lab", title)
        self.assertEqual(argv[:3], ["cmd.exe", "/c", "start"])
        self.assertTrue(argv[-1].endswith(r"perf\perf.bat"))
        with mock.patch.object(L.shutil, "which", return_value=None):
            with self.assertRaises(ValueError):
                app._job("perf_lab", {}, "S")
        html = self.read("tools", "launcher", "web", "index.html").decode()
        self.assertIn('data-job-btn="perf_lab"', html)


@unittest.skipUnless(HAVE_LAPSIM, "lapsim/LR26_LapSim not present")
class TestGuide(unittest.TestCase):
    def test_guide_and_docs_route(self):
        from http.server import ThreadingHTTPServer

        import app as perf_app
        from perflab.lab import Lab

        tmp = tempfile.mkdtemp()
        try:
            a = perf_app.App(Lab(os.path.join(ROOT, "lapsim", "LR26_LapSim"), configs_dir=tmp))
            httpd = ThreadingHTTPServer(("127.0.0.1", 0), perf_app.make_handler(a, lambda: None))
            threading.Thread(target=httpd.serve_forever, daemon=True).start()
            port = httpd.server_address[1]

            def get(path):
                c = http.client.HTTPConnection("127.0.0.1", port, timeout=20)
                c.request("GET", path)
                r = c.getresponse()
                return r.status, r.getheader("Content-Type"), r.read()

            st, _, page = get("/")
            self.assertIn(b'id="page-guide"', page)
            self.assertIn(b"How do I", page)
            self.assertIn(b'id="help-page"', page)
            for anchor in (b"guide-overview", b"guide-setup", b"guide-studies", b"guide-transient", b"guide-energy",
                           b"guide-ggv", b"guide-logs"):
                self.assertIn(anchor, page)
            st, ct, body = get("/docs/PERF_APP.md")
            self.assertEqual((st, ct.split(";")[0]), (200, "text/plain"))
            if os.path.isfile(os.path.join(ROOT, "docs", "paper", "paper2_performance_lab.pdf")):
                st, ct, body = get("/docs/paper2_performance_lab.pdf")
                self.assertEqual((st, ct), (200, "application/pdf"))
            for bad in ("/docs/../app.py", "/docs/README.md", "/docs/%2e%2e/perf/app.py"):
                self.assertEqual(get(bad)[0], 404, bad)
            httpd.shutdown()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


def _tk_ok():
    try:
        import tkinter
        r = tkinter.Tk()
        r.destroy()
        return True
    except Exception:  # noqa: BLE001
        return False


@unittest.skipUnless(HAVE_LAPSIM and _tk_ok(), "needs Tk and a display")
class TestWindow(LauncherCase):
    def test_window_starts_checks_and_closes(self):
        import runpy
        ns = runpy.run_path(os.path.join(ROOT, "perf", "launcher", "perf_launcher.pyw"), run_name="perf_launcher")
        core = ns["core"]
        core.save_settings({"port": free_port(), "auto_start": False, "stop_on_close": True})
        w = ns["Launcher"](auto_start=False)
        import time
        t0 = time.time()
        while time.time() - t0 < 60 and not getattr(w, "checks", None):
            w.update()
            time.sleep(0.05)
        self.assertTrue(w.checks and all(c.ok for c in w.checks), [(c.name, c.detail) for c in getattr(w, "checks", [])])
        w.on_close()


if __name__ == "__main__":
    unittest.main()
