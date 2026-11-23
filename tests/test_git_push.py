"""tools/git_push.py: one commit pushed to two remotes (local bare repositories stand in for GitHub and GitLab)."""

import io
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
GIT = shutil.which("git")


def _git(*args, cwd=None):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout


@unittest.skipUnless(GIT, "git not installed")
class TestGitPush(unittest.TestCase):
    def setUp(self):
        import git_push

        self.gp = git_push
        self.tmp = tempfile.mkdtemp()
        self.work = os.path.join(self.tmp, "project")
        os.makedirs(os.path.join(self.work, "tools"))
        os.makedirs(os.path.join(self.work, "assets", "lr26"))
        shutil.copy(os.path.join(ROOT, ".gitignore"), self.work)
        with open(os.path.join(self.work, "tools", "a.py"), "w") as f:
            f.write("print('a')\n")
        with open(os.path.join(self.work, "assets", "lr26", "lr26_body.usda"), "w") as f:
            f.write("#usda 1.0\n")                       # confidential: .gitignore keeps it out
        self.remotes = {}
        for name in ("github", "gitlab"):
            path = os.path.join(self.tmp, f"{name}.git")
            _git("init", "-q", "--bare", path)
            self.remotes[name] = path

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_tool(self, *extra):
        args = ["--repo", self.work, "--git", GIT, "--github", self.remotes["github"], "--gitlab", self.remotes["gitlab"],
                "--name", "Test", "--email", "test@example.com", *extra]
        out = io.StringIO()
        with redirect_stdout(out):
            rc = self.gp.main(args)
        return rc, out.getvalue()

    def head(self, bare):
        return _git("--git-dir", bare, "rev-parse", "refs/heads/main").strip()

    def test_first_push_reaches_both_without_confidential_files(self):
        rc, out = self.run_tool("--init", "--message", "first")
        self.assertEqual(rc, 0, out)
        self.assertEqual(self.head(self.remotes["github"]), self.head(self.remotes["gitlab"]))
        files = _git("--git-dir", self.remotes["gitlab"], "ls-tree", "-r", "--name-only", "main").split()
        self.assertIn("tools/a.py", files)
        self.assertNotIn("assets/lr26/lr26_body.usda", files)
        # a second push without changes is fine and needs no message
        rc, out = self.run_tool()
        self.assertEqual(rc, 0, out)
        self.assertIn("nothing new to commit", out)

    def test_confidential_file_stops_everything(self):
        with open(os.path.join(self.work, ".gitignore"), "w") as f:
            f.write("")                                   # someone removed the rules
        rc, out = self.run_tool("--init", "--message", "oops")
        self.assertEqual(rc, 1)
        self.assertIn("confidential", out)
        self.assertEqual(_git("ls-files", cwd=self.work), "")                    # unstaged again
        self.assertNotEqual(subprocess.run(["git", "--git-dir", self.remotes["github"], "rev-parse", "-q", "--verify",
                                            "refs/heads/main"], capture_output=True).returncode, 0)

    def test_one_failing_remote_does_not_stop_the_other(self):
        self.remotes["gitlab"] = os.path.join(self.tmp, "missing.git")
        rc, out = self.run_tool("--init", "--message", "first")
        self.assertEqual(rc, 1)
        self.assertIn("github: OK", out)
        self.assertIn("gitlab: FAILED", out)
        self.assertTrue(self.head(self.remotes["github"]))

    def test_changes_need_a_message(self):
        rc, out = self.run_tool("--init")
        self.assertEqual(rc, 1)
        self.assertIn("commit message is needed", out)

    def test_check_does_not_change_anything(self):
        rc, out = self.run_tool("--check")
        self.assertEqual(rc, 0, out)
        self.assertFalse(os.path.exists(os.path.join(self.work, ".git")))

    def test_rules(self):
        self.assertTrue(self.gp.is_confidential("cad/lr26/LR26_FW_final.stp"))
        self.assertTrue(self.gp.is_confidential("lapsim/LR26_LapSim/config/lr26_params.m"))
        self.assertTrue(self.gp.is_confidential("assets/lr26/lr26_wheel_fl.usda"))
        self.assertFalse(self.gp.is_confidential("models/cones/cones_301026/model.onnx"))
        self.assertFalse(self.gp.is_confidential("rl/policies/universal/policy.json"))
        self.assertIn("never force-pushes", self.gp.hint_for("! [rejected] main -> main (non-fast-forward)"))
        self.assertTrue(self.gp.URL_RE.match("https://github.com/lions-racing/fsg-dv-sim.git"))
        self.assertTrue(self.gp.URL_RE.match("git@gitlab.com:lions/fsg-dv-sim.git"))
        self.assertFalse(self.gp.URL_RE.match("https://github.com/x/y.git; rm -rf ~"))




class TestLauncherGit(unittest.TestCase):
    """Launcher: settings checks and the two jobs (no git is run here)."""

    def test_settings_and_jobs(self):
        import tempfile as _t
        from unittest import mock

        sys.path.insert(0, os.path.join(ROOT, "tools", "launcher"))
        os.environ.setdefault("FSG_LAUNCHER_DIR", _t.mkdtemp(prefix="fsg-launcher-test-"))
        import launcher as L

        app = L.Launcher(probe=False)
        with mock.patch.object(L, "SETTINGS_FILE", L.Path(_t.mkdtemp()) / "launcher.json"):
            with self.assertRaises(ValueError):
                app._job("git_push", {"message": "x"}, "S")          # no URL yet
            for bad in ({"git_github": "https://github.com/x/y.git; rm -rf ~"}, {"git_branch": "a..b"},
                        {"git_email": "not an address"}, {"git_name": 'a"b'}):
                self.assertFalse(app.save_settings(bad)[0], bad)
            ok, msg = app.save_settings({"git_github": "https://github.com/lions/fsg-dv-sim.git",
                                         "git_gitlab": "git@gitlab.com:lions/fsg-dv-sim.git", "git_branch": "",
                                         "git_name": "Miguel", "git_email": "m@example.com"})
            self.assertTrue(ok, msg)
            self.assertEqual(app.settings["git_branch"], "main")
            title, argv, _, _ = app._job("git_push", {"message": "new launcher panel"}, "S")
            self.assertIn("GitHub + GitLab", title)
            self.assertEqual(argv[argv.index("--message") + 1], "new launcher panel")
            self.assertIn("--init", argv)
            _, argv, _, _ = app._job("git_check", {}, "S")
            self.assertIn("--check", argv)
            self.assertNotIn("--init", argv)


@unittest.skipUnless(GIT, "git not installed")
class TestIgnoreKeepsCode(unittest.TestCase):
    """.gitignore must not drop code: an unanchored `cad/` once hid ros2_ws/src/fsdsim/fsdsim/cad/ (code) along
    with the confidential top-level cad/ folder."""

    def test_no_source_file_is_ignored(self):
        src = []
        for top in ("ros2_ws/src", "perf", "tools", "tests", "rl", "isaac", "isaaclab"):
            for dirpath, dirs, files in os.walk(os.path.join(ROOT, top)):
                dirs[:] = [d for d in dirs if d not in ("__pycache__", "uploads", "build", "install", "log")]
                src += [os.path.relpath(os.path.join(dirpath, f), ROOT) for f in files
                        if f.endswith((".py", ".js", ".html", ".css", ".yaml", ".m", ".csv", ".bat", ".pyw"))]
        self.assertGreater(len(src), 100)
        tmp = tempfile.mkdtemp()
        try:                                             # a scratch repository with this project's .gitignore
            shutil.copy(os.path.join(ROOT, ".gitignore"), tmp)
            _git("init", "-q", cwd=tmp)
            r = subprocess.run(["git", "check-ignore", "--no-index", "--stdin"], cwd=tmp, input="\n".join(src),
                               capture_output=True, text=True)
            ignored = [p for p in r.stdout.splitlines() if p.strip()]
            conf = ["cad/part.stp", "assets/lr26/lr26_body.usda", "lapsim/LR26_LapSim/config/lr26_params.m"]
            r = subprocess.run(["git", "check-ignore", "--no-index", "--stdin"], cwd=tmp, input="\n".join(conf),
                               capture_output=True, text=True)
            kept_out = r.stdout.split()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        self.assertEqual(ignored, [], "code files ignored by .gitignore")
        self.assertEqual(sorted(kept_out), sorted(conf), "confidential files must stay ignored")


if __name__ == "__main__":
    unittest.main()
