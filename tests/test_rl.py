"""RL throttle / brake: policy, features, rollouts, trainer, driver selection, dashboard integration."""

import json
import os
import shutil
import sys
import tempfile
import time
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEEDS_LAPSIM = unittest.skipUnless(os.path.isfile(os.path.join(ROOT, "lapsim", "LR26_LapSim", "config", "lr26_params.m")),
                                   "lapsim/LR26_LapSim not present (confidential, not in git): CI runs without it")
SRC = os.path.join(ROOT, "ros2_ws", "src")
sys.path[:0] = [os.path.join(SRC, "fsdsim"), os.path.join(SRC, "fsd_dashboard")]

from fsdsim.config import load_all_configs  # noqa: E402
from fsdsim.control import RLDriver, make_vehicle_driver  # noqa: E402
from fsdsim.control.rl_driver import N_FEATURES  # noqa: E402
from fsdsim.rl import registry  # noqa: E402
from fsdsim.rl.env import Scenario, ScenarioCache, full_run, rollout  # noqa: E402
from fsdsim.rl.es import spec_hash, train  # noqa: E402
from fsdsim.rl.policy import MLPPolicy  # noqa: E402

CONFIG = os.path.join(SRC, "fsd_sim_bringup", "config")


class TestPolicy(unittest.TestCase):
    def test_params_roundtrip_and_save(self):
        p = MLPPolicy(N_FEATURES, (16, 8), seed=3)
        th = p.get_params()
        self.assertEqual(th.size, p.n_params)
        q = MLPPolicy(N_FEATURES, (16, 8)).set_params(th)
        x = np.random.default_rng(0).standard_normal(N_FEATURES)
        self.assertAlmostEqual(p.act(x), q.act(x))
        with tempfile.TemporaryDirectory() as d:
            p.save(d, {"name": "t", "spec_hash": "x"})
            r, meta = MLPPolicy.load(d)
        self.assertAlmostEqual(p.act(x), r.act(x))
        self.assertEqual(meta["sizes"], [N_FEATURES, 16, 8, 1])

    def test_imitation_fit_learns_a_function(self):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((2000, 4))
        y = np.tanh(0.8 * X[:, 0] - 0.5 * X[:, 2])
        p = MLPPolicy(4, (16,))
        self.assertLess(p.fit(X, y, epochs=60), 0.01)


class TestRollouts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cache = ScenarioCache(CONFIG)

    def test_reference_rollouts_are_clean(self):
        for sc in (Scenario("acceleration", 0, 4.0), Scenario("skidpad", 0, 8.0), Scenario("autocross", 3, 8.0, 0.4, 0.6)):
            r = rollout(None, self.cache, sc, driver="reference")
            self.assertEqual(r["ended"], "", sc)
            self.assertGreater(r["distance"], 20.0, sc)
            self.assertEqual(r["cones"], 0)

    def test_features_shape_and_launch_rule(self):
        sim, drv = self.cache.get(Scenario("acceleration", 0))
        f = drv.features((0.0, 0.0, 0.0), 0.0, 0, {})
        self.assertEqual(f.shape, (N_FEATURES,))
        brake = MLPPolicy(N_FEATURES)
        brake.layers[-1][1][:] = -5.0               # always full brake
        drv.policy = brake
        self.assertGreaterEqual(drv.act(f, 0.0), RLDriver.LAUNCH_MIN)   # cannot brake at standstill
        self.assertLess(drv.act(f, 10.0), -0.9)

    def test_crash_is_penalised(self):
        full = MLPPolicy(N_FEATURES)
        full.layers[-1][1][:] = 5.0                 # always full throttle: leaves the skidpad circle
        r = rollout(full, self.cache, Scenario("skidpad", 0, 14.0))
        ref = rollout(None, self.cache, Scenario("skidpad", 0, 14.0), driver="reference")
        self.assertLess(r["fitness"], ref["fitness"])


class TestConeReset(unittest.TestCase):
    def test_reset_puts_knocked_cones_back(self):
        """Regression: knocked cones used to stay where they were pushed after Simulation.reset()."""
        from fsdsim.sim import Simulation

        v, s, c = load_all_configs(CONFIG)
        sim = Simulation(v, s, c, mission="skidpad", enable_lidar=False, enable_camera=False)
        home = [(k.x, k.y) for k in sim.track.cones]
        cone = sim.track.cones[5]
        st = sim.vehicle.state
        st.x, st.y = cone.x, cone.y                    # drive the car onto a cone
        sim._check_cones()
        self.assertTrue(cone.knocked)
        self.assertNotEqual((cone.x, cone.y), home[5])
        sim.reset()
        self.assertEqual([(k.x, k.y) for k in sim.track.cones], home)
        self.assertFalse(any(k.knocked for k in sim.track.cones))


class TestTraining(unittest.TestCase):
    def test_tiny_training_run_saves_a_usable_policy(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "pol")
            meta = train(CONFIG, ["m=270"], out, generations=1, pairs=1, workers=1, missions=("acceleration",),
                         train_seeds=(100,), val_seeds=(), log=lambda m: None)
            self.assertTrue(os.path.isfile(os.path.join(out, "policy.npz")))
            self.assertEqual(meta["changes"], ["m=270"])
            veh = load_all_configs(CONFIG, ["m=270"])[0]
            self.assertEqual(meta["spec_hash"], spec_hash(veh))
            self.assertNotEqual(meta["spec_hash"], spec_hash(load_all_configs(CONFIG)[0]))
            meta2 = train(CONFIG, ["m=270"], os.path.join(d, "pol2"), generations=1, pairs=1, workers=1,
                          missions=("acceleration",), train_seeds=(100,), val_seeds=(), log=lambda m: None, init=out)
            self.assertEqual(meta2["continued_from"], out)
            ev = meta["evaluation"][0]
            self.assertEqual(ev["mission"], "acceleration")
            self.assertIsNotNone(ev["rl"]["raw_time"], "the imitation-initialised policy must finish")
            os.environ["FSG_RL_POLICIES"] = d
            try:
                self.assertIn(registry.find_for_vehicle(veh)["dir_name"], ("pol", "pol2"))
                self.assertIsNone(registry.find_for_vehicle(load_all_configs(CONFIG)[0]))
                drv, desc = make_vehicle_driver(veh, "acceleration", longitudinal="rl")
                self.assertIsInstance(drv, RLDriver, desc)
                drv2, desc2 = make_vehicle_driver(load_all_configs(CONFIG)[0], "acceleration", longitudinal="rl")
                self.assertNotIsInstance(drv2, RLDriver)
                self.assertIn("no policy", desc2)
            finally:
                os.environ.pop("FSG_RL_POLICIES")


class TestIntegration(unittest.TestCase):
    def test_driver_node_and_launch(self):
        node = open(os.path.join(SRC, "fsd_sim_bridge", "fsd_sim_bridge", "demo_driver_node.py")).read()
        for s in ('"longitudinal"', '"/sim/driver_set"', '"/sim/driver"', "drv.idle()"):
            self.assertIn(s, node)
        for f in ("isaac.launch.py", "lite.launch.py"):
            text = open(os.path.join(SRC, "fsd_sim_bringup", "launch", f)).read()
            self.assertIn('DeclareLaunchArgument("longitudinal"', text)

    @NEEDS_LAPSIM
    def test_dashboard_switches_driver_and_trains(self):
        from fsd_dashboard.demo import DemoRunner

        tmp = tempfile.mkdtemp()
        os.environ["FSG_RL_POLICIES"] = tmp
        try:
            r = DemoRunner(CONFIG, "acceleration", rtf=8.0)
            st = r.setup
            ok, msg = st.handle({"action": "driver", "longitudinal": "rl"})
            self.assertIn("no policy", msg)                       # nothing trained yet: reference driver
            st.rl.extra_args = ["--pairs", "1", "--missions", "acceleration"]
            ok, msg = st.handle({"action": "rl_train", "generations": 1, "workers": 1})
            self.assertTrue(ok, msg)
            for _ in range(600):
                if st.rl.job.get("state") != "running":
                    break
                time.sleep(0.5)
            s = st.state()["rl"]
            self.assertEqual(s["job"]["state"], "done", s["job"].get("log_tail"))
            self.assertEqual(s["match"], "lr26_baseline")
            self.assertEqual(r.driver_desc.split(",")[0], "RL driver")   # re-sent automatically after training
        finally:
            os.environ.pop("FSG_RL_POLICIES")
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
