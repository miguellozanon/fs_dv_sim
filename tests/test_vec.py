"""Batched (GPU) car model, universal policy environment, PPO, export and integration.

NumPy everywhere; the PyTorch tests (same results as NumPy) run only where torch is installed."""

import copy
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "ros2_ws", "src")
sys.path[:0] = [os.path.join(SRC, "fsdsim"), os.path.join(SRC, "fsd_dashboard")]

from fsdsim.config import load_all_configs  # noqa: E402
from fsdsim.control.demo_driver import make_driver, make_vehicle_driver  # noqa: E402
from fsdsim.control.rl_driver import RLDriver  # noqa: E402
from fsdsim.dynamics import VehicleModel  # noqa: E402
from fsdsim.perf import model as pmodel  # noqa: E402
from fsdsim.perf import qss  # noqa: E402
from fsdsim.rl import registry  # noqa: E402
from fsdsim.rl.env import ScenarioCache, full_run  # noqa: E402
from fsdsim.rl.policy import MLPPolicy  # noqa: E402
from fsdsim.track import build_track  # noqa: E402
from fsdsim.vec import limits as L  # noqa: E402
from fsdsim.vec.backend import Backend  # noqa: E402
from fsdsim.vec.car import BatchCar  # noqa: E402
from fsdsim.vec.env import EnvConfig, VecLonEnv, event_time  # noqa: E402
from fsdsim.vec.features import FEATURES_VERSION, N_FEATURES  # noqa: E402
from fsdsim.vec.ppo import PPO, PPOConfig  # noqa: E402
from fsdsim.vec.spec import Realism, SpecBatch, SpecRanges, coverage, spec_values  # noqa: E402
from fsdsim.vec.steer import BatchFollower  # noqa: E402
from fsdsim.vec.tracks import TrackBank  # noqa: E402
from fsdsim.vec.train import EvalSuite, make_meta, save, train  # noqa: E402

CONFIG = os.path.join(SRC, "fsd_sim_bringup", "config")
VARIANTS = ([], ["m=250", "c_z=3.2"], ["drive_awd=1", "brake_bias=0.62", "P_max=50000"])

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

_BANK = {}


def bank(items=(("acceleration", 0), ("skidpad", 0), ("autocross", 7), ("autocross", 11))):
    key = tuple(items)
    if key not in _BANK:
        _BANK[key] = TrackBank.build(CONFIG, list(items))
    return _BANK[key]


def vehicles():
    return [load_all_configs(CONFIG, vehicle_overrides=list(ch))[0] for ch in VARIANTS]


class TestCarModel(unittest.TestCase):
    def test_matches_fsdsim_vehicle_model(self):
        """Same trajectories as fsdsim.dynamics.VehicleModel (performance mode), three specs at once."""
        vehs = vehicles()
        B = Backend("numpy")
        car = BatchCar(B, SpecBatch.from_values(B, [spec_values(v) for v in vehs]))
        models = [VehicleModel(v) for v in vehs]
        for m in models:
            m.reset(0, 0, 0)
        car.reset(np.arange(3), 0.0, 0.0, 0.0)
        vmax = np.zeros(3)
        for k in range(100):
            st = np.array([0.25 * np.sin(0.7 * k * 0.04 + i) for i in range(3)])
            ac = np.array([14 * np.sin(0.9 * k * 0.04 + 0.3 * i) + 2.5 for i in range(3)])
            for i, m in enumerate(models):
                m.set_command(st[i], ac[i])
                for _ in range(20):
                    m.step(0.002)
            car.step(st, ac, 0.04)
            vmax = np.maximum(vmax, car.vx)
        for i, m in enumerate(models):
            s = m.state.to_array()
            b = np.array([getattr(car, k)[i] for k in car.STATE])
            self.assertLess(np.abs(s - b).max(), 1e-9)
            self.assertGreater(vmax[i], 8.0)       # the cars really drove: a meaningful comparison

    def test_limit_tables_match_lapsim_port(self):
        veh = vehicles()[2]
        perf = veh.perf()
        B = Backend("numpy")
        sb = SpecBatch.from_vehicle(B, veh)
        vc = L.corner_speed_table(B, sb)[0]
        ref = np.array([qss.corner_speed(perf, k) for k in L.K_GRID])
        self.assertLess(np.abs(vc - ref).max(), 1e-3)
        acc, dec = L.ax_tables(B, sb)
        for i in (0, 5, 12):
            for j in (0, 10, 30):
                a, d, _, _ = pmodel.ax_limits(perf, L.V_GRID[i], L.K_GRID[j])
                self.assertAlmostEqual(acc[0, i, j], a, places=9)
                self.assertAlmostEqual(dec[0, i, j], d, places=9)


class TestSteeringAndTracks(unittest.TestCase):
    def test_pure_pursuit_matches_reference_follower(self):
        veh, _, sc = load_all_configs(CONFIG)
        bk = bank()
        B = Backend("numpy")
        bk.to(B)
        n = len(bk.infos)
        sb = SpecBatch(B, spec_values(veh), n)
        car, fol = BatchCar(B, sb), BatchFollower(B, bk)
        tid = np.arange(n)
        sp = bk.start_pose[tid]
        car.reset(np.arange(n), sp[:, 0], sp[:, 1], sp[:, 2])
        idx = bk.start_idx[tid].copy()
        refs = []
        for t in bk.infos:
            ev = copy.deepcopy(sc.events)
            ev.track_generator.seed = t.seed
            tr = build_track(t.mission, ev, veh, t.seed if t.mission == "autocross" else None)
            d = make_driver(veh, t.mission, "map", tr.width)
            d.follower.set_path(tr.centerline, tr.closed)
            d.follower.initialise(tr.start_pose[:2])
            m = VehicleModel(veh)
            m.reset(*tr.start_pose)
            refs.append((d, m))
        for _ in range(200):
            idx = fol.nearest(tid, idx, car.x, car.y)
            st = fol.steer(tid, idx, car.x, car.y, car.yaw, car.vx, sb.lf + sb.lr, sb.lr, sb.max_steer)
            for i, (d, m) in enumerate(refs):
                s = m.state
                steer, _, _ = d.follower.control((s.x, s.y, s.yaw), s.vx)
                self.assertEqual(d.follower.idx, idx[i])
                self.assertAlmostEqual(steer, st[i], places=9)
                m.set_command(steer, float(np.clip(2.0 * (11.0 - s.vx), -10, 10)))
                for _ in range(20):
                    m.step(0.002)
            car.step(st, np.clip(2.0 * (11.0 - car.vx), -10, 10), 0.04)

    def test_bank_save_load_and_marks(self):
        bk = bank()
        with tempfile.TemporaryDirectory() as d:
            f = os.path.join(d, "bank.npz")
            bk.save(f)
            b2 = TrackBank.load(f)
        self.assertEqual([t.mission for t in b2.infos], [t.mission for t in bk.infos])
        np.testing.assert_allclose(b2.infos[2].path, bk.infos[2].path)
        acc, skid = bk.infos[0], bk.infos[1]
        self.assertAlmostEqual(acc.marks[1] - acc.marks[0], 75.0, places=6)
        self.assertEqual(len(skid.marks), 5)
        self.assertAlmostEqual(event_time("skidpad", [0, 1, 6, 11, 16.4]), 5.2)


class TestEnv(unittest.TestCase):
    def test_full_throttle_acceleration_matches_fsdsim(self):
        """Timed acceleration run at full throttle: batched model vs the complete simulator."""
        veh = load_all_configs(CONFIG)[0]
        B = Backend("numpy")
        env = VecLonEnv(B, bank((("acceleration", 0),)), spec_values(veh), EnvConfig(n_envs=1, randomise=False))
        r = env.evaluate(lambda o: np.ones(len(o)), [0])[0]

        class Full:
            features_version = 1

            def act(self, f):
                return 1.0

        ref = full_run(Full(), ScenarioCache(CONFIG, []), "acceleration", 0)["raw_time"]
        self.assertAlmostEqual(r["time"], ref, delta=0.01)
        self.assertFalse(r["failed"])

    def test_step_reset_randomisation(self):
        veh = load_all_configs(CONFIG)[0]
        B = Backend("numpy", seed=3)
        env = VecLonEnv(B, bank(), spec_values(veh), EnvConfig(n_envs=32), teacher=True)
        obs = env.reset()
        self.assertEqual(obs.shape, (32, N_FEATURES))
        self.assertGreater(np.std(env.spec.m), 1.0)                 # specs differ per car
        done_any = False
        for _ in range(150):
            obs, rew, term, trunc, info = env.step(env.teacher_a)
            self.assertTrue(np.isfinite(obs).all() and np.isfinite(rew).all())
            done_any |= bool((term | trunc).any())
        self.assertTrue(done_any)
        self.assertTrue((env.t < 25).all())                          # finished cars restarted

    def test_rl_driver_features_equal_env(self):
        """The simulator's RL driver computes exactly the training inputs (features v2)."""
        veh, _, sc = load_all_configs(CONFIG)
        B = Backend("numpy")
        bk = bank((("autocross", 7), ("skidpad", 0)))
        env = VecLonEnv(B, bk, spec_values(veh), EnvConfig(n_envs=2, randomise=False, realism=Realism.off(),
                                                             flying_start=0.0), teacher=True)
        env.noise_on = False
        env.reset(tid=[0, 1])
        for _ in range(120):
            env.step(env.teacher_a)
        pol = MLPPolicy(N_FEATURES, (8,))
        pol.features_version = 2
        for i, t in enumerate(bk.infos):
            ev = copy.deepcopy(sc.events)
            ev.track_generator.seed = t.seed
            tr = build_track(t.mission, ev, veh, t.seed if t.mission == "autocross" else None)
            d = RLDriver(pol, veh, t.mission, tr.width)
            d.set_reference(tr.centerline, tr.closed)
            d.prev_a = float(env.prev_a[i])
            c = env.car
            f = d.features((c.x[i], c.y[i], c.yaw[i]), c.vx[i], int(env.idx[i]),
                           {"vy": c.vy[i], "r": c.r[i], "ax": c.ax_b[i], "ay": c.ay_b[i]})
            self.assertLess(np.abs(f - env.obs[i]).max(), 1e-12)


class TestSpecs(unittest.TestCase):
    def test_coverage(self):
        base = spec_values(load_all_configs(CONFIG)[0])
        r = SpecRanges()
        self.assertEqual(coverage(base, base, r), [])
        for ch in (["m=260"], ["c_z*1.2"], ["mu*0.9"], ["drive_awd=1"], ["brake_bias=0.6"], ["P_max=60000"]):
            self.assertEqual(coverage(spec_values(load_all_configs(CONFIG, vehicle_overrides=ch)[0]), base, r), [], ch)
        for ch in (["m=400"], ["n_max=15000"], ["c_z*2"]):
            self.assertTrue(coverage(spec_values(load_all_configs(CONFIG, vehicle_overrides=ch)[0]), base, r), ch)

    def test_randomise_stays_in_ranges(self):
        base = spec_values(load_all_configs(CONFIG)[0])
        B = Backend("numpy", seed=1)
        sb = SpecBatch(B, base, 200)
        sb.randomise(B.arange(200), SpecRanges(), Realism.off())
        for row in sb.rows_numpy()[:50]:
            self.assertEqual(coverage(row, base, SpecRanges()), [])


class TestPPOAndRegistry(unittest.TestCase):
    def test_export_folds_normalisation(self):
        a = PPO(N_FEATURES, PPOConfig(hidden=(16, 16)), seed=1)
        x = np.random.default_rng(0).normal(size=(64, N_FEATURES)) * 3 + 1
        a.norm.update(x)
        mu = a.actor.forward(a.norm(x))[:, 0]
        np.testing.assert_allclose(a.export_policy().forward(x), mu, atol=1e-12)

    def test_tiny_training_and_universal_selection(self):
        veh = load_all_configs(CONFIG)[0]
        base = spec_values(veh)
        with tempfile.TemporaryDirectory() as pol_dir:
            out = os.path.join(pol_dir, "uni_test")
            bk = bank((("acceleration", 0), ("autocross", 101)))
            suite = EvalSuite(CONFIG, base, SpecRanges(), items=[("acceleration", 0)], n_specs=1)
            meta = train(CONFIG, out_dir=out, name="uni_test", iterations=2, n_envs=16, eval_every=2,
                         ppo_cfg=PPOConfig(hidden=(16, 16), n_steps=8, bc_rounds=1, bc_steps=8, bc_epochs=2,
                                           critic_warmup=1), bank=bk, eval_suite=suite, log=lambda m: None)
            self.assertTrue(meta["universal"])
            pol, m2 = MLPPolicy.load(out)
            self.assertEqual(pol.features_version, FEATURES_VERSION)
            self.assertIn("runs", m2["validation"])
            old = os.environ.get("FSG_RL_POLICIES")
            os.environ["FSG_RL_POLICIES"] = pol_dir
            try:
                drv, desc = make_vehicle_driver(veh, "autocross", "map", 3.0, longitudinal="rl")
                self.assertIsInstance(drv, RLDriver)
                self.assertIn("universal policy 'uni_test'", desc)
                far = load_all_configs(CONFIG, vehicle_overrides=["m=420"])[0]
                drv, desc = make_vehicle_driver(far, "autocross", "map", 3.0, longitudinal="rl")
                self.assertNotIsInstance(drv, RLDriver)
                self.assertIn("does not cover", desc)
                # the RL driver drives a complete mission in fsdsim
                r = full_run(pol, ScenarioCache(CONFIG, []), "acceleration", 0, tmax=30)
                self.assertIsNotNone(r["raw_time"])
            finally:
                if old is None:
                    os.environ.pop("FSG_RL_POLICIES")
                else:
                    os.environ["FSG_RL_POLICIES"] = old

    def test_clip_output_policy(self):
        p = MLPPolicy(4, (3,))
        p.activation, p.output = "elu", "clip"
        p.layers[-1][0] *= 1000
        x = np.random.default_rng(2).normal(size=(10, 4))
        y = p.forward(x)
        self.assertTrue((np.abs(y) <= 1).all() and (np.abs(y) == 1).any())
        self.assertAlmostEqual(p.act(x[0]), y[0])
        with tempfile.TemporaryDirectory() as d:
            p.save(d, {"name": "c"})
            q, _ = MLPPolicy.load(d)
        np.testing.assert_allclose(q.forward(x), y)


class TestTools(unittest.TestCase):
    def test_spec_sweep_tool(self):
        veh = load_all_configs(CONFIG)[0]
        base = spec_values(veh)
        with tempfile.TemporaryDirectory() as d:
            pol = PPO(N_FEATURES, PPOConfig(hidden=(8,))).export_policy()
            ec = EnvConfig()
            save(os.path.join(d, "pols", "uni"), pol, make_meta("uni", base, ec.ranges, ec.realism, ec, "test", [], {}, (8,)))
            env = dict(os.environ, FSG_RL_POLICIES=os.path.join(d, "pols"))
            r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "rl_spec_sweep.py"), "--vary", "m=260,300",
                                "--events", "acceleration", "--tracks", "0", "--out", os.path.join(d, "o")],
                               capture_output=True, text=True, env=env, timeout=300)
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            with open(os.path.join(d, "o", "sweep.csv"), encoding="utf-8") as f:
                rows = f.read().splitlines()
            self.assertEqual(len(rows), 3)
            self.assertIn("lapsim_acceleration_s", rows[0])


@unittest.skipUnless(HAVE_TORCH, "PyTorch not installed")
class TestTorchBackend(unittest.TestCase):
    def test_torch_env_matches_numpy(self):
        """Same physics, features and rewards on PyTorch (float64) as on NumPy."""
        veh = load_all_configs(CONFIG)[0]
        base = spec_values(veh)
        bk = bank()
        out = []
        for B in (Backend("numpy"), Backend("torch", device="cpu", dtype="float64")):
            env = VecLonEnv(B, TrackBank(bk.infos), base, EnvConfig(n_envs=4, randomise=False, realism=Realism.off(),
                                                                    flying_start=0.0), teacher=True)
            env.noise_on = False
            obs = env.reset(tid=[0, 1, 2, 3])
            for _ in range(60):
                obs, rew, *_ = env.step(env.teacher_a)
            out.append((B.to_numpy(obs), B.to_numpy(env.car.x), B.to_numpy(rew)))
        for a, b in zip(*out):
            np.testing.assert_allclose(a, b, atol=1e-8)

    def test_torch_ppo_smoke(self):
        from fsdsim.vec.ppo_torch import train_torch

        with tempfile.TemporaryDirectory() as d:
            meta = train_torch(CONFIG, out_dir=os.path.join(d, "u"), iterations=2, n_envs=16, device="cpu",
                               ppo_cfg=PPOConfig(hidden=(16, 16), n_steps=8, bc_rounds=1, bc_steps=8, bc_epochs=1,
                                                 critic_warmup=1),
                               train_items=[("acceleration", 0), ("autocross", 101)], eval_every=2, log=lambda m: None)
            self.assertTrue(meta["universal"])
            p, _ = MLPPolicy.load(os.path.join(d, "u"))
            self.assertEqual(p.sizes[0], N_FEATURES)


    def _flaky_training(self, fail_from, fail_to, iterations=4):
        """train_torch with a car environment that runs out of GPU memory on rollout steps
        fail_from ... fail_to (counted after the imitation warm start)."""
        from fsdsim.vec import ppo_torch

        class FlakyEnv(VecLonEnv):
            calls = 0

            def step(self, action, auto_reset=True):
                if not self.teacher:
                    FlakyEnv.calls += 1
                    if fail_from <= FlakyEnv.calls <= fail_to:
                        raise RuntimeError("CUDA error: out of memory")
                return super().step(action, auto_reset)

        logs = []
        d = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(d, ignore_errors=True))
        with mock.patch.object(ppo_torch, "VecLonEnv", FlakyEnv), mock.patch.object(ppo_torch, "OOM_WAIT_S", 0.0):
            meta = ppo_torch.train_torch(
                CONFIG, out_dir=os.path.join(d, "u"), iterations=iterations, n_envs=16, device="cpu",
                ppo_cfg=PPOConfig(hidden=(16, 16), n_steps=8, bc_rounds=1, bc_steps=8, bc_epochs=1, critic_warmup=1),
                train_items=[("acceleration", 0), ("autocross", 101)], eval_every=2, log=logs.append)
        return meta, logs, os.path.join(d, "u")

    def test_torch_recovers_from_gpu_out_of_memory(self):
        """One out-of-memory error (another program on the GPU): the training waits, restarts the cars, finishes."""
        meta, logs, out = self._flaky_training(12, 12)
        self.assertTrue(any("ran out of memory" in m for m in logs), logs)
        self.assertTrue(meta["complete"])
        self.assertEqual(meta["iterations_done"], 4)
        self.assertNotIn("aborted", meta)
        self.assertTrue(os.path.exists(os.path.join(out, "agent_torch.pt")))

    def test_torch_out_of_memory_keeps_the_best_policy(self):
        """Out of memory for good: the best policy so far is saved (incomplete) and can be continued from."""
        meta, logs, out = self._flaky_training(20, 10 ** 9, iterations=6)
        self.assertEqual(meta.get("aborted"), "out of GPU memory")
        self.assertFalse(meta["complete"])
        self.assertEqual(meta["iterations_done"], 2)
        p, m = MLPPolicy.load(out)
        self.assertEqual(p.sizes[0], N_FEATURES)
        self.assertFalse(m["complete"])
        self.assertTrue(os.path.exists(os.path.join(out, "agent_torch.pt")))
        self.assertTrue(any("Continue from" in x for x in logs), logs)


if __name__ == "__main__":
    unittest.main()
