"""Unit tests for the simulator core (no ROS / Isaac needed).

Run with either:
    python3 -m unittest discover -s tests -v
    python3 -m pytest tests            (if pytest is installed)
"""

import math
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ros2_ws", "src", "fsdsim"))
CONFIG = os.path.join(ROOT, "ros2_ws", "src", "fsd_sim_bringup", "config")

import numpy as np  # noqa: E402

from fsdsim import dv_can  # noqa: E402
from fsdsim.config import EventsConfig, LidarConfig, SensorsConfig, SimConfig, VehicleParams, load_all_configs  # noqa: E402
from fsdsim.dynamics import VehicleModel  # noqa: E402
from fsdsim.geometry import polyline_tangents  # noqa: E402
from fsdsim.mission import ASState  # noqa: E402
from fsdsim.sensors.camera import StereoCameraModel  # noqa: E402
from fsdsim.sensors.lidar import LidarModel  # noqa: E402
from fsdsim.sim import Simulation  # noqa: E402
from fsdsim.track import Cone, ConeColor, build_track, load_csv, save_csv  # noqa: E402


class TestConfig(unittest.TestCase):
    def test_yaml_files_load_without_unknown_keys(self):
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            veh, sens, sim = load_all_configs(CONFIG)
        msgs = [str(x.message) for x in w]
        if not os.path.isfile(os.path.join(ROOT, "lapsim", "LR26_LapSim", "config", "lr26_params.m")):
            msgs = [m for m in msgs if "LR26_LapSim folder not found" not in m]     # CI: no confidential LapSim
        self.assertEqual(msgs, [])
        self.assertEqual(sens.lidar.channels, 64)
        self.assertAlmostEqual(veh.lr, veh.wheelbase - veh.lf)
        self.assertEqual(sim.events.skidpad.inner_diameter, 15.25)


class TestTracks(unittest.TestCase):
    def setUp(self):
        self.ev, self.veh = EventsConfig(), VehicleParams()

    def _side_check(self, tr):
        """Blue cones must be left of the path, yellow right."""
        cl = tr.centerline
        tang = polyline_tangents(cl, tr.closed)
        for c in tr.cones:
            if c.color not in (ConeColor.BLUE, ConeColor.YELLOW):
                continue
            d = np.hypot(cl[:, 0] - c.x, cl[:, 1] - c.y)
            i = int(np.argmin(d))
            v = np.array([c.x, c.y]) - cl[i]
            cross = tang[i, 0] * v[1] - tang[i, 1] * v[0]
            if c.color == ConeColor.BLUE:
                self.assertGreater(cross, 0, f"blue cone {c.id} not on the left")
            else:
                self.assertLess(cross, 0, f"yellow cone {c.id} not on the right")

    def test_acceleration(self):
        tr = build_track("acceleration", self.ev, self.veh)
        self.assertEqual(tr.finish_line.p1[0], 75.0)
        self._side_check(tr)
        # front of the car 0.3 m behind the start line
        self.assertAlmostEqual(tr.start_pose[0] + self.veh.front_bumper, -0.3)

    def test_skidpad_geometry(self):
        tr = build_track("skidpad", self.ev, self.veh)
        for c in tr.cones:
            if c.color in (ConeColor.BLUE, ConeColor.YELLOW):
                d = min(math.hypot(c.x - 9.125, c.y), math.hypot(c.x + 9.125, c.y))
                self.assertTrue(abs(d - 7.625) < 0.01 or abs(d - 10.625) < 0.01, d)
        # right circle (x > 0): inner yellow, outer blue
        right_inner = [c for c in tr.cones if c.x > 0 and abs(math.hypot(c.x - 9.125, c.y) - 7.625) < 0.01]
        self.assertTrue(all(c.color == ConeColor.YELLOW for c in right_inner))
        self.assertEqual(tr.laps, 4)

    def test_closed_tracks_respect_rules(self):
        for seed in range(5):
            tr = build_track("trackdrive", self.ev, self.veh, seed=seed)
            lo, hi = self.ev.track_generator.length_range
            self.assertTrue(lo * 0.97 <= tr.length <= hi * 1.03, tr.length)
            self._side_check(tr)
            for color in (ConeColor.BLUE, ConeColor.YELLOW):
                pts = np.array([[c.x, c.y] for c in tr.cones if c.color == color])
                # each cone has a neighbour of the same colour within the max spacing (+ tolerance)
                d = np.linalg.norm(pts[:, None] - pts[None], axis=2)
                np.fill_diagonal(d, np.inf)
                self.assertLess(d.min(axis=1).max(), self.ev.track_generator.max_cone_spacing + 0.6)

    def test_csv_roundtrip(self):
        tr = build_track("trackdrive", self.ev, self.veh)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.csv")
            save_csv(tr, p)
            tr2 = load_csv(p, "trackdrive", self.ev, self.veh)
        self.assertEqual(len(tr2.cones), len(tr.cones))
        # reconstructed centerline stays close to the original one
        err = [np.min(np.linalg.norm(tr.centerline - q, axis=1)) for q in tr2.centerline[::10]]
        self.assertLess(max(err), 0.6)
        self.assertAlmostEqual(tr2.width, tr.width, delta=0.3)


class TestDynamics(unittest.TestCase):
    def test_no_reverse_and_braking(self):
        m = VehicleModel(VehicleParams())
        m.reset(0, 0, 0)
        m.set_command(0.0, -10.0)
        for _ in range(200):
            m.step(0.005)
        self.assertEqual(m.state.vx, 0.0)
        self.assertAlmostEqual(m.state.x, 0.0, places=6)

    def test_steady_state_circle(self):
        p = VehicleParams()
        m = VehicleModel(p)
        m.reset(0, 0, 0)
        m.state.vx = 8.0
        delta = math.atan(p.wheelbase / 9.125)
        for _ in range(int(5 / 0.005)):
            m.set_command(delta, 3 * (8.0 - m.state.vx))
            m.step(0.005)
        radius = m.state.speed / abs(m.state.r)
        self.assertAlmostEqual(radius, 9.125, delta=0.6)  # mild understeer
        self.assertAlmostEqual(m.ay_body, 8.0**2 / radius, delta=0.5)

    def test_kinematic_low_speed(self):
        p = VehicleParams()
        m = VehicleModel(p)
        m.reset(0, 0, 0)
        m.set_command(0.3, 1.0)
        for _ in range(400):
            m.step(0.005)
        s = m.state
        self.assertAlmostEqual(s.r, s.vx * math.tan(s.steer) / p.wheelbase, places=3)


class TestLidar(unittest.TestCase):
    def test_cone_range_and_fields(self):
        cfg = LidarConfig(dropout_prob=0.0, range_noise_std=0.0, motion_distortion=False)
        lidar = LidarModel(cfg, np.random.default_rng(0))
        cone = Cone(cfg.mount.x + 10.0, 0.0, ConeColor.BLUE)
        out = lidar.scan((0, 0, 0), (0, 0, 0), [cone])
        xyz = out["xyz"]
        above = xyz[:, 2] > -cfg.mount.z + 0.02
        cone_pts = xyz[above & (np.abs(np.arctan2(xyz[:, 1], xyz[:, 0])) < 0.05)]
        self.assertGreater(len(cone_pts), 3)
        self.assertTrue(np.all(np.abs(cone_pts[:, 0] - 10.0) < 0.12))
        self.assertEqual(out["ring"].dtype, np.uint16)
        self.assertTrue(np.all(out["t"] < 1e8))  # within one 10 Hz scan

    def test_ground_ring_distance(self):
        cfg = LidarConfig(dropout_prob=0.0, range_noise_std=0.0)
        lidar = LidarModel(cfg)
        out = lidar.scan((0, 0, 0), (0, 0, 0), [])
        ring = out["ring"]
        k = 33  # third beam below the horizon
        r = np.linalg.norm(out["xyz"][ring == k][:, :2], axis=1)
        expected = cfg.mount.z / math.tan(-lidar.elev[k])
        self.assertAlmostEqual(float(np.median(r)), expected, delta=0.05)

    def test_motion_distortion_skews_points(self):
        cfg = LidarConfig(dropout_prob=0.0, range_noise_std=0.0, motion_distortion=True)
        lidar = LidarModel(cfg)
        cone = Cone(cfg.mount.x, -5.0, ConeColor.YELLOW)  # to the right (fired late in the sweep)
        static = lidar.scan((0, 0, 0), (0, 0, 0), [cone])
        moving = lidar.scan((0, 0, 0), (2.0, 0, 0), [cone])  # 20 m/s
        def cone_x(o):
            m = (o["xyz"][:, 2] > -cfg.mount.z + 0.02) & (o["xyz"][:, 1] < -4)
            return float(np.mean(o["xyz"][m, 0]))
        self.assertLess(cone_x(moving), cone_x(static) - 0.3)


class TestCamera(unittest.TestCase):
    def test_detects_cone_ahead(self):
        sc = SensorsConfig()
        cam = StereoCameraModel(sc.camera, np.random.default_rng(0))
        cone = Cone(8.0, 0.0, ConeColor.YELLOW, id=3)
        dets = [cam.detect((0, 0, 0), [cone]) for _ in range(20)]
        hits = [d[0] for d in dets if d]
        self.assertGreater(len(hits), 15)
        z = np.mean([h["position"][2] for h in hits])
        self.assertAlmostEqual(z, 8.0 - sc.camera.mount.x, delta=0.5)
        self.assertEqual(hits[0]["color"], int(ConeColor.YELLOW))


class TestMissionAndSim(unittest.TestCase):
    def make(self, mission="acceleration", **sim_overrides):
        veh, sens, simcfg = load_all_configs(CONFIG)
        simcfg.sim.mission = mission
        for k, v in sim_overrides.items():
            setattr(simcfg.sim, k, v)
        return Simulation(veh, sens, simcfg, enable_lidar=False, enable_camera=False)

    def test_go_needs_ready_time(self):
        sim = self.make(auto_go=False)
        sim.step(0.1)
        self.assertEqual(sim.mission.state, ASState.READY)
        ok, _ = sim.go()
        self.assertFalse(ok)
        sim.step(5.0)
        ok, _ = sim.go()
        self.assertTrue(ok)
        self.assertEqual(sim.mission.state, ASState.DRIVING)

    def test_watchdog_triggers_emergency(self):
        sim = self.make(auto_go=False)
        sim.step(5.2)
        sim.go()
        sim.set_command(0.0, 2.0)
        sim.step(2.0)  # no further commands -> timeout
        self.assertEqual(sim.mission.state, ASState.EMERGENCY)

    def test_cone_hit_counts_penalty(self):
        sim = self.make(auto_go=False)
        sim.step(5.2)
        sim.go()
        sim.track.cones.append(Cone(sim.vehicle.state.x + 4.0, 0.0, ConeColor.BLUE, id=len(sim.track.cones)))
        sim._cones_xy = sim.track.cone_xy()
        sim._cone_r = np.array([c.hit_radius for c in sim.track.cones])
        for _ in range(100):
            sim.set_command(0.0, 3.0)
            sim.step(0.02)
        self.assertEqual(sim.mission.cones_hit, 1)
        self.assertEqual(sim.mission.penalty_time(), 2.0)

    def test_event_rates(self):
        sim = self.make(auto_go=False)
        counts = {}
        for _ in range(100):
            for ev in sim.step(0.01):
                counts[ev["type"]] = counts.get(ev["type"], 0) + 1
        self.assertAlmostEqual(counts["imu"], 200, delta=2)
        self.assertAlmostEqual(counts["gnss"], 10, delta=1)
        self.assertAlmostEqual(counts["vehicle_sensors"], 100, delta=1)
        self.assertIn("track", counts)
        self.assertIn("cones", counts)

    def test_imu_at_rest_measures_gravity(self):
        sim = self.make(auto_go=False)
        acc = []
        for _ in range(50):
            for ev in sim.step(0.01):
                if ev["type"] == "imu":
                    acc.append(ev["linear_acceleration"])
        self.assertAlmostEqual(float(np.mean(np.array(acc)[:, 2])), 9.81, delta=0.05)


class TestDvCan(unittest.TestCase):
    def test_status_roundtrip(self):
        data = dv_can.encode_dv_system_status(3, 2, 6, 1, 3, 7, 41, 1234)
        self.assertEqual(len(data), 5)
        d = dv_can.decode_dv_system_status(data)
        self.assertEqual(
            d,
            {"as_state": 3, "ebs_state": 2, "ami_state": 6, "steering_state": 1, "service_brake_state": 3,
             "lap_counter": 7, "cones_count_actual": 41, "cones_count_all": 1234},
        )

    def test_dynamics_frames(self):
        f1 = dv_can.encode_dv_driving_dynamics_1(54.0, 60.0, -12.5, -13.0, 0, 0, 40, 45)
        self.assertEqual(len(f1), 8)
        self.assertEqual(f1[2], (256 - 25))  # -12.5 deg / 0.5 = -25 -> two's complement
        f2 = dv_can.encode_dv_driving_dynamics_2(1.0, -2.0, 90.0)
        self.assertEqual(len(f2), 6)


if __name__ == "__main__":
    unittest.main()
