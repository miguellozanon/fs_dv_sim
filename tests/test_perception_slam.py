"""LiDAR cone detection, LiDAR + camera fusion, cone SLAM, explore-then-race and sensor fault injection."""

import math
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "ros2_ws", "src", "fsdsim"), os.path.join(ROOT, "tools")]
CONFIG = os.path.join(ROOT, "ros2_ws", "src", "fsd_sim_bringup", "config")

import numpy as np  # noqa: E402

from fsdsim import faults as F  # noqa: E402
from fsdsim.config import load_all_configs  # noqa: E402
from fsdsim.perception import ConeSlam, LidarConeDetector, fuse  # noqa: E402
from fsdsim.sensors.lidar import LidarModel  # noqa: E402
from fsdsim.track import Cone, ConeColor  # noqa: E402


def local_truth(cones, pose):
    x0, y0, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c * (q.x - x0) + s * (q.y - y0), -s * (q.x - x0) + c * (q.y - y0)] for q in cones])


class TestLidarCones(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.veh, cls.sens, _ = load_all_configs(CONFIG)
        cls.cones = [Cone(6.0, 1.7, ConeColor.BLUE), Cone(6.0, -1.7, ConeColor.YELLOW), Cone(10.0, 1.8, ConeColor.BLUE),
                     Cone(10.0, -1.8, ConeColor.YELLOW), Cone(8.0, 2.0, ConeColor.ORANGE_BIG), Cone(14.0, -1.5, ConeColor.YELLOW),
                     Cone(4.0, 0.0, ConeColor.ORANGE_SMALL)]
        cls.lidar = LidarModel(cls.sens.lidar, np.random.default_rng(0))

    def detect(self, attitude=(0.0, 0.0, 0.0), cloud_edit=None):
        cloud = self.lidar.scan((0, 0, 0), (0, 0, 0), self.cones, attitude)
        if cloud_edit:
            cloud = cloud_edit(cloud)
        det = LidarConeDetector(self.sens.lidar, self.veh)
        return det.detect(cloud), det

    def check(self, found, tol=0.06):
        T = local_truth(self.cones, (0, 0, 0))
        self.assertEqual(len(found), len(self.cones))
        unknown = 0
        for c in found:
            d = np.linalg.norm(T - [c["x"], c["y"]], axis=1)
            self.assertLess(d.min(), tol)
            # a single scan may leave a colour open (too few points on the stripe); it is never a wrong colour
            self.assertIn(c["color"], (int(self.cones[int(d.argmin())].color), 4))
            unknown += c["color"] == 4
        self.assertLessEqual(unknown, 3)

    def test_positions_colours_and_big_cone(self):
        found, det = self.detect()
        self.check(found)
        self.assertEqual(sum(c["big"] for c in found), 1)
        self.assertAlmostEqual(det.last["ground_pitch_deg"], 0.0, delta=0.05)

    def test_pitched_and_rolled_body_still_finds_the_cones(self):
        found, det = self.detect(attitude=(math.radians(1.0), math.radians(0.8), -0.008))
        self.check(found, tol=0.25)                    # the detector does not know the attitude: positions shift a little
        self.assertAlmostEqual(abs(det.last["ground_pitch_deg"]), 0.8, delta=0.15)
        self.assertAlmostEqual(abs(det.last["ground_roll_deg"]), 1.0, delta=0.15)

    def test_own_car_points_are_ignored(self):
        def add_car(cloud):                            # a blob where the front tyre is seen from the sensor on the nose
            R, t = LidarConeDetector(self.sens.lidar, self.veh).R, np.array([self.sens.lidar.mount.x, 0.0, self.sens.lidar.mount.z])
            blob = np.random.default_rng(1).normal([0.85, 0.61, 0.25], 0.08, (300, 3))
            pts = (blob - t) @ R
            out = {k: np.concatenate([v, np.zeros(300, v.dtype)]) for k, v in cloud.items() if k != "xyz"}
            out["xyz"] = np.vstack([cloud["xyz"], pts.astype(np.float32)])
            return out
        found, det = self.detect(cloud_edit=add_car)
        self.check(found)
        self.assertGreaterEqual(det.last["own_car"], 300)

    def test_fusion_takes_the_camera_colour(self):
        lid = [{"x": 6.0, "y": 1.7, "range": 6.2, "color": 4}, {"x": 10.0, "y": -1.8, "range": 10.2, "color": 4}]
        cam = [{"x": 6.3, "y": 1.75, "range": 6.5, "color": 0}, {"x": 20.0, "y": 0.0, "range": 20.0, "color": 1},
               {"x": 5.0, "y": -3.0, "range": 5.8, "color": 1}]
        out = fuse(lid, cam)
        self.assertEqual([c["color"] for c in out], [0, 4])            # second LiDAR cone: no camera cone in its direction
        self.assertEqual(len(fuse(lid, cam, keep_camera_only=True)), 3)   # + the near camera-only cone (not the one at 20 m)


class TestSlam(unittest.TestCase):
    def test_dead_reckoning_with_side_slip(self):
        s = ConeSlam()
        s.set_pose(0, 0, 0)
        for i in range(1001):
            s.predict(i * 0.01, 10.0, 0.5, vy=0.0)
        r = 10.0 / 0.5
        self.assertAlmostEqual(s.x, r * math.sin(5.0), places=2)
        self.assertAlmostEqual(s.y, r * (1 - math.cos(5.0)), places=2)

    def test_map_corrects_a_wrong_pose(self):
        rng = np.random.default_rng(0)
        truth = np.array([[x, y] for x in range(0, 40, 4) for y in (-1.75, 1.75)], float)
        s = ConeSlam(gain=0.5)
        s.set_pose(0, 0, 0)

        def observe(pose):
            T = local_truth([Cone(a, b, ConeColor.BLUE) for a, b in truth], pose) + rng.normal(0, 0.02, truth.shape)
            return [{"x": float(a), "y": float(b), "color": 0} for a, b in T if 0.5 < a < 16]

        for k in range(40):                              # stand still: build the map
            s.predict(k * 0.1, 0.0, 0.0)
            s.update(k * 0.1, observe((0, 0, 0)))
        self.assertGreaterEqual(len(s.cones()), 6)
        s.set_pose(0.4, -0.3, 0.03)                      # the odometry is suddenly wrong; the car has not moved
        for k in range(40, 70):
            s.predict(k * 0.1, 0.0, 0.0)
            s.update(k * 0.1, observe((0, 0, 0)))
        self.assertLess(math.hypot(s.x, s.y), 0.05)
        self.assertLess(abs(s.yaw), 0.005)

    def test_false_detections_are_dropped(self):
        s = ConeSlam()
        s.set_pose(0, 0, 0)
        s.predict(0.0, 0, 0)
        s.update(0.0, [{"x": 5.0, "y": 0.0, "color": 0}])
        s.update(3.0, [])
        self.assertEqual(len(s.lm), 0)


class TestClosedLoopPerception(unittest.TestCase):
    def run_cl(self, **kw):
        import closed_loop_eval as C
        sim, res, log, _ = C.run(kw.pop("mission", "autocross"), CONFIG, verbose=False, **kw)
        acc = sim.pipeline.accuracy([(c.x, c.y, int(c.color)) for c in sim.track.cones if not c.knocked])
        s = sim.vehicle.state
        p = sim.pipeline.pose()
        return sim, res, acc, math.hypot(p[0] - s.x, p[1] - s.y)

    def test_fusion_slam_drives_a_clean_lap_without_ground_truth(self):
        sim, res, acc, pose_err = self.run_cl(perception="fusion", slam=True)
        self.assertEqual((res.state, res.cones_hit, res.off_course), ("FINISHED", 0, 0))
        self.assertGreaterEqual(acc["seen_share"], 0.98)
        self.assertLessEqual(acc["false"], 1)
        self.assertLess(acc["pos_rms"], 0.12)
        self.assertGreater(acc["colour_ok"], 0.97)
        self.assertLess(pose_err, 0.25)

    def test_explore_then_race_on_the_map(self):
        sim, res, acc, pose_err = self.run_cl(mission="trackdrive", laps=3, perception="lidar", slam=True, race_after=1,
                                              line="racing", dt=0.04)
        self.assertEqual((res.state, res.cones_hit, res.off_course), ("FINISHED", 0, 0))
        self.assertNotIn("error", sim.race_info)
        self.assertLess(res.lap_times[2], 0.55 * res.lap_times[0])      # the mapped lap is driven more than 45 % faster
        self.assertLess(pose_err, 0.4)

    def test_lidar_dropout_degrades_but_is_survived(self):
        sim, res, acc, _ = self.run_cl(perception="fusion", slam=True, faults="lidar:dropout@14+1.5")
        self.assertEqual(res.state, "FINISHED")
        self.assertGreater(sim.faults.counts["lidar:dropout"], 10)


class TestFaults(unittest.TestCase):
    def test_parse_and_errors(self):
        f = F.parse("lidar:dropout@20+3, gnss:jump=1.5@10+30; imu:bias=0.02@0")
        self.assertEqual([(x.sensor, x.kind, x.value, x.start, x.end) for x in f],
                         [("lidar", "dropout", 0.0, 20.0, 23.0), ("gnss", "jump", 1.5, 10.0, 40.0), ("imu", "bias", 0.02, 0.0, math.inf)])
        for bad in ("lidar:dropout", "radar:dropout@1", "lidar:melt@1", "lidar:bias=1@1", "imu:bias@1"):
            with self.assertRaises(ValueError):
                F.parse(bad)
        self.assertFalse(F.FaultInjector(""))

    def test_kinds(self):
        inj = F.FaultInjector("imu:dropout@1+1,wheels:scale=1.1@0,imu:bias=0.5@3+1,gnss:jump=2@0,wheels:freeze@5+1,camera:latency=0.3@0")
        imu = lambda t: {"type": "imu", "stamp": t, "angular_velocity": [0.0, 0.0, 0.1]}      # noqa: E731
        wh = lambda t, r: {"type": "vehicle_sensors", "stamp": t, "wheel_rpm": [r] * 4}       # noqa: E731
        self.assertEqual(len(inj.apply([imu(0.5)], 0.5)), 1)
        self.assertEqual(inj.apply([imu(1.5)], 1.5), [])
        self.assertAlmostEqual(inj.apply([imu(3.5)], 3.5)[0]["angular_velocity"][2], 0.6)
        self.assertAlmostEqual(inj.apply([wh(4.0, 100.0)], 4.0)[0]["wheel_rpm"][0], 110.0)
        frozen = inj.apply([wh(5.5, 500.0)], 5.5)[0]
        self.assertAlmostEqual(frozen["wheel_rpm"][0], 110.0)             # the last good value ...
        self.assertEqual(frozen["stamp"], 5.5)                            # ... with a fresh stamp
        g = inj.apply([{"type": "gnss", "stamp": 1.0, "lat": 49.0, "lon": 8.0}], 1.0)[0]
        self.assertAlmostEqual((g["lon"] - 8.0) * math.pi / 180 * 6378137.0 * math.cos(math.radians(49.0)), 2.0, places=6)
        cam = {"type": "camera_detections", "stamp": 6.0, "detections": []}
        self.assertEqual(inj.apply([cam], 6.0), [])
        self.assertEqual(inj.apply([], 6.2), [])
        self.assertEqual(inj.apply([], 6.31), [cam])
        gt = {"type": "ground_truth", "stamp": 1.5}
        self.assertEqual(inj.apply([gt], 1.5), [gt])                      # never touched

    def test_lidar_thin(self):
        inj = F.FaultInjector("lidar:thin=0.25@0")
        cloud = {"xyz": np.zeros((4000, 3), np.float32), "ring": np.zeros(4000, np.uint16)}
        out = inj.apply([{"type": "lidar", "stamp": 1.0, "cloud": cloud}], 1.0)[0]["cloud"]
        self.assertAlmostEqual(len(out["xyz"]) / 4000, 0.25, delta=0.04)
        self.assertEqual(len(out["ring"]), len(out["xyz"]))


class TestFixedLayoutEvents(unittest.TestCase):
    def test_skidpad_follows_the_known_path_with_slam_pose(self):
        """Skidpad with perception: map mode on the layout fixed by the rules (cone midpoints cannot drive a
        figure of eight: 4 cones hit before 10 Nov 2026), pose from SLAM."""
        import closed_loop_eval as C
        sim, res, _, _ = C.run("skidpad", CONFIG, verbose=False, perception="lidar", slam=True)
        self.assertEqual((res.state, res.cones_hit, res.off_course), ("FINISHED", 0, 0))
        err = [r[1] for r in sim.slam_log]
        self.assertLess(max(err), 0.25)

    def test_launch_files_leave_the_mode_to_the_driver(self):
        for name in ("lite", "isaac"):
            with open(os.path.join(ROOT, "ros2_ws", "src", "fsd_sim_bringup", "launch", name + ".launch.py")) as f:
                self.assertNotIn('"cones" if p_nodes', f.read())


if __name__ == "__main__":
    unittest.main()
