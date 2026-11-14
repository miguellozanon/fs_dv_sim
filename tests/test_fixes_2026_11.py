"""Regression tests for the bugs found in the full check of 9 Nov 2026 (no ROS / Isaac needed).

* a track CSV whose car_start puts the (CAD-length) nose past the start line: lap never timed, EBS
* acceleration from a CSV: start / finish lines snapped to a 0.25 m grid, no exit lane (stop zone, length)
* /as/select_mission with an unknown code raised inside the ROS service callback
* closed_loop_eval timed out on long imported circuits (FSA endurance, 1.3 km)
* reference driver ran ~0.2 m wide on the skidpad circles: a cone touched by the CAD front wing at 20-25 Hz
* launcher / offline demo offered 'inspection', which has no track (the simulation crashed at start)
"""

import math
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "ros2_ws", "src", "fsdsim"), os.path.join(ROOT, "tools"),
                os.path.join(ROOT, "tools", "launcher"), os.path.join(ROOT, "ros2_ws", "src", "fsd_dashboard")]
CONFIG = os.path.join(ROOT, "ros2_ws", "src", "fsd_sim_bringup", "config")
TRACKS = os.path.join(ROOT, "ros2_ws", "src", "fsd_sim_bringup", "tracks")

import numpy as np  # noqa: E402

from fsdsim.config import load_all_configs  # noqa: E402
from fsdsim.mission import ASState  # noqa: E402
from fsdsim.track import MISSIONS, build_track, load_csv, save_csv  # noqa: E402


def _front_ahead_of_line(track, veh):
    x, y, yaw = track.start_pose
    front = np.array([x + veh.front_bumper * math.cos(yaw), y + veh.front_bumper * math.sin(yaw)])
    tl = track.timing_line
    return float(np.dot(front - 0.5 * (tl.p1 + tl.p2), tl.forward))


class TestCsvStart(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.veh, _, cls.sim = load_all_configs(CONFIG)

    def test_shipped_csvs_stage_the_car_behind_the_line(self):
        for name in ("acceleration", "trackdrive_seed7", "trackdrive_seed11", "trackdrive_seed23"):
            mission = "acceleration" if name == "acceleration" else "trackdrive"
            tr = load_csv(os.path.join(TRACKS, name + ".csv"), mission, self.sim.events, self.veh)
            self.assertLess(_front_ahead_of_line(tr, self.veh), -0.2, name)

    def test_old_car_start_is_moved_back(self):
        """A file written for the old 0.60 m front overhang (car_start x = -1.70)."""
        gen = build_track("acceleration", self.sim.events, self.veh)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "acc.csv")
            save_csv(gen, p)
            with open(p) as fh:
                txt = fh.read().splitlines()
            txt = [ln if not ln.startswith("car_start") else "car_start,-1.7000,0.0000,0.000000,0,0,0" for ln in txt]
            with open(p, "w") as fh:
                fh.write("\n".join(txt) + "\n")
            tr = load_csv(p, "acceleration", self.sim.events, self.veh)
        self.assertAlmostEqual(_front_ahead_of_line(tr, self.veh), -self.sim.events.acceleration.staging_gap, places=3)

    def test_acceleration_csv_matches_generated_layout(self):
        gen = build_track("acceleration", self.sim.events, self.veh)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "acc.csv")
            save_csv(gen, p)
            tr = load_csv(p, "acceleration", self.sim.events, self.veh)
        for a, b in ((gen.timing_line, tr.timing_line), (gen.finish_line, tr.finish_line)):
            self.assertLess(np.abs(a.p1 - b.p1).max(), 1e-6)       # exactly on the big orange cones
        self.assertAlmostEqual(tr.meta["length"], gen.meta["length"], places=6)
        self.assertGreater(tr.meta["stop_length"], 90.0)
        self.assertIsNotNone(tr.stop_zone)

    def test_acceleration_from_csv_is_timed_and_stops(self):
        import closed_loop_eval as C
        _, res, _, _ = C.run("acceleration", CONFIG, verbose=False, track_file=os.path.join(TRACKS, "acceleration.csv"))
        self.assertEqual(res.state, "FINISHED")
        self.assertEqual((res.cones_hit, res.off_course), (0, 0))
        self.assertTrue(res.extra.get("stopped_in_zone"))


class TestMissionSelect(unittest.TestCase):
    def test_unknown_code_is_refused_not_raised(self):
        veh, sens, sim = load_all_configs(CONFIG)
        sim.sim.mission = "skidpad"
        s = __import__("fsdsim.sim", fromlist=["Simulation"]).Simulation(veh, sens, sim)
        ok, msg = s.mission.select_mission(42)
        self.assertFalse(ok)
        self.assertIn("unknown mission", msg)
        self.assertEqual(s.mission.state, ASState.OFF)


class TestClosedLoopTimeout(unittest.TestCase):
    def test_long_imported_circuit_gets_more_time(self):
        import closed_loop_eval as C

        class T:
            closed, laps = True, 10
            centerline = np.stack([np.cos(np.linspace(0, 2 * np.pi, 400)) * 200,
                                   np.sin(np.linspace(0, 2 * np.pi, 400)) * 200], 1)   # 1.26 km lap
        self.assertGreater(C.timeout_for("trackdrive", T), 10 * 1250 / 6.0)
        T.laps = 1
        self.assertGreater(C.timeout_for("autocross", T), 1250 / 6.0)
        T.centerline = T.centerline / 10                                                 # 126 m: the fixed value
        self.assertEqual(C.timeout_for("autocross", T), C.TIMEOUT["autocross"])


class TestSkidpadTracking(unittest.TestCase):
    def test_no_cone_at_25_hz_and_tight_line(self):
        """The ROS driver / RL harness run at 20-25 Hz; the CAD front wing (1.48 m) leaves ~0.5 m to the cones."""
        import closed_loop_eval as C
        sim, res, log, _ = C.run("skidpad", CONFIG, verbose=False, dt=0.04)
        self.assertEqual(res.state, "FINISHED")
        self.assertEqual((res.cones_hit, res.off_course), (0, 0))
        xy, v = np.stack([log["x"], log["y"]], 1), np.array(log["v"])
        cl = sim.track.centerline
        e = np.array([np.min(np.linalg.norm(cl - p, axis=1)) for p in xy[v > 1][::4]])
        self.assertLess(float(np.sqrt(np.mean(e ** 2))), 0.15)


class TestMissionLists(unittest.TestCase):
    def test_launcher_offers_only_missions_with_a_track(self):
        import launcher as L
        self.assertTrue(set(L.MISSIONS) <= set(MISSIONS), set(L.MISSIONS) - set(MISSIONS))

    def test_demo_refuses_inspection(self):
        from fsd_dashboard import demo as D
        self.assertNotIn("inspection", D.TRACK_MISSIONS)


if __name__ == "__main__":
    unittest.main()
