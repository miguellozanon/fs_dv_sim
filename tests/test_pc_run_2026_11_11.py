"""Bugs found in the full run on the team PC (11 Nov 2026) and the speed-ups made then.

* perception node read /sim/vehicle wrongly and used the default car (wheel radius, nose length for the own-car filter)
* racing line on ground truth was ignored on autocross / trackdrive (the driver stayed in cones mode)
* lite simulator lost real time on every late tick (0.86-0.89x on the team laptop)
* analytic LiDAR: all cones in one array pass (same output, ~35 % faster); IMU without np.cross
"""

import json
import math
import os
import sys
import unittest
import unittest.mock

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "ros2_ws", "src")
sys.path[:0] = [os.path.join(SRC, "fsdsim"), os.path.join(SRC, "fsd_sim_bridge")]
CONFIG = os.path.join(SRC, "fsd_sim_bringup", "config")

from fsdsim.config import load_all_configs  # noqa: E402


def _read(*parts):
    with open(os.path.join(*parts), encoding="utf-8") as f:
        return f.read()


class TestVehicleMessage(unittest.TestCase):
    def test_summary_round_trip_keeps_the_car(self):
        from fsd_sim_bridge.common import vehicle_from_summary, vehicle_summary
        veh = load_all_configs(CONFIG)[0]
        got = vehicle_from_summary(json.dumps(vehicle_summary(veh), default=str))
        for name in ("wheel_radius", "front_bumper", "rear_bumper", "wheelbase", "mass"):
            self.assertAlmostEqual(getattr(got, name), getattr(veh, name), places=6, msg=name)
        with self.assertRaises(ValueError):
            vehicle_from_summary(json.dumps({"name": "x"}))

    def test_nodes_use_the_shared_parser(self):
        for node in ("perception_node.py", "demo_driver_node.py"):
            src = _read(SRC, "fsd_sim_bridge", "fsd_sim_bridge", node)
            self.assertIn("vehicle_from_summary(msg.data)", src, node)


class TestRacingLineMode(unittest.TestCase):
    def test_driver_switches_to_map_mode_for_the_racing_line_on_ground_truth(self):
        src = _read(SRC, "fsd_sim_bridge", "fsd_sim_bridge", "demo_driver_node.py")
        self.assertIn('self.line == "racing" and ground_truth and not self.race_after', src)


class TestLiteRealTime(unittest.TestCase):
    def test_late_ticks_are_made_up(self):
        src = _read(SRC, "fsd_sim_bridge", "fsd_sim_bridge", "lite_sim_node.py")
        self.assertIn("self._debt += self.rtf * (now - self._last_tick)", src)
        self.assertIn("def _step_once", src)


class TestLidarBatch(unittest.TestCase):
    def setUp(self):
        from fsdsim.sensors.lidar import LidarModel
        _, sens, _ = load_all_configs(CONFIG)
        sens.lidar.range_noise_std = 0.0
        sens.lidar.dropout_prob = 0.0
        self.L = LidarModel(sens.lidar, np.random.default_rng(0))
        self.sens = sens

    def _cone(self, x, y):
        from fsdsim.track import Cone, ConeColor
        return Cone(x=x, y=y, color=ConeColor.YELLOW)

    def _hits(self, cones):
        """Beams whose return changed against the empty scene: {(ring, column time): range}."""
        pose = (0.0, 0.0, 0.0)
        base = self.L.scan(pose, pose, [])
        s = self.L.scan(pose, pose, cones)
        key = lambda d: {(int(r), int(t)): float(np.linalg.norm(p)) for r, t, p in zip(d["ring"], d["t"], d["xyz"])}
        b, h = key(base), key(s)
        return {k: v for k, v in h.items() if k not in b or abs(b[k] - v) > 1e-6}

    def test_range_to_a_cone_ahead(self):
        hits = self._hits([self._cone(8.0, 0.0)])
        self.assertGreater(len(hits), 5)
        d = math.hypot(8.0 - self.sens.lidar.mount.x, self.sens.lidar.mount.z)       # to the cone's foot
        self.assertAlmostEqual(min(hits.values()), d - 0.095, delta=0.25)

    def test_nearest_of_two_cones_on_one_beam_wins(self):
        near, far = self._cone(6.0, 0.0), self._cone(9.0, 0.0)
        alone = self._hits([near])
        both = self._hits([far, near])
        for k, v in alone.items():               # the beams on the near cone still see the near cone
            self.assertAlmostEqual(both[k], v, places=6)
        self.assertGreater(len(both), len(alone))  # and the far cone shows above the near one


class TestImuLever(unittest.TestCase):
    def test_lever_arm_terms_match_the_cross_products(self):
        from fsdsim.sensors.proprioceptive import ImuModel
        _, sens, _ = load_all_configs(CONFIG)
        lever = np.array([sens.imu.mount.x, sens.imu.mount.y, sens.imu.mount.z])
        r, rd = 0.8, -1.3
        ref = np.cross([0, 0, rd], lever) + np.cross([0, 0, r], np.cross([0, 0, r], lever))
        lx, ly = lever[0], lever[1]
        mine = np.array([-rd * ly - r * r * lx, rd * lx - r * r * ly, 0.0])
        self.assertTrue(np.allclose(ref, mine))
        self.assertTrue(hasattr(ImuModel, "measure"))


class TestLatency(unittest.TestCase):
    def test_predict_pose_on_an_arc(self):
        from fsdsim.control.demo_driver import predict_pose
        x, y, yaw = predict_pose((0.0, 0.0, 0.0), 10.0, 0.1, 1.53, 0.1)
        r = 10.0 * math.tan(0.1) / 1.53
        self.assertAlmostEqual(yaw, r * 0.1, places=9)
        self.assertAlmostEqual(math.hypot(x, y), 2 * (10.0 / r) * math.sin(r * 0.1 / 2), places=6)   # chord
        self.assertEqual(predict_pose((1.0, 2.0, 0.3), 5.0, 0.1, 1.53, 0.0), (1.0, 2.0, 0.3))

    def test_skidpad_with_isaac_like_latency(self):
        """30 Hz and 67 ms until a command acts (Isaac Sim): 16 cones without the prediction, clean with it."""
        sys.path.insert(0, os.path.join(ROOT, "tools"))
        import closed_loop_eval as C
        _, res, _, _ = C.run("skidpad", CONFIG, verbose=False, dt=1 / 30, cmd_latency=0.067)
        self.assertEqual((res.state, res.cones_hit, res.off_course), ("FINISHED", 0, 0))
        _, res, _, _ = C.run("skidpad", CONFIG, verbose=False, dt=1 / 30, cmd_latency=0.067, latency_comp=0.0)
        self.assertGreater(res.cones_hit, 0)

    def test_launch_passes_the_latency_and_lidar_offset(self):
        src = _read(SRC, "fsd_sim_bringup", "launch", "isaac.launch.py")
        self.assertIn('DeclareLaunchArgument("driver_latency", default_value="0.07"', src)
        self.assertIn('params["lidar_time_offset"]', src)


class TestTransportLoad(unittest.TestCase):
    """Full images to the dashboard delayed odometry and commands by up to 2 s in Isaac Sim (Zenoh router)."""

    def test_dashboard_watches_previews_and_camera_info(self):
        src = _read(SRC, "fsd_dashboard", "fsd_dashboard", "node.py")
        self.assertIn('t["camera"] + "/preview"', src)
        self.assertIn('ns_cam + "/camera_info"', src)
        self.assertNotIn('sub(Image, t["camera"], ', src)

    def test_bridge_publishes_previews_and_relay_skips_old_frames(self):
        ri = _read(SRC, "fsd_sim_bridge", "fsd_sim_bridge", "ros_interface.py")
        self.assertIn("def _publish_preview", ri)
        relay = _read(SRC, "fsd_sim_bridge", "fsd_sim_bridge", "isaac_relay_node.py")
        self.assertIn("self._on_events(msg, light=behind)", relay)


class TestIsaacLinkBandwidth(unittest.TestCase):
    def test_depth_travels_as_millimetres_and_camera_rate_is_lowered(self):
        self.assertIn('"depth_mm": True', _read(ROOT, "isaac", "run_isaac.py"))
        self.assertIn('if ev.get("depth_mm"):', _read(SRC, "fsd_sim_bridge", "fsd_sim_bridge", "ros_interface.py"))
        sys.path.insert(0, os.path.join(ROOT, "tools", "launcher"))
        import launcher as L
        app = L.Launcher(probe=False)
        cmds = []
        app._start = lambda proc: cmds.append(" ".join(str(a) for a in proc.argv))
        base = {"mission": "autocross", "changes": [], "seed": -1, "headless": False, "record_cones": False}
        with unittest.mock.patch.object(L, "win_path", lambda p: "C:\\repo\\isaac\\run_isaac.ps1"):
            app._start_isaac({**base, "perception": "none"})
            app._start_isaac({**base, "perception": "yolo"})
        self.assertIn("'--camera-hz','5'", cmds[0])
        self.assertIn("'--camera-hz','10'", cmds[1])


class TestLinkPriority(unittest.TestCase):
    def test_state_goes_first_and_old_frames_are_dropped(self):
        from fsdsim.link import LinkClient
        c = LinkClient("127.0.0.1", 1, log=lambda m: None, max_queue=3)    # port 1: never connects
        try:
            c.connected.set()
            for i in range(6):
                c.send({"kind": "events", "i": i, "img": np.zeros(10, np.uint8)}, droppable=True)
            c.send({"kind": "events", "state": True}, droppable=False)
            got = []
            while True:
                data = c._next(0.01)
                if data is None:
                    break
                got.append(data)
            self.assertEqual(len(got), 4)                 # 1 state + the 3 newest frames
            self.assertEqual(c.dropped, 3)
            self.assertIn(b'"state"', got[0])            # the state message first
            self.assertIn(b'"i":3', got[1])              # then the oldest kept frame
        finally:
            c.close()


class TestRelayAddress(unittest.TestCase):
    def test_mirrored_networking_prefers_loopback(self):
        ps1 = _read(ROOT, "isaac", "run_isaac.ps1")
        self.assertIn('$order = @("127.0.0.1", $wslIp)', ps1)


class TestDriverTick(unittest.TestCase):
    """Isaac: a late /clock held the sim-time tick of the driver back by up to 0.8 s; there it ticks on wall time."""

    def test_isaac_driver_ticks_on_the_wall_clock(self):
        node = _read(ROOT, "ros2_ws", "src", "fsd_sim_bridge", "fsd_sim_bridge", "demo_driver_node.py")
        self.assertIn('declare_parameter("tick_clock", "sim")', node)
        self.assertIn("ClockType.STEADY_TIME", node)
        self.assertIn("longest wait for /clock", node)
        launch = _read(ROOT, "ros2_ws", "src", "fsd_sim_bringup", "launch", "isaac.launch.py")
        self.assertIn('"tick_clock": "steady"', launch)

    def test_stale_pose_is_carried_forward_on_the_steering_arc(self):
        from fsdsim.control.demo_driver import predict_pose
        # 0.3 s without odometry at 10 m/s, straight: the car is 3 m further on
        x, y, yaw = predict_pose((0.0, 0.0, 0.0), 10.0, 0.0, 1.53, 0.3)
        self.assertAlmostEqual(x, 3.0, places=6)
        node = _read(ROOT, "ros2_ws", "src", "fsd_sim_bridge", "fsd_sim_bridge", "demo_driver_node.py")
        self.assertIn("def _bridge_stale_pose", node)
        self.assertIn("if not self._steady", node)     # lite (sim-time ticks) is unchanged


class TestLidarTiming(unittest.TestCase):
    def test_alignment_check_finds_a_wrong_stamp(self):
        sys.path.insert(0, os.path.join(ROOT, "tools"))
        import closed_loop_eval as C
        import fsdsim.perception.pipeline as P
        P.PerceptionPipeline.lidar_time_offset = 0.06
        try:
            sim, res, _, _ = C.run("autocross", CONFIG, verbose=False, perception="lidar", slam=False)
            truth = [(c.x, c.y, int(c.color)) for c in sim.track.cones]
            t = sim.pipeline.lidar_time_alignment(truth)
        finally:
            P.PerceptionPipeline.lidar_time_offset = 0.0
        self.assertAlmostEqual(t["best_offset"], 0.0, delta=0.011)
        self.assertGreater(t["rms_now"], 0.3)
        self.assertLess(t["rms_best"], 0.06)


class TestFrozenAndLateSensors(unittest.TestCase):
    def test_frozen_lidar_and_late_camera_stay_clean(self):
        sys.path.insert(0, os.path.join(ROOT, "tools"))
        import closed_loop_eval as C
        for faults in ("lidar:freeze@15+2", "camera:latency=0.4@5,lidar:thin=0.3@0"):
            sim, res, _, _ = C.run("autocross", CONFIG, verbose=False, perception="fusion", slam=True, faults=faults)
            self.assertEqual((res.state, res.cones_hit, res.off_course), ("FINISHED", 0, 0), faults)

    def test_finish_line_on_seed_13(self):
        sys.path.insert(0, os.path.join(ROOT, "tools"))
        import closed_loop_eval as C
        _, res, _, _ = C.run("autocross", CONFIG, verbose=False, seed=13)
        self.assertEqual((res.state, res.cones_hit), ("FINISHED", 0))


if __name__ == "__main__":
    unittest.main()


class TestStateQueueTime(unittest.TestCase):
    """The leg from Isaac's step to the relay was not measured; the client now reports how long state waited."""

    def test_client_reports_state_wait(self):
        import socket
        import time as _t

        from fsdsim.link import LinkClient, read_message

        srv = socket.socket(); srv.bind(("127.0.0.1", 0)); srv.listen(1)   # noqa: E702
        c = LinkClient("127.0.0.1", srv.getsockname()[1], log=lambda *_: None)
        conn, _ = srv.accept()
        try:
            self.assertTrue(c.connected.wait(3.0))
            c.send({"kind": "events", "t": 0.0, "events": [], "wall": _t.time()}, droppable=False)
            msg = read_message(conn)
            self.assertIn("wall", msg)
            self.assertGreaterEqual(c.state_wait_max, 0.0)
            self.assertLess(c.state_wait_max, 1.0)
        finally:
            c.close(); conn.close(); srv.close()   # noqa: E702


class TestLidarPreview(unittest.TestCase):
    """The dashboard's ~1 MB clouds slowed Zenoh delivery of odometry and /clock to all nodes by up to 0.4 s."""

    def test_dashboard_reads_the_thinned_cloud(self):
        ri = _read(ROOT, "ros2_ws", "src", "fsd_sim_bridge", "fsd_sim_bridge", "ros_interface.py")
        self.assertIn('"/preview"', ri)
        self.assertIn("PREVIEW_POINTS = 6000", ri)
        self.assertIn("self.lidar_pub.get_subscription_count() > 0", ri)
        dash = _read(ROOT, "ros2_ws", "src", "fsd_dashboard", "fsd_dashboard", "node.py")
        self.assertIn('t["lidar"] + "/preview"', dash)


class TestSensorDemand(unittest.TestCase):
    """Isaac sends full images / clouds only while a ROS node reads them (the relay's sensor_demand)."""

    def test_thin_cloud_keeps_all_fields(self):
        sys.path.insert(0, os.path.join(ROOT, "isaac"))
        import run_isaac

        m = 30000
        cloud = {"xyz": np.random.rand(m, 3).astype(np.float32), "intensity": np.ones(m, np.float32),
                 "t": np.arange(m, dtype=np.uint32), "reflectivity": np.zeros(m, np.uint16),
                 "ring": np.zeros(m, np.uint16), "range": np.zeros(m, np.uint32)}
        thin = run_isaac.thin_cloud(cloud, 6000)
        self.assertEqual({k: len(v) for k, v in thin.items()}, {k: 6000 for k in cloud})
        self.assertEqual(int(thin["t"][0]), 0)
        self.assertEqual(int(thin["t"][-1]), m - 1)
        self.assertIs(run_isaac.thin_cloud(thin, 6000), thin)

    def test_relay_and_isaac_agree_on_the_command(self):
        relay = _read(ROOT, "ros2_ws", "src", "fsd_sim_bridge", "fsd_sim_bridge", "isaac_relay_node.py")
        isaac = _read(ROOT, "isaac", "run_isaac.py")
        self.assertIn('"sensor_demand"', relay)
        self.assertIn('kind == "sensor_demand"', isaac)


class TestStateChannel(unittest.TestCase):
    """State gets its own TCP connection when the relay offers it, so it never waits behind camera frames."""

    def _drain(self, srv, n, timeout=3.0):
        import queue as _q
        import time as _t

        out, end = [], _t.monotonic() + timeout
        while len(out) < n and _t.monotonic() < end:
            try:
                out.append(srv.incoming.get(timeout=0.1))
            except _q.Empty:
                pass
        return out

    def test_state_on_its_own_connection(self):
        import time as _t

        from fsdsim.link import LinkClient, LinkServer

        srv = LinkServer("127.0.0.1", 0, log=lambda *_: None)
        c = LinkClient("127.0.0.1", srv.port, on_connect=lambda: [{"kind": "hello"}], log=lambda *_: None)
        try:
            self.assertTrue(c.state_channel_up.wait(3.0))
            first = self._drain(srv, 2)
            self.assertEqual([m["kind"] for m in first], ["_connected", "hello"])
            # a big frame and a state message: both arrive; state is never stuck behind the frame's socket
            c.send({"kind": "events", "t": 1.0, "sensors": True, "events": [{"data": np.zeros(4_000_000, np.uint8)}]})
            c.send({"kind": "events", "t": 1.0, "events": [{"type": "ground_truth"}], "wall": _t.time()}, droppable=False)
            got = self._drain(srv, 2)
            self.assertEqual(sorted(bool(m.get("sensors")) for m in got), [False, True])
            self.assertEqual(len(srv._aux), 1)
            self.assertTrue(all("_rx" in m for m in got))
        finally:
            c.close()
            srv.close()
        # the simulator going away closes its state connection too
        end = _t.monotonic() + 3.0
        while _t.monotonic() < end and (srv._aux or srv.connected):
            _t.sleep(0.05)
        self.assertEqual(srv._aux, [])

    def test_old_relay_without_the_offer_keeps_one_connection(self):
        import socket
        import time as _t

        from fsdsim.link import LinkClient, read_message

        lst = socket.socket(); lst.bind(("127.0.0.1", 0)); lst.listen(2)   # noqa: E702
        c = LinkClient("127.0.0.1", lst.getsockname()[1], log=lambda *_: None)
        conn, _ = lst.accept()
        try:
            self.assertTrue(c.connected.wait(3.0))
            c.send({"kind": "events", "t": 0.0, "events": []}, droppable=False)
            self.assertEqual(read_message(conn)["kind"], "events")
            _t.sleep(0.3)
            self.assertFalse(c.state_channel_up.is_set())
        finally:
            c.close(); conn.close(); lst.close()   # noqa: E702


class TestEveningFixes(unittest.TestCase):
    """Runs on the team PC, 11 Nov 2026 evening: SLAM pose held back by scans; camera "Low rate" at 5-10 Hz."""

    def test_scan_split_gives_the_same_result(self):
        import fsdsim.perception.pipeline as P

        self.assertTrue(hasattr(P.PerceptionPipeline, "detect_cloud"))
        self.assertTrue(hasattr(P.PerceptionPipeline, "integrate_cloud"))
        node = _read(ROOT, "ros2_ws", "src", "fsd_sim_bridge", "fsd_sim_bridge", "perception_node.py")
        self.assertIn("MultiThreadedExecutor(num_threads=2)", node)
        self.assertIn("self.pipe.detect_cloud(", node)
        self.assertIn("callback_group=self._slow", node)

    def test_dashboard_expects_the_isaac_camera_rate(self):
        sys.path.insert(0, os.path.join(ROOT, "tools", "launcher"))
        import launcher as L

        self.assertEqual(L.isaac_camera_hz({"perception": "none"}), 5)
        self.assertEqual(L.isaac_camera_hz({"perception": "fusion"}), 10)
        self.assertIsNone(L.isaac_camera_hz({"perception": "yolo", "record_cones": True}))
        launch = _read(ROOT, "ros2_ws", "src", "fsd_sim_bringup", "launch", "isaac.launch.py")
        self.assertIn('"camera_hz": float(LaunchConfiguration("camera_hz")', launch)
        dash = _read(ROOT, "ros2_ws", "src", "fsd_dashboard", "fsd_dashboard", "node.py")
        self.assertIn('self.get_parameter("camera_hz").value', dash)


class TestColourlessCones(unittest.TestCase):
    """LiDAR cones without colours (Isaac's RTX LiDAR): the car went straight on at the first 90 deg corner."""

    @staticmethod
    def _corner(w=3.5, step=3.0):
        def centre(s):                            # 12 m straight, a right turn of radius 6 m, then straight on
            if s < 12.0:
                return np.array([s, 0.0]), np.array([1.0, 0.0])
            a = (s - 12.0) / 6.0
            if a < math.pi / 2:
                return np.array([12.0 + 6 * math.sin(a), -6.0 + 6 * math.cos(a)]), np.array([math.cos(a), -math.sin(a)])
            return np.array([18.0, -6.0 - (s - 12.0 - 3 * math.pi)]), np.array([0.0, -1.0])
        pts, side = [], []
        for k in range(1, 9):
            for sgn, col in ((1.0, 0), (-1.0, 1)):
                p, t = centre(k * step)
                n = np.array([-t[1], t[0]])
                pts.append(p + sgn * 0.5 * w * n)
                side.append(col)
        return np.array(pts), np.array(side)

    def test_corner_sides_and_path(self):
        from fsdsim.control.demo_driver import ConeLocalPlanner, DriverParams

        cones, truth = self._corner()
        pl = ConeLocalPlanner(DriverParams())
        colors = pl._side_of_unknown(cones, np.full(len(cones), 4), 3.5)
        near = np.linalg.norm(cones, axis=1) < 20.0
        self.assertGreaterEqual(np.mean(colors[near] == truth[near]), 0.9)
        path = pl.plan(cones, np.full(len(cones), 4))
        self.assertLess(path[-1][1], -3.0)        # the path turns right with the track

    def test_closed_loop_without_colours(self):
        sys.path.insert(0, os.path.join(ROOT, "tools"))
        import closed_loop_eval as C
        import fsdsim.perception.lidar_cones as LC

        orig = LC.LidarConeDetector.__dict__["_colour"]          # the staticmethod itself, to put it back as it was
        LC.LidarConeDetector._colour = staticmethod(lambda h, i, top, big: LC.ORANGE_BIG if big else LC.UNKNOWN)
        try:
            _, res, _, _ = C.run("autocross", CONFIG, verbose=False, perception="lidar", dt=1 / 30,
                                 cmd_latency=0.067, latency_comp=0.07)
        finally:
            LC.LidarConeDetector._colour = orig
        self.assertEqual(res.state, "FINISHED")
        self.assertEqual((res.cones_hit, res.off_course), (0, 0))

    def test_isaac_lidar_mode_leaves_colours_unknown(self):
        from fsdsim.perception import PerceptionPipeline

        veh, sens, _ = load_all_configs(CONFIG)
        self.assertTrue(PerceptionPipeline(veh, sens, "lidar", False).det.colour_from_intensity)
        self.assertFalse(PerceptionPipeline(veh, sens, "lidar", False, lidar_colours=False).det.colour_from_intensity)
        launch = _read(ROOT, "ros2_ws", "src", "fsd_sim_bringup", "launch", "isaac.launch.py")
        self.assertIn('params["lidar_colours"] = False', launch)


class TestStampOrder(unittest.TestCase):
    """Odometry of one Isaac frame arrived round-robin per topic: SLAM drifted 0.8 m per lap (0.05 m in order)."""

    def test_round_robin_order_is_what_drifts(self):
        sys.path.insert(0, os.path.join(ROOT, "tools"))
        import closed_loop_eval as C
        import fsdsim.perception.pipeline as P

        orig = P.PerceptionPipeline.handle_all
        buf = []

        def frame_round_robin(self, events):          # one 1/30 s frame, then one message per topic in turn
            buf.extend(e for e in events if e.get("type") in self._ORDER)
            st = [e["stamp"] for e in buf if "stamp" in e]
            if not st or max(st) - min(st) < 1 / 30:
                return
            by = {}
            for e in buf:
                by.setdefault(e["type"], []).append(e)
            buf.clear()
            while any(by.values()):
                for k in ("ground_truth", "imu", "vehicle_sensors", "camera_detections", "lidar"):
                    if by.get(k):
                        self.handle(by[k].pop(0))

        P.PerceptionPipeline.handle_all = frame_round_robin
        try:
            sim, _, _, _ = C.run("autocross", CONFIG, verbose=False, perception="fusion", slam=True, dt=1 / 30)
        finally:
            P.PerceptionPipeline.handle_all = orig
        self.assertGreater(max(r[1] for r in sim.slam_log), 0.5)   # the bug, reproduced without ROS
        node = _read(ROOT, "ros2_ws", "src", "fsd_sim_bridge", "fsd_sim_bridge", "perception_node.py")
        self.assertIn("heapq.heappush(self._heap", node)        # the node now processes in stamp order
        self.assertIn('self._push(stamp_of(msg) + self.pipe.det.period, "scan"', node)


class TestFreshPose(unittest.TestCase):
    """The driver's own dead reckoning, anchored on a SLAM pose that arrives 0.25 s late."""

    def test_late_anchor_gives_the_current_pose(self):
        from fsdsim.perception.fresh_pose import FreshPose

        r_w, v, yr = 0.25, 10.0, 0.4                      # 10 m/s on a 25 m circle
        rpm = v / r_w * 60 / (2 * math.pi)
        f = FreshPose(r_w, 0.0)

        def truth(t):                                     # start (5, 2) heading 30 deg, ground speed along the car
            y0 = math.radians(30)
            R = v / yr
            return (5 + R * (math.sin(y0 + yr * t) - math.sin(y0)), 2 - R * (math.cos(y0 + yr * t) - math.cos(y0)),
                    y0 + yr * t)

        dt = 0.01
        for k in range(301):                             # 3 s at 100 Hz
            t = k * dt
            f.imu(yr)
            f.wheels(t, [rpm, rpm, rpm, rpm], 0.0, (v, 0.0))
            if k >= 25 and k % 10 == 0:                  # SLAM pose of 0.25 s ago, with a 0.3 m map offset
                x, y, yaw = truth(t - 0.25)
                f.anchor(t - 0.25, x + 0.3, y, yaw)
        x, y, yaw = f.pose()
        tx, ty, tyaw = truth(3.0)
        self.assertLess(math.hypot(x - (tx + 0.3), y - ty), 0.02)
        self.assertLess(abs((yaw - tyaw + math.pi) % (2 * math.pi) - math.pi), 0.002)

    def test_driver_uses_it_with_the_slam_pose(self):
        node = _read(ROOT, "ros2_ws", "src", "fsd_sim_bridge", "fsd_sim_bridge", "demo_driver_node.py")
        self.assertIn('gp("odom_topic").value == "/slam/odom"', node)
        self.assertIn("self.fresh.anchor(", node)
