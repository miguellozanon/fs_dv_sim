"""Tests for the Windows<->WSL link protocol and the Isaac RTX LiDAR conversion
(no Isaac Sim / ROS needed)."""

import math
import os
import sys
import time
import types
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ros2_ws", "src", "fsdsim"))
sys.path.insert(0, os.path.join(ROOT, "isaac"))

import numpy as np  # noqa: E402

from fsdsim.config import (  # noqa: E402
    LidarConfig, SensorsConfig, SimConfig, VehicleParams, config_fingerprint, sensors_from_dict, to_dict,
)
from fsdsim.link import LinkClient, LinkServer, decode, encode  # noqa: E402
from fsdsim.sensors.lidar import LidarModel  # noqa: E402
from fsdsim.track import Cone, ConeColor  # noqa: E402
from sensor_streams import gmo_to_cloud, pointcloud_dict_to_cloud, rings_from_elevation  # noqa: E402


def _decode_frame(b):
    import struct

    hl, pl = struct.unpack("<II", b[:8])
    return decode(b[8:8 + hl], b[8 + hl:8 + hl + pl])


class TestProtocol(unittest.TestCase):
    def test_roundtrip_nested_arrays(self):
        msg = {"kind": "events", "t": np.float64(1.5), "events": [
            {"type": "lidar", "cloud": {"xyz": np.random.rand(100, 3).astype(np.float32),
                                        "ring": np.arange(100, dtype=np.uint16)}},
            {"type": "image", "data": np.zeros((4, 5, 3), np.uint8), "K": [1.0, 2.0, 3.0, 4.0]},
        ]}
        out = _decode_frame(encode(msg))
        self.assertEqual(out["t"], 1.5)
        np.testing.assert_array_equal(out["events"][0]["cloud"]["xyz"], msg["events"][0]["cloud"]["xyz"])
        self.assertEqual(out["events"][0]["cloud"]["ring"].dtype, np.uint16)
        self.assertEqual(out["events"][1]["data"].shape, (4, 5, 3))

    def test_config_survives_handshake(self):
        sens = SensorsConfig()
        sens.lidar.mount.x = 1.23
        d = _decode_frame(encode({"sensors": to_dict(sens)}))["sensors"]
        back = sensors_from_dict(d)
        self.assertEqual(back.lidar.mount.x, 1.23)
        self.assertEqual(back.camera.resolution, (1280, 720))  # list -> tuple restored
        self.assertEqual(config_fingerprint(back), config_fingerprint(sens))

    def test_socket_roundtrip_and_reconnect(self):
        srv = LinkServer("127.0.0.1", 0, log=lambda m: None)
        hello = {"kind": "hello", "n": 1}
        cli = LinkClient("127.0.0.1", srv.port, on_connect=lambda: [hello], log=lambda m: None)
        self.assertTrue(cli.connected.wait(5))
        cli.send({"kind": "events", "data": np.ones(1000, np.float32)}, droppable=False)
        got = [srv.incoming.get(timeout=5) for _ in range(3)]
        kinds = [g.get("kind") for g in got]
        self.assertEqual(kinds, ["_connected", "hello", "events"])
        self.assertEqual(float(got[2]["data"].sum()), 1000.0)
        srv.send({"kind": "cmd", "cmd": "go"})
        for _ in range(50):
            cmds = cli.poll()
            if cmds:
                break
            time.sleep(0.05)
        self.assertEqual(cmds[0]["cmd"], "go")
        cli.close()
        srv.close()


class TestGmoConversion(unittest.TestCase):
    def _scan(self):
        cfg = LidarConfig(dropout_prob=0.0, range_noise_std=0.0, motion_distortion=False)
        cones = [Cone(cfg.mount.x + 6.0, 1.0, ConeColor.BLUE), Cone(cfg.mount.x + 3.0, -2.0, ConeColor.YELLOW)]
        return cfg, LidarModel(cfg, np.random.default_rng(0)).scan((0, 0, 0), (0, 0, 0), cones)

    def test_spherical_gmo_recovers_points_and_rings(self):
        cfg, c = self._scan()
        xyz = c["xyz"].astype(np.float64)
        r = np.linalg.norm(xyz, axis=1)
        gmo = types.SimpleNamespace(
            numElements=len(xyz), elementsCoordsType="CoordsType.SPHERICAL",
            x=np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0])), y=np.degrees(np.arcsin(xyz[:, 2] / r)), z=r,
            scalar=c["intensity"] / 255.0, flags=np.ones(len(xyz), np.uint8),
            emitterId=(c["ring"].astype(np.uint32) * 7 + 3) % 64, timeOffsetNs=c["t"].astype(np.int64) + 5000,
        )
        out = gmo_to_cloud(gmo, cfg.channels, cfg.vertical_fov_deg)
        np.testing.assert_allclose(out["xyz"], c["xyz"], atol=2e-3)
        np.testing.assert_array_equal(out["ring"], c["ring"])
        self.assertEqual(int(out["t"].min()), 0)
        self.assertGreater(float(out["intensity"].max()), 50.0)  # rescaled from [0, 1]

    def test_cartesian_and_elements_layout(self):
        cfg, c = self._scan()
        elems = types.SimpleNamespace(x=c["xyz"][:, 0], y=c["xyz"][:, 1], z=c["xyz"][:, 2])
        gmo = types.SimpleNamespace(elements=elems, elementsCoordsType="CARTESIAN", numElements=len(c["xyz"]))
        out = gmo_to_cloud(gmo, cfg.channels, cfg.vertical_fov_deg)
        np.testing.assert_allclose(out["xyz"], c["xyz"], atol=1e-5)
        np.testing.assert_array_equal(out["ring"], c["ring"])

    def test_replicator_dict(self):
        _, c = self._scan()
        out = pointcloud_dict_to_cloud({"data": c["xyz"], "info": {"intensity": c["intensity"]}})
        self.assertEqual(len(out["xyz"]), len(c["xyz"]))

    def test_rings_ignore_missing_upper_beams(self):
        el = np.radians(np.array([-22.5, -10.0, 0.0]))
        self.assertEqual(rings_from_elevation(el, None, 64).tolist(), [63, 46, 32])
        self.assertTrue(math.isclose(22.5 * 2 / 63, 0.714, rel_tol=1e-2))


class TestFingerprintDiffers(unittest.TestCase):
    def test_changes_with_config(self):
        a = config_fingerprint(VehicleParams(), SensorsConfig(), SimConfig())
        v = VehicleParams()
        v.mass += 1
        self.assertNotEqual(a, config_fingerprint(v, SensorsConfig(), SimConfig()))


if __name__ == "__main__":
    unittest.main()


class TestClockSourceNodes(unittest.TestCase):
    """The nodes that publish /clock must never run their timers on sim time (deadlock)."""

    def test_relay_and_lite_use_steady_timers(self):
        src = os.path.join(ROOT, "ros2_ws", "src", "fsd_sim_bridge", "fsd_sim_bridge")
        for name in ("isaac_relay_node.py", "lite_sim_node.py"):
            text = open(os.path.join(src, name), encoding="utf-8").read()
            timers = [ln for ln in text.splitlines() if "self.create_timer(" in ln]
            self.assertTrue(timers, name)
            for ln in timers:
                self.assertIn("clock=", ln, f"{name}: timer without steady clock: {ln.strip()}")

    def test_isaac_launch_relay_not_on_sim_time(self):
        launch = open(os.path.join(ROOT, "ros2_ws", "src", "fsd_sim_bringup", "launch", "isaac.launch.py"),
                      encoding="utf-8").read()
        relay_block = launch.split('executable="isaac_relay"')[1].split("Node(")[0]
        self.assertIn('"use_sim_time": False', relay_block)


class TestMonotonicTimeline(unittest.TestCase):
    """ROS stamps must never go backwards when Isaac restarts or /sim/reset is called:
    tf2 would reject them as TF_OLD_DATA (wall-time RViz draws nothing) or wipe its buffer."""

    def _run(self, tl, ts):
        return [tl.advance(t) for t in ts]

    def test_continuous_run_is_identity(self):
        from fsdsim.timeline import MonotonicTimeline

        tl = MonotonicTimeline()
        ts = [i / 30 for i in range(300)]
        self.assertEqual(self._run(tl, ts), ts)

    def test_restart_and_reset_continue_forward(self):
        from fsdsim.timeline import MonotonicTimeline

        tl = MonotonicTimeline()
        self._run(tl, [i / 30 for i in range(1, 4380)])   # first run ends near 146 s
        last = tl.last
        tl.new_session()
        out = self._run(tl, [i / 30 for i in range(1, 100)])  # second Isaac run starts at 0
        self.assertGreater(out[0], last)
        self.assertTrue(all(b > a for a, b in zip(out, out[1:])))
        self.assertAlmostEqual(tl.to_sim(tl.to_ros(2.5)), 2.5)
        before_reset = tl.last
        out = self._run(tl, [0.0, 1 / 30])                 # /sim/reset
        self.assertGreater(out[0], before_reset)

    def test_short_previous_run_and_reconnect(self):
        from fsdsim.timeline import MonotonicTimeline

        tl = MonotonicTimeline()
        self._run(tl, [0.1, 0.2])                          # run shorter than the reorder window
        tl.new_session()
        self.assertGreater(tl.advance(0.03), 0.2)
        tl2 = MonotonicTimeline()
        self._run(tl2, [10.0, 10.1])
        tl2.new_session()                                  # same Isaac run reconnecting after a link drop
        self.assertAlmostEqual(tl2.advance(10.2), 10.2)

    def test_out_of_order_frame_is_not_a_restart(self):
        from fsdsim.timeline import MonotonicTimeline

        tl = MonotonicTimeline()
        self._run(tl, [5.0, 5.1])
        self.assertAlmostEqual(tl.advance(5.0), 5.0)
        self.assertEqual(tl.offset, 0.0)

    def test_rviz_launched_like_ros2_run(self):
        """RViz from the launch files must match the manual command that works:
        no __node remap (name=), use_sim_time as a plain --ros-args parameter."""
        for name in ("isaac.launch.py", "lite.launch.py"):
            launch = open(os.path.join(ROOT, "ros2_ws", "src", "fsd_sim_bringup", "launch", name),
                          encoding="utf-8").read()
            block = launch.split('executable="rviz2"')[1].split("Node(")[0]
            self.assertNotIn("name=", block, name)
            self.assertIn('ros_arguments=["-p", "use_sim_time:="', block, name)

    def test_isaac_launch_waits_for_relay_before_rviz(self):
        """RViz started before Isaac connected never showed data on WSL; the launch waits for
        the relay's TF self-check line, which the relay must keep printing."""
        launch = open(os.path.join(ROOT, "ros2_ws", "src", "fsd_sim_bringup", "launch", "isaac.launch.py"),
                      encoding="utf-8").read()
        self.assertIn("OnProcessIO(target_action=relay", launch)
        self.assertIn('READY_MARK = "TF self-check"', launch)
        relay = open(os.path.join(ROOT, "ros2_ws", "src", "fsd_sim_bridge", "fsd_sim_bridge", "isaac_relay_node.py"),
                     encoding="utf-8").read()
        self.assertIn('"TF self-check OK', relay.replace("f\"", "\""))
        self.assertIn('"TF self-check FAILED', relay.replace("f\"", "\""))

    def test_stamps_never_negative(self):
        src = os.path.join(ROOT, "ros2_ws", "src", "fsd_sim_bridge", "fsd_sim_bridge", "ros_interface.py")
        self.assertIn("to_time(max(0.0, self.timeline.to_ros(", open(src, encoding="utf-8").read())

    def test_ros_side_uses_timeline(self):
        src = os.path.join(ROOT, "ros2_ws", "src", "fsd_sim_bridge", "fsd_sim_bridge")
        ri = open(os.path.join(src, "ros_interface.py"), encoding="utf-8").read()
        self.assertIn("to_time(max(0.0, self.timeline.to_ros(ev[\"stamp\"])))", ri)
        relay = open(os.path.join(src, "isaac_relay_node.py"), encoding="utf-8").read()
        self.assertIn("self.io.new_session()", relay)
        self.assertIn("self.io.begin_frame(", relay)
        self.assertIn("self.io.begin_frame(", open(os.path.join(src, "lite_sim_node.py"), encoding="utf-8").read())


class TestRingDecimation(unittest.TestCase):
    def test_os1_64_from_128_profile(self):
        from sensor_streams import decimate_rings

        cfg = LidarConfig(channels=128, dropout_prob=0.0, range_noise_std=0.0, motion_distortion=False)
        cones = [Cone(cfg.mount.x + 5.0, 0.5, ConeColor.BLUE)]
        c128 = LidarModel(cfg, np.random.default_rng(0)).scan((0, 0, 0), (0, 0, 0), cones)
        c64 = decimate_rings(c128, 128, 64)
        self.assertEqual(int(c64["ring"].max()), 63)
        self.assertTrue(np.all(np.isin(np.unique(c128["ring"][c128["ring"] % 2 == 0]) // 2, c64["ring"])))
        self.assertAlmostEqual(len(c64["xyz"]) / len(c128["xyz"]), 0.5, delta=0.05)
        self.assertIs(decimate_rings(c128, 128, 128), c128)

    def test_profile_channels_from_variant(self):
        import isaac_compat as ic

        self.assertEqual(ic.profile_channels("OS1_REV7_128ch10hz1024res", 64), 128)
        self.assertEqual(ic.profile_channels("custom", 64), 64)
