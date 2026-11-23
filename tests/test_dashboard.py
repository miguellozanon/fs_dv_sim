"""Dashboard tests: model, imaging, HTTP/SSE server, offline demo, and the ROS node
against lightweight stand-ins for rclpy and the message packages (no ROS needed)."""

import base64
import json
import math
import os
import sys
import time
import types
import unittest
import urllib.request
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEEDS_LAPSIM = unittest.skipUnless(os.path.isfile(os.path.join(ROOT, "lapsim", "LR26_LapSim", "config", "lr26_params.m")),
                                   "lapsim/LR26_LapSim not present (confidential, not in git): CI runs without it")
SRC = os.path.join(ROOT, "ros2_ws", "src")
sys.path[:0] = [os.path.join(SRC, "fsdsim"), os.path.join(SRC, "fsd_dashboard"), os.path.join(SRC, "fsd_sim_bridge")]

import numpy as np  # noqa: E402

from fsd_dashboard import imaging  # noqa: E402
from fsd_dashboard.model import ClientCursor, DashboardModel, Stream  # noqa: E402
from fsd_dashboard.server import DashboardServer  # noqa: E402

CONFIG = os.path.join(SRC, "fsd_sim_bringup", "config")


def _status(**kw):
    st = {"state": 3, "mission": 3, "lap_count": 0, "laps_target": 10, "mission_complete": False, "cones_hit": 0,
          "off_course": 0, "mission_time": 1.0, "last_lap_time": 0.0, "best_lap_time": 0.0, "penalty_time": 0.0,
          "message": "AS Driving"}
    st.update(kw)
    return st


class TestModel(unittest.TestCase):
    def test_laps_penalties_and_events(self):
        m = DashboardModel()
        m.set_status(_status(state=2, message="AS Ready"))
        m.set_status(_status())
        m.set_status(_status(lap_count=1, last_lap_time=22.5, mission_time=23.0))
        m.set_status(_status(lap_count=1, cones_hit=1, penalty_time=2.0, mission_time=24.0))
        texts = [e["text"] for e in m.delta(ClientCursor())["events"]]
        self.assertIn("Lap 1 completed in 22.500 s", texts)
        self.assertTrue(any(t.startswith("Cone hit - 1") for t in texts))
        self.assertEqual(m.laps, [{"lap": 1, "time": 22.5}])

    def test_delta_sends_static_data_once_and_only_new_rows(self):
        m = DashboardModel()
        m.set_track([[0, 0], [10, 0], [10, 10]], True)
        m.set_cones([(0, 0, 1.0, 2.0, False)])
        for i in range(10):
            m.set_pose(i * 0.1, i, 0, 0, 5, 0, 0)
        cur = ClientCursor()
        d1 = m.delta(cur)
        self.assertTrue(d1["reset"])
        self.assertIn("track", d1)
        self.assertIn("cones", d1)
        self.assertEqual(len(d1["telemetry"]), 10)
        m.set_pose(1.0, 10, 0, 0, 5, 0, 0)
        d2 = m.delta(cur)
        self.assertNotIn("track", d2)
        self.assertNotIn("cones", d2)
        self.assertEqual(len(d2["telemetry"]), 1)
        self.assertNotIn("events", d2)

    def test_restart_clears_trail_and_logs(self):
        m = DashboardModel()
        for i in range(50):
            m.set_pose(i * 0.1, i, 0, 0, 5, 0, 0)
        m.set_pose(0.0, 0, 0, 0, 0, 0, 0)
        d = m.delta(ClientCursor())
        self.assertIn("Simulation restarted", [e["text"] for e in d["events"]])
        self.assertLessEqual(len(d["trail"]), 1)

    def test_lidar_roundtrip(self):
        m = DashboardModel()
        pts = np.array([[1.23, -4.56, 0.3], [100.0, 50.0, 0.0], [np.nan, 0, 0]])
        m.set_lidar(1.0, pts)
        d = m.delta(ClientCursor())["lidar"]
        q = np.frombuffer(base64.b64decode(d["data"]), "<i2").reshape(-1, 3) * d["res"]
        self.assertEqual(d["n"], 2)
        np.testing.assert_allclose(q, pts[:2], atol=0.011)

    def test_stream_rate_uses_stamps_and_detects_stall(self):
        s = Stream("lidar", "LiDAR", "/lidar/points", 10.0)
        for i in range(30):
            s.tick(i * 0.1, wall=100.0 + i * 0.125)   # sim running at 0.8x real time
        self.assertAlmostEqual(s.rate(), 10.0, places=3)
        self.assertEqual(s.status(103.7)[0], "good")
        self.assertEqual(s.status(110.0), ("critical", "Stalled"))
        self.assertEqual(Stream("x", "X", "/x").status(0.0)[0], "none")
        ctl = Stream("control", "Control command", "/control/command", None, active_states=(2, 3))
        ctl.tick(1.0, wall=100.0)
        self.assertEqual(ctl.status(105.0, as_state=3)[0], "critical")    # stack died while driving
        self.assertEqual(ctl.status(105.0, as_state=5), ("none", "Idle"))  # nothing expected after Finished


class TestImaging(unittest.TestCase):
    def test_png_and_depth(self):
        rgb = (np.arange(4 * 6 * 3) % 255).astype(np.uint8).reshape(4, 6, 3)
        png = imaging.encode_png(rgb)
        self.assertTrue(png.startswith(b"\x89PNG"))
        idat = png[png.index(b"IDAT") + 4: png.index(b"IEND") - 8]
        raw = zlib.decompress(idat)
        self.assertEqual(raw[1:19], rgb[0].tobytes())
        depth = np.array([[1.0, np.inf], [np.nan, 50.0]], np.float32)
        col = imaging.colorize_depth(depth)
        self.assertEqual(col.shape, (2, 2, 3))
        np.testing.assert_array_equal(col[0, 1], imaging.NO_DEPTH)
        self.assertGreater(int(col[0, 0].sum()), int(col[1, 1].sum()))   # near is lighter

    def test_from_ros_image(self):
        bgr = np.zeros((2, 3, 3), np.uint8)
        bgr[..., 0] = 200                                        # blue in BGR
        kind, rgb = imaging.from_ros_image("bgr8", 2, 3, 9, bgr.tobytes())
        self.assertEqual(kind, "rgb")
        self.assertEqual(int(rgb[0, 0, 2]), 200)
        d = np.full((2, 3), 7.5, np.float32)
        kind, arr = imaging.from_ros_image("32FC1", 2, 3, 12, d.tobytes())
        self.assertEqual(kind, "depth")
        self.assertAlmostEqual(float(arr[1, 2]), 7.5)
        self.assertIsNone(imaging.from_ros_image("yuv422", 2, 3, 6, b"\0" * 12))


class TestServer(unittest.TestCase):
    def setUp(self):
        self.m = DashboardModel(source="test")
        self.m.command_handler = lambda cmd: (cmd["cmd"] == "go", "ok" if cmd["cmd"] == "go" else "no")
        self.srv = DashboardServer(self.m, "127.0.0.1", 0).start()
        self.base = f"http://127.0.0.1:{self.srv.port}"

    def tearDown(self):
        self.srv.stop()

    def _post(self, body):
        req = urllib.request.Request(self.base + "/api/command", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=3) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_page_state_commands_and_stream(self):
        with urllib.request.urlopen(self.base + "/", timeout=3) as r:
            self.assertIn(b"ASSI", r.read())
        for f in ("app.js", "style.css"):
            with urllib.request.urlopen(f"{self.base}/{f}", timeout=3) as r:
                self.assertEqual(r.status, 200)
        with urllib.request.urlopen(self.base + "/api/state", timeout=3) as r:
            self.assertEqual(json.loads(r.read())["source"], "test")
        self.assertEqual(self._post({"cmd": "go"}), (200, {"ok": True, "message": "ok"}))
        self.assertEqual(self._post({"cmd": "reset"})[1]["ok"], False)
        self.assertEqual(self._post({"cmd": "rm -rf"})[0], 400)
        with self.assertRaises(urllib.error.HTTPError):
            urllib.request.urlopen(self.base + "/../server.py", timeout=3)
        with urllib.request.urlopen(self.base + "/events", timeout=3) as r:
            line = r.readline()
        self.assertTrue(line.startswith(b"data: {"))
        self.assertEqual(json.loads(line[6:])["source"], "test")


class TestDemo(unittest.TestCase):
    def test_skidpad_runs_and_buttons_work(self):
        from fsd_dashboard.demo import DemoRunner

        r = DemoRunner(CONFIG, "skidpad", auto=False)
        self.assertEqual(r.command({"cmd": "select_mission", "mission": "skidpad"})[0], True)
        self.assertEqual(r.command({"cmd": "activate"})[0], True)
        for _ in range(300):                       # > 5 s in Ready, driver already commanding
            r.step(0.02)
        self.assertTrue(r.command({"cmd": "go"})[0])
        for _ in range(1000):
            r.step(0.02)
        d = r.model.delta(ClientCursor())
        self.assertEqual(d["status"]["state_name"], "Driving")
        self.assertGreaterEqual(d["status"]["lap_count"], 1)
        self.assertIn("lidar", d)
        health = {h["key"]: h for h in d["health"]}
        self.assertEqual(health["lidar"]["level"], "good")
        self.assertTrue(r.command({"cmd": "emergency"})[0])
        r.step(0.1)
        self.assertEqual(r.model.status["state_name"], "Emergency")


# ---------------------------------------------------------------------------
# ROS node against stand-ins
# ---------------------------------------------------------------------------
class _Msg:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _install_ros_stubs():
    """Minimal fake rclpy / message modules so fsd_dashboard.node can be imported and driven."""
    mods = {}

    def mod(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        mods[name] = m
        return m

    class Param:
        def __init__(self, v):
            self.value = v

    class Logger:
        def info(self, *_a, **_k): pass
        warn = error = debug = info

    class Node:
        def __init__(self, name):
            self._params, self.subs, self.clients = {}, {}, {}

        def declare_parameter(self, n, v):
            self._params.setdefault(n, v)

        def get_parameter(self, n):
            return Param(self._params[n])

        def get_logger(self):
            return Logger()

        def create_subscription(self, typ, topic, fn, qos, callback_group=None):
            self.subs[topic] = fn

        def create_client(self, typ, name, callback_group=None):
            c = FakeClient(typ, name)
            self.clients[name] = c
            return c

        def create_publisher(self, typ, topic, qos):
            pub = FakePublisher(topic)
            self.__dict__.setdefault("pubs", {})[topic] = pub
            return pub

        def create_timer(self, *_a, **_k):
            return None

        def destroy_node(self):
            pass

    class FakePublisher:
        def __init__(self, topic):
            self.topic, self.sent, self.subscribers = topic, [], 1

        def publish(self, msg):
            self.sent.append(msg)

        def get_subscription_count(self):
            return self.subscribers

    class FakeFuture:
        def __init__(self, res):
            self._res = res

        def add_done_callback(self, fn):
            fn(self)

        def result(self):
            return self._res

    class FakeClient:
        def __init__(self, typ, name):
            self.srv_name, self.calls = name, []

        def wait_for_service(self, timeout_sec=0):
            return True

        def call_async(self, req):
            self.calls.append(req)
            return FakeFuture(_Msg(success=True, message=f"{self.srv_name} ok"))

    class Time:
        def __init__(self, nanoseconds=0):
            self.nanoseconds = nanoseconds

    class Buffer:
        transforms = {}

        def lookup_transform(self, target, source, t):
            if (target, source) not in self.transforms:
                raise LookupError(source)
            return self.transforms[(target, source)]

    class Req(_Msg):
        pass

    srv_type = lambda: type("Srv", (), {"Request": Req})  # noqa: E731
    mod("rclpy", init=lambda **k: None, ok=lambda: False, shutdown=lambda: None, try_shutdown=lambda: None)
    mod("rclpy.callback_groups", ReentrantCallbackGroup=lambda: None)
    mod("rclpy.clock", Clock=lambda **k: None, ClockType=types.SimpleNamespace(STEADY_TIME=1))
    mod("rclpy.executors", MultiThreadedExecutor=object, ExternalShutdownException=type("ExternalShutdownException", (Exception,), {}))
    mod("rclpy.node", Node=Node)
    mod("rclpy.qos", DurabilityPolicy=types.SimpleNamespace(TRANSIENT_LOCAL=1),
        HistoryPolicy=types.SimpleNamespace(KEEP_LAST=1), ReliabilityPolicy=types.SimpleNamespace(RELIABLE=1, BEST_EFFORT=2),
        QoSProfile=lambda **k: k, qos_profile_sensor_data=None)
    mod("rclpy.time", Time=Time)
    for pkg, names in (("ackermann_msgs.msg", ["AckermannDriveStamped"]), ("nav_msgs.msg", ["Odometry", "Path"]),
                       ("sensor_msgs.msg", ["CameraInfo", "Image", "Imu", "NavSatFix", "PointCloud2"]),
                       ("visualization_msgs.msg", ["MarkerArray"]),
                       ("fsd_sim_msgs.msg", ["ASStatus", "ConeArray", "VehicleSensors"]),
                       ("std_msgs.msg", ["String"])):
        mod(pkg.split(".")[0])
        mod(pkg, **{n: type(n, (_Msg,), {}) for n in names})     # keyword constructor like real messages
    mod("std_srvs"); mod("std_srvs.srv", Trigger=srv_type())  # noqa: E702
    mod("fsd_sim_msgs.srv", SelectMission=srv_type())
    mod("tf2_ros", Buffer=Buffer, TransformListener=lambda *a, **k: None)
    saved = {k: sys.modules.get(k) for k in mods}
    sys.modules.update(mods)
    return saved, Buffer


def _hdr(t, frame="map"):
    return _Msg(stamp=_Msg(sec=int(t), nanosec=int(round((t - int(t)) * 1e9))), frame_id=frame)


def _q(yaw):
    return _Msg(x=0.0, y=0.0, z=math.sin(yaw / 2), w=math.cos(yaw / 2))


class TestRosNode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.saved, cls.Buffer = _install_ros_stubs()
        sys.modules.pop("fsd_dashboard.node", None)
        import fsd_dashboard.node as node_mod  # noqa: PLC0415

        cls.node_mod = node_mod

    @classmethod
    def tearDownClass(cls):
        for k, v in cls.saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        sys.modules.pop("fsd_dashboard.node", None)

    def setUp(self):
        orig = self.node_mod.DashboardNode.declare_parameter

        def declare(node, n, v):
            overrides = {"port": 0, "host": "127.0.0.1", "config_dir": CONFIG}
            return orig(node, n, overrides.get(n, v))

        self.node_mod.DashboardNode.declare_parameter = declare
        self.n = self.node_mod.DashboardNode()
        self.node_mod.DashboardNode.declare_parameter = orig

    def tearDown(self):
        self.n.server.stop()

    def test_topics_to_model(self):
        n, subs = self.n, self.n.subs
        subs["/as/status"](_Msg(header=_hdr(1.0), state=3, mission=2, lap_count=0, laps_target=4, mission_complete=False,
                                cones_hit=0, off_course=0, mission_time=0.5, last_lap_time=0.0, best_lap_time=0.0,
                                penalty_time=0.0, message="AS Driving"))
        subs["/ground_truth/odom"](_Msg(header=_hdr(1.0), pose=_Msg(pose=_Msg(position=_Msg(x=3.0, y=4.0, z=0.0),
                                                                            orientation=_q(0.5))),
                                        twist=_Msg(twist=_Msg(linear=_Msg(x=10.0, y=0.0), angular=_Msg(z=0.2)))))
        subs["/vehicle/sensors"](_Msg(header=_hdr(1.0), wheel_rpm_fl=1.0, wheel_rpm_fr=2.0, wheel_rpm_rl=3.0,
                                      wheel_rpm_rr=4.0, steering_angle=0.1))
        subs["/control/command"](_Msg(header=_hdr(1.0), drive=_Msg(steering_angle=0.12, acceleration=2.0)))
        subs["/ground_truth/cones"](_Msg(cones=[_Msg(id=5, color=1, position=_Msg(x=1.0, y=2.0), knocked=False)]))
        subs["/ground_truth/centerline"](_Msg(poses=[_Msg(pose=_Msg(position=_Msg(x=float(i), y=0.0))) for i in range(5)]))
        subs["/ground_truth/markers"](_Msg(markers=[
            _Msg(ns="timing", points=[_Msg(x=0.0, y=-2.0), _Msg(x=0.0, y=2.0)]),
            _Msg(ns="vehicle", points=[], pose=_Msg(position=_Msg(x=0.1)), scale=_Msg(x=3.0, y=1.4))]))
        # LiDAR: x/y/z + an extra field, transformed by TF (sensor 1 m ahead of map origin, rotated 90 deg)
        dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("intensity", "<f4")])
        pts = np.zeros(2, dt)
        pts["x"], pts["y"], pts["z"] = [5.0, 0.0], [0.0, 2.0], [0.5, 0.0]
        fields = [_Msg(name=k, offset=dt.fields[k][1]) for k in dt.names]
        self.Buffer.transforms[("map", "os_lidar")] = _Msg(header=_hdr(1.0), transform=_Msg(
            translation=_Msg(x=1.0, y=0.0, z=0.0), rotation=_q(math.pi / 2)))
        subs["/lidar/points/preview"](_Msg(header=_hdr(1.0, "os_lidar"), fields=fields, width=2, height=1,
                                           point_step=dt.itemsize, data=pts.tobytes()))
        img = np.zeros((4, 8, 3), np.uint8)
        subs["/zed/zed_node/left/image_rect_color/preview"](_Msg(header=_hdr(1.0), encoding="rgb8", height=4, width=8,
                                                                 step=24, data=img.tobytes()))
        d = n.model.delta(ClientCursor())
        self.assertEqual(d["status"]["mission_name"], "Skidpad")
        self.assertAlmostEqual(d["pose"]["yaw"], 0.5, places=3)
        self.assertEqual(d["vehicle"]["wheel_rpm"], [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(d["command"]["accel"], 2.0)
        self.assertEqual(d["cones"], [[5, 1, 1.0, 2.0, False]])
        self.assertEqual(d["track"]["timing_line"], [[0.0, -2.0], [0.0, 2.0]])
        self.assertAlmostEqual(d["car"]["front"], 1.6)
        q = np.frombuffer(base64.b64decode(d["lidar"]["data"]), "<i2").reshape(-1, 3) * d["lidar"]["res"]
        np.testing.assert_allclose(q, [[1.0, 5.0, 0.5], [-1.0, 0.0, 0.0]], atol=0.011)
        self.assertEqual(d["images"]["left"]["w"], 8)

    def test_vehicle_topic_sets_info_and_prediction(self):
        from fsdsim.config import load_all_configs, to_dict

        veh, _, _ = load_all_configs(CONFIG, vehicle_overrides=["m=300"])
        n = self.n
        n.subs["/as/status"](_Msg(header=_hdr(1.0), state=2, mission=2, lap_count=0, laps_target=4,
                                  mission_complete=False, cones_hit=0, off_course=0, mission_time=0.0,
                                  last_lap_time=0.0, best_lap_time=0.0, penalty_time=0.0, message="AS Ready"))
        n.subs["/sim/vehicle"](_Msg(data=json.dumps({"vehicle": to_dict(veh)}, default=str)))
        for _ in range(100):
            if n.model.prediction:
                break
            time.sleep(0.05)
        d = n.model.delta(ClientCursor())
        self.assertEqual(d["vehicle_info"]["changes"], ["m=300"])
        self.assertEqual(d["prediction"]["event"], "skidpad")
        self.assertGreater(d["prediction"]["lapsim"]["time"], 5.15)   # heavier than the 5.154 s baseline

    @NEEDS_LAPSIM
    def test_vehicle_setup_publishes_vehicle_set(self):
        n = self.n
        self.assertIsNotNone(n.setup, "Vehicle setup should load lapsim/LR26_LapSim")
        n.setup.matlab.exe = None          # no MATLAB in this test
        ok, msg = n.model.setup_handle({"action": "apply", "changes": ["m=270", "skidpad.util_y=0.85"], "matlab": False})
        self.assertTrue(ok, msg)
        sent = n.pubs["/sim/vehicle_set"].sent
        self.assertEqual(json.loads(sent[-1].data), {"changes": ["m=270"]})   # event utilisation is LapSim-only
        n.pubs["/sim/vehicle_set"].subscribers = 0
        n.setup.apply(["m=271"], run_matlab=False)
        self.assertEqual(n.setup.sim["state"], "error")
        self.assertIn("no simulator", n.setup.sim["message"])

    def test_commands_call_services(self):
        ok, msg = self.n.model.handle_command({"cmd": "select_mission", "mission": "autocross"})
        self.assertTrue(ok)
        self.assertEqual(self.n.clients["/as/select_mission"].calls[-1].mission, 6)
        ok, _ = self.n.model.handle_command({"cmd": "emergency"})
        self.assertTrue(ok)
        self.assertEqual(len(self.n.clients["/as/emergency"].calls), 1)
        self.assertEqual(self.n.model.handle_command({"cmd": "select_mission", "mission": "drag"})[0], False)


class TestLaunchIntegration(unittest.TestCase):
    def test_launch_files_start_dashboard(self):
        for name in ("isaac.launch.py", "lite.launch.py"):
            text = open(os.path.join(SRC, "fsd_sim_bringup", "launch", name), encoding="utf-8").read()
            self.assertIn('executable="dashboard"', text, name)
            self.assertIn('DeclareLaunchArgument("dashboard", default_value="true"', text, name)
            self.assertIn("OpaqueFunction(function=_dashboard)", text, name)


if __name__ == "__main__":
    unittest.main()
