"""The camera perception ROS 2 nodes (cone_dataset_recorder, cone_perception) without ROS 2: a small stand-in
for rclpy, the messages and tf2 feeds them rendered camera frames (tests/test_vision.render)."""

import json
import math
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ros2_ws", "src", "fsdsim"))
sys.path.insert(0, os.path.join(ROOT, "tests"))


# ---------------------------------------------------------------------------- ROS 2 stand-in
class _Msg:
    def __init__(self, **kw):
        for k, v in self._defaults().items():
            setattr(self, k, v)
        for k, v in kw.items():
            setattr(self, k, v)

    def _defaults(self):
        return {}


class Stamp(_Msg):
    def _defaults(self):
        return {"sec": 0, "nanosec": 0}


class Header(_Msg):
    def _defaults(self):
        return {"stamp": Stamp(), "frame_id": ""}


def _with_header(name, **fields):
    def defaults(self):
        d = {"header": Header()}
        d.update({k: (v() if callable(v) else v) for k, v in fields.items()})
        return d
    return type(name, (_Msg,), {"_defaults": defaults})


Image = _with_header("Image", height=0, width=0, encoding="", step=0, data=b"")
CameraInfo = _with_header("CameraInfo", width=0, height=0, k=list)
ConeArray = _with_header("ConeArray", cones=list)
Cone = type("Cone", (_Msg,), {})
Point = type("Point", (_Msg,), {})
String = type("String", (_Msg,), {})


class ASStatus(_Msg):
    AS_OFF, AS_READY, AS_DRIVING, AS_EMERGENCY, AS_FINISHED = 1, 2, 3, 4, 5

    def _defaults(self):
        return {"header": Header(), "state": 3, "mission": 6}


class _Param:
    def __init__(self, v):
        self.value = v


class _Log:
    def __init__(self):
        self.lines = []

    def info(self, m):
        self.lines.append(("info", m))

    def warn(self, m):
        self.lines.append(("warn", m))

    error = warn


class Pub:
    def __init__(self):
        self.msgs = []

    def publish(self, m):
        self.msgs.append(m)


class Node:
    def __init__(self, name):
        self.params, self.subs, self.pubs, self.timers, self.log = {}, {}, {}, [], _Log()

    def declare_parameter(self, name, default):
        self.params.setdefault(name, default)

    def get_parameter(self, name):
        return _Param(OVERRIDES.get(name, self.params[name]))

    def create_subscription(self, typ, topic, cb, qos):
        self.subs[topic] = cb

    def create_publisher(self, typ, topic, qos):
        return self.pubs.setdefault(topic, Pub())

    def create_timer(self, period, cb):
        self.timers.append(cb)

    def get_logger(self):
        return self.log

    def destroy_node(self):
        pass


class Buffer:
    T = None

    def lookup_transform(self, parent, child, stamp):
        T = Buffer.T
        q = _quat(T[:3, :3])
        tr = types.SimpleNamespace(x=T[0, 3], y=T[1, 3], z=T[2, 3])
        rot = types.SimpleNamespace(x=q[0], y=q[1], z=q[2], w=q[3])
        return types.SimpleNamespace(transform=types.SimpleNamespace(translation=tr, rotation=rot))


def _quat(R):
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = math.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2
    y = math.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2
    z = math.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2
    x = math.copysign(x, R[2, 1] - R[1, 2])
    y = math.copysign(y, R[0, 2] - R[2, 0])
    z = math.copysign(z, R[1, 0] - R[0, 1])
    return x, y, z, w


OVERRIDES = {}


def install_stubs():
    def mod(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        sys.modules[name] = m
        return m

    class _Enum:
        RELIABLE = TRANSIENT_LOCAL = KEEP_LAST = None

    rclpy = mod("rclpy", init=lambda args=None: None, ok=lambda: False, shutdown=lambda: None, spin=lambda n: None)
    rclpy.time = mod("rclpy.time", Time=lambda *a, **k: None)
    rclpy.executors = mod("rclpy.executors", ExternalShutdownException=type("ExternalShutdownException", (Exception,), {}),
                          MultiThreadedExecutor=object)
    rclpy.node = mod("rclpy.node", Node=Node)
    rclpy.qos = mod("rclpy.qos", QoSProfile=lambda **k: k, ReliabilityPolicy=_Enum, DurabilityPolicy=_Enum,
                    HistoryPolicy=_Enum, qos_profile_sensor_data=None)
    rclpy.logging = mod("rclpy.logging", get_logger=lambda n: _Log())
    mod("sensor_msgs"), mod("sensor_msgs.msg", Image=Image, CameraInfo=CameraInfo)
    mod("fsd_sim_msgs"), mod("fsd_sim_msgs.msg", ConeArray=ConeArray, Cone=Cone, ASStatus=ASStatus)
    mod("geometry_msgs"), mod("geometry_msgs.msg", Point=Point)
    mod("std_msgs"), mod("std_msgs.msg", Header=Header, String=String)
    mod("tf2_ros", Buffer=Buffer, TransformListener=lambda buf, node: None)
    sys.path.insert(0, os.path.join(ROOT, "ros2_ws", "src", "fsd_sim_bridge"))


# ---------------------------------------------------------------------------- tests
class TestPerceptionNodes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._saved = dict(sys.modules)
        install_stubs()
        import test_vision as tv

        cls.tv = tv
        cls.cc, cls.K, cls.size = tv.camera()
        # a slalom-free straight: blue left, yellow right, every 3 m
        cls.truth = [(3.0 * i, 1.7, 0) for i in range(2, 12)] + [(3.0 * i, -1.7, 1) for i in range(2, 12)]

    @classmethod
    def tearDownClass(cls):
        for k in list(sys.modules):
            if k not in cls._saved:
                del sys.modules[k]

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp, ignore_errors=True))
        OVERRIDES.clear()

    def feed(self, node, x, t):
        """One camera frame of the car at (x, 0) heading +x, stamped t."""
        from fsdsim.vision.geometry import camera_pose

        T = camera_pose(x, 0.0, 0.0, self.cc.mount, self.cc.baseline / 2.0)
        Buffer.T = T
        rgb, depth = self.tv.render(self.truth, T, self.K, self.size)
        st = Stamp(sec=int(t), nanosec=int((t - int(t)) * 1e9))
        w, h = self.size
        subs = node.subs
        subs["/zed/zed_node/left/camera_info"](CameraInfo(header=Header(stamp=st, frame_id="zed_left_camera_optical_frame"),
                                                          width=w, height=h, k=[self.K[0], 0, self.K[2], 0, self.K[1], self.K[3], 0, 0, 1]))
        subs["/zed/zed_node/depth/depth_registered"](Image(header=Header(stamp=st), height=h, width=w, encoding="32FC1",
                                                           step=w * 4, data=depth.astype(np.float32).tobytes()))
        subs["/zed/zed_node/left/image_rect_color"](Image(header=Header(stamp=st, frame_id="zed_left_camera_optical_frame"),
                                                          height=h, width=w, encoding="rgb8", step=w * 3, data=rgb.tobytes()))
        return T, depth

    def truth_msg(self):
        return ConeArray(header=Header(frame_id="map"), cones=[
            Cone(id=i, color=c, position=Point(x=x, y=y, z=0.0), knocked=False) for i, (x, y, c) in enumerate(self.truth)])

    def test_recorder_writes_images_and_labels(self):
        from fsd_sim_bridge import cone_dataset_recorder_node as R

        OVERRIDES.update(out_dir=self.tmp, session="unit", min_move=1.0)
        n = R.ConeDatasetRecorder()
        n.subs["/ground_truth/cones"](self.truth_msg())
        n.subs["/as/status"](ASStatus(state=ASStatus.AS_DRIVING))
        for k, x in enumerate((0.0, 0.5, 2.0, 4.0)):           # 0.5 m: not far enough for a new image
            self.feed(n, x, 1.0 + k * 0.5)
            n._tick()
        n._report()
        n.meta.close()
        d = os.path.join(self.tmp, "unit")
        imgs = sorted(os.listdir(os.path.join(d, "images")))
        self.assertEqual(len(imgs), 3)
        rows = open(os.path.join(d, "labels", imgs[0].replace(".jpg", ".txt"))).read().split("\n")
        rows = [r for r in rows if r]
        self.assertGreaterEqual(len(rows), 6)
        self.assertEqual({r.split()[0] for r in rows}, {"0", "1"})
        meta = [json.loads(line) for line in open(os.path.join(d, "frames.jsonl"))]
        self.assertEqual(len(meta), 3)
        self.assertTrue(any("depth confirms 100 %" in m for _, m in n.log.lines), n.log.lines)
        import cv2

        self.assertEqual(cv2.imread(os.path.join(d, "images", imgs[0])).shape, (self.size[1], self.size[0], 3))

    def test_perception_maps_the_cones_the_detector_sees(self):
        from fsd_sim_bridge import cone_perception_node as P
        from fsdsim.vision.labels import frame_labels

        test = self

        class TruthDetector:
            """Stand-in for the YOLO model: the boxes of the visible cones (labels), slightly jittered."""
            backend, input_hw, last_ms = "test", (384, 640), 5.0

            def __init__(self, *a, **k):
                self.rng = np.random.default_rng(1)

            def detect(self, rgb):
                lab = frame_labels(test.truth, Buffer.T, test.K, test.size, test.depth)
                return [{"cls": g["cls"], "conf": 0.8, "box": tuple(v + self.rng.normal(0, 0.5) for v in g["box"])} for g in lab]

        model = os.path.join(self.tmp, "models", "cones", "unit")
        os.makedirs(model)
        OVERRIDES.update(model=model)
        with mock.patch.object(P, "YoloDetector", TruthDetector):
            n = P.ConePerception()
        n.subs["/ground_truth/cones"](self.truth_msg())
        n.subs["/as/status"](ASStatus(state=ASStatus.AS_DRIVING))
        for k in range(8):
            _, self.depth = self.feed(n, 0.5 * k, 1.0 + 0.1 * k)
            n._tick()
        n._stats()
        cones = n.pubs["/perception/cones"].msgs[-1].cones
        self.assertGreaterEqual(len(cones), 10)
        for c in cones:
            err = min(math.hypot(c.position.x - x, c.position.y - y) for x, y, _ in self.truth)
            self.assertLess(err, 0.3)
            gx, gy, gc = min(self.truth, key=lambda g: math.hypot(c.position.x - g[0], c.position.y - g[1]))
            self.assertEqual(c.color, gc)
        det = json.loads(n.pubs["/perception/detections"].msgs[-1].data)
        self.assertEqual((det["w"], det["h"]), self.size)
        self.assertTrue(det["dets"] and det["dets"][0]["x"] is not None)
        st = json.loads(n.pubs["/perception/stats"].msgs[-1].data)
        self.assertGreater(st["frame"]["recall"], 0.9)
        self.assertGreater(st["frame"]["precision"], 0.9)
        self.assertEqual(st["frame"]["colour"], 1.0)
        self.assertEqual(st["map"]["false"], 0)
        # a new run (Ready again) clears the map
        n.subs["/as/status"](ASStatus(state=ASStatus.AS_READY))
        n._new_run()
        self.assertEqual(n.map.confirmed(), [])


if __name__ == "__main__":
    unittest.main()
