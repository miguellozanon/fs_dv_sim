"""Camera cone perception (fsdsim.vision): labels, detector post-processing, 3D cones, cone map, accuracy.

A tiny renderer draws cones into an RGB and a depth image with the same camera model as the simulator,
so every step can be checked without Isaac Sim or a trained model."""

import math
import os
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ros2_ws", "src", "fsdsim"))

from fsdsim.config import load_all_configs  # noqa: E402
from fsdsim.vision import CLASS_NAMES  # noqa: E402
from fsdsim.vision import detector as D  # noqa: E402
from fsdsim.vision import labels as Lb  # noqa: E402
from fsdsim.vision import mapping as M  # noqa: E402
from fsdsim.vision.geometry import camera_pose, ground_point, project  # noqa: E402

CONFIG = os.path.join(ROOT, "ros2_ws", "src", "fsd_sim_bringup", "config")
COLORS = {0: (30, 60, 220), 1: (240, 220, 30), 2: (250, 120, 20), 3: (250, 120, 20)}


def camera():
    sens = load_all_configs(CONFIG)[1]
    cc = sens.camera
    w, h = cc.resolution
    fx = (w / 2.0) / math.tan(math.radians(cc.hfov_deg) / 2.0)
    return cc, (fx, fx, w / 2.0, h / 2.0), (w, h)


def render(cones, T, K, size):
    """RGB + depth (optical z) of flat-shaded cones on a grey ground plane."""
    import cv2

    w, h = size
    rgb = np.full((h, w, 3), 90, np.uint8)
    depth = np.full((h, w), np.inf, np.float32)
    vv, uu = np.mgrid[0:h, 0:w]
    fx, fy, cx, cy = K
    d = np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones_like(uu, float)], -1) @ T[:3, :3].T
    s = -T[2, 3] / np.where(d[..., 2] < -1e-6, d[..., 2], np.nan)
    zg = s                                     # optical z of the ground point = s (ray z component is 1)
    depth = np.where(np.isfinite(zg) & (zg > 0), zg, np.inf).astype(np.float32)
    for x, y, c in sorted(cones, key=lambda c: -math.hypot(c[0] - T[0, 3], c[1] - T[1, 3])):   # far to near
        uv, z = project(Lb.cone_points(x, y, c), T, K)
        if np.any(z < 0.3):
            continue
        hull = cv2.convexHull(np.round(uv).astype(np.int32))
        mask = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(mask, hull, 1)
        sel = mask.astype(bool)
        rgb[sel] = COLORS[c]
        depth[sel] = float(z.mean())
    return rgb, depth


class TestVision(unittest.TestCase):
    def setUp(self):
        self.cc, self.K, self.size = camera()
        self.T = camera_pose(0.0, 0.0, 0.0, self.cc.mount, self.cc.baseline / 2.0)

    def test_projection_matches_the_simulator_frames(self):
        # a ground point straight ahead lands in the centre column; the camera (1.15 m high) looks 0.12 rad down,
        # so ground 5 m ahead is below the image centre and 20 m ahead above it
        y0 = self.cc.mount.y + self.cc.baseline / 2.0
        uv, z = project(np.array([[5.0, y0, 0.0], [20.0, y0, 0.0]]), self.T, self.K)
        self.assertAlmostEqual(uv[0, 0], self.size[0] / 2.0, delta=1.0)
        self.assertGreater(uv[0, 1], self.size[1] / 2.0)
        self.assertLess(uv[1, 1], self.size[1] / 2.0)
        self.assertGreater(z[0], 4.5)
        g = ground_point(uv[0, 0], uv[0, 1], self.T, self.K)
        np.testing.assert_allclose(g[:2], [5.0, y0], atol=1e-6)

    def test_labels_skip_hidden_and_far_cones(self):
        cones = [(6.0, 1.5, 0), (6.0, -1.5, 1), (12.0, 1.5, 0), (12.0, -1.5, 1),
                 (4.0, 0.0, 3), (4.6, 0.0, 2),           # the small orange cone hides behind the big one
                 (-5.0, 0.0, 0), (60.0, 0.0, 0)]          # behind the car, too far
        rgb, depth = render(cones, self.T, self.K, self.size)
        lab = Lb.frame_labels(cones, self.T, self.K, self.size, depth)
        got = sorted((d["cone"], d["cls"]) for d in lab)
        self.assertIn(((4.0, 0.0), 3), got)
        self.assertNotIn(((4.6, 0.0), 2), got)
        self.assertNotIn(((-5.0, 0.0), 0), got)
        self.assertNotIn(((60.0, 0.0), 0), got)
        self.assertEqual(len(got), 5)
        rows = Lb.yolo_lines(lab, self.size)
        self.assertEqual(len(rows), 5)
        c, cx, cy, w, h = rows[0].split()
        self.assertTrue(0 < float(cx) < 1 and 0 < float(h) < 1)
        # without a depth image every cone in view is labelled, the hidden one too
        self.assertEqual(len(Lb.frame_labels(cones, self.T, self.K, self.size)), 6)

    def test_locate_from_depth_and_ground(self):
        cones = [(7.0, 1.6, 0), (11.0, -1.7, 1), (16.0, 2.0, 0)]
        _, depth = render(cones, self.T, self.K, self.size)
        for d in Lb.frame_labels(cones, self.T, self.K, self.size, depth):
            det = {"box": d["box"], "cls": d["cls"], "conf": 0.9}
            x, y, r = M.locate(det, depth, self.K, self.T)
            self.assertLess(math.hypot(x - d["cone"][0], y - d["cone"][1]), 0.25, (d["cone"], x, y))
            x, y, r = M.locate(det, None, self.K, self.T)              # ground plane only
            self.assertLess(math.hypot(x - d["cone"][0], y - d["cone"][1]), 0.6, (d["cone"], x, y))

    def test_postprocess_and_nms(self):
        # model output for a 640 x 384 input: two overlapping boxes on one cone + one other cone
        nc = len(CLASS_NAMES)
        img_size = (1280, 720)
        r = min(384 / 720, 640 / 1280)
        pad = (0, int(round((384 - 720 * r) / 2 - 0.1)))
        out = np.zeros((1, 4 + nc, 3), np.float32)
        out[0, :4, 0] = [100, 200, 20, 30]
        out[0, 4 + 0, 0] = 0.9
        out[0, :4, 1] = [101, 201, 20, 30]
        out[0, 4 + 1, 1] = 0.6
        out[0, :4, 2] = [400, 220, 30, 40]
        out[0, 4 + 3, 2] = 0.8
        dets = D.postprocess(out, r, pad, img_size, nc, conf=0.3, iou=0.5)
        self.assertEqual([d["cls"] for d in dets], [0, 3])
        x1, y1, x2, y2 = dets[0]["box"]
        self.assertAlmostEqual((x1 + x2) / 2, 100 / r, delta=0.01)
        self.assertAlmostEqual((y1 + y2) / 2, (200 - pad[1]) / r, delta=0.01)
        self.assertEqual(D.postprocess(out, r, pad, img_size, nc, conf=0.95), [])

    def test_letterbox(self):
        img = np.zeros((720, 1280, 3), np.uint8)
        out, r, pad = D.letterbox(img, (384, 640))
        self.assertEqual(out.shape, (384, 640, 3))
        self.assertAlmostEqual(r, 0.5)
        self.assertEqual(pad, (0, 12))

    def test_cone_map_and_accuracy(self):
        truth = [(5.0 + 3 * i, 1.6 * (1 if i % 2 else -1), i % 2) for i in range(8)]
        cm = M.ConeMap()
        rng = np.random.default_rng(0)
        for k in range(6):
            dets = [{"x": x + rng.normal(0, 0.08), "y": y + rng.normal(0, 0.08), "cls": c if k else 1 - c,
                     "conf": 0.8, "range": 8.0} for x, y, c in truth]
            dets.append({"x": 30 + k * 7.0, "y": 9.0, "cls": 0, "conf": 0.4, "range": 20.0})    # spurious, never repeated
            cm.update(dets, t=k * 0.1)
        conf = cm.confirmed()
        self.assertEqual(len(conf), len(truth))                  # one cone each, the noise is not confirmed
        acc = M.map_accuracy(conf, truth)
        self.assertEqual((acc["matched"], acc["colour_ok"], acc["false"]), (8, 8, 0))   # colour vote fixes frame 0
        self.assertLess(acc["error_m"], 0.1)
        a = M.Accuracy()
        a.add_frame([{"x": 5.1, "y": -1.6, "cls": 0}, {"x": 20.0, "y": 0.0, "cls": 1}],
                    [{"cone": (5.0, -1.6), "cls": 0}, {"cone": (8.0, 1.6), "cls": 1}])
        s = a.summary()
        self.assertEqual((s["recall"], s["precision"], s["colour"]), (0.5, 0.5, 1.0))

    def test_detector_end_to_end_with_a_stand_in_model(self):
        """YoloDetector: image -> letterbox -> (ONNX Runtime session, replaced here) -> boxes in image pixels."""
        import json
        from unittest import mock

        try:
            import onnxruntime as ort
        except ImportError:
            self.skipTest("onnxruntime not installed")
        cones = [(7.0, 1.6, 0), (9.0, -1.7, 1)]
        rgb, depth = render(cones, self.T, self.K, self.size)
        truth = Lb.frame_labels(cones, self.T, self.K, self.size, depth)
        seen = {}

        class FakeSession:
            def __init__(self, path, sess_options=None, providers=None):
                pass

            def get_inputs(self):
                return [type("I", (), {"name": "images", "shape": [1, 3, 384, 640]})()]

            def get_providers(self):
                return ["CPUExecutionProvider"]

            def run(self, _, feeds):
                x = feeds["images"]
                seen["shape"] = x.shape
                r, top = 0.5, 12                               # 1280 x 720 -> 640 x 360 + 12 px padding
                out = np.zeros((1, 8, len(truth)), np.float32)
                for i, g in enumerate(truth):
                    x1, y1, x2, y2 = g["box"]
                    out[0, :4, i] = [(x1 + x2) / 2 * r, (y1 + y2) / 2 * r + top, (x2 - x1) * r, (y2 - y1) * r]
                    out[0, 4 + g["cls"], i] = 0.9
                return [out]

        root = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(root, ignore_errors=True))
        d = os.path.join(root, "sim_cones")
        os.makedirs(d)
        open(os.path.join(d, "model.onnx"), "wb").close()
        with open(os.path.join(d, "model.json"), "w") as f:
            json.dump({"names": CLASS_NAMES, "imgsz": [384, 640]}, f)
        with mock.patch.object(ort, "InferenceSession", FakeSession):
            det = D.YoloDetector(d)
            dets = det.detect(rgb)
        self.assertEqual(seen["shape"], (1, 3, 384, 640))
        self.assertEqual(det.backend, "onnxruntime CPU")
        self.assertEqual(sorted(x["cls"] for x in dets), sorted(g["cls"] for g in truth))
        for x in dets:
            pos = M.locate(x, depth, self.K, self.T)
            self.assertLess(min(math.hypot(pos[0] - c[0], pos[1] - c[1]) for c in cones), 0.25)
        self.assertEqual([m["name"] for m in D.list_models(root)], ["sim_cones"])


if __name__ == "__main__":
    unittest.main()
