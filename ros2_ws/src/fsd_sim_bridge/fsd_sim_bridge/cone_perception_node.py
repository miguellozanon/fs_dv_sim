"""Camera cone perception: YOLO on the ZED image -> cones with colour on the map, for the driver.

    ros2 run fsd_sim_bridge cone_perception --ros-args -p model:=<repo>/models/cones/<name>

Per camera frame: YOLO boxes (class = cone colour), each cone's position from the depth image (the
ground plane without depth), transformed to the map with the camera pose; frames are merged into a cone
map (fsdsim.vision.mapping.ConeMap) that is reset when a new run starts.

Publishes
    /perception/cones        fsd_sim_msgs/ConeArray (map, latched)   the cone map, for the driver (cones_topic)
    /perception/detections   std_msgs/String (JSON)                   boxes of the last frame, for the dashboard
    /perception/stats        std_msgs/String (JSON, 1 Hz)             speed and accuracy vs the ground truth
The ground truth (/ground_truth/cones) is used only for the accuracy numbers, never for the cones.
"""

from __future__ import annotations

import json
import os
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from fsd_sim_msgs.msg import Cone, ConeArray
from geometry_msgs.msg import Point
from rclpy.node import Node
from std_msgs.msg import Header, String

from fsdsim.vision import CLASS_NAMES
from fsdsim.vision.detector import YoloDetector, list_models
from fsdsim.vision.labels import frame_labels
from fsdsim.vision.mapping import Accuracy, ConeMap, locate, map_accuracy

from .camera_input import LATCHED, CameraInput, repo_root
from .common import shutdown_quietly


class ConePerception(Node):
    def __init__(self):
        super().__init__("fsd_cone_perception")
        self.declare_parameter("model", "")            # folder with model.onnx; "" = the newest in models/cones
        self.declare_parameter("conf", 0.35)
        self.declare_parameter("iou", 0.5)
        self.declare_parameter("rate_hz", 10.0)
        self.declare_parameter("max_range", 20.0)
        self.declare_parameter("min_hits", 2)
        self.declare_parameter("namespace", "/zed/zed_node")
        self.declare_parameter("cones_topic", "/perception/cones")   # the camera cone map (fusion: /perception/camera_cones)
        gp = self.get_parameter
        model = gp("model").value or ""
        if not model:
            models = sorted(list_models(os.path.join(repo_root(), "models", "cones")), key=lambda m: m.get("trained") or "")
            if not models:
                raise RuntimeError("no trained cone detector in models/cones (launcher: Train the cone detector)")
            model = models[-1]["path"]
        self.det = YoloDetector(model, conf=float(gp("conf").value), iou=float(gp("iou").value))
        self.model_name = os.path.basename(os.path.normpath(model if not model.endswith(".onnx") else os.path.dirname(model)))
        self.max_range = float(gp("max_range").value)
        self.cam = CameraInput(self, gp("namespace").value)
        self.map = ConeMap(min_hits=int(gp("min_hits").value))
        self.acc = Accuracy()
        self.cones_pub = self.create_publisher(ConeArray, gp("cones_topic").value, LATCHED)
        # this frame's cones in the left camera optical frame (as /zed/detections of the lite backend): the colour
        # source of the LiDAR + camera fusion (fsd_perception)
        self.frame_pub = self.create_publisher(ConeArray, "/perception/frame_cones", 5)
        self.det_pub = self.create_publisher(String, "/perception/detections", 5)
        self.stats_pub = self.create_publisher(String, "/perception/stats", 5)
        self._run_key = None
        self._ms, self._frames, self._t0 = [], 0, time.monotonic()
        self.create_timer(1.0 / max(float(gp("rate_hz").value), 0.5), self._tick)
        self.create_timer(1.0, self._stats)
        self.get_logger().info(f"cone detector '{self.model_name}' ({self.det.backend}), input {self.det.input_hw}")

    def _new_run(self) -> None:
        """Reset the cone map when a run starts again (mission change, Ready after Finished / reset)."""
        st = self.cam.status
        if st is None:
            return
        key = (st.mission, st.state in (st.AS_READY, st.AS_OFF))
        if self._run_key is not None and key != self._run_key and key[1]:
            self.map.reset()
            self.get_logger().info("new run: cone map cleared")
        self._run_key = key

    def _tick(self) -> None:
        self._new_run()
        f = self.cam.frame(need_depth=False)
        if f is None:
            return
        dets = self.det.detect(f["rgb"])
        self._ms.append(self.det.last_ms)
        self._frames += 1
        placed = []
        for d in dets:
            pos = locate(d, f["depth"], f["K"], f["T"], self.max_range)
            d["x"], d["y"] = (pos[0], pos[1]) if pos else (None, None)
            if pos is not None:
                placed.append({"x": pos[0], "y": pos[1], "cls": d["cls"], "conf": d["conf"], "range": pos[2]})
        self._publish_frame(placed, f)
        self.map.update(placed, f["t"])
        if self.cam.truth:
            truth = frame_labels(self.cam.truth, f["T"], f["K"], f["size"], f["depth"], self.max_range)
            self.acc.add_frame(placed, truth)
        self._publish_map(f["t"])
        self.det_pub.publish(String(data=json.dumps({
            "t": round(f["t"], 3), "w": f["size"][0], "h": f["size"][1], "ms": round(self.det.last_ms, 1),
            "dets": [{"cls": d["cls"], "conf": round(d["conf"], 3), "box": [round(v, 1) for v in d["box"]],
                      "x": None if d["x"] is None else round(d["x"], 2), "y": None if d["y"] is None else round(d["y"], 2)}
                     for d in dets]}, separators=(",", ":"))))

    def _publish_frame(self, placed: list, f: dict) -> None:
        import numpy as np

        msg = ConeArray(header=Header(frame_id="zed_left_camera_optical_frame"))
        msg.header.stamp.sec = int(f["t"])
        msg.header.stamp.nanosec = int((f["t"] - int(f["t"])) * 1e9)
        T_cam_map = np.linalg.inv(f["T"])
        for i, c in enumerate(placed):
            p = T_cam_map @ np.array([c["x"], c["y"], 0.16, 1.0])
            msg.cones.append(Cone(id=i + 1, color=int(c["cls"]), position=Point(x=float(p[0]), y=float(p[1]), z=float(p[2])),
                                  confidence=float(c["conf"]), knocked=False))
        self.frame_pub.publish(msg)

    def _publish_map(self, t: float) -> None:
        msg = ConeArray(header=Header(frame_id=self.cam.map_frame))
        msg.header.stamp.sec = int(t)
        msg.header.stamp.nanosec = int((t - int(t)) * 1e9)
        for c in self.map.confirmed():
            msg.cones.append(Cone(id=int(c["id"]), color=int(c["cls"]), position=Point(x=c["x"], y=c["y"], z=0.0),
                                  confidence=float(c["conf"]), knocked=False))
        self.cones_pub.publish(msg)

    def _stats(self) -> None:
        now = time.monotonic()
        fps = self._frames / max(now - self._t0, 1e-6)
        self._frames, self._t0 = 0, now
        ms = sorted(self._ms)[len(self._ms) // 2] if self._ms else None
        self._ms = []
        conf = self.map.confirmed()
        truth = [(x, y, c) for x, y, c, knocked in self.cam.truth if not knocked]
        mapped_truth = []
        if conf and truth:
            # only the part of the track the camera has seen counts (cones near a mapped cone)
            mapped_truth = [g for g in truth if min((g[0] - c["x"]) ** 2 + (g[1] - c["y"]) ** 2 for c in conf) < 9.0]
        self.stats_pub.publish(String(data=json.dumps({
            "model": self.model_name, "backend": self.det.backend, "classes": CLASS_NAMES,
            "fps": round(fps, 1), "ms": None if ms is None else round(ms, 1),
            "frame": self.acc.summary(), "map": map_accuracy(conf, mapped_truth) if mapped_truth else {"mapped": len(conf)},
        }, separators=(",", ":"))))


def main(args=None) -> None:
    rclpy.init(args=args)
    try:
        node = ConePerception()
    except Exception as e:  # noqa: BLE001
        rclpy.logging.get_logger("fsd_cone_perception").error(str(e))
        shutdown_quietly()  # the SIGINT handler may already have shut the context down
        raise SystemExit(1) from None
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass  # Ctrl+C / launch shutdown
    finally:
        node.destroy_node()
        shutdown_quietly()  # the SIGINT handler may already have shut the context down


if __name__ == "__main__":
    main()
