"""Record a YOLO training set while the simulation drives: ZED images + cone labels from the ground truth.

    ros2 run fsd_sim_bridge cone_dataset_recorder --ros-args -p rate_hz:=2.0

Writes datasets/cones/<session>/images/*.jpg and labels/*.txt (YOLO format: class cx cy w h), one image
per `rate_hz` and only after the car moved `min_move` metres (no identical frames while it stands).
Labels come from the ground-truth cones projected into the camera; the depth image removes cones that are
hidden. tools/train_cone_yolo.py trains on every recorded session.

The node also checks that the camera model matches the image: the share of labelled cones that the depth
image confirms. A low share means the camera pose or intrinsics differ from sensors.yaml.
"""

from __future__ import annotations

import json
import math
import os
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from fsdsim.vision import CLASS_NAMES
from fsdsim.vision.labels import frame_labels, yolo_lines

from .camera_input import CameraInput, repo_root
from .common import shutdown_quietly

# 2: camera images stamped with the time of the frame they show (Isaac isaac_frame_lag) and the main hoop no
# longer in front of the camera. Sessions without session.json (version 1) have shifted labels: not used.
LABELS_VERSION = 2


class ConeDatasetRecorder(Node):
    def __init__(self):
        super().__init__("fsd_cone_dataset_recorder")
        self.declare_parameter("out_dir", "")
        self.declare_parameter("session", "")
        self.declare_parameter("rate_hz", 2.0)
        self.declare_parameter("min_move", 1.0)
        self.declare_parameter("max_range", 25.0)
        self.declare_parameter("jpeg_quality", 92)
        self.declare_parameter("namespace", "/zed/zed_node")
        gp = self.get_parameter
        out = gp("out_dir").value or os.path.join(repo_root(), "datasets", "cones")
        session = gp("session").value or time.strftime("sim_%Y%m%d-%H%M%S")
        self.dir = os.path.join(out, session)
        self.img_dir, self.lab_dir = os.path.join(self.dir, "images"), os.path.join(self.dir, "labels")
        os.makedirs(self.img_dir, exist_ok=True)
        os.makedirs(self.lab_dir, exist_ok=True)
        with open(os.path.join(self.dir, "classes.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(CLASS_NAMES) + "\n")
        with open(os.path.join(self.dir, "session.json"), "w", encoding="utf-8") as f:
            json.dump({"labels_version": LABELS_VERSION, "created": time.strftime("%Y-%m-%d %H:%M:%S")}, f)
        self.cam = CameraInput(self, gp("namespace").value)
        self.max_range = float(gp("max_range").value)
        self.min_move = float(gp("min_move").value)
        self.quality = int(gp("jpeg_quality").value)
        self.n = 0
        self.boxes = 0
        self.last_xy = None
        self.check = [0, 0]                   # cones in view, of which the depth image confirms
        self.meta = open(os.path.join(self.dir, "frames.jsonl"), "a", encoding="utf-8")
        self.create_timer(1.0 / max(float(gp("rate_hz").value), 0.05), self._tick)
        self.create_timer(30.0, self._report)
        self.get_logger().info(f"recording YOLO training images to {self.dir}")

    def _tick(self) -> None:
        f = self.cam.frame(need_depth=True)
        if f is None or not self.cam.truth:
            return
        st = self.cam.status
        if st is not None and st.state not in (st.AS_READY, st.AS_DRIVING):
            return                                     # finished / emergency: the car stands, nothing new
        cam_xy = (f["T"][0, 3], f["T"][1, 3])
        if self.last_xy is not None and math.hypot(cam_xy[0] - self.last_xy[0], cam_xy[1] - self.last_xy[1]) < self.min_move:
            return
        self.last_xy = cam_xy
        labels = frame_labels(self.cam.truth, f["T"], f["K"], f["size"], f["depth"], self.max_range)
        in_view = frame_labels(self.cam.truth, f["T"], f["K"], f["size"], None, self.max_range)
        self.check[0] += len(in_view)
        self.check[1] += len(labels)
        import cv2  # noqa: PLC0415

        name = f"{os.path.basename(self.dir)}_{self.n:06d}"
        cv2.imwrite(os.path.join(self.img_dir, name + ".jpg"), f["rgb"][..., ::-1],
                    [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
        with open(os.path.join(self.lab_dir, name + ".txt"), "w", encoding="utf-8") as fh:
            rows = yolo_lines(labels, f["size"])
            fh.write("\n".join(rows) + ("\n" if rows else ""))
        self.meta.write(json.dumps({"name": name, "t": round(f["t"], 3), "x": round(cam_xy[0], 3), "y": round(cam_xy[1], 3),
                                    "boxes": len(labels), "in_view": len(in_view)}) + "\n")
        self.meta.flush()
        self.n += 1
        self.boxes += len(labels)

    def _report(self) -> None:
        seen, ok = self.check
        share = ok / seen if seen else None
        self.get_logger().info(f"dataset: {self.n} images, {self.boxes} cone boxes in {self.dir}"
                               + (f"; depth confirms {100 * share:.0f} % of the cones in view" if share is not None else ""))
        if share is not None and seen > 50 and share < 0.4:
            self.get_logger().warn("the depth image confirms few projected cones: the camera pose or intrinsics may "
                                   "differ from sensors.yaml (labels would be misplaced) - check docs/PERCEPTION.md")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ConeDatasetRecorder()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass  # Ctrl+C / launch shutdown
    finally:
        node._report()
        node.meta.close()
        node.destroy_node()
        shutdown_quietly()  # the SIGINT handler may already have shut the context down


if __name__ == "__main__":
    main()
