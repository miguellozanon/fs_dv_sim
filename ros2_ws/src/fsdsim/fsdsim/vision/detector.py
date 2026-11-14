"""YOLO cone detector on a camera image (an Ultralytics YOLO model exported to ONNX).

Runs with ONNX Runtime (CPU, or CUDA if onnxruntime-gpu is installed) and falls back to OpenCV's DNN module.
The model folder (models/cones/<name>/) holds model.onnx and model.json:
    {"names": ["blue_cone", ...], "imgsz": [384, 640], "trained": "...", "metrics": {...}}
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

from . import CLASS_NAMES


def letterbox(img: np.ndarray, shape: tuple[int, int], color=114):
    """Resize keeping the aspect ratio and pad to shape (h, w). Returns image, scale, (pad_x, pad_y)."""
    import cv2  # noqa: PLC0415

    h0, w0 = img.shape[:2]
    H, W = shape
    r = min(H / h0, W / w0)
    nw, nh = int(round(w0 * r)), int(round(h0 * r))
    if (nw, nh) != (w0, h0):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    px, py = (W - nw) / 2.0, (H - nh) / 2.0
    top, left = int(round(py - 0.1)), int(round(px - 0.1))
    out = np.full((H, W, 3), color, dtype=np.uint8)
    out[top:top + nh, left:left + nw] = img
    return out, r, (left, top)


def nms(boxes: np.ndarray, scores: np.ndarray, iou: float) -> list[int]:
    """Greedy non-maximum suppression; boxes (N, 4) xyxy."""
    order = np.argsort(-scores)
    keep = []
    area = (boxes[:, 2] - boxes[:, 0]).clip(0) * (boxes[:, 3] - boxes[:, 1]).clip(0)
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = (xx2 - xx1).clip(0) * (yy2 - yy1).clip(0)
        ov = inter / (area[i] + area[rest] - inter + 1e-9)
        order = rest[ov <= iou]
    return keep


def postprocess(out: np.ndarray, scale: float, pad: tuple[int, int], size: tuple[int, int], n_cls: int,
                conf: float = 0.35, iou: float = 0.5, max_det: int = 100) -> list[dict]:
    """Ultralytics detection output (1, 4 + n_cls, N) (cx, cy, w, h in model pixels, class scores) ->
    [{cls, conf, box (x1, y1, x2, y2) in image pixels}]. Class-agnostic NMS: a cone has one colour."""
    p = np.asarray(out)
    if p.ndim == 3:
        p = p[0]
    if p.shape[0] != 4 + n_cls and p.shape[1] == 4 + n_cls:
        p = p.T
    scores_all = p[4:4 + n_cls].T
    cls = scores_all.argmax(1)
    sc = scores_all[np.arange(len(cls)), cls]
    m = sc >= conf
    if not m.any():
        return []
    cx, cy, w, h = p[0, m], p[1, m], p[2, m], p[3, m]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad[0]) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad[1]) / scale
    W, H = size
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, W - 1)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, H - 1)
    sc, cls = sc[m], cls[m]
    keep = nms(boxes, sc, iou)[:max_det]
    return [{"cls": int(cls[i]), "conf": float(sc[i]), "box": tuple(float(v) for v in boxes[i])} for i in keep]


class YoloDetector:
    def __init__(self, model_dir: str, conf: float = 0.35, iou: float = 0.5, prefer: str = "auto"):
        """model_dir: folder with model.onnx (+ model.json), or the .onnx file itself."""
        path = model_dir if model_dir.endswith(".onnx") else os.path.join(model_dir, "model.onnx")
        if not os.path.exists(path):
            raise FileNotFoundError(f"no YOLO model at {path} (train one: tools/train_cone_yolo.py)")
        meta_p = os.path.join(os.path.dirname(path), "model.json")
        self.meta = {}
        if os.path.exists(meta_p):
            with open(meta_p, encoding="utf-8") as f:
                self.meta = json.load(f)
        self.names = list(self.meta.get("names") or CLASS_NAMES)
        self.conf, self.iou = conf, iou
        self.path = path
        self.backend = ""
        self._sess = self._net = None
        if prefer in ("auto", "onnxruntime"):
            try:
                import onnxruntime as ort  # noqa: PLC0415

                prov = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
                so = ort.SessionOptions()
                so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                # a few threads only: by default ONNX Runtime takes every core, and the Isaac relay in the same WSL
                # then fell seconds behind (the car drove on old data, 11 Nov 2026). FSG_ORT_THREADS overrides.
                so.intra_op_num_threads = max(1, int(os.environ.get("FSG_ORT_THREADS", "2")))
                so.inter_op_num_threads = 1
                self._sess = ort.InferenceSession(path, sess_options=so, providers=prov)
                self._in = self._sess.get_inputs()[0]
                self.backend = "onnxruntime " + ("CUDA" if self._sess.get_providers()[0].startswith("CUDA") else "CPU")
                shape = self._in.shape
            except ImportError:
                if prefer == "onnxruntime":
                    raise
        if self._sess is None:
            import cv2  # noqa: PLC0415

            self._net = cv2.dnn.readNetFromONNX(path)
            self.backend = f"OpenCV DNN {cv2.__version__}"
            shape = [1, 3] + list(self.meta.get("imgsz") or [640, 640])
        hw = self.meta.get("imgsz")
        if not hw:
            hw = [int(shape[2]) if isinstance(shape[2], int) else 640, int(shape[3]) if isinstance(shape[3], int) else 640]
        self.input_hw = (int(hw[0]), int(hw[1])) if isinstance(hw, (list, tuple)) else (int(hw), int(hw))
        self.last_ms = 0.0

    def detect(self, rgb: np.ndarray) -> list[dict]:
        """RGB uint8 image (H, W, 3) -> detections [{cls, conf, box}] in image pixels."""
        t0 = time.perf_counter()
        img, r, pad = letterbox(rgb, self.input_hw)
        x = np.ascontiguousarray(img.transpose(2, 0, 1)[None].astype(np.float32) / 255.0)
        if self._sess is not None:
            out = self._sess.run(None, {self._in.name: x})[0]
        else:
            self._net.setInput(x)
            out = self._net.forward()
        dets = postprocess(out, r, pad, (rgb.shape[1], rgb.shape[0]), len(self.names), self.conf, self.iou)
        self.last_ms = (time.perf_counter() - t0) * 1000.0
        return dets


def list_models(root: str) -> list[dict]:
    """Trained cone detectors under root (models/cones): [{name, path, trained, metrics}]."""
    out = []
    if os.path.isdir(root):
        for name in sorted(os.listdir(root)):
            d = os.path.join(root, name)
            if os.path.isfile(os.path.join(d, "model.onnx")):
                meta = {}
                try:
                    with open(os.path.join(d, "model.json"), encoding="utf-8") as f:
                        meta = json.load(f)
                except (OSError, ValueError):
                    pass
                out.append({"name": name, "path": d, "trained": meta.get("trained"), "metrics": meta.get("metrics", {}),
                            "images": meta.get("images")})
    return out
