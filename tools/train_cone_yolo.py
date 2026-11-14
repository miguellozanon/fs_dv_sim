#!/usr/bin/env python3
"""Train the YOLO cone detector on the images recorded in the simulator, and export it for the car.

    python tools/train_cone_yolo.py --name sim_cones --epochs 60          (Windows Python with CUDA PyTorch)

Input: every session in datasets/cones/<session>/{images,labels} (record them with the launcher option
"Record camera images for YOLO training", node cone_dataset_recorder). Classes: blue_cone, yellow_cone,
orange_cone, large_orange_cone (= fsd_sim_msgs/Cone colours, FSOCO names).
Validation images are whole sessions (or whole stretches of driving), never neighbours of training images.

Output: models/cones/<name>/model.onnx + model.json (classes, input size, accuracy); the Ultralytics run
(weights, plots, metrics) in models/cones/<name>/run. The simulator uses it with perception:=yolo.

Needs: pip install ultralytics (Ultralytics YOLO11, AGPL-3.0), PyTorch with CUDA for the GPU.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import shutil
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ros2_ws", "src", "fsdsim"))

from fsdsim.vision import CLASS_NAMES  # noqa: E402

INPUT_HW = [384, 640]          # 16:9 camera images letterboxed to 640 x 384 (multiple of 32)


LABELS_VERSION = 2       # cone_dataset_recorder_node.LABELS_VERSION: older recordings have shifted labels


def labels_version(d: str) -> int:
    try:
        with open(os.path.join(d, "session.json"), encoding="utf-8") as f:
            return int(json.load(f).get("labels_version", 1))
    except (OSError, ValueError):
        # a recording of the simulator (frames.jsonl) without session.json is version 1; other folders
        # (e.g. real images in YOLO format) are used as they are
        return 1 if os.path.isfile(os.path.join(d, "frames.jsonl")) else 0


def sessions(data_dir: str) -> list[dict]:
    out = []
    for d in sorted(glob.glob(os.path.join(data_dir, "*"))):
        v = labels_version(d)
        if v == 1:
            print(f"[yolo] skipping {os.path.basename(d)}: recorded before the camera timing fix (its boxes lag the "
                  "cones); record again or delete the folder", flush=True)
            continue
        imgs = sorted(glob.glob(os.path.join(d, "images", "*.jpg")) + glob.glob(os.path.join(d, "images", "*.png")))
        imgs = [p for p in imgs if os.path.exists(os.path.join(d, "labels", os.path.splitext(os.path.basename(p))[0] + ".txt"))]
        if imgs:
            out.append({"name": os.path.basename(d), "images": imgs})
    return out


def split(sess: list[dict], val_share: float, seed: int = 0) -> tuple[list[str], list[str]]:
    """Whole sessions for validation when there are 3 or more; else whole stretches of 25 images."""
    total = sum(len(s["images"]) for s in sess)
    if len(sess) >= 3:
        order = sess[:]
        random.Random(seed).shuffle(order)
        val, n = [], 0
        for s in order:
            if n >= val_share * total:
                break
            val.append(s["name"])
            n += len(s["images"])
        tr = [p for s in sess if s["name"] not in val for p in s["images"]]
        va = [p for s in sess if s["name"] in val for p in s["images"]]
        return tr, va
    blocks = [s["images"][i:i + 25] for s in sess for i in range(0, len(s["images"]), 25)]
    random.Random(seed).shuffle(blocks)
    k = max(1, round(val_share * len(blocks)))
    return [p for b in blocks[k:] for p in b], [p for b in blocks[:k] for p in b]


def count_boxes(images: list[str]) -> list[int]:
    n = [0] * len(CLASS_NAMES)
    for p in images:
        lab = os.path.join(os.path.dirname(os.path.dirname(p)), "labels", os.path.splitext(os.path.basename(p))[0] + ".txt")
        with open(lab, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    c = int(line.split()[0])
                    if 0 <= c < len(n):
                        n[c] += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.path.join(ROOT, "datasets", "cones"))
    ap.add_argument("--name", default=time.strftime("cones_%Y%m%d-%H%M"))
    ap.add_argument("--model", default="yolo11n.pt", help="start weights: yolo11n.pt (fast) or yolo11s.pt (more accurate)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0", help="0 = first CUDA GPU, cpu")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--val-share", type=float, default=0.15)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--check", action="store_true", help="only check the dataset and write the split, no training")
    args = ap.parse_args()

    sess = sessions(args.data)
    n_img = sum(len(s["images"]) for s in sess)
    print(f"[yolo] {n_img} images in {len(sess)} sessions under {args.data}", flush=True)
    if n_img < 50:
        sys.exit(f"[yolo] too few images ({n_img}); record more: launcher -> Simulation -> Isaac Sim, "
                 "'Record camera images for YOLO training', and drive a few autocross / trackdrive runs")
    tr, va = split(sess, args.val_share)
    out = os.path.join(ROOT, "models", "cones", args.name)
    data_dir = os.path.join(out, "data")
    os.makedirs(data_dir, exist_ok=True)
    for fn, lst in (("train.txt", tr), ("val.txt", va)):
        with open(os.path.join(data_dir, fn), "w", encoding="utf-8") as f:
            f.write("\n".join(os.path.abspath(p) for p in lst) + "\n")
    yaml_p = os.path.join(data_dir, "data.yaml")
    with open(yaml_p, "w", encoding="utf-8") as f:
        f.write(f"path: {os.path.abspath(data_dir)}\ntrain: train.txt\nval: val.txt\nnames:\n")
        f.writelines(f"  {i}: {n}\n" for i, n in enumerate(CLASS_NAMES))
    boxes = count_boxes(tr)
    print(f"[yolo] train {len(tr)} images, val {len(va)} images; training boxes per class: "
          + ", ".join(f"{n} {c}" for n, c in zip(CLASS_NAMES, boxes)), flush=True)
    if args.check:
        return

    try:
        from ultralytics import YOLO  # noqa: PLC0415
    except ImportError:
        sys.exit("[yolo] Ultralytics is not installed for this Python: pip install ultralytics "
                 "(launcher: Maintenance -> Install YOLO (Ultralytics) for GPU jobs)")
    t0 = time.time()
    model = YOLO(args.model)
    model.train(data=yaml_p, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=args.device,
                workers=args.workers, patience=args.patience, project=out, name="run", exist_ok=True, seed=0,
                plots=True, verbose=True)
    best = os.path.join(out, "run", "weights", "best.pt")
    trained = YOLO(best)
    met = trained.val(data=yaml_p, imgsz=args.imgsz, device=args.device, plots=False, verbose=False)
    per_class = {}
    try:
        for k, cls_id in enumerate(met.box.ap_class_index):       # classes that have validation boxes
            p, r, ap50, ap = met.box.class_result(k)
            per_class[CLASS_NAMES[int(cls_id)]] = {"precision": round(float(p), 4), "recall": round(float(r), 4),
                                                   "mAP50": round(float(ap50), 4)}
    except Exception:  # noqa: BLE001 - a class without validation boxes
        pass
    metrics = {"mAP50": round(float(met.box.map50), 4), "mAP50_95": round(float(met.box.map), 4),
               "precision": round(float(met.box.mp), 4), "recall": round(float(met.box.mr), 4), "classes": per_class}
    print(f"[yolo] validation: mAP50 {metrics['mAP50']:.3f}, mAP50-95 {metrics['mAP50_95']:.3f}, "
          f"precision {metrics['precision']:.3f}, recall {metrics['recall']:.3f}", flush=True)
    onnx = trained.export(format="onnx", imgsz=INPUT_HW, opset=12, dynamic=False, simplify=True)
    shutil.copyfile(onnx, os.path.join(out, "model.onnx"))
    meta = {"names": CLASS_NAMES, "imgsz": INPUT_HW, "trained": time.strftime("%Y-%m-%d %H:%M"),
            "base_model": args.model, "epochs": args.epochs, "train_imgsz": args.imgsz, "images": n_img,
            "train_images": len(tr), "val_images": len(va), "sessions": [s["name"] for s in sess],
            "train_boxes": dict(zip(CLASS_NAMES, boxes)), "metrics": metrics,
            "train_minutes": round((time.time() - t0) / 60, 1), "software": "Ultralytics YOLO (AGPL-3.0)"}
    with open(os.path.join(out, "model.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(out, "results.md"), "w", encoding="utf-8") as f:
        f.write(f"# Cone detector {args.name}\n\n{n_img} images ({len(tr)} train, {len(va)} validation), "
                f"{args.model}, {args.epochs} epochs, {meta['train_minutes']} min.\n\n"
                "| class | precision | recall | mAP50 |\n|---|---|---|---|\n")
        for c, m in per_class.items():
            f.write(f"| {c} | {m['precision']:.3f} | {m['recall']:.3f} | {m['mAP50']:.3f} |\n")
        f.write(f"| **all** | {metrics['precision']:.3f} | {metrics['recall']:.3f} | {metrics['mAP50']:.3f} |\n\n"
                f"mAP50-95: {metrics['mAP50_95']:.3f}\n")
    print(f"[yolo] model: {os.path.join(out, 'model.onnx')}", flush=True)


if __name__ == "__main__":
    main()
