"""From 2D detections to cones on the map, and how good they are.

* locate(): a detection box + the depth image -> the cone's position on the map (depth of the cone body,
  pushed back by the cone radius to its centre); without depth, the box's foot point on the ground plane.
* ConeMap: detections of many frames merged into one map of cones (position average, colour vote),
  as a driver needs it. Uses the car pose of the simulator (no SLAM).
* Accuracy: frame detections and the map against the simulator's ground truth.
"""

from __future__ import annotations

import math

import numpy as np

from .geometry import back_project, ground_point, to_map

CONE_RADIUS = 0.07      # m: body radius at the height the depth is read (small cone)


def locate(det: dict, depth: np.ndarray | None, K, T_map_cam: np.ndarray, max_depth: float = 25.0,
           ground_z: float = 0.0) -> tuple[float, float, float] | None:
    """(x, y, range) of the detected cone on the map, or None."""
    x1, y1, x2, y2 = det["box"]
    w, h = x2 - x1, y2 - y1
    u = 0.5 * (x1 + x2)
    cam = T_map_cam[:3, 3]
    if depth is not None:
        H, W = depth.shape[:2]
        us = np.clip(np.round(np.linspace(x1 + 0.3 * w, x2 - 0.3 * w, 4)).astype(int), 0, W - 1)
        vs = np.clip(np.round(np.linspace(y1 + 0.4 * h, y2 - 0.15 * h, 5)).astype(int), 0, H - 1)
        d = depth[np.ix_(vs, us)].astype(float).ravel()
        d = d[np.isfinite(d) & (d > 0.3) & (d < max_depth)]
        if d.size >= 3:
            z = float(np.percentile(d, 40))           # the nearer part of the samples: the cone, not behind it
            v = y1 + 0.6 * h
            p = to_map(back_project(u, v, z, K), T_map_cam)
            dx, dy = p[0] - cam[0], p[1] - cam[1]
            r = math.hypot(dx, dy)
            if r > 1e-6:
                return p[0] + CONE_RADIUS * dx / r, p[1] + CONE_RADIUS * dy / r, r + CONE_RADIUS
    g = ground_point(u, y2, T_map_cam, K, ground_z)
    if g is None:
        return None
    r = math.hypot(g[0] - cam[0], g[1] - cam[1])
    return (float(g[0]), float(g[1]), r) if r <= max_depth else None


class ConeMap:
    """Cones seen over many frames: nearest-neighbour association, running mean, colour votes."""

    def __init__(self, gate: float = 0.8, min_hits: int = 2, forget_s: float = 3.0):
        self.gate, self.min_hits, self.forget_s = gate, min_hits, forget_s
        self.cones: list[dict] = []
        self._next = 1

    def reset(self) -> None:
        self.cones, self._next = [], 1

    def update(self, dets: list[dict], t: float) -> None:
        """dets: [{x, y, cls, conf, range}] in the map frame (one camera frame)."""
        used = set()
        pairs = []
        for i, d in enumerate(dets):
            for j, c in enumerate(self.cones):
                dist = math.hypot(d["x"] - c["x"], d["y"] - c["y"])
                if dist <= self.gate:
                    pairs.append((dist, i, j))
        pairs.sort()
        done_d = set()
        for dist, i, j in pairs:
            if i in done_d or j in used:
                continue
            done_d.add(i)
            used.add(j)
            d, c = dets[i], self.cones[j]
            wgt = d["conf"] / (1.0 + 0.05 * d.get("range", 0.0) ** 2)      # near detections are more precise
            tot = c["w"] + wgt
            c["x"] = (c["x"] * c["w"] + d["x"] * wgt) / tot
            c["y"] = (c["y"] * c["w"] + d["y"] * wgt) / tot
            c["w"], c["hits"], c["last"] = tot, c["hits"] + 1, t
            c["votes"][d["cls"]] += d["conf"]
        for i, d in enumerate(dets):
            if i in done_d:
                continue
            votes = [0.0, 0.0, 0.0, 0.0]
            votes[d["cls"]] += d["conf"]
            self.cones.append({"id": self._next, "x": d["x"], "y": d["y"], "w": d["conf"] / (1.0 + 0.05 * d.get("range", 0.0) ** 2),
                               "hits": 1, "first": t, "last": t, "votes": votes})
            self._next += 1
        # one-off detections that were never confirmed are noise
        self.cones = [c for c in self.cones if c["hits"] >= self.min_hits or t - c["last"] <= self.forget_s]

    def confirmed(self) -> list[dict]:
        out = []
        for c in self.cones:
            if c["hits"] < self.min_hits:
                continue
            v = c["votes"]
            k = int(np.argmax(v))
            out.append({"id": c["id"], "x": c["x"], "y": c["y"], "cls": k,
                        "conf": float(v[k] / (sum(v) or 1.0) * min(1.0, c["hits"] / 5.0)), "hits": c["hits"]})
        return out


def _match(pred: list[tuple], truth: list[tuple], gate: float) -> list[tuple[int, int, float]]:
    """Greedy nearest matches (pred index, truth index, distance) within gate."""
    pairs = sorted((math.hypot(p[0] - g[0], p[1] - g[1]), i, j)
                   for i, p in enumerate(pred) for j, g in enumerate(truth))
    pi, tj, out = set(), set(), []
    for d, i, j in pairs:
        if d > gate:
            break
        if i in pi or j in tj:
            continue
        pi.add(i)
        tj.add(j)
        out.append((i, j, d))
    return out


class Accuracy:
    """Running accuracy of the camera perception against the ground truth (last `window` frames)."""

    def __init__(self, window: int = 150, gate: float = 1.0):
        self.window, self.gate = window, gate
        self.frames: list[dict] = []

    def add_frame(self, dets_map: list[dict], visible_truth: list[dict]) -> dict:
        """dets_map: [{x, y, cls}]; visible_truth: frame_labels() of this frame ({cone: (x, y), cls})."""
        pred = [(d["x"], d["y"], d["cls"]) for d in dets_map]
        truth = [(g["cone"][0], g["cone"][1], g["cls"]) for g in visible_truth]
        m = _match(pred, truth, self.gate)
        f = {"pred": len(pred), "truth": len(truth), "tp": len(m),
             "colour_ok": sum(1 for i, j, _ in m if pred[i][2] == truth[j][2]),
             "err_sum": sum(d for _, _, d in m)}
        self.frames.append(f)
        del self.frames[:-self.window]
        return f

    def summary(self) -> dict:
        tp = sum(f["tp"] for f in self.frames)
        pred = sum(f["pred"] for f in self.frames)
        truth = sum(f["truth"] for f in self.frames)
        return {"frames": len(self.frames),
                "recall": tp / truth if truth else None,          # visible cones that were found
                "precision": tp / pred if pred else None,          # detections that are real cones
                "colour": sum(f["colour_ok"] for f in self.frames) / tp if tp else None,
                "error_m": sum(f["err_sum"] for f in self.frames) / tp if tp else None}


def map_accuracy(cmap: list[dict], truth: list[tuple], gate: float = 1.0) -> dict:
    """Confirmed map cones vs all ground-truth cones (x, y, color): how many are right, wrong colour, false."""
    pred = [(c["x"], c["y"], c["cls"]) for c in cmap]
    m = _match(pred, [(t[0], t[1], int(t[2])) for t in truth], gate)
    ok = sum(1 for i, j, _ in m if pred[i][2] == int(truth[j][2]))
    return {"mapped": len(pred), "matched": len(m), "colour_ok": ok, "false": len(pred) - len(m),
            "error_m": (sum(d for _, _, d in m) / len(m)) if m else None}
