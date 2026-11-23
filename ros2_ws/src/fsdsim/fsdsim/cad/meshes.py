"""Triangle meshes of the car -> USD files for the Isaac Sim backend.

Readers: binary / ASCII STL, OBJ (objects / groups become parts), STEP through `cadquery` when it is
installed (`pip install cadquery`; it carries OpenCascade, which tessellates the B-rep solids). Then:

* `Frame`: CAD coordinates -> simulator frame (x forward, y left, z up, origin at the centre of gravity on
  the ground, metres). The LR26 CAD has the front axle at x = 0, x pointing to the rear, millimetres.
* `split_wheels`: the tyres go to the wheel prims so they can steer and spin; everything else is the body.
  Parts that lie completely inside a wheel's envelope become wheel; a mesh that is a single part (one STL of
  the whole car) is split by triangle instead.
* `write_usda`: ASCII USD with points, faces and smooth normals (hard edges kept above a crease angle).
"""

from __future__ import annotations

import math
import os
import re
import struct
from dataclasses import dataclass

import numpy as np


@dataclass
class Part:
    name: str
    v: np.ndarray                  # (N, 3) float64
    f: np.ndarray                  # (M, 3) int
    colour: tuple | None = None

    def copy(self, **kw) -> "Part":
        d = dict(name=self.name, v=self.v.copy(), f=self.f.copy(), colour=self.colour)
        d.update(kw)
        return Part(**d)

    @property
    def bounds(self):
        return self.v.min(axis=0), self.v.max(axis=0)


# --------------------------------------------------------------------------- #
# readers
# --------------------------------------------------------------------------- #
def _weld(tri: np.ndarray, decimals: int = 6):
    """(T, 3, 3) triangle soup -> shared vertices and faces; degenerate triangles are dropped."""
    pts = tri.reshape(-1, 3)
    key = np.round(pts, decimals)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    f = inv.reshape(-1, 3)
    ok = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
    return uniq, f[ok]


def read_stl(path: str, name: str | None = None) -> list[Part]:
    name = name or os.path.splitext(os.path.basename(path))[0]
    with open(path, "rb") as fh:
        data = fh.read()
    n = struct.unpack_from("<I", data, 80)[0] if len(data) >= 84 else -1
    if len(data) == 84 + 50 * n:                              # binary
        dt = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
        tri = np.frombuffer(data, dtype=dt, count=n, offset=84)["v"].astype(np.float64)
    else:                                                     # ASCII
        txt = data.decode("utf-8", "replace")
        nums = re.findall(r"vertex\s+(\S+)\s+(\S+)\s+(\S+)", txt)
        if not nums:
            raise ValueError(f"{path}: not an STL file")
        tri = np.array(nums, dtype=np.float64).reshape(-1, 3, 3)
    v, f = _weld(tri)
    return [Part(name, v, f)]


def read_obj(path: str) -> list[Part]:
    verts: list[list[float]] = []
    groups: dict[str, list[list[int]]] = {}
    cur = os.path.splitext(os.path.basename(path))[0]
    colours: dict[str, tuple] = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("usemtl "):                      # Autodesk ATF names the colour: Opaque(156,168,171)
                m = re.search(r"\((\d+)\s*,\s*(\d+)\s*,\s*(\d+)\)", line)
                if m:
                    colours[cur] = tuple(int(x) / 255.0 for x in m.groups())
            elif line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith(("o ", "g ")):
                cur = line[2:].strip() or cur
            elif line.startswith("f "):
                idx = []
                for tok in line.split()[1:]:
                    i = int(tok.split("/")[0])
                    idx.append(i - 1 if i > 0 else len(verts) + i)
                for k in range(1, len(idx) - 1):               # fan
                    groups.setdefault(cur, []).append([idx[0], idx[k], idx[k + 1]])
    V = np.array(verts, float)
    parts = []
    for name, faces in groups.items():
        f = np.array(faces)
        used, inv = np.unique(f, return_inverse=True)
        parts.append(Part(name, V[used], inv.reshape(-1, 3), colours.get(name)))
    return parts


def read_step(path: str, tolerance_mm: float = 0.3, angular: float = 0.2) -> list[Part]:
    """STEP -> one part per solid (millimetres), with cadquery. Not available offline: see docs/CAD.md."""
    try:
        import cadquery as cq  # noqa: PLC0415
    except ImportError as e:  # pragma: no cover - environment dependent
        raise ImportError("reading STEP needs cadquery (python3 -m pip install cadquery), or export STL / OBJ from the CAD") from e
    shape = cq.importers.importStep(path).val()
    base = os.path.splitext(os.path.basename(path))[0]
    parts = []
    for i, solid in enumerate(shape.Solids()):
        verts, tris = solid.tessellate(tolerance_mm, angular)
        if not tris:
            continue
        v = np.array([[p.x, p.y, p.z] for p in verts], float)
        parts.append(Part(f"{base}_{i:03d}", v, np.array(tris, int)))
    return parts


def read_any(path: str, **kw) -> list[Part]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".stl":
        return read_stl(path)
    if ext == ".obj":
        return read_obj(path)
    if ext in (".stp", ".step"):
        return read_step(path, **kw)
    raise ValueError(f"{path}: unsupported format (use .stl, .obj or .stp)")


# --------------------------------------------------------------------------- #
# frame and wheel split
# --------------------------------------------------------------------------- #
@dataclass
class Frame:
    """CAD -> simulator. Defaults: the LR26 CAD (mm, front axle at x = 0, x to the rear)."""

    scale: float = 1e-3
    x_front_axle: float = 0.0      # [CAD units]
    lf: float = 0.85065            # CoG -> front axle [m]
    x_rearward: bool = True

    def apply(self, v: np.ndarray) -> np.ndarray:
        p = np.asarray(v, float) * self.scale
        xa = self.x_front_axle * self.scale
        if self.x_rearward:
            return np.stack([self.lf - (p[:, 0] - xa), -p[:, 1], p[:, 2]], axis=1)
        return np.stack([self.lf + (p[:, 0] - xa), p[:, 1], p[:, 2]], axis=1)


def to_sim(parts: list[Part], frame: Frame) -> list[Part]:
    """Parts in the simulator frame (the CAD -> simulator change is a rotation or a shift, so winding is kept)."""
    return [Part(p.name, frame.apply(p.v), p.f.copy(), p.colour) for p in parts]


@dataclass
class WheelSpec:
    name: str
    centre: tuple                  # [m] simulator frame
    radius: float                  # outer radius incl. tyre [m]
    half_width: float


def default_wheels(lf: float, lr: float, track_f: float, track_r: float, outer_radius: float, width: float,
                   hub_z: float | None = None) -> list[WheelSpec]:
    z = hub_z if hub_z is not None else outer_radius
    return [WheelSpec("fl", (lf, track_f / 2, z), outer_radius, width / 2), WheelSpec("fr", (lf, -track_f / 2, z), outer_radius, width / 2),
            WheelSpec("rl", (-lr, track_r / 2, z), outer_radius, width / 2), WheelSpec("rr", (-lr, -track_r / 2, z), outer_radius, width / 2)]


def _sub(part: Part, mask: np.ndarray, name: str) -> Part | None:
    if not mask.any():
        return None
    f = part.f[mask]
    used, inv = np.unique(f, return_inverse=True)
    return Part(name, part.v[used], inv.reshape(-1, 3), part.colour)


def split_wheels(parts: list[Part], wheels: list[WheelSpec], tol: float = 0.015, hub_radius: float = 0.09):
    """(body parts, {wheel name: parts centred on the hub}).

    A part completely inside a wheel's envelope (radius + tol, half width + tol) is that wheel's. Triangles are
    split only when the whole model is one part (a single STL), from the hub radius outwards, so the upright and
    the suspension stay with the body."""
    body: list[Part] = []
    out: dict[str, list[Part]] = {w.name: [] for w in wheels}

    def centred(p: Part, w: WheelSpec) -> Part:
        return p.copy(v=p.v - np.array(w.centre))

    def inside(lo, hi, w):
        c = np.array(w.centre)
        return (abs(lo[0] - c[0]) <= w.radius + tol and abs(hi[0] - c[0]) <= w.radius + tol and abs(lo[1] - c[1]) <= w.half_width + tol
                and abs(hi[1] - c[1]) <= w.half_width + tol and lo[2] >= c[2] - w.radius - tol and hi[2] <= c[2] + w.radius + tol)

    single = len(parts) == 1
    for p in parts:
        lo, hi = p.bounds
        owner = next((w for w in wheels if inside(lo, hi, w)), None)
        if owner is not None:
            out[owner.name].append(centred(p, owner))
            continue
        if not single:
            body.append(p)
            continue
        cen = p.v[p.f].mean(axis=1)
        rest = np.ones(len(p.f), bool)
        for w in wheels:
            d = cen - np.array(w.centre)
            r = np.hypot(d[:, 0], d[:, 2])
            m = rest & (np.abs(d[:, 1]) <= w.half_width + tol) & (r <= w.radius + tol) & (r >= hub_radius)
            sub = _sub(p, m, f"{p.name}_{w.name}")
            if sub is not None:
                out[w.name].append(centred(sub, w))
            rest &= ~m
        sub = _sub(p, rest, p.name)
        if sub is not None:
            body.append(sub)
    return body, out


def simplify(part: Part, max_tris: int) -> Part:
    """Vertex clustering: merge vertices on a grid, growing the cell until at most `max_tris` triangles remain.

    Meant for small hardware tessellated far finer than it can be seen (springs, threads); the shape stays within a
    few cell widths. Returns the part unchanged if it is small enough."""
    if max_tris <= 0 or len(part.f) <= max_tris:
        return part
    lo, hi = part.bounds
    diag = float(np.linalg.norm(hi - lo)) or 1.0
    best = part
    a, b = diag / 2000.0, diag / 4.0
    for _ in range(14):                                            # bisection on the cell size
        cell = 0.5 * (a + b)
        q = np.floor((part.v - lo) / cell).astype(np.int64)
        _, first, inv = np.unique(q, axis=0, return_index=True, return_inverse=True)
        inv = inv.reshape(-1)
        # representative = mean of the cluster
        cnt = np.bincount(inv)
        nv = np.stack([np.bincount(inv, weights=part.v[:, k]) / cnt for k in range(3)], axis=1)
        f = inv[part.f]
        ok = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
        f = f[ok]
        if len(f):
            key = np.sort(f, axis=1)
            _, uniq = np.unique(key, axis=0, return_index=True)
            f = f[np.sort(uniq)]
        if len(f) <= max_tris:
            used, back = np.unique(f, return_inverse=True)
            best = Part(part.name, nv[used], back.reshape(-1, 3), part.colour)
            b = cell
        else:
            a = cell
    return best


def clearance(parts: list[Part], points: dict[str, tuple], radius: float = 0.10) -> dict[str, int]:
    """Number of vertices within `radius` of each point (sensor mounts), to see whether the model blocks a sensor."""
    res = {}
    for k, pt in points.items():
        n = 0
        for p in parts:
            n += int(np.count_nonzero(np.linalg.norm(p.v - np.array(pt), axis=1) < radius))
        res[k] = n
    return res


def inside_cylinder(parts: list[Part], centre: tuple, radius: float, half_height: float) -> int:
    """How much of the model is inside a vertical cylinder (a sensor's own body, e.g. the OS1 at its mount):
    vertices inside it, plus triangles that cut its axis or its rim (large triangles of a coarse surface can pass
    through the sensor with no vertex inside it). 0 = the sensor is free."""
    c = np.asarray(centre, float)
    lo, hi = c - [radius, radius, half_height], c + [radius, radius, half_height]
    segs = [(c - [0, 0, half_height], c + [0, 0, half_height])]
    for a in np.linspace(0.0, 2 * np.pi, 8, endpoint=False):
        r = radius * np.array([np.cos(a), np.sin(a), 0.0])
        segs += [(c + r - [0, 0, half_height], c + r + [0, 0, half_height]), (c, c + r)]
    n = 0
    for p in parts:
        d = p.v - c
        n += int(np.count_nonzero((np.hypot(d[:, 0], d[:, 1]) < radius) & (np.abs(d[:, 2]) < half_height)))
        tri = p.v[p.f]
        near = np.all(tri.max(axis=1) >= lo, axis=1) & np.all(tri.min(axis=1) <= hi, axis=1)
        if near.any():
            t = tri[near]
            for a, b in segs:
                n += int(np.count_nonzero(_segment_hits(a, b, t)))
    return n


def _segment_hits(a, b, tri: np.ndarray) -> np.ndarray:
    """Moller-Trumbore: which triangles (N, 3, 3) the segment a-b crosses."""
    d = np.asarray(b, float) - np.asarray(a, float)
    e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    pv = np.cross(d, e2)
    det = np.einsum("ij,ij->i", e1, pv)
    ok = np.abs(det) > 1e-12
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    tv = np.asarray(a, float) - tri[:, 0]
    u = np.einsum("ij,ij->i", tv, pv) * inv
    qv = np.cross(tv, e1)
    v = (qv @ d) * inv
    t = np.einsum("ij,ij->i", e2, qv) * inv
    return ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t >= 0) & (t <= 1)


# --------------------------------------------------------------------------- #
# USD
# --------------------------------------------------------------------------- #
def corner_normals(part: Part, crease_deg: float = 40.0, max_valence: int = 24) -> np.ndarray:
    """(M*3, 3) face-varying normals: area-weighted over the faces at a vertex whose normal is within the crease angle."""
    v, f = part.v, part.f
    tri = v[f]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(fn, axis=1)
    fn = fn / np.maximum(area[:, None], 1e-300)
    nv = len(v)
    corner_v = f.reshape(-1)
    face_of = np.repeat(np.arange(len(f)), 3)
    order = np.argsort(corner_v, kind="stable")
    sv, sf = corner_v[order], face_of[order]
    first = np.searchsorted(sv, np.arange(nv))
    counts = np.bincount(sv, minlength=nv)
    k = int(min(max(counts.max(), 1), max_valence))
    pad = np.full((nv, k), -1, int)
    for j in range(k):                                         # column j = j-th face of every vertex
        has = counts > j
        pad[has, j] = sf[first[has] + j]
    cos_c = math.cos(math.radians(crease_deg))
    nb = pad[corner_v]                                          # (3M, k) neighbouring faces of every corner
    valid = nb >= 0
    nbn = fn[np.where(valid, nb, 0)]
    dots = np.einsum("ckd,cd->ck", nbn, fn[face_of])
    w = np.where(valid & (dots >= cos_c), area[np.where(valid, nb, 0)], 0.0)
    n = np.einsum("ck,ckd->cd", w, nbn)
    ln = np.linalg.norm(n, axis=1)
    return np.where(ln[:, None] > 0, n / np.maximum(ln, 1e-300)[:, None], fn[face_of])


def _arr(a: np.ndarray, fmt: str) -> str:
    return "[" + ", ".join(fmt % tuple(r) if a.ndim > 1 else fmt % r for r in a) + "]"


def _ident(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_]", "_", name)
    return ("p_" + s) if not re.match(r"[A-Za-z_]", s) else s


def write_usda(path: str, root_name: str, parts: list[Part], default_colour=(0.6, 0.6, 0.6), normals: bool = True,
               crease_deg: float = 40.0, comment: str = "") -> dict:
    """Write `parts` under a Z-up, metre USD whose default prim is `root_name`. Returns {parts, triangles}."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    seen: dict[str, int] = {}
    tris = 0
    with open(path, "w", encoding="utf-8") as fh:
        fh.write('#usda 1.0\n(\n    defaultPrim = "%s"\n    metersPerUnit = 1\n    upAxis = "Z"\n' % root_name)
        if comment:
            fh.write('    doc = "%s"\n' % comment.replace('"', "'"))
        fh.write(")\n\n")
        fh.write('def Xform "%s"\n{\n' % root_name)
        for p in parts:
            base = _ident(p.name)
            seen[base] = seen.get(base, 0) + 1
            nm = base if seen[base] == 1 else f"{base}_{seen[base]}"
            col = p.colour or default_colour
            fh.write('    def Mesh "%s"\n    {\n' % nm)
            fh.write("        int[] faceVertexCounts = [%s]\n" % ", ".join(["3"] * len(p.f)))
            fh.write("        int[] faceVertexIndices = [%s]\n" % ", ".join(map(str, p.f.reshape(-1).tolist())))
            fh.write("        point3f[] points = %s\n" % _arr(p.v, "(%.5f, %.5f, %.5f)"))
            if normals:
                n = corner_normals(p, crease_deg)
                fh.write("        normal3f[] normals = %s (\n            interpolation = \"faceVarying\"\n        )\n" % _arr(n, "(%.3f, %.3f, %.3f)"))
            fh.write("        color3f[] primvars:displayColor = [(%.3f, %.3f, %.3f)]\n" % tuple(col))
            fh.write('        uniform token subdivisionScheme = "none"\n        bool doubleSided = 1\n    }\n')
            tris += len(p.f)
        fh.write("}\n")
    return {"parts": len(parts), "triangles": tris}


def read_usda(path: str) -> dict:
    """Minimal reader of what `write_usda` writes (for tests): {mesh name: (points, faceVertexIndices)}."""
    with open(path, encoding="utf-8") as fh:
        txt = fh.read()
    out = {}
    for m in re.finditer(r'def Mesh "([^"]+)"\s*\{(.*?)\n    \}', txt, re.S):
        body = m.group(2)
        pts = re.search(r"point3f\[\] points = \[(.*?)\]\n", body, re.S).group(1)
        idx = re.search(r"faceVertexIndices = \[(.*?)\]\n", body, re.S).group(1)
        P = np.array(re.findall(r"\(([^)]*)\)", pts), dtype=object)
        P = np.array([[float(x) for x in s.split(",")] for s in P])
        out[m.group(1)] = (P, np.array([int(x) for x in idx.split(",")]) if idx.strip() else np.zeros(0, int))
    return out
