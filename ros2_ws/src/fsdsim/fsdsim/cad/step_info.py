"""What a STEP (AP203 / AP214 / AP242) file says about the car, read as text, with no CAD kernel.

STEP files from NX hold boundary-representation solids (planes, cylinders, B-spline surfaces, ...), not
triangles, so they cannot be shown without a tessellator (OpenCascade). This module does what text parsing
can: the unit, the solids with an approximate bounding box, and the wheels, found from the toroidal tyre
faces (centre, axis, radii). From the wheel centres it derives wheelbase, track widths and the wheel radius,
which can be checked against the vehicle parameters.

Bounding boxes use the vertices of the solid, the control points of its B-spline surfaces and curves, and
the analytic extent of its toroidal and spherical faces. Curved faces whose extremes lie between vertices
(cylinders, arcs) can be a few millimetres to centimetres off; control points of B-splines make the box
slightly larger than the surface. Use them to place and compare, not to dimension.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np

_ENT = re.compile(r"#(\d+)\s*=\s*([A-Z_0-9]+)\s*\((.*?)\)\s*;", re.S)
_REF = re.compile(r"#(\d+)")
_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[Ee][-+]?\d+)?"
_VEC = re.compile(rf"\(\s*({_NUM})\s*,\s*({_NUM})\s*,\s*({_NUM})\s*\)")
_SOLID_NAME = re.compile(r"MANIFOLD_SOLID_BREP\s*\(\s*'([^']*)'")


@dataclass
class Solid:
    id: int
    name: str
    lo: np.ndarray                 # bounding box min [m]
    hi: np.ndarray
    faces: int
    torus_faces: int

    @property
    def centre(self) -> np.ndarray:
        return 0.5 * (self.lo + self.hi)

    @property
    def size(self) -> np.ndarray:
        return self.hi - self.lo


@dataclass
class Wheel:
    centre: np.ndarray             # [m], CAD frame
    axis: np.ndarray               # unit vector
    major: float                   # torus major radius [m]
    minor: float                   # tube radius [m]
    y_min: float = 0.0             # extent along the axis [m] (min / max of the tyre shoulders)
    y_max: float = 0.0

    @property
    def outer_radius(self) -> float:
        return self.major + self.minor


@dataclass
class StepInfo:
    path: str
    unit_to_m: float
    solids: list[Solid] = field(default_factory=list)
    wheels: list[Wheel] = field(default_factory=list)
    entities: int = 0

    def bounds(self):
        if not self.solids:
            return None
        return np.min([s.lo for s in self.solids], axis=0), np.max([s.hi for s in self.solids], axis=0)


def _unit_scale(text: str) -> float:
    """Length unit of the file in metres."""
    m = re.search(r"SI_UNIT\s*\(\s*(\.[A-Z]+\.|\$)\s*,\s*\.METRE\.\s*\)", text)
    if m and "LENGTH_UNIT" in text[max(0, m.start() - 120):m.start() + 120]:
        prefix = m.group(1)
        return {".MILLI.": 1e-3, ".CENTI.": 1e-2, ".KILO.": 1e3, "$": 1.0}.get(prefix, 1.0)
    if re.search(r"\.MILLI\.\s*,\s*\.METRE\.", text):
        return 1e-3
    if re.search(r"INCH", text):
        return 0.0254
    return 1e-3                    # most CAD exports are millimetres


def read(path: str) -> StepInfo:
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    ents: dict[int, tuple[str, str]] = {}
    for i, n, a in _ENT.findall(text):
        ents[int(i)] = (n, re.sub(r"\s+", "", a))
    info = StepInfo(path, _unit_scale(text), entities=len(ents))
    k = info.unit_to_m

    pts: dict[int, np.ndarray] = {}
    for i, (n, a) in ents.items():
        if n == "CARTESIAN_POINT":
            m = _VEC.search(a)
            if m:
                pts[i] = np.array([float(x) for x in m.groups()]) * k

    def refs(i):
        return [int(x) for x in _REF.findall(ents[i][1]) if int(x) in ents]

    def placement(i):
        """(origin, z axis) of an AXIS2_PLACEMENT_3D."""
        r = refs(i)
        o = pts[r[0]]
        z = np.array([float(x) for x in _VEC.search(ents[r[1]][1]).groups()]) if len(r) > 1 else np.array([0.0, 0.0, 1.0])
        return o, z / (np.linalg.norm(z) or 1.0)

    solids = [(i, a) for i, (n, a) in ents.items() if n == "MANIFOLD_SOLID_BREP"]
    seen_wheel: dict[tuple, Wheel] = {}
    for sid, a in solids:
        name = (re.match(r"'([^']*)'", a) or [None, f"solid_{sid}"])[1]
        stack, seen = refs(sid), set()
        lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
        faces = tori = 0

        def grow(p, r=0.0):
            nonlocal lo, hi
            lo = np.minimum(lo, p - r)
            hi = np.maximum(hi, p + r)

        while stack:
            i = stack.pop()
            if i in seen:
                continue
            seen.add(i)
            n, aa = ents[i]
            if n == "ADVANCED_FACE":
                faces += 1
            if n == "VERTEX_POINT":
                for r in refs(i):
                    if r in pts:
                        grow(pts[r])
                continue
            if n.startswith("B_SPLINE"):
                for r in refs(i):
                    if r in pts:
                        grow(pts[r])
                continue
            if n == "TOROIDAL_SURFACE":
                tori += 1
                r0 = refs(i)[0]
                o, z = placement(r0)
                nums = [float(x) for x in re.findall(_NUM, aa.split(",", 2)[2])] if aa.count(",") >= 2 else []
                if len(nums) >= 2:
                    R, rr = nums[0] * k, nums[1] * k
                    ext = (R + rr) * np.sqrt(np.clip(1 - z * z, 0, 1)) + rr * np.abs(z)
                    grow(o, ext)
                    key = tuple(np.round(o, 4)) + tuple(np.round(np.abs(z), 3))
                    seen_wheel[key] = (o, z, R, rr, None)
                continue
            if n == "SPHERICAL_SURFACE":
                r0 = refs(i)[0]
                o, _ = placement(r0)
                nums = re.findall(_NUM, aa.split(",", 2)[2]) if aa.count(",") >= 2 else []
                if nums:
                    grow(o, float(nums[0]) * k)
                continue
            if n in ("CARTESIAN_POINT", "DIRECTION", "AXIS2_PLACEMENT_3D", "VECTOR", "LINE", "CIRCLE", "PLANE",
                     "CYLINDRICAL_SURFACE", "CONICAL_SURFACE"):
                continue
            stack.extend(refs(i))
        if np.all(np.isfinite(lo)):
            info.solids.append(Solid(sid, name, lo, hi, faces, tori))

    # wheels: group the toroidal faces by centre x/z, radii and axis direction (two faces per tyre, one per shoulder)
    groups: dict[tuple, list] = {}
    for key, v in seen_wheel.items():
        o, z, R, rr, _ = v
        side = 1.0 if z[int(np.argmax(np.abs(z)))] > 0 else -1.0
        groups.setdefault((round(o[0], 3), round(o[2], 3), round(R, 4), round(rr, 4), side), []).append(o)
    for (x, zc, R, rr, side), items in groups.items():
        ys = [o[1] for o in items]
        c = np.array([x, float(np.mean(ys)), zc])
        info.wheels.append(Wheel(c, np.array([0.0, side, 0.0]), R, rr, min(ys) - rr, max(ys) + rr))
    info.wheels.sort(key=lambda w: (w.centre[0], w.centre[1]))
    return info


def vehicle_dimensions(info: StepInfo) -> dict | None:
    """Wheelbase, track widths, wheel radius from the wheel centres (needs four wheels)."""
    if len(info.wheels) < 4:
        return None
    xs = sorted({round(w.centre[0], 3) for w in info.wheels})
    if len(xs) < 2:
        return None
    front, rear = xs[0], xs[-1]

    def track(x):
        ys = [w.centre[1] for w in info.wheels if abs(w.centre[0] - x) < 1e-3]
        return float(max(ys) - min(ys)) if len(ys) >= 2 else math.nan

    w0 = info.wheels[0]
    return {"wheelbase": rear - front, "track_front": track(front), "track_rear": track(rear),
            "wheel_radius_centre_height": float(np.mean([w.centre[2] for w in info.wheels])),
            "tyre_outer_radius": float(w0.outer_radius), "tyre_width": float(w0.y_max - w0.y_min),
            "x_front_axle": front, "x_rear_axle": rear}
