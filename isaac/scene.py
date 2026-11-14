"""USD scene authoring for the Isaac Sim backend (plain pxr API, version-stable).

/World
  /Ground                 asphalt plane
  /Lights                 sun + dome
  /Cones/cone_XXXX        instanced references to /_ConeProtos/<COLOR>
  /Car                    base_link (moved every frame from the fsdsim vehicle model)
    /body                 procedural car or your USD (vehicle.yaml: usd_path; tools/cad_to_usd.py)
    /wheel_fl ...         steer (rotateZ) -> spin (rotateY); tyre = cylinder or your USD (usd_wheel_path)
    /os_lidar_mount       RTX LiDAR parent (sensors.yaml: lidar.mount)
    /zed_mount            ZED centre (sensors.yaml: camera.mount)
      /left, /right       USD cameras (looking along +X of the car)
/_ConeProtos              class prims (not rendered), one per cone type
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, Vt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "ros2_ws", "src", "fsdsim"))

from fsdsim.geometry import rpy_to_matrix  # noqa: E402
from fsdsim.track import CONE_SPECS, ConeColor  # noqa: E402

from isaac_compat import quat_wxyz_from_matrix  # noqa: E402

COLORS = {
    "blue": (0.05, 0.18, 0.75),
    "yellow": (0.95, 0.80, 0.02),
    "orange": (0.95, 0.38, 0.03),
    "white": (0.92, 0.92, 0.92),
    "black": (0.03, 0.03, 0.03),
    "asphalt": (0.18, 0.18, 0.19),
    "car": (0.75, 0.05, 0.05),
    "carbon": (0.06, 0.06, 0.07),
    "tyre": (0.02, 0.02, 0.02),
}

# USD camera looks down -Z with +Y up; ROS camera body is x forward, z up.
R_CAM_CONV = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def _quat(R: np.ndarray) -> Gf.Quatd:
    w, x, y, z = quat_wxyz_from_matrix(R)
    return Gf.Quatd(w, x, y, z)


def _xform(stage, path, t=(0, 0, 0), R=None):
    x = UsdGeom.Xform.Define(stage, path)
    x.ClearXformOpOrder()
    top = x.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
    top.Set(Gf.Vec3d(*map(float, t)))
    oop = x.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
    oop.Set(_quat(np.eye(3) if R is None else R))
    return x, top, oop


class Scene:
    def __init__(self, stage, track, vehicle, sensors):
        self.stage = stage
        self.track = track
        self.vp = vehicle
        self.sc = sensors
        self.materials = {}
        self._car_t = self._car_o = None
        self._wheels = []
        self._wheel_z = []
        self._spin = [0.0, 0.0, 0.0, 0.0]
        self._cone_ops = {}
        self._knocked_done = set()

    # ------------------------------------------------------------------ #
    def build(self):
        st = self.stage
        UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(st, 1.0)
        if not st.GetPrimAtPath("/World"):
            UsdGeom.Xform.Define(st, "/World")
        for name, rgb in COLORS.items():
            self.materials[name] = self._material(name, rgb)
        self._ground()
        self._lights()
        self._cone_prototypes()
        self._cones()
        self._car()

    def _material(self, name, rgb, roughness=0.6, root="/World/Looks"):
        path = f"{root}/{name}"
        mat = UsdShade.Material.Define(self.stage, path)
        sh = UsdShade.Shader.Define(self.stage, path + "/Shader")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        return mat

    def _bind(self, prim, name, materials=None):
        mat = (materials or self.materials)[name]
        UsdShade.MaterialBindingAPI.Apply(prim.GetPrim() if hasattr(prim, "GetPrim") else prim).Bind(mat)

    def _ground(self):
        half = 400.0
        m = UsdGeom.Mesh.Define(self.stage, "/World/Ground")
        m.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(-half, -half, 0), Gf.Vec3f(half, -half, 0),
                                          Gf.Vec3f(half, half, 0), Gf.Vec3f(-half, half, 0)]))
        m.CreateFaceVertexCountsAttr([4])
        m.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        m.CreateNormalsAttr(Vt.Vec3fArray([Gf.Vec3f(0, 0, 1)] * 4))
        m.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        self._bind(m, "asphalt")

    def _lights(self):
        sun = UsdLux.DistantLight.Define(self.stage, "/World/Lights/Sun")
        sun.CreateIntensityAttr(2500.0)
        sun.CreateAngleAttr(0.53)
        UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-40.0, 25.0, 0.0))
        dome = UsdLux.DomeLight.Define(self.stage, "/World/Lights/Sky")
        dome.CreateIntensityAttr(600.0)

    # ------------------------------------------------------------------ #
    def _frustum(self, path, z0, z1, r0, r1, n=24):
        m = UsdGeom.Mesh.Define(self.stage, path)
        pts, counts, idx = [], [], []
        for k in range(n):
            a = 2 * math.pi * k / n
            pts.append(Gf.Vec3f(r0 * math.cos(a), r0 * math.sin(a), z0))
            pts.append(Gf.Vec3f(r1 * math.cos(a), r1 * math.sin(a), z1))
        for k in range(n):
            a, b = 2 * k, 2 * ((k + 1) % n)
            counts.append(4)
            idx += [a, b, b + 1, a + 1]
        m.CreatePointsAttr(Vt.Vec3fArray(pts))
        m.CreateFaceVertexCountsAttr(counts)
        m.CreateFaceVertexIndicesAttr(idx)
        m.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        return m

    def _cone_prototypes(self):
        self.stage.CreateClassPrim("/_ConeProtos")
        for color in ConeColor:
            spec = CONE_SPECS[color]
            root = f"/_ConeProtos/{color.name}"
            UsdGeom.Xform.Define(self.stage, root)
            # materials must live inside the prototype: bindings that point outside a
            # referenced prim are out of scope in USD composition and would be dropped
            needed = {spec.body, "black", "white", "orange"} | {st for _, _, st in spec.stripes}
            mats = {n: self._material(n, COLORS[n], root=f"{root}/Looks") for n in needed}
            plate = UsdGeom.Cube.Define(self.stage, root + "/base")
            plate.CreateSizeAttr(1.0)
            UsdGeom.Xformable(plate).AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.0125))
            UsdGeom.Xformable(plate).AddScaleOp().Set(Gf.Vec3f(spec.base, spec.base, 0.025))
            self._bind(plate, "black" if color != ConeColor.ORANGE_BIG else "orange", mats)
            # body split into colour bands (stripes)
            bands, z = [], 0.0
            for z_from, z_to, colour in spec.stripes:
                bands.append((z, z_from, spec.body))
                bands.append((z_from, z_to, colour))
                z = z_to
            bands.append((z, 1.0, spec.body))

            def radius(f):
                return spec.r_bottom + (spec.r_top - spec.r_bottom) * f

            for i, (f0, f1, colour) in enumerate(bands):
                if f1 - f0 < 1e-6:
                    continue
                m = self._frustum(f"{root}/band{i}", 0.025 + f0 * (spec.height - 0.025),
                                  0.025 + f1 * (spec.height - 0.025), radius(f0), radius(f1))
                self._bind(m, colour, mats)

    def _cones(self):
        UsdGeom.Scope.Define(self.stage, "/World/Cones")
        for c in self.track.cones:
            path = f"/World/Cones/cone_{c.id:04d}"
            prim = self.stage.DefinePrim(path, "Xform")
            prim.GetReferences().AddInternalReference(f"/_ConeProtos/{c.color.name}")
            prim.SetInstanceable(True)
            x = UsdGeom.Xformable(prim)
            x.ClearXformOpOrder()
            t = x.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
            t.Set(Gf.Vec3d(float(c.x), float(c.y), 0.0))
            r = x.AddRotateXYZOp()
            r.Set(Gf.Vec3f(0, 0, 0))
            self._cone_ops[c.id] = (t, r)

    # ------------------------------------------------------------------ #
    def _car(self):
        p, st = self.vp, self.stage
        x0, y0, yaw0 = self.track.start_pose
        _, self._car_t, self._car_o = _xform(st, "/World/Car", (x0, y0, 0.0), rpy_to_matrix(0, 0, yaw0))
        usd = self._resolve(p.usd_path)
        if usd:
            body = st.DefinePrim("/World/Car/body", "Xform")
            body.GetReferences().AddReference(usd)
            if p.usd_fill_body or p.usd_fill_hoop:
                self._body_filler(p.usd_fill_body, p.usd_fill_hoop)
            self._paint_body(body)
        else:
            self._procedural_body()
        # wheels
        tf, tr = p.track_front / 2, p.track_rear / 2
        for name, x, y in (("fl", p.lf, tf), ("fr", p.lf, -tf), ("rl", -p.lr, tr), ("rr", -p.lr, -tr)):
            steer = UsdGeom.Xform.Define(st, f"/World/Car/wheel_{name}")
            wt = steer.AddTranslateOp()
            wt.Set(Gf.Vec3d(x, y, p.wheel_radius))
            self._wheel_z.append((wt, x, y))
            rz = steer.AddRotateZOp()
            spin = UsdGeom.Xform.Define(st, f"/World/Car/wheel_{name}/spin")
            ry = spin.AddRotateYOp()
            wheel_usd = self._resolve(p.usd_wheel_path.replace("{name}", name)) if p.usd_wheel_path else ""
            if wheel_usd:
                tyre = st.DefinePrim(f"/World/Car/wheel_{name}/spin/tyre", "Xform")
                tyre.GetReferences().AddReference(wheel_usd)
                self._paint_wheel(tyre, p.wheel_radius)
            else:
                cyl = UsdGeom.Cylinder.Define(st, f"/World/Car/wheel_{name}/spin/tyre")
                cyl.CreateAxisAttr("Y")
                cyl.CreateRadiusAttr(p.wheel_radius)
                cyl.CreateHeightAttr(0.2)
                self._bind(cyl, "tyre")
            self._wheels.append((rz, ry, name.startswith("f")))
        # sensor mounts
        lm = self.sc.lidar.mount
        R_l = rpy_to_matrix(lm.roll, lm.pitch, lm.yaw + math.radians(self.sc.lidar.isaac.yaw_offset_deg))
        _xform(st, "/World/Car/os_lidar_mount", (lm.x, lm.y, lm.z), R_l)
        cm = self.sc.camera.mount
        _xform(st, "/World/Car/zed_mount", (cm.x, cm.y, cm.z), rpy_to_matrix(cm.roll, cm.pitch, cm.yaw))

    @staticmethod
    def _resolve(path: str) -> str:
        """A USD path from vehicle.yaml: relative paths are relative to the repository; "" if the file is missing."""
        if not path:
            return ""
        full = path if os.path.isabs(path) else os.path.join(os.path.dirname(HERE), path)
        if not os.path.isfile(full):
            print(f"[fsd-isaac] USD file not found: {full} (using the procedural model)", flush=True)
            return ""
        full = os.path.normpath(full).replace("\\", "/")     # USD asset paths: forward slashes on Windows too
        print(f"[fsd-isaac] USD model: {full}", flush=True)
        return full

    # ------------------------------------------------------------------ #
    # Materials for the CAD car. tools/cad_to_usd.py writes only a flat displayColor per part, which the RTX renderer
    # does not shade like paint, carbon or rubber, and its colour guess painted every part longer than 0.3 m red,
    # the suspension tubes too; one wheel kept a light grey from the CAD file while the others were black
    # (team PC, 13 Nov 2026). Here each part gets a real material, chosen from its colour and its shape.
    CAR_LOOKS = {   # name: (base colour, roughness, metallic)
        "paint": ((0.75, 0.05, 0.05), 0.35, 0.0),
        "carbon": ((0.05, 0.05, 0.055), 0.45, 0.0),
        "metal": ((0.55, 0.55, 0.58), 0.35, 0.8),
        "steel": ((0.62, 0.66, 0.67), 0.3, 0.9),
        "rubber": ((0.02, 0.02, 0.02), 0.9, 0.0),
        "rim": ((0.32, 0.32, 0.34), 0.3, 0.85),
    }

    @staticmethod
    def classify_part(colour, points) -> str:
        """Material for one CAD part from its displayColor and its points (car frame, metres)."""
        c = np.asarray(colour, float)
        P = np.asarray(points, float)
        if len(P) >= 3:
            q = P - P.mean(0)
            ext = np.ptp(q @ np.linalg.svd(q, full_matrices=False)[2].T, axis=0)
            if ext[0] > 0.2 and 0.005 < ext[1] < 0.03 and ext[2] > 0.5 * ext[1]:
                return "metal"            # a rod (round section): wishbones, push / pull rods, tie rods
            # (the nose and the wings are made of long thin ribbons too, but flat: those keep their colour)
        if c[0] > 0.5 and c[1] < 0.2 and c[2] < 0.2:
            return "paint"
        if c.max() < 0.12:
            return "carbon"
        if abs(c[0] - c[2]) > 0.15:       # a stray CAD colour (beige on the nose): body paint
            return "paint"
        return "steel" if c.mean() > 0.6 else "metal"

    def _car_material(self, name):
        key = f"car_{name}"
        if key not in self.materials:
            rgb, rough, metal = self.CAR_LOOKS[name]
            mat = self._material(key, rgb, roughness=rough, root="/World/Car/Looks")
            UsdShade.Shader(self.stage.GetPrimAtPath(f"/World/Car/Looks/{key}/Shader")).GetInput("metallic").Set(metal)
            self.materials[key] = mat
        return self.materials[key]

    def _paint_body(self, body_prim) -> dict:
        """Bind a material to every mesh of the referenced CAD body; returns {material: number of parts}."""
        counts = {}
        for prim in Usd.PrimRange(body_prim):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            mesh = UsdGeom.Mesh(prim)
            pv = UsdGeom.PrimvarsAPI(prim).GetPrimvar("displayColor")
            col = pv.Get() if pv else None
            colour = tuple(col[0]) if col is not None and len(col) else (0.55, 0.55, 0.58)
            name = self.classify_part(colour, mesh.GetPointsAttr().Get() or [])
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(self._car_material(name))
            mesh.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set([Gf.Vec3f(*self.CAR_LOOKS[name][0])])
            counts[name] = counts.get(name, 0) + 1
        return counts

    def _paint_wheel(self, wheel_prim, radius: float) -> None:
        """Rubber tyre, metal rim: faces inside 80 % of the wheel radius (about the wheel's Y axis) are the rim."""
        for prim in Usd.PrimRange(wheel_prim):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            mesh = UsdGeom.Mesh(prim)
            P = np.asarray(mesh.GetPointsAttr().Get() or [], float)
            idx = np.asarray(mesh.GetFaceVertexIndicesAttr().Get() or [], int)
            counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get() or [], int)
            binding = UsdShade.MaterialBindingAPI.Apply(prim)
            binding.Bind(self._car_material("rubber"))
            mesh.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set([Gf.Vec3f(*self.CAR_LOOKS["rubber"][0])])
            if not len(P) or not len(counts):
                continue
            r = np.hypot(P[:, 0], P[:, 2])
            starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
            rmax = np.maximum.reduceat(r[idx], starts)
            rim = np.nonzero(rmax < 0.8 * radius)[0]
            if len(rim):
                sub = binding.CreateMaterialBindSubset("rim", Vt.IntArray([int(i) for i in rim]), UsdGeom.Tokens.face)
                UsdShade.MaterialBindingAPI.Apply(sub.GetPrim()).Bind(self._car_material("rim"))

    def _body_filler(self, tub: bool = True, hoop: bool = True):
        """Parts the CAD may lack: a simple monocoque tub, and the main hoop under the camera (the ZED sits on top of it)."""
        st = self.stage
        if tub:
            c = UsdGeom.Cube.Define(st, "/World/Car/body_filler_tub")
            c.CreateSizeAttr(1.0)
            c.AddTranslateOp().Set(Gf.Vec3d(0.05, 0.0, 0.28))
            c.AddScaleOp().Set(Gf.Vec3f(1.3, 0.46, 0.36))
            self._bind(c, "car")
        if hoop:
            cm = self.sc.camera.mount
            top = cm.z - 0.08
            h = UsdGeom.Cylinder.Define(st, "/World/Car/body_filler_hoop")
            h.CreateRadiusAttr(0.03)
            h.CreateHeightAttr(top - 0.4)
            h.AddTranslateOp().Set(Gf.Vec3d(cm.x, 0.0, 0.4 + (top - 0.4) / 2))
            self._bind(h, "carbon")

    def _procedural_body(self):
        p, st = self.vp, self.stage
        L = p.front_bumper + p.rear_bumper
        cx = 0.5 * (p.front_bumper - p.rear_bumper)

        def box(name, center, size, mat):
            c = UsdGeom.Cube.Define(st, f"/World/Car/body/{name}")
            c.CreateSizeAttr(1.0)
            c.AddTranslateOp().Set(Gf.Vec3d(*center))
            c.AddScaleOp().Set(Gf.Vec3f(*size))
            self._bind(c, mat)

        UsdGeom.Xform.Define(st, "/World/Car/body")
        box("monocoque", (cx, 0.0, 0.32), (L * 0.78, 0.55, 0.32), "car")
        box("nose", (p.front_bumper - 0.35, 0.0, 0.25), (0.7, 0.32, 0.18), "car")
        box("front_wing", (p.front_bumper - 0.12, 0.0, 0.08), (0.25, 1.35, 0.03), "carbon")
        box("rear_wing", (-p.rear_bumper + 0.15, 0.0, 1.05), (0.3, 1.1, 0.03), "carbon")
        box("sidepod_l", (0.0, 0.42, 0.28), (1.0, 0.28, 0.25), "car")
        box("sidepod_r", (0.0, -0.42, 0.28), (1.0, 0.28, 0.25), "car")
        # main hoop right under the ZED (the camera sits on top of it, as on the car), so it never blocks the view
        cm = self.sc.camera.mount
        top = cm.z - 0.08
        hoop = UsdGeom.Cylinder.Define(st, "/World/Car/body/main_hoop")
        hoop.CreateRadiusAttr(0.03)
        hoop.CreateHeightAttr(top - 0.4)
        hoop.AddTranslateOp().Set(Gf.Vec3d(cm.x, 0.0, 0.4 + (top - 0.4) / 2))
        self._bind(hoop, "carbon")

    # ------------------------------------------------------------------ #
    def update(self, sim, dt: float) -> None:
        """Move the car / wheels and knock cones according to the fsdsim state."""
        s = sim.vehicle.state
        # the body rolls, pitches and heaves on its suspension (fsdsim.chassis): the LiDAR and the cameras are
        # children of /World/Car and move with it, so the rendered sensors see what the car's sensors see.
        # (The wheels tilt with it by under a degree; their height is corrected so the tyres stay on the ground.)
        roll, pitch, heave = sim.vehicle.chassis.state.attitude
        self._car_t.Set(Gf.Vec3d(s.x, s.y, heave))
        self._car_o.Set(_quat(rpy_to_matrix(roll, pitch, s.yaw)))
        for tz, x, y in self._wheel_z:
            tz.Set(Gf.Vec3d(x, y, self.vp.wheel_radius - heave + x * math.sin(pitch) - y * math.sin(roll)))
        omegas = sim.vehicle.wheel_speeds()
        for i, (rz, ry, front) in enumerate(self._wheels):
            self._spin[i] = (self._spin[i] + math.degrees(omegas[i]) * dt) % 360.0
            ry.Set(float(self._spin[i]))
            rz.Set(float(math.degrees(s.steer)) if front else 0.0)
        for c in self.track.cones:
            if c.knocked and c.id not in self._knocked_done:
                t, r = self._cone_ops[c.id]
                t.Set(Gf.Vec3d(float(c.x), float(c.y), CONE_SPECS[c.color].base / 2))
                r.Set(Gf.Vec3f(90.0, 0.0, float(np.degrees(s.yaw))))
                self._knocked_done.add(c.id)
            elif not c.knocked and c.id in self._knocked_done:  # sim reset
                t, r = self._cone_ops[c.id]
                t.Set(Gf.Vec3d(float(c.x), float(c.y), 0.0))
                r.Set(Gf.Vec3f(0, 0, 0))
                self._knocked_done.discard(c.id)

    # ------------------------------------------------------------------ #
    def create_zed_cameras(self) -> dict:
        """Two USD cameras matching the ZED intrinsics; returns their prim paths."""
        cc = self.sc.camera
        st = self.stage
        f = cc.focal_length_mm
        h_ap = 2.0 * f * math.tan(math.radians(cc.hfov_deg) / 2.0)
        v_ap = h_ap * cc.resolution[1] / cc.resolution[0]
        paths = {}
        for side, y in (("left", cc.baseline / 2), ("right", -cc.baseline / 2)):
            path = f"/World/Car/zed_mount/{side}"
            cam = UsdGeom.Camera.Define(st, path)
            cam.ClearXformOpOrder()
            cam.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(0.0, y, 0.0))
            cam.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(_quat(R_CAM_CONV))
            cam.CreateFocalLengthAttr(f)
            cam.CreateHorizontalApertureAttr(h_ap)
            cam.CreateVerticalApertureAttr(v_ap)
            cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 200.0))
            paths[side] = path
        return paths
