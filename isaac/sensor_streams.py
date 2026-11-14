"""Read RTX LiDAR and camera data inside Isaac Sim and turn it into fsdsim events.

Used with the TCP sensor transport (default on Windows): data is read here with
Replicator annotators / the Isaac Sim 6.x RTX sensor API and sent to the WSL
relay over fsdsim.link, which publishes it as ROS 2 messages.

The conversion functions (`gmo_to_cloud`, `pointcloud_dict_to_cloud`) are
pure NumPy so they can be unit-tested without Isaac Sim.
"""

from __future__ import annotations

import math
import queue

import numpy as np

# --------------------------------------------------------------------------- #
# Pure conversion helpers
# --------------------------------------------------------------------------- #
CLOUD_KEYS = ("xyz", "intensity", "reflectivity", "ring", "t", "range")


def _np(a):
    if a is None:
        return None
    if hasattr(a, "numpy"):  # warp array / torch tensor
        a = a.numpy()
    return np.asarray(a)


def _field(obj, *names):
    """First available attribute / key among names, also looking inside obj.elements."""
    for src in (obj, getattr(obj, "elements", None)):
        if src is None:
            continue
        for n in names:
            v = src.get(n) if isinstance(src, dict) else getattr(src, n, None)
            if v is not None:
                return v
    return None


def rings_from_elevation(elev: np.ndarray, emitter: np.ndarray | None, channels: int,
                         fov_deg: tuple = (-22.5, 22.5)) -> np.ndarray:
    """Ouster convention: ring 0 = top beam.

    Rings come from the configured beam geometry (uniform spacing over the
    vertical FOV, like OS1 'uniform' beams), not from the beams that happened
    to return: in an open field the upper beams never hit anything. When
    emitter ids are available, each emitter's mean elevation is used so all
    points of a beam get the same ring.
    """
    n = elev.size
    if n == 0:
        return np.zeros(0, np.uint16)
    el = np.degrees(np.asarray(elev, np.float64))
    if emitter is not None and emitter.size == n:
        ids, inv = np.unique(emitter, return_inverse=True)
        el = (np.bincount(inv, weights=el) / np.maximum(np.bincount(inv), 1))[inv]
    lo, hi = float(min(fov_deg)), float(max(fov_deg))
    step = (hi - lo) / max(channels - 1, 1)
    return np.clip(np.round((hi - el) / step), 0, channels - 1).astype(np.uint16)


def _finish_cloud(xyz, intensity, emitter, t_ns, channels, fov_deg=(-22.5, 22.5)):
    xyz = np.asarray(xyz, np.float32).reshape(-1, 3)
    n = len(xyz)
    rng = np.linalg.norm(xyz, axis=1)
    elev = np.arctan2(xyz[:, 2], np.hypot(xyz[:, 0], xyz[:, 1]))
    if intensity is None or len(intensity) != n:
        intensity = np.full(n, 50.0, np.float32)
    intensity = np.asarray(intensity, np.float32)
    if n and float(np.nanmax(intensity)) <= 1.5:  # normalised [0, 1] -> 0..255 like Ouster reflectivity
        intensity = intensity * 255.0
    if t_ns is None or len(t_ns) != n:
        t_ns = np.zeros(n)
    t_ns = np.asarray(t_ns, np.int64)
    if n:
        t_ns = t_ns - t_ns.min()
    return {
        "xyz": xyz,
        "intensity": intensity.astype(np.float32),
        "reflectivity": np.clip(intensity, 0, 65535).astype(np.uint16),
        "ring": rings_from_elevation(elev, None if emitter is None else np.asarray(emitter), channels, fov_deg),
        "t": np.clip(t_ns, 0, 2**32 - 1).astype(np.uint32),
        "range": (rng * 1000.0).astype(np.uint32),
    }


def gmo_to_cloud(gmo, channels: int = 64, fov_deg: tuple = (-22.5, 22.5)) -> dict | None:
    """GenericModelOutput (parsed) -> fsdsim cloud dict in the sensor frame.

    Handles spherical (azimuth deg, elevation deg, range m) and Cartesian
    element coordinates, VALID flags, intensity scaling and emitter -> ring.
    """
    n = _field(gmo, "numElements", "num_elements")
    x, y, z = (_np(_field(gmo, k)) for k in ("x", "y", "z"))
    if x is None or y is None or z is None:
        return None
    n = int(n) if n is not None else len(x)
    x, y, z = x[:n].astype(np.float64), y[:n].astype(np.float64), z[:n].astype(np.float64)
    ctype = str(_field(gmo, "elementsCoordsType", "elements_coords_type", "coordsType") or "SPHERICAL").upper()
    if "CART" in ctype or ctype.endswith("1"):
        xyz = np.stack([x, y, z], axis=1)
    else:
        az, el, r = np.radians(x), np.radians(y), z
        xyz = np.stack([r * np.cos(el) * np.cos(az), r * np.cos(el) * np.sin(az), r * np.sin(el)], axis=1)
    keep = np.isfinite(xyz).all(axis=1) & (np.linalg.norm(xyz, axis=1) > 1e-3)
    flags = _np(_field(gmo, "flags"))
    if flags is not None and len(flags) >= n:
        valid = (flags[:n].astype(np.uint8) & 1) > 0
        if (keep & valid).any():
            keep &= valid
    scalar = _np(_field(gmo, "scalar", "intensity"))
    emitter = _np(_field(gmo, "emitterId", "emitter_id", "channelId", "channel_id"))
    t_off = _np(_field(gmo, "timeOffsetNs", "time_offset_ns"))
    sl = lambda a: None if a is None or len(a) < n else a[:n][keep]  # noqa: E731
    return _finish_cloud(xyz[keep], sl(scalar), sl(emitter), sl(t_off), channels, fov_deg)


def pointcloud_dict_to_cloud(d: dict, channels: int = 64, fov_deg: tuple = (-22.5, 22.5)) -> dict | None:
    """Replicator point-cloud annotator output (Isaac 4.x / 5.x style) -> cloud dict."""
    if not isinstance(d, dict):
        return None
    xyz = _np(d.get("data", d.get("points", d.get("xyz"))))
    if xyz is None or xyz.size == 0:
        return None
    xyz = xyz.reshape(-1, 3)
    info = d.get("info", {}) if isinstance(d.get("info"), dict) else {}
    get = lambda *k: next((_np(src[x]) for src in (d, info) for x in k if x in src and src[x] is not None), None)  # noqa: E731
    return _finish_cloud(xyz, get("intensity", "intensities"), get("emitterId", "emitterIds", "channelId"),
                         get("timeOffsetNs", "timestamp"), channels, fov_deg)


# --------------------------------------------------------------------------- #
# Isaac Sim readers
# --------------------------------------------------------------------------- #
def _is_empty_buffer(data) -> bool:
    """True for an empty / all-zero GenericModelOutput buffer (before the first full scan)."""
    try:
        head = _np(data)
    except Exception:  # noqa: BLE001 - unknown container: let the parser decide
        return False
    if head is None:
        return True
    head = head.reshape(-1)
    return head.size == 0 or not head[:16].any()


def decimate_rings(cloud: dict, profile_channels: int, target_channels: int) -> dict:
    """Emulate a sensor with fewer beams from a denser profile (e.g. OS1-64 from the OS1-128
    profile: same 45 deg FOV, keep every 2nd beam). `cloud["ring"]` must be in profile rings."""
    if not cloud or profile_channels <= target_channels or profile_channels % target_channels:
        return cloud
    step = profile_channels // target_channels
    keep = (cloud["ring"] % step) == 0
    out = {k: v[keep] for k, v in cloud.items()}
    out["ring"] = (out["ring"] // step).astype(np.uint16)
    return out


class RtxLidarStream:
    """Full scans from an RTX LiDAR prim. Raises RuntimeError if no API works.

    Isaac Sim 6.x: a Replicator Writer receives every GenericModelOutput frame
    (NVIDIA's recommended pattern - polling get_data() each tick drops frames
    under multi-tick rendering). Fallbacks: polling, then 4.5/5.x annotators.
    """

    _writer_registered = False
    _queue: "queue.Queue" = None  # filled by the writer thread
    _active = None  # stream instance the writer reports to

    def __init__(self, prim_path: str, lcfg, log, lidar_obj=None, profile_channels: int = 0):
        self.cfg = lcfg
        self.log = log
        self.mode = None
        self._last_key = None
        self.scans = 0
        self.empty_reads = 0
        self.writer_calls = 0
        self.zero_element_frames = 0
        self.profile_channels = int(profile_channels or lcfg.channels)
        errors = []
        try:  # Isaac Sim 6.x
            from isaacsim.sensors.experimental.rtx import Lidar, LidarSensor, parse_generic_model_output_data

            self._parse = parse_generic_model_output_data
            lidar = lidar_obj if lidar_obj is not None else Lidar(prim_path)
            try:
                self._setup_writer(LidarSensor, lidar, prim_path)
                self.mode = "experimental+writer"
            except Exception as e:  # noqa: BLE001
                errors.append(f"writer: {type(e).__name__}: {e}")
                self.sensor = LidarSensor(lidar, annotators=["generic-model-output"])
                self.mode = "experimental+polling"
        except Exception as e:  # noqa: BLE001
            errors.append(f"experimental API: {type(e).__name__}: {e}")
        if self.mode is None:  # Isaac Sim 4.5 / 5.x replicator annotators
            try:
                import omni.replicator.core as rep

                rp = rep.create.render_product(prim_path, [1, 1], name="os_lidar_tcp")
                for name in ("IsaacExtractRTXSensorPointCloudNoAccumulator", "IsaacExtractRTXSensorPointCloud",
                             "RtxSensorCpuIsaacCreateRTXLidarScanBuffer"):
                    try:
                        ann = rep.AnnotatorRegistry.get_annotator(name)
                        ann.attach([rp])
                        self.annotator, self.mode = ann, f"replicator:{name}"
                        break
                    except Exception as e:  # noqa: BLE001
                        errors.append(f"{name}: {e}")
            except Exception as e:  # noqa: BLE001
                errors.append(f"replicator: {e}")
        if self.mode is None:
            raise RuntimeError("; ".join(errors))
        note = f" (notes: {'; '.join(errors)})" if errors else ""
        log(f"RTX LiDAR readout via {self.mode}{note}")

    # ------------------------------------------------------------------ #
    def _setup_writer(self, LidarSensor, lidar, prim_path):
        import omni.replicator.core as rep

        cls = RtxLidarStream
        cls._queue = queue.Queue(maxsize=4)
        cls._active = self
        if not cls._writer_registered:
            get_ann = getattr(getattr(rep, "annotators", None), "get", None) or rep.AnnotatorRegistry.get_annotator

            class FsdGmoWriter(rep.Writer):
                def __init__(self, **kwargs):
                    self.annotators = [get_ann("GenericModelOutput")]

                def write(self, data):
                    stream = RtxLidarStream._active
                    if stream is not None:
                        stream._on_gmo(data)

            rep.WriterRegistry.register(FsdGmoWriter)
            cls._writer_registered = True
        self.sensor = LidarSensor(lidar, annotators=[])
        if hasattr(self.sensor, "attach_writer"):
            self.sensor.attach_writer("FsdGmoWriter")
        else:  # attach the writer to our own render product
            writer = rep.WriterRegistry.get("FsdGmoWriter")
            writer.initialize()
            writer.attach([rep.create.render_product(prim_path, [1, 1], name="os_lidar_gmo")])

    def _on_gmo(self, data) -> None:
        """Writer callback (Replicator thread): parse now, while the buffer is valid."""
        self.writer_calls += 1
        try:
            buf = None
            items = data.items() if isinstance(data, dict) else []
            for k, v in items:
                if "GenericModelOutput" in str(k):
                    buf = v.get("data", v) if isinstance(v, dict) else v
                    break
            if buf is None or _is_empty_buffer(buf):
                self.empty_reads += 1
                return
            gmo = self._parse(buf)
            n = _field(gmo, "numElements", "num_elements")
            if n is not None and int(n) == 0:
                self.zero_element_frames += 1
                return
            cloud = self._to_cloud(gmo)
            if cloud is None or len(cloud["xyz"]) == 0:
                self.empty_reads += 1
                return
            q = RtxLidarStream._queue
            if q.full():
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
            q.put_nowait(cloud)
        except Exception as e:  # noqa: BLE001
            self.empty_reads += 1
            if self.empty_reads < 3:
                self.log(f"LiDAR writer parse failed: {type(e).__name__}: {e}")

    def _to_cloud(self, gmo):
        cloud = gmo_to_cloud(gmo, self.profile_channels, tuple(self.cfg.vertical_fov_deg))
        return decimate_rings(cloud, self.profile_channels, int(self.cfg.channels))

    def diagnostics(self) -> str:
        return (f"mode {self.mode}, writer calls {self.writer_calls}, empty reads {self.empty_reads}, "
                f"zero-element frames {self.zero_element_frames}")

    # ------------------------------------------------------------------ #
    def poll(self) -> dict | None:
        """Return the newest full scan (cloud dict) or None if nothing new."""
        try:
            if self.mode == "experimental+writer":
                latest = None
                while True:
                    try:
                        latest = RtxLidarStream._queue.get_nowait()
                    except queue.Empty:
                        break
                if latest is not None:
                    self.scans += 1
                return latest
            if self.mode == "experimental+polling":
                data, _ = self.sensor.get_data("generic-model-output")
                if data is None or _is_empty_buffer(data):
                    self.empty_reads += 1  # the RTX LiDAR has not produced a full scan yet
                    return None
                gmo = self._parse(data)
                key = _field(gmo, "frameId", "timestampNs")
                key = int(key) if key is not None else None
                if key is not None and key == self._last_key:
                    return None
                self._last_key = key
                cloud = self._to_cloud(gmo)
                if cloud is None or len(cloud["xyz"]) == 0:
                    self.empty_reads += 1
                    return None
                self.scans += 1
                return cloud
            data = self.annotator.get_data()
            cloud = pointcloud_dict_to_cloud(data, self.profile_channels, tuple(self.cfg.vertical_fov_deg))
            cloud = decimate_rings(cloud, self.profile_channels, int(self.cfg.channels))
            if cloud is None:
                return None
            key = (len(cloud["xyz"]), float(cloud["xyz"][:8].sum()) if len(cloud["xyz"]) else 0.0)
            if key == self._last_key:
                return None
            self._last_key = key
            self.scans += 1
            return cloud
        except Exception as e:  # noqa: BLE001
            self.log(f"LiDAR read failed: {type(e).__name__}: {e}")
            return None


class CameraStream:
    """RGB (+ optional depth) from a USD camera via Replicator annotators."""

    def __init__(self, cam_path: str, resolution: tuple[int, int], with_depth: bool, log):
        import omni.replicator.core as rep

        self.w, self.h = int(resolution[0]), int(resolution[1])
        self.rp = rep.create.render_product(cam_path, [self.w, self.h])
        self.rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        self.rgb.attach([self.rp])
        self.depth = None
        if with_depth:
            self.depth = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
            self.depth.attach([self.rp])
        log(f"camera {cam_path}: {self.w}x{self.h} rgb{' + depth' if with_depth else ''}")

    def read(self):
        rgb = _np(self.rgb.get_data())
        if rgb is None or rgb.size == 0:
            return None, None
        if rgb.ndim == 3 and rgb.shape[2] == 4:
            rgb = rgb[:, :, :3]
        depth = None
        if self.depth is not None:
            depth = _np(self.depth.get_data())
            if depth is not None and depth.size:
                depth = depth.astype(np.float32).reshape(self.h, self.w)
                depth[~np.isfinite(depth)] = np.nan
        return np.ascontiguousarray(rgb, dtype=np.uint8), depth


def zed_intrinsics(ccfg) -> tuple[float, float, float, float]:
    w, h = ccfg.resolution
    fx = (w / 2.0) / math.tan(math.radians(ccfg.hfov_deg) / 2.0)
    return fx, fx, w / 2.0, h / 2.0
