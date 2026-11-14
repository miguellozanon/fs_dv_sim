"""Thin compatibility layer over Isaac Sim APIs.

Isaac Sim renamed most modules between 4.x (omni.isaac.*), 5.x (isaacsim.*)
and 6.x (isaacsim.*.experimental, ROS nodes moved to isaacsim.ros2.nodes).
Everything version-specific lives here so the rest of the backend stays
stable. Primary target: Isaac Sim 6.x; fallbacks for 5.x / 4.5.

Every function logs which path it took. If something fails on your version,
this is the only file you should need to touch.
"""

from __future__ import annotations

import math

import numpy as np

LOG_PREFIX = "[fsd-isaac]"


def log(msg: str) -> None:
    print(f"{LOG_PREFIX} {msg}", flush=True)


# --------------------------------------------------------------------------- #
# App / extensions / stage
# --------------------------------------------------------------------------- #
def create_app(headless: bool, width: int = 1280, height: int = 720, motion_bvh: bool = False):
    """Motion BVH must be chosen at startup (NVIDIA: SimulationApp({"enable_motion_bvh": True}));
    toggling /renderer/raytracingMotion/enabled at runtime breaks the Hydra engines."""
    try:
        from isaacsim import SimulationApp  # 4.5+
    except ImportError:  # pragma: no cover - very old releases
        from omni.isaac.kit import SimulationApp
    cfg = {"headless": headless, "width": width, "height": height}
    if motion_bvh:
        cfg["enable_motion_bvh"] = True
    return SimulationApp(cfg)


def kit_log_path() -> str:
    try:
        import carb

        return str(carb.settings.get_settings().get("/log/file") or "")
    except Exception:  # noqa: BLE001
        return ""


def profile_channels(variant: str, default: int) -> int:
    """'OS1_REV7_128ch10hz1024res' -> 128."""
    import re

    m = re.search(r"_(\d+)ch", variant or "")
    return int(m.group(1)) if m else default


def isaac_version() -> str:
    for mod in ("isaacsim.core.version", "omni.isaac.version"):
        try:
            v = __import__(mod, fromlist=["get_version"]).get_version()
            first = str(v[0])
            return first if "." in first else ".".join(str(x) for x in v[:3])
        except Exception:
            continue
    return "unknown"


def enable_extensions(names: list[str]) -> list[str]:
    """Enable the first available extension of each '|' separated alternative."""
    import omni.kit.app

    mgr = omni.kit.app.get_app().get_extension_manager()
    enabled = []
    for spec in names:
        for name in spec.split("|"):
            try:
                if mgr.set_extension_enabled_immediate(name, True):
                    enabled.append(name)
                    break
            except Exception:
                continue
        else:
            log(f"WARNING: none of {spec} could be enabled")
    log(f"extensions enabled: {enabled}")
    return enabled


def get_stage():
    import omni.usd

    return omni.usd.get_context().get_stage()


# --------------------------------------------------------------------------- #
# Fixed-step stepping
# --------------------------------------------------------------------------- #
class Stepper:
    """Fixed-dt stepping with rendering every step."""

    def __init__(self, simulation_app, dt: float):
        self.app = simulation_app
        self.dt = dt
        self.world = None
        for mod in ("isaacsim.core.api", "omni.isaac.core"):
            try:
                World = __import__(mod, fromlist=["World"]).World
                self.world = World(stage_units_in_meters=1.0, physics_dt=dt, rendering_dt=dt)
                log(f"stepping with {mod}.World (dt={dt:.4f}s)")
                break
            except Exception as e:  # noqa: PERF203
                log(f"{mod}.World unavailable ({type(e).__name__}: {e})")
        if self.world is None:
            import carb
            import omni.timeline

            s = carb.settings.get_settings()
            try:
                s.set("/app/player/useFixedTimeStepping", True)
                s.set("/app/player/fixedTimeStep", dt)
            except Exception as e:
                log(f"could not enable fixed time stepping: {e}")
            self.timeline = omni.timeline.get_timeline_interface()
            log("stepping with omni.timeline + app.update() (fallback)")

    def start(self) -> None:
        if self.world is not None:
            self.world.reset()
            self.world.play()
        else:
            self.timeline.play()

    def ensure_playing(self) -> None:
        try:
            import omni.timeline

            tl = omni.timeline.get_timeline_interface()
            if not tl.is_playing():
                tl.play()
        except Exception as e:  # noqa: BLE001
            log(f"could not restart the timeline: {e}")

    def step(self) -> None:
        if self.world is not None:
            self.world.step(render=True)
        else:
            self.app.update()

    def time(self) -> float:
        if self.world is not None:
            return float(self.world.current_time)
        return float(self.timeline.get_current_time())


# --------------------------------------------------------------------------- #
# RTX LiDAR (Ouster OS1)
# --------------------------------------------------------------------------- #
def list_lidar_variants(keyword: str = "OS1") -> list[str]:
    try:
        from isaacsim.sensors.experimental.rtx import SUPPORTED_LIDAR_CONFIGS
    except Exception:
        return []
    out = []
    for path, variants in SUPPORTED_LIDAR_CONFIGS.items():
        if keyword.lower() in str(path).lower():
            out += [str(v) for v in (variants or [])]
    return sorted(set(out))


def resolve_lidar_variant(lcfg) -> str:
    """Pick the Isaac OS1 variant that best matches channels / rate / resolution."""
    if lcfg.isaac.variant and lcfg.isaac.variant != "auto":
        return lcfg.isaac.variant
    rate, res, ch = int(round(lcfg.rate_hz)), int(lcfg.horizontal_resolution), int(lcfg.channels)
    available = list_lidar_variants(lcfg.isaac.config)
    if available:
        log(f"available {lcfg.isaac.config} variants: {available}")
    for c in (ch, 128, 32):
        for rev in ("REV7", "REV6"):
            name = f"OS1_{rev}_{c}ch{rate}hz{res}res"
            if not available or name in available:
                if c != ch:
                    log(f"WARNING: no {ch}-channel OS1 profile shipped; using {name}. Point density differs from your "
                        f"sensor - author a custom profile for an exact match (see docs/ISAAC.md).")
                return name
    return available[0] if available else f"OS1_REV6_{ch}ch{rate}hz{res}res"


def find_omnilidar(root_path: str) -> str:
    """Path of the OmniLidar prim at or below root_path (vendor assets nest it)."""
    from pxr import Usd

    stage = get_stage()
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return root_path
    for prim in Usd.PrimRange(root):
        if prim.GetTypeName() == "OmniLidar" or "OmniSensorGenericLidarCoreAPI" in [str(a) for a in prim.GetAppliedSchemas()]:
            return prim.GetPath().pathString
    return root_path


def create_rtx_lidar(parent_path: str, lcfg):
    """Create the RTX LiDAR under parent_path.

    Returns (omnilidar_prim_path, lidar_object_or_None, variant). The object is the
    Isaac Sim 6.x `isaacsim.sensors.experimental.rtx.Lidar` when that API is used.
    """
    variant = resolve_lidar_variant(lcfg)
    path = f"{parent_path}/os_lidar"
    # Isaac Sim 6.x experimental API
    try:
        from isaacsim.sensors.experimental.rtx import Lidar

        lidar = Lidar.create(
            path=path,
            config=lcfg.isaac.config,
            variant=variant,
            aux_output_level=lcfg.isaac.aux_output_level,
            accumulate_outputs=True,         # full scans (NVIDIA: required for complete output)
            tick_rate=float(lcfg.rate_hz),   # must equal the profile's scanRateBaseHz
            translations=[[0.0, 0.0, 0.0]],
        )
        prim_path = find_omnilidar(path)
        log(f"RTX LiDAR via isaacsim.sensors.experimental.rtx: {prim_path} ({variant})")
        return prim_path, lidar, variant
    except Exception as e:
        log(f"experimental Lidar API failed ({type(e).__name__}: {e}); trying command API")
    # Isaac Sim 4.5 / 5.x command API
    import omni.kit.commands
    from pxr import Gf

    for kwargs in (
        dict(config=lcfg.isaac.config, variant=variant),
        dict(config=variant),
        dict(config=f"Ouster_{variant}"),
    ):
        try:
            _, prim = omni.kit.commands.execute(
                "IsaacSensorCreateRtxLidar",
                path="/os_lidar",
                parent=parent_path,
                translation=Gf.Vec3d(0, 0, 0),
                orientation=Gf.Quatd(1, 0, 0, 0),
                **kwargs,
            )
            if prim:
                prim_path = find_omnilidar(prim.GetPath().pathString)
                log(f"RTX LiDAR via IsaacSensorCreateRtxLidar{kwargs}: {prim_path}")
                return prim_path, None, variant
        except Exception as e:  # noqa: PERF203
            log(f"IsaacSensorCreateRtxLidar{kwargs} failed: {e}")
    raise RuntimeError("could not create an RTX LiDAR - see docs/ISAAC.md")


def create_render_product(prim_path: str, resolution: tuple[int, int], name: str) -> str:
    import omni.replicator.core as rep

    rp = rep.create.render_product(prim_path, list(resolution), name=name)
    path = rp.path if hasattr(rp, "path") else str(rp)
    log(f"render product {name}: {path}")
    return path


# --------------------------------------------------------------------------- #
# OmniGraph: ROS 2 publishers for clock, LiDAR and cameras
# --------------------------------------------------------------------------- #
NODE_PREFIXES = [
    {  # Isaac Sim 5.x / 6.x
        "ros2": "isaacsim.ros2.bridge",
        "core": "isaacsim.core.nodes",
    },
    {  # Isaac Sim 4.x
        "ros2": "omni.isaac.ros2_bridge",
        "core": "omni.isaac.core_nodes",
    },
]


def build_ros_graph(lidar_rp: str | None, lcfg, cameras: list[dict], graph_path: str = "/World/ROS2Graph") -> None:
    """cameras: [{"rp": render_product, "type": "rgb"|"depth"|"info", "topic": ..., "frame": ..., "ns": ...}]"""
    import omni.graph.core as og

    last_err = None
    for pref in NODE_PREFIXES:
        R, C = pref["ros2"], pref["core"]
        nodes = [
            ("tick", "omni.graph.action.OnPlaybackTick"),
            ("simtime", f"{C}.IsaacReadSimulationTime"),
            ("clock", f"{R}.ROS2PublishClock"),
        ]
        connects = [
            ("tick.outputs:tick", "clock.inputs:execIn"),
            ("simtime.outputs:simulationTime", "clock.inputs:timeStamp"),
        ]
        values = [("clock.inputs:topicName", "clock")]
        if lidar_rp:
            nodes.append(("lidar", f"{R}.ROS2RtxLidarHelper"))
            connects.append(("tick.outputs:tick", "lidar.inputs:execIn"))
            values += [
                ("lidar.inputs:renderProductPath", lidar_rp),
                ("lidar.inputs:topicName", lcfg.topic.lstrip("/")),
                ("lidar.inputs:frameId", lcfg.frame_id),
                ("lidar.inputs:type", "point_cloud"),
            ]
        for i, cam in enumerate(cameras):
            n = f"cam{i}"
            kind = "ROS2CameraInfoHelper" if cam["type"] == "info" else "ROS2CameraHelper"
            nodes.append((n, f"{R}.{kind}"))
            connects.append(("tick.outputs:tick", f"{n}.inputs:execIn"))
            values += [
                (f"{n}.inputs:renderProductPath", cam["rp"]),
                (f"{n}.inputs:topicName", cam["topic"]),
                (f"{n}.inputs:frameId", cam["frame"]),
                (f"{n}.inputs:nodeNamespace", cam["ns"]),
            ]
            if cam["type"] != "info":
                values.append((f"{n}.inputs:type", cam["type"]))
        try:
            keys = og.Controller.Keys
            og.Controller.edit(
                {"graph_path": graph_path, "evaluator_name": "execution"},
                {keys.CREATE_NODES: nodes, keys.CONNECT: connects, keys.SET_VALUES: values},
            )
            log(f"ROS 2 OmniGraph built with node prefix '{R}' ({len(nodes)} nodes)")
            if lidar_rp and lcfg.isaac.publish_metadata:
                _try_set_lidar_metadata(og, f"{graph_path}/lidar", lcfg.isaac.publish_metadata)
            return
        except Exception as e:  # noqa: PERF203
            last_err = e
            log(f"OmniGraph with prefix '{R}' failed: {e}")
            try:
                import omni.usd

                omni.usd.get_context().get_stage().RemovePrim(graph_path)
            except Exception:
                pass
    raise RuntimeError(f"could not build the ROS 2 OmniGraph: {last_err}")


def _try_set_lidar_metadata(og, node_path: str, metadata: list[str]) -> None:
    """Ask the LiDAR helper for ring / per-point time fields (Isaac 5.1+)."""
    try:
        attr = og.Controller.attribute(f"{node_path}.inputs:selectedMetadata")
        og.Controller.set(attr, list(metadata))
        log(f"LiDAR metadata fields requested: {metadata}")
    except Exception as e:
        log(f"LiDAR metadata selection not supported here ({e}); cloud will have x,y,z,intensity only")


# --------------------------------------------------------------------------- #
# Math helpers for USD
# --------------------------------------------------------------------------- #
def quat_wxyz_from_matrix(R: np.ndarray) -> tuple[float, float, float, float]:
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        return (0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s)
    if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        return ((R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s)
    if R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        return ((R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s)
    s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
    return ((R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s)
