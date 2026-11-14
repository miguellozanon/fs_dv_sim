"""Typed configuration loaded from YAML (vehicle.yaml, sensors.yaml, sim.yaml).

Every field has a default so a partial YAML file is fine. Unknown keys raise a
warning instead of being silently ignored, which catches typos early.
"""

from __future__ import annotations

import dataclasses
import os
import warnings
from dataclasses import dataclass, field
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - Isaac Sim python may lack pyyaml
    yaml = None


# --------------------------------------------------------------------------- #
# Vehicle
# --------------------------------------------------------------------------- #
@dataclass
class AeroParams:
    rho: float = 1.225  # air density [kg/m^3]
    cd_a: float = 1.25  # drag coefficient * frontal area [m^2]
    cl_a: float = 3.6  # downforce coefficient * area [m^2]
    balance_front: float = 0.45  # share of downforce on the front axle


@dataclass
class TireParams:
    """Simplified Pacejka magic formula, lateral only, same front/rear."""

    B: float = 11.0
    C: float = 1.45
    E: float = 0.3
    mu: float = 1.6  # peak friction coefficient


@dataclass
class DrivetrainParams:
    drive_split_front: float = 0.5  # 0 = RWD, 0.5 = 4WD even split
    brake_bias_front: float = 0.65
    max_power: float = 80_000.0  # [W] at the wheels (FS limit is 80 kW at the accumulator)


@dataclass
class ActuatorParams:
    max_steer: float = 0.42  # road-wheel angle limit [rad]
    max_steer_rate: float = 1.5  # [rad/s]
    steer_time_constant: float = 0.05  # first-order lag [s]
    accel_time_constant: float = 0.06
    max_accel: float = 14.0  # [m/s^2] request limit
    max_decel: float = 16.0  # [m/s^2] service brake request limit
    max_speed: float = 30.0  # [m/s]


@dataclass
class FootprintParams:
    length: float = 2.95
    width: float = 1.45
    front_overhang: float = 0.60  # beyond front axle [m]
    rear_overhang: float = 0.82  # beyond rear axle [m]


@dataclass
class ChassisParams:
    """Body motion on the suspension, wheel slip and the handling model (fsdsim.chassis).

    The planar car (x, y, yaw) keeps the LapSim force limits. On top of it the sprung mass rolls, pitches and
    heaves (second-order responses to lateral / longitudinal acceleration and downforce), which tilts the LiDAR,
    the camera and the IMU, and the wheels turn with longitudinal slip. Values are typical for an FS car with
    aero and are NOT measured on the LR26: replace them with the vehicle dynamics department's numbers."""

    attitude: bool = True          # roll / pitch / heave of the body (sensors move with it)
    roll_gradient_deg_g: float = 0.7      # [deg per g lateral]
    pitch_gradient_deg_g: float = 0.45    # [deg per g longitudinal] (dive under braking, squat under drive)
    wheel_rate_front: float = 45_000.0    # [N/m] per wheel (ride): heave / pitch from downforce
    wheel_rate_rear: float = 50_000.0
    roll_freq_hz: float = 3.2             # natural frequencies of the sprung mass
    pitch_freq_hz: float = 3.6
    heave_freq_hz: float = 3.4
    damping_ratio: float = 0.65
    wheel_slip: bool = True        # wheel speeds include longitudinal slip (drive / brake force vs tyre potential)
    slip_B: float = 12.0           # longitudinal force = potential * sin(slip_C * atan(slip_B * slip)); peak at ~0.14
    slip_C: float = 1.5
    # "single_track": axle forces as in LapSim (default; the RL policies and LapSim comparisons use it).
    # "double_track": slip angle, load and force per wheel, yaw moment from left / right force differences.
    handling: str = "single_track"


@dataclass
class VehicleParams:
    name: str = "generic_fs_ev_dv"
    mass: float = 255.0  # [kg] incl. DV hardware, no driver
    yaw_inertia: float = 110.0  # [kg m^2]
    wheelbase: float = 1.53
    cog_to_front: float = 0.80  # lf [m]; lr = wheelbase - lf
    cog_height: float = 0.28
    track_front: float = 1.22
    track_rear: float = 1.18
    wheel_radius: float = 0.235
    rolling_resistance: float = 0.015
    low_speed_blend: tuple = (3.0, 5.0)  # [m/s] kinematic -> dynamic model blend window
    usd_path: str = ""  # optional CAD/USD model of the body for Isaac Sim (visual only; relative = to the repository)
    usd_wheel_path: str = ""  # optional USD of one wheel; "{name}" -> fl / fr / rl / rr (centred on the hub)
    usd_fill_body: bool = True  # with a USD body: add a simple monocoque tub (CFD models have none)
    usd_fill_hoop: bool = True  # with a USD body: add the main hoop under the camera (the CAD models have none)
    # "single_track": generic Pacejka single-track model with the values above.
    # "performance": tyre, aero, powertrain and brake limits from the Performance department's
    #   LapSim model (fsdsim.perf) with the LR26 parameter set; mass and geometry follow it.
    model: str = "single_track"
    # Where the LapSim parameters come from (first that works):
    #   performance_source: "lapsim" = the MATLAB folder <repo>/lapsim/LR26_LapSim (or $FSG_LAPSIM_DIR),
    #                       or a path to an LR26_LapSim folder. lr26_params.m and competition_*.m are
    #                       read directly, so MATLAB and the simulator always use the same car.
    #   performance_file:   YAML with LapSim names (fallback, e.g. when the MATLAB folder is absent)
    performance_source: str = ""
    performance_file: str = ""            # YAML with LapSim names, relative to the config dir
    performance: dict = field(default_factory=dict)        # loaded / inline LapSim parameters
    performance_changes: list = field(default_factory=list)  # e.g. ["m-50", "mux*1.02"] applied on top
    efficiency_map: str = ""     # motor + inverter efficiency map (perf.efficiency JSON) for the energy meter; "" = eta_motor
    aero: AeroParams = field(default_factory=AeroParams)
    tire: TireParams = field(default_factory=TireParams)
    drivetrain: DrivetrainParams = field(default_factory=DrivetrainParams)
    actuators: ActuatorParams = field(default_factory=ActuatorParams)
    footprint: FootprintParams = field(default_factory=FootprintParams)
    chassis: ChassisParams = field(default_factory=ChassisParams)

    def perf(self):
        """LapSim parameters (PerfParams) with performance_changes applied, or None."""
        if not self.performance:
            return None
        from .perf import PerfParams  # noqa: PLC0415

        return PerfParams.from_dict(self.performance).with_changes(*self.performance_changes)

    @property
    def lf(self) -> float:
        return self.cog_to_front

    @property
    def lr(self) -> float:
        return self.wheelbase - self.cog_to_front

    @property
    def front_bumper(self) -> float:
        """Distance from CoG to the front of the car [m]."""
        return self.lf + self.footprint.front_overhang

    @property
    def rear_bumper(self) -> float:
        return self.lr + self.footprint.rear_overhang


# --------------------------------------------------------------------------- #
# Sensors
# --------------------------------------------------------------------------- #
@dataclass
class Mount:
    """Sensor pose relative to base_link (CoG projected to the ground)."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0


@dataclass
class IsaacLidarOptions:
    config: str = "OS1"
    variant: str = "auto"  # "auto" picks the closest OS1 variant to channels/rate/resolution
    aux_output_level: str = "FULL"  # needed for ring / per-point time metadata
    yaw_offset_deg: float = 0.0  # fix-up if the vendor USD asset is rotated
    publish_metadata: list = field(default_factory=lambda: ["emitter_id", "timestamp"])
    # the RTX scan read after a render shows the car this many rendered frames earlier (as the camera,
    # camera.isaac_frame_lag); stamps are moved back by it. Check: dashboard map line "LiDAR timing" (11 Nov 2026)
    frame_lag: int = 2


@dataclass
class LidarConfig:
    enabled: bool = True
    model: str = "ouster_os1_64"
    frame_id: str = "os_lidar"
    topic: str = "/lidar/points"
    mount: Mount = field(default_factory=lambda: Mount(x=1.55, z=0.22))
    channels: int = 64
    vertical_fov_deg: tuple = (-22.5, 22.5)
    horizontal_resolution: int = 1024
    rate_hz: float = 10.0
    min_range: float = 0.5
    max_range: float = 120.0
    max_cone_range: float = 60.0  # analytic model: ignore cones further than this
    range_noise_std: float = 0.015
    dropout_prob: float = 0.01
    motion_distortion: bool = True
    isaac: IsaacLidarOptions = field(default_factory=IsaacLidarOptions)


@dataclass
class CameraConfig:
    enabled: bool = True
    model: str = "zed2i"
    namespace: str = "/zed/zed_node"
    frame_prefix: str = "zed"  # -> zed_camera_center, zed_left_camera_optical_frame, ...
    mount: Mount = field(default_factory=lambda: Mount(x=-0.35, z=1.15, pitch=0.12))
    baseline: float = 0.12
    resolution: tuple = (1280, 720)
    rate_hz: float = 15.0
    hfov_deg: float = 110.0
    focal_length_mm: float = 2.12
    min_depth: float = 0.3
    max_depth: float = 20.0
    disparity_noise_px: float = 0.15
    detection_max_range: float = 18.0
    detections_topic: str = "/zed/detections"  # analytic (lite) detector output
    isaac_mode: str = "native"  # native | stereolabs_extension
    # Isaac Sim delivers a camera image (Replicator annotator) this many rendered frames after the frame it shows;
    # images are stamped with the time of the frame they show (measured: 2 frames at 30 Hz, docs/PERCEPTION.md)
    isaac_frame_lag: int = 2


@dataclass
class ImuConfig:
    enabled: bool = True
    model: str = "generic_mems_ins"
    frame_id: str = "imu_link"
    topic: str = "/imu/data"
    mount: Mount = field(default_factory=lambda: Mount(z=0.25))
    rate_hz: float = 200.0
    gyro_noise_density: float = 1.2e-4  # [rad/s/sqrt(Hz)]
    gyro_bias_random_walk: float = 1.0e-5  # [rad/s^2/sqrt(Hz)]
    accel_noise_density: float = 6.0e-4  # [m/s^2/sqrt(Hz)]
    accel_bias_random_walk: float = 1.0e-4  # [m/s^3/sqrt(Hz)]
    provide_orientation: bool = True
    orientation_noise_std: float = 0.005  # [rad]


@dataclass
class GnssOrigin:
    lat: float = 49.3278  # approx. Hockenheimring (placeholder, set your own)
    lon: float = 8.5658
    alt: float = 105.0


@dataclass
class GnssConfig:
    enabled: bool = True
    model: str = "rtk_gnss"
    frame_id: str = "gnss_link"
    topic: str = "/gnss/fix"
    velocity_topic: str = "/gnss/vel"
    mount: Mount = field(default_factory=lambda: Mount(x=-0.20, z=1.25))
    rate_hz: float = 10.0
    horizontal_std: float = 0.02  # RTK fixed; use ~1.5 for single-point
    vertical_std: float = 0.04
    velocity_std: float = 0.03
    origin: GnssOrigin = field(default_factory=GnssOrigin)


@dataclass
class VehicleSensorsConfig:
    enabled: bool = True
    topic: str = "/vehicle/sensors"
    rate_hz: float = 100.0
    wheel_rpm_noise_std: float = 2.0
    steering_noise_std: float = 0.002  # [rad]
    ground_speed_enabled: bool = True
    ground_speed_noise_std: float = 0.02  # [m/s]


@dataclass
class SensorsConfig:
    base_frame: str = "base_link"
    map_frame: str = "map"
    lidar: LidarConfig = field(default_factory=LidarConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    imu: ImuConfig = field(default_factory=ImuConfig)
    gnss: GnssConfig = field(default_factory=GnssConfig)
    vehicle_sensors: VehicleSensorsConfig = field(default_factory=VehicleSensorsConfig)


# --------------------------------------------------------------------------- #
# Simulation, events, AS
# --------------------------------------------------------------------------- #
@dataclass
class AccelerationEvent:
    length: float = 75.0  # start line -> finish line [m]
    width: float = 4.0
    cone_spacing: float = 5.0
    stop_length: float = 100.0  # exit lane after the finish line [m]
    staging_gap: float = 0.3  # front of car behind start line [m]


@dataclass
class SkidpadEvent:
    inner_diameter: float = 15.25
    outer_diameter: float = 21.25
    center_distance: float = 18.25
    entry_length: float = 15.0
    exit_length: float = 25.0
    laps_per_side: int = 2
    inner_cones: int = 16
    outer_cones: int = 13


@dataclass
class TrackGenerator:
    width: float = 3.5
    min_radius: float = 4.5  # centerline radius [m]
    length_range: tuple = (280.0, 450.0)
    max_straight: float = 80.0
    box: tuple = (110.0, 70.0)
    seed: int = 7
    staging_distance: float = 6.0  # front axle behind the start line [m]
    max_cone_spacing: float = 5.0
    min_cone_spacing: float = 2.0


@dataclass
class EventsConfig:
    acceleration: AccelerationEvent = field(default_factory=AccelerationEvent)
    skidpad: SkidpadEvent = field(default_factory=SkidpadEvent)
    track_generator: TrackGenerator = field(default_factory=TrackGenerator)
    autocross_laps: int = 1
    trackdrive_laps: int = 10
    ebs_test_trigger_distance: float = 40.0  # RES pressed this far past the start line [m]


@dataclass
class PenaltyConfig:
    cone_down_s: float = 2.0  # verify against FS Rules (D 9 / D 10)
    off_course_s: float = 10.0


@dataclass
class ASConfig:
    ready_min_time_s: float = 5.0  # AS Ready must be held before Go is accepted
    ebs_decel: float = 10.0  # [m/s^2]
    hold_decel: float = 5.0  # brake applied in AS Off / Ready / Finished [m/s^2]
    standstill_speed: float = 0.1  # [m/s]
    standstill_time_s: float = 1.0
    command_timeout_s: float = 1.0  # control watchdog in AS Driving -> AS Emergency
    leave_track_distance: float = 8.0  # beyond track edge -> AS Emergency (RES)


@dataclass
class SimSettings:
    physics_rate_hz: float = 500.0
    realtime_factor: float = 1.0
    seed: int = 42
    mission: str = "trackdrive"
    track_file: str = ""
    auto_activate: bool = True
    auto_go: bool = True
    go_delay_s: float = 5.5
    auto_go_wait_for_stack: bool = True  # only send Go once /control/command is being received
    publish_ground_truth: bool = True
    publish_gt_tf: bool = True
    analytic_lidar: bool = True
    analytic_camera: bool = True
    candump_log: str = ""
    faults: str = ""     # sensor fault injection, e.g. "lidar:dropout@20+3,gnss:jump=1.5@10+30" (fsdsim.faults)


@dataclass
class SimConfig:
    sim: SimSettings = field(default_factory=SimSettings)
    events: EventsConfig = field(default_factory=EventsConfig)
    penalties: PenaltyConfig = field(default_factory=PenaltyConfig)
    autonomous_system: ASConfig = field(default_factory=ASConfig)


# --------------------------------------------------------------------------- #
# Loader
# --------------------------------------------------------------------------- #
def _merge(obj: Any, data: dict, path: str = "") -> Any:
    """Recursively copy values from a dict into a dataclass instance."""
    if data is None:
        return obj
    names = {f.name: f for f in dataclasses.fields(obj)}
    for key, value in data.items():
        if key not in names:
            warnings.warn(f"config: unknown key '{path}{key}' ignored")
            continue
        current = getattr(obj, key)
        if dataclasses.is_dataclass(current) and isinstance(value, dict):
            _merge(current, value, f"{path}{key}.")
        elif isinstance(current, tuple) and isinstance(value, (list, tuple)):
            setattr(obj, key, tuple(value))
        elif isinstance(current, float) and isinstance(value, int):
            setattr(obj, key, float(value))
        else:
            setattr(obj, key, value)
    return obj


def _read_yaml(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}
    if yaml is None:
        raise RuntimeError("pyyaml is required to read config files (pip install pyyaml)")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_vehicle(path: str = "", overrides=None) -> VehicleParams:
    v = _merge(VehicleParams(), _read_yaml(path))
    return finalize_vehicle(v, os.path.dirname(path) if path else "", overrides)


def finalize_vehicle(v: VehicleParams, config_dir: str = "", overrides=None) -> VehicleParams:
    """Load the performance file, apply overrides and sync mass / geometry from the LapSim set."""
    if v.performance_source and not v.performance:
        v.performance = _performance_from_lapsim(v.performance_source, config_dir)
    if v.performance_file and not v.performance:
        pf = v.performance_file if os.path.isabs(v.performance_file) else os.path.join(config_dir, v.performance_file)
        if not os.path.exists(pf):
            raise FileNotFoundError(f"vehicle.performance_file '{v.performance_file}' not found (looked for {pf})")
        v.performance = _read_yaml(pf)
    if v.model == "performance" and not v.performance:
        raise ValueError("vehicle.model is 'performance' but no performance parameters were given")
    if overrides:
        apply_vehicle_overrides(v, overrides, sync=False)
    sync_vehicle_from_performance(v)
    return v


def _performance_from_lapsim(source: str, config_dir: str) -> dict:
    """LapSim parameters read from the MATLAB files (lr26_params.m + competition_*.m); {} if not found."""
    from .perf.lapsim import LapSim, find_lapsim_dir  # noqa: PLC0415

    if source.lower() in ("lapsim", "auto"):
        root = find_lapsim_dir(config_dir or None)
    else:
        root = source if os.path.isabs(source) else os.path.join(config_dir, source)
        root = root if os.path.isfile(os.path.join(root, "config", "lr26_params.m")) else None
    if not root:
        import warnings  # noqa: PLC0415

        warnings.warn(f"vehicle.performance_source '{source}': LR26_LapSim folder not found "
                      "(set FSG_LAPSIM_DIR); using performance_file instead", stacklevel=3)
        return {}
    ls = LapSim.load(root)
    d = ls.veh.to_dict()
    d["source"] = f"{ls.params_file} + {os.path.basename(ls.competition_file)}"
    return d


def sync_vehicle_from_performance(v: VehicleParams) -> VehicleParams:
    """With model 'performance', mass and geometry come from the LapSim parameters (one source of truth)."""
    if v.model != "performance":
        return v
    p = v.perf()
    v.mass = p.m
    v.wheelbase = p.lf + p.lr
    v.cog_to_front = p.lf
    v.cog_height = p.h_cog
    v.track_front, v.track_rear = p.sf, p.sr
    v.wheel_radius = p.r
    from .perf.model import powertrain  # noqa: PLC0415

    v.actuators.max_speed = max(v.actuators.max_speed, powertrain(p, 0.0)[2] + 0.5)
    return v


def apply_vehicle_overrides(v: VehicleParams, changes, sync: bool = True) -> VehicleParams:
    """Apply "name=value" / "name*factor" / "name+delta" / "name-delta" changes.

    LapSim names and aliases (m, mux, mu, c_z, i_trans, ...) change the performance set;
    other names are VehicleParams fields, dotted for nested ones (yaw_inertia=120, aero.cd_a=1.3)."""
    import re

    from .perf.params import ALIASES, PerfParams, _eval  # noqa: PLC0415

    perf_names = {f.name for f in dataclasses.fields(PerfParams)} | set(ALIASES)
    for raw in changes or []:
        for ch in [c for c in re.split(r"[,\s]+", str(raw)) if c]:
            m = re.fullmatch(r"([A-Za-z_][\w.]*)\s*([=*+-])\s*(.+)", ch)
            if not m:
                raise ValueError(f"cannot parse vehicle change '{ch}' (use name=value, name*factor, name+delta)")
            name, op, val = m.groups()
            if name in perf_names:
                if not v.performance:
                    raise ValueError(f"'{name}' is a performance (LapSim) parameter, but vehicle.yaml has no "
                                     "performance set (model: performance, performance_file: ...)")
                v.performance_changes = list(v.performance_changes) + [ch]
                continue
            obj, parts = v, name.split(".")
            for part in parts[:-1]:
                obj = getattr(obj, part)
            if not hasattr(obj, parts[-1]):
                raise KeyError(f"unknown vehicle parameter '{name}'")
            cur = getattr(obj, parts[-1])
            if isinstance(cur, (str, bool)):            # e.g. chassis.handling=double_track, chassis.attitude=false
                if op != "=":
                    raise ValueError(f"'{name}' is not a number: use {name}=value")
                setattr(obj, parts[-1], val.strip().lower() in ("1", "true", "yes", "on") if isinstance(cur, bool) else val.strip())
                continue
            cur, x = float(cur), _eval(val, {})
            setattr(obj, parts[-1], x if op == "=" else cur * x if op == "*" else cur + x if op == "+" else cur - x)
    if sync:
        sync_vehicle_from_performance(v)
    return v


def load_sensors(path: str = "") -> SensorsConfig:
    return _merge(SensorsConfig(), _read_yaml(path))


def load_sim(path: str = "") -> SimConfig:
    return _merge(SimConfig(), _read_yaml(path))


def load_all_configs(config_dir: str = "", vehicle_overrides=None, vehicle_file: str = ""):
    """Load (vehicle, sensors, sim) from a directory with the three YAML files.

    vehicle_overrides: e.g. ["m=300", "mux*1.05"] (see apply_vehicle_overrides).
    vehicle_file: another vehicle YAML in the same directory (default vehicle.yaml)."""
    def p(name):
        return os.path.join(config_dir, name) if config_dir else ""

    return (load_vehicle(p(vehicle_file or "vehicle.yaml"), vehicle_overrides), load_sensors(p("sensors.yaml")),
            load_sim(p("sim.yaml")))


def to_dict(obj: Any) -> dict:
    return dataclasses.asdict(obj)


def vehicle_from_dict(d: dict) -> VehicleParams:
    v = _merge(VehicleParams(), d)
    return sync_vehicle_from_performance(v) if v.model == "performance" and v.performance else v


def sensors_from_dict(d: dict) -> SensorsConfig:
    return _merge(SensorsConfig(), d)


def sim_from_dict(d: dict) -> SimConfig:
    return _merge(SimConfig(), d)


def config_fingerprint(*objs) -> str:
    """Short hash of configs, used to detect sim / relay mismatches."""
    import hashlib
    import json

    blob = json.dumps([to_dict(o) for o in objs], sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:10]
