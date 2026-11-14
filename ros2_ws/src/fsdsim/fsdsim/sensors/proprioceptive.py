"""IMU, GNSS and vehicle (wheel speed / steering / ground speed) sensor models.

These are computed from the vehicle model in both backends (also in Isaac Sim,
where the chassis is driven kinematically by fsdsim and a PhysX IMU would only
see the teleported pose).
"""

from __future__ import annotations

import math

import numpy as np

from ..config import GnssConfig, ImuConfig, VehicleSensorsConfig
from ..dynamics import G, VehicleModel
from ..geometry import quat_from_rpy, rpy_to_matrix

WGS84_A = 6378137.0
WGS84_E2 = 6.69437999014e-3


class ImuModel:
    def __init__(self, cfg: ImuConfig, rng: np.random.Generator | None = None):
        self.cfg = cfg
        self.rng = rng or np.random.default_rng(2)
        self.dt = 1.0 / cfg.rate_hz
        self.gyro_bias = np.zeros(3)
        self.accel_bias = np.zeros(3)
        m = cfg.mount
        self.R = rpy_to_matrix(m.roll, m.pitch, m.yaw)  # imu -> base_link
        self.lever = np.array([m.x, m.y, m.z])

    def measure(self, vm: VehicleModel) -> dict:
        cfg, s = self.cfg, vm.state
        # Specific force at the IMU in the level vehicle frame (flat ground), then into the rolled / pitched body
        ch = vm.chassis.state
        # rotation only about z: alpha x l + omega x (omega x l), written out (np.cross was 10 % of a lite step)
        lx, ly = self.lever[0], self.lever[1]
        r, rd = s.r, vm.r_dot
        a_p = np.array([vm.ax_body - rd * ly - r * r * lx, vm.ay_body + rd * lx - r * r * ly, 0.0])
        omega = np.array([0.0, 0.0, r])
        R_att = rpy_to_matrix(ch.roll, ch.pitch, 0.0)          # body -> level frame
        f = R_att.T @ (a_p + np.array([0.0, 0.0, G]))
        omega = R_att.T @ omega + np.array([ch.roll_rate, ch.pitch_rate, 0.0])
        # Bias random walk + white noise
        sq = math.sqrt(self.dt)
        self.gyro_bias += self.rng.normal(0.0, cfg.gyro_bias_random_walk * sq, 3)
        self.accel_bias += self.rng.normal(0.0, cfg.accel_bias_random_walk * sq, 3)
        gyro = self.R.T @ omega + self.gyro_bias + self.rng.normal(0.0, cfg.gyro_noise_density / sq, 3)
        acc = self.R.T @ f + self.accel_bias + self.rng.normal(0.0, cfg.accel_noise_density / sq, 3)
        out = {
            "angular_velocity": [float(v) for v in gyro],
            "linear_acceleration": [float(v) for v in acc],
            "gyro_var": (cfg.gyro_noise_density / sq) ** 2,
            "accel_var": (cfg.accel_noise_density / sq) ** 2,
        }
        if cfg.provide_orientation:
            n = cfg.orientation_noise_std
            q = quat_from_rpy(ch.roll + self.rng.normal(0, n), ch.pitch + self.rng.normal(0, n), s.yaw + self.rng.normal(0, n))
            out["orientation"] = list(q)
            out["orientation_var"] = n * n
        return out


class GnssModel:
    def __init__(self, cfg: GnssConfig, rng: np.random.Generator | None = None):
        self.cfg = cfg
        self.rng = rng or np.random.default_rng(3)
        lat0 = math.radians(cfg.origin.lat)
        sin2 = math.sin(lat0) ** 2
        self.r_n = WGS84_A * (1 - WGS84_E2) / (1 - WGS84_E2 * sin2) ** 1.5  # meridian radius
        self.r_e = WGS84_A / math.sqrt(1 - WGS84_E2 * sin2)  # prime vertical radius
        self.cos_lat0 = math.cos(lat0)

    def enu_to_lla(self, e: float, n: float, u: float) -> tuple[float, float, float]:
        o = self.cfg.origin
        lat = o.lat + math.degrees(n / self.r_n)
        lon = o.lon + math.degrees(e / (self.r_e * self.cos_lat0))
        return lat, lon, o.alt + u

    def measure(self, vm: VehicleModel) -> dict:
        cfg, s = self.cfg, vm.state
        m = cfg.mount
        c, sn = math.cos(s.yaw), math.sin(s.yaw)
        e = s.x + c * m.x - sn * m.y + self.rng.normal(0, cfg.horizontal_std)
        n = s.y + sn * m.x + c * m.y + self.rng.normal(0, cfg.horizontal_std)
        u = m.z + self.rng.normal(0, cfg.vertical_std)
        lat, lon, alt = self.enu_to_lla(e, n, u)
        # antenna velocity in ENU
        ve = s.vx * c - s.vy * sn - s.r * (sn * m.x + c * m.y)
        vn = s.vx * sn + s.vy * c + s.r * (c * m.x - sn * m.y)
        vel = [ve + self.rng.normal(0, cfg.velocity_std), vn + self.rng.normal(0, cfg.velocity_std), 0.0]
        return {
            "lat": lat,
            "lon": lon,
            "alt": alt,
            "pos_var": [cfg.horizontal_std**2, cfg.horizontal_std**2, cfg.vertical_std**2],
            "vel_enu": vel,
            "vel_var": cfg.velocity_std**2,
        }


class VehicleSensorModel:
    def __init__(self, cfg: VehicleSensorsConfig, rng: np.random.Generator | None = None):
        self.cfg = cfg
        self.rng = rng or np.random.default_rng(4)

    def measure(self, vm: VehicleModel) -> dict:
        cfg, s = self.cfg, vm.state
        rpm = [w * 60.0 / (2 * math.pi) + self.rng.normal(0, cfg.wheel_rpm_noise_std) for w in vm.wheel_speeds()]
        if s.vx < 0.05:
            rpm = [0.0 if abs(v) < 3 * cfg.wheel_rpm_noise_std else v for v in rpm]
        out = {
            "wheel_rpm": [float(v) for v in rpm],
            "steering_angle": float(s.steer + self.rng.normal(0, cfg.steering_noise_std)),
        }
        if cfg.ground_speed_enabled:
            out["ground_speed"] = [
                float(s.vx + self.rng.normal(0, cfg.ground_speed_noise_std)),
                float(s.vy + self.rng.normal(0, cfg.ground_speed_noise_std)),
            ]
        return out
