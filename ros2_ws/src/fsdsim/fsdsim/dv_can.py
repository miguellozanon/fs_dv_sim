"""Encoder for the mandatory FS Driverless data-logger CAN frames.

Message layout from the FS Driverless Specification (DV driving dynamics 1/2,
DV system status). All signals little-endian, sent every 100 ms. Double-check
scaling against the specification version used at your event before relying on
it in the car - the simulator uses it so logs from sim and car look identical.

    0x500  DV driving dynamics 1 (8 bytes)
           b0 speed actual [km/h, u8]       b1 speed target [km/h, u8]
           b2 steering actual [0.5 deg, s8] b3 steering target [0.5 deg, s8]
           b4 brake hydr. actual [%, u8]    b5 brake hydr. target [%, u8]
           b6 motor moment actual [%, s8]   b7 motor moment target [%, s8]
    0x501  DV driving dynamics 2 (6 bytes)
           acc long [s16, 1/512 m/s^2], acc lat [s16, 1/512 m/s^2],
           yaw rate [s16, 1/128 deg/s]
    0x502  DV system status (5 bytes, bit-packed)
           AS state 0-2, EBS state 3-4, AMI state 5-7, steering state 8,
           service brake / redundancy 9-10, lap counter 11-14,
           cones count actual 15-22, cones count all 23-39
"""

from __future__ import annotations

import math
import struct


def _u8(v):
    return int(max(0, min(255, round(v))))


def _s8(v):
    return int(max(-128, min(127, round(v))))


def _s16(v):
    return int(max(-32768, min(32767, round(v))))


def encode_dv_driving_dynamics_1(
    speed_actual_kmh: float,
    speed_target_kmh: float,
    steering_actual_deg: float,
    steering_target_deg: float,
    brake_actual_pct: float,
    brake_target_pct: float,
    motor_moment_actual_pct: float,
    motor_moment_target_pct: float,
) -> bytes:
    return struct.pack(
        "<BBbbBBbb",
        _u8(speed_actual_kmh),
        _u8(speed_target_kmh),
        _s8(steering_actual_deg / 0.5),
        _s8(steering_target_deg / 0.5),
        _u8(brake_actual_pct),
        _u8(brake_target_pct),
        _s8(motor_moment_actual_pct),
        _s8(motor_moment_target_pct),
    )


def encode_dv_driving_dynamics_2(acc_long: float, acc_lat: float, yaw_rate_deg_s: float) -> bytes:
    return struct.pack("<hhh", _s16(acc_long * 512), _s16(acc_lat * 512), _s16(yaw_rate_deg_s * 128))


def encode_dv_system_status(
    as_state: int,
    ebs_state: int,
    ami_state: int,
    steering_state: int,
    service_brake_state: int,
    lap_counter: int,
    cones_count_actual: int,
    cones_count_all: int,
) -> bytes:
    v = 0
    v |= (as_state & 0x7) << 0
    v |= (ebs_state & 0x3) << 3
    v |= (ami_state & 0x7) << 5
    v |= (steering_state & 0x1) << 8
    v |= (service_brake_state & 0x3) << 9
    v |= (min(lap_counter, 15) & 0xF) << 11
    v |= (min(cones_count_actual, 255) & 0xFF) << 15
    v |= (min(cones_count_all, (1 << 17) - 1) & 0x1FFFF) << 23
    return v.to_bytes(5, "little")


def decode_dv_system_status(data: bytes) -> dict:
    v = int.from_bytes(data[:5], "little")
    return {
        "as_state": v & 0x7,
        "ebs_state": (v >> 3) & 0x3,
        "ami_state": (v >> 5) & 0x7,
        "steering_state": (v >> 8) & 0x1,
        "service_brake_state": (v >> 9) & 0x3,
        "lap_counter": (v >> 11) & 0xF,
        "cones_count_actual": (v >> 15) & 0xFF,
        "cones_count_all": (v >> 23) & 0x1FFFF,
    }


class CandumpWriter:
    """Writes frames in `candump -L` format: (time) iface ID#HEXDATA."""

    def __init__(self, path: str, iface: str = "can0"):
        self.f = open(path, "w", encoding="ascii")
        self.iface = iface

    def write(self, stamp: float, can_id: int, data: bytes) -> None:
        self.f.write(f"({stamp:017.6f}) {self.iface} {can_id:03X}#{data.hex().upper()}\n")

    def close(self) -> None:
        self.f.close()


class DvLogger:
    """Produces the three DV frames from simulator state every 100 ms."""

    def __init__(self, writer: CandumpWriter, max_steer_rad: float, max_accel: float, max_decel: float):
        self.w = writer
        self.max_steer = max_steer_rad
        self.max_accel = max_accel
        self.max_decel = max_decel
        self.next_t = 0.0
        self.cones_all = 0

    def maybe_log(self, t, gt: dict, status: dict, cmd: tuple[float, float], cones_seen: int = 0, v_target: float = 0.0):
        if t < self.next_t:
            return
        self.next_t = t + 0.1
        vx = gt["twist"][0]
        steer_deg = math.degrees(gt["steer"])
        steer_cmd_deg = math.degrees(cmd[0])
        accel = cmd[1]
        moment = 100.0 * max(accel, 0.0) / self.max_accel
        brake = 100.0 * max(-accel, 0.0) / self.max_decel
        self.w.write(t, 0x500, encode_dv_driving_dynamics_1(vx * 3.6, v_target * 3.6, steer_deg, steer_cmd_deg, brake, brake, moment, moment))
        ax, ay = gt["accel"]
        self.w.write(t, 0x501, encode_dv_driving_dynamics_2(ax, ay, math.degrees(gt["twist"][2])))
        self.cones_all += cones_seen
        ebs = 3 if status["state"] == 4 else 2
        self.w.write(
            t,
            0x502,
            encode_dv_system_status(status["state"], ebs, status["mission"], 1, 3, status["lap_count"], cones_seen, self.cones_all),
        )
