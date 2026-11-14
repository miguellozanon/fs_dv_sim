"""Sensor fault injection: what does the stack do when a sensor fails?

Specification (sim.yaml `sim.faults`, launch argument `faults:=...`, tools' --faults): comma-separated items

    <sensor>:<kind>[=<value>]@<start>[+<duration>]        times in seconds of simulation time; no duration = for good

    sensor   lidar | camera | imu | gnss | wheels          (camera: images and camera detections; wheels: /vehicle/sensors)
    kind     dropout            no messages
             freeze             the last message is repeated with new stamps (a hung driver)
             latency=<s>        messages arrive <s> seconds late (stamps unchanged)
             thin=<share>       lidar: only this share of the points survives (dirt, rain), e.g. thin=0.3
             bias=<rad/s>       imu: added to the yaw rate
             scale=<factor>     wheels: wheel speeds x factor (wrong tyre radius, slip)
             jump=<m>           gnss: position offset to the east (multipath, lost RTK fix)

    lidar:dropout@20+3,gnss:jump=1.5@10+30,imu:bias=0.02@0,camera:latency=0.25@5

Faults act on the events the simulator hands out, so they work the same in the lite backend, in the tools and
(for LiDAR and camera) on the rendered sensors of Isaac Sim. The ground truth is never touched.
"""

from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass

import numpy as np

SENSOR_EVENTS = {"lidar": ("lidar",), "camera": ("image", "camera_detections"), "imu": ("imu",), "gnss": ("gnss",),
                 "wheels": ("vehicle_sensors",)}
KINDS = {"dropout": None, "freeze": None, "latency": "any", "thin": "lidar", "bias": "imu", "scale": "wheels", "jump": "gnss"}
_ITEM = re.compile(r"^\s*(\w+)\s*:\s*(\w+)\s*(?:=\s*([-+0-9.eE]+))?\s*@\s*([0-9.]+)\s*(?:\+\s*([0-9.]+))?\s*$")


@dataclass
class Fault:
    sensor: str
    kind: str
    value: float
    start: float
    end: float

    def active(self, t: float) -> bool:
        return self.start <= t < self.end


def parse(spec: str) -> list[Fault]:
    out = []
    for item in [i for i in re.split(r"[,;]", spec or "") if i.strip()]:
        m = _ITEM.match(item)
        if not m:
            raise ValueError(f"fault '{item.strip()}': expected <sensor>:<kind>[=<value>]@<start>[+<duration>]")
        sensor, kind, val, start, dur = m.groups()
        if sensor not in SENSOR_EVENTS:
            raise ValueError(f"fault '{item.strip()}': unknown sensor '{sensor}' (one of {', '.join(SENSOR_EVENTS)})")
        if kind not in KINDS:
            raise ValueError(f"fault '{item.strip()}': unknown kind '{kind}' (one of {', '.join(KINDS)})")
        if KINDS[kind] not in (None, "any", sensor):
            raise ValueError(f"fault '{item.strip()}': '{kind}' is for the {KINDS[kind]} sensor")
        if KINDS[kind] is not None and val is None:
            raise ValueError(f"fault '{item.strip()}': '{kind}' needs a value ({kind}=...)")
        start = float(start)
        out.append(Fault(sensor, kind, float(val) if val is not None else 0.0, start,
                         start + float(dur) if dur is not None else math.inf))
    return out


class FaultInjector:
    def __init__(self, spec: str = "", seed: int = 0):
        self.faults = parse(spec)
        self.rng = np.random.default_rng(seed)
        self._last: dict = {}          # sensor event type -> last good event (for freeze)
        self._held: list = []          # (release time, event) for latency
        self.counts: dict = {}         # "sensor:kind" -> events affected

    def __bool__(self) -> bool:
        return bool(self.faults)

    def reset(self) -> None:
        self._last, self._held, self.counts = {}, [], {}

    def describe(self, t: float) -> list[str]:
        return [f"{f.sensor}:{f.kind}" for f in self.faults if f.active(t)]

    def _count(self, f: Fault) -> None:
        k = f"{f.sensor}:{f.kind}"
        self.counts[k] = self.counts.get(k, 0) + 1

    def apply(self, events: list, t: float) -> list:
        """Events of one step -> what the stack receives at time t."""
        if not self.faults:
            return events
        out = []
        for ev in events:
            kind = ev.get("type")
            sensor = next((s for s, kinds in SENSOR_EVENTS.items() if kind in kinds), None)
            if sensor is None:
                out.append(ev)
                continue
            key = (kind, ev.get("topic", ""))
            act = [f for f in self.faults if f.sensor == sensor and f.active(float(ev.get("stamp", t)))]
            drop, delay = False, 0.0
            for f in act:
                if f.kind == "dropout":
                    drop = True
                elif f.kind == "freeze":
                    last = self._last.get(key)
                    if last is not None:
                        ev = {**last, "stamp": ev["stamp"]}
                    else:
                        drop = True
                elif f.kind == "latency":
                    delay = max(delay, f.value)
                elif f.kind == "thin":
                    c = ev["cloud"]
                    keep = self.rng.random(len(c["xyz"])) < max(min(f.value, 1.0), 0.0)
                    ev = {**ev, "cloud": {k: v[keep] for k, v in c.items()}}
                elif f.kind == "bias":
                    ev = copy.copy(ev)
                    w = list(ev["angular_velocity"])
                    w[2] += f.value
                    ev["angular_velocity"] = w
                elif f.kind == "scale":
                    ev = {**ev, "wheel_rpm": [float(r) * f.value for r in ev["wheel_rpm"]]}
                elif f.kind == "jump":
                    ev = copy.copy(ev)
                    ev["lon"] = ev["lon"] + math.degrees(f.value / (6378137.0 * math.cos(math.radians(ev["lat"]))))
                self._count(f)
            if not any(f.kind == "freeze" for f in act) and not drop:
                self._last[key] = ev
            if drop:
                continue
            if delay > 0.0:
                self._held.append((t + delay, ev))
            else:
                out.append(ev)
        if self._held:
            due = [e for rel, e in self._held if rel <= t + 1e-9]
            self._held = [(rel, e) for rel, e in self._held if rel > t + 1e-9]
            out.extend(due)
        return out
