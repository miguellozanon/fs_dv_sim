"""Minimal reader for ASAM MDF 4 (.mf4) measurement files, as written by asammdf and CAN loggers.

No third-party packages (NumPy only). Reads the channel groups, channels, linear / rational / table
conversions, uncompressed (DT) and zlib-compressed (DZ) data. Not a general MDF library: no raw bus-logging
frames, no variable-length signal data, no attachments, no data lists. Use asammdf for those.

    f = MDF4("run.mf4")
    t, rpm = f.signal("act_Speed_R_DCUA")           # first group that has the channel, invalid values -> NaN
    for g, t, x in f.signals("Current_Car_State"):   # every group (e.g. the same message on CAN1 and CAN3)
        ...

"Invalid" means the all-ones raw value of an unsigned channel (0xFFFF: "signal not available" in many CAN
databases, which a linear conversion would turn into a plausible-looking number such as 33535 rpm).
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Channel:
    name: str
    group: int
    cn_type: int               # 0 fixed length, 2 master (time), ...
    data_type: int
    bit_offset: int
    byte_offset: int
    bit_count: int
    unit: str = ""
    conv: dict | None = None
    comment: str = ""


@dataclass
class Group:
    index: int
    name: str
    cycles: int
    rec_id_size: int
    rec_bytes: int             # data bytes
    inval_bytes: int
    channels: list[Channel] = field(default_factory=list)
    data_link: int = 0


class MDF4:
    def __init__(self, path: str):
        with open(path, "rb") as f:
            self.buf = f.read()
        if self.buf[:8] != b"MDF     ":
            raise ValueError("not an MDF file")
        self.version = self.buf[8:16].decode("ascii", "replace").strip()
        if not self.version.startswith("4"):
            raise ValueError(f"only MDF 4 is supported (file is {self.version})")
        self.groups: list[Group] = []
        self._read_tree()
        self._cache: dict[int, np.ndarray] = {}

    # ---- blocks -----------------------------------------------------------------------------
    def _hdr(self, pos: int):
        bid, _, length, nlinks = struct.unpack_from("<4sIQQ", self.buf, pos)
        links = struct.unpack_from(f"<{nlinks}Q", self.buf, pos + 24) if nlinks else ()
        return bid, length, links, pos + 24 + 8 * nlinks

    def _text(self, pos: int) -> str:
        if not pos:
            return ""
        bid, length, _, d = self._hdr(pos)
        if bid not in (b"##TX", b"##MD"):
            return ""
        raw = self.buf[d:pos + length].split(b"\x00", 1)[0]
        return raw.decode("utf-8", "replace")

    def _conversion(self, pos: int):
        if not pos:
            return None
        bid, length, links, d = self._hdr(pos)
        cc_type, prec, flags, nref, nval = struct.unpack_from("<BBHHH", self.buf, d)
        vals = struct.unpack_from(f"<{nval}d", self.buf, d + 8 + 16) if nval else ()
        conv = {"type": cc_type, "vals": vals}
        if cc_type in (7, 8, 9, 10):
            conv["texts"] = [self._text(p) for p in links[4:4 + nref]]
        return conv

    def _read_tree(self) -> None:
        _, _, hl, _ = self._hdr(64)
        dg = hl[0]
        while dg:
            _, _, dl, dd = self._hdr(dg)
            rec_id = self.buf[dd]
            cg = dl[1]
            while cg:
                _, _, gl, gd = self._hdr(cg)
                rec_id_val, cycles, flags, sep, _r, dbytes, ibytes = struct.unpack_from("<QQHHIII", self.buf, gd)
                g = Group(len(self.groups), self._text(gl[2]), cycles, rec_id, dbytes, ibytes, data_link=dl[2])
                cn = gl[1]
                while cn:
                    _, _, nl, nd = self._hdr(cn)
                    (cn_type, sync, dtype, bit_off, byte_off, bit_cnt, cflags, inval,
                     prec, _r2, natt) = struct.unpack_from("<BBBBIIIIBBH", self.buf, nd)
                    g.channels.append(Channel(self._text(nl[2]), g.index, cn_type, dtype, bit_off, byte_off, bit_cnt,
                                              unit=self._text(nl[6]), conv=self._conversion(nl[4]),
                                              comment=self._text(nl[7])))
                    cn = nl[0]
                self.groups.append(g)
                cg = gl[0]
            dg = dl[0]

    # ---- data -------------------------------------------------------------------------------
    def _raw(self, g: Group) -> np.ndarray:
        if g.index in self._cache:
            return self._cache[g.index]
        n = g.rec_id_size + g.rec_bytes + g.inval_bytes
        pos = g.data_link
        if not pos or g.cycles == 0:
            arr = np.zeros((0, n), np.uint8)
        else:
            bid, length, links, d = self._hdr(pos)
            if bid == b"##DT":
                data = self.buf[d:pos + length]
            elif bid == b"##DZ":
                org, ztype, _r, zparam, org_len, zlen = struct.unpack_from("<2sBBIQQ", self.buf, d)
                raw = zlib.decompress(self.buf[d + 24:d + 24 + zlen])
                if ztype == 1:           # transposed: columns = zparam bytes per record
                    rows = len(raw) // zparam
                    body = np.frombuffer(raw[:rows * zparam], np.uint8).reshape(zparam, rows).T.tobytes()
                    data = body + raw[rows * zparam:]
                else:
                    data = raw
            else:
                raise ValueError(f"unsupported data block {bid!r} (data lists are not supported)")
            arr = np.frombuffer(data, np.uint8)
            arr = arr[: (len(arr) // n) * n].reshape(-1, n)
        self._cache[g.index] = arr
        return arr

    @staticmethod
    def _convert(x: np.ndarray, conv) -> np.ndarray:
        if not conv or conv["type"] == 0:
            return x
        t, v = conv["type"], conv["vals"]
        x = x.astype(np.float64)
        if t == 1 and len(v) >= 2:
            return v[0] + v[1] * x
        if t == 2 and len(v) >= 6:
            return (v[0] * x * x + v[1] * x + v[2]) / (v[3] * x * x + v[4] * x + v[5])
        if t == 4 and len(v) >= 4:           # table with interpolation: pairs (int, phys)
            xs, ys = np.array(v[0::2]), np.array(v[1::2])
            return np.interp(x, xs, ys)
        if t == 5 and len(v) >= 4:           # table without interpolation
            xs, ys = np.array(v[0::2]), np.array(v[1::2])
            return ys[np.abs(x[:, None] - xs[None, :]).argmin(axis=1)]
        return x                              # text tables etc.: keep the raw numbers

    def channel_names(self) -> list[tuple[int, str]]:
        return [(g.index, c.name) for g in self.groups for c in g.channels]

    def read(self, group: int, channel: str | Channel, convert: bool = True, mask_invalid: bool = False) -> np.ndarray:
        g = self.groups[group]
        ch = channel if isinstance(channel, Channel) else next(c for c in g.channels if c.name == channel)
        arr = self._raw(g)
        o = g.rec_id_size + ch.byte_offset
        nbytes = (ch.bit_offset + ch.bit_count + 7) // 8
        cols = arr[:, o:o + nbytes]
        if ch.data_type in (0, 1, 2, 3):
            big = ch.data_type in (1, 3)
            val = np.zeros(len(arr), np.uint64)
            for i in range(nbytes):
                shift = 8 * (nbytes - 1 - i) if big else 8 * i
                val |= cols[:, i].astype(np.uint64) << np.uint64(shift)
            val >>= np.uint64(ch.bit_offset)
            if ch.bit_count < 64:
                val &= np.uint64((1 << ch.bit_count) - 1)
            if ch.data_type in (2, 3):
                sign = np.uint64(1 << (ch.bit_count - 1))
                out = val.astype(np.int64)
                out = np.where(val & sign, out - (1 << ch.bit_count), out)
            else:
                out = val
        elif ch.data_type in (4, 5):
            dt = ("<" if ch.data_type == 4 else ">") + ("f4" if ch.bit_count == 32 else "f8")
            out = np.ascontiguousarray(cols).view(dt).reshape(-1)
        else:
            raise ValueError(f"unsupported data type {ch.data_type} for {ch.name}")
        if mask_invalid and ch.data_type in (0, 1) and 1 < ch.bit_count < 64:
            bad = out == np.uint64((1 << ch.bit_count) - 1)
            out = out.astype(np.float64)
            out[bad] = np.nan
        return self._convert(out, ch.conv) if convert else out

    def time(self, group: int) -> np.ndarray | None:
        g = self.groups[group]
        for c in g.channels:
            if c.cn_type == 2:
                return self.read(group, c)
        return None

    def find(self, name: str) -> list[tuple[int, Channel]]:
        return [(g.index, c) for g in self.groups for c in g.channels if c.name == name]

    def signals(self, name: str, mask_invalid: bool = True) -> list[tuple[str, np.ndarray, np.ndarray]]:
        """[(group name, time [s], values)] of every channel called `name` (needs a time channel in the group)."""
        out = []
        for gi, ch in self.find(name):
            t = self.time(gi)
            if t is None or ch.cn_type == 2:
                continue
            out.append((self.groups[gi].name, t, self.read(gi, ch, mask_invalid=mask_invalid)))
        return out

    def signal(self, name: str, group: str | None = None, mask_invalid: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """(time, values) of `name`; `group` is a part of the group name (e.g. 'CAN3') to choose between buses."""
        for gname, t, x in self.signals(name, mask_invalid):
            if group is None or group in gname:
                return t, x
        raise KeyError(f"no channel '{name}'" + (f" in a group matching '{group}'" if group else ""))


def resample(series: dict[str, tuple[np.ndarray, np.ndarray]], rate_hz: float, t0: float | None = None,
             t1: float | None = None) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Put signals on one uniform time base by holding the last value (NaN before a signal's first sample)."""
    starts = [t[0] for t, _ in series.values() if len(t)]
    ends = [t[-1] for t, _ in series.values() if len(t)]
    if not starts:
        return np.zeros(0), {k: np.zeros(0) for k in series}
    t0 = min(starts) if t0 is None else t0
    t1 = max(ends) if t1 is None else t1
    grid = np.arange(t0, t1 + 0.5 / rate_hz, 1.0 / rate_hz)
    out = {}
    for k, (t, x) in series.items():
        idx = np.searchsorted(t, grid, side="right") - 1
        y = np.asarray(x, np.float64)[np.clip(idx, 0, max(len(t) - 1, 0))] if len(t) else np.full(len(grid), np.nan)
        y = np.where(idx < 0, np.nan, y)
        out[k] = y
    return grid, out
