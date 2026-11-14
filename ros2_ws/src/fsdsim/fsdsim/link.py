"""TCP link between the simulator process (Isaac Sim on Windows) and the ROS 2
relay (WSL / Linux). Standard library + NumPy only, so it runs in Isaac Sim's
Python and in the ROS 2 Python without extra packages.

Why not DDS? Isaac Sim (Windows) <-> ROS 2 (WSL2) over DDS needs WSL network
mode tweaks, Fast DDS peer profiles and firewall rules, and NVIDIA has
deprecated that path since Isaac Sim 6.x. One TCP connection from Windows to
WSL avoids all of it.

Wire format, one message per frame:

    uint32 header_len | uint32 payload_len | header (UTF-8 JSON) | payload

NumPy arrays anywhere inside the message are moved into the binary payload and
replaced in the JSON by {"__nd__": [offset, nbytes, dtype, shape]}.
"""

from __future__ import annotations

import collections
import json
import queue
import socket
import struct
import threading
import time
from typing import Any, Callable

import numpy as np

DEFAULT_PORT = 47800
PROTOCOL_VERSION = 1
_HDR = struct.Struct("<II")
MAX_HEADER = 64 * 1024 * 1024
MAX_PAYLOAD = 512 * 1024 * 1024


# --------------------------------------------------------------------------- #
# Encoding
# --------------------------------------------------------------------------- #
def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, (set, tuple)):
        return list(o)
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def encode(msg: Any) -> bytes:
    blobs: list[bytes] = []
    offset = [0]

    def walk(o):
        if isinstance(o, np.ndarray):
            a = np.ascontiguousarray(o)
            b = a.tobytes()
            ref = {"__nd__": [offset[0], len(b), a.dtype.str, list(a.shape)]}
            blobs.append(b)
            offset[0] += len(b)
            return ref
        if isinstance(o, (bytes, bytearray, memoryview)):
            b = bytes(o)
            ref = {"__nd__": [offset[0], len(b), "|u1", [len(b)]]}
            blobs.append(b)
            offset[0] += len(b)
            return ref
        if isinstance(o, dict):
            return {k: walk(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [walk(v) for v in o]
        return o

    header = json.dumps(walk(msg), default=_json_default, separators=(",", ":")).encode("utf-8")
    payload = b"".join(blobs)
    return _HDR.pack(len(header), len(payload)) + header + payload


def decode(header: bytes, payload: bytes) -> Any:
    view = memoryview(payload)

    def hook(d):
        ref = d.get("__nd__") if len(d) == 1 else None
        if ref is not None:
            off, n, dtype, shape = ref
            return np.frombuffer(view[off:off + n], dtype=np.dtype(dtype)).reshape(shape)
        return d

    return json.loads(header.decode("utf-8"), object_hook=hook)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        k = sock.recv_into(view[got:], n - got)
        if k == 0:
            raise ConnectionError("peer closed the connection")
        got += k
    return bytes(buf)


def read_message(sock: socket.socket, stats: dict | None = None) -> Any:
    hlen, plen = _HDR.unpack(_recv_exact(sock, _HDR.size))
    if stats is not None:
        stats["bytes"] = stats.get("bytes", 0) + _HDR.size + hlen + plen
    if hlen > MAX_HEADER or plen > MAX_PAYLOAD:
        raise ConnectionError(f"frame too large ({hlen}, {plen}) - protocol mismatch?")
    header = _recv_exact(sock, hlen)
    payload = _recv_exact(sock, plen) if plen else b""
    return decode(header, payload)


def _tune(sock: socket.socket, sndbuf: int = 8 * 1024 * 1024) -> None:
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    for opt, size in ((socket.SO_SNDBUF, sndbuf), (socket.SO_RCVBUF, 8 * 1024 * 1024)):
        try:
            sock.setsockopt(socket.SOL_SOCKET, opt, size)
        except OSError:
            pass


# --------------------------------------------------------------------------- #
# Client (simulator side, e.g. Isaac Sim on Windows)
# --------------------------------------------------------------------------- #
class LinkClient:
    """Connects (and reconnects) to the relay; never blocks the render loop.

    send(msg, droppable=True) queues a message. When the queue backs up (slow
    link), droppable messages (sensor frames) are discarded so the simulation
    keeps running in real time; non-droppable ones (AS state) are kept.
    """

    def __init__(self, host: str, port: int = DEFAULT_PORT, on_connect: Callable[[], list] | None = None,
                 log: Callable[[str], None] = print, max_queue: int = 3, state_channel: bool = True):
        self.host, self.port = host, port
        self.on_connect = on_connect
        self.log = log
        # State (pose, IMU, wheels, AS) goes first and is never dropped; sensor frames (images, scans) wait in a
        # short queue that drops the OLDEST frame. Before 11 Nov 2026 one FIFO held up to 64 frames of several MB:
        # when the Windows -> WSL link slowed down, the car's pose waited behind them for 1-2 s.
        self.max_queue = max_queue
        self._q: "queue.Queue[tuple[bytes, float]]" = queue.Queue()         # state messages, time queued
        self._frames: "collections.deque[bytes]" = collections.deque()    # sensor frames, newest last
        self._frames_lock = threading.Lock()
        self._wake = threading.Event()
        self._incoming: "queue.Queue[Any]" = queue.Queue()
        self._sock: socket.socket | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.connected = threading.Event()
        self.dropped = 0
        self.sent_bytes = 0
        self.state_wait_max = 0.0     # s from send() until a state message was handed to the socket (read and reset)
        self.write_max = (0.0, 0)     # longest sendall in s and its size in bytes (read and reset)
        # Second connection for state (when the relay offers it): on the team PC a few-kB state message waited up to
        # 0.4 s behind multi-MB camera frames in the one Windows -> WSL socket (11 Nov 2026).
        self.use_state_channel = state_channel
        self._caps = threading.Event()             # the relay accepts a state connection
        self._state_sock: socket.socket | None = None
        self.state_channel_up = threading.Event()
        threading.Thread(target=self._run, name="fsd-link-client", daemon=True).start()

    def send(self, msg: Any, droppable: bool = True) -> None:
        if not self.connected.is_set():
            return  # nothing to send to; state is re-sent on connect
        data = encode(msg)
        if droppable:
            with self._frames_lock:
                self._frames.append(data)
                while len(self._frames) > self.max_queue:
                    self._frames.popleft()          # the oldest frame is the least useful one
                    self.dropped += 1
        else:
            self._q.put((data, time.monotonic()))
        self._wake.set()

    def _next(self, timeout: float):
        """State messages first (unless they have their own connection), then the oldest waiting sensor frame;
        None after `timeout` without data."""
        deadline = time.monotonic() + timeout
        while True:
            if not self.state_channel_up.is_set():
                try:
                    data, t_q = self._q.get_nowait()
                    self.state_wait_max = max(self.state_wait_max, time.monotonic() - t_q)
                    return data
                except queue.Empty:
                    pass
            with self._frames_lock:
                if self._frames:
                    return self._frames.popleft()
                self._wake.clear()
            left = deadline - time.monotonic()
            if left <= 0 or not self._wake.wait(left):
                return None

    def poll(self) -> list:
        out = []
        while True:
            try:
                out.append(self._incoming.get_nowait())
            except queue.Empty:
                return out

    def close(self) -> None:
        self._stop.set()
        with self._lock:
            for sk in (self._sock, self._state_sock):
                if sk is not None:
                    try:
                        sk.close()
                    except OSError:
                        pass

    # ------------------------------------------------------------------ #
    def _run(self) -> None:
        warned = False
        while not self._stop.is_set():
            try:
                sock = socket.create_connection((self.host, self.port), timeout=3.0)
            except OSError as e:
                if not warned:
                    self.log(f"link: waiting for the ROS 2 relay at {self.host}:{self.port} ({e})")
                    warned = True
                time.sleep(1.0)
                continue
            sock.settimeout(None)
            _tune(sock, sndbuf=2 * 1024 * 1024)       # less data in flight: lower latency when the link is slow
            with self._lock:
                self._sock = sock
            warned = False
            # drop stale frames from a previous connection
            while not self._q.empty():
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    break
            with self._frames_lock:
                self._frames.clear()
            try:
                for m in (self.on_connect() if self.on_connect else []):
                    sock.sendall(encode(m))
            except OSError:
                self._close_sock()
                continue
            self._caps.clear()
            self.connected.set()
            self.log(f"link: connected to relay {self.host}:{self.port}")
            reader = threading.Thread(target=self._reader, args=(sock,), name="fsd-link-reader", daemon=True)
            reader.start()
            tried_state = False
            try:
                while not self._stop.is_set() and reader.is_alive():
                    if self._caps.is_set() and not tried_state and self.use_state_channel:
                        tried_state = True
                        self._open_state_channel()
                    data = self._next(0.2)
                    if data is None:
                        continue
                    t_w = time.monotonic()
                    sock.sendall(data)
                    t_w = time.monotonic() - t_w
                    if t_w > self.write_max[0]:
                        self.write_max = (t_w, len(data))
                    self.sent_bytes += len(data)
            except OSError as e:
                self.log(f"link: send failed ({e}); reconnecting")
            self.connected.clear()
            self._close_sock()

    def _open_state_channel(self) -> None:
        try:
            st = socket.create_connection((self.host, self.port), timeout=3.0)
            st.settimeout(None)
            _tune(st, sndbuf=256 * 1024)
            st.sendall(encode({"kind": "_channel", "role": "state"}))
        except OSError as e:
            self.log(f"link: no separate state connection ({e}); state shares the sensor connection")
            return
        with self._lock:
            self._state_sock = st
        self.state_channel_up.set()
        threading.Thread(target=self._state_sender, args=(st,), name="fsd-link-state", daemon=True).start()
        self.log("link: state (pose, IMU, wheels, AS) on its own connection")

    def _state_sender(self, st: socket.socket) -> None:
        try:
            while not self._stop.is_set() and self.state_channel_up.is_set():
                try:
                    data, t_q = self._q.get(timeout=0.2)
                except queue.Empty:
                    continue
                self.state_wait_max = max(self.state_wait_max, time.monotonic() - t_q)
                t_w = time.monotonic()
                st.sendall(data)
                t_w = time.monotonic() - t_w
                if t_w > self.write_max[0]:
                    self.write_max = (t_w, len(data))
                self.sent_bytes += len(data)
        except OSError as e:
            self.log(f"link: state connection failed ({e}); reconnecting")
            self._close_sock()            # the main loop sees the closed socket and reconnects both

    def _reader(self, sock: socket.socket) -> None:
        try:
            while True:
                msg = read_message(sock)
                if isinstance(msg, dict):
                    if msg.get("kind") == "_caps":
                        if msg.get("state_channel"):
                            self._caps.set()
                        continue
                    msg["_rx"] = time.monotonic()
                self._incoming.put(msg)
        except (OSError, ConnectionError, ValueError):
            pass

    def _close_sock(self) -> None:
        self.state_channel_up.clear()
        with self._lock:
            for sk in (self._sock, self._state_sock):
                if sk is not None:
                    try:
                        sk.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                    try:
                        sk.close()
                    except OSError:
                        pass
            self._sock = self._state_sock = None


# --------------------------------------------------------------------------- #
# Server (ROS 2 relay side, WSL / Linux)
# --------------------------------------------------------------------------- #
class LinkServer:
    """Accepts one simulator at a time; a new connection replaces the old one. A simulator may open a second
    connection for its small state messages (first message `{"kind": "_channel", "role": "state"}`), offered with
    `{"kind": "_caps", "state_channel": true}` when it connects.

    Received messages are put on `self.incoming` (drain it from your own
    thread, e.g. a ROS timer) so no ROS calls happen on the socket thread.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = DEFAULT_PORT, log: Callable[[str], None] = print):
        self.log = log
        self.incoming: "queue.Queue[Any]" = queue.Queue()
        self._client: socket.socket | None = None
        self._aux: list[socket.socket] = []      # the simulator's state connection(s)
        self._lock = threading.Lock()
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind((host, port))
        self._srv.listen(1)
        self.port = self._srv.getsockname()[1]
        self.peer: str | None = None
        self._stats: dict = {"bytes": 0}
        threading.Thread(target=self._accept_loop, name="fsd-link-accept", daemon=True).start()

    def send(self, msg: Any) -> bool:
        with self._lock:
            c = self._client
            if c is None:
                return False
            try:
                c.sendall(encode(msg))
                return True
            except OSError:
                return False

    @property
    def recv_bytes(self) -> int:
        return self._stats["bytes"]

    @property
    def connected(self) -> bool:
        return self._client is not None

    def close(self) -> None:
        try:
            self._srv.close()
        except OSError:
            pass

    def _accept_loop(self) -> None:
        while True:
            try:
                sock, addr = self._srv.accept()
            except OSError:
                return
            _tune(sock)
            threading.Thread(target=self._serve, args=(sock, addr), name="fsd-link-server-reader", daemon=True).start()

    def _serve(self, sock: socket.socket, addr) -> None:
        """The first message tells a simulator's main connection from its state connection (`_channel`)."""
        try:
            first = read_message(sock, self._stats)
        except (OSError, ConnectionError, ValueError):
            sock.close()
            return
        if isinstance(first, dict) and first.get("kind") == "_channel" and first.get("role") == "state":
            with self._lock:
                self._aux.append(sock)
            try:
                while True:
                    msg = read_message(sock, self._stats)
                    if isinstance(msg, dict):
                        msg["_rx"] = time.monotonic()
                    self.incoming.put(msg)
            except (OSError, ConnectionError, ValueError):
                pass
            with self._lock:
                if sock in self._aux:
                    self._aux.remove(sock)
            return
        with self._lock:
            for old in [self._client, *self._aux]:       # a new simulator replaces the old one
                if old is not None:
                    try:
                        old.close()
                    except OSError:
                        pass
            self._aux = []
            self._client = sock
            try:
                sock.sendall(encode({"kind": "_caps", "state_channel": True}))
            except OSError:
                pass
        self.peer = f"{addr[0]}:{addr[1]}"
        self.incoming.put({"kind": "_connected", "peer": self.peer})
        if isinstance(first, dict):
            first["_rx"] = time.monotonic()
        self.incoming.put(first)
        self._reader(sock)

    def _reader(self, sock: socket.socket) -> None:
        try:
            while True:
                msg = read_message(sock, self._stats)
                if isinstance(msg, dict):
                    msg["_rx"] = time.monotonic()      # when it arrived: the relay reports how long it waited
                self.incoming.put(msg)
        except (OSError, ConnectionError, ValueError) as e:
            reason = str(e)
        with self._lock:
            if self._client is sock:
                self._client = None
                for a in self._aux:                     # the state connection belongs to this simulator
                    try:
                        a.close()
                    except OSError:
                        pass
                self._aux = []
                self.incoming.put({"kind": "_disconnected", "reason": reason})
