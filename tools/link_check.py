#!/usr/bin/env python3
"""Check the Windows <-> WSL link before starting Isaac Sim.

In WSL (where the relay will run):
    python3 tools/link_check.py serve
On Windows (any Python with numpy, e.g. C:\\isaacsim\\python.bat):
    C:\\isaacsim\\python.bat tools\\link_check.py connect --host <WSL IP>     # IP: run `hostname -I` in WSL

It reports round-trip latency and the throughput available for LiDAR and camera
frames. Rough needs at default settings: ~10 MB/s for the OS1-64 and ~100 MB/s
for ZED left RGB + depth at 1280x720 / 15 Hz.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ros2_ws", "src", "fsdsim"))

import numpy as np  # noqa: E402

from fsdsim.link import DEFAULT_PORT, LinkClient, LinkServer  # noqa: E402


def serve(port: int) -> None:
    srv = LinkServer("0.0.0.0", port)
    print(f"listening on 0.0.0.0:{port} - now run 'link_check.py connect --host <this IP>' on Windows")
    t0, b0 = time.time(), 0
    while True:
        msg = srv.incoming.get()
        k = msg.get("kind")
        if k == "_connected":
            print(f"connected: {msg['peer']}")
            t0, b0 = time.time(), srv.recv_bytes
        elif k == "_disconnected":
            print("disconnected")
        elif k == "ping":
            srv.send({"kind": "pong", "t": msg["t"]})
        elif k == "bulk" and msg.get("last"):
            dt = time.time() - t0
            print(f"received {(srv.recv_bytes - b0) / 1e6:.1f} MB in {dt:.2f} s -> {(srv.recv_bytes - b0) / 1e6 / dt:.0f} MB/s")
            srv.send({"kind": "done"})


def connect(host: str, port: int, mb: int) -> None:
    cli = LinkClient(host, port, max_queue=10_000)
    if not cli.connected.wait(10):
        sys.exit(f"could not connect to {host}:{port} - see docs/WINDOWS_WSL.md (troubleshooting)")
    rtts = []
    for _ in range(20):
        t = time.perf_counter()
        cli.send({"kind": "ping", "t": t}, droppable=False)
        while True:
            replies = [m for m in cli.poll() if m.get("kind") == "pong"]
            if replies:
                rtts.append((time.perf_counter() - t) * 1000)
                break
            time.sleep(0.0005)
    print(f"round trip: median {np.median(rtts):.2f} ms, max {max(rtts):.2f} ms")
    frame = np.random.randint(0, 255, (720, 1280, 3), np.uint8)  # one ZED RGB frame (2.76 MB)
    n = max(1, int(mb * 1e6 / frame.nbytes))
    t0 = time.perf_counter()
    for i in range(n):
        cli.send({"kind": "bulk", "data": frame, "last": i == n - 1}, droppable=False)
    while not any(m.get("kind") == "done" for m in cli.poll()):
        time.sleep(0.01)
    dt = time.perf_counter() - t0
    rate = n * frame.nbytes / 1e6 / dt
    print(f"sent {n} camera frames ({n * frame.nbytes / 1e6:.0f} MB) in {dt:.2f} s -> {rate:.0f} MB/s")
    if rate < 120:
        print("note: below ~120 MB/s; use --camera-streams left and/or --camera-hz 10 in run_isaac")
    cli.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--port", type=int, default=DEFAULT_PORT)
    c = sub.add_parser("connect")
    c.add_argument("--host", required=True)
    c.add_argument("--port", type=int, default=DEFAULT_PORT)
    c.add_argument("--mb", type=int, default=500, help="data volume for the throughput test")
    a = ap.parse_args()
    serve(a.port) if a.mode == "serve" else connect(a.host, a.port, a.mb)


if __name__ == "__main__":
    main()
