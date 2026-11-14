#!/usr/bin/env python3
"""Is this computer ready for the simulator? Checks every part and says how to fix what is missing.

    python3 tools/check_install.py            # in WSL (Ubuntu 24.04) or on Linux
    python3 tools/check_install.py --json

Required for the tools and the offline demo: Python >= 3.10, NumPy, PyYAML, the LapSim folder.
For the ROS 2 simulators: ROS 2 Jazzy, rmw_zenoh, ackermann_msgs, colcon, a built workspace.
For Isaac Sim: Windows interop from WSL and the Isaac Sim folder. Optional: PyTorch, ONNX Runtime, matplotlib,
MATLAB, the CAD model, a trained cone detector. Exit code 1 if something required for the tools is missing.
The full installation guide is docs/INSTALL.md.
"""

import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _mod(name):
    try:
        m = importlib.import_module(name)
        return getattr(m, "__version__", "installed")
    except Exception:  # noqa: BLE001
        return None


def _ros_has(pkg: str) -> bool:
    return any(os.path.isdir(os.path.join(p, "share", pkg)) for p in ("/opt/ros/jazzy",) + tuple(
        x for x in os.environ.get("AMENT_PREFIX_PATH", "").split(":") if x))


def checks() -> list[dict]:
    out = []

    def add(group, name, ok, detail="", fix="", required=False):
        out.append({"group": group, "name": name, "ok": bool(ok), "detail": str(detail or ""), "fix": fix, "required": required})

    v = sys.version_info
    add("tools", "Python >= 3.10", v >= (3, 10), f"{v.major}.{v.minor}.{v.micro}", "sudo apt install python3", True)
    for mod, pkg, req in (("numpy", "python3-numpy", True), ("yaml", "python3-yaml", True), ("matplotlib", "python3-matplotlib", False)):
        ver = _mod(mod)
        add("tools", mod if mod != "yaml" else "PyYAML", ver, ver, f"sudo apt install {pkg}", req)
    lap = os.path.isfile(os.path.join(ROOT, "lapsim", "LR26_LapSim", "config", "lr26_params.m"))
    add("tools", "LapSim folder (lapsim/LR26_LapSim)", lap, "", "copy the Performance department's LR26_LapSim folder into lapsim/", True)
    add("tools", "unit tests present", os.path.isdir(os.path.join(ROOT, "tests")), "", "", False)

    jazzy = os.path.isfile("/opt/ros/jazzy/setup.bash")
    add("ros", "ROS 2 Jazzy", jazzy, "/opt/ros/jazzy" if jazzy else "", "docs/INSTALL.md step 3 (sudo apt install ros-jazzy-desktop)")
    add("ros", "rmw_zenoh_cpp", _ros_has("rmw_zenoh_cpp"), "", "sudo apt install ros-jazzy-rmw-zenoh-cpp")
    add("ros", "ackermann_msgs", _ros_has("ackermann_msgs"), "", "sudo apt install ros-jazzy-ackermann-msgs")
    add("ros", "colcon", shutil.which("colcon"), shutil.which("colcon") or "", "sudo apt install python3-colcon-common-extensions")
    built = [os.path.join(b, "install", "fsd_sim_bringup") for b in (os.path.expanduser("~/.fsg-dv-sim/colcon"), os.path.join(ROOT, "ros2_ws"))]
    add("ros", "workspace built", any(os.path.isdir(b) for b in built), next((b for b in built if os.path.isdir(b)), ""),
        "bash tools/wsl/fsgbuild.sh  (or the launcher: Maintenance > Build workspace)")
    try:
        with open(os.path.expanduser("~/.bashrc"), errors="ignore") as f:
            alias = "fsgsim" in f.read()
    except OSError:
        alias = False
    add("ros", "fsgsim / fsgrouter / fsgbuild aliases", alias, "", "bash tools/wsl/setup_shell.sh && source ~/.bashrc")

    ps = shutil.which("powershell.exe")
    add("isaac", "Windows interop (powershell.exe from WSL)", ps, ps or "", "run inside WSL 2 on Windows; in /etc/wsl.conf keep [interop] enabled=true")
    isaac = next((p for p in (os.environ.get("ISAAC_SIM", ""), "/mnt/c/isaacsim", os.path.expanduser("~/isaacsim"))
                  if p and os.path.isdir(p)), "")
    add("isaac", "Isaac Sim 6.1 folder", isaac, isaac, "install Isaac Sim to C:\\isaacsim (docs/INSTALL.md step 5) or set it in the launcher's Settings")
    usd = os.path.isfile(os.path.join(ROOT, "assets", "lr26", "lr26_body.usda"))
    add("isaac", "LR26 CAD model (assets/lr26)", usd, "", "python3 tools/cad_to_usd.py Assem1.obj --unit-scale 0.01 --set-yaml --hoop  (else the simple car is drawn)")

    t = _mod("torch")
    add("optional", "PyTorch (universal RL training)", t, t, "pip install torch  (GPU: on Windows, launcher > Maintenance > Install packages for GPU jobs)")
    o = _mod("onnxruntime")
    add("optional", "ONNX Runtime (camera cones)", o, o, "launcher > Maintenance > Install ONNX Runtime in WSL")
    models = os.path.join(ROOT, "models", "cones")
    n = len([d for d in os.listdir(models) if os.path.isfile(os.path.join(models, d, "model.onnx"))]) if os.path.isdir(models) else 0
    add("optional", "trained cone detector (models/cones)", n, f"{n} model(s)", "record images in Isaac Sim, then launcher > RL & Performance > Train the cone detector")
    pol = os.path.join(ROOT, "rl", "policies")
    n = len([d for d in os.listdir(pol) if os.path.isfile(os.path.join(pol, d, "policy.npz"))]) if os.path.isdir(pol) else 0
    add("optional", "RL policies (rl/policies)", n, f"{n} policy folder(s)", "python3 tools/train_rl.py")
    ml = os.environ.get("FSG_MATLAB") or shutil.which("matlab") or next(
        (os.path.join(r, d, "bin", "matlab.exe") for r in ("/mnt/c/Program Files/MATLAB",) if os.path.isdir(r)
         for d in sorted(os.listdir(r), reverse=True) if os.path.isfile(os.path.join(r, d, "bin", "matlab.exe"))), "")
    add("optional", "MATLAB (LapSim in MATLAB from the dashboard)", ml, ml, "optional: install MATLAB or set FSG_MATLAB")
    try:
        nv = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True, timeout=5)
        gpu = nv.stdout.strip().splitlines()[0] if nv.returncode == 0 and nv.stdout.strip() else ""
    except Exception:  # noqa: BLE001
        gpu = ""
    add("optional", "NVIDIA GPU visible", gpu, gpu, "Isaac Sim needs an RTX GPU on Windows; in WSL this only matters for PyTorch")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    res = checks()
    if args.json:
        print(json.dumps(res, indent=1))
    else:
        titles = {"tools": "Tools and offline demo (required)", "ros": "ROS 2 simulators (lite and the Isaac relay)",
                  "isaac": "Isaac Sim", "optional": "Optional"}
        for g, title in titles.items():
            print(f"\n{title}")
            for r in (x for x in res if x["group"] == g):
                print(f"  [{'ok' if r['ok'] else '--'}] {r['name']}" + (f": {r['detail']}" if r["ok"] and r["detail"] else ""))
                if not r["ok"] and r["fix"]:
                    print(f"         fix: {r['fix']}")
        ready = {g: all(r["ok"] for r in res if r["group"] == g) for g in ("tools", "ros", "isaac")}
        print("\nready: tools and offline demo " + ("yes" if ready["tools"] else "NO") + " | ROS 2 simulators "
              + ("yes" if ready["tools"] and ready["ros"] else "no") + " | Isaac Sim "
              + ("yes" if all(ready.values()) else "no") + "   (guide: docs/INSTALL.md)")
    return 0 if all(r["ok"] for r in res if r["required"]) else 1


if __name__ == "__main__":
    sys.exit(main())
