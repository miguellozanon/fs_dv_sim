# Windows 11 + WSL2 setup

This is the setup it was adapted for:

| Component | Runs on | Version |
|---|---|---|
| Isaac Sim | Windows 11 | 6.1.0 (Python 3.12) |
| Isaac Lab | Windows 11 | 3.0.0 (optional; its Python env can run the simulator) |
| ROS 2 | WSL2, Ubuntu 24.04 | Jazzy (Python 3.12) |

```
 Windows 11                                        WSL2 Ubuntu 24.04
 ┌───────────────────────────────┐                 ┌──────────────────────────────────────┐
 │ Isaac Sim 6.1 (run_isaac.py)  │   TCP :47800    │ isaac_relay  ──> /lidar/points        │
 │  fsdsim: dynamics, AS, IMU... │ ──────────────> │                  /zed/zed_node/...    │
 │  RTX OS1 LiDAR, ZED cameras   │ <────────────── │                  /imu/data, /as/status│
 │  no ROS needed here           │  commands       │                  /clock, TF ...       │
 └───────────────────────────────┘                 │ your stack / demo_driver / RViz       │
                                                   └──────────────────────────────────────┘
```

**Why TCP and not DDS?** With Isaac Sim 6.x, NVIDIA deprecated running the
ROS 2 bridge on Windows against ROS 2 in WSL. That route needs:
* explicit Fast DDS peers that must be updated whenever the WSL IP changes,
* Hyper-V firewall rules,
* a large DDS UDP stream for point clouds and images across the VM boundary.

Here, Isaac opens one TCP connection to the relay in WSL instead. Isaac on
Windows needs no ROS environment at all.

**Inside WSL, ROS 2 uses Zenoh**, set up the same way as the Go2 project:
* every node uses `RMW_IMPLEMENTATION=rmw_zenoh_cpp`;
* nodes find each other through one Zenoh router (`ros2 run rmw_zenoh_cpp rmw_zenohd`, TCP port 7447), instead of DDS multicast, which is unreliable in WSL mirrored networking;
* nothing ROS-related is sourced by `~/.bashrc`, so new terminals open instantly. The `fsgsim` alias sources ROS 2 and the workspace only when you need them, like `go2sim`.

One router serves both projects. The simulator uses `ROS_DOMAIN_ID=18` (Go2: 17), so both can run at the same time without their `/clock`, `/tf` or `/lidar/points` mixing.

---


> **No terminals needed:** keep the project on the Windows drive (e.g. `C:\formula_student\fsg-dv-sim`)
> and double-click `FSG DV Launcher.bat` next to it, or `run.bat` inside it. It starts the launcher in WSL
> and opens http://localhost:8070, which builds the workspace when needed and starts the router, the
> launch files and Isaac Sim for you. See [LAUNCHER.md](LAUNCHER.md). The terminal steps below do the same
> by hand. For a copy on C:, run `bash /mnt/c/formula_student/fsg-dv-sim/tools/wsl/setup_shell.sh` once
> so that `fsgsim` / `fsgbuild` use it. Its build files then go to `~/.fsg-dv-sim/colcon`.

## 1. One-time setup

### WSL (Ubuntu 24.04, ROS 2 Jazzy)

```bash
sudo apt update
sudo apt install ros-jazzy-rmw-zenoh-cpp ros-jazzy-ackermann-msgs ros-jazzy-rviz2 \
                 python3-numpy python3-yaml python3-colcon-common-extensions
# put the project in ~/fsg-dv-sim (unzip fsg-dv-sim.zip in ~, or git clone your team's copy)

bash ~/fsg-dv-sim/tools/wsl/setup_shell.sh     # aliases; add --dry-run to preview the .bashrc change
source ~/.bashrc
fsgbuild                                        # = fsgsim && colcon build --symlink-install && fsgsim
```

`setup_shell.sh` backs up `~/.bashrc` and comments out the `source /opt/ros/jazzy/...` and workspace lines from the earlier instructions; it does not touch other lines such as your `go2sim` alias. It then adds:

| Alias | Does |
|---|---|
| `fsgsim` | sources ROS 2 Jazzy and this workspace, sets `RMW_IMPLEMENTATION=rmw_zenoh_cpp` and `ROS_DOMAIN_ID=18`, and says whether the Zenoh router is running |
| `fsgrouter` | `fsgsim && ros2 run rmw_zenoh_cpp rmw_zenohd` (the router; keep it open) |
| `fsgbuild` | rebuilds the workspace and re-sources it |

Change the domain with `export FSG_ROS_DOMAIN_ID=...` in `~/.bashrc`. For DDS instead of Zenoh, use `FSG_RMW=rmw_fastrtps_cpp`.

**After updating the project:** `cd ~/fsg-dv-sim/ros2_ws && rm -rf build install log && fsgbuild`.

Keep the repository **in the WSL filesystem** (`~/...`, not `/mnt/c/...`).
colcon is much faster there, and Windows can still run the same files through
`\\wsl.localhost\Ubuntu-24.04\home\<you>\fsg-dv-sim`.

RViz opens on the Windows desktop through WSLg, which is built into Windows 11.

### Windows (Isaac Sim 6.1)

You need nothing beyond your existing install. The launcher installs `pyyaml`
into Isaac's Python on first use if it is missing.

PowerShell blocks scripts from a WSL path by default. Use
`-ExecutionPolicy Bypass` for this script, as below, rather than changing the
machine policy.

---

## 2. Run

**Terminal 1: WSL, the Zenoh router** (one per machine, also serves Go2; keep it open)

```bash
fsgrouter
```

**Terminal 2: WSL, relay + reference driver + RViz + dashboard**

```bash
fsgsim && ros2 launch fsd_sim_bringup isaac.launch.py
```

Expected output: `[fsd_isaac_relay]: fsg-dv-sim 0.6.0 relay - waiting for Isaac Sim on tcp://0.0.0.0:47800`.
Without the router, the launch stops at once with `no Zenoh router on tcp/127.0.0.1:7447`.

**Terminal 3: Windows PowerShell**

```powershell
powershell -ExecutionPolicy Bypass -File \\wsl.localhost\Ubuntu-24.04\home\<you>\fsg-dv-sim\isaac\run_isaac.ps1 -Mission skidpad
```

Other WSL terminals: `fsgsim` first, then `ros2 topic list`, `ros2 node list`, etc.

What the launcher does:
* uses `C:\isaacsim\python.bat` (change with `-IsaacPath` or `$env:ISAAC_SIM`);
* asks WSL for its IP and probes the WSL IP, then `localhost`, for the relay;
* starts `run_isaac.py`.

The relay prints `hello from Isaac Sim 6.1.0 | mission skidpad | ...`. The
reference driver sends Go after 5 s in AS Ready and drives the event.

**Dashboard.** Open **http://localhost:8080** in a Windows browser. It shows:
* AS state;
* the track with LiDAR returns;
* the ZED camera;
* telemetry, laps and sensor health;
* Go / Emergency stop buttons.

See [DASHBOARD.md](DASHBOARD.md). RViz opens by itself a few seconds after Isaac connects (`rviz_wait`).

Useful options:

```powershell
# Isaac Lab 3.0 environment instead of the standalone Isaac Sim (point at its python.exe)
... run_isaac.ps1 -Python C:\path\to\isaaclab-env\Scripts\python.exe -Mission trackdrive

# headless, other circuit, both ZED RGB images + depth
... run_isaac.ps1 -Mission autocross -Headless -Extra "--seed","11","--camera-streams","left,right,depth"

# lower camera bandwidth
... run_isaac.ps1 -Mission trackdrive -Extra "--camera-streams","left","--camera-hz","10"

# a WSL distro other than the default
... run_isaac.ps1 -Distro Ubuntu-24.04 -Mission acceleration

# change vehicle parameters for this run (LapSim names, see docs/PERFORMANCE.md)
... run_isaac.ps1 -Mission autocross -Set "m=300","c_z=4.5"
```

Without the launcher:

```powershell
C:\isaacsim\python.bat \\wsl.localhost\Ubuntu-24.04\home\<you>\fsg-dv-sim\isaac\run_isaac.py --mission skidpad --relay-host 127.0.0.1
```

### Config is sent automatically

Isaac reads `vehicle.yaml`, `sensors.yaml` and `sim.yaml` and sends them to the
relay when it connects. The relay builds its topics, frames and static TFs from
that copy, so the two sides can never disagree.

If you edit the YAML, restart Isaac. If you changed topics or frames, restart
the relay too; it warns you when the config fingerprint changes.

---

### MATLAB LapSim and the Vehicle setup page

The MATLAB LapSim is part of the project, in `~/fsg-dv-sim/lapsim/LR26_LapSim`. The simulator reads the car from its `config/lr26_params.m`.

* **In MATLAB:** set the current folder to `\\wsl.localhost\Ubuntu-24.04\home\<you>\fsg-dv-sim\lapsim\LR26_LapSim` and run `run_lapsim` as before. If your MATLAB release does not accept that path, map it to a drive letter first: `net use L: \\wsl.localhost\Ubuntu-24.04\home\<you>\fsg-dv-sim\lapsim`.
* **RL driver:** `ros2 launch fsd_sim_bringup isaac.launch.py longitudinal:=rl` drives with the throttle and brake policy trained for the running vehicle, or with the universal policy when the setup is inside its range. Train a per-setup one with `python3 tools/train_rl.py` or from the dashboard. See [RL.md](RL.md).
* **Universal RL policy on the GPU:** in WSL with the NVIDIA driver for WSL, `python3 -m pip install --user --break-system-packages torch` and `python3 tools/train_rl_universal.py --backend torch --device cuda:0`. Run `python3 -m unittest tests.test_vec` first: its PyTorch tests check the GPU path against NumPy. With Isaac Lab on Windows, see [isaaclab/README.md](../isaaclab/README.md) and [RL_UNIVERSAL.md](RL_UNIVERSAL.md).
* **In the dashboard** (http://localhost:8080/#setup): enter parameter values and press **Apply**. LapSim, MATLAB in the background, and the running Isaac or lite simulation all use them. MATLAB is found automatically under `C:\Program Files\MATLAB`. Otherwise, set `export FSG_MATLAB="C:\...\bin\matlab.exe"` in the dashboard's terminal before the launch. See [DASHBOARD.md](DASHBOARD.md#vehicle-setup).

## 3. Check the link before the first Isaac run (optional)

```bash
# WSL
python3 ~/fsg-dv-sim/tools/link_check.py serve
```

```powershell
# Windows (Isaac's Python has numpy). Get the IP with `hostname -I` in WSL.
C:\isaacsim\python.bat \\wsl.localhost\Ubuntu-24.04\home\<you>\fsg-dv-sim\tools\link_check.py connect --host <WSL IP>
```

The checker reports round-trip time and throughput.

| Stream | Bandwidth |
|---|---|
| OS1-64 LiDAR at 10 Hz | ~10 MB/s |
| ZED left RGB at 1280x720, 15 Hz | ~40 MB/s |
| ZED depth (float32) | ~55 MB/s |
| ZED right RGB | ~40 MB/s |

When the link cannot keep up, Isaac drops sensor frames rather than slowing
down. AS state and commands are never dropped. Isaac's log reports
`link up (… MB/s, N frames dropped)` every 10 s.

---

## 4. Topics you get in WSL

The topics are the same as with the lite backend (see README), plus the rendered camera topics:

* **Point cloud.** `/lidar/points` uses ouster-ros fields:
  * `ring` 0 = top beam
  * `t` = ns from scan start
* **ZED RGB.** `/zed/zed_node/left/image_rect_color` and `.../right/image_rect_color` (if enabled) are `rgb8`, with their `camera_info`. The right camera's `camera_info` carries the stereo baseline in `P[3]`.
* **ZED depth.** `/zed/zed_node/depth/depth_registered` is `32FC1` in metres, plus `camera_info`.
* **`/clock`.** Published by the relay from Isaac's simulation time. Run every node with `use_sim_time:=true`.

---

## 5. Troubleshooting

| Symptom | Fix |
|---|---|
| Launcher: "Relay not reachable" | Start `isaac.launch.py` in WSL first; Isaac keeps retrying anyway. Check `ss -ltnp \| grep 47800` in WSL. |
| Reachable from WSL, not from Windows (mirrored networking) | Allow inbound to WSL through the Hyper-V firewall (admin PowerShell, from Microsoft's WSL networking docs): `Set-NetFirewallHyperVVMSetting -Name '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}' -DefaultInboundAction Allow` |
| `localhost` fails in NAT mode | `localhostForwarding` may be disabled in `%UserProfile%\.wslconfig`. Use `-RelayHost <WSL IP>` instead. |
| WSL IP changed after reboot | Nothing to do; the launcher reads it on every start. |
| Many dropped frames / low MB/s | Run `link_check.py`; reduce `--camera-streams` / `--camera-hz`; prefer the WSL IP over localhost in NAT mode. |
| Real-time factor < 1 in Isaac's log | Normal for the first minutes while shaders compile. If it persists: `-Headless`, fewer camera streams, a lower `--render-hz`. |
| `RTX LiDAR unavailable … analytic` | The RTX sensor could not be created or read. You still get a LiDAR cloud from the analytic OS1 model. Check the variant names Isaac lists in the log and set `lidar.isaac.variant`. |
| Relay: "protocol mismatch" | The Windows and WSL copies are different versions. Run both from the same clone (the `\\wsl.localhost` path). |
| AS Emergency 1 s after Go | No `/control/command` is arriving (watchdog). Start the demo driver or your stack. |
| Isaac window closes / relay shows "Isaac Sim disconnected" while running | Isaac crashed. The relay prints the Kit log path, e.g. `c:/isaacsim/kit/logs/Kit/Isaac-Sim Python/6.1/kit_<date>.log`. Send the last ~100 lines. |
| Endless `failed to create Hydra Engine thread for viewport` warnings | Motion BVH was switched on while Isaac was running (fixed). Use `--motion-bvh` at startup only if you need LiDAR motion compensation. |
| `no RTX LiDAR scan after 3.0 s (...)` | The diagnostics in brackets show whether the Writer was called and whether frames were empty. Send that line. You still get the analytic LiDAR cloud meanwhile. |
| Relay logs a short connect / disconnect right before Isaac starts | That is the launcher checking the port; the relay ignores connections that do not send a hello. |
| `Isaac timeline jumped back ...` | Stop/Play was pressed in the Isaac window. The simulation keeps its own clock and continues; to restart a run, use `ros2 service call /sim/reset std_srvs/srv/Trigger` instead. |
| `pip install pyyaml` fails | Isaac Python offline: copy a `pyyaml` wheel, or install with `python.bat -m pip install <wheel>`. |
| Launch stops: `no Zenoh router on tcp/127.0.0.1:7447` | Start `fsgrouter` in its own terminal first and keep it open. |
| `ros2 topic list` shows nothing / nodes do not see each other | Every terminal needs `fsgsim` (same `RMW_IMPLEMENTATION` and `ROS_DOMAIN_ID`), and the router must be running. A plain `source /opt/ros/jazzy/setup.bash` terminal uses DDS and sees nothing. |
| `Package 'fsd_sim_bringup' not found` | That terminal has not loaded the workspace. Run `fsgsim` first, and check that its line says `workspace sourced`. If it says `workspace NOT built`, run `fsgbuild`. |
| Ctrl+C prints `RCLError: failed to shutdown` and `process has died` | Harmless with 0.4.0 (a shutdown race in the nodes); fixed in 0.4.1. |
| `ros2 node list` hangs or times out | The CLI daemon is stale: `pkill -f _ros2_daemon` (or add `--no-daemon`). It only affects the command-line tools. |

### RViz shows "No tf data. Frame [map] does not exist", or shows nothing

**Isaac restarted or `/sim/reset`.** Every Isaac run, and every `/sim/reset`, starts simulation time at 0 again. If the ROS stamps followed it:
* RViz on the wall clock would reject the new transforms as older than the ones it already holds (`TF_OLD_DATA`) and draw nothing;
* RViz on sim time would wipe its TF buffer.

The relay (and the lite simulator) therefore keep ROS time moving forward. After a restart they continue 1 s past the last stamp and log `simulator time restarted (t=...) ; ROS time continues at ...`. Lap and mission times are not affected. If you run an **older relay** that lacks this, restart the launch (relay + RViz) whenever you restart Isaac.

**RViz started before Isaac connects.** On this WSL setup, an RViz started before Isaac connected showed nothing, while the same RViz started later worked. So `isaac.launch.py` now waits: it starts RViz when the relay prints its `TF self-check` line, about 4 s after Isaac connects. Use `rviz_wait:=false` to start it immediately.

**Discovery in WSL.** About 15 s after Isaac connects, the relay logs `ROS graph seen by the relay: nodes ...; /tf subscribers ...`. If RViz (`/rviz`) is missing from that list, processes are not discovering each other.
* **With Zenoh** (the default since 0.4.0): every terminal must run `fsgsim`, and `fsgrouter` must keep running.
* **With DDS:** multicast is unreliable with WSL mirrored networking. The Zenoh setup above avoids that.

If `ros2 node list` fails with `TimeoutError: [Errno 110] Connection timed out` inside `is_daemon_running`, the CLI's background daemon is hung. That affects only the command-line tools, not the simulation. Fix it with `pkill -f _ros2_daemon` (`ros2 daemon stop` itself may hang), or add `--no-daemon` to the command.

**RViz from the launch file vs. started by hand.** The launch starts RViz exactly like `ros2 run rviz2 rviz2 -d .../fsd_sim.rviz --ros-args -p use_sim_time:=false`, and prints RViz's output in the launch terminal. If only a hand-started RViz works:
1. Check that the relay's first line says `fsg-dv-sim 0.6.0 relay`. If it doesn't, the build is old: look for a nested `~/fsg-dv-sim/fsg-dv-sim` from unzipping, then rebuild.
2. Run `ros2 param get /rviz use_sim_time` (`ros2 node list` shows RViz's node name). It should print `False`.
3. As a workaround, use `ros2 launch fsd_sim_bringup isaac.launch.py rviz:=false` and start RViz by hand.

**RViz clock.** RViz now runs on the wall clock by default (`rviz_sim_time:=false`). It still draws the sim-stamped data correctly. The textbook sim-time setup has one weakness: RViz deletes all its TF data every time `/clock` jumps backwards, which shows exactly this error even though `tf2_echo` sees the transforms.

**Two `/clock` publishers.** The most common cause is a second node publishing `/clock`, for example:
* a lite simulator left running;
* `ros2 bag play --clock`;
* Isaac's own ROS 2 bridge.

The relay checks for this every 5 s and logs `another node also publishes /clock: ...` with the node name. Stop that node, or find it with `ros2 node list` and `ros2 topic info /clock -v`.

**The relay's own check.** The relay also logs `TF self-check OK/FAILED` about 4 s after Isaac connects. **OK** means the transforms are published, so any remaining problem is in RViz: click **Reset** (bottom left).

Checks in a second WSL terminal while the simulator runs:

```bash
ros2 topic info /clock -v              # exactly ONE publisher: fsd_isaac_relay (or fsd_lite_sim)
ros2 run tf2_ros tf2_echo map base_link
ros2 topic hz /lidar/points            # ~10 Hz of sim time
```

### Optional: Isaac's own ROS 2 bridge over Zenoh (`-SensorTransport ros2`)

Like the Go2 setup (`run_go2_sim_ros.bat`), Isaac can also publish the LiDAR, cameras and `/clock` itself with its bundled ROS 2 Jazzy, through the Zenoh router in WSL:

```powershell
... run_isaac.ps1 -SensorTransport ros2 -Mission trackdrive
```

* **Router check.** The launcher refuses to start without the router on `127.0.0.1:7447` (`-ZenohRouter` to change).
* **Environment.** It runs Isaac's `setup_ros_env.bat`, sets `RMW_IMPLEMENTATION=rmw_zenoh_cpp` and `ROS_DOMAIN_ID=18` (`-DomainId`), and sets `ZENOH_SESSION_CONFIG_URI`.
* **Session file.** It uses the Go2 session file `C:\robots\zenoh_session_isaac.json5` by default (`-ZenohConfig`). That file must use `tcp/127.0.0.1:...` for `connect` and `listen`, because on Windows `localhost` resolves to `::1`, which WSL cannot reach in mirrored mode.
* **What still uses TCP.** AS state, commands and vehicle dynamics still go over the TCP link. The relay then leaves `/clock` to Isaac (`publish_clock:=auto`).
* **Why TCP stays the default.** It carries the RTX LiDAR and both ZED streams at about 80 MB/s with no drops on this machine. The Zenoh path is useful when other Isaac-side ROS tools must join the same graph.
