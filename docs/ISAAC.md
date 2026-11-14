# Isaac Sim backend

The Isaac backend renders the **Ouster OS1 LiDAR** with RTX ray tracing and the
**ZED stereo camera** as real images. Vehicle dynamics, AS logic and the
proprioceptive sensors still come from `fsdsim`, so both backends behave the same.

* **Windows 11 + WSL2 (your setup):** see **[WINDOWS_WSL.md](WINDOWS_WSL.md)** for step-by-step instructions.
* **Native Linux:** same idea. Run `ISAAC_SIM=~/isaacsim ./isaac/run_isaac.sh --mission skidpad` next to `ros2 launch fsd_sim_bringup isaac.launch.py`.

Targeted versions: **Isaac Sim 6.1** (Python 3.12), Isaac Lab 3.0 environments,
ROS 2 Jazzy. Fallbacks for Isaac Sim 5.x / 4.5 APIs are kept in `isaac_compat.py`.

## How it fits together

```
run_isaac.py  (Isaac Sim python, Windows or Linux)
  fsdsim.Simulation   car pose, AS state, IMU / GNSS / wheels, timing
  scene.py            USD stage: ground, instanced cones, car, sensor mounts
  isaac_compat.py     version-specific Isaac calls (RTX LiDAR, OmniGraph, stepping)
  sensor_streams.py   RTX LiDAR (GenericModelOutput) + camera annotators -> arrays
  fsdsim.link         TCP client  ──────────────►  isaac_relay (ROS 2 node, WSL/Linux)
                                  ◄──────────────  /control/command, /as/* services
```

**Sensor transport**
* **`tcp` (default).** LiDAR and camera data travel through the link and the relay publishes them. Isaac needs no ROS installation.
* **`ros2`.** Isaac's OmniGraph ROS 2 bridge publishes LiDAR, cameras and `/clock` over DDS. Use it on native Linux or NVIDIA's Pixi workspace. AS state and commands still use the link.

## Isaac Sim 6.1 specifics

* **RTX LiDAR creation.** `isaacsim.sensors.experimental.rtx.Lidar.create(config="OS1", variant=...)` is called with:
  * `accumulate_outputs=True`, so the sensor outputs full scans;
  * `tick_rate` equal to the scan rate (10 Hz), as NVIDIA requires for complete output.

  The vendor asset nests the `OmniLidar` prim, so the backend searches for it before attaching readers.
* **LiDAR readout.** A Replicator **Writer** (`FsdGmoWriter`) is attached to `LidarSensor` and receives every `GenericModelOutput` frame. This is NVIDIA's recommended pattern: polling `get_data()` each frame drops scans under multi-tick rendering. Polling is kept as a fallback.
  * The data is parsed with `parse_generic_model_output_data`.
  * Spherical and Cartesian output are both handled.
  * `ring` comes from beam elevation (0 = top beam) and `t` from `timeOffsetNs`.
* **Cameras.** USD cameras with the ZED baseline, FOV and resolution, read with Replicator `rgb` and `distance_to_image_plane` annotators.
* **Fallback.** If no RTX scan arrives within 3 s of sim time, the log prints diagnostics (writer calls, empty reads, zero-element frames) and switches to the analytic OS1 model. ROS still gets a cloud.
* **Motion BVH.** It is **off by default.** NVIDIA's docs say it can only be chosen when the app starts. Turning it on mid-run caused endless `failed to create Hydra Engine thread for viewport` warnings and a crash. Enable it with `--motion-bvh`, which passes `SimulationApp({"enable_motion_bvh": True})`.
  * It is only needed for LiDAR motion compensation (rolling-shutter skew at speed).
  * It costs VRAM, which matters on an 8 GB laptop GPU.
* **Time.** fsdsim is advanced to the time of the frame about to be rendered, so all stamps share Isaac's simulation time. The relay publishes it as `/clock`.

## The 64-channel OS1

Isaac Sim 6.1 ships OS1 profiles with 32 and 128 channels only; your log lists them at startup. With `lidar.isaac.variant: auto` and `channels: 64`:
1. The backend uses `OS1_REV7_128ch10hz1024res`, which has the same 45° vertical FOV as the OS1-64.
2. It **keeps every 2nd beam**, renumbering rings 0-63, so the point density and ring layout match an OS1-64 with uniform beam spacing.

For a beam-exact match:
1. Copy the OS1 USD asset.
2. Set its emitter elevation and azimuth arrays to your unit's `beam_altitude_angles` / `beam_azimuth_angles`, from the sensor's metadata JSON.
3. Load it with `Lidar.create(usd_path=...)` in `isaac_compat.create_rtx_lidar`.

## ZED camera options

* **`camera.isaac_mode: native`** (default). The cameras described above. Depth is ground-truth renderer depth.
* **`camera.isaac_mode: stereolabs_extension`**. Stereolabs' ZED extension streams the simulated camera into the real ZED SDK / ZED ROS 2 wrapper, so stereo depth matches the car. The ZED SDK must be installed on the machine running Isaac.
  1. Place the ZED asset under `/World/Car/zed_mount`.
  2. Follow Stereolabs' "ZED Isaac Sim Extension" guide.

## Using your car's CAD

Set `usd_path` in `vehicle.yaml` to a USD export of your car, with its origin at
the CoG projected to the ground, x forward and z up. The model is visual only;
it does not change the dynamics. `usd_wheel_path` (with `{name}` for fl / fr / rl / rr) replaces the tyre
cylinders by wheels that steer and spin. `tools/cad_to_usd.py` makes both from the LR26 CAD (STEP, STL or
OBJ) in the right frame; see [CAD.md](CAD.md).

**Materials (13 Nov 2026).** The converted USD carries only a flat colour per part, which the RTX renderer does
not shade like paint, carbon or rubber. `isaac/scene.py` binds real materials when it loads the car: red paint
(glossy), carbon, metal for the rest and for every rod-shaped part (wishbones, push / pull and tie rods, which the
colour guess had painted red), and on all four wheels the same rubber tyre with a metal hub (one wheel had kept a
light grey from the CAD file). Nothing in `assets/lr26` is changed.

**Reset simulation (13 Nov 2026).** After a reset the simulation clock starts at 0 again; the ZED camera's next
frame was still scheduled for the old time, so no image came for as long as the run had lasted and YOLO drove
blind. The camera schedule now restarts with the clock, and the reference driver starts the new run fresh.

## Isaac Lab 3.0

* **Running in an Isaac Lab env.** `run_isaac.py` only needs a Python with Isaac Sim importable (`from isaacsim import SimulationApp`). It runs from an Isaac Lab 3.0 environment as well as from the standalone `python.bat`; pass `-Python <env>\python.exe` to the launcher.
* **Quaternion convention.** Isaac Lab 3.0 changed its quaternion order to (x, y, z, w). This project writes USD directly and is not affected.
* **Kit-less mode.** Not applicable here, because the LiDAR and cameras need RTX rendering.
* **Possible next step.** An Isaac Lab task that wraps `fsdsim` (same tracks, AS logic and vehicle model) for batch evaluation or RL-trained controllers.

## First-run checklist

1. **Isaac log.** The `[fsd-isaac]` lines show the Isaac version, the LiDAR API path and variant, the camera resolution and `link: connected to relay`.
2. **Topic rates (WSL).**
   * `ros2 topic hz /lidar/points`: about 10 Hz.
   * `ros2 topic hz /zed/zed_node/left/image_rect_color`: about the camera rate.
3. **Alignment in RViz.** LiDAR cone returns must sit on the ground-truth cone markers.
   * If the cloud is rotated, set `lidar.isaac.yaw_offset_deg`.
   * If it is offset, check `lidar.mount`.
4. **Point fields.** In `ros2 topic echo /lidar/points --once --field fields`, `ring` should span about 30-63 on flat ground; upper beams see only sky.
5. **Drift.** Every 10 s, Isaac's log prints the Isaac-fsdsim drift, measured from the offset at start-up. It should stay near 0 ms. fsdsim is the time authority and every stamp uses its time, so a small constant value is harmless; only a growing drift indicates a problem.
6. **Real-time factor.** It is also printed every 10 s. Below 1.0, the simulation runs slower than wall-clock time. Everything stays consistent because all stamps are sim time; only the 'live' feel suffers. On an 8 GB laptop GPU these options help:
   * `--camera-hz 10`;
   * `--camera-streams left`, which drops depth;
   * a lower ZED resolution in `sensors.yaml`.

## Added in Nov 2026

* **Attitude:** the car body rolls, pitches and heaves (`fsdsim.chassis`); the wheels stay on the ground. The
  RTX LiDAR and the ZED are children of the body, so they tilt with it.
* **Perception:** `isaac.launch.py perception:=fusion slam:=true` runs the LiDAR cone detector on the RTX point
  cloud with colours from YOLO (`/perception/frame_cones`); `perception:=lidar` works without a detector but
  gives no colours. `line:=racing` and `race_after:=N` as in the lite backend.
* **Sensor faults:** `run_isaac.ps1 -Faults "lidar:dropout@20+3"` (launcher: Sensor faults) acts on the rendered
  LiDAR and camera frames and on the IMU / GNSS / wheel events.

## Timing and load (found in the run on the team PC, 11 Nov 2026)

* **Relay address (the main fix).** With WSL in mirrored networking, Isaac reached the relay over the PC's LAN IP
  (`192.168.x.x`): 35-80 MB/s with pauses of 0.3-0.6 s, frames dropped and commands up to 2 s old. Over
  `127.0.0.1` the same runs show 0 frames dropped and a command age of at most 67 ms. `run_isaac.ps1` now uses
  `127.0.0.1` when WSL shares the Windows IPs (mirrored); the log line `[fsd] ROS 2 relay : tcp://...` shows it.
* **Command age.** Every 10 s Isaac's log prints `command age median ... ms, max ... ms`: how old the driver's
  command is when the car gets it (Isaac's sim time minus the command stamp). Healthy: about 70-100 ms (one render
  frame each way plus the driver). Seconds mean the ROS side in WSL is behind; the car then steers on old data and
  leaves the track even on ground truth. Check the launcher's **Maintenance → Diagnose WSL** for busy processes.
* **Where the remaining 0.3-0.5 s spikes came from (11 Nov 2026, evening).** New log fields split the path:
  Isaac's line now ends with `state queued before sending max ... ms (longest write ... ms, ... MB)` and the
  relay's `link` line with `state on the way from Isaac up to ... ms longer than the fastest`; the driver logs
  `timing: ... longest wait for /clock ..., pose carried forward up to ...`. On the team PC a write of a few kB
  blocked for 0.4 s: the Windows -> WSL socket was full of images and clouds (~23 MB/s) that no ROS node read.
  Four changes:
  1. **Sensor demand.** Every second the relay tells Isaac whether any ROS node subscribes to the full camera
     images or the full LiDAR cloud (`sensor_demand`); if not, Isaac sends 320 px images and 6000-point clouds,
     enough for the dashboard. YOLO, fusion, RViz or the recorder switch it back to full within a second. The
     relay logs `Isaac sends full / small ... images and full / thinned ... LiDAR clouds`.
  2. **LiDAR preview.** The dashboard reads `/lidar/points/preview` (x, y, z, 6000 points per scan) instead of
     the ~1 MB cloud: with the full cloud Zenoh delivered odometry and `/clock` to every node up to 0.4 s late
     (reproduced without Isaac; with the preview: none). `/lidar/points` is only built when something subscribes.
  3. **Driver.** In Isaac the reference driver ticks on the wall clock (`tick_clock:=steady`; a late `/clock`
     held its sim-time tick back), keeps only the newest odometry and, when no odometry came for over 80 ms,
     carries the pose forward on its steering arc for the missing time (x the measured real-time factor).
  4. **State connection.** Isaac opens a second TCP connection for state (pose, IMU, wheels, AS) when the relay
     offers it (`link: state ... on its own connection` in the Isaac log), so state never waits behind a 5 MB
     camera frame; the relay skips sensor frames more than 0.25 s older than the state it already published.
     An older relay or Isaac falls back to the single connection.
  Result: Isaac's `state queued before sending` at most 16 ms (was 0.3-0.44 s), command age at most 67 ms with full
  images; trackdrive on ground truth with the racing line 10 laps at 18.85 s, 0 cones (before: laps up to 40 s,
  then off the track); explore then race with fusion + SLAM 18.82-18.88 s (3 cones). Runs: [TEST_PLAN.md](TEST_PLAN.md).
* **Driver latency.** The reference driver predicts its pose 70 ms ahead (`isaac.launch.py driver_latency:=0.07`):
  Isaac renders at 30 Hz and a command acts one frame later. Without it the skidpad failed from the third lap.
  `tools/closed_loop_eval.py --rate 30 --cmd-latency 0.067` reproduces this without Isaac.
* **Load on the relay.** The dashboard now watches small image previews (`<image topic>/preview`, 5 Hz, 320 px)
  instead of the full 1280 x 720 colour and depth images; full images are only converted when something subscribes
  (YOLO, RViz, the recorder); when the relay falls behind it skips old images and scans (pose, IMU, wheels and AS
  state are never skipped) and says so in its `link ...` log line. ONNX Runtime (YOLO) uses 2 threads
  (`FSG_ORT_THREADS`).
* **RTX LiDAR stamps.** A scan read after a render shows the car `lidar.isaac.frame_lag` (2) renders earlier, as the
  camera does; the stamps are moved back by that. The first scan logs whether the RTX output has per-point times.
  With perception and the ground truth on, the dashboard's map line shows **LiDAR timing**: the stamp shift that
  fits the true cones best (should be 0 ms; else set `isaac.launch.py lidar_time_offset:=`).
