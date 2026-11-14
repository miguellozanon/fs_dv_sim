# Perception

Two parts: **camera cones with YOLO** (first sections, Isaac Sim only) and, since Nov 2026, **LiDAR cones,
LiDAR + camera fusion, SLAM and sensor faults** (the section "LiDAR, fusion, SLAM, faults" below; lite backend,
tools and Isaac Sim).

# Camera cones: YOLO on the ZED camera

The car can find the cones itself: a YOLO model detects every cone and its colour in the ZED left image,
the depth image gives its distance, and the detections of many frames become a cone map the driver steers
from. Without it the driver gets the simulator's exact cone positions (ground truth).

```
ZED left image ──► YOLO (ONNX) ──► boxes + colour ─┐
ZED depth image ───────────────────────────────────┼─► 3D cone (map frame, TF) ──► cone map ──► /perception/cones ──► driver
camera pose (TF map → zed_left_camera_optical_frame) ┘
```

## Workflow (all from the launcher)

1. **Record training images.** Simulation → Isaac Sim, tick **Record camera images for YOLO training**, and
   drive a few runs: autocross and trackdrive with different **track layouts (seeds)**, a few setups. Two images
   per second are saved while the car moves (`datasets/cones/<run>/images`, `labels`). A few hundred images is
   a start, 2000+ is better.
2. **Install YOLO once.** Maintenance → **Install YOLO (Ultralytics) for the Windows Python** and **Install
   ONNX Runtime in WSL**.
3. **Train.** RL & Performance → **Train the cone detector (YOLO)** (YOLO11n: fast; YOLO11s: more accurate).
   It trains on the GPU with the Windows Python, checks itself on whole runs it did not train on, and writes
   `models/cones/<name>/model.onnx`, `model.json` and `results.md` (precision, recall, mAP per colour).
   Close Isaac Sim while it trains: both need most of the 8 GB GPU memory.
4. **Drive with it.** Simulation → Isaac Sim, **Cones for the driver: Camera (YOLO)**, pick the detector (or
   Newest). The dashboard shows the boxes on the camera image, the cones it mapped (rings on the map) and how
   good it is (below the camera).

## How the labels are made

`fsdsim.vision.labels`: every ground-truth cone is projected into the image with the camera pose (TF) and the
camera intrinsics (camera_info); its box is the projection of the base plate and the tip (FS Driverless
Specification sizes). The depth image decides what the camera really sees: a cone behind another cone, the
car or the image border is left out. Classes are the `fsd_sim_msgs/Cone` colours with the FSOCO names:
`blue_cone`, `yellow_cone`, `orange_cone`, `large_orange_cone`.

The recorder checks the camera model every 30 s: "depth confirms N % of the cones in view". It should be
close to 100 %. A low value means the rendered camera differs from `sensors.yaml` (mount, field of view), and
the boxes would be misplaced: report it with a saved launcher report.

### Camera timing and view (fixed after the first training, 30 Oct 2026)

The first recording (888 images) trained to only mAP50 0.63. Drawing its labels on its images showed why:

* **Isaac Sim delivers a camera image two rendered frames (≈ 67 ms) after the frame it shows.** The images
  were stamped with the time they were read, so the labels came from a pose 0.3–0.6 m further along the track
  and missed the cones at speed (exact while the car stood still). Images are now stamped with the time of the
  frame they show: `sensors.yaml` `camera.isaac_frame_lag: 2` (measured by fitting the labels to the cones in the
  images: lag 0.06–0.07 s at 4–9 m/s; 0.1 s overshoots).
* **The procedural car's main hoop stood 10 cm in front of the camera** and hid the middle of the image
  (fewer yellow labels: the right-hand cones were behind it). The hoop is now under the camera, as on the car.

Recordings carry `session.json` with `labels_version: 2`; the trainer skips older recordings (their boxes lag the
cones). With matching labels the recorder's "depth confirms" share should be close to 100 % (it was 57 %).

## What the car does with the detections

`fsdsim.vision.mapping`:

* **Position:** the depth of the cone body (lower middle of the box), pushed back by the cone radius to its
  centre; without depth, where the box's foot meets the ground.
* **Cone map:** detections of all frames are merged (nearest neighbour within 0.8 m, weighted mean, colour
  vote); a cone counts once it was seen twice. The map is cleared when a new run starts. The car pose comes
  from the simulator (no SLAM yet).
* **Driver:** the reference driver in cones mode (autocross, trackdrive) uses `/perception/cones` instead of
  `/ground_truth/cones`. Acceleration and skidpad, and the RL driver, follow the known path.

## Accuracy of YOLO (dashboard, `/perception/stats`)

Per frame, against the cones the camera really sees (same visibility test as the labels): **found** (recall),
**boxes that are cones** (precision), **colour right**, **position error**. For the map: mapped cones, false
cones (no real cone within 1 m), right colour.

## Files

| | |
|---|---|
| `ros2_ws/src/fsdsim/fsdsim/vision/` | labels, detector (ONNX Runtime, OpenCV DNN fallback), mapping, accuracy |
| `ros2_ws/src/fsd_sim_bridge/fsd_sim_bridge/cone_perception_node.py` | the detector node (`perception:=yolo`) |
| `ros2_ws/src/fsd_sim_bridge/fsd_sim_bridge/cone_dataset_recorder_node.py` | the recorder (`record_cones:=true`) |
| `tools/train_cone_yolo.py` | training + ONNX export (`--check` only checks the dataset) |
| `datasets/cones/`, `models/cones/` | recorded images, trained detectors |

By hand: `ros2 launch fsd_sim_bringup isaac.launch.py perception:=yolo cone_model:=<repo>/models/cones/<name>`
and `record_cones:=true`; `python tools\train_cone_yolo.py --name <name> --epochs 60` on Windows.

## LiDAR, fusion, SLAM, faults (`fsdsim.perception`, `fsdsim.faults`)

No ground truth goes into this chain; the ground truth is only used to print the accuracy.

| Step | Code | What it does |
|---|---|---|
| LiDAR cones | `perception/lidar_cones.py` | removes the car's own body (box from the CAD dimensions: the OS1 sees nose, tyres, wings), fits the ground plane, clusters the rest on a grid, keeps cone-sized clusters, colour from the stripe intensity pattern (lite backend) |
| Fusion | `perception/fusion.py` | LiDAR positions + camera colours (nearest camera detection within a gate); camera-only cones are added at short range |
| SLAM | `perception/slam.py` | pose from wheel speeds and gyro, corrected by matching each frame's cones to the map (rigid 2-D fit); the map keeps cones seen several times |
| Pipeline | `perception/pipeline.py` | sensor events in, `cones()`, `pose()`, `mapped_track()` out |
| ROS node | `fsd_sim_bridge/perception_node.py` (`perception`) | `/lidar/points`, `/imu/data`, `/vehicle/sensors`, camera cones → `/perception/cones`, `/slam/odom`, `/perception/map_stats` |

**Modes** (`perception:=`, launcher: Cones for the driver): `lidar`, `fusion`, `camera` (camera cones through
the map; weakest), and `yolo` (the older path above: camera cones with the ground-truth pose). `slam:=true`
also takes the car's pose from the node. In **Isaac Sim** the RTX LiDAR has no stripe intensities: `lidar` mode
leaves the cones without colour there (`isaac.launch.py` sets the node's `lidar_colours:=false`; guessed from the
RTX intensities the colours were right for 23 % of the cones) and the driver follows the boundaries by geometry
(autocross finished in 45.2 s, 0 cones, on the team PC). `fusion` gives coloured cones (YOLO, needs a trained detector).

**Explore, then race** (`race_after:=1 line:=racing`, trackdrive): the first lap follows the cones, the node
builds the map, the driver then plans on the mapped track (centre line from the driven lap and the cones; with
`line:=racing` its racing line, 0.25 m extra margin) and drives it from lap 2 on.

```bash
python3 tools/closed_loop_eval.py --mission autocross --perception fusion --slam
python3 tools/closed_loop_eval.py --mission trackdrive --laps 3 --perception lidar --slam --race-after 1 --line racing
ros2 launch fsd_sim_bringup lite.launch.py mission:=trackdrive perception:=fusion slam:=true race_after:=1 line:=racing
```

Results in the lite model (10 Nov 2026): fusion + SLAM on autocross: all 113 cones mapped, position error
0.04-0.10 m rms, pose error about 0.05-0.1 m, 0 cones hit. Explore then race: lap 1 about 40 s, then 18.7-19.6 s.

**Sensor faults** (`faults:=`, launcher: Sensor faults, `run_isaac.ps1 -Faults`, tools `--faults`):

```
<sensor>:<kind>[=<value>]@<start>[+<duration>]      sensor: lidar | camera | imu | gnss | wheels
lidar:dropout@20+3, lidar:thin=0.3@0, camera:latency=0.25@5, imu:bias=0.02@10, wheels:scale=1.05@0, gnss:jump=1.5@10+30
```
kinds: `dropout`, `freeze` (last message repeated), `latency=<s>`, `thin=<share>` (LiDAR points kept),
`bias=<rad/s>` (gyro), `scale=<factor>` (wheel speeds), `jump=<m>` (GNSS). The dashboard's sensor table shows
the result (Stalled, Low rate, Delay). In Isaac Sim the faults act on the rendered LiDAR and camera.

**Robustness (added 11 Nov 2026 after the run on the team PC).**
* Camera colours are matched to the LiDAR cones in the map frame, each placed with the pose at its own stamp, from
  the camera frames of the last 0.8 s. A late camera (0.25-0.5 s) no longer loses the colours.
* A frozen sensor (the same scan or detections repeated with new stamps) is skipped; odometry carries the pose.
  Before, a 2 s LiDAR freeze moved the SLAM pose by up to 25 m.
* The cones planner uses big orange cones near the car as the boundary (the timing line), and gives cones without
  colour the side of the path planned a moment ago. This bridges a few seconds without camera colours.

**Fixes from the Isaac Sim runs (11 Nov 2026, evening).**
* **Cones without colour, whole laps.** The planner follows each boundary from the car cone by cone (next cone
  1-7 m on, turning at most 70 deg, left and right in turn), then the old path rule for what is left. Before, the
  cones after the first 90 deg corner stayed unknown and the car went straight on. Autocross, trackdrive and three
  other seeds now finish in the lite model with no colours at all (`tests/test_pc_run_2026_11_11.py`).
* **Stamp order in the ROS node.** Isaac's relay publishes a whole render frame at once (~7 IMU, ~3 wheel-speed
  samples, maybe a scan) and the executor takes one message per topic in turn: wheel speeds were integrated with
  yaw rates of other times and scans applied before the odometry had reached them. The SLAM pose drifted 0.8 m in
  one lap (reproduced without ROS: 0.34 m mean against 0.05 m in order). The node now processes odometry, camera
  and scans in stamp order (at most 0.1 s held back) and detects cones on a second thread, so odometry is never
  blocked by a scan.
* **The driver's pose with SLAM.** /slam/odom reached the driver 0.15-0.27 s after its stamp. The driver now
  integrates wheel speeds and gyro itself and takes only the SLAM correction (`fsdsim/perception/fresh_pose.py`,
  `demo_driver fresh_pose:=auto`, on with `/slam/odom`). Carrying the late pose forward on the steering arc was
  tried first and is worse: the racing line left the track.
* **SLAM error in the map line** is now the SLAM pose at the ground truth's own stamp; it was the newest pose
  against a ground truth up to one frame older, which alone read as 0.3 m at 10 m/s.
* New fields in `/perception/map_stats` (dashboard map line / JSON): `wheel_lag_ms`, `pose_lag_ms`,
  `correction_max_m`, `correction_max_deg` (how far each scan moves the pose), `odometry_err_1s_m` (wheel speeds
  + gyro alone after 1 s, against the ground truth) and, under `lidar_timing`, `bias` (the cone offset left in
  the car frame: LiDAR mounting / centre estimate).

**Known limits.** Camera-only SLAM is weak (short range, low gain); a camera freeze in camera mode leaves the car
without cones. A gyro bias of 0.02 rad/s moves the SLAM pose by metres (no bias estimation). In Isaac Sim the
SLAM pose is still 0.2-0.4 m off (lite 0.05 m): the RTX LiDAR cones are 0.08-0.15 m off with the true pose
(lite 0.01-0.03 m; no offset, no timing error, `bias` ~0), so each scan corrects the pose by 0.1-0.3 m (lite
0.01-0.03 m). Odometry alone is off by 0.06-0.9 m after 1 s in both. Trackdrive with explore then race finishes,
with a few cones on the racing line (3-9 in a 10-lap run).

## Limits

* **Simulator images only.** A detector trained on Isaac Sim images works in the simulator. For the real car,
  add real images (e.g. the FSOCO dataset, available to Formula Student teams) to `datasets/cones/` in the same
  YOLO format and train again.
* **Licence.** Ultralytics YOLO is AGPL-3.0: fine for the team; code that includes it and is published must be
  published under AGPL too. The inference code here only needs the exported ONNX model and ONNX Runtime.
* **Speed.** YOLO11n at 640 × 384 on the CPU in WSL: about 20–50 ms per image (shown in the dashboard). With
  `onnxruntime-gpu` it can use the GPU, which Isaac Sim already loads heavily.
* Checked on the real setup: training with Ultralytics 8.4 on Windows Python 3.14 (RTX 5060), the exported
  model in the car's detector code (ONNX Runtime and OpenCV DNN give the same boxes; 18–34 ms per image on the
  CPU). Not yet: driving from the camera cones in Isaac Sim.
