# Test plan: what only the team PC can check

Everything that runs without Isaac Sim, MATLAB, a GPU or the real ROS 2 install was checked on 9 and 10 Nov 2026
(see "Already checked" at the end). Sections 7 and 8 are new on 10 Nov (chassis, racing line, perception, SLAM,
faults, launcher options). This page lists the runs left for the PC, in order, with what to expect and
what to send back. After each block, the launcher's **Share with Claude > Save report to the project** (Simulation
page, right column) writes `shared\LATEST.md` and the logs; Claude reads that folder directly.

## 1. WSL, no simulator (5 min)

```bash
cd /mnt/c/formula_student/fsg-dv-sim
python3 -m unittest discover -s tests                 # 258 tests, OK (a few skipped without PyTorch / a real .mf4)
python3 tools/check_install.py                        # ready: tools yes | ROS 2 simulators yes | Isaac Sim yes
python3 tools/regression.py                           # 33 of 33 ok (2-3 min); send results/regression_*/report.md if not
python3 tools/closed_loop_eval.py --strict            # acc 4.77 s, skidpad 5.38 s, autocross 44.85 s, trackdrive 10 laps clean
python3 tools/closed_loop_eval.py --rate 20 --strict  # same, driver at 20 Hz: all clean
```
Send: the last lines of each (or a saved report).

## 2. ROS 2 lite backend in WSL (10 min)

`fsgbuild` first (new files changed nothing in the packages' setup, but the build stamp should be fresh), then in
the launcher **Lite**, or:

```bash
fsgsim && ros2 launch fsd_sim_bringup lite.launch.py mission:=skidpad
fsgsim && ros2 launch fsd_sim_bringup lite.launch.py mission:=acceleration \
    track_file:=$(ros2 pkg prefix fsd_sim_bringup)/share/fsd_sim_bringup/tracks/acceleration.csv
```
Expect: skidpad FINISHED 0 cones (`ros2 topic echo /as/status --once`: `cones_hit: 0`, about 5.4 s); acceleration
from the CSV FINISHED, stopped in the exit lane (before 9 Nov it never started the timer and ended in EBS).
Dashboard on :8080: car, laps, no penalties.

## 3. Isaac Sim with the LR26 CAD (the main open item)

Launcher: **Isaac Sim**, mission **Skidpad**, reference driver, ground-truth cones, RViz on. Then the same with
**Trackdrive**.

**Isaac log** (`[fsd-isaac]` lines):
* five lines `USD model: .../assets/lr26/lr26_body.usda` and `..._wheel_fl|fr|rl|rr.usda`; no `USD file not found`
* how long from start to `running:` (the body is a 45 MB text USD), the real-time factor (was 0.75-0.8 with the
  simple car) and `RTX LiDAR scans` growing

**Viewport** (screenshots: side, top, front three-quarter, and the car at the skidpad crossing):
* wheels exactly at the hubs, turning when the car steers and spinning when it drives; tyres just touching the
  asphalt (7 mm into it is expected: tyre deflection)
* body neither floating nor sunk (floor about 4 cm above the ground); car pointing forward (nose = driving direction)
* colours: red bodywork, carbon wings, black tyres. If the car is all grey or white, the renderer ignores the
  display colours: tell Claude (materials then go into the USD)
* a thin carbon post under the camera (the main-hoop filler) and the OS1 sitting on top of the nose

**ZED** (dashboard camera panel or RViz image): the car fills the bottom third: nose, both front tyres, push rods.
Compare with `docs/img/cad_zed_view.png` (ray-cast of the USD). Cones in front must be visible above the nose.

**LiDAR** (RViz, `/lidar/points`, decay 0):
* `ros2 topic hz /lidar/points` about 10 Hz
* returns on the car itself: expected behind the sensor (front tyres at about +-130 deg, helmet and rear wing at
  180 deg) and maybe a ring on the nose just below the sensor. Nothing in the front half closer than the cones
* cone returns still on the ground-truth cone markers
* screenshot of RViz from above with the car at standstill

**Results**: skidpad and 10-lap trackdrive finished with 0 cones (the reference driver at 50 Hz; Isaac adds
latency: report any cone).

## 4. Camera cones (YOLO) with the new car body

Launcher: Isaac Sim, **Autocross**, cones = **Camera (YOLO)**, detector `cones_301026`. The detector was trained
with the old simple car in the bottom of the image. Report: whether the car finishes, cones hit, and the detection
accuracy in the dashboard. If worse, record a new set (option "Record camera images for YOLO training", one
trackdrive) and retrain (RL & Performance > Camera cones (YOLO)).

## 5. MATLAB (15 min)

```matlab
cd C:\formula_student\fsg-dv-sim\lapsim\LR26_LapSim; startup_lapsim
run_lapsim                                         % 4.662 / 5.154 / 49.944 / 1526.97 s, 164.35 points
comp = competition_fsa2025_real(); [R, tbl, total] = lts_run_all(lr26_params(), comp, struct('verbose', true))
                                                   % autocross 49.26 s, endurance lap 96.33 s, 163.32 points
c0 = competition_fsa2025();
o = lts_endurance_power(lr26_params(), c0.events.endurance, 5.5)    % cap_kW 17.98
o = lts_endurance_power(lr26_params(), comp.events.endurance, 5.5)  % cap_kW 17.46
```
And the MATLAB half of a parameter study: launcher job "Parameter study" (c_z 3.5, 4.15, 4.8), then in MATLAB the
`run('...\matlab_study.m')` line it prints. Expect `matlab_results.csv` next to it with the same times as the report.
These were run in GNU Octave here (same numbers); the table code at the end of `matlab_study.m` needs MATLAB.

## 6. Optional

* Windows GPU tests: launcher RL & Performance > "Check the GPU code (PyTorch tests)" (CUDA; run on the CPU here).
* A real imported circuit in Isaac: `python3 tools/track_import.py lapsim/LR26_LapSim/tracks/FSA2025_endurance_real.csv --closed --cones tracks_out/fsa_end.csv`,
  then mission trackdrive with that track file. The exploration-style driver laps it in about 215 s and touched
  2 cones in the 3.6 m hairpin in the lite check.

## 7. New on 10 Nov: lite backend from the launcher (10 min)

`fsgbuild` first (a new node `perception` was added to `fsd_sim_bridge`; the launcher builds by itself).
Launcher → **Lite**, each with RViz off or on:

| Run | Expect (dashboard) |
|---|---|
| Autocross, cones **LiDAR + camera**, SLAM ticked | finished, 0 cones; under the map "Cone map (LiDAR + camera): 113 cones, 100 % ... pose from SLAM, error about 0.05-0.1 m"; sensor table has a Delay column |
| Trackdrive, **LiDAR**, SLAM, race after **1 lap**, path **Racing line** | lap 1 about 40 s, later laps 18.7-19.6 s, 0 cones; the driver log says it switched to the mapped track |
| Autocross, ground truth, **Racing line** | about 19.5 s (centre line 20.1 s), 0 cones |
| Skidpad, handling **Double track** | about 5.39 s, 0 cones |
| Autocross, LiDAR + camera, SLAM, faults `lidar:dropout@14+1.5` | LiDAR row "Stalled" for 1.5 s, still finished with 0 cones |

On the PC the real-time factor is 1; here (2 CPU cores) the ROS runs were slower than real time and the LiDAR
rate showed "Low rate". Report if that also happens on the PC.

## 8. New on 10 Nov: Isaac Sim (the open items)

1. **Attitude.** Any mission: the body should lean out in corners (under 1.5 deg at 1.5 g), dive on braking and
   sit a few mm lower at speed; wheels stay on the ground and do not sink or float. `ros2 topic echo
   /imu/data --once` at standstill: z acceleration about 9.81, x and y about 0. Screenshot in a corner.
2. **LiDAR cones on the RTX LiDAR.** Autocross, cones **LiDAR**, SLAM off. Expect cones in the map panel without
   colours (grey / unknown). The cones planner cannot drive a whole lap on colourless cones, so leaving the track
   here is expected; what matters is the "Cone map" line (cones found, position error, own-car points removed). Send the "Cone map" line from the dashboard and the `fsd_perception` log lines
   (own-car points removed, ms per scan).
3. **Fusion.** Autocross, cones **LiDAR + camera**, detector `cones_301026`, SLAM off, then SLAM on. Expect
   coloured cones, finished. Send the "Cone map" line (cones found, position error, colour right, pose error).
4. **Racing line.** Autocross and trackdrive, ground truth, path **Racing line**: lap times against the centre
   line, cones hit (Isaac adds latency; the margin is 0.40 m).
5. **Faults.** Autocross, LiDAR + camera, faults `lidar:dropout@14+1.5`: the Isaac log prints the fault plan
   at start; the LiDAR row stalls for 1.5 s.

If Isaac Sim does not start at all with the new code: the Isaac log (`[fsd-isaac]` lines and the Python
traceback) is enough; `chassis.attitude=false` as a vehicle change switches the body motion off.

## Checked on the team PC (11 Nov 2026, RTX 5060 Laptop, Isaac Sim 6.1, by Claude through the launcher)

**Lite backend** (all after the fixes of the day; 0 cones unless noted): skidpad 5.34-5.37 s per lap; autocross
racing line 19.50 s; trackdrive racing line 10 laps at 18.84 s; seed 13 clean; the fault cases clean; explore then
race 38.6 s then 18.78 s; no real-time warnings (the timer catches up now). Regression run: 37 cases.

**Isaac Sim**, final runs with all fixes (S4 / S5 in the session log):

| Run | Result |
|---|---|
| Skidpad, ground truth | finished, 0 cones; command age max 67 ms; real-time factor 0.84-0.92 |
| Autocross, ground truth, racing line | finished (t = 31.1 s), command age max 67 ms |
| Trackdrive, ground truth, racing line | 10 laps, 18.85 s each (lite 18.84 s), 0 cones, 0 off course |
| Autocross, LiDAR + camera (YOLO), SLAM | finished, 0 cones; command age max 67 ms with full images (34 MB/s) |
| Trackdrive, LiDAR + camera, SLAM, explore then race, racing line | finished: lap 1 41.3 s, lap 2 21.8 s, then 8 laps at 18.82-18.88 s; 3 cones, 0 off course; command age max 100 ms |
| Autocross, YOLO only (camera cones) | finished, lap 48.1 s, 0 cones; command age max 67 ms |
| Autocross, RL driver | finished, lap 19.62 s, 0 cones (before the link fixes 20.7 s, 5 cones) |
| Earlier in the day (driver latency fixed, before the link fixes) | acceleration, EBS test, skidpad double track, fusion with LiDAR dropout and with the camera 0.25 s late: all finished |
| Autocross, LiDAR only (no colours) | finished, 45.2 s, 0 cones (evening fix: boundaries followed by geometry; before: off the track at 17.8 s) |

What was wrong on the PC and is fixed (details: [ISAAC.md](ISAAC.md) "Timing and load"):

1. Relay address: the LAN IP in mirrored networking paused the link for 0.3-0.6 s; now `127.0.0.1`.
2. Driver latency: the driver predicts its pose 70 ms ahead (`driver_latency`); skidpad failed without it.
3. Images and clouds nobody read (~23 MB/s) filled the Windows -> WSL socket: Isaac now sends full ones only
   while a ROS node subscribes (`sensor_demand`): 2.7 MB/s on ground truth, real-time factor up from 0.77 to ~0.85.
4. State (pose, IMU, wheels, AS) has its own TCP connection, so it never waits behind a 5 MB camera frame
   (state queued at most 16 ms, was up to 0.44 s).
5. The dashboard reads a thinned LiDAR cloud (`/lidar/points/preview`); the full one delayed Zenoh for every node.
6. The driver ticks on the wall clock in Isaac, keeps only the newest pose and carries a late pose forward.
7. Lite: racing line on ground truth, the real-time timer, LiDAR / IMU speed-ups, the cones planner with big
   orange cones and unknown colours, the frozen-frame watchdog, camera / LiDAR timing (see the tests in
   `tests/test_pc_run_2026_11_11.py`).

Fixed in the evening (second round on the PC, `tests/test_pc_run_2026_11_11.py`):

* The late `/clock` in the first table came from the single Isaac -> relay connection (before item 4), not from
  Zenoh: with the state connection the driver waits at most 100-120 ms for `/clock` (2-3 frames).
* LiDAR-only cones in Isaac: no colours from the RTX intensities (they were right for 23 %), and the cones
  planner follows colourless boundaries by geometry. Autocross finished.
* SLAM: the perception node handles odometry, camera and scans in stamp order (the round-robin order of a frame's
  messages made the pose drift 0.8 m per lap), detects cones on its own thread, and the SLAM error shown is
  measured at the ground truth's stamp. The driver integrates wheel speeds + gyro itself and only takes the SLAM
  correction (its pose was 0.15-0.27 s old).
* Camera "Low rate" at 5-10 Hz: the launcher tells the dashboard the rate Isaac renders at (`camera_hz`). Needs a
  restart of `run.bat` (the launcher itself changed).
* Every node printed a traceback (`rcl_shutdown already called`, a second Ctrl+C in the dashboard) when a run
  stopped: shut down quietly now.

Still open:

* SLAM in Isaac 0.2-0.4 m (lite 0.05 m): the RTX LiDAR cones are noisier (0.08-0.15 m with the true pose, no
  bias, no timing error), so each scan moves the pose by 0.1-0.3 m. Explore then race finishes 10 laps at
  18.77-18.80 s with 3-9 cones on the racing line. Ideas: weight cones by range, a lower SLAM gain with a better
  odometry model, or more margin on a SLAM map.
* Not run: MATLAB, GPU training (RL, YOLO), headless Isaac, RViz on the PC.

## Already checked (10 Nov 2026, Linux, no Isaac / MATLAB)

* 258 unit tests; the regression run (33 cases); the lite backend through ROS 2 with fusion + SLAM, racing line,
  explore then race, a camera latency fault (dashboard map line and Delay column read back over HTTP)
* launcher page in a headless browser: new controls, fault check, no console errors; offline demo with the
  racing line
* `isaac/scene.py` with roll / pitch / heave on an OpenUSD stage and the fault path of `run_isaac.py` with the
  stand-in for Isaac (same link protocol). **Not** run: Isaac Sim itself, the RTX LiDAR point cloud through the
  detector, YOLO fusion, MATLAB, CUDA, Windows.

## Already checked (9 Nov 2026, Linux, no Isaac / MATLAB)

* unit tests (PyTorch CPU included), pyflakes, JS / YAML / JSON / XML / shell syntax
* every tool end-to-end: closed-loop eval (all events, 4 seeds, map and cone modes, 20 / 25 / 50 Hz, track files),
  parameter study (incl. RL driver), endurance strategy, track import (real FSA tracks driven), CAD inspect, CAD
  to USD (round trip of the shipped car), MF4 summary and log correlation (synthetic log), LapSim CSV export,
  LionsLTS and LapSim importers, link check, RL evaluate / universal training / spec sweep, YOLO model load
* the offline dashboard and the launcher in a headless browser (no console errors)
* ROS 2 Jazzy (RoboStack) build of the workspace; lite launch on all missions; isaac.launch.py with a stand-in for
  Isaac (same link protocol, the real `isaac/scene.py` on an OpenUSD stage): relay, driver, dashboard, skidpad clean
* `isaac/scene.py` with the LR26 USD built in OpenUSD: geometry, wheels, sensor mounts (see docs/CAD.md)
* LapSim MATLAB code in GNU Octave: baseline equals the MATLAB results; the new real-track and endurance-power files
