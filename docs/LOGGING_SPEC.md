# Logging and correlation spec (LR26 test days)

What to log on the car, how, and which test gives which model parameter, so that the Performance Lab and the
driverless simulator can be checked against the real car. Today they are only checked against each other and
against MATLAB: the team's only car log (D1F58) has inverter channels but no vehicle speed, GPS or IMU.

**For whom:** the data / electronics people who set up the logger, the test-day lead, and whoever loads the logs
into the Performance Lab (**Logs** page; [PERF_APP.md](PERF_APP.md)).

## 1. What we want to learn

| Model parameter (LapSim name) | Today | Test that measures it | Lab tool |
|---|---|---|---|
| Driver utilisation `util_x`, `util_y` per event | assumed (0.8–1.0) | any fast lap, skidpad, autocross | Logs → **Identify** → *Grip use (ggV)* |
| Lateral grip `muy` (with the driver at the limit) | assumption | steady-state circle, increasing speed | *Steady cornering* |
| Longitudinal grip / traction `mux` | assumption | straight-line acceleration and braking | *Grip use (ggV)* (its longitudinal part) |
| Rolling resistance `f_roll`, drag `c_w·A` | assumption | coast-down | *Coast-down* |
| Downforce `c_z` (only roughly) | CFD | steady circle at two speeds; coast-down with / without wing | *Steady cornering* |
| Motor / inverter efficiency `eta_motor`, map | done (D1F58) | any driving with inverter channels | *Powertrain* (exists) |
| Endurance energy per lap | model only | endurance-like stint | Logs → *Correlate*, Energy page |
| Speed trace of a lap | model only | autocross / endurance lap | *Correlate* (exists) |
| Track geometry | documentation tables | GPS of a driven lap, or a walk of the cones | Logs → **Track from GPS** |

## 2. Channels

**Must have** (no correlation without them):

| Quantity | Unit | Rate | Accuracy | Sign / note |
|---|---|---|---|---|
| Time | s | – | one clock for all channels | all channels in one file, same time base |
| Vehicle speed over ground | km/h or m/s | ≥ 50 Hz | ±0.2 km/h | from GNSS (Doppler) or GNSS/INS. Wheel speed is **not** enough (slip, tyre radius) |
| Longitudinal acceleration ax | m/s² or g | ≥ 100 Hz | ±0.02 g | + forward. Body frame, gravity removed or the sensor mounted level |
| Lateral acceleration ay | m/s² or g | ≥ 100 Hz | ±0.02 g | + left (ISO 8855). Say so if your sensor uses + right |
| Yaw rate r | deg/s | ≥ 100 Hz | ±0.5 deg/s | + counter-clockwise (left turn) |
| GNSS latitude, longitude | deg | ≥ 10 Hz | ≤ 0.5 m (RTK: 2 cm) | WGS84; for the track geometry |
| Steering wheel angle (or rack position) | deg | ≥ 100 Hz | ±0.5 deg | + left; give the steering ratio |
| Motor speed per motor | rpm | ≥ 100 Hz | – | already logged |
| Motor torque (actual) per motor | Nm | ≥ 100 Hz | – | already logged |
| DC bus voltage and current (accumulator) | V, A | ≥ 50 Hz | – | for energy; already logged per inverter |
| Brake pressure front, rear | bar | ≥ 100 Hz | ±1 bar | for coast-down windows and brake balance |
| Throttle / APPS | % | ≥ 50 Hz | – | driver input |

**Should have** (better identification):

| Quantity | Why |
|---|---|
| Four wheel speeds | slip ratio, tyre radius check against GNSS speed |
| Heading (GNSS/INS) and sideslip angle β | balance, transient-model check (sideslip is what the transient lap and the OCP show) |
| Damper potentiometers (4) | ride height → aero; load transfer check |
| Tyre temperatures, pressures (TPMS) | explains grip differences between runs |
| Ambient temperature, air pressure, wind | air density `rho`, coast-down wind correction |
| Accumulator state of charge, cell temperatures | energy, derating |

**Driverless runs** (rosbag, for the DV simulator): the same vehicle channels as topics, plus the planner's
reference path and speed, the controller commands (steering, acceleration), the cone detections and the
localisation estimate. Record the whole run with `ros2 bag record` (sensor raw data only when there is space).

## 3. File and naming

* **Format:** MF4 (the data logger) or CSV with a header line and units in the names (`speed_kmh`, `ax_g`). The Lab
  reads both. One file per run.
* **Name:** `<date>_<venue>_<test>_<run>.mf4`, e.g. `2027-03-14_Hockenheim_coastdown_03.mf4`.
* **Run sheet** (one line per run, CSV or the team sheet): file name, driver, test, setup name (the Lab's saved
  setup), mass with driver and ballast, tyre set and pressures cold/hot, wing configuration, ambient temperature,
  wind, track surface (dry / damp), remarks (cone hit, spin, aborted).
* **Storage:** team drive, folder per test day. Logs are not in git; the Lab keeps uploaded logs in `perf/uploads/`
  (also kept out of git).

## 4. Before the test day

1. **Mount the IMU** near the centre of gravity, axes aligned with the car (x forward, y left, z up). Note the offset
   from the CoG. Measure the mounting angles on the level ground (car with driver, static logging for 30 s).
2. **GNSS antenna** on the highest point with a clear sky view; note its position relative to the CoG.
3. **Calibrate** the steering angle (straight ahead = 0, full lock left and right) and the brake pressure sensors
   (zero when released).
4. **Weigh** the car with the driver and ballast on four scales: mass and static weight distribution (`m`, `lf`,
   `lr`).
5. **Test log:** drive 2 minutes on the parking lot, load the file in the Lab (Logs page, *Inspect*), check every
   must-have channel is there, with sensible units and signs (left turn → ay > 0, yaw rate > 0).

## 5. Test-day plan

Order chosen so that tyres and brakes are warm for the grip tests and the battery lasts. Every test **at least 3
valid runs**; the Lab takes the median.

| # | Test | How | What it gives | Duration |
|---|---|---|---|---|
| 0 | Static | 30 s standing, engine on, level ground | IMU offsets | 1 min |
| 1 | Warm-up | 2 laps of the handling course | tyres, brakes warm | 5 min |
| 2 | **Coast-down** | straight ≥ 150 m, both directions. Accelerate to ~80 km/h, then torque 0 (no regen), no brake, steering straight, coast to ~20 km/h | `f_roll`, `c_w·A` | 3 × 2 directions |
| 3 | **Steady cornering** | circle of known radius (skidpad circle, R ≈ 9.1 m; and a larger one, R ≈ 20–30 m if space). Increase speed slowly over 3–4 laps until the car runs wide | maximum ay at two speeds → `muy`, `util_y`, rough `c_z` | 2 radii × 2 directions |
| 4 | **Skidpad** | the official figure of eight | skidpad time, `util_y` of the event | 3 runs |
| 5 | **Acceleration** | 75 m from standstill, launch as in competition | traction-limited and power-limited ax(v), `util_x`, launch | 3 runs |
| 6 | **Braking** | from 60–80 km/h, maximum braking without locking, straight | brake-limited ax(v), brake balance | 3 runs |
| 7 | **Autocross lap** | a course of known geometry (cones measured with GPS, or the course walked with the GNSS receiver) | speed trace, lap time, `util_x/util_y` | 3 laps |
| 8 | **Endurance stint** | 5–10 laps of the autocross course at endurance pace | energy per lap, power cap behaviour, efficiency map | 1 stint |

Log continuously during the whole session; mark the start of each run with a button / marker channel if the logger
has one, otherwise note the clock time in the run sheet.

**Safety and validity:** drive the tests only on a closed area with the usual test-day safety setup. A run is not
valid if: the car hit a cone, spun or left the course; the driver lifted mid-test; the surface changed (damp patch);
the wind was above ~3 m/s for a coast-down.

## 6. In the Performance Lab

All on the **Logs** page. Upload the file, choose the channels (speed, ax, ay, and for some tools yaw rate,
latitude/longitude, torque, brake pressure), choose the time window of one run.

| Tool | Input | Output |
|---|---|---|
| *Correlate* (exists) | one lap, an event | speed RMSE and bias, time difference along the lap |
| *Powertrain* (exists) | MF4 with inverter channels | efficiency map, loss model, roll-out check |
| **Identify → Coast-down** | one coast-down window (speed; brake pressure if logged) | `a(v) = c0 + c2 v²`, fitted to the speed with the exact solution of the coast → `f_roll` and `c_w` (with the setup's mass, area, downforce and air density), fit quality; compare with the setup. One direction per window: run it for both directions and average |
| **Identify → Grip use (ggV)** | any fast running, or straight-line acceleration / braking runs (speed, ax, ay) | the log's accelerations against the model's tyre limits per speed; the share of the limit used laterally, in braking and in grip-limited traction (95th percentile) → suggested `util_y`, `util_x`; ax(v) against the drive and brake limits |
| **Identify → Steady cornering** | constant-radius circle (speed, ay; radius) | maximum steady ay against speed, against the model's `ay_limit(v)` → the model's grip scale that matches |
| **Identify → Track from GPS** | one lap or more (latitude, longitude) | one lap cut where the car returns to the start, smoothing splines (smoothing chosen from the data), x, y every 0.5 m in `perf/uploads/tracks/<name>.csv`: copy it into `lapsim/LR26_LapSim/tracks/` and reference it with `lts_track_csv` (LapSim computes the curvature, as for the FSA "real" tracks) |

Checked on synthetic logs of the CI test car (tests/test_testcar.py): coast-down recovers f_roll and c_w within
2 % and 4 % with 0.1 km/h speed noise; steady cornering recovers a 0.9 grip scale within 0.02; grip use recovers
the event's utilisation from a simulated lap; a GPS lap with 0.3 m noise gives the same QSS lap time as the true
line within 1.5 %. Not yet checked on a real log.

Each identification shows the suggested value next to the setup's value. Nothing is changed in `lr26_params.m`
automatically: the Performance department decides, then edits the file (or saves a setup with the change and
compares it first).

## 7. What a good correlation looks like

| Check | Target |
|---|---|
| Lap speed RMSE, autocross / endurance | < 4 km/h after the utilisation is fitted |
| Lap time | within 2 % |
| Coast-down fit | rms residual < 0.05 m/s², the two directions within 10 % |
| Endurance energy per lap | within 5 % |
| Steady cornering ay_max | within 3 % of the model with the identified grip |

If the speed trace is right but the time is not, check the track geometry (Track from GPS) before the car.

## 8. Not covered yet

* Tyre identification from a rig (TTC data): the Magic-Formula shape of the dynamic models uses `vehicle.yaml`
  defaults.
* Aero maps (ride height, yaw): only an overall `c_z`, `c_w` and balance.
* Transient tests (step steer, sine sweep) for the yaw inertia and the transient model: worth a later test day once
  the steady-state parameters are known.
