# Architecture and design notes

## Design goals

1. **One interface for sim and car.** The autonomy stack only sees ROS 2
   topics whose types, frames and names match what the car's drivers publish:
   ouster-ros point fields, ZED wrapper topic layout, and the FS DV CAN state
   codes. Swapping the simulator for the real car should not require code
   changes.
2. **One source of truth for behaviour.** Vehicle dynamics, the AS state
   machine, timing, penalties and proprioceptive sensors live in `fsdsim` and
   are shared by every backend. The renderer only decides how the LiDAR and
   cameras are produced.
3. **Testable without a GPU.** The core is pure Python with NumPy, so unit
   tests and closed-loop regression run on any laptop or CI runner.
4. **Everything identifiable is a parameter.** Vehicle, sensors, events and
   rules are YAML. That is what makes the step from simulator to digital twin
   a data problem rather than a code problem.

## Simulation loop

`Simulation.step(dt)` sub-steps physics at `physics_rate_hz` (500 Hz default,
RK4 inside the vehicle model with ≤ 2 ms substeps). Each substep:

1. The AS auto-sequence runs: mission select, ASMS on, then Go after
   `go_delay_s` once the control stream is alive.
2. The control watchdog is checked. In AS Driving, a stale `/control/command`
   leads to AS Emergency.
3. The brake override is applied:
   * AS Off / Ready / Finished: brakes held
   * AS Emergency: EBS deceleration
4. The vehicle model is integrated.
5. Cone contact: the car footprint rectangle is checked against cone base
   circles. A hit knocks the cone and adds a penalty.
6. Mission update: timing-line crossings (interpolated to sub-step time), lap
   count, off-course and leaving-track checks, and standstill leading to
   AS Finished.
7. Sensors are sampled when due, each on its own rate scheduler.

The outputs are plain-dict **events** (`imu`, `gnss`, `vehicle_sensors`,
`lidar`, `camera_detections`, `as_status`, `ground_truth`, `cones`, `track`).
`fsd_sim_bridge/ros_interface.py` is the only place that converts events into
ROS messages. It is used both by the lite node and by the Isaac relay.

## Vehicle model

* **Dynamic single-track model.** Simplified Pacejka lateral tyres, a friction
  ellipse with longitudinal force, longitudinal load transfer, aero drag and
  downforce with front/rear balance, rolling resistance, a power limit, a
  drive split and brake bias.
* **Kinematic single-track model at low speed.** It is blended in over
  `low_speed_blend` (3-5 m/s), where the dynamic model is singular. This is
  the FSSIM / AMZ approach.
* **Actuators.** First-order lags for steering and acceleration request, plus
  a steering rate limit.
* **Chassis attitude** (`fsdsim.chassis`, since Nov 2026). Roll, pitch and heave follow the accelerations and
  the downforce as second-order responses (roll / pitch gradient, wheel rates, natural frequencies and damping in
  `vehicle.yaml` → `chassis:`). The planar motion is unchanged; the attitude tilts what the sensors see: LiDAR
  rays, the IMU (gravity components, roll and pitch rate, orientation), odometry and TF, and the car body in
  Isaac Sim. `chassis.attitude: false` gives the old flat car.
* **Wheel slip.** The wheel-speed sensors include longitudinal slip from the drive or brake force of each wheel
  against its grip (`slip_B`, `slip_C`); SLAM and odometry code therefore sees what the car's sensors see.
* **Double-track handling** (`chassis.handling: double_track`, launcher: Handling model). Slip angle, load
  (longitudinal and lateral transfer) and force per wheel, yaw moment from the left / right difference. The
  default `single_track` keeps the LapSim axle forces, so times stay equal to the Performance model; double
  track changes the skidpad by about 0.01 s and is meant for work on yaw control and torque vectoring.
* **Energy.** The model integrates electrical power (drive force x speed over the drivetrain efficiency, with an
  optional motor efficiency map, see PERFORMANCE.md); `energy_kwh` and `power_kw` are in the ground-truth event,
  in the closed-loop CSV and in the regression report.
* **Assumed values.** The `chassis:` numbers are typical Formula Student values, **not measured on the LR26**.
  Replace them when the suspension department has roll / pitch gradients and wheel rates.
* **Not modelled.** Suspension kinematics (camber, bump steer), tyre temperature, torque vectoring control.

## Driver: reference and RL

`fsdsim.control.DemoDriver` closes the loop until the team's own stack replaces it:
* **Steering:** pure pursuit on the known path (map mode) or on cone midpoints (cones mode).
* **Speed:** targets from the LapSim QSS model of the running car, at a grip margin.

On circuits the map-mode path is the centre line or, with `line:=racing`, the racing line inside the cones
(`fsdsim.perf.raceline`, see PERFORMANCE.md). In cones mode with a perception node the driver can switch to the
mapped track after N laps (`race_after:=N`: explore, then race; see PERCEPTION.md).

`fsdsim.control.RLDriver` keeps the same steering and stop logic but takes throttle and brake from
a policy learned for the vehicle specification (`fsdsim.rl`: imitation, then evolution strategies
on headless fsdsim runs). The driver node switches between them with `longitudinal:=rl` or at run
time over `/sim/driver_set`. Details: [RL.md](RL.md).

`fsdsim.vec` is the same car (dynamics in performance mode, LapSim limits, pure pursuit, cone and
timing checks) as array code for N cars with N specifications at once, on NumPy or PyTorch / CUDA.
PPO trains one **universal** policy on it (features v2: speed envelope from the car's own LapSim
limits, plus its specification), from `tools/train_rl_universal.py` or Isaac Lab (`isaaclab/fsd_rl`,
RSL-RL). `RLDriver` computes the v2 inputs with the same functions, and `rl.registry` picks the
universal policy for every setup inside its training ranges. Details: [RL_UNIVERSAL.md](RL_UNIVERSAL.md).

## Sensor models

**LiDAR (analytic, lite backend)**
* Every beam of the configured spinning LiDAR is ray-cast against the ground
  plane and against each cone as an exact frustum. Stripes give the
  reflectivity bands that many teams use for colour classification.
* The vehicle pose is interpolated per firing column across the scan, so the
  output is motion-distorted exactly like the real sensor.
* Points are in the per-column sensor frame, with `t` in ns from scan start
  and `ring` 0 = top beam.

**Camera (analytic, lite backend)**
* Pinhole projection with ZED intrinsics and occlusion by nearer cones.
* Range-dependent detection probability.
* Stereo depth noise σz = z²·σd / (f·B).

**IMU**
* Specific force from body accelerations, with lever-arm terms for the mount
  offset.
* White noise and bias random walk, given as continuous densities converted
  to the sensor rate.
* Optional orientation output, as from an INS.

**GNSS**
* Antenna lever arm, ENU to WGS84 via the local tangent plane at the origin.
* RTK or single-point noise; ENU velocity.

**Vehicle sensors**
* Per-wheel rpm from the kinematics, including steering.
* Steering encoder.
* Optical ground-speed sensor.

## Perception, SLAM and sensor faults

`fsdsim.perception` turns the simulated sensors into what a driverless stack needs, without ground truth: cones
from the LiDAR points, colours from the camera, a cone map and the car's pose from wheel speeds, gyro and the
cones (SLAM). `fsdsim.faults` makes sensors fail on a schedule. Both run in the tools, in the lite backend and
with Isaac Sim's rendered sensors. Details: [PERCEPTION.md](PERCEPTION.md).

## Isaac Sim backend: why a kinematic chassis

The car prim in Isaac is moved every frame from the fsdsim state, so PhysX does
not simulate the car. This keeps vehicle behaviour identical between backends
and fully parameterised by `vehicle.yaml`, and avoids tuning a second,
different PhysX vehicle model.

RTX sensors ray-trace the scene at render time, so they are unaffected by how
the chassis moves. Motion distortion within a LiDAR scan comes from the RTX
LiDAR model itself, which ray-traces the rotating sensor at sub-frame time
across consecutive render frames.

### Windows ↔ WSL link

Isaac Sim runs on Windows; ROS 2 runs in WSL2. NVIDIA deprecated the DDS route
between the two with Isaac Sim 6.x. Instead, `fsdsim.link` uses one TCP
connection: framed JSON with NumPy arrays as raw binary.
* **Hello.** On connect Isaac sends a hello with the full configuration, then the track and cones.
* **Data.** Events, LiDAR scans and camera frames follow; commands flow back.
* **Backpressure.** Sensor frames are droppable; AS state and commands are never dropped.

All DDS traffic stays inside WSL.

### Time

* **Isaac backend.** fsdsim is advanced to the timestamp of the frame about to
  be rendered, so event, LiDAR and camera stamps share Isaac's time base. The
  relay publishes it as `/clock`; with the `ros2` sensor transport, Isaac's
  OmniGraph publishes it instead. The log reports any drift.
* **Lite backend.** `lite_sim` publishes `/clock`.
* **All other nodes** run with `use_sim_time:=true`.

## Path to a digital twin

| Step | What changes | Where |
|---|---|---|
| Vehicle identification | Fit mass / inertia / tyre / aero / actuator lags to coast-down, step-steer, skidpad and acceleration logs | `vehicle.yaml` (+ optional new model terms in `dynamics.py`) |
| Sensor calibration | Noise, bias, latency, dropout from recorded rosbags; real mount poses from calibration | `sensors.yaml` |
| Real tracks | Surveyed or SLAM cone maps exported as CSV | `track_file` |
| Replay / comparison | Feed recorded `/control/command` into the sim and compare the trajectory with car odometry | new tool next to `closed_loop_eval.py` |
| Hardware-in-the-loop | The DV computer talks CAN; the sim already speaks the DV logging frames (`fsdsim/dv_can.py`) and can be extended to the team's VCU messages | new bridge node |
| Performance model | `fsdsim.perf`: LR26 LapSim port; `vehicle.yaml` model `performance` drives the dynamic model with the same limits; `tools/param_study.py` | v1 done (docs/PERFORMANCE.md) |
| Dashboard | `fsd_dashboard`: subscribes to the same topics, identical for sim, replay and car; HTTP + Server-Sent Events, no external web libraries | v1 done (docs/DASHBOARD.md) |
