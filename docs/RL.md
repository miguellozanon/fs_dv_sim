# RL driver: learned throttle and brake

> One policy for **every** setup in a range (trained on the GPU / Isaac Lab, no retraining when a
> parameter changes): see [RL_UNIVERSAL.md](RL_UNIVERSAL.md). This page covers the policy trained for
> one specification.

The simulator can learn **how to use the accelerator and the brakes** for a given vehicle
specification (the LapSim parameters you set in `lr26_params.m`, the dashboard's Vehicle setup
or `-Set` / `vehicle_set:=`). Steering stays with the reference path follower; the stop after the
finish stays with the reference driver's stop logic (a rules and safety function).

```
vehicle spec (LapSim parameters, actuator limits)
   │
   ├─ tools/train_rl.py  ── imitation (DAgger) of the reference driver ──► evolution strategies (RL)
   │        headless fsdsim runs, same physics as Isaac          │
   │                                                             ▼
   │                                   rl/policies/<name>/policy.npz + policy.json
   │                                                             │
   └─ simulator (Isaac or lite) ── demo_driver longitudinal:=rl ─┘  (picks the policy of the running spec)
```

## What the policy sees and does

Every 40 ms (it also runs at the driver node's 50 Hz) the policy gets 26 numbers:

| Input | Why |
|---|---|
| speed | |
| speed the **LapSim model of this car** allows at the tyre limit at 0, 4, 8, 14, 22, 32, 45 and 60 m ahead | the vehicle specification: grip, aero, power, mass enter here |
| path curvature at the same points | where the corners are |
| lateral velocity, yaw rate, longitudinal and lateral acceleration, side-slip angle | is the car stable, how much grip is in use |
| lateral and heading error to the path | is the steering coping |
| speed minus the tyre-limit speed here, previous action | |

It answers with one number `a` in [-1, 1]: `a > 0` throttle (`a × max_accel`), `a < 0` brake
(`a × max_decel`). The car's traction control and ABS limits are in the dynamics (LapSim's friction
ellipse), so "full throttle" means "as much as the rear tyres take". Below 2 m/s, while racing, the
policy cannot brake (launch rule: standstill must not be a trap).

The policy is a small neural network (26 → 32 → 32 → 1, tanh), stored as plain NumPy arrays.
Running it needs only NumPy: the ROS nodes, the dashboard and Isaac's Python all can.

## How it learns

1. **Imitation (DAgger).** The policy first learns the reference driver's pedal work. In 4 rounds
   it drives itself and the reference driver labels what it would do in those states, so it also
   learns to recover. The result drives like the reference driver (same lap times within ~0.04 s).
2. **Reinforcement learning, evolution strategies** (OpenAI-ES, Salimans et al. 2017), on the
   network's output layer. The hidden layers keep the features learned by imitation; RL searches
   the 33 output weights, much like Augmented Random Search (Mania et al. 2018). Searching all
   ~1,900 weights at once was tried first and did not improve on the imitation: the useful gain
   lies just next to the grip limit, where large random changes spin the car. Every generation:
   * 16 slightly perturbed copies (8 antithetic pairs) drive the same scenarios:
     acceleration from standstill (4 s); skidpad entry and right-hand circles (14 s); three
     generated autocross layouts (18 s each), from the start line or from a random point at speed.
   * **Fitness:** metres driven along the path in the time window; minus 10 per cone or off-course
     excursion; minus 80 if the car leaves the track or spins; minus a little for jerky pedals.
   * **Update:** the rank-weighted perturbations give the direction, applied with Adam, with a
     pull back towards the imitation solution unless the change pays off. Rollouts run in
     parallel on the CPU cores.
3. **Selection and evaluation.** Every 5 generations the policy drives held-out tracks; the best
   one is kept. At the end it races complete missions (official times, penalties) against the
   reference driver, and the results are stored in `policy.json`.

**Expect small gains.** The reference driver is already tuned to this car: its speed targets come
from LapSim at 80 % of the grip. There is little room before the car spins. The reference driver
itself is about 2 % faster at 85 % grip and 80 % on braking, and spins at 95 %. RL finds this
margin per vehicle spec, and per situation (corner type, speed), instead of one fixed number. The
gain grows when a spec moves away from the one the reference margins were tuned on.

Training uses the CPU only. A generation runs 16 rollouts, so up to 16 workers help:
* about 10 min for 50 generations on a 16-thread laptop;
* about 40 min on 2 cores.

## Results: `rl/policies/lr26_baseline` (LR26 as in `lr26_params.m`)

The project ships a policy for the baseline LR26. It was trained on 2 CPU cores: imitation, then
50 + 25 + 30 generations (`--init` continues a run). Results on complete missions in fsdsim:

Reference times from training (Oct 2026). Since the reference driver's steering preview (9 Nov 2026) it gives
skidpad 5.375 s and autocross 20.13 s; `tools/train_rl.py --evaluate` shows the current numbers.

| Mission | Reference driver | RL driver |
|---|---|---|
| Acceleration | 4.767 s | **4.555 s** |
| Skidpad (timed laps) | **5.494 s** | 5.512 s |
| Trackdrive, 10 laps (layout 7) | 197.05 s | **194.18 s** |
| Autocross on 10 layouts never used in training | | faster on all 10 (−0.22 to −0.58 s, mean −0.42 s, about −2 %) |

Penalties: on the 10 unseen autocross layouts each driver hit one cone, on a different layout.
Everything else is clean, and both drivers stop in the stop zone on acceleration and skidpad.

Where the time comes from:
* **Acceleration:** the policy floors it from the start. The reference driver follows a speed
  profile with a P controller, which lags behind it.
* **Corners:** the policy carries more speed into the slower corners and brakes a little later.
  This is the margin the reference driver leaves with its fixed 80 % / 70 % grip settings.
* **Skidpad:** no gain. The reference driver is already at the limit there, and a policy that
  pushed harder hit cones on the left-hand circles.

## Train

**Dashboard** (http://localhost:8080/#setup): apply a setup, then **Driver → Train RL driver**.
It trains for the applied setup (its vehicle changes) in the background, shows progress, and,
when finished, the policy appears in the list with its gain over the reference driver.

**Command line** (WSL, no ROS needed):

```bash
cd ~/fsg-dv-sim
python3 tools/train_rl.py                                      # LR26 as in lapsim/LR26_LapSim
python3 tools/train_rl.py --set m=270 "mu*1.05" --name light   # another spec
python3 tools/train_rl.py --generations 120 --tracks 4        # longer, less noisy
python3 tools/train_rl.py --evaluate rl/policies/lr26_baseline # RL vs reference on complete missions
```

Policies go to `rl/policies/<name>/` (`policy.npz`, and `policy.json` with the vehicle spec,
training curve and evaluation). Commit them to share them.

## Drive with it

* **Dashboard:** **Driver → RL (learned)**. The running simulator switches at once; finished missions
  show an **RL** tag in the DV sim column.
* **Launch:** `ros2 launch fsd_sim_bringup isaac.launch.py longitudinal:=rl` (or `lite.launch.py`).
  `rl_policy:=auto` (default) picks the policy trained for the running vehicle spec;
  `rl_policy:=<name>` forces one.
* **Offline demo:** `ros2 run fsd_dashboard dashboard_demo --mission autocross --longitudinal rl`.
* **Without a matching policy** the driver node stays with the reference driver and says why
  (`/sim/driver`, dashboard, log).

A policy is tied to the vehicle specification it was trained for (the hash of the LapSim
parameters, actuator limits and yaw inertia). Change a parameter and train again; the old
policies stay in the list.

## Limits

* Map mode only: the policy needs the known path (acceleration, skidpad, autocross and
  trackdrive with a map). In cones mode the driver node uses the reference driver.
* It learns throttle and brake, not steering or the racing line.
* It is trained on fsdsim's physics (the same model Isaac uses for the car). Grip on the real car
  differs; treat the policy as a study tool for "how much can this car take", not as a
  competition controller.
* Evolution strategies need many runs but no gradients or GPU. The gradient-based, GPU-scale
  version is the universal policy (PPO): [RL_UNIVERSAL.md](RL_UNIVERSAL.md).
