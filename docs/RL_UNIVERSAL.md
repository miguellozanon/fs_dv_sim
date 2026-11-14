# Universal RL driver: one policy for every vehicle setup (GPU, Isaac Lab)

The [per-setup RL driver](RL.md) learns throttle and brake for **one** vehicle specification and
has to be retrained whenever the Performance department changes a parameter. The universal policy
is trained once over a **range** of specifications and is told the specification of the car it
drives. Every setup inside the range is driven at once: change `m`, `c_z` or `P_max` in the
dashboard, press Apply, and the learned driver drives the new car.

```
LapSim parameters (lr26_params.m)  ──►  spec ranges around them (fsdsim/vec/spec.py)
                                              │  sample a car per episode (+ hidden grip, lags, latency, noise)
batched LR26 model (fsdsim.vec, NumPy or PyTorch/CUDA)  ◄── thousands of cars, each its own spec and track
                                              │
PPO  (tools/train_rl_universal.py  |  Isaac Lab + RSL-RL: isaaclab/fsd_rl)
                                              │
rl/policies/<name>/policy.npz  (features v2, "universal": true, the ranges it covers)
                                              │
simulator / dashboard / param_study --driver rl / rl_spec_sweep   (picks it for every setup in range)
```

## The car model on the GPU

`fsdsim/vec/car.py` is fsdsim's dynamic model in `performance` mode, written with array operations so
N cars with N different specifications step together:

* wheel loads from static weight, aero (`c_z`, `aero_bal_f`, `aero_eff`) and longitudinal / lateral load
  transfer (`h_cog`, track widths), load-sensitive tyre potentials (`mux`, `muy`, `dmu/dFz`, `Fz0`),
* Pacejka lateral shape, friction ellipse (`tire_exp`), cornering priority (traction control / ABS),
* powertrain `min(T_motor, P / omega)` through `i_trans`, `eta_g`, `n_max`, RWD or AWD; brakes with a fixed
  bias or ideal balance; drag, rolling resistance, `e_i`,
* actuator lag and steering rate limit, kinematic blend at low speed, RK4 at 500 Hz,
* the reference driver's pure pursuit for steering, fsdsim's cone contact check (footprint vs cone base,
  every 8 ms) and off-course rule, the official timing marks (start / finish line, skidpad laps 2 and 4).

It is checked against the simulator in the tests: trajectories equal to 1e-9 m for three specifications
at once, steering identical to the reference driver, the acceleration run at full throttle within 0.001 s
of fsdsim, LapSim's corner speeds and acceleration limits to the port's precision. On a GPU the same code
runs in PyTorch (`fsdsim/vec/backend.py` switches NumPy / torch).

## What the policy sees (features v2)

* speed; the **speed envelope** at 0 ... 52 m ahead: the highest speed from which *this* car can still brake
  for every corner further ahead at the tyre limit (LapSim corner speeds and braking limits of its spec,
  lookup tables built per car when it gets a new spec),
* the curvature of 12 intervals ahead, the full traction and braking acceleration it has now,
* body motion (lateral velocity, yaw rate, ax, ay, side slip), tracking error, previous action,
* **the specification** (15 numbers: mass, mux, muy, downforce, drag, aero balance, CoG height, weight
  distribution, power, drive force, top speed, RWD / AWD, brake bias, yaw inertia).

The same code (`fsdsim/vec/features.py`) computes the inputs in training and in the simulator's RL driver.

## Realism: what the policy is *not* told

Every episode the car also gets hidden variations, so the policy keeps a margin like a real driver:
track grip ×0.95 ... 1.03, longitudinal actuator lag ×0.8 ... 1.4, steering lag ×0.8 ... 1.3, a control
latency of 0 or 40 ms, sensor noise on speed, IMU and tracking error. Episodes: acceleration (5 s),
skidpad (whole figure of eight), 60 generated autocross layouts (20 s, half of them from a random point
at speed). Reward: metres of progress, −30 per cone (≈ the 2 s penalty), −100 and end of the episode for
leaving the track or spinning, a small cost for jerky pedals.

Default ranges around the LapSim baseline (`SpecRanges`, change with `--ranges file.json`):

| Parameter | Range |
|---|---|
| m | ×0.85 ... 1.15 (241 ... 327 kg) |
| mu (mux and muy) | ×0.85 ... 1.12, mux / muy ratio ×0.93 ... 1.07 |
| c_z / c_w | ×0.70 ... 1.30 / ×0.75 ... 1.25 |
| aero_bal_f | 0.40 ... 0.58 |
| h_cog | ×0.85 ... 1.20 |
| static front weight | 42 ... 56 % (wheelbase kept) |
| P_max | 40 ... 80 kW |
| T_motor, i_trans | ×0.80 ... 1.20, ×0.85 ... 1.15 |
| yaw inertia | ×0.80 ... 1.25 |
| drive | RWD, 25 % AWD |
| brake bias | ideal, 35 % fixed 0.50 ... 0.72 |

Other parameters (wheelbase, track widths, `n_max`, tyre shape, ...) must stay at the baseline; the
simulator says which parameter is outside (`/sim/driver`, dashboard) and uses the reference driver then.

## Train

**GPU from the launcher:** **RL & Performance → Train the universal policy → Compute: GPU**. It runs on the
Windows Python with CUDA PyTorch (the project folder is on `C:`). Check the GPU code first with
**Maintenance → Check the GPU code** ([LAUNCHER.md](LAUNCHER.md)).

**GPU, PyTorch, by hand** (Windows: `python tools\train_rl_universal.py --backend torch`; or WSL with an NVIDIA driver and `pip install torch`):

```bash
cd ~/fsg-dv-sim
python3 tools/train_rl_universal.py --backend torch --device cuda:0 --num-envs 4096 --iterations 1500
```

The GPU run saves the best policy and `agent_torch.pt` whenever the validation improves, so a crash or a
stop keeps the progress (`policy.json`: `"complete": false`, `iterations_done`); `--init rl/policies/<name>`
goes on from there. Out of GPU memory (another program on the GPU, e.g. Isaac Sim) it pauses and retries
three times, then saves and exits with code 3.

**Speed:** the batched car model launches many small GPU operations per step (20 physics sub-steps of
0.002 s per 0.04 s control step), so with 4096 cars the GPU mostly waits for Python and the driver: about
4400 car-steps/s on an RTX 5060 Laptop GPU, no faster than one CPU core with NumPy. More cars cost almost
the same time per step, so try `--num-envs 16384` (still well inside 8 GB with Isaac Sim closed) and compare the `steps/s` in the log.

**Isaac Lab 3.0 + RSL-RL** (Windows): see [isaaclab/README.md](../isaaclab/README.md). Train with Isaac Lab's
`train.py --task FSD-Longitudinal-Universal-v0 --external_callback fsd_rl.register`, then
`tools/export_rsl_rl_policy.py <model.pt>` writes the simulator policy (and checks it against the checkpoint).

**CPU, NumPy only** (any machine, slower: about 8 000 car steps/s per core):

```bash
python3 tools/train_rl_universal.py --iterations 2000            # ~1 h per 1000 iterations on one core
python3 tools/train_rl_universal.py --init rl/policies/universal --iterations 1000   # continue
```

All three: imitation of the reference driver's speed law as a warm start, then PPO (clipped surrogate,
GAE, adaptive learning rate, observation normalisation), the best policy on held-out tracks and setups
is kept. Output: `rl/policies/<name>/` (`policy.npz`, `policy.json` with ranges, curve and validation).

## Use it (Performance department)

* **Dashboard**, Vehicle setup: apply a setup; the Driver panel shows *"The universal policy ... drives
  this setup"*; choose **RL (learned)**. *RL policy* chooses automatic / universal / a named policy.
* **Launch:** `longitudinal:=rl` (`rl_policy:=auto` takes a policy trained for exactly this setup first,
  else the universal one; `rl_policy:=universal` always the universal one).
* **Parameter studies with the learned driver:** `python3 tools/param_study.py --param m --values 260,284,310 --driver rl`
  (LapSim vs the closed-loop simulation driven at the limit by the policy).
* **Fast sweeps** (all setups × tracks at once in the batched model, GPU optional):

  ```bash
  python3 tools/rl_spec_sweep.py --vary m=250:320:8
  python3 tools/rl_spec_sweep.py --vary c_z=3.2:5.0:5 --vary aero_bal_f=0.42,0.48,0.54 --tracks 8 --compare
  ```

  Per setup: acceleration / skidpad / autocross times, cones, top speed, peak lateral acceleration, the
  share of time traction, motor or power limited, LapSim's QSS times next to them, and the sensitivity
  (s per unit) when one parameter varies. `results/sweep_<time>/sweep.csv` and `sweep.md`.
* **Official runs in the full simulator** for any setup: `python3 tools/train_rl_universal.py --evaluate rl/policies/universal --set m=260`.

## Results: `rl/policies/universal`

Trained on the CPU with the NumPy trainer, 1 core, about 3.5 h:
* imitation and PPO, 3000 iterations with 256 cars;
* 1500 iterations after switching to fsdsim's exact cone check;
* 700 iterations of fine-tuning at a lower learning rate.

That is about 60 million car-steps. The default GPU run (1500 iterations × 4096 cars × 32 steps) is about 200 million.

**Held-out check in the batched model.** Acceleration, skidpad and 8 autocross layouts the policy never
trained on, for the baseline car and 8 randomly drawn setups (90 timed runs). Score = time + 2 s per cone,
compared with the reference driver's speed law on the same runs:

| | Universal RL | Reference law |
|---|---|---|
| Mean score | **−7.4 %** | 0 |
| Autocross | **−9.4 %** | |
| Acceleration / skidpad | +0.4 % / +1.4 % | |
| Cones (90 runs) | **12** | 22 |
| Spins / off track | **0** | 0 |

**Complete missions in the full simulator** (official timing, cones, off-course; `--evaluate`). The first two
setups are inside the range, the third is a low-grip AWD car:

Reference times from Oct 2026; since the reference driver's steering preview (9 Nov 2026) its baseline skidpad is
5.375 s and the RL cars steer the same way (the policy only works throttle and brakes).

| Setup | Mission | Reference driver | Universal RL |
|---|---|---|---|
| LapSim baseline | Acceleration | 4.767 s | **4.387 s** |
| | Skidpad | 5.494 s | **5.392 s** |
| | Autocross 7 / 11 / 23 / 42 | 20.29 / 20.84 / 20.28 / 18.33 s | **18.94 / 18.89 / 18.50 / 17.03 s** |
| | Trackdrive 10 laps (layout 7) | 197.05 s | **183.09 s** |
| m = 260, c_z = 4.6 | Acceleration / skidpad | 4.655 / 5.432 s | **4.212 / 5.256 s** |
| | Autocross 7 / 11 / 23 / 42 | 19.96 / 20.51 / 19.98 / 17.99 s | **18.52 / 18.48 / 18.12 / 16.63 s** |
| AWD, P_max 60 kW, mu ×0.92 | Acceleration / skidpad | 4.163 / 5.725 s | **3.703 / 5.528 s** |
| | Autocross 7 / 11 / 23 / 42 | 20.41 +3 cones +1 OC / 20.79 / 20.28 +2 cones +1 OC / 18.27 s | **18.68 / 18.48 / 18.26 / 16.78 s**, clean |

All RL runs are clean, and every car stops in the stop zone. For comparison, the per-setup policy of [RL.md](RL.md)
(evolution strategies) reached 4.555 s, 5.512 s and 194.18 s on the baseline.

Where the time comes from:
* **Acceleration:** full throttle from the release of the brakes. The LR26 is motor-torque limited for
  85–100 % of the run (see the sweep below), so there is nothing left to learn there.
* **Autocross:** the policy carries speed to the envelope of the car's own limits and brakes at the
  last moment the envelope allows. The reference driver keeps a fixed 80 % / 56 % grip margin.
* **Skidpad:** the circles are steady state, so the gain is small. The time comes from the entry and
  the change of direction.

Stop logic, changed for both drivers: after the finish line the stop braking now never lets the car
go faster than the corner profile ahead. A fast finish no longer runs wide in the first corner after the
line, which removed the reference driver's post-finish cones too.

**Example sweep** (`rl_spec_sweep.py --vary m=250:320:4 --tracks 4 --compare`, 21 s on 1 CPU core):

| m (kg) | Accel (s) | LapSim accel | Motor-limited | Skidpad (s) | Autocross (s) | Reference law autocross |
|---|---|---|---|---|---|---|
| 250 | 4.139 | 4.390 | 86 % | 5.261 | 17.92 | 19.32 |
| 273 | 4.311 | 4.578 | 91 % | 5.366 | 18.22 | 19.60 |
| 297 | 4.478 | 4.759 | 97 % | 5.407 | 18.48 | 19.86 |
| 320 | 4.642 | 4.934 | 100 % | 5.451 | 18.75 | 20.11 |

Sensitivities: +7.2 ms/kg acceleration, +2.6 ms/kg skidpad, +11.9 ms/kg autocross (4 layouts).
LapSim's acceleration time is longer because it models the launch separately (`ax_max_launch`); the
simulation releases the brakes and applies full torque with traction control.

## Limits

* Throttle and brake only; steering is the reference driver's pure pursuit, in map mode.
* The policy is as good as the model: grip, load transfer and the powertrain come from the LapSim
  parameters; tyre transients, suspension and the real car's control loops are not modelled. Read times
  as "what this setup can do with a driver at the limit of this model", next to LapSim's QSS numbers.
* Outside the ranges the simulator does not use the policy. Retrain with wider `--ranges` or around a new
  baseline (`vehicle_changes` in Isaac Lab, `--set` is not used for training on purpose).
* The GPU path is checked by `tests/test_vec.py` (PyTorch gives the same results as NumPy; passed on an RTX
  5060 Laptop GPU with Windows Python 3.14 and torch 2.14.1+cu132). The Isaac Lab path has not been run yet.
