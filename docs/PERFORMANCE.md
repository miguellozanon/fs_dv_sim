# Performance model (LR26 LapSim) inside the simulator

The Performance department's lap time simulation (LR26 LapSim, MATLAB) lives in this repository,
in `lapsim/LR26_LapSim`, and the driverless simulator reads its car from there. There is one
parameter file for both: change a value in `lr26_params.m` and MATLAB, the Python LapSim, the
reference driver and the dynamic DV car all use it.

```
lapsim/LR26_LapSim/                        (the MATLAB code, unchanged: run_lapsim, lts_*.m, tracks)
  config/lr26_params.m        ─┐  Table 3.1: m, lf, lr, h_cog, c_w, c_z, mux, muy, i_trans, T_motor, ...
  config/competition_fsa2025.m ┤  events, FSA 2025 tracks, driver utilisation, laps, scoring
  tracks/*.csv                ─┘
        │  read directly (vehicle.yaml: performance_source: lapsim)
        ├── fsdsim/perf/lapsim.py  port of lts_run_event / lts_qss_solve / lts_corner_speed / lts_score:
        │                          all four events with points, identical to MATLAB
        ├── fsdsim/perf/model.py   port of lts_resistance, lts_powertrain, lts_wheel_loads,
        │                          lts_axle_state, lts_ax_limits
        ├── vehicle.yaml (model: performance) -> fsdsim/dynamics.py: the dynamic DV car uses the same
        │       wheel loads, tyre potentials, friction ellipse, powertrain, brakes and aero
        ├── reference driver: speed targets from the same model on the simulator's track
        └── dashboard "Vehicle setup": change parameters -> LapSim, MATLAB and the simulation
```

**Performance Lab:** the lap time simulation alone, for the Performance department (Windows, double-click
`perf\perf.bat`, no WSL/ROS/Isaac): sensitivities, sweeps, setup comparison, transient lap, energy, ggV and log
correlation. See [PERF_APP.md](PERF_APP.md). Its lap time methods (QSS, transient, minimum-time optimal control,
Monte-Carlo robustness), their hypotheses and limits: [PERF_METHODS.md](PERF_METHODS.md). The parameter table of the dashboard's Vehicle setup page and of the
Performance Lab is one list, `fsdsim/perf/schema.py`.

## Where the numbers come from

* **Parameters:** `config/lr26_params.m` and the competition file are parsed at start-up
  (`veh.<field> = <expression>;`, `e.util_x = ...`, `lts_track_csv(...)`, `score_def(...)`).
  If the folder is missing, the simulator falls back to `performance_lr26.yaml`, a copy written by
  `tools/import_lapsim_params.py`. `FSG_LAPSIM_DIR` points it at another LapSim folder.
* **Physics:** the LapSim functions ported line by line. Against LapSim's own results in
  `lts_results_sensitivities.mat` (10 swept parameters, 85 values, all four events), the largest
  difference is below 0.005 ms, the rounding of the `.mat` file. Baseline: acceleration 4.662 s,
  skidpad 5.154 s, autocross 49.944 s, endurance 1527.0 s, 164.35 points.
  `tests/test_vehicle_setup.py` and `tests/test_performance.py` check this on every run.
* **Event settings** (`P_max_endurance`, `endurance_torque_scale`, `v_lim`, driver utilisation, scoring)
  apply to LapSim's events only. The DV car on the simulator's tracks has its own driver (the reference
  driver keeps a grip margin) and always runs with the full motor power.

## Changing parameters

Both sides use the same names: the LapSim fields (`m`, `mux`, `muy`, `c_z`, `c_w`, `A`,
`aero_bal_f`, `h_cog`, `i_trans`, `T_motor`, `n_max`, `P_max`, `brake_bias`, `tire_exp`, ...) and the
aliases of `lts_set_param`: `mu`, `mass`, `gear`, `cd`/`cw`, `cl`/`cz`, `cog`, `power`,
`cop`/`aero_balance`, `torque` (case-insensitive). `mu=1.8` behaves as in MATLAB (`muy` = 1.8, `mux`
keeps its ratio). Forms: `name=value`, `name*factor`, `name+delta`, `name-delta`.
`<event>.util_x` / `<event>.util_y` (e.g. `autocross.util_y=0.85`) change LapSim's driver
utilisation; `vehicle.yaml` fields (`yaw_inertia`, `actuators.max_steer`) change only the DV car.

What to compare: acceleration and skidpad are the same event in MATLAB and in the simulator.
Autocross differs on purpose (MATLAB: FSA 2025 track; simulator: its own cone layout) and
endurance is 16 laps of the FSA track against 10 laps of trackdrive, so compare the *change*.

### 0. The dashboard: Vehicle setup page (recommended)

http://localhost:8080/#setup, with the simulator running (or `ros2 run fsd_dashboard dashboard_demo`
without ROS):

1. The form lists every Table 3.1 parameter with its value from `lr26_params.m`. Type new values
   (or a factor like `*1.05`), or load a saved configuration.
2. **Apply** runs:
   * **LapSim** (the Python port, under a second): times and points of all four events, with the
     difference to the baseline;
   * **MATLAB** in the background (Windows MATLAB through WSL; 20–60 s): the same changes in the real
     LR26_LapSim, shown next to it when it finishes;
   * **the simulation**: the running simulator (Isaac Sim or lite) swaps the vehicle and restarts the
     mission. The time of every mission that finishes is added to the table.
3. **Save** stores the configuration in `lapsim/configs/<name>.json`, so the team can share it.

Details: [DASHBOARD.md](DASHBOARD.md#vehicle-setup).

### 1. The car changes (permanent)

Edit `lapsim/LR26_LapSim/config/lr26_params.m` (in MATLAB, open the folder
`\\wsl.localhost\Ubuntu-24.04\home\<you>\fsg-dv-sim\lapsim\LR26_LapSim`). The next simulator
start uses it; a running dashboard shows the new baseline after **Re-read MATLAB files** or the next
Apply. There is nothing to import.

DV-only differences (no driver, DV hardware) stay in `vehicle.yaml` -> `performance_changes`
(e.g. `["m-75", "m+20"]`), applied on top of the MATLAB values.

### 2. What-if for one run (no file changes)

| Simulator | MATLAB, same change |
|---|---|
| Isaac: `run_isaac.ps1 -Mission skidpad -Set "m=300","mu*1.05"` | `veh = lr26_params();` |
| Lite: `ros2 launch fsd_sim_bringup lite.launch.py mission:=skidpad vehicle_set:="m=300 mu*1.05"` | `veh = lts_set_param(veh, 'm', 300);` |
| No ROS: `tools/closed_loop_eval.py --missions skidpad --set m=300 mu*1.05` | `veh = lts_set_param(veh, 'mu', 1.05, 'scale');` |
| Offline dashboard: `ros2 run fsd_dashboard dashboard_demo --mission skidpad --set m=300` | `lts_run_all(veh, competition_fsa2025(), struct('verbose', true));` |

### 3. Parameter study (several values)

```bash
python3 tools/param_study.py --param m --values 260,284,310
```

writes the simulator report (`report.html`) and `matlab_study.m`, the same cases for MATLAB. It prints
the MATLAB command, e.g.

```matlab
cd <repo>/lapsim/LR26_LapSim; startup_lapsim
run('\\wsl.localhost\Ubuntu-24.04\home\laptop\fsg-dv-sim\results\m_<time>\matlab_study.m')
```

which prints Time_s and Points per case and writes `matlab_results.csv` next to the report. For a
plain one-parameter study it also prints the equivalent `lts_sweep(...)` + `lts_plot_sweep(S)` call.
Add `--out /mnt/c/Users/<you>/Documents/MATLAB/study_m` to keep the study in a Windows folder.

## Parameter studies

```bash
python3 tools/param_study.py --param c_z --values 3.5,4.15,4.8                 # acceleration, skidpad, autocross
python3 tools/param_study.py --param mu --scale 0.9,1.0,1.1 --events skidpad autocross trackdrive
python3 tools/param_study.py --table                                           # +-10 % on mu, m, i_trans, h_cog, c_z, c_w
python3 tools/param_study.py --param i_trans --values 12,13,14.5,16,17.5,19 --no-sim   # LapSim only, seconds
python3 tools/param_study.py --param m --values 260,284,310 --set "brake_bias=0.65" --grip-util 0.85
```

Output in `results/<param>_<time>/`:
* `results.csv` and `results.json` (including speed traces);
* `report.html`: event table, time-vs-parameter charts and speed-over-distance (LapSim dashed, simulation solid). Open it in a browser; it needs no internet.

Runs are parallel (`--jobs`). Acceleration, skidpad and autocross take a few seconds each; trackdrive (10 laps) about a minute.

## First results (LR26 from lr26_params.m, default reference driver)

`docs/examples/study_table_lr26/` (`python3 tools/param_study.py --table`). Baseline: acceleration
4.662 s LapSim / 4.767 s sim, skidpad 5.154 / 5.494 s, autocross (fsdsim layout) 19.12 / 20.28 s.
Since the steering preview of the reference driver (9 Nov 2026, docs/CAD.md): skidpad 5.375 s, autocross 20.13 s;
the table below is from before and still shows the sensitivities.

Sensitivity, seconds per +1 % of the parameter (central difference over ±10 %):

| Parameter | Accel LapSim | Accel Sim | Skidpad LapSim | Skidpad Sim | Autocross LapSim | Autocross Sim |
|---|---|---|---|---|---|---|
| mu (x and y) | 0.000 | -0.006 | -0.028 | -0.029 | -0.083 | -0.093 |
| m | +0.022 | +0.015 | +0.005 | +0.004 | +0.038 | +0.026 |
| i_trans | -0.024 | -0.018 | 0.000 | 0.000 | -0.025 | -0.014 |
| h_cog | 0.000 | 0.000 | +0.001 | +0.001 | -0.001 | 0.000 |
| c_z | 0.000 | 0.000 | -0.002 | -0.002 | -0.008 | -0.009 |
| c_w | +0.001 | +0.002 | 0.000 | +0.001 | +0.002 | +0.003 |

All 39 runs finished without cones or off-course penalties.

What it shows:
* **Grip** matters slightly more to the DV car on autocross (−0.093 s/%) than LapSim predicts (−0.083 s/%).
* **Mass and gear ratio** matter less in the simulation than in LapSim, because the DV driver is not at the traction or power limit most of the lap.
* **Acceleration with more grip:** no change in either model. The car is motor-limited, not traction-limited.
* **Absolute times:** the simulation is 0.10 s (acceleration), 0.34 s (skidpad) and 1.2 s (autocross) slower than LapSim, because:
  * the reference driver keeps a margin (80 % of lateral grip, 70 % of that for braking: `--grip-util`, `--long-util`);
  * the simulation has yaw dynamics, actuator lag and path tracking.

## How the dynamic model uses LapSim

In `performance` mode, `fsdsim.dynamics` keeps the single-track equations, actuator lags and low-speed kinematic blend. Every force limit comes from the LapSim functions:
* **Wheel loads:** `lts_wheel_loads`, using the body accelerations of the previous step.
* **Lateral force:** the axle potential from `lts_axle_state` times a Pacejka shape (`B`, `C`, `E` in `vehicle.yaml`, peak 1).
* **Cornering priority, as in LapSim:** drive and brake forces only use the friction-ellipse remainder `Fx_rem = pxAx (1 - u^e)^(1/e)`. On the car this is what traction control and ABS do. Past the peak slip angle the tyre counts as fully used.
* **Drive:** `lts_powertrain` per driven axle, capped by `F_cap`. **Brakes:** `brake_bias`, or the ideal balance. **Aero:** `aero_eff`, `aero_bal_f`, `f_roll`. **Longitudinal mass:** `m·e_i`.

LapSim has no yaw inertia; `vehicle.yaml` provides `yaw_inertia` (120 kg m²).

`model: single_track` with `vehicle_generic.yaml` keeps the previous generic car.

## Real tracks, endurance energy budget, logs (from the LionsLTS review)

Details and caveats in [LIONSLTS.md](LIONSLTS.md).

* **Real FSA 2025 tracks:** `FSG_LAPSIM_COMPETITION=competition_fsa2025_real.m` (or `LapSim.load(root, competition=...)`) selects the competition file that uses the team's x,y track files with a cubic spline and a 5 m curvature filter (autocross 49.3 s, endurance lap 96.3 s against 49.9 s / 95.4 s in the documentation; the filter width moves the lap time by about ±6 %).
* **Endurance power cap from energy:** `python3 tools/endurance_strategy.py --budget-kwh 5.5` finds the cap whose 16-lap energy fits the budget (5.5 kWh gives 17.98 kW, the cap inferred from the documentation). In code: `fsdsim.perf.strategy.power_for_energy`.
* **LionsLTS vehicle files:** `python3 tools/import_lionslts_vehicle.py car.yml --save --run` creates a saved configuration (`lapsim/configs/`).
* **CSV for Marple:** `python3 tools/export_run_csv.py` writes a run with the LionsLTS signal names.
* **Car logs:** `python3 tools/mf4_summary.py log.mf4` reads `.mf4` files (no asammdf needed). `python3 tools/log_correlate.py log.mf4 --lapsim-effect` compares the logged motor efficiency with `eta_motor` (0.855 measured at up to 14 Nm against 0.90) and shows what it does to the endurance power cap.

## Racing line, efficiency map, ggv map, solver plug-ins, closed-loop endurance (Nov 2026)

All of this is own code on top of the LapSim port; nothing was copied from the TUM tools (ideas only).

**Racing line** (`fsdsim.perf.raceline`). A family of lines inside the cones (bounded least squares between
minimum curvature and minimum length, the car's width plus a margin kept free); the fastest by the car's QSS
model wins.

```bash
python3 tools/racing_line.py --mission trackdrive --seed 7 --png out/line.png     # line, gain, picture
python3 tools/closed_loop_eval.py --mission autocross --mode map --line racing    # driven in the closed loop
```
* LapSim (QSS) gain on the generated circuits: about 6 %. Driven by the reference driver: 2-3 %
  (seed 7: 20.12 → 19.48 s, seed 42: 18.23 → 17.72 s), clean at 50 and 20 Hz on all tested seeds. The driver
  keeps 0.40 m to the cones and uses 75 % of the lateral grip on the line (`RACING_MARGIN`, `RACING_GRIP_UTIL`
  in `control/demo_driver.py`).
* ROS: `line:=racing` (lite and Isaac launch files), launcher: **Path on circuits**. Reference driver only (the
  RL policies were trained on the centre line).
* The output CSV (`x_m,y_m`) can be used as a LapSim track. Note: the real FSA 2025 track files are already a
  driven line, not a centre line; do not optimise them again with the full track width.

**Motor efficiency map** (`fsdsim.perf.efficiency`). `eta(torque, rpm)` instead of the constant `eta_motor`,
used by the LapSim energy numbers and by the simulator's energy meter.
```bash
python3 tools/log_correlate.py log.mf4 --save-map lapsim/configs/eta_map.json --motors-per-axle 2
python3 tools/endurance_strategy.py --budget-kwh 5.5 --eta-map lapsim/configs/eta_map.json
```
`vehicle.yaml: efficiency_map:` or the environment variable `FSG_ETA_MAP` selects it for the simulator.
Assumption to confirm: two motors per axle share the axle torque equally (`motors_per_set`).

**ggv map** (`fsdsim.perf.ggv`): `python3 tools/ggv_map.py --out results/ggv.csv --png` writes the drive and
brake limits against speed and lateral acceleration (optionally with an event's driver utilisation).

**Solver plug-ins** (`fsdsim.perf.solvers`). A lap-time solver is a function registered under a name; files in
`lapsim/solvers/*.py` are loaded automatically (`example_solver.py` shows the interface). Built in: `qss` (the
LapSim port) and `point_mass` (no load transfer; 154 points against 164.35). `python3 tools/solver_compare.py`
compares them on the four events; `LapSim.run(..., solver="name")` uses one.

**Endurance in the closed loop** (`tools/endurance_closed_loop.py --budget-kwh 0.45`): drives N laps with the
reference driver and finds the power cap whose driven energy fits the budget (the closed-loop counterpart of
`endurance_strategy.py`). `closed_loop_eval.py --laps 10 --csv run.csv` writes any run with the LionsLTS signal
names for **Marple** (same columns as `export_run_csv.py`, plus the driven path and energy).

**Regression run**: `python3 tools/regression.py` (launcher: Maintenance) runs 33 closed-loop cases (events at
50 and 20 Hz, four circuits on centre and racing line, double track, perception with SLAM, sensor faults) and
writes `results/regression_<time>/report.md`. Run it before and after changing the model or the driver.
