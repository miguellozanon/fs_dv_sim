# LionsLTS (previous Performance code): review, what was taken over, what comes next

In November 2026 the Performance department's previous simulator, **LionsLTS** (Python, quasi-steady-state
lap-time simulation built on the TUM framework), was handed over as six branches: `master`, `dev`, `tumgit`,
`Bhuvan`, `Restructuring`, `InitialRefactoring`. This page records what is in them, how it compares with our
LapSim port (`fsdsim.perf`), what was adopted in this repository, what was *not*, and what is worth doing next.

Everything marked **to confirm** needs an answer from the Performance department; nothing there was changed
silently.

## 1. What the branches contain

| Branch | Content | Use for us |
|---|---|---|
| `master` | latest working version (README: "extension happens in dev") | reference |
| `dev` | the framework: `Wrapper → Simulation → solver plug-ins` (`TUMQSS`, `TUMQSSNW`, `TUMQSSSKIDPAD`), `InputLoader`, `ResultExporter` (CSV with named signals for Marple), YAML settings and vehicle files (LR25, LR26, LR26_emrax, ...), sweeps of 1 or 2 parameters, FSA points evaluation, `optimize_power` (endurance power cap from an energy budget), track helpers (OptimumLap, Google Earth), real FSA 2025 track files | main source of ideas and data |
| `tumgit` | TUM *global race trajectory optimisation*: minimum-time optimal control problem, powertrain models, friction maps. Licence LGPL-3.0; README: "do not touch code, dependency check off" | ideas only, nothing copied |
| `Bhuvan` | real CAN logs (`.mf4`) from the car and a PyQt task that trims MF4 files | real data for correlation |
| `Restructuring`, `InitialRefactoring` | refactoring of the TUM OCP simulator (on hold / closed) | ignored |

## 2. Comparison with our LapSim port

| Topic | LionsLTS | This repo | Verdict |
|---|---|---|---|
| Solver | forward/backward QSS, 1 solver per event type, plug-in classes | forward/backward QSS in `perf/lapsim.py`, line-by-line port of the MATLAB LapSim | equivalent idea; our port is validated against the MATLAB results |
| Track | spline of arc length, analytic curvature, moving average of curvature (`curv_filt_width` 5 m) | moving-average smoothing of x, y | **adopted as option** (`method='spline'`) |
| Real tracks | FSA 2025 autocross / endurance as x,y files | generated tracks | **adopted** (`competition_fsa2025_real.m`) |
| Endurance power | `optimize_power`: lower the power until the lap energy fits the budget | fixed cap 18 kW inferred from the documentation | **adopted**: `perf/strategy.py`, `tools/endurance_strategy.py` |
| Vehicle files | YAML `general / engine / gearbox / tires` | `lr26_params.m` + saved configurations | **adopted**: importer to our configurations |
| Output | CSV with named signals (Marple) | plots, HTML reports | **adopted**: `perf/export.py` writes the same column names |
| Real logs | MF4 files, no analysis code in the repo | none | **adopted**: dependency-free MF4 reader + summary tool |
| Optimal control | TUM min-time OCP (`tumgit`) | none | not copied (licence); see roadmap |
| Scoring constants | max points 75 / 75 / 100 / 325 / 100 | fitted to 2026 maxima (50 / 50 / 100 / 250 / 75) | ours kept; theirs are outdated (**to confirm**) |

## 3. What was changed in this repository

### Track method: spline + curvature filter
`track_csv(path, closed, ..., method="spline", curv_filter=5.0)` in `fsdsim/perf/lapsim.py` (Python) and
`tracks/lts_track_csv.m` (MATLAB, same options; written without MATLAB, **run in GNU Octave 8.4 on 9 Nov 2026**: the real tracks give
autocross 49.26 s and an endurance lap of 96.33 s, as the Python port; still to be run once in MATLAB).
The competition files select it with `'method','spline','curv_filter',5`.

### The team's real FSA 2025 tracks
`lapsim/LR26_LapSim/tracks/FSA2025_autocross_real.csv`, `FSA2025_endurance_real.csv` and
`config/competition_fsa2025_real.m`. With the untuned LionsLTS settings our solver gives autocross
**49.26 s** (documentation 49.90 s) and endurance lap **96.33 s** (95.4 s). The result depends on the
curvature filter: autocross +1.1 % / −1.9 % and endurance lap +6.5 % / −5.6 % for 3 m / 8 m, so the filter
width is the first thing to correlate against the car. Select the file with `FSG_LAPSIM_COMPETITION=competition_fsa2025_real.m`
(name under `config/` or full path) or `LapSim.load(root, competition=...)`.

### Endurance energy budget
`fsdsim/perf/strategy.py` (`power_for_energy`, `sweep_power`, `run_with_cap`) and
`python3 tools/endurance_strategy.py --budget-kwh 5.9`:

* 5.5 kWh budget → cap 17.98 kW with the traced tracks of `competition_fsa2025.m` (17.46 kW with the real tracks), i.e. **our inferred 18 kW is a 5.5 kWh budget**;
* LionsLTS' 5.9 kWh → cap 20.4 kW, lap 94.5 s, total 1511.7 s, 80.1 points;
* uncapped, 16 laps need 7.73 kWh, more than the 7 kWh pack.

The efficiency score is *not* computed: the FS Rules 2026 formula (D9.4) could not be read. LionsLTS uses an
older formula, so a number from it would not be a 2026 number (**to confirm**).
A MATLAB mirror is in `analysis/lts_endurance_power.m` (run in GNU Octave on 9 Nov 2026: 17.98 / 17.46 / 20.39 kW,
the same caps as the Python tool; still to be run once in MATLAB).

### LionsLTS vehicle files → configurations
`python3 tools/import_lionslts_vehicle.py LR26_emrax.yml --save --run` converts a vehicle file into
`name=value` changes relative to `lr26_params.m`, saves it in `lapsim/configs/` (the launcher and dashboard
list it) and prints what could not be mapped. Saved: `LionsLTS_LR25`, `LionsLTS_LR26_as_committed`
(**a what-if: as committed, LR26.yml has almost no aero and AWD**), `LionsLTS_LR26_actual_aero`,
`..._emrax`, `..._fischer`, `..._himu`, `LionsLTS_experimental`. Test copies of three files are in
`tests/data/lionslts/`.

### MF4 logs
`fsdsim/logdata/mdf4.py` (NumPy only, no asammdf because it cannot be installed offline) reads channel
groups, linear / rational / table conversions and DT / DZ blocks. Values that are all-ones raw (0xFFFF,
"signal not available") become NaN, otherwise they appear as e.g. 33 535 rpm. `tools/mf4_summary.py file.mf4`
prints a statistics table; `--csv --signals NAME NAME@CAN3 --rate 100` writes a resampled CSV.
Checked on the real `D1F58_*.mf4` files: low-power test session, p99 about 14–17 kW per inverter, BMS ≤ 72 A,
bus about 520 V, motor ≤ about 11.7 krpm, torque set-points ≤ 25 Nm.

### CSV export with LionsLTS column names
`fsdsim/perf/export.py` and `tools/export_run_csv.py`: time, cumulative distance, raceline x/y, velocity,
reached ax, init ay, curvature, powertrain / resistance / drag / rolling force, consumed energy, energy storage
state, accumulator power. The file loads in Marple next to LionsLTS runs.

### Real tracks in the driverless simulator
`python3 tools/track_import.py line.csv --closed --cones out.csv --png` (or `--latlon` for GPS) fits the
line and puts cones around it. `tools/closed_loop_eval.py --missions trackdrive --track-file out.csv` drives it.
On the real FSA endurance layout (1312 m, tightest radius 3.6 m, 579 cones) the reference driver runs clean
(0 cones, 0 off course): in `map` mode with a **104.3 s** lap against LapSim's 96.3 s (the reference
driver's grip margin, not the layout), in `cones` mode 215 s (its 9 m/s cone-mode speed cap). The tool's
700 s timeout ends the run before 10 laps, which it reports as "not clean".

### Correlation with the real log (first result, 4 Nov 2026)
`python3 tools/log_correlate.py D1F58_decoded.mf4 --lapsim-effect` (module `fsdsim/logdata/powertrain.py`):

* **Motor + inverter efficiency (shaft / DC) is 0.855 median** (p10 0.80, p90 0.92) against LapSim's constant `eta_motor` 0.90. It depends on the point: about 0.90 at low torque and 9–12 krpm, falling to 0.77–0.84 at 10–14 Nm (the loss grows with torque squared). This assumes `act_Torque` is shaft torque in Nm and `act_Power` the DC power in kW; the log does not say (**to confirm**).
* **Effect on endurance** (real FSA track, 5.5 kWh budget): the power cap falls from 17.5 kW (eta 0.90) to 16.0 kW (eta 0.855), endurance 1546 s → 1560 s, 67.3 → 62.5 points. Indicative only: this log has torque ≤ 14 Nm, the car's motors go to 38 Nm, and the loss fit must not be extrapolated to it.
* **This log cannot fit the vehicle dynamics.** With the gear ratio and wheel size of `lr26_params.m` the logged torque explains only about 40 % of the measured acceleration, and at zero torque with no brake the speed falls by about 2 m/s² (20 → 9 m/s in two seconds), several times the roll-out of a car. The motors were not driving a car freely rolling on the road (stand, dyno or other load). There is also no GPS, IMU or wheel speed in the file. A road log with GPS or wheel speed (acceleration run, skidpad, coast-down) is needed for mass, resistances and the curvature filter.
* The BMS current times bus voltage is 17 % above the bus power in the log (sampling or scaling); the inverter DC power was used.

## 4. Discrepancies to confirm with the Performance department

1. Scoring maxima: LionsLTS 75 / 75 / 100 / 325 / 100 (old) against the official 2026 rules 50 / 50 / 100 / 250 / 75 used here. And the D9.4 efficiency formula.
2. Which LR26 vehicle file is the car: LR26.yml as committed has almost no aero, 270 kg and AWD; `LR26_emrax` has 290 kg, aero and rear drive.
3. `aero_eff`: LionsLTS multiplies the downforce only, our model drag and downforce. The importer divides the drag area by `aero_eff` to keep the same drag; the physical convention should be confirmed.
4. Motor power: LionsLTS files use 72 kW (= 80 kW × 0.9) in places, our `P_max` is 80 kW at the motor output.
5. `torque_e_motor_max` for AWD files: total or per axle? We put an identical motor set on the front.
6. `circ_ref` 1.5651 m (LionsLTS) against 1.57 m (ours); ignored as noise by the importer.
7. Endurance energy budget: 5.5 kWh (our inferred cap) or 5.9 kWh (LionsLTS)?
8. Curvature filter width for the real tracks (5 m untuned): correlate against the car's logs.
9. What was the D1F58 log (stand / dyno / car)? Units of `act_Torque` and `act_Power`; is there a log of a car on the road with GPS or wheel speed?
10. Which motor / inverter combination is the efficiency 0.855 of, and is `eta_motor` 0.90 in `lr26_params.m` from a map or an assumption?

## 5. Not adopted, and why
* The TUM code and its minimum-time OCP: LGPL-3.0 and marked "do not touch"; only ideas are reused.
* LionsLTS' outdated scoring constants.
* Its multi-gear tyre-per-axle model: our LapSim port has one tyre set; the importer warns when front and rear differ.

## 6. Roadmap ideas (see the Gantt plan)
1. **Minimum-time racing line** (own implementation of an OCP or a curvature-minimising line) feeding the DV driver, instead of the centre line.
2. **Correlation with real road logs**: speed and power from MF4 against LapSim on the same track; fit `curv_filter`, resistances; replace the constant `eta_motor` with an efficiency map (torque, speed) once a log with the full torque range exists. The tool for the powertrain part exists (`tools/log_correlate.py`).
3. **Vehicle identification** from the logs (resistance, motor efficiency map, regeneration).
4. **Friction / ggv maps** and temperature-dependent powertrain limits.
5. **Solver plug-ins**: one interface for QSS, dynamic and RL solvers so the same event definitions run through each.
6. **Marple workflow**: standard CSV per run, from LapSim, the simulator and the car.
7. Run the 10-lap endurance in the closed loop with the energy budget and compare with `strategy.py`.
