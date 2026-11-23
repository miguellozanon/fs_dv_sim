# Performance Lab (`perf/`)

A stand-alone lap time simulation for the Performance department. It runs only the LR26 LapSim and the tools
around it, on the Windows Python, with no WSL, ROS or Isaac Sim. The page opens in the browser at
http://localhost:8090.

## For everyone: how to start it

1. Double-click **`Performance Lab.bat`** in the project folder (or `perf\perf.bat`, or the **Performance Lab**
   shortcut on the desktop once it exists).
2. **The first time on a PC** it asks two questions and answers them itself if you say yes (internet needed):
   * install **Python 3.12** for your Windows user, with no administrator rights (winget, else the python.org
     installer);
   * install the packages **numpy, scipy and pyyaml** (pip, for your user), and **casadi** (optional: the
     optimal-control models of the Methods page).
3. The **Performance Lab window** opens. It checks everything, starts the Lab and opens it in the web browser.
   * Next time: double-click again, or press **Start and open**.
   * In the browser, **Guide** (top right) explains every page in plain words, with step-by-step recipes.
   * **? Help** in the top bar opens the guide for the page you are on.

**The launcher window** (`perf/launcher/perf_launcher.pyw`):

| Part | What it does |
|---|---|
| Start and open / Open in the browser / Stop | runs the Lab in the background (no black console window) |
| Is everything ready? | checks Python, the packages, CasADi (optional: a grey mark, the Lab starts without it), the car data (`lapsim/LR26_LapSim`), the setups folder, the port, and does one test run (shows the points) |
| Repair | installs missing packages, including the optional CasADi (asks first) |
| Back up setups / Restore from a backup | one zip in `Documents\Performance Lab` with every saved setup and efficiency map. Restore never overwrites your files: a different setup comes back as "... (restored)" |
| Open the user guide / Read the technical paper | the guide in the browser; `docs/paper/paper2_performance_lab.pdf` |
| Save a help report | a zip with the checks and logs (no car data), to send to whoever helps |
| Put a shortcut on the desktop | "Performance Lab" on the desktop and in the Start menu, with the team icon |
| Settings | start the Lab when the window opens (on), stop it when the window closes (on), port (8090) |

Settings and logs live in `%APPDATA%\LionsRacing\PerformanceLab`, outside the project, so an update of the project
keeps them.

**Other ways in:**

* **The DV launcher:** **RL & Performance → Open the Performance Lab** starts the same window on Windows.
* **From a terminal:** `python perf/app.py [--port 8091] [--no-browser]`, or `perf\perf.bat --console` for the Lab
  in the console window without the launcher.
* The Lab listens on this computer only (127.0.0.1).

## Keeping it alive (no Performance member or programmer around)

* **The car changes:** edit `lapsim/LR26_LapSim/config/lr26_params.m`, the same file MATLAB uses (each line is
  `veh.<name> = <value>;`). The Lab reads it again by itself; reload the page. The Setup page lists every name
  with its meaning.
* **New competition or tracks:** copy `competition_fsa2025.m` to `competition_<event><year>.m` and change its
  values; it appears in the top bar.
* **Back up** the setups before big changes (launcher window), and keep the `lapsim/LR26_LapSim` folder on the
  team drive. It is not in git: it is confidential.
* **New PC:** copy the project folder, double-click `Performance Lab.bat`. That is all.
* **Something broken:** launcher window → **Save a help report**, and send the zip with a description to the
  team's software contact (or the driverless team). The **Check again** results usually say what is wrong and what
  to do.
* **How it computes:** the guide (plain words), then the technical paper `docs/paper/paper2_performance_lab.pdf`
  (every equation and check).

**The car** comes from `lapsim/LR26_LapSim`, the same as for the simulator and MATLAB:

* `config/lr26_params.m` for the vehicle;
* `config/competition_*.m` for the events, tracks, driver utilisation and scoring.

The files are re-read when they change. The app never writes them.

## What it does

| Page | Content |
|---|---|
| Overview | All events with time, lap time, points, average and top speed, energy and driver utilisation, plus the change against the baseline. Speed, ax, ay and power traces over distance. A track map coloured by speed. CSV export. |
| Setup | The parameter table (the same as the dashboard's Vehicle setup page, `fsdsim/perf/schema.py`). Free changes such as `mu*1.02` or `autocross.util_y=0.85`. Saved setups. |
| Studies | **Sensitivity**: ±δ per parameter, a tornado chart and the time change per event (`lts_sensitivity_table`). **Sweep**: one parameter, total and per-event points and times, speed traces. **2-D map**: points over two parameters, with the best point. **Compare**: the active setup against up to seven saved ones, with speed traces and time difference over distance. |
| Transient | The event driven by the simulator's dynamic vehicle model and reference driver, against the QSS lap. Shows the time lost along the lap, steering, sideslip, lateral offset and ay. Yaw inertia and the steering limit act here; the QSS LapSim does not have them. **Line smoothing**: the driver's line is a smooth curve within this distance of the track's points (automatic: 0.25 m for measured x, y tracks, none for curvature tables). |
| Methods | The same car with several lap time methods, to choose the one that fits the question (table on the page). **One method, one event**: minimum-time optimal control (point mass, single track, double track ± torque vectoring, fixed or free line) with speed, accelerations, line, steering and sideslip. **Compare the methods**: QSS, point-mass QSS, transient and the OCP models on one event, baseline and one change; whether they agree on its sign. **Robustness**: Monte-Carlo on QSS over the uncertain parameters, spread of the points and probability that a change gains. Details: [PERF_METHODS.md](PERF_METHODS.md). |
| Energy | Where the energy of an endurance lap goes (drivetrain losses, drag, rolling, friction brakes, recuperation). Endurance points, energy and lap time against the power cap. The cap for a given battery budget (LionsLTS `optimize_power` idea). The efficiency factor T²·E. **Recuperation**: endurance energy, energy recovered and, with a battery budget, the power cap and endurance points for each maximum regen power. |
| ggV | Drive and brake limits against lateral acceleration per speed. Choose the tyre limit or an event's driver utilisation. |
| Logs | Upload `.mf4` or `.csv` (stored in `perf/uploads/`, kept out of git), or use LapSim's reference traces. Map the speed, ax and ay channels; motor rpm can be turned into speed through the gear and tyre. Correlation on distance, with the offset searched within ±30 m: speed RMSE and bias, time difference along the lap. **Powertrain** (MF4 with the inverter channels): efficiency table, loss model, roll-out check. The result can be saved as `lapsim/configs/eta_map_<name>.json`. **Identify the car from the log** ([LOGGING_SPEC.md](LOGGING_SPEC.md)): *coast-down* → rolling resistance and c_w; *grip use (ggV)* → driver utilisation util_y, util_x; *steady cornering* → grip scale; *track from GPS* → a track file (x, y every 0.5 m) in `perf/uploads/tracks/`. Results are suggestions next to the setup's values; nothing is written to `lr26_params.m`. Extra channels: yaw rate, latitude, longitude, brake pressure. |

**Context bar (top of every page):**

* the competition (`competition_fsa2025.m` or `_real.m`);
* the active setup;
* the motor efficiency: constant `eta_motor`, or a saved map.

Everything the app computes uses these.

**Setups** are the dashboard's saved configurations (`lapsim/configs/<name>.json`). A setup saved in one app
appears in the other. An existing setup is replaced only when **replace** is ticked.

Long studies run one at a time in the background, with a progress bar and **Cancel**. Every result can be
exported as CSV.

## How the numbers relate

* **QSS results** come from `fsdsim.perf.lapsim`, the line-by-line port of LR26_LapSim. With no changes the
  total is 164.35 points, the same as MATLAB (tests/test_vehicle_setup.py). The points cover the dynamic events
  only.
* **Sensitivity and sweeps** use the definitions of `analysis/lts_sensitivity_table.m` and `lts_sweep*.m`.
  "Improving direction" applies, and mass is also given per kg.
* **The transient lap** uses:
  * the vehicle: `fsdsim.dynamics.VehicleModel` in performance mode, with the event's LapSim car at the tyre
    limit (endurance torque scale and power cap included);
  * the driver: the reference driver with grip utilisation = the event's `util_y`, longitudinal
    `util_x / util_y`.
  * Closed tracks: two laps from standstill, and the second lap is timed (flying lap as in LapSim).
  * Open tracks: standstill to the end.
  * Each event takes 0.2 to 10 s to compute.

## Checked on 13 Nov 2026 (container, no Windows PC)

Baseline, FSA 2025 tracks (documentation version):

| Event | QSS | Transient | Notes |
|---|---|---|---|
| Acceleration | 4.66 s | 4.83 s | |
| Skidpad | 5.15 s | 5.38 s | |
| Autocross | 49.94 s | 53.71 s | lateral offset ≤ 0.5 m |
| Endurance lap | 95.4 s | 100.0 s | |

On the FSA 2025 real tracks the transient endurance lap used to be 112 s against 96 s for QSS: the driver followed
the measured x, y points with their point-to-point noise. Since 15 Nov 2026 its line on measured tracks is
smoothed (within 0.25 m of the points): autocross 53.5 s (QSS 49.3 s), endurance lap 103.5 s (QSS 96.3 s), never
off the track. The documentation tracks are unchanged (no smoothing). The lateral offset shown is measured from
the smoothed line.

The transient endurance lap uses ~35 % less energy than QSS (0.22 against 0.34 kWh). The driver's targets sit
below the limit, so it cruises at part throttle and brakes less. This is a property of the reference driver,
not a measured fact.

The documentation endurance track (`FSA2025_endurance.csv`) is a curvature table. Its x, y end 17 m from the
start. LapSim only uses the curvature, so this does not matter there. For the transient lap the gap is spread
along the lap, about 1 cm per metre (`perflab.transient.close_loop`). The page says when it did this.

Documentation trace against the port (Logs page, `FSA2025_autocross_doc_speed.csv`): speed RMSE 3.2 km/h, lap
time 49.30 against 49.30 s.

**Not tested:**

* `perf.bat` on Windows: its Python / package installation (winget, python.org installer, pip), and the desktop
  shortcut (PowerShell). It was written for Windows `cmd` and could not be run here. The launcher window and all its
  functions were tested on Linux with Tk under a virtual display;
* the pip installation step, including CasADi; CasADi / IPOPT on Windows (the optimal-control models were checked
  on Linux with CasADi 3.8.2);
* a real car log with GPS or IMU speed (the team's only log, D1F58, has none);
* uploads above a few MB.

Python 3.9 was checked with numpy, scipy and pyyaml from PyPI.

## Files

```
Performance Lab.bat      double-click start in the project folder (calls perf/perf.bat)
perf/perf.bat            Windows start: installs Python / packages if needed, opens the launcher window (CRLF)
perf/launcher/           launcher window (perf_launcher.pyw: Tk), its logic (core.py: checks, start/stop,
                         backup/restore, help report, shortcuts), icon and logo
perf/app.py              HTTP server (stdlib), API, jobs
perf/perflab/lab.py      LapSim folder, competitions, setups, efficiency maps, runs + traces
perf/perflab/studies.py  sensitivity, sweep, 2-D map, compare
perf/perflab/energy.py   energy breakdown, power cap / budget, ggV
perf/perflab/transient.py transient lap (VehicleModel + reference driver)
perf/perflab/logs.py     log upload / reading / correlation, powertrain efficiency map
perf/perflab/jobs.py     background jobs with progress and cancel
perf/web/                page (index.html, perf.js, charts.js: own SVG charts, works offline)
perf/perflab/ocp.py      minimum-time optimal control (CasADi + IPOPT): point mass, single / double track
perf/perflab/methods.py  method comparison, Monte-Carlo robustness, which method for which question
perf/perflab/ident.py    parameters from logs: coast-down, grip use, steady cornering, track from GPS
tests/test_perf_app.py, tests/test_perf_launcher.py, tests/test_perf_ocp.py   tests
tests/test_testcar.py, tests/fixtures/testcar_lapsim/   the Lab on a fictional car (CI, no confidential data)
```

API (POSTs need the header `X-Perf-Lab: 1`):

* `GET /api/info`
* `POST /api/run {changes, competition}`
* `POST /api/job {kind: sensitivity|sweep|sweep2d|compare|energy|regen|transient|ocp|methods|monte_carlo|ident, args}`, then
  `GET /api/job/<id>`
* `GET /api/methods` (the methods, the guidance table, whether CasADi is installed)
* `POST /api/ggv`
* `POST /api/logs/{upload,inspect,correlate,powertrain,save_map,save_track}`
* `POST /api/setup/save`
* `POST /api/eta_map`

**Environment variables** (tests, demos): `FSG_LAPSIM_DIR` another LapSim folder (e.g. the test car
`tests/fixtures/testcar_lapsim`), `PERF_LAB_UPLOADS` another folder for uploaded logs.
