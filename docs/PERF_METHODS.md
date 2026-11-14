# Lap time methods in the Performance Lab

The Performance Lab (`perf/`, see [PERF_APP.md](PERF_APP.md)) solves the same LR26 with several lap time methods.
This file gives, for each method, what it is for, what it assumes, its equations and how it is solved, its limits,
and how it was checked. The **Methods** page of the Lab runs them and compares them. The technical paper
(`docs/paper/paper2_performance_lab.pdf`, section "Lap time methods") has the same content in paper form.

All methods read the same car (`lapsim/LR26_LapSim/config/lr26_params.m`), the same competition file (tracks, driver
utilisation, scoring), and give the same outputs: event time, points, and speed along the track.

## Which method for which question

| Question | Method | Why |
|---|---|---|
| Ranking of setup changes, sensitivities, sweeps (hundreds of runs) | **LapSim QSS** | 0.2 s for all events, equal to MATLAB, driver utilisation calibrated against logs |
| Is a gain larger than what we do not know about the car? | **Monte-Carlo on QSS** | spread of the points, and probability that a change gains |
| Is the QSS solver itself the limit? | **OCP point mass, fixed line** | same physics as QSS, solved as one optimisation: agrees within 0.3 % |
| Yaw inertia, steering limits, load transfer, tyre shape, balance | **OCP single / double track** | these quantities do not exist in QSS; the OCP gives the limit of a perfect driver |
| Torque vectoring, other control allocation | **OCP double track + torque vectoring** | needs individual wheels |
| How a (DV or human) driver gets on with the car, stability | **Transient lap** | closed loop with a driver; realistic behaviour, not the limit |
| Racing line on a known track width | **OCP with free line** | only on a true centre line with its width (not the FSA tables, see below) |
| First-order check, comparison with other teams' simple tools | **QSS point mass (friction circle)** | the textbook model; shows what LapSim's ggV adds |

Rule of thumb: **decide with QSS, check with the OCP, test robustness with Monte-Carlo.** A change is worth it when
the methods agree on its sign (the Methods page says so) and the Monte-Carlo probability of a gain is high (> 90 %).

## Common definitions

| Symbol | Meaning |
|---|---|
| s | distance along the event's line (the track table), m |
| κ(s) | curvature of the line, 1/m (left turn > 0) |
| n, ξ | lateral offset of the car from the line (left > 0), m; heading of the car relative to the line, rad |
| v, vx, vy, r | speed; body longitudinal and lateral velocity, m/s; yaw rate, rad/s |
| ax, ay | longitudinal and lateral acceleration, m/s² |
| m, m_eff | mass; mass including rotating inertia (`m·e_i`) |
| F_res(v) | drag and rolling resistance `k_D v² + f_roll (m g + k_DF v²)` |
| util_x, util_y | the event's driver utilisation of the tyre potential (competition file) |

Event time T: the time of the run (acceleration, skidpad, autocross), or `laps × lap time` for the endurance
(flying laps, as LapSim). Points: the competition's scoring formula from T, identical for every method.

---

## 1. LapSim QSS (quasi-steady-state)

**Purpose.** The team's reference: the line-by-line port of the MATLAB LR26_LapSim (`fsdsim.perf.lapsim`). All
studies (sensitivity, sweeps, 2-D maps, compare, energy) use it. It gives 164.35 points with the baseline, the same as
MATLAB.

**Hypotheses.**

* The car is in a steady state at every point: the accelerations change instantly (no yaw inertia, no transient
  tyre forces, no steering limit).
* The car follows the given line exactly (the track table is the driven line).
* Tyre limits per axle with quasi-static load transfer, load-sensitive friction; the combined limit is an ellipse.
* Motor torque, power and speed limits; endurance torque scale and power cap.
* The driver uses `util_x`, `util_y` of the tyre potential.

**Equations.**

1. Cornering speed at every point: the largest v with `v² |κ| ≤ ay_max(v)` from the axle model (bisection).
2. Forward pass: `v_{i+1}² = v_i² + 2 ds ax_max(v_i, ay_i)`, limited by the cornering speed.
3. Backward pass: the same with the braking limit.
4. The speed is the minimum of both passes. The time is `Σ ds / v̄`.

Closed tracks are solved as three laps in a row and the middle lap is kept (no start effect).

**Limits.** No transients, no steering or yaw limits, line fixed. Its optimism or pessimism against reality is absorbed
by the utilisation factors, which were fitted to logs: a change that acts through transients (yaw inertia, damping,
steering ratio) cannot be evaluated.

**Checked.** 4.662 / 5.154 / 49.944 / 1526.97 s, 164.35 points, equal to MATLAB (tests/test_vehicle_setup.py).

## 2. QSS point mass (friction circle)

**Purpose.** The textbook model, used by many teams as a first lap time tool. Here as a reference, to show what the
LapSim ggV adds.

**Hypotheses.** As QSS, but the combined limit is a circle with a single radius taken from the ggV corner of the car,
not LapSim's axle ellipse. It is the `point_mass` solver of `fsdsim.perf.solvers`, run through LapSim's events and
scoring.

**Limits.** Optimistic where power or load transfer limit the car, pessimistic where the ellipse is wider than the
circle. Not for decisions.

**Checked.** 4.704 / 5.010 / 50.918 / 1561.69 s, 154.24 points (baseline), against 164.35 for LapSim QSS.

## 3. Transient lap (time simulation with a driver)

**Purpose.** The simulator's dynamic vehicle model (`fsdsim.dynamics.VehicleModel`, performance mode) driven in closed
loop by the reference driver (`perflab.transient`). It shows how a driver gets on with the car: time lost along the
lap, steering, sideslip, lateral offset.

**Hypotheses.**

* Single-track vehicle with yaw inertia, Pacejka tyres, steering angle and rate limits, powertrain limits.
* A driver model: pure-pursuit steering and a speed target from the track curvature at grip utilisation `util_y`
  (longitudinal `util_x / util_y`).
* Explicit time integration, step 0.02 s.

**Equations.** The vehicle model's ODEs (see paper I) integrated in time; the driver closes the loop.

**Limits.** The result depends on the driver model: it is not the limit of the car. The driver's targets sit below
the limit, so the transient lap is slower (+3.6 % on the skidpad) and uses less energy than QSS. The driver works on
the raw x, y of the track: on the "real" FSA tracks it is too slow (see PERF_APP.md).

**Checked.** 4.83 / 5.38 / 53.71 / 100.0 s per endurance lap (baseline, documentation tracks).

## 4. Minimum-time optimal control (OCP)

`perf/perflab/ocp.py`. The question "what is the fastest way round for a perfect driver?" written as one
optimisation over the whole event, instead of the point-by-point limits of QSS:

```
minimise   T = ∫ dt = ∫ (1 / ṡ) ds
subject to the vehicle's equations of motion (in s), the tyre, powertrain and steering limits,
           |n| ≤ (track width − car width) / 2  (free line)  or  n = 0  (fixed line),
           start conditions (open tracks) or periodicity (closed tracks).
```

### Common to all OCP models

* **Curvilinear coordinates.** The car's position is s (along the line) and n (lateral offset); ξ is its heading
  relative to the line. With the speed components vx, vy:

  `ṡ = (vx cos ξ − vy sin ξ) / (1 − n κ)`, `ṅ = vx sin ξ + vy cos ξ`, `ξ̇ = r − κ ṡ`.

  Every state derivative is divided by ṡ to integrate in s, so the grid is fixed and the event time is the
  objective.
* **Discretisation.** Uniform grid, ds = 1 m (curvature averaged per metre, heading kept). Trapezoidal collocation:
  `x_{k+1} − x_k = ds/2 · (x'_k + x'_{k+1})`. Closed tracks: the last node wraps to the first (periodic: the car
  starts the lap as it ends it, i.e. a flying lap).
* **Solver.** CasADi (`Opti`) builds the problem with exact derivatives, IPOPT (interior point, MUMPS) solves it.
  Default tolerance 1e-6, at most 3000 iterations.
* **Hierarchy of initial guesses.** QSS speeds → point mass → single track → double track. Every model starts from
  the solution of the simpler one. Without that the dynamic models often do not converge.
* **Same car as QSS.** The tyre potentials, load sensitivity, load transfer formulas, aero, drag, rolling resistance,
  motor torque / power / speed limits and the event's utilisation are taken from the LapSim parameters; yaw inertia,
  steering limits and the Pacejka shape (B, C, E) from the simulator's `vehicle.yaml` (the same car as the transient
  lap).
* **Grid effect.** `qss.t_lap_same_grid` is LapSim QSS on the same 1 m grid. The difference to the OCP is the
  method; the difference between QSS on its own grid and on the 1 m grid is the grid (autocross: 49.94 against 49.85 s).
* **Open tracks.** The point mass starts at v = 0. The dynamic models start at 1 m/s (the equations divide by the
  speed); the time to reach 1 m/s at the traction limit is added (`t_start_correction`, ~0.07 s).
* **Failure.** When a stage does not converge, `ok` is false and no time or points are given (the traces of the last
  iterate are returned for diagnosis only).

### 4a. Point mass on the ggV envelope

**Purpose.** A check of QSS: the same physics solved as one optimisation. With the fixed line it must agree with QSS;
if it did not, the QSS solver (not the car model) would be the limit.

**States** n, ξ, v at the nodes. **Inputs** ax and the path curvature κ_p, constant over each interval.

**Equations.**

* `v_{k+1}² − v_k² = 2 ds h ax`, `ξ_{k+1} − ξ_k = ds (κ_p h − κ̄)`, n from the trapezoid of `(1 − nκ) tan ξ`,
  with `h = (1 − nκ)/cos ξ` (path length per unit of s).
* **Envelope** (fitted to LapSim's ggV at 18 speeds, `Envelope`):
  * `ay_max(v) = a0 + a1 v + a2 v²`, braking `ab(v)` and traction `at(v)` quadratics;
  * combined: `(|ay| / ay_max)^p + (ax / a_lim)^p ≤ 1` with separate exponents p for braking and traction (fitted,
    1.9 and 2.1 for the baseline, so close to an ellipse); smooth powers (`(z² + ε)^{p/2}`) keep the Hessian finite.
* **Powertrain** exact: `m_eff ax + F_res ≤ F_T` (motor torque at the wheel, per driven axle), `(m_eff ax + F_res) v
  ≤ P η_g`, `v ≤ v_max`.
* Limits hold at both ends of each interval. (With node inputs the path curvature zig-zags around the track
  curvature and the optimiser "beats" the envelope: the interval form removes this.)

**Hypotheses.** As QSS, plus the fitted envelope (rms error of the fit is in the result) and a constant-input
interval.

**Limits.** No yaw, no steering; the envelope is a fit (the endurance power cap ends the steady-state speeds below
the motor speed limit: only feasible speeds are fitted).

**Checked (fixed line, same 1 m grid as QSS).**

| Event | QSS (1 m grid) | OCP point mass | Difference | Solve |
|---|---|---|---|---|
| Acceleration | 4.662 s | 4.669 s | +0.15 % | 0.1 s |
| Skidpad | 5.154 s | 5.142 s | −0.23 % | 0.1 s |
| Autocross | 49.854 s | 49.79 s | −0.13 % | 1.3 s |
| Endurance lap | 94.471 s | 94.21 s | −0.28 % | 2 s |

The OCP is slightly faster because it optimises over the whole lap and the envelope is a smooth fit of the axle
model; the agreement shows that LapSim's forward/backward passes find the optimum of its own model.

### 4b. Dynamic single track (bicycle)

**Purpose.** The fastest lap of a perfect driver with the vehicle dynamics QSS leaves out: yaw inertia, sideslip,
steering angle and rate limits, the tyre slip characteristic, front/rear balance with load transfer.

**States** n, ξ, vx, vy, r, δ. **Inputs** steering rate δ̇ and the longitudinal tyre forces Fx,f, Fx,r (in units of
m g). **Algebraic variables** ax, ay (in g), which set the load transfer.

**Equations.**

* Slip angles: `α_f = δ − atan((vy + l_f r)/vx)`, `α_r = −atan((vy − l_r r)/vx)`.
* Lateral force: `Fy = py · sin(C atan(Bα − E(Bα − atan Bα)))` (Magic Formula shape), with the axle potential py =
  sum of the two wheels' load-sensitive potentials `Fz (μ + dμ/dFz (Fz − Fz0)) · util`.
* Wheel loads: quasi-static, the LapSim formulas (longitudinal transfer `m ax h/L`, lateral transfer per axle
  `m ay h/s · share`, aero balance). ax and ay enter as algebraic variables with
  `ax = (ΣFx_body − F_res)/(m g)`, `ay = ΣFy_body/(m g)`.
* Combined slip: `(Fx/px)² + S² ≤ 1` per axle (friction ellipse, S the normalised lateral force).
* Body equations: `v̇x = (ΣFx − F_res)/m_eff + vy r`, `v̇y = ΣFy/m − vx r`, `ṙ = Mz/Iz`, `δ̇` = input.
* Powertrain: driven-axle force ≤ F_T, `Fx_drive vx ≤ P η_g`, no front drive force on an RWD car, `vx ≤ v_max`;
  `|δ| ≤ δ_max`, `|δ̇| ≤ δ̇_max`.
* Small regularisation of input changes (1e-3) to avoid chattering; it changes the time by less than 1 ms.

**Hypotheses.** Rigid body in the plane, quasi-static load transfer (no roll or pitch dynamics), tyre forces follow
the slip angle without relaxation, longitudinal force as an input (wheel speeds not modelled), a perfect driver who
knows the whole track.

**Limits.** Gives the limit of the car, not what a driver achieves (compare with the transient lap). Compute time
30–40 s on autocross and endurance. Results depend on the Pacejka shape and yaw inertia from `vehicle.yaml`, which are
less well known than LapSim's parameters.

**Checked (fixed line).**

| Event | QSS | Single track | Compute (all stages) |
|---|---|---|---|
| Acceleration | 4.662 s | 4.662 s | 0.1 s |
| Skidpad | 5.154 s | 5.231 s (+1.5 %) | 0.1 s |
| Autocross | 49.94 s | 52.09 s (+4.3 %) | 39 s |
| Endurance lap | 95.44 s | 99.97 s (+4.7 %) | 36 s |

Slower than QSS because yaw and steering need time and the Pacejka curve falls beyond its peak; close to the
transient lap (53.7 s / 100.0 s), which is driven at the event's utilisation.

### 4c. Dynamic double track (± torque vectoring)

**Purpose.** As the single track with four wheels: lateral load transfer reduces the grip of the inner wheels, each
wheel has its own slip angle, and the yaw moment of different left / right forces is available (torque vectoring).

**Differences to 4b.** Inputs are the four longitudinal tyre forces. Each wheel has its slip angle from its position
(±track/2), its potential from its own load, and its own friction ellipse. Without torque vectoring the left and
right forces of each axle are equal (open differential with equal torque); with torque vectoring they are free within
the per-motor torque limit (half the axle's force) and the total power limit.

**Hypotheses / limits.** As 4b. The torque-vectoring result is the best possible allocation for a perfect
controller: an upper bound of what a real TV controller gains.

**Checked (fixed line).**

| Event | Double track | + torque vectoring | Compute (all stages) |
|---|---|---|---|
| Acceleration | 4.662 s | – | 0.1 s |
| Skidpad | 5.398 s | 5.282 s | 0.2 s |
| Autocross | 54.03 s | 52.07 s | 51 / 65 s |
| Endurance lap | 102.72 s | – | 49 s |

Torque vectoring gains 0.12 s on the skidpad and 2.0 s on the autocross: the yaw moment it creates replaces the
grip the inner wheels lose with load transfer.

### Free line (all OCP models)

With `fixed_line = false` the car may use the track width around the given line (width set in the app, default
3.5 m; the car's centre stays `(width − car width)/2` from the line). The OCP then finds the racing line.

**Use it only on a true centre line with a known width.** The FSA tables are already a driven line: the
documentation tracks are curvature tables with small wiggles, the "real" tracks the team's driven line ("do not
optimise again with full width"). On them a free line straightens a line that is already optimised and the result is
far too fast (autocross point mass 42.6 s, endurance 72.9 s per lap: −24 %). The result carries a warning.

## 5. Monte-Carlo robustness (on QSS)

**Purpose.** A lap time model is only as good as its parameters. The Monte-Carlo study asks whether a result holds
when the parameters we do not know exactly are off within their uncertainty, and whether a change gains on (almost)
all of them.

**Method.**

1. Draw N parameter sets: each uncertain parameter multiplied by `1 + σ·z`, z standard normal, clipped at ±3σ. Default
   σ (1σ, relative): μ 3 %, mass 1 %, c_L 5 %, c_D 5 %, power 2 %, driver utilisation util_y 2 % (one factor on all
   events). Editable in the app.
2. Run LapSim QSS for every draw: distribution of the total and per-event points (mean, σ, 5/50/95 %).
3. With a change: run each draw again with the change (paired: the same draws). The gain distribution, its mean with
   standard error, and the probability of a gain `P(gain > 0)`.

**Hypotheses.** Independent, normal parameter errors; the QSS model structure is right; the scoring's reference times
are fixed.

**Limits.** Only parameter uncertainty, not model error (use the method comparison for that). The spread of the
absolute points is large (σ ≈ 19 points for the defaults) because the scoring is steep around the reference times; the
paired gain is much sharper (5 kg less: 3.9 ± 0.3 points, P(gain) = 100 %). 200 draws take ~90 s (with a change).

## 6. Method comparison

`methods.compare` runs the chosen methods on one event for the active setup and with one change. It reports per
method the time, the difference to LapSim QSS, the time and point change of the change, the compute time, and whether
all methods agree on the sign of the change.

Skidpad, `mass-5` (baseline):

| Method | Time | vs QSS | Δ time of −5 kg | Compute |
|---|---|---|---|---|
| LapSim QSS | 5.154 s | 0 | −8.1 ms | 0.0 s |
| QSS point mass | 5.010 s | −2.8 % | −9.8 ms | 0.0 s |
| Transient lap | 5.375 s | +4.3 % | −7.5 ms | 0.9 s |
| OCP point mass | 5.142 s | −0.2 % | −8.3 ms | 0.4 s |
| OCP single track | 5.231 s | +1.5 % | −7.9 ms | 0.6 s |
| OCP double track | 5.398 s | +4.7 % | −6.3 ms | 0.9 s |
| OCP double track + TV | 5.282 s | +2.5 % | −7.4 ms | 0.8 s |

All methods agree on the sign; the size of the gain differs by about ±20 % from QSS between methods, which is the
model uncertainty of the gain.

## Implementation

| File | Content |
|---|---|
| `perf/perflab/ocp.py` | Track, Envelope, Car, the point-mass and dynamic OCPs, `run()` |
| `perf/perflab/methods.py` | method list, `compare`, `monte_carlo`, `guidance` |
| `perf/app.py` | jobs `ocp`, `methods`, `monte_carlo`; `GET /api/methods` |
| `perf/web/` | the Methods page |
| `tests/test_perf_ocp.py` | tests (OCP tests skip without CasADi) |

**CasADi** (`pip install casadi`, ~40 MB, includes IPOPT) is optional: without it the Lab runs and the Methods page
offers QSS, the point-mass QSS, the transient lap and Monte-Carlo. The launcher's check lists it as optional and
**Repair** installs it; `perf.bat` installs it with the other packages.

From Python:

```python
from perflab.lab import Lab
from perflab import ocp, methods
lab = Lab()
r = ocp.run(lab, "autocross", "single_track")            # fixed line; r["T"], r["points"], r["trace"]
c = methods.compare(lab, "skidpad", "mass-5", ["qss", "ocp_point_mass", "ocp_single_track"])
m = methods.monte_carlo(lab, 200, "mass-5")
```

## Not tested

* CasADi / IPOPT on Windows (checked on Linux, CasADi 3.8.2, Python 3.12); the `perf.bat` installation of CasADi.
* The free-line OCP against a real centre line with a measured width (no such track file exists yet).
* OCP results against car data: no log with GPS / IMU speed exists.
