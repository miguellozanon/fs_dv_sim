# Technical papers

| File | Content |
|---|---|
| `paper1_dv_simulator.pdf` | fsg-dv-sim: architecture, vehicle model (LapSim force limits), rules, sensor models, Isaac Sim link, perception and SLAM, reference driver, racing line, RL, verification, results |
| `paper2_performance_lab.pdf` | Performance Lab: LapSim vehicle model, QSS solver, tracks, scoring, sensitivities and sweeps, energy and power cap, ggV, transient lap, log correlation, efficiency maps, results |

**Rebuild:**

```bash
python3 docs/paper/figures/make_figures.py      # figures + numbers.json (about 10 min; needs lapsim/LR26_LapSim, and CasADi for the lap time methods)
cd docs/paper && latexmk -pdf paper1_dv_simulator.tex paper2_performance_lab.tex && latexmk -c
```

* `python3 docs/paper/figures/make_figures.py dv` (or `perf`) redoes one paper's figures only; `methods` only paper II's
  section on the lap time methods (OCP, comparison, Monte-Carlo; ~6 min).
* Every number in the papers comes from the code in this repository. They are written to
  `figures/numbers.json` when the figures are made.
* Change the author line in the `\author{...}` of each `.tex`.

**Confidentiality.** The papers contain LR26 parameter values from `lr26_params.m`, the same values that
docs/PERFORMANCE.md already lists. Check with the team before publishing the PDFs outside the team.
