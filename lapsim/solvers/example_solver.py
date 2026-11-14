"""Example solver plug-in for the Python LapSim (fsdsim.perf.solvers). Copy this file and change it.

Every .py file in this folder that has a function register(registry) is loaded automatically
(tools/solver_compare.py lists them). A solver gets the vehicle, the curvature profile and the grid spacing and
returns the speed at every point:

    solve(veh, k, ds, v0, v_end) -> (v, v_corner, limit)

    veh     fsdsim.perf.PerfParams: every lr26_params.m value as an attribute (veh.m, veh.muy, veh.P_max, ...)
    k       curvature [1/m] at every grid point (numpy array; sign = direction)
    ds      grid spacing [m]
    v0      start speed [m/s], NaN = free (flying lap)
    v_end   end speed [m/s], NaN = free
    v       speed [m/s] at every point; v_corner the cornering limit there; limit: anything or None

This example is LapSim's own solver with a safety factor on the cornering speed, to show the mechanics.
"""

import numpy as np


def solve_cautious(veh, k, ds, v0, v_end, factor=0.95):
    """LapSim QSS with 95 % of the cornering speed (example plug-in)."""
    from fsdsim.perf.lapsim import qss_solve

    v, vc, limit = qss_solve(veh, k, ds, v0, v_end)
    return np.minimum(v, factor * vc), factor * vc, limit


def register(registry):
    registry.add("example_cautious", solve_cautious)
