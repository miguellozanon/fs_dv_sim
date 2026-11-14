"""Speed-profile solvers for LapSim's events, behind one interface.

A solver is a function

    solve(veh: PerfParams, k: ndarray, ds: float, v0: float, v_end: float) -> (v, v_corner, limit)

with the curvature profile k [1/m] at spacing ds, the start / end speeds (NaN = free) and, as results, the speed
at every point, the cornering-limit speed and (optional, may be None) which limit set it. `run_event` builds the
track, calls the solver and computes times, energy and points from v, so a new solver is compared with the
built-in one on the same events with the same scoring:

    from fsdsim.perf import solvers, lapsim
    solvers.register("mine", my_solve)
    ls = lapsim.LapSim.load(".../LR26_LapSim");  ls.run(solver="mine")

Built in: "qss" (the port of LapSim's lts_qss_solve) and "point_mass" (friction circle from the ggv corner of the
car: the usual first-order check, optimistic where power or load transfer limit the car).
Plug-ins: every lapsim/solvers/*.py (or a file / folder in $FSG_SOLVER_PLUGINS) with a function
`register(registry)` is loaded by `load_plugins()`; see lapsim/solvers/example_solver.py.
"""

from __future__ import annotations

import importlib.util
import math
import os

import numpy as np

_REGISTRY: dict = {}
_loaded = False


def register(name: str, fn, description: str = "") -> None:
    _REGISTRY[name] = (fn, description or (fn.__doc__ or "").strip().split("\n")[0])


def names() -> dict:
    load_plugins()
    return {n: d for n, (_, d) in _REGISTRY.items()}


def get(name: str):
    load_plugins()
    if name not in _REGISTRY:
        raise KeyError(f"unknown solver '{name}'. Known: {', '.join(sorted(_REGISTRY))}")
    return _REGISTRY[name][0]


def load_plugins(paths=None) -> list[str]:
    """Import plug-in files once; returns the files loaded."""
    global _loaded
    if _loaded and paths is None:
        return []
    _loaded = True
    cands = list(paths or [])
    if paths is None:
        env = os.environ.get("FSG_SOLVER_PLUGINS", "")
        cands += [p for p in env.split(os.pathsep) if p]
        from .lapsim import find_lapsim_dir  # noqa: PLC0415

        root = find_lapsim_dir()
        if root:
            cands.append(os.path.join(os.path.dirname(root), "solvers"))
    files = []
    for c in cands:
        if os.path.isdir(c):
            files += [os.path.join(c, f) for f in sorted(os.listdir(c)) if f.endswith(".py") and not f.startswith("_")]
        elif os.path.isfile(c):
            files.append(c)
    done = []
    for f in files:
        spec = importlib.util.spec_from_file_location("fsg_solver_" + os.path.splitext(os.path.basename(f))[0], f)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if hasattr(mod, "register"):
            mod.register(_ModuleRegistry())
            done.append(f)
    return done


class _ModuleRegistry:
    """What a plug-in's register(registry) gets."""

    @staticmethod
    def add(name: str, fn, description: str = "") -> None:
        register(name, fn, description)


# ---------------------------------------------------------------------------- #
def _qss(veh, k, ds, v0, v_end):
    """LapSim's quasi-steady-state solver (lts_qss_solve): load transfer, friction ellipse, powertrain."""
    from .lapsim import qss_solve  # noqa: PLC0415

    return qss_solve(veh, k, ds, v0, v_end)


def _point_mass(veh, k, ds, v0, v_end):
    """Point mass on a friction circle (ax, ay limits at 15 m/s) with the power limit; no load transfer."""
    from . import model  # noqa: PLC0415
    from .ggv import ay_limit  # noqa: PLC0415

    k = np.abs(np.asarray(k, float).ravel())
    n = len(k)
    v_ref = 15.0
    ay_max = ay_limit(veh, v_ref)
    acc0, dec0, _, _ = model.ax_limits(veh, v_ref, 0.0)
    meff = veh.m * veh.e_i
    v_top = model.powertrain(veh, 0.0)[2]
    vc = np.minimum(np.sqrt(ay_max / np.maximum(k, 1e-9)), v_top)

    def left(v, kk):                       # share of the circle left for ax
        return math.sqrt(max(1.0 - min(v * v * kk / ay_max, 1.0) ** 2, 0.0))

    vf = np.zeros(n)
    vf[0] = vc[0] if math.isnan(v0) else min(v0, vc[0])
    for i in range(n - 1):
        v = vf[i]
        _, F_cap, _ = model.powertrain(veh, v)
        a = min(acc0 * left(v, k[i]), (F_cap - model.resistance(veh, v)) / meff)
        vf[i + 1] = min(math.sqrt(max(v * v + 2 * a * ds, 0.0)), vc[i + 1])
    vb = np.zeros(n)
    vb[-1] = vc[-1] if math.isnan(v_end) else min(v_end, vc[-1])
    for i in range(n - 1, 0, -1):
        v = vb[i]
        a = dec0 * left(v, k[i])
        vb[i - 1] = min(math.sqrt(v * v - 2 * a * ds), vc[i - 1])
    return np.minimum(vf, vb), vc, None


register("qss", _qss)
register("point_mass", _point_mass)
