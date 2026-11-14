"""Performance department vehicle model (LR26 LapSim) inside fsdsim.

    from fsdsim.perf import PerfParams, qss
    p = PerfParams.from_yaml(".../performance_lr26.yaml").with_changes("m=300")
    qss.acceleration(p).time, qss.skidpad(p).time
"""

from . import efficiency, model, qss
from .params import ALIASES, PerfParams

__all__ = ["ALIASES", "PerfParams", "efficiency", "model", "qss"]
