"""Performance Lab: the LR26 lap time simulation on its own (no ROS, no Isaac Sim, no WSL).

Runs on the Windows Python with numpy, scipy and pyyaml. It uses the LapSim port and the vehicle model of
ros2_ws/src/fsdsim (the same code the driverless simulator uses), so the numbers match the dashboard's
Vehicle setup page and LR26_LapSim in MATLAB. See docs/PERF_APP.md.
"""

from __future__ import annotations

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FSDSIM = os.path.join(REPO, "ros2_ws", "src", "fsdsim")
if FSDSIM not in sys.path:
    sys.path.insert(0, FSDSIM)
SIM_CONFIG = os.path.join(REPO, "ros2_ws", "src", "fsd_sim_bringup", "config")
