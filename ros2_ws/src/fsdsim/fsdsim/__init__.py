"""fsdsim - simulator core for the FSG driverless digital twin.

Pure Python + NumPy. No ROS and no Isaac Sim imports live in this package, so
the same code runs inside a ROS 2 node, inside Isaac Sim's Python, and in
offline tools / unit tests.

Typical use::

    from fsdsim.config import load_all_configs
    from fsdsim.sim import Simulation

    vehicle, sensors, simcfg = load_all_configs("path/to/config")
    sim = Simulation(vehicle, sensors, simcfg)
    events = sim.step(0.01)
"""

__version__ = "0.7.1"
