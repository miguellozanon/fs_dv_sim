"""Configuration of the FSD longitudinal (throttle / brake) task for Isaac Lab."""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from fsdsim.vec.features import N_FEATURES

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@configclass
class FsdLonSceneCfg(InteractiveSceneCfg):
    """No articulated assets: the cars are fsdsim's batched model (fsdsim.vec.car), not PhysX bodies."""

    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())


@configclass
class FsdLonEnvCfg(DirectRLEnvCfg):
    # one RL step = one policy step of 40 ms; inside it the car model runs its own 500 Hz RK4 steps
    decimation = 1
    episode_length_s = 30.0               # the scenarios end themselves (5 / 24 / 20 s), see fsdsim.vec.env
    action_space = 1                      # a in [-1, 1]: throttle (> 0) / brake (< 0)
    observation_space = N_FEATURES        # features v2 (fsdsim.vec.features)
    state_space = 0
    is_finite_horizon = False

    sim: SimulationCfg = SimulationCfg(dt=0.04, render_interval=1)
    scene: FsdLonSceneCfg = FsdLonSceneCfg(num_envs=4096, env_spacing=0.0, replicate_physics=True)

    # fsdsim
    config_dir: str = os.path.join(REPO, "ros2_ws", "src", "fsd_sim_bringup", "config")
    vehicle_changes: list = []            # baseline of the spec ranges (empty: LapSim lr26_params.m)
    autocross_seeds: tuple = tuple(range(100, 160))
    physics_hz: float = 500.0
    realism: bool = True                  # hidden grip / actuator lag / latency / sensor noise
    spec_ranges: dict = {}                # empty: fsdsim.vec.spec.SpecRanges defaults
