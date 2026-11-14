"""Isaac Lab task: the universal throttle / brake policy of the LR26 driverless car (RSL-RL, PPO).

The car is fsdsim's own vehicle model (LapSim forces), batched on the GPU (fsdsim.vec); Isaac Lab
provides the vectorised RL environment API, the PPO implementation (RSL-RL), logging, checkpoints
and multi-GPU training. Registered with Isaac Lab's unified training script through a callback:

    isaaclab.bat -p scripts\\reinforcement_learning\\train.py --rl_library rsl_rl ^
        --task FSD-Longitudinal-Universal-v0 --external_callback fsd_rl.register --headless --num_envs 4096

(see README.md; PYTHONPATH must contain <repo>/isaaclab and <repo>/ros2_ws/src/fsdsim).
"""

import gymnasium as gym

TASK_ID = "FSD-Longitudinal-Universal-v0"
_done = False


def register():
    """--external_callback fsd_rl.register: registers the task. Returns None (no arguments consumed)."""
    global _done
    if not _done:
        gym.register(
            id=TASK_ID,
            entry_point=f"{__name__}.env:FsdLonEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.env_cfg:FsdLonEnvCfg",
                "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:FsdLonPPORunnerCfg",
                "default_agent": "rsl_rl",
            },
        )
        _done = True
    return None
