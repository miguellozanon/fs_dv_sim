# Isaac Lab: train the universal throttle / brake policy with RSL-RL

The task `FSD-Longitudinal-Universal-v0` puts fsdsim's own LR26 model (LapSim forces, batched on the
GPU, `fsdsim.vec`) behind Isaac Lab's `DirectRLEnv` interface. Isaac Lab provides PPO (RSL-RL), logging
(TensorBoard / W&B), checkpoints and multi-GPU training; the physics, scenarios and rewards are the same
as `tools/train_rl_universal.py`. There is no PhysX car: train headless.

## Train (Windows, Isaac Lab 3.0 installed in `C:\IsaacLab`)

```bat
set PYTHONPATH=%USERPROFILE%\fsg-dv-sim\isaaclab;%USERPROFILE%\fsg-dv-sim\ros2_ws\src\fsdsim
cd C:\IsaacLab
isaaclab.bat -p scripts\reinforcement_learning\train.py --rl_library rsl_rl ^
    --task FSD-Longitudinal-Universal-v0 --external_callback fsd_rl.register ^
    --headless --num_envs 4096 --max_iterations 1500
```

Use the path of the repo on Windows (copy it from WSL, or `\\wsl$\Ubuntu\home\<you>\fsg-dv-sim`).
fsdsim needs only NumPy and PyYAML, which Isaac Lab's Python has.

Logs and checkpoints: `C:\IsaacLab\logs\rsl_rl\fsd_longitudinal_universal\<date>\model_<it>.pt`
(`tensorboard --logdir logs\rsl_rl` shows the return, `Episode/progress_m`, `Episode/cones`, `Episode/failed`).

## Export to the simulator

```bat
isaaclab.bat -p %USERPROFILE%\fsg-dv-sim\tools\export_rsl_rl_policy.py ^
    logs\rsl_rl\fsd_longitudinal_universal\<date>\model_1499.pt --name universal_isaaclab
```

This writes `rl\policies\universal_isaaclab` (NumPy weights, observation normalisation folded in),
checks it against the checkpoint and runs the validation suite. Copy the folder into the repo's
`rl/policies` in WSL; the simulator then uses it for every setup inside its training range
(`longitudinal:=rl`, or Driver -> RL in the dashboard).

## Settings

`fsd_rl/env_cfg.py`: tracks (`autocross_seeds`), `vehicle_changes` (the centre of the ranges),
`spec_ranges` (empty: the defaults of `fsdsim/vec/spec.py`), `realism`, `physics_hz`.
`fsd_rl/agents/rsl_rl_ppo_cfg.py`: network (tanh 64-64, required by the exporter's default
`--activation tanh`), PPO settings.

Without Isaac Lab the same training runs with plain PyTorch:
`python3 tools/train_rl_universal.py --backend torch --device cuda:0`.
