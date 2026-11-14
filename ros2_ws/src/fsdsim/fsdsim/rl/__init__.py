"""Reinforcement learning of the throttle / brake (longitudinal) control for a given vehicle spec.

    fsdsim.rl.policy   MLPPolicy (NumPy), saved as policy.npz + policy.json
    fsdsim.rl.env      headless training scenarios and rollouts (fsdsim physics)
    fsdsim.rl.es       trainer: imitation warm start + evolution strategies (OpenAI-ES)
    fsdsim.rl.registry where policies live and which one matches a vehicle

Train:  python3 tools/train_rl.py --set m=270 --name light
Drive:  ros2 launch fsd_sim_bringup isaac.launch.py longitudinal:=rl
"""
