"""Batched (GPU) LR26 car model and the universal throttle / brake policy's environment.

backend   NumPy or PyTorch arrays (same code on CPU and CUDA)
spec      vehicle specifications per car, randomisation ranges, hidden realism
car       the dynamic car model of fsdsim (LapSim forces) for N cars at once
limits    LapSim limits per car (corner speed, accel / brake capability), lookup tables
tracks    track bank: reference paths of the FSG events and generated autocross layouts
steer     the reference driver's pure pursuit for N cars
features  what the policy sees (features v2: spec-conditioned), shared with the RL driver
env       the vectorised RL environment (rewards, resets, randomisation, evaluation runs)
"""
