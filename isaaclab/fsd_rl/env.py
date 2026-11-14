"""Isaac Lab DirectRLEnv around fsdsim's vectorised longitudinal environment (fsdsim.vec.env).

All of the physics, the scenarios, the rewards and the resets are fsdsim.vec.VecLonEnv on the GPU
(PyTorch backend); this class maps it onto Isaac Lab's step / reset protocol so RSL-RL (or skrl,
rl_games) can train on it and log through Isaac Lab.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs import DirectRLEnv

from fsdsim.vec.backend import Backend
from fsdsim.vec.env import EnvConfig, VecLonEnv
from fsdsim.vec.spec import Realism, SpecRanges
from fsdsim.vec.tracks import TrackBank
from fsdsim.vec.train import TRAIN_ITEMS, baseline

from .env_cfg import FsdLonEnvCfg


class FsdLonEnv(DirectRLEnv):
    cfg: FsdLonEnvCfg

    def __init__(self, cfg: FsdLonEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        B = Backend("torch", device=str(self.device), seed=int(cfg.seed or 0))
        _, base = baseline(cfg.config_dir, cfg.vehicle_changes)
        items = [(m, s) for m, s in TRAIN_ITEMS if m != "autocross"] + [("autocross", s) for s in cfg.autocross_seeds]
        bank = TrackBank.build(cfg.config_dir, items, cfg.vehicle_changes)
        ecfg = EnvConfig(n_envs=self.num_envs, physics_hz=cfg.physics_hz,
                         ranges=SpecRanges.from_dict(cfg.spec_ranges) if cfg.spec_ranges else SpecRanges(),
                         realism=Realism() if cfg.realism else Realism.off())
        self.core = VecLonEnv(B, bank, base, ecfg, teacher=False)
        self.core.reset()
        self._actions = torch.zeros(self.num_envs, device=self.device)
        self._rew = torch.zeros(self.num_envs, device=self.device)
        self._term = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._trunc = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    # ------------------------------------------------------------------ #
    def _setup_scene(self):
        pass

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._actions = actions[:, 0].clamp(-1.0, 1.0).to(self.core.B.float)

    def _apply_action(self) -> None:
        # decimation = 1: one call per RL step; the car model integrates the 40 ms itself
        _, rew, term, trunc, info = self.core.step(self._actions, auto_reset=False)
        self._rew, self._term, self._trunc = rew.float(), term, trunc
        done = term | trunc
        if bool(done.any()):
            log = self.extras.setdefault("log", {})
            log["Episode/progress_m"] = info["prog"][done].mean()
            log["Episode/cones"] = info["cones"][done].mean()
            log["Episode/failed"] = term[done].float().mean()

    def _get_observations(self) -> dict:
        return {"policy": self.core._observe().float()}

    def _get_rewards(self) -> torch.Tensor:
        return self._rew

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._term, self._trunc

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        super()._reset_idx(env_ids)
        ids = torch.as_tensor(env_ids, device=self.device).long()
        if hasattr(self, "core"):
            self.core._reset_idx(ids)
