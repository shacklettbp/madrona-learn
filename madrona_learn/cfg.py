from dataclasses import dataclass
from typing import Callable, List

import torch

@dataclass(frozen=True)
class PPOConfig:
    num_epochs: int
    num_mini_batches: int
    value_loss_coef: float
    entropy_coef: float
    max_grad_norm: float

@dataclass(frozen=True)
class TrainConfig:
    num_updates: int
    steps_per_update: int
    lr: float
    gamma: float
    gae_lambda: float
    ppo_cfg: PPOConfig

    def __repr__(self):
        rep = "TrainConfig:"

        for k, v in self.__dict__.items():
            rep += f"\n  {k}: {v}" 

        return rep

@dataclass(frozen=True)
class SimData:
    step: Callable
    obs: List[torch.Tensor]
    actions: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor
