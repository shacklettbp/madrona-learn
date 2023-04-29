from dataclasses import dataclass
from typing import Callable, List

import torch

@dataclass(frozen=True)
class TrainConfig:
    num_updates: int
    steps_per_update: int
    lr: float
    gamma: float

    def __repr__(self):
        rep = "TrainConfig:"

        for k, v in self.__dict__.items():
            rep += f"\n  {k}: {v}" 

        return rep

@dataclass(frozen=True)
class SimConfig:
    step: Callable
    process_obs: Callable
    obs: List[torch.Tensor]
    actions: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor
