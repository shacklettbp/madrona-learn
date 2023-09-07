from dataclasses import dataclass
from typing import Callable, List
from .typing_utils import DataclassProtocol

import torch

@dataclass(frozen=True)
class AlgoConfig:
    name: str
    update_iter_fn: Callable

@dataclass(frozen=True)
class TrainConfig:
    num_updates: int
    steps_per_update: int
    lr: float
    algo: AlgoConfig
    num_bptt_chunks: int
    gamma: float
    gae_lambda: float = 1.0
    normalize_advantages: bool = True
    normalize_values : bool = True
    value_normalizer_decay : float = 0.99999
    mixed_precision : bool = False

    def __repr__(self):
        rep = "TrainConfig:"

        for k, v in self.__dict__.items():
            if k == 'algo':
                rep += f"\n  {v.name}:"
                for algo_cfg_k, algo_cfg_v in self.algo.__dict__.items():
                    if algo_cfg_k in ['name', 'update_iter_fn']:
                        continue

                    rep += f"\n    {algo_cfg_k}: {algo_cfg_v}"
            else:
                rep += f"\n  {k}: {v}" 

        return rep

@dataclass(frozen=True)
class SimInterface:
    step: Callable
    obs: List[torch.Tensor]
    actions: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor
