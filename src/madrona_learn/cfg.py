from dataclasses import dataclass
from typing import Callable, List, Optional

import jax
from jax import lax, random, numpy as jnp

class AlgoConfig:
    def name(self):
        raise NotImplementedError

    def setup(self):
        raise NotImplementedError


class PBTConfig:
    population_size: int
    history_len: int
    batching_cfg: List[int]
    update_interval: int


@dataclass(frozen=True)
class TrainConfig:
    num_worlds: int
    num_updates: int
    steps_per_update: int
    lr: float
    algo: AlgoConfig
    num_bptt_chunks: int
    gamma: float
    seed: int
    gae_lambda: float = 1.0
    pbt: Optional[PBTConfig] = None
    compute_advantages: bool = True
    normalize_advantages: bool = True # Only used if compute_advantages = True
    normalize_returns: bool = True # Only used if compute_advantages = False
    normalize_values: bool = True
    value_normalizer_decay: float = 0.99999
    mixed_precision: bool = False

    def __repr__(self):
        rep = "TrainConfig:"

        for k, v in self.__dict__.items():
            if k == 'algo':
                rep += f"\n  {v.name()}:"
                for algo_cfg_k, algo_cfg_v in self.algo.__dict__.items():
                    rep += f"\n    {algo_cfg_k}: {algo_cfg_v}"
            else:
                rep += f"\n  {k}: {v}" 

        return rep
