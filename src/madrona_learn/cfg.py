from dataclasses import dataclass
from typing import Callable, List, Optional

import jax
from jax import lax, random, numpy as jnp

class AlgoConfig:
    def name(self):
        raise NotImplementedError

    def setup(self, cfg: "TrainConfig"):
        raise NotImplementedError

@dataclass(frozen=True)
class TrainConfig:
    num_worlds: int
    num_updates: int
    steps_per_update: int
    lr: float
    algo: AlgoConfig
    num_bptt_chunks: int
    gamma: float
    gae_lambda: float = 1.0
    team_size: int = 1
    num_teams: int = 1
    pbt_ensemble_size: int = 1
    pbt_history_len: int = 1
    normalize_advantages: bool = True
    normalize_values : bool = True
    value_normalizer_decay : float = 0.99999
    mixed_precision : bool = False

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

@dataclass(frozen=True)
class SimInterface:
    step: Callable
    obs: List[jax.Array]
    actions: jax.Array
    dones: jax.Array
    rewards: jax.Array
    policy_assignments: Optional[jax.Array]
