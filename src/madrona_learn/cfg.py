from dataclasses import dataclass
from typing import Callable, List, Optional

import jax
from jax import lax, random, numpy as jnp

class AlgoConfig:
    def name(self):
        raise NotImplementedError

    def setup(self):
        raise NotImplementedError


@dataclass(frozen=True)
class PBTConfig:
    num_teams: int
    team_size: int
    num_train_policies: int
    num_past_policies: int
    past_policy_update_interval: int
    # Must add to 1 and cleanly subdivide total rollout batch size
    self_play_portion: float
    cross_play_portion: float
    past_play_portion: float
    # Purely a speed / memory parameter
    rollout_policy_batch_size_override: int = 0


@dataclass(frozen=True)
class TrainConfig:
    num_worlds: int
    num_agents_per_world: int
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
            elif k == 'pbt':
                if v == 'None':
                    rep += "\n  ppt: Disabled"
                else:
                    rep += "\n  pbt:"
                    for pbt_k, pbt_v in self.pbt.__dict__.items():
                        rep += f"\n    {pbt_k}: {pbt_v}"
            else:
                rep += f"\n  {k}: {v}" 

        return rep
