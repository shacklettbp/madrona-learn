from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import jax
from jax import lax, random, numpy as jnp

class AlgoConfig:
    def name(self):
        raise NotImplementedError

    def setup(self):
        raise NotImplementedError


@dataclass(frozen=True)
class ParamRange:
    min: float
    max: float
    log10_space: bool = False
    ln_space: bool = False

    def __repr__(self):
        if self.log10_space:
            type_str = "[log10]"
        elif self.ln_space:
            type_str = "[ln]"
        else:
            type_str = ""

        return f"{self.min}, {self.max} {type_str}"


@dataclass(frozen=True)
class PBTConfig:
    num_teams: int
    team_size: int
    num_train_policies: int
    num_past_policies: int
    train_policy_cull_interval: int
    num_cull_policies: int
    past_policy_update_interval: int
    # Must add to 1 and cleanly subdivide total rollout batch size
    self_play_portion: float
    cross_play_portion: float
    past_play_portion: float
    # During past policy updating & culling of train policies, the
    # policy being copied must have an expected winrate greater than this
    # threshold over the overwritten policy or the copy will be skipped
    policy_overwrite_threshold: float = 0.7
    lr_explore_range: Optional[ParamRange] = None
    entropy_explore_range: Optional[ParamRange] = None
    # Purely a speed / memory parameter
    rollout_policy_chunk_size_override: int = 0


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
    compute_dtype: jnp.dtype = jnp.float32

    def __repr__(self):
        rep = "TrainConfig:"

        for k, v in self.__dict__.items():
            if k == 'algo':
                rep += f"\n  {v.name()}:"
                for algo_cfg_k, algo_cfg_v in self.algo.__dict__.items():
                    rep += f"\n    {algo_cfg_k}: {algo_cfg_v}"
            elif k == 'pbt':
                if v == None:
                    rep += "\n  pbt: Disabled"
                else:
                    rep += "\n  pbt:"
                    for pbt_k, pbt_v in self.pbt.__dict__.items():
                        rep += f"\n    {pbt_k}: {pbt_v}"
            elif k == 'compute_dtype':
                rep += '\n  compute_dtype: '
                if v == jnp.float32:
                    rep += 'fp32'
                elif v == jnp.float16:
                    rep += 'fp16'
                elif v == jnp.bfloat16:
                    rep += 'bf16'
            else:
                rep += f"\n  {k}: {v}" 

        return rep
