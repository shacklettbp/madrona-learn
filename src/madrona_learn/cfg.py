import dataclasses
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Dict, Union

import jax
from jax import lax, random, numpy as jnp
from flax.core import FrozenDict

@dataclass(frozen=True)
class ActionsConfig:
    actions_num_buckets: List[int]


class AlgoConfig:
    def name(self):
        raise NotImplementedError

    def setup(self):
        raise NotImplementedError


@dataclass(frozen=True)
class ParamExplore:
    base: float
    min_scale: float
    max_scale: float
    log10_scale: bool = False
    ln_scale: bool = False
    clip_perturb: bool = False
    perturb_rnd_min: float = 0.8
    perturb_rnd_max: float = 1.2

    def __repr__(self):
        if self.log10_scale:
            type_str = "log10, "
        elif self.ln_scale:
            type_str = "ln, "
        else:
            type_str = ""

        return f"{self.base * self.min_scale}, {self.base * self.max_scale} [{type_str}{self.perturb_rnd_min, self.perturb_rnd_max}]"


@dataclass(frozen=True)
class PBTConfig:
    num_teams: int
    team_size: int
    num_train_policies: int
    num_past_policies: int
    # Must add to 1 and evenly subdivide total rollout batch size
    self_play_portion: float
    cross_play_portion: float
    past_play_portion: float
    # During past policy updating & culling of train policies, the
    # policy being copied must have an expected winrate greater than this
    # threshold over the overwritten policy or the copy will be skipped
    policy_overwrite_threshold: float = 0.7
    reward_hyper_params_explore: Dict[str, ParamExplore] = FrozenDict({})
    # Purely a speed / memory parameter
    rollout_policy_chunk_size_override: int = 0


@dataclass(frozen=True)
class TrainConfig:
    num_worlds: int
    num_agents_per_world: int
    num_updates: int
    actions: ActionsConfig
    steps_per_update: int
    lr: Union[float, ParamExplore]
    algo: AlgoConfig
    num_bptt_chunks: int
    gamma: float
    seed: int
    metrics_buffer_size: int
    baseline_policy_id: int = 0
    custom_policy_ids: List[int] = dataclasses.field(default_factory=lambda: [])
    gae_lambda: float = 1.0
    pbt: Optional[PBTConfig] = None
    dreamer_v3_critic: bool = True 
    compute_advantages: bool = True
    normalize_advantages: bool = True # Only used if compute_advantages = True
    normalize_returns: bool = True # Only used if compute_advantages = False
    normalize_values: bool = False
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
            elif k == 'actions':
                rep += '\n  actions: '
                rep += str(self.actions)
            else:
                rep += f"\n  {k}: {v}" 

        return rep


@dataclass(frozen=True)
class EvalConfig:
    num_worlds: int
    num_teams: int
    team_size: int
    num_eval_steps: int
    actions: ActionsConfig
    reward_gamma: float
    policy_dtype: jnp.dtype
    eval_competitive: bool
    use_deterministic_policy: bool = True
    clear_fitness: bool = True
    custom_policy_ids: List[int] = dataclasses.field(default_factory=lambda: [])
