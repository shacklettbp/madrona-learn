import jax
from jax import lax, random, numpy as jnp
import flax
from flax import linen as nn
from flax.core import frozen_dict, FrozenDict

from dataclasses import dataclass
from typing import List, Optional, Dict, Callable

from .actor_critic import ActorCritic
from .observations import ObservationsPreprocess

@dataclass(frozen = True)
class Policy:
    actor_critic: ActorCritic
    obs_preprocess: Optional[ObservationsPreprocess] = None
    init_reward_hyper_params: Optional[Callable] = None
    mutate_reward_hyper_params: Optional[Callable] = None
    get_team_a_score: Optional[Callable] = None
