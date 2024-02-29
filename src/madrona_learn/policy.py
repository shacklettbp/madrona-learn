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
    get_episode_scores: Optional[Callable] = None
