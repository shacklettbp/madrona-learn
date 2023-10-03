import jax
from jax import lax, random, numpy as jnp

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

@runtime_checkable
@dataclass
class DataclassProtocol(Protocol):
    pass


@dataclass
class InternalConfig:
    rollout_batch_size : int
    num_train_agents : int
    num_train_seqs : int
    num_bptt_steps : int
    float_storage_type : jnp.dtype

    def __init__(self, dev, cfg, float_compute_type):
        self.rollout_batch_size = \
            cfg.num_teams * cfg.team_size * cfg.num_worlds

        assert(cfg.num_worlds % cfg.pbt_ensemble_size == 0)
        self.num_train_agents = cfg.team_size * cfg.num_worlds

        assert(cfg.steps_per_update % cfg.num_bptt_chunks == 0)
        self.num_train_seqs = self.num_train_agents * cfg.num_bptt_chunks
        self.num_bptt_steps = cfg.steps_per_update // cfg.num_bptt_chunks

        if float_compute_type == jnp.bfloat16:
            self.float_storage_type = jnp.bfloat16
        else:
            self.float_storage_type = jnp.float16
