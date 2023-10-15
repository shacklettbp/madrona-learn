import jax
from jax import lax, random, numpy as jnp
import flax
from flax import linen as nn
import flax.training.dynamic_scale
import flax.training.train_state
from flax.training import orbax_utils
import optax
import orbax.checkpoint

from dataclasses import dataclass
from typing import Optional, Any, Callable

from .actor_critic import ActorCritic
from .moving_avg import EMANormalizer

class HyperParams(flax.struct.PyTreeNode):
    lr: float
    gamma: float
    gae_lambda: float


class PolicyTrainState(flax.struct.PyTreeNode):
    apply_fn: Callable = flax.struct.field(pytree_node=False)
    params: flax.core.FrozenDict[str, Any]
    batch_stats: flax.core.FrozenDict[str, Any]
    value_normalize_fn: Callable = flax.struct.field(pytree_node=False)
    value_normalize_stats: flax.core.FrozenDict[str, Any]
    hyper_params: HyperParams
    tx: optax.GradientTransformation = flax.struct.field(pytree_node=False)
    opt_state: optax.OptState
    scheduler: Optional[optax.Schedule]
    scaler: Optional[flax.training.dynamic_scale.DynamicScale]
    update_prng_key: random.PRNGKey

    def update(
        self,
        params=None,
        batch_stats=None,
        value_normalize_stats=None,
        hyper_params=None,
        tx=None,
        opt_state=None,
        scheduler=None,
        scaler=None,
        update_prng_key=None,
    ):
        return PolicyTrainState(
            apply_fn = self.apply_fn,
            params = params if params != None else self.params,
            batch_stats = (
                batch_stats if batch_stats != None else self.batch_stats),
            value_normalize_fn = self.value_normalize_fn,
            value_normalize_stats = (
                value_normalize_stats if value_normalize_stats != None else
                    self.value_normalize_stats),
            hyper_params = (
                hyper_params if hyper_params != None else self.hyper_params),
            tx = tx if tx != None else self.tx,
            opt_state = opt_state if opt_state != None else self.opt_state,
            scheduler = scheduler if scheduler != None else self.scheduler,
            scaler = scaler if scaler != None else self.scaler,
            update_prng_key = (
                update_prng_key if update_prng_key != None else 
                    self.update_prng_key),
        )

    def gen_update_rnd(self):
        rnd, next_key = random.split(self.update_prng_key)
        return rnd, self.update(update_prng_key=next_key)


class TrainStateManager(flax.struct.PyTreeNode):
    train_states: PolicyTrainState

    def save(self, update_idx, path):
        ckpt = {
            'next_update': update_idx + 1,
            'train_states': self.train_states,
        }

        checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(ckpt)
        checkpointer.save(path, ckpt, save_args=save_args)

    @staticmethod
    def load(path):
        checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        loaded = checkpointer.restore(path)

        self.train_states = loaded['train_states']

        return TrainStateManager(
            train_states = loaded['train_states'],
        ), loaded['next_update']

    @staticmethod
    def load_policy_weights(path, policy_idx):
        checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        loaded = checkpointer.restore(path)

        loaded = torch.load(path)
        return {
            'params': loaded['train_states'][policy_idx].params,
            'batch_stats': loaded['train_states'][policy_idx].batch_stats,
        }
