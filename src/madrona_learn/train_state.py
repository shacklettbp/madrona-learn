import jax
from jax import lax, random, numpy as jnp
from jax.experimental import checkify
import flax
from flax import linen as nn
from flax.training.dynamic_scale import DynamicScale
import flax.training.train_state
from flax.training import orbax_utils
import optax
import orbax.checkpoint

from dataclasses import dataclass
from typing import Optional, Any, Callable
from functools import partial

from .actor_critic import ActorCritic
from .moving_avg import EMANormalizer

class HyperParams(flax.struct.PyTreeNode):
    lr: float
    gamma: float
    gae_lambda: float
    normalize_values: bool
    value_normalizer_decay: float


class PolicyTrainState(flax.struct.PyTreeNode):
    apply_fn: Callable = flax.struct.field(pytree_node=False)
    value_normalize_fn: Callable = flax.struct.field(pytree_node=False)
    rnn_reset_fn: Callable = flax.struct.field(pytree_node=False)
    tx: optax.GradientTransformation = flax.struct.field(pytree_node=False)
    params: flax.core.FrozenDict[str, Any]
    batch_stats: flax.core.FrozenDict[str, Any]
    value_normalize_stats: flax.core.FrozenDict[str, Any]
    hyper_params: HyperParams
    opt_state: optax.OptState
    scheduler: Optional[optax.Schedule]
    scaler: Optional[DynamicScale]
    update_prng_key: random.PRNGKey

    def update(
        self,
        tx=None,
        params=None,
        batch_stats=None,
        value_normalize_stats=None,
        hyper_params=None,
        opt_state=None,
        scheduler=None,
        scaler=None,
        update_prng_key=None,
    ):
        return PolicyTrainState(
            apply_fn = self.apply_fn,
            value_normalize_fn = self.value_normalize_fn,
            rnn_reset_fn = self.rnn_reset_fn,
            params = params if params != None else self.params,
            tx = tx if tx != None else self.tx,
            batch_stats = (
                batch_stats if batch_stats != None else self.batch_stats),
            value_normalize_stats = (
                value_normalize_stats if value_normalize_stats != None else
                    self.value_normalize_stats),
            hyper_params = (
                hyper_params if hyper_params != None else self.hyper_params),
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

    def load(self, path):
        checkpointer = orbax.checkpoint.PyTreeCheckpointer()

        restore_desc = {
            'next_update': 0,
            'train_states': self.train_states,
        }
        
        loaded = checkpointer.restore(path, item=restore_desc)

        return TrainStateManager(
            train_states = loaded['train_states'],
        ), loaded['next_update']

    @staticmethod
    def load_policy_weights(path, policy_idx):
        checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        loaded = checkpointer.restore(path)

        return {
            'params': jax.tree_map(lambda x: x[policy_idx],
                                   loaded['train_states'].params),
            'batch_stats': jax.tree_map(lambda x: x[policy_idx],
                                        loaded['train_states'].batch_stats),
        }

    @staticmethod
    def create(
        policy,
        hyper_params,
        mixed_precision,
        num_policies,
        batch_size_per_policy,
        base_init_rng,
        example_obs,
        example_rnn_states,
        checkify_errors,
    ):
        train_states = _setup_train_states(policy, hyper_params,
            mixed_precision, num_policies, batch_size_per_policy,
            base_init_rng, example_obs, example_rnn_states, checkify_errors)

        return TrainStateManager(train_states=train_states)

def _setup_value_normalizer(hyper_params, rng_key, fake_values):
    value_norm_decay = (hyper_params.value_normalizer_decay 
                        if hyper_params.normalize_values else 1.0)

    value_normalizer = EMANormalizer(
        value_norm_decay, disable=not hyper_params.normalize_values)

    value_normalizer_vars = value_normalizer.init(
        rng_key, 'normalize', False, fake_values)

    return value_normalizer.apply, value_normalizer_vars['batch_stats']

def _setup_new_policy(
    policy,
    hyper_params,
    mixed_precision,
    prng_key,
    rnn_states,
    obs,
):
    model_init_rng, value_norm_rng, update_rng = random.split(prng_key, 3)
    fake_outs, variables = policy.init_with_output(
        model_init_rng, random.PRNGKey(0), rnn_states, obs,
        method='rollout')

    params = variables['params']

    if 'batch_stats' in variables:
        batch_stats = variables['batch_stats']
    else:
        batch_stats = {}

    value_norm_fn, value_norm_stats = _setup_value_normalizer(
        hyper_params, value_norm_rng, fake_outs[2])

    rnn_reset_fn = policy.clear_recurrent_state

    optimizer = optax.adam(learning_rate=hyper_params.lr)
    opt_state = optimizer.init(params)

    if mixed_precision:
        scaler = DynamicScale()
    else:
        scaler = None

    return PolicyTrainState(
        apply_fn = policy.apply,
        value_normalize_fn = value_norm_fn,
        rnn_reset_fn = rnn_reset_fn,
        tx = optimizer,
        params = params,
        batch_stats = batch_stats,
        value_normalize_stats = value_norm_stats,
        hyper_params = hyper_params,
        opt_state = opt_state,
        scheduler = None,
        scaler = scaler,
        update_prng_key = update_rng,
    )

def _setup_train_states(
    policy,
    hyper_params,
    mixed_precision,
    num_policies,
    batch_size_per_policy,
    base_init_rng,
    example_obs,
    example_rnn_states,
    checkify_errors,
):
    setup_new_policy = partial(_setup_new_policy,
        policy, hyper_params, mixed_precision)

    setup_new_policies = jax.vmap(setup_new_policy)

    obs = jax.tree_map(lambda x: x.reshape(
            num_policies,
            batch_size_per_policy,
            *x.shape[1:],
        ), example_obs)

    setup_new_policies = jax.jit(
        checkify.checkify(setup_new_policies, errors=checkify_errors))

    init_rngs = random.split(base_init_rng, num_policies)

    err, train_states = setup_new_policies(
        init_rngs, example_rnn_states, obs)
    err.throw()

    return train_states
