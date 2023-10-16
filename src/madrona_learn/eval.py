import jax
from jax import lax, random, numpy as jnp
from jax.experimental import checkify
import flax
from flax import linen as nn
from flax.training.dynamic_scale import DynamicScale
from flax.core import frozen_dict, FrozenDict
import optax

from os import environ as env_vars
from dataclasses import dataclass
from functools import partial
from typing import List, Optional, Dict, Callable, Any

from .actor_critic import ActorCritic
from .train_state import TrainStateManager, HyperParams
from .utils import init_recurrent_states

def infer(
    dev: jax.Device,
    num_policies: int,
    mixed_precision: bool,
    sim_step: Callable,
    init_sim_data: FrozenDict,
    policy: ActorCritic,
    step_cb: Callable,
    ckpt_path: str,
):
    with jax.default_device(dev):
        _infer_impl(num_policies, mixed_precision, sim_step, 
                    init_sim_data, policy, step_cb, ckpt_path)

def eval_loop(
    step_cb: Callable,
    train_state_mgr: TrainStateManager,
    rnn_states: Any,
    sim_data: FrozenDict,
):
    pass

def _infer_impl(
    num_policies: int,
    mixed_precision: bool,
    sim_step: Callable,
    init_sim_data: FrozenDict,
    policy: ActorCritic,
    step_cb: Callable,
    ckpt_path: str,
):
    checkify_errors = checkify.user_checks
    if 'MADRONA_LEARN_FULL_CHECKIFY' in env_vars and \
            env_vars['MADRONA_LEARN_FULL_CHECKIFY'] == '1':
        checkify_errors |= (
            checkify.float_checks |
            checkify.nan_checks |
            checkify.div_checks |
            checkify.index_checks
        )

    init_sim_data = frozen_dict.freeze(init_sim_data)

    batch_size_per_policy = init_sim_data['actions'].shape[0] // num_policies

    fake_hyper_params = HyperParams(
        lr = 0.0,
        gamma = 0.0,
        gae_lambda = 0.0,
        normalize_values = True,
        value_normalizer_decay = 1.0,
    )

    rnn_states = init_recurrent_states(policy,
        batch_size_per_policy, num_policies)

    train_state_mgr = TrainStateManager.create(
        policy = policy, 
        hyper_params = fake_hyper_params,
        num_policies = num_policies,
        batch_size_per_policy = batch_size_per_policy,
        base_init_rng = random.PRNGKey(0),
        example_obs = init_sim_data['obs'],
        example_rnn_states = rnn_states,
        mixed_precision = mixed_precision,
        checkify_errors = checkify_errors,
    )

    train_state_mgr, _ = train_state_mgr.load(ckpt_path)

    sim_data = jax.tree_map(jnp.copy, init_sim_data)

    eval_loop = partial(eval_loop, step_cb)
    eval_loop = jax.jit(
        checkify.checkify(eval_loop, errors=checkify_errors))

    eval_loop_args = (train_state_mgr, rnn_states, sim_data)

    lowered_eval_loop = eval_loop.lower(*eval_loop_args)

    if 'MADRONA_LEARN_PRINT_LOWERED' in env_vars and \
            env_vars['MADRONA_LEARN_PRINT_LOWERED'] == '1':
        print(lowered_eval_loop.as_text())

    compiled_eval_loop = lowered_eval_loop.compile()

    err, _ = compiled_eval_loop(*eval_loop_args)
    err.throw()
