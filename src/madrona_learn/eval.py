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
from .cfg import TrainConfig
from .train_state import TrainStateManager, HyperParams
from .utils import init_recurrent_states, make_pbt_reorder_funcs

def eval_ckpt(
    dev: jax.Device,
    ckpt_path: str,
    num_eval_steps: int,
    train_cfg: TrainConfig,
    sim_step: Callable,
    init_sim_data: FrozenDict,
    policy: ActorCritic,
    step_cb: Callable,
):
    with jax.default_device(dev):
        _eval_ckpt_impl(ckpt_path, num_eval_steps, train_cfg, sim_step,
                        init_sim_data, policy, step_cb)


class InferenceState(flax.struct.PyTreeNode):
    sim_step_fn: Callable = flax.struct.field(pytree_node=False)
    user_step_cb: Callable = flax.struct.field(pytree_node=False)
    rnn_states: Any
    sim_data: FrozenDict
    reorder_idxs: Optional[jax.Array]


def inference_loop(
    num_steps: int,
    num_policies: int,
    train_state_mgr: TrainStateManager,
    inference_state: InferenceState,
):
    def policy_infer(state, rnn_states, obs):
        return state.apply_fn(
            {
                'params': state.params,
                'batch_stats': state.batch_stats,
            },
            rnn_states,
            obs,
            train=False,
            method='debug',
        )

    policy_infer = jax.vmap(policy_infer)
    rnn_reset_fn = jax.vmap(train_state_mgr.train_states.rnn_reset_fn)

    prep_for_policy, prep_for_sim = make_pbt_reorder_funcs(
        inference_state.reorder_idxs != None, num_policies)

    def inference_iter(step_idx, inference_state):
        rnn_states = inference_state.rnn_states
        sim_data = inference_state.sim_data
        reorder_idxs = inference_state.reorder_idxs

        policy_obs = prep_for_policy(sim_data['obs'], reorder_idxs)

        actions, action_probs, values, rnn_states = policy_infer(
            train_state_mgr.train_states, rnn_states, policy_obs)

        sim_actions = prep_for_sim(actions, reorder_idxs)

        sim_data = sim_data.copy({
            'actions': sim_actions,
        })

        sim_data = frozen_dict.freeze(inference_state.sim_step_fn(sim_data))

        if reorder_idxs != None:
            reorder_idxs = jnp.argsort(sim_data['policy_assignments'])

        dones = jnp.asarray(sim_data['dones'], dtype=jnp.bool_)
        rewards = jnp.asarray(sim_data['rewards'], dtype=values.dtype)

        dones, rewards = prep_for_policy((dones, rewards), reorder_idxs)

        rnn_states = rnn_reset_fn(rnn_states, dones)

        inference_state.user_step_cb(
            policy_obs, actions, action_probs, values, dones, rewards)

        inference_state = inference_state.replace(
            rnn_states = rnn_states,
            sim_data = sim_data,
            reorder_idxs = reorder_idxs,
        )

        return inference_state

    return lax.fori_loop(0, num_steps, inference_iter, inference_state)

def _eval_ckpt_impl(
    ckpt_path: str,
    num_eval_steps: int,
    train_cfg: TrainConfig,
    sim_step: Callable,
    init_sim_data: FrozenDict,
    policy: ActorCritic,
    step_cb: Callable,
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

    batch_size_per_policy = \
        init_sim_data['actions'].shape[0] // train_cfg.pbt_ensemble_size

    rnn_states = init_recurrent_states(policy,
        batch_size_per_policy, train_cfg.pbt_ensemble_size)

    algo = train_cfg.algo.setup()
    hyper_params = algo.init_hyperparams(train_cfg)
    optimizer = algo.make_optimizer(hyper_params)

    train_state_mgr = TrainStateManager.create(
        policy = policy, 
        optimizer = optimizer,
        hyper_params = hyper_params,
        num_policies = train_cfg.pbt_ensemble_size,
        batch_size_per_policy = batch_size_per_policy,
        base_init_rng = random.PRNGKey(0),
        example_obs = init_sim_data['obs'],
        example_rnn_states = rnn_states,
        mixed_precision = train_cfg.mixed_precision,
        checkify_errors = checkify_errors,
    )

    train_state_mgr, _ = train_state_mgr.load(ckpt_path)

    sim_data = jax.tree_map(jnp.copy, init_sim_data)

    if 'policy_assignments' in sim_data:
        init_reorder_idxs = jnp.argsort(sim_data['policy_assignments'])
    else:
        init_reorder_idxs = None

    inference_state = InferenceState(
        sim_step_fn = sim_step,
        user_step_cb = step_cb,
        rnn_states = rnn_states,
        sim_data = sim_data,
        reorder_idxs = init_reorder_idxs,
    )

    inference_loop_fn = partial(inference_loop, num_eval_steps,
        train_cfg.pbt_ensemble_size, train_state_mgr)

    inference_loop_fn = jax.jit(
        checkify.checkify(inference_loop_fn, errors=checkify_errors))

    lowered_inference_loop = inference_loop_fn.lower(inference_state)

    if 'MADRONA_LEARN_PRINT_LOWERED' in env_vars and \
            env_vars['MADRONA_LEARN_PRINT_LOWERED'] == '1':
        print(lowered_inference_loop.as_text())

    compiled_inference_loop = lowered_inference_loop.compile()

    err, _ = compiled_inference_loop(inference_state)
    err.throw()
