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
from .rollouts import RolloutState, rollout_loop
from .train_state import TrainStateManager

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

    train_state_mgr = TrainStateManager.create(
        policy = policy, 
        cfg = train_cfg,
        algo = algo,
        rollout_agents_per_policy = batch_size_per_policy,
        base_init_rng = random.PRNGKey(0),
        example_obs = init_sim_data['obs'],
        example_rnn_states = rnn_states,
        checkify_errors = checkify_errors,
    )

    train_state_mgr, _ = train_state_mgr.load(ckpt_path)

    rollout_state = RolloutState.create(
        step_fn = sim_step,
        prng_key = random.PRNGKey(0),
        rnn_states = rnn_states,
        init_sim_data = init_sim_data,
    )

    def post_policy_cb(step_idx, policy_obs, policy_out, cb_state):
        return policy_out.copy({
            'obs': policy_obs,
        })

    def post_step_cb(step_idx, dones, rewards, cb_state):
        step_data = cb_state.copy({
            'dones': dones,
            'rewards': rewards,
        })

        step_cb(step_data)

        return None

    rollout_loop_fn = partial(rollout_loop, 
        train_cfg.pbt_ensemble_size, num_eval_steps,
        train_state_mgr.policy_states, rollout_state,
        post_policy_cb, post_step_cb, None,
        jnp.float16 if train_cfg.mixed_precision else jnp.float32,
        sample_actions = False, return_debug = True)

    rollout_loop_fn = jax.jit(
        checkify.checkify(rollout_loop_fn, errors=checkify_errors))

    lowered_rollout_loop = rollout_loop_fn.lower(rollout_state)

    if 'MADRONA_LEARN_PRINT_LOWERED' in env_vars and \
            env_vars['MADRONA_LEARN_PRINT_LOWERED'] == '1':
        print(lowered_rollout_loop.as_text())

    compiled_rollout_loop = lowered_inference_loop.compile()

    err, _ = compiled_rollout_loop(rollout_state)
    err.throw()
