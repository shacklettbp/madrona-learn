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
from .rollouts import RolloutConfig, RolloutState, rollout_loop
from .train_state import TrainStateManager

def eval_ckpt(
    dev: jax.Device,
    ckpt_path: str,
    num_eval_steps: int,
    sim_step: Callable,
    init_sim_data: FrozenDict,
    policy: ActorCritic,
    step_cb: Callable,
    policy_dtype: jnp.dtype,
    load_policies: List[int] = [0],
    use_deterministic_policy: bool = True,
):
    with jax.default_device(dev):
        _eval_ckpt_impl(ckpt_path, num_eval_steps, sim_step,
            init_sim_data, policy, step_cb, policy_dtype, load_policies,
            use_deterministic_policy)

def _eval_ckpt_impl(
    ckpt_path: str,
    num_eval_steps: int,
    sim_step: Callable,
    init_sim_data: FrozenDict,
    policy: ActorCritic,
    step_cb: Callable,
    policy_dtype: jnp.dtype,
    load_policies: List[int],
    use_deterministic_policy: bool,
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

    num_policies = len(load_policies)

    total_batch_size = init_sim_data['actions'].shape[0]
    batch_size_per_policy = total_batch_size // num_policies

    policy_states = TrainStateManager.load_policies(policy, ckpt_path)
    policy_states = jax.tree_map(
        lambda x: x[load_policies], policy_states)

    rollout_cfg = RolloutConfig.setup(
        num_current_policies = num_policies,
        num_past_policies = 0,
        num_teams = 0,
        team_size = 0,
        total_batch_size = total_batch_size,
        self_play_portion = 1.0,
        cross_play_portion = 0.0,
        past_play_portion = 0.0,
        float_dtype = policy_dtype,
    )

    @jax.jit
    def init_rollout_state():
        rnn_states = policy.init_recurrent_state(total_batch_size)

        return RolloutState.create(
            rollout_cfg = rollout_cfg,
            step_fn = sim_step,
            prng_key = random.PRNGKey(0),
            rnn_states = rnn_states,
            init_sim_data = init_sim_data,
        )

    rollout_state = init_rollout_state()

    def post_policy_cb(step_idx, policy_obs, policy_out,
                       reorder_state, cb_state):
        return policy_out.copy({
            'obs': policy_obs,
        })

    def post_step_cb(step_idx, dones, rewards, reorder_state, cb_state):
        step_data = cb_state.copy({
            'dones': dones,
            'rewards': rewards,
        })

        step_cb(step_data)

        return None

    rollout_loop_fn = partial(rollout_loop,
        rollout_cfg = rollout_cfg,
        policy_states = policy_states,
        num_policies = num_policies,
        num_steps = num_eval_steps,
        post_inference_cb = post_policy_cb,
        post_step_cb = post_step_cb,
        cb_state = None,
        sample_actions = not use_deterministic_policy,
        return_debug = True,
    )

    rollout_loop_fn = jax.jit(
        checkify.checkify(rollout_loop_fn, errors=checkify_errors),
        donate_argnums=0)

    lowered_rollout_loop = rollout_loop_fn.lower(rollout_state)

    if 'MADRONA_LEARN_PRINT_LOWERED' in env_vars and \
            env_vars['MADRONA_LEARN_PRINT_LOWERED'] == '1':
        print(lowered_rollout_loop.as_text())

    compiled_rollout_loop = lowered_rollout_loop.compile()

    err, _ = compiled_rollout_loop(rollout_state)
    err.throw()
