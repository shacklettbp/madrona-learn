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
import itertools

from .actor_critic import ActorCritic
from .rollouts import RolloutConfig, RolloutState, rollout_loop
from .train_state import TrainStateManager

@dataclass(frozen=True)
class SinglePolicyEvalConfig:
    policy_idx: int

@dataclass(frozen=True)
class MultiPolicyEvalConfig:
    num_teams: int
    team_size: int

def eval_ckpt(
    dev: jax.Device,
    ckpt_path: str,
    num_eval_steps: int,
    sim_step: Callable,
    init_sim_data: FrozenDict,
    policy: ActorCritic,
    obs_preprocess: Optional[nn.Module],
    step_cb: Callable,
    policy_dtype: jnp.dtype,
    single_policy_cfg: Optional[SinglePolicyEvalConfig] = None,
    multi_policy_cfg: Optional[MultiPolicyEvalConfig] = None,
    use_deterministic_policy: bool = True,
):
    assert single_policy_cfg != None or multi_policy_cfg != None
    assert single_policy_cfg == None or multi_policy_cfg == None

    with jax.default_device(dev):
        _eval_ckpt_impl(ckpt_path, num_eval_steps, sim_step,
            init_sim_data, policy, obs_preprocess, step_cb, policy_dtype,
            single_policy_cfg, multi_policy_cfg, use_deterministic_policy)

def _eval_ckpt_impl(
    ckpt_path: str,
    num_eval_steps: int,
    sim_step: Callable,
    init_sim_data: FrozenDict,
    policy: ActorCritic,
    obs_preprocess: Optional[nn.Module],
    step_cb: Callable,
    policy_dtype: jnp.dtype,
    single_policy_cfg: Optional[SinglePolicyEvalConfig],
    multi_policy_cfg: Optional[MultiPolicyEvalConfig],
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

    sim_batch_size = init_sim_data['actions'].shape[0]

    policy_states, num_train_policies = TrainStateManager.load_policies(
        policy, obs_preprocess, ckpt_path)

    if single_policy_cfg != None:
        policy_states = jax.tree_map(
            lambda x: x[jnp.asarray((single_policy_cfg.policy_idx,))], policy_states)

        rollout_cfg = RolloutConfig.setup(
            num_current_policies = 1,
            num_past_policies = 0,
            num_teams = 1,
            team_size = 1,
            sim_batch_size = sim_batch_size,
            self_play_portion = 1.0,
            cross_play_portion = 0.0,
            past_play_portion = 0.0,
            static_play_portion = 0.0,
            policy_dtype = policy_dtype,
        )

        static_play_assignments = None

    elif multi_policy_cfg != None:
        assert (
            sim_batch_size %
                (multi_policy_cfg.num_teams * multi_policy_cfg.team_size) ==
            0
        )

        rollout_cfg = RolloutConfig.setup(
            num_current_policies = num_train_policies,
            num_past_policies = 0,
            num_teams = multi_policy_cfg.num_teams,
            team_size = multi_policy_cfg.team_size,
            sim_batch_size = sim_batch_size,
            self_play_portion = 0.0,
            cross_play_portion = 0.0,
            past_play_portion = 0.0,
            static_play_portion = 1.0,
            policy_dtype = policy_dtype,
        )

        num_unique_static_assignments = num_train_policies * num_train_policies

        num_static_repeats = sim_batch_size // num_unique_static_assignments

        assert sim_batch_size % num_unique_static_assignments == 0

        static_assignments_list = []

        for combo in itertools.product(
                range(num_train_policies), repeat=multi_policy_cfg.num_teams):
            for i in combo:
                static_assignments_list.append(i)

        num_assignment_duplicates = (
            (sim_batch_size // multi_policy_cfg.team_size) //
            len(static_assignments_list))

        @jax.jit
        def gen_static_assignments():
            assignments = jnp.array(
                static_assignments_list, dtype=jnp.int32)

            assignments = assignments.reshape(-1, multi_policy_cfg.num_teams)
            assignments = jnp.repeat(
                assignments, num_assignment_duplicates, axis=0)
            assignments = jnp.repeat(
                assignments.reshape(-1), multi_policy_cfg.team_size)

            return assignments

        static_play_assignments = gen_static_assignments()

        assert (static_play_assignments.shape[0] ==
                rollout_cfg.static_play_batch_size)

    @jax.jit
    def init_rollout_state(init_sim_data, static_play_assignments):
        rnn_states = policy.init_recurrent_state(rollout_cfg.sim_batch_size)

        return RolloutState.create(
            rollout_cfg = rollout_cfg,
            step_fn = sim_step,
            prng_key = random.PRNGKey(0),
            rnn_states = rnn_states,
            init_sim_data = init_sim_data,
            static_play_assignments = static_play_assignments
        )

    rollout_state = init_rollout_state(init_sim_data, static_play_assignments)

    def post_policy_cb(step_idx, obs, preprocessed_obs, policy_out,
                       reorder_state, cb_state):
        return reorder_state.to_sim(policy_out.copy({
            'obs': obs,
        }))

    def post_step_cb(step_idx, dones, rewards, reorder_state, cb_state):
        step_data = cb_state.copy({
            'dones': dones,
            'rewards': rewards,
        })

        step_cb(step_data)

        return None

    rollout_loop_fn = partial(rollout_loop,
        rollout_cfg = rollout_cfg,
        num_steps = num_eval_steps,
        post_inference_cb = post_policy_cb,
        post_step_cb = post_step_cb,
        cb_state = None,
        sample_actions = not use_deterministic_policy,
        return_debug = True,
    )

    rollout_loop_args = (rollout_state, policy_states)

    rollout_loop_fn = jax.jit(
        checkify.checkify(rollout_loop_fn, errors=checkify_errors),
        donate_argnums=0)

    lowered_rollout_loop = rollout_loop_fn.lower(*rollout_loop_args)

    if 'MADRONA_LEARN_PRINT_LOWERED' in env_vars and \
            env_vars['MADRONA_LEARN_PRINT_LOWERED'] == '1':
        print(lowered_rollout_loop.as_text())

    compiled_rollout_loop = lowered_rollout_loop.compile()

    err, _ = compiled_rollout_loop(*rollout_loop_args)
    err.throw()
