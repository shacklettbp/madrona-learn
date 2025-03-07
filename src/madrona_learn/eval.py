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
from .policy import Policy
from .rollouts import RolloutConfig, RolloutState, rollout_loop
from .train_state import PolicyState, TrainStateManager
from .cfg import EvalConfig

def eval_load_ckpt(
    policy: Policy,
    ckpt_path: str,
    train_only: bool = True,
    single_policy: Optional[int] = None,
):
    policy_states, num_train_policies, total_num_policies = \
        TrainStateManager.load_policies(policy, ckpt_path)

    if single_policy != None:
        policy_states = jax.tree_map(
            lambda x: x[jnp.asarray((single_policy,))], policy_states)

        return policy_states, 1

    if train_only:
        policy_states = jax.tree_map(
            lambda x: x[jnp.arange(num_train_policies)], policy_states)

        return policy_states, num_train_policies

    return policy_states, total_num_policies


def eval_policies(
    dev: jax.Device,
    eval_cfg: EvalConfig,
    sim_fns: Dict['str', Callable],
    policy: Policy,
    init_sim_ctrl: jax.Array,
    policy_states: PolicyState,
    step_cb: Callable,
):
    with jax.default_device(dev):
        return _eval_policies_impl(eval_cfg, sim_fns, policy, init_sim_ctrl,
                                   policy_states, step_cb)

def _eval_policies_impl(
    eval_cfg: EvalConfig,
    sim_fns: Dict['str', Callable],
    policy: Policy,
    init_sim_ctrl: jax.Array,
    policy_states: PolicyState,
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

    num_agents_per_world = eval_cfg.team_size * eval_cfg.num_teams
    sim_batch_size = eval_cfg.num_worlds * num_agents_per_world

    if hasattr(policy_states, 'mmr') and policy_states.mmr != None:
        num_eval_policies = policy_states.mmr.elo.shape[0]
    elif (hasattr(policy_states, 'episode_score') and
          policy_states.episode_score != None):
        num_eval_policies = policy_states.episode_score.mean.shape[0]
    else:
        num_eval_policies = 1

    if eval_cfg.clear_fitness:
        def reset_mmr(mmr):
            if mmr == None:
                return None

            return mmr.replace(elo=mmr.elo.at[:].set(1500))

        def reset_episode_score(episode_score):
            if episode_score == None:
                return None

            return jax.tree_map(lambda x: x.at[:].set(0), episode_score)

        policy_states = policy_states.update(
            mmr = reset_mmr(policy_states.mmr),
            episode_score = reset_episode_score(policy_states.episode_score),
        )

    if num_eval_policies == 1 or not eval_cfg.eval_competitive:
        rollout_cfg = RolloutConfig.setup(
            num_current_policies = num_eval_policies,
            num_past_policies = 0,
            num_teams = 1,
            team_size = num_agents_per_world,
            sim_batch_size = sim_batch_size,
            actions_cfg = eval_cfg.actions,
            self_play_portion = 1.0,
            cross_play_portion = 0.0,
            past_play_portion = 0.0,
            static_play_portion = 0.0,
            reward_gamma = eval_cfg.reward_gamma,
            custom_policy_ids = eval_cfg.custom_policy_ids,
            policy_dtype = eval_cfg.policy_dtype,
        )

        static_play_assignments = jnp.zeros((sim_batch_size, 1), dtype=jnp.int32)
    else:
        rollout_cfg = RolloutConfig.setup(
            num_current_policies = num_eval_policies,
            num_past_policies = 0,
            num_teams = eval_cfg.num_teams,
            team_size = eval_cfg.team_size,
            sim_batch_size = sim_batch_size,
            actions_cfg = eval_cfg.actions,
            self_play_portion = 0.0,
            cross_play_portion = 0.0,
            past_play_portion = 0.0,
            static_play_portion = 1.0,
            reward_gamma = eval_cfg.reward_gamma,
            custom_policy_ids = eval_cfg.custom_policy_ids,
            policy_dtype = eval_cfg.policy_dtype,
        )

        total_num_policies = num_eval_policies + len(eval_cfg.custom_policy_ids)

        num_unique_static_assignments = total_num_policies * total_num_policies 

        num_static_repeats = sim_batch_size // num_unique_static_assignments

        assert sim_batch_size % num_unique_static_assignments == 0

        static_assignments_list = []

        for team_a_policy in range(num_eval_policies):
            for team_b_policy in range(num_eval_policies):
                static_assignments_list.append(team_a_policy)
                static_assignments_list.append(team_b_policy)

            for custom_id in eval_cfg.custom_policy_ids:
                static_assignments_list.append(team_a_policy)
                static_assignments_list.append(custom_id)

        for custom_id in eval_cfg.custom_policy_ids:
            for team_b_policy in range(num_eval_policies):
                static_assignments_list.append(custom_id)
                static_assignments_list.append(team_b_policy)

            for other_custom_id in eval_cfg.custom_policy_ids:
                static_assignments_list.append(custom_id)
                static_assignments_list.append(other_custom_id)

        num_assignment_duplicates = (
            (sim_batch_size // eval_cfg.team_size) //
            len(static_assignments_list))

        @jax.jit
        def gen_static_assignments():
            assignments = jnp.array(
                static_assignments_list, dtype=jnp.int32)

            assignments = assignments.reshape(-1, rollout_cfg.pbt.num_teams)
            assignments = jnp.repeat(
                assignments, num_assignment_duplicates, axis=0)
            assignments = jnp.repeat(
                assignments.reshape(-1), rollout_cfg.pbt.team_size)

            return assignments

        static_play_assignments = gen_static_assignments()

        assert (static_play_assignments.shape[0] ==
                rollout_cfg.pbt.static_play_batch_size)

    @jax.jit
    def init_rollout_state(sim_ctrl, static_play_assignments):
        rnn_states = policy.actor_critic.init_recurrent_state(
            rollout_cfg.sim_batch_size)

        return RolloutState.create(
            rollout_cfg = rollout_cfg,
            sim_fns = sim_fns,
            prng_key = random.PRNGKey(0),
            rnn_states = rnn_states,
            init_sim_ctrl = sim_ctrl,
            static_play_assignments = static_play_assignments
        )

    rollout_state = init_rollout_state(init_sim_ctrl, static_play_assignments)

    def post_policy_cb(step_idx, obs, preprocessed_obs, policy_out,
                       reorder_state, cb_state):
        return reorder_state.to_sim(policy_out.copy({
            'obs': obs,
        }))

    def post_step_cb(step_idx, rollout_state, dones, rewards,
                     episode_results, cb_state):
        sim_state = rollout_state.sim_state
        env_returns = rollout_state.env_returns

        step_data = cb_state.copy({
            'sim_state': sim_state,
            'dones': dones,
            'rewards': rewards,
            'returns': env_returns,
            'episode_results': episode_results,
            'rnn_states': rollout_state.rnn_states
        })

        sim_state = step_cb(step_data)

        rollout_state = rollout_state.update(
            sim_state = sim_state,
        )

        return rollout_state, None

    rollout_loop_fn = partial(rollout_loop,
        num_steps = eval_cfg.num_eval_steps,
        post_inference_cb = post_policy_cb,
        post_step_cb = post_step_cb,
        cb_state = None,
        sample_actions = not eval_cfg.use_deterministic_policy,
        return_debug = True,
    )

    rollout_loop_args = (rollout_state, policy_states)

    rollout_loop_fn = jax.jit(
        checkify.checkify(rollout_loop_fn, errors=checkify_errors),
        donate_argnums=[0, 1])

    lowered_rollout_loop = rollout_loop_fn.lower(*rollout_loop_args)

    if 'MADRONA_LEARN_PRINT_LOWERED' in env_vars and \
            env_vars['MADRONA_LEARN_PRINT_LOWERED'] == '1':
        print(lowered_rollout_loop.as_text())

    compiled_rollout_loop = lowered_rollout_loop.compile()

    err, (rollout_state, policy_states, _) = compiled_rollout_loop(
        *rollout_loop_args)
    err.throw()

    if eval_cfg.eval_competitive and hasattr(policy_states, 'mmr'):
        return policy_states.mmr
    elif hasattr(policy_states, 'episode_score'):
        return policy_states.episode_score
    else:
        return jnp.zeros((1,))
