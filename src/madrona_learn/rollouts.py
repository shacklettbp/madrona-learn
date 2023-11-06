import jax
from jax import lax, random, numpy as jnp
import flax
from flax import linen as nn
from flax.core import frozen_dict, FrozenDict

from dataclasses import dataclass
from typing import List, Optional, Callable, Any
from functools import partial
import itertools

from .cfg import TrainConfig
from .actor_critic import ActorCritic
from .algo_common import compute_advantages, compute_returns
from .metrics import TrainingMetrics, Metric
from .profile import profile
from .train_state import TrainStateManager, PolicyState, PolicyTrainState
from .utils import TypedShape, convert_float_leaves

@dataclass(frozen = True)
class RolloutConfig:
    num_current_policies: int
    num_past_policies: int
    total_num_policies: int
    num_teams: int
    team_size: int
    policy_batch_size: int
    total_batch_size: int
    self_play_batch_size: int
    cross_play_batch_size: int
    past_play_batch_size: int
    num_cross_play_matches: int
    num_past_play_matches: int
    float_dtype: jnp.dtype
    has_matchmaking: bool

    @staticmethod
    def setup(
        num_current_policies: int,
        num_past_policies: int,
        num_teams: int,
        team_size: int,
        total_batch_size: int,
        self_play_portion: float,
        cross_play_portion: float,
        past_play_portion: float,
        float_dtype: jnp.dtype,
        policy_batch_size_override: int = 0,
    ):
        total_num_policies = num_current_policies + num_past_policies

        assert (self_play_portion + cross_play_portion +
            past_play_portion == 1.0)

        self_play_batch_size = int(total_batch_size * self_play_portion)
        cross_play_batch_size = int(total_batch_size * cross_play_portion)
        past_play_batch_size = int(total_batch_size * past_play_portion)

        assert (self_play_batch_size + cross_play_batch_size +
                past_play_batch_size == total_batch_size)

        agents_per_world = num_teams * team_size

        assert cross_play_batch_size % agents_per_world == 0
        assert past_play_batch_size % agents_per_world == 0

        num_cross_play_matches = cross_play_batch_size // agents_per_world
        num_past_play_matches = past_play_batch_size // agents_per_world

        assert num_cross_play_matches % num_current_policies == 0
        assert num_past_play_matches % num_current_policies == 0
        
        has_matchmaking = self_play_portion != 1.0

        if has_matchmaking:
            assert num_teams > 1

            min_policy_batch_size = min(
                self_play_batch_size // num_current_policies,
                cross_play_batch_size // num_current_policies)

            if past_play_batch_size > 0:
                # FIXME: this doesn't make much sense, think more
                # about auto-picking policy batch size
                min_policy_batch_size = min(min_policy_batch_size,
                    past_play_batch_size // num_past_policies)

            # Round to nearest power of 2
            policy_batch_size = 1 << ((min_policy_batch_size - 1).bit_length())
            policy_batch_size = max(
                policy_batch_size, min(64, total_batch_size))
        else:
            assert num_past_policies == 0
            min_policy_batch_size = 0
            policy_batch_size = total_batch_size // num_current_policies

        if policy_batch_size_override != 0:
            assert policy_batch_size_override > min_policy_batch_size
            policy_batch_size = policy_batch_size_override

        assert total_batch_size % policy_batch_size == 0

        return RolloutConfig(
            num_current_policies = num_current_policies,
            num_past_policies = num_past_policies,
            total_num_policies = total_num_policies,
            num_teams = num_teams,
            team_size = team_size,
            policy_batch_size = policy_batch_size ,
            total_batch_size = total_batch_size,
            self_play_batch_size = self_play_batch_size,
            cross_play_batch_size = cross_play_batch_size,
            past_play_batch_size = past_play_batch_size,
            num_cross_play_matches = num_cross_play_matches,
            num_past_play_matches = num_past_play_matches,
            float_dtype = float_dtype,
            has_matchmaking = has_matchmaking,
        )


class PolicyBatchReorderState(flax.struct.PyTreeNode):
    to_policy_idxs: jax.Array
    to_sim_idxs: jax.Array


class RolloutState(flax.struct.PyTreeNode):
    step_fn: Callable = flax.struct.field(pytree_node=False)
    prng_key: random.PRNGKey
    rnn_states: Any
    sim_data: FrozenDict
    reorder_state: Optional[PolicyBatchReorderState]
    policy_assignments: Optional[jax.Array]
    to_policy: Callable = flax.struct.field(pytree_node=False)
    to_sim: Callable = flax.struct.field(pytree_node=False)

    @staticmethod
    def create(
        rollout_cfg,
        step_fn,
        prng_key,
        rnn_states,
        init_sim_data,
    ):
        if 'policy_assignments' in init_sim_data:
            policy_assignments = \
                init_sim_data['policy_assignments'].squeeze(axis=-1)
        elif (rollout_cfg.cross_play_batch_size > 0 or
              rollout_cfg.past_play_batch_size > 0): 
            prng_key, assign_rnd = random.split(prng_key)
            policy_assignments = _init_matchmake_assignments(
                assign_rnd, rollout_cfg)
        else:
            policy_assignments = None

        if policy_assignments:
            def to_policy(args, reorder_state):
                return jax.tree_map(
                    lambda x: jnp.take(x, reorder_state.to_policy_idxs, 0),
                    args,
                )

            def to_sim(args, reorder_state):
                args = jax.tree_map(lambda x: x.reshape(
                    rollout_cfg.total_batch_size, *x.shape[2:]), args)
                return jax.tree_map(
                    lambda x: jnp.take(x, reorder_state.to_sim_idxs, 0),
                    args,
                )

            reorder_state = _compute_reorder_state(
                policy_assignments,
                rollout_cfg,
            )
        else:
            def to_policy(args, reorder_state):
                assert reorder_state == None
                return jax.tree_map(lambda x: x.reshape(
                    rollout_cfg.total_num_policies,
                    rollout_cfg.policy_batch_size, *x.shape[1:]), args)

            def to_sim(args, reorder_state):
                assert reorder_state == None
                return jax.tree_map(lambda x: x.reshape(
                    rollout_cfg.total_batch_size, *x.shape[2:]), args)

            reorder_state = None

        sim_data = jax.tree_map(jnp.copy, init_sim_data)

        return RolloutState(
            step_fn = step_fn,
            prng_key = prng_key,
            rnn_states = rnn_states,
            sim_data = sim_data,
            reorder_state = reorder_state,
            policy_assignments = (policy_assignments
                if 'policy_assignments' not in init_sim_data else None),
            to_policy = to_policy,
            to_sim = to_sim,
        )

    def update(
        self,
        prng_key=None,
        rnn_states=None,
        sim_data=None,
        reorder_state=None,
        policy_assignments=None,
    ):
        return RolloutState(
            step_fn = self.step_fn,
            prng_key = prng_key if prng_key != None else self.prng_key,
            rnn_states = rnn_states if rnn_states != None else self.rnn_states,
            sim_data = sim_data if sim_data != None else self.sim_data,
            reorder_state = (
                reorder_state if reorder_state != None else self.reorder_state),
            policy_assignments = (
                policy_assignments if policy_assignments != None else
                    self.policy_assignments),
            to_policy = self.to_policy,
            to_sim = self.to_sim,
        )


class RolloutData(flax.struct.PyTreeNode):
    data: FrozenDict[str, Any]
    num_train_seqs_per_policy: int = flax.struct.field(pytree_node=False)
    num_train_policies: int = flax.struct.field(pytree_node=False)

    def all(self):
        return self.data

    def minibatch(self, indices):
        mb = jax.tree_map(lambda x: jnp.take(x, indices, 0), self.data)

        mb, rnn_start_states = mb.pop('rnn_start_states')

        # Make time leading axis
        mb = jax.tree_map(lambda x: jnp.swapaxes(x, 0, 1), mb)

        return mb.copy({
            'rnn_start_states': rnn_start_states,
        })


class RolloutStore(flax.struct.PyTreeNode):
    store : FrozenDict

    @staticmethod
    def create_from_tree(typed_shapes : FrozenDict):
        return RolloutStore(
            store = jax.tree_map(
                lambda x: jnp.empty(x.shape, x.dtype), typed_shapes),
        )

    def save(self, indices, data):
        def save_leaf(v, store): 
            return store.at[indices].set(v)

        data = frozen_dict.freeze(data)

        new_store = self.store
        for k, v in data.items():
            new_store = new_store.copy(
                {k: jax.tree_map(save_leaf, v, new_store[k])})

        return RolloutStore(store = new_store)


class RolloutManager:
    def __init__(
        self,
        train_cfg: TrainConfig,
        rollout_cfg: RolloutConfig,
        train_state_mgr: TrainStateManager,
        init_rollout_state: RolloutState,
    ):
        cpu_dev = jax.devices('cpu')[0]

        self._cfg = rollout_cfg

        self._num_bptt_chunks = train_cfg.num_bptt_chunks
        assert train_cfg.steps_per_update % train_cfg.num_bptt_chunks == 0
        self._num_bptt_steps = (
            train_cfg.steps_per_update // train_cfg.num_bptt_chunks)

        self._num_train_policies = rollout_cfg.num_current_policies
        self._num_train_agents_per_policy = \
            _compute_num_train_agents_per_policy(rollout_cfg)

        self._num_train_seqs_per_policy = (
            self._num_train_agents_per_policy * self._num_bptt_chunks)

        def compute_sim_to_train_indices():
            return _compute_sim_to_train_indices(rollout_cfg)

        self._sim_to_train_idxs = jax.jit(compute_sim_to_train_indices)()
        assert (self._sim_to_train_idxs.shape[1] ==
                self._num_train_agents_per_policy)

        self._use_advantages = train_cfg.compute_advantages
        self._compute_advantages_fn = partial(compute_advantages, train_cfg)
        self._compute_returns_fn = partial(compute_returns, train_cfg)

        typed_shapes = {}

        sim_data = init_rollout_state.sim_data
        
        def get_typed_shape(x):
            return TypedShape(x.shape, x.dtype)

        typed_shapes['obs'] = jax.tree_map(get_typed_shape, sim_data['obs'])

        typed_shapes['actions'] = TypedShape(
            sim_data['actions'].shape, sim_data['actions'].dtype)

        typed_shapes['log_probs'] = TypedShape(
            typed_shapes['actions'].shape, self._cfg.float_dtype)

        typed_shapes['rewards'] = TypedShape(
          sim_data['rewards'].shape, self._cfg.float_dtype)

        typed_shapes['dones'] = TypedShape(
            sim_data['dones'].shape, jnp.bool_)

        typed_shapes['values'] = TypedShape(
            typed_shapes['rewards'].shape, self._cfg.float_dtype)

        def expand_per_step_shapes(x):
            if jnp.issubdtype(x.dtype, jnp.floating):
                dtype = self._cfg.float_dtype
            else:
                dtype = x.dtype

            return TypedShape((
                    self._num_bptt_chunks,
                    self._num_bptt_steps,
                    self._num_train_policies,
                    self._num_train_agents_per_policy,
                    *x.shape[1:],
                ), dtype=dtype)

        typed_shapes = jax.tree_map(expand_per_step_shapes, typed_shapes)

        typed_shapes['bootstrap_values'] = TypedShape(
            typed_shapes['values'].shape[2:],
            self._cfg.float_dtype)

        typed_shapes['rnn_start_states'] = jax.tree_map(
            lambda x: TypedShape((
                    self._num_bptt_chunks,
                    self._num_train_policies,
                    self._num_train_agents_per_policy,
                    *x.shape[2:],
                ), x.dtype),
            init_rollout_state.rnn_states)

        self._store_typed_shape_tree = frozen_dict.freeze(typed_shapes)

    def add_metrics(
        self, 
        train_cfg: TrainConfig,
        metrics: FrozenDict[str, Metric],
    ):
        new_metrics = {
            'Rewards': Metric.init(True),
            'Returns': Metric.init(True),
            'Values': Metric.init(True),
        }

        if train_cfg.compute_advantages:
            new_metrics['Advantages'] = Metric.init(True)

        new_metrics['Bootstrap Values'] = Metric.init(True)

        return metrics.copy(new_metrics)

    def collect(
        self,
        train_state_mgr: TrainStateManager,
        rollout_state: RolloutState,
        metrics: TrainingMetrics,
    ):
        def iter_bptt_chunk(bptt_chunk, inputs):
            rollout_state, rollout_store = inputs

            post_inference_cb = partial(self._post_inference_cb,
                train_state_mgr.train_states, bptt_chunk)
            post_step_cb = partial(self._post_step_cb, bptt_chunk)

            with profile("Cache RNN state"):
                rollout_store = rollout_store.save(bptt_chunk, {
                    'rnn_start_states': self._sim_to_train(
                        rollout_state.rnn_states, rollout_state.reorder_state),
                })

            rollout_state, rollout_store = rollout_loop(
                self._cfg, rollout_state, train_state_mgr.policy_states,
                self._num_bptt_steps,  post_inference_cb, post_step_cb,
                rollout_store, sample_actions = True, return_debug = False)

            return rollout_state, rollout_store

        rollout_store = RolloutStore.create_from_tree(
            self._store_typed_shape_tree)

        rollout_state, rollout_store = lax.fori_loop(
            0, self._num_bptt_chunks, iter_bptt_chunk,
            (rollout_state, rollout_store))

        with profile("Bootstrap Values"):
            bootstrap_values = self._bootstrap_values(
                train_state_mgr.policy_states, train_state_mgr.train_states,
                rollout_state)

            rollout_store = rollout_store.save(
                slice(None), {'bootstrap_values': bootstrap_values})

        with profile("Finalize Rollouts"):
            rollout_data, metrics = self._finalize_rollouts(
                rollout_store.store, metrics)

        return rollout_state, rollout_data, metrics

    def _sim_to_train(self, data, reorder_state):
        if self._cfg.has_matchmaking:
            def to_train(x):
                return x[self._sim_to_train_idxs]
        else:
            def to_train(x):
                return x.reshape(self._num_train_policies, -1, *x.shape[1:])

        return jax.tree_map(to_train, data)

    def _policy_to_train(self, data, reorder_state):
        if not self._cfg.has_matchmaking:
            # Already in train ordering!
            return data
        
        def to_train(x):
            # FIXME
            sim_ordering = x.reshape(self._cfg.total_batch_size, *x.shape[2:])[
                reorder_state.to_sim_idxs]

            return sim_ordering[self._sim_to_train_idxs]

        return jax.tree_map(to_train, data)

    def _invert_value_normalization(self, train_states, values):
        @jax.vmap
        def invert_vn(train_state, v):
            return train_state.value_normalize_fn(
                { 'batch_stats': train_state.value_normalize_stats },
                mode='invert',
                update_stats=False,
                x=v,
            )

        return invert_vn(train_states, values)

    def _bootstrap_values(self, policy_states, train_states, rollout_state):
        rnn_states = rollout_state.rnn_states
        obs = rollout_state.sim_data['obs']

        obs = convert_float_leaves(obs, self._cfg.float_dtype)

        rnn_states, obs = self._sim_to_train(
            (rnn_states, obs), rollout_state.reorder_state)

        policy_states = jax.tree_map(
            lambda x: x[0:self._num_train_policies], policy_states)

        @jax.vmap
        def critic_fn(state, rnn_states, obs):
            policy_out, rnn_states = state.apply_fn(
                {
                    'params': state.params,
                    'batch_stats': state.batch_stats,
                },
                rnn_states,
                obs,
                train=False,
                method='critic_only',
            )

            return policy_out['values']

        values = critic_fn(policy_states, rnn_states, obs)

        return self._invert_value_normalization(train_states, values)

    def _post_inference_cb(
        self,
        train_states: PolicyTrainState,
        bptt_chunk: int,
        bptt_step: int,
        policy_obs: FrozenDict[str, Any],
        policy_out: FrozenDict[str, Any],
        reorder_state: PolicyBatchReorderState,
        rollout_store: RolloutStore,
    ):
        with profile('Pre Step Rollout Store'):
            values = self._policy_to_train(policy_out['values'], reorder_state)
            values = self._invert_value_normalization(train_states, values)

            obs, actions, log_probs = self._policy_to_train(
                (policy_obs, policy_out['actions'], policy_out['log_probs']),
                reorder_state)

            save_data = {
                'obs': obs,
                'actions': actions,
                'log_probs': log_probs,
                'values': values,
            }

            return rollout_store.save((bptt_chunk, bptt_step), save_data)

    def _post_step_cb(
        self,
        bptt_chunk: int,
        bptt_step: int,
        dones: jax.Array,
        rewards: jax.Array,
        reorder_state: PolicyBatchReorderState,
        rollout_store: RolloutStore,
    ):
        with profile('Post Step Rollout Store'):
            save_data = self._sim_to_train({
                'dones': dones,
                'rewards': rewards,
            }, reorder_state)
            return rollout_store.save(
                (bptt_chunk, bptt_step), save_data)

    def _finalize_rollouts(self, rollouts, metrics):
        rollouts, bootstrap_values = rollouts.pop('bootstrap_values')

        if self._use_advantages:
            advantages = self._compute_advantages_fn(rollouts['rewards'],
                rollouts['values'], rollouts['dones'], bootstrap_values)

            rollouts = rollouts.copy({
                'advantages': advantages,
            })

            metrics = metrics.record({
                'Advantages': advantages,
            })

            returns = advantages + rollouts['values']
        else:
            returns = self._compute_returns_fn(rollouts['rewards'],
                rollouts['dones'], bootstrap_values)

        rollouts = rollouts.copy({
            'returns': returns,
        })

        metrics = metrics.record({
            'Rewards': rollouts['rewards'],
            'Values': rollouts['values'],
            'Returns': rollouts['returns'],
            'Bootstrap Values': bootstrap_values,
        })

        rollouts, rnn_start_states = rollouts.pop('rnn_start_states')

        # Per Step rollouts reshaped / transposed as follows:
        # [C, T / C, P, B, ...] => [P, C * B, T / C, ...]

        def reorder_seq_data(x):
            t = x.transpose(2, 0, 3, 1, *range(4, len(x.shape)))
            return t.reshape(t.shape[0], -1, *t.shape[3:])

        rollouts = jax.tree_map(reorder_seq_data, rollouts)

        # RNN states reshaped / transposed as follows:
        # [C, P, B, ...] => [P, C * B, ...]

        def reorder_rnn_data(x):
            t = x.transpose(1, 0, 2, *range(3, len(x.shape)))
            return t.reshape(t.shape[0], -1, *t.shape[3:])

        rnn_start_states = jax.tree_map(reorder_rnn_data, rnn_start_states)
            
        return RolloutData(
            data = rollouts.copy({
                'rnn_start_states': rnn_start_states,
            }),
            num_train_seqs_per_policy = self._num_train_seqs_per_policy,
            num_train_policies = self._num_train_policies,
        ), metrics


def rollout_loop(
    rollout_cfg: RolloutConfig,
    rollout_state: RolloutState,
    policy_states: PolicyState,
    num_steps: int,
    post_inference_cb: Callable,
    post_step_cb: Callable,
    cb_state: Any,
    **policy_kwargs,
):
    def policy_fn(state, sample_key, rnn_states, obs):
        return state.apply_fn(
            {
                'params': state.params,
                'batch_stats': state.batch_stats,
            },
            sample_key,
            rnn_states,
            obs,
            train = False,
            **policy_kwargs,
            method = 'rollout',
        )

    policy_fn = jax.vmap(policy_fn)
    rnn_reset_fn = policy_states.rnn_reset_fn

    def rollout_iter(step_idx, iter_state):
        rollout_state, cb_state = iter_state

        prng_key = rollout_state.prng_key
        rnn_states = rollout_state.rnn_states
        sim_data = rollout_state.sim_data
        reorder_state = rollout_state.reorder_state
        policy_assignments = rollout_state.policy_assignments
        to_policy = rollout_state.to_policy
        to_sim = rollout_state.to_sim

        with profile('Policy Inference'):
            prng_key, step_key = random.split(prng_key)
            step_keys = random.split(
                step_key, rollout_cfg.total_num_policies)

            policy_obs = convert_float_leaves(
                sim_data['obs'], rollout_cfg.float_dtype)

            rnn_states, policy_obs = to_policy(
                (rnn_states, policy_obs), reorder_state)

            policy_out, rnn_states = policy_fn(
                policy_states, step_keys, rnn_states, policy_obs)

            cb_state = post_inference_cb(
                step_idx, policy_obs, policy_out, reorder_state, cb_state)

            # rnn_states are kept across the loop in sim ordering,
            # because the ordering is stable, unlike policy batches that 
            # can shift when assignments change
            rnn_states = to_sim(rnn_states, reorder_state)

        with profile('Rollout Step'):
            sim_data = sim_data.copy({
                'actions': to_sim(policy_out['actions'], reorder_state),
            })

            sim_data = frozen_dict.freeze(rollout_state.step_fn(sim_data))

            dones = sim_data['dones'].astype(jnp.bool_)
            rewards = sim_data['rewards'].astype(rollout_cfg.float_dtype)

            # rnn states kept in sim ordering
            rnn_states = rnn_reset_fn(rnn_states, dones)

            if reorder_state != None:
                if 'policy_assignments' in sim_data:
                    policy_assignments = \
                        sim_data['policy_assignments'].squeeze(axis=-1)
                else:
                    assert policy_assignments != None
                    policy_assignments, prng_key = _update_policy_assignments(
                        policy_assignments, dones, prng_key, rollout_cfg)

                reorder_state = _compute_reorder_state(
                    policy_assignments,
                    rollout_cfg,
                )

            cb_state = post_step_cb(
                step_idx, dones, rewards, reorder_state, cb_state)

        rollout_state = rollout_state.update(
            prng_key = prng_key,
            rnn_states = rnn_states,
            sim_data = sim_data,
            reorder_state = reorder_state,
            policy_assignments = (policy_assignments
                if 'policy_assignments' not in sim_data else None),
        )

        return rollout_state, cb_state

    return lax.fori_loop(0, num_steps, rollout_iter, (rollout_state, cb_state))


def _compute_num_train_agents_per_policy(rollout_cfg):
    assert rollout_cfg.cross_play_batch_size % rollout_cfg.num_teams == 0
    assert rollout_cfg.past_play_batch_size % rollout_cfg.num_teams == 0

    # FIXME: currently we only collect training data for team 1 in each
    # world in cross-play and past-play to avoid having variable amounts
    # of training data in each step
    total_num_train_agents = (
        rollout_cfg.self_play_batch_size +
        rollout_cfg.cross_play_batch_size // rollout_cfg.num_teams +
        rollout_cfg.past_play_batch_size // rollout_cfg.num_teams
    )

    assert total_num_train_agents % rollout_cfg.num_current_policies == 0
    return total_num_train_agents // rollout_cfg.num_current_policies


def _compute_sim_to_train_indices(rollout_cfg):
    global_indices = jnp.arange(rollout_cfg.total_batch_size)

    def setup_match_indices(start, stop):
        return global_indices[start:stop].reshape(
            rollout_cfg.num_current_policies, -1,
            rollout_cfg.num_teams, rollout_cfg.team_size)

    self_play_indices = setup_match_indices(
        0, rollout_cfg.self_play_batch_size)

    cross_play_indices = setup_match_indices(
        rollout_cfg.self_play_batch_size,
        rollout_cfg.self_play_batch_size + rollout_cfg.cross_play_batch_size)

    past_play_indices = setup_match_indices(
        rollout_cfg.self_play_batch_size + rollout_cfg.cross_play_batch_size,
        rollout_cfg.self_play_batch_size + rollout_cfg.cross_play_batch_size +
            rollout_cfg.past_play_batch_size)

    self_play_gather = self_play_indices.reshape(
        rollout_cfg.num_current_policies, -1)

    cross_play_gather = cross_play_indices[:, :, 0, :].reshape(
        rollout_cfg.num_current_policies, -1)

    past_play_gather = past_play_indices[:, :, 0, :].reshape(
        rollout_cfg.num_current_policies, -1)

    return jnp.concatenate(
        [self_play_gather, cross_play_gather, past_play_gather], axis=1)


def _cross_play_matchmake(
    assignments,
    dones,
    assign_rnd,
    rollout_cfg,
):
    num_matches = rollout_cfg.num_cross_play_matches
    num_teams = rollout_cfg.num_teams
    team_size = rollout_cfg.team_size

    assignments = assignments.reshape(num_matches, num_teams, team_size)
    dones = dones.reshape(num_matches, num_teams, team_size)

    new_assignments = random.randint(assign_rnd, (num_matches, num_teams - 1),
        0, rollout_cfg.num_current_policies - 1)

    new_assignments = jnp.where(new_assignments >= assignments[:, 0:1, 0],
        new_assignments + 1, new_assignments)

    new_assignments = jnp.where(
        dones[:, 1:, :], new_assignments[:, :, None], assignments[:, 1:, :])

    return assignments.at[:, 1:, :].set(new_assignments).reshape(-1)


def _past_play_matchmake(
    assignments,
    dones,
    assign_rnd,
    rollout_cfg,
):
    num_matches = rollout_cfg.num_cross_play_matches
    num_teams = rollout_cfg.num_teams
    team_size = rollout_cfg.team_size
    
    assignments = assignments.reshape(num_matches, num_teams, team_size)
    dones = dones.reshape(num_matches, num_teams, team_size)

    new_assignments = random.randint(assign_rnd, (num_matches, num_teams - 1),
        rollout_cfg.num_current_policies,
        rollout_cfg.num_current_policies + rollout_cfg.num_past_policies)

    new_assignments = jnp.where(
        dones[:, 1:, :], new_assignments[:, :, None], assignments[:, 1:, :])

    return assignments.at[:, 1:, :].set(new_assignments).reshape(-1)


def _init_matchmake_assignments(assign_rnd, rollout_cfg):
    def self_play_matchmake(batch_size):
        return jnp.repeat(
            jnp.arange(rollout_cfg.num_current_policies),
            batch_size // rollout_cfg.num_current_policies)

    self_play_batch_size = rollout_cfg.self_play_batch_size
    cross_play_batch_size = rollout_cfg.cross_play_batch_size
    past_play_batch_size = rollout_cfg.past_play_batch_size

    sub_assignments = []

    if self_play_batch_size > 0:
        self_assignments = self_play_matchmake(self_play_batch_size)
        assert self_assignments.shape[0] == self_play_batch_size
        sub_assignments.append(self_assignments)

    if cross_play_batch_size > 0:
        assign_rnd, cross_rnd = random.split(assign_rnd)

        cross_assignments = self_play_matchmake(cross_play_batch_size)
        assert cross_assignments.shape[0] == cross_play_batch_size
        cross_assignments = _cross_play_matchmake(
            cross_assignments,
            jnp.ones(cross_play_batch_size, dtype=jnp.bool_),
            cross_rnd,
            rollout_cfg,
        )

        sub_assignments.append(cross_assignments)

    if past_play_batch_size > 0:
        past_assignments = self_play_matchmake(past_play_batch_size)
        assert past_assignments.shape[0] == past_play_batch_size
        past_assignments = _past_play_matchmake(
            past_assignments,
            jnp.ones(past_play_batch_size, dtype=jnp.bool_),
            assign_rnd,
            rollout_cfg,
        )

        sub_assignments.append(past_assignments)

    policy_assignments = jnp.concatenate(sub_assignments, axis=0)

    assert policy_assignments.shape[0] == rollout_cfg.total_batch_size
    return policy_assignments


def _update_policy_assignments(
    assignments,
    dones,
    assign_rnd,
    rollout_cfg,
):
    if rollout_cfg.cross_play_batch_size > 0:
        assign_rnd, cross_rnd = random.split(assign_rnd)

        cross_start = self_play_batch_size
        cross_end = self_play_batch_size + cross_play_batch_size
        assignments = assignments.at[cross_start:cross_end].set(
            _cross_play_matchmake(assignments[cross_start:cross_end],
                dones[cross_start:cross_end], cross_rnd, rollout_cfg))

    if past_play_batch_size > 0:
        assign_rnd, past_rnd = random.split(assign_rnd)

        past_start = self_play_batch_size + cross_play_batch_size 
        past_end = past_start + past_play_batch_size
        assignments = assignments.at[past_start:past_end].set(
            _past_play_matchmake(assignments[past_start:past_end],
                dones[past_start:past_end], past_rnd, rollout_cfg))

    return assignments, assign_rnd

def _compute_reorder_chunks(assignments, P, C, B):
    assert assignments.ndim == 1

    sort_idxs = jnp.argsort(assignments)
    sorted_assignments = assignments.at[sort_idxs].get(unique_indices=True)

    ne_mask = jnp.ones(assignments.shape[0], dtype=jnp.bool_).at[1:].set(
        lax.ne(sorted_assignments[1:], sorted_assignments[:-1]))
    transitions = jnp.nonzero(
        ne_mask, size=P + 1, fill_value=assignments.size)[0]
    transitions_diff = jnp.diff(transitions)
    transitions = transitions[:-1]

    # Need to scatter here to handle the case where there are 0 instances
    # of certain values. Care is needed here since transitions will
    # contain out of bounds indices in this case.
    transition_assignments = sorted_assignments.at[transitions].get(
        mode='fill', indices_are_sorted=True, fill_value=P)
    assignment_starts = jnp.full(P, assignments.size, dtype=jnp.int32).at[
        transition_assignments].set(transitions, mode='drop')
    assignment_counts = jnp.zeros(P, dtype=jnp.int32).at[
        transition_assignments].set(transitions_diff, mode='drop')

    num_full_chunks, partial_sizes = jnp.divmod(assignment_counts, C)

    # Compute each item's offset from the start of items in its class
    expanded_assignment_starts = jnp.take(
        assignment_starts, sorted_assignments, indices_are_sorted=True)
    offsets_from_starts = (
        jnp.arange(assignments.size) - expanded_assignment_starts)

    full_chunk_counts = num_full_chunks * C
    full_chunk_cumsum = jnp.cumsum(full_chunk_counts)
    # Base offset for partial chunks is after all full chunks
    partial_base = full_chunk_cumsum[-1]

    # Prefix sum to get starting offsets for full chunks
    full_chunk_starts = full_chunk_cumsum - full_chunk_counts

    # Computer scatter indices for items in full chunks
    expanded_full_chunk_starts = jnp.take(
        full_chunk_starts, sorted_assignments, indices_are_sorted=True)
    expanded_full_chunk_counts = jnp.take(
        full_chunk_counts, sorted_assignments, indices_are_sorted=True)
    full_chunk_indices = expanded_full_chunk_starts + offsets_from_starts

    # Compute scatter indices for items in partial chunks
    partial_chunk_starts = (
        partial_base + C * jnp.arange(P) - full_chunk_counts)

    # Alternatively, if partial chunks need to be compacted
    # to the beginning, do the following prefix sum over occupied chunks.
    # Note that this doesn't change the worst-case size bounds.
    # has_partial_mask = jnp.where(partial_sizes != 0, 1, 0)
    # partial_chunk_prefix = jnp.cumsum(has_partial_mask) - has_partial_mask
    # partial_chunk_starts = (
    #     partial_base + C * partial_chunk_prefix - full_chunk_counts)

    expanded_partial_chunk_starts = jnp.take(
        partial_chunk_starts, sorted_assignments, indices_are_sorted=True)

    partial_chunk_indices = (expanded_partial_chunk_starts +
        offsets_from_starts)

    full_partial_mask = jnp.logical_and(
        expanded_full_chunk_counts > 0,
        offsets_from_starts < expanded_full_chunk_counts)
    scatter_positions = jnp.where(
        full_partial_mask, full_chunk_indices, partial_chunk_indices)

    to_policy_idxs = jnp.full((B * C), assignments.size, jnp.int32).at[
        scatter_positions].set(sort_idxs, unique_indices=True).reshape(B, C)

    to_sim_idxs = jnp.empty_like(assignments).at[
        sort_idxs].set(scatter_positions, unique_indices=True)

    return to_policy_idxs, to_sim_idxs


def _compute_reorder_state(
    assignments,
    rollout_cfg,
):
    policy_batch_size = rollout_cfg.policy_batch_size
    assert assignments.shape[0] % policy_batch_size == 0


    # Allocate enough chunks to evenly divide the full batch, plus
    # num_policies - 1 extras to handle worst case usage from unused space
    # in chunks
    num_policy_batches = assignments.shape[0] // policy_batch_size
    if rollout_cfg.has_matchmaking:
        num_policy_batches += rollout_cfg.total_num_policies - 1

    to_policy_idxs, to_sim_idxs = _compute_reorder_chunks(
        assignments, rollout_cfg.total_num_policies, 
        rollout_cfg.policy_batch_size, num_policy_batches)

    return PolicyBatchReorderState(
        to_policy_idxs = to_policy_idxs,
        to_sim_idxs = to_sim_idxs,
    )


'''
def _update_reorder_state(
    reorder_state,
    assignments,
    rollout_cfg,
):
    if (rollout_cfg.num_cross_play_matches == 0 and
            rollout_cfg.num_past_play_matches == 0):
        return reorder_state

    assert assignments.ndim == 1
    policy_batch_size = rollout_cfg.policy_batch_size
    num_policies = rollout_cfg.total_num_policies

    reorder_assigns = assignments[rollout_cfg.self_play_batch_size:]
    num_reorder_matches = (rollout_cfg.num_cross_play_matches +
        rollout_cfg.num_past_play_matches)

    assert matchmake_assigns.shape[0] % policy_batch_size == 0

    # Allocate enough matchmake batches to evenly divide the non-self play
    # matches, plus num policies - 1 extras to handle worst case usage
    # from unused space in chunks
    num_matchmake_batches = (
        matchmake_assigns.shape[0] // policy_batch_size + num_policies - 1)

    non_self_play_assigns = non_self_play_assigns.reshape(
        num_matchmake_matches, rollout_cfg.num_teams, rollout_cfg.team_size)

    first_team_assigns = matchmake_assigns[:, 0:1, :]
    reorder_assigns = matchmake_assigns[:, 1:, :]

    reorder_policy_idxs, reorder_sim_idxs = _compute_reorder_chunks(
        reorder_assigns.reshape(-1), num_policies,
        policy_batch_size, num_reorder_batches)

    to_policy_idxs = reorder_state.to_policy_idxs.at[
        rollout_cfg.num_static_policy_batches:].set(reorder_policy_idxs)

    to_sim_idxs = reorder_state.to_sim_idxs.reshape(
        -1, rollout_cfg.num_teams, rollout_cfg.team_size)

    to_sim_idxs = to_sim_idxs.at[
        rollout_cfg.self_play_batch_size:, 1:, :].set(reorder_sim_idxs)

    to_sim_idxs = to_sim_idxs.reshape(-1)

    return PolicyBatchReorderState(
        to_policy_idxs = to_policy_idxs,
        to_sim_idxs = to_sim_idxs,
    )


def _init_reorder_state(
    assignments,
    rollout_cfg,
):
    pass
'''
