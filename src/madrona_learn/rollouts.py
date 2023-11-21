import jax
from jax import lax, random, numpy as jnp
import flax
from flax import linen as nn
from flax.core import frozen_dict, FrozenDict

from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable, Any
from functools import partial
import itertools

from .cfg import TrainConfig
from .actor_critic import ActorCritic
from .algo_common import compute_advantages, compute_returns
from .metrics import TrainingMetrics, Metric
from .profile import profile
from .train_state import TrainStateManager, PolicyState, PolicyTrainState
from .utils import TypedShape
from .observations import ObservationsPreprocess

@dataclass(frozen = True)
class RolloutConfig:
    num_current_policies: int
    num_past_policies: int
    total_num_policies: int
    num_teams: int
    team_size: int
    sim_batch_size: int
    policy_chunk_size: int
    num_policy_chunks: int
    total_policy_batch_size: int
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
        sim_batch_size: int,
        self_play_portion: float,
        cross_play_portion: float,
        past_play_portion: float,
        float_dtype: jnp.dtype,
        policy_chunk_size_override: int = 0,
    ):
        total_num_policies = num_current_policies + num_past_policies

        assert (self_play_portion + cross_play_portion +
            past_play_portion == 1.0)

        self_play_batch_size = int(sim_batch_size * self_play_portion)
        cross_play_batch_size = int(sim_batch_size * cross_play_portion)
        past_play_batch_size = int(sim_batch_size * past_play_portion)

        assert (self_play_batch_size + cross_play_batch_size +
                past_play_batch_size == sim_batch_size)

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
            assert num_current_policies > 1 or num_past_policies > 0

            min_policy_chunk_size = min(
                self_play_batch_size // num_current_policies,
                cross_play_batch_size // num_current_policies)

            if past_play_batch_size > 0:
                # FIXME: this doesn't make much sense, think more
                # about auto-picking policy batch size
                min_policy_chunk_size = min(min_policy_chunk_size,
                    past_play_batch_size // num_past_policies)

            # Round to nearest power of 2
            policy_chunk_size = 1 << ((min_policy_chunk_size - 1).bit_length())
            policy_chunk_size = max(
                policy_chunk_size, min(64, sim_batch_size))
        else:
            assert num_past_policies == 0

            min_policy_chunk_size = 0
            policy_chunk_size = sim_batch_size // num_current_policies

        if policy_chunk_size_override != 0:
            assert policy_chunk_size_override > min_policy_chunk_size
            policy_chunk_size = policy_chunk_size_override

        assert sim_batch_size % policy_chunk_size == 0
        # Allocate enough policy sub-batches to evenly divide the full batch,
        # plus num_policies - 1 extras to handle worst case usage from unused
        # space in each subbatch
        num_policy_chunks = sim_batch_size // policy_chunk_size
        if has_matchmaking:
            num_policy_chunks += total_num_policies - 1

        total_policy_batch_size = num_policy_chunks * policy_chunk_size

        return RolloutConfig(
            num_current_policies = num_current_policies,
            num_past_policies = num_past_policies,
            total_num_policies = total_num_policies,
            num_teams = num_teams,
            team_size = team_size,
            sim_batch_size = sim_batch_size,
            policy_chunk_size = policy_chunk_size,
            num_policy_chunks = num_policy_chunks,
            total_policy_batch_size = total_policy_batch_size,
            self_play_batch_size = self_play_batch_size,
            cross_play_batch_size = cross_play_batch_size,
            past_play_batch_size = past_play_batch_size,
            num_cross_play_matches = num_cross_play_matches,
            num_past_play_matches = num_past_play_matches,
            float_dtype = float_dtype,
            has_matchmaking = has_matchmaking,
        )


class PolicyBatchReorderState(flax.struct.PyTreeNode):
    to_policy_idxs: Optional[jax.Array]
    to_sim_idxs: Optional[jax.Array]
    policy_dims: Tuple[int] = flax.struct.field(pytree_node=False)
    sim_dims: Tuple[int] = flax.struct.field(pytree_node=False)

    def to_policy(self, data):
        def txfm(x):
            if self.to_policy_idxs == None:
                return x.reshape(*self.policy_dims, *x.shape[1:])
            else:
                # FIXME: can we clean to to_policy_idxs to not have OOB
                # indices?
                return x.at[self.to_policy_idxs].get(mode='clip')

        return jax.tree_map(txfm, data)

    def to_sim(self, data):
        if self.to_policy_idxs != None:
            num_flattened_policy_chunks = (
                self.to_policy_idxs.shape[0] * self.to_policy_idxs.shape[1])
        def txfm(x):
            if self.to_sim_idxs == None:
                return x.reshape(*self.sim_dims, *x.shape[2:])
            else:
                flattened_chunks = x.reshape(
                    num_flattened_policy_chunks, *x.shape[2:])

                return flattened_chunks.at[self.to_sim_idxs].get(
                    unique_indices=True)

        return jax.tree_map(txfm, data)


class RolloutState(flax.struct.PyTreeNode):
    step_fn: Callable = flax.struct.field(pytree_node=False)
    prng_key: random.PRNGKey
    rnn_states: Any
    sim_data: FrozenDict
    reorder_state: PolicyBatchReorderState
    policy_assignments: Optional[jax.Array]

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

        reorder_state = _compute_reorder_state(policy_assignments, rollout_cfg)

        sim_data = jax.tree_map(jnp.copy, init_sim_data)

        return RolloutState(
            step_fn = step_fn,
            prng_key = prng_key,
            rnn_states = rnn_states,
            sim_data = sim_data,
            reorder_state = reorder_state,
            policy_assignments = (policy_assignments
                if 'policy_assignments' not in init_sim_data else None),
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


class RolloutCollectState(flax.struct.PyTreeNode):
    store: FrozenDict[str, Any]
    obs_stats: FrozenDict[str, Any]

    @staticmethod
    def create(store_typed_shapes, init_obs_stats):
        return RolloutCollectState(
            store = jax.tree_map(
                lambda x: jnp.empty(x.shape, x.dtype), store_typed_shapes),
            obs_stats = init_obs_stats,
        )

    def save(self, indices, data):
        def save_leaf(v, store): 
            return store.at[indices].set(v)

        data = frozen_dict.freeze(data)

        new_store = self.store
        for k, v in data.items():
            new_store = new_store.copy(
                {k: jax.tree_map(save_leaf, v, new_store[k])})

        return self.replace(store=new_store)

    def set_obs_stats(self, obs_stats):
        return self.replace(obs_stats = obs_stats)


class RolloutManager:
    def __init__(
        self,
        train_cfg: TrainConfig,
        rollout_cfg: RolloutConfig,
        init_rollout_state: RolloutState,
        obs_preprocess: ObservationsPreprocess,
        obs_preprocess_state: FrozenDict[str, Any],
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
        
        def get_obs_typed_shape(o):
            return TypedShape(o.shape, o.dtype)

        typed_shapes['obs'] = jax.tree_map(
            get_obs_typed_shape, sim_data['obs'])

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
            return TypedShape((
                    self._num_bptt_chunks,
                    self._num_bptt_steps,
                    self._num_train_policies,
                    self._num_train_agents_per_policy,
                    *x.shape[1:],
                ), dtype=x.dtype)

        typed_shapes = jax.tree_map(expand_per_step_shapes, typed_shapes)

        typed_shapes['bootstrap_values'] = TypedShape(
            typed_shapes['values'].shape[2:],
            self._cfg.float_dtype)

        typed_shapes['rnn_start_states'] = jax.tree_map(
            lambda x: TypedShape((
                    self._num_bptt_chunks,
                    self._num_train_policies,
                    self._num_train_agents_per_policy,
                    *x.shape[1:],
                ), x.dtype),
            init_rollout_state.rnn_states)

        # Preprocessed observations are stored in the rollouts. Figure
        # out observation shape / dtype after preprocessing
        def preprocess_shape_helper(obs):
            return obs_preprocess.preprocess(
                jax.tree_map(lambda x: x[0:self._num_train_policies],
                    obs_preprocess_state), obs, True)

        preprocessed_obs_shape_dtypes = jax.eval_shape(preprocess_shape_helper,
            jax.tree_map(lambda x: jax.ShapeDtypeStruct(x.shape[2:], x.dtype),
                typed_shapes['obs']))

        typed_shapes['obs'] = jax.tree_map(
            lambda o, n: TypedShape((*o.shape[0:2], *n.shape), n.dtype),
            typed_shapes['obs'], preprocessed_obs_shape_dtypes)

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
        obs_preprocess = train_state_mgr.policy_states.obs_preprocess
        obs_preprocess_train_state = jax.tree_map(
            lambda s: s[0:self._num_train_policies],
            train_state_mgr.policy_states.obs_preprocess_state)

        def iter_bptt_chunk(bptt_chunk, inputs):
            rollout_state, collect_state = inputs

            post_inference_cb = partial(self._post_inference_cb,
                obs_preprocess,
                obs_preprocess_train_state,
                train_state_mgr.train_states,
                bptt_chunk)
            post_step_cb = partial(self._post_step_cb, bptt_chunk)

            with profile("Cache RNN state"):
                collect_state = collect_state.save(bptt_chunk, {
                    'rnn_start_states': self._sim_to_train(
                        rollout_state.rnn_states, rollout_state.reorder_state),
                })

            rollout_state, collect_state = rollout_loop(
                rollout_state, train_state_mgr.policy_states, self._cfg,
                self._num_bptt_steps,  post_inference_cb, post_step_cb,
                collect_state, sample_actions = True, return_debug = False)

            return rollout_state, collect_state 

        collect_state = RolloutCollectState.create(
            self._store_typed_shape_tree,
            obs_preprocess.init_obs_stats(obs_preprocess_train_state, True),
        )

        rollout_state, collect_state = lax.fori_loop(
            0, self._num_bptt_chunks, iter_bptt_chunk,
            (rollout_state, collect_state))

        with profile("Bootstrap Values"):
            bootstrap_values = self._bootstrap_values(
                train_state_mgr.policy_states, train_state_mgr.train_states,
                rollout_state)

            collect_state = collect_state.save(
                slice(None), {'bootstrap_values': bootstrap_values})

        with profile("Finalize Rollouts"):
            rollout_data, metrics = self._finalize_rollouts(
                collect_state.store, metrics)

        return rollout_state, rollout_data, collect_state.obs_stats, metrics

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
            sim_ordering = reorder_state.to_sim(x)
            return sim_ordering[self._sim_to_train_idxs]

        return jax.tree_map(to_train, data)

    def _invert_value_normalization(self, train_states, values):
        @jax.vmap
        def invert_vn(train_state, v):
            return train_state.value_normalizer.invert(
                train_state.value_normalizer_state, v)

        return invert_vn(train_states, values)

    def _bootstrap_values(self, policy_states, train_states, rollout_state):
        rnn_states = rollout_state.rnn_states
        obs = rollout_state.sim_data['obs']
        reorder_state = rollout_state.reorder_state

        rnn_states, obs = self._sim_to_train((rnn_states, obs), reorder_state)

        policy_states = jax.tree_map(
            lambda x: x[0:self._num_train_policies], policy_states)

        @jax.vmap
        def critic_fn(state, rnn_states, obs):
            preprocessed_obs = state.obs_preprocess.preprocess(
                state.obs_preprocess_state, obs, False)

            policy_out, rnn_states = state.apply_fn(
                {
                    'params': state.params,
                    'batch_stats': state.batch_stats,
                },
                rnn_states,
                preprocessed_obs,
                train = False,
                method='critic_only',
            )

            return policy_out['values']

        values = critic_fn(policy_states, rnn_states, obs)

        return self._invert_value_normalization(train_states, values)

    def _post_inference_cb(
        self,
        obs_preprocess: ObservationsPreprocess,
        obs_preprocess_state: FrozenDict[str, Any],
        train_states: PolicyTrainState,
        bptt_chunk: int,
        bptt_step: int,
        obs: FrozenDict[str, Any],
        preprocessed_obs: FrozenDict[str, Any],
        policy_out: FrozenDict[str, Any],
        reorder_state: PolicyBatchReorderState,
        collect_state: RolloutCollectState,
    ):
        with profile('Pre Step Rollout Store'):
            values = self._policy_to_train(policy_out['values'], reorder_state)
            values = self._invert_value_normalization(train_states, values)

            preprocessed_obs, actions, log_probs = self._policy_to_train(
                (preprocessed_obs,
                 policy_out['actions'],
                 policy_out['log_probs']),
                reorder_state)

            save_data = {
                'obs': preprocessed_obs,
                'actions': actions,
                'log_probs': log_probs,
                'values': values,
            }

            collect_state = collect_state.save(
                (bptt_chunk, bptt_step), save_data)

            new_obs_stats = obs_preprocess.update_obs_stats(
                obs_preprocess_state,
                collect_state.obs_stats,
                bptt_chunk * self._num_bptt_steps + bptt_step,
                self._policy_to_train(obs, reorder_state),
                True,
            )

            collect_state = collect_state.set_obs_stats(new_obs_stats)

            return collect_state

    def _post_step_cb(
        self,
        bptt_chunk: int,
        bptt_step: int,
        dones: jax.Array,
        rewards: jax.Array,
        reorder_state: PolicyBatchReorderState,
        collect_state: RolloutCollectState,
    ):
        with profile('Post Step Rollout Store'):
            save_data = self._sim_to_train({
                'dones': dones,
                'rewards': rewards,
            }, reorder_state)
            return collect_state.save(
                (bptt_chunk, bptt_step), save_data)

    def _finalize_rollouts(self, rollouts, metrics):
        rollouts, bootstrap_values = rollouts.pop('bootstrap_values')

        if self._use_advantages:
            advantages = self._compute_advantages_fn(rollouts['rewards'],
                rollouts['values'], rollouts['dones'], bootstrap_values)

            rollouts = rollouts.copy({
                'advantages': advantages,
            })

            returns = advantages + rollouts['values']
        else:
            returns = self._compute_returns_fn(rollouts['rewards'],
                rollouts['dones'], bootstrap_values)

        rollouts = rollouts.copy({
            'returns': returns,
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

        metrics = metrics.record({
            'Rewards': rollouts['rewards'],
            'Values': rollouts['values'],
            'Returns': rollouts['returns'],
            'Bootstrap Values': bootstrap_values,
        })

        if self._use_advantages:
            metrics = metrics.record({
                'Advantages': rollouts['advantages'],
            })

        return RolloutData(
            data = rollouts.copy({
                'rnn_start_states': rnn_start_states,
            }),
            num_train_seqs_per_policy = self._num_train_seqs_per_policy,
            num_train_policies = self._num_train_policies,
        ), metrics


def rollout_loop(
    rollout_state: RolloutState,
    policy_states: PolicyState,
    rollout_cfg: RolloutConfig,
    num_steps: int,
    post_inference_cb: Callable,
    post_step_cb: Callable,
    cb_state: Any,
    **policy_kwargs,
):
    def obs_preprocess_fn(state, obs):
        return state.obs_preprocess.preprocess(
            state.obs_preprocess_state, obs, True)

    @jax.vmap
    def policy_fn(state, sample_key, rnn_states, preprocessed_obs):
        return state.apply_fn(
            {
                'params': state.params,
                'batch_stats': state.batch_stats,
            },
            sample_key,
            rnn_states,
            preprocessed_obs,
            train = False,
            **policy_kwargs,
            method = 'rollout',
        )

    rnn_reset_fn = policy_states.rnn_reset_fn

    def reorder_policy_states(states, assignments, reorder_state):
        if not rollout_cfg.has_matchmaking:
            return states 
        else:
            # FIXME: more efficient way to get this?
            state_idxs = reorder_state.to_policy(assignments)[:, 0]
            return jax.tree_map(lambda x: x[state_idxs], states)

    def rollout_iter(step_idx, iter_state):
        rollout_state, cb_state = iter_state

        prng_key = rollout_state.prng_key
        rnn_states = rollout_state.rnn_states
        sim_data = rollout_state.sim_data
        reorder_state = rollout_state.reorder_state
        policy_assignments = rollout_state.policy_assignments

        with profile('Policy Inference'):
            prng_key, step_key = random.split(prng_key)
            step_keys = random.split(
                step_key, rollout_cfg.num_policy_chunks)

            reordered_policy_states = reorder_policy_states(
                policy_states, policy_assignments, reorder_state)

            rnn_states, policy_obs = reorder_state.to_policy(
                (rnn_states, sim_data['obs']))

            preprocessed_obs = obs_preprocess_fn(
                reordered_policy_states, policy_obs)

            policy_out, rnn_states = policy_fn(reordered_policy_states,
                step_keys, rnn_states, preprocessed_obs)

            cb_state = post_inference_cb(
                step_idx, policy_obs, preprocessed_obs,
                policy_out, reorder_state, cb_state)

            # rnn_states are kept across the loop in sim ordering,
            # because the ordering is stable, unlike policy batches that 
            # can shift when assignments change
            rnn_states = reorder_state.to_sim(rnn_states)

        with profile('Rollout Step'):
            sim_data = sim_data.copy({
                'actions': reorder_state.to_sim(policy_out['actions']),
            })

            sim_data = frozen_dict.freeze(rollout_state.step_fn(sim_data))

            dones = sim_data['dones'].astype(jnp.bool_)
            rewards = sim_data['rewards'].astype(rollout_cfg.float_dtype)

            # rnn_states kept in sim ordering
            rnn_states = rnn_reset_fn(rnn_states, dones)

            if rollout_cfg.has_matchmaking:
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
    global_indices = jnp.arange(rollout_cfg.sim_batch_size)

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
    num_matches = rollout_cfg.num_past_play_matches
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

    assert policy_assignments.shape[0] == rollout_cfg.sim_batch_size
    return policy_assignments


def _update_policy_assignments(
    assignments,
    dones,
    assign_rnd,
    rollout_cfg,
):
    cross_start = rollout_cfg.self_play_batch_size
    cross_end = cross_start + rollout_cfg.cross_play_batch_size
    
    past_start = cross_end
    past_end = past_start + rollout_cfg.past_play_batch_size

    if rollout_cfg.cross_play_batch_size > 0:
        assign_rnd, cross_rnd = random.split(assign_rnd)

        assignments = assignments.at[cross_start:cross_end].set(
            _cross_play_matchmake(assignments[cross_start:cross_end],
                dones[cross_start:cross_end], cross_rnd, rollout_cfg))

    if rollout_cfg.past_play_batch_size > 0:
        assign_rnd, past_rnd = random.split(assign_rnd)

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
        partial_base + jnp.arange(0, P * C, C) - full_chunk_counts)

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

    full_partial_mask = offsets_from_starts < expanded_full_chunk_counts
    scatter_positions = jnp.where(
        full_partial_mask, full_chunk_indices, partial_chunk_indices)

    to_policy_idxs = jnp.full((B * C), assignments.size, jnp.int32).at[
        scatter_positions].set(sort_idxs, unique_indices=True).reshape(B, C)

    # This last step isn't strictly necessary, but ensures that each chunk will
    # only gather data for its own policy. Could go a step further and also 
    # remove OOB indices from empty chunks, but it will still work correctly
    # because JAX clamps out of bounds indices. It's worth noting this makes
    # it harder to identify invalid training data if trying to go from
    # policy => train ordering directly in future.
    to_policy_idxs = jnp.where(to_policy_idxs != assignments.size,
        to_policy_idxs, to_policy_idxs[:, 0:1])

    to_sim_idxs = jnp.empty_like(assignments).at[
        sort_idxs].set(scatter_positions, unique_indices=True)

    return to_policy_idxs, to_sim_idxs


def _compute_reorder_state(
    assignments,
    rollout_cfg,
):
    if rollout_cfg.has_matchmaking:
        to_policy_idxs, to_sim_idxs = _compute_reorder_chunks(
            assignments, rollout_cfg.total_num_policies, 
            rollout_cfg.policy_chunk_size, rollout_cfg.num_policy_chunks)
    else:
        to_policy_idxs = None
        to_sim_idxs = None

    return PolicyBatchReorderState(
        to_policy_idxs = to_policy_idxs,
        to_sim_idxs = to_sim_idxs,
        policy_dims = (
            rollout_cfg.total_num_policies, rollout_cfg.policy_chunk_size),
        sim_dims = (rollout_cfg.sim_batch_size,),
    )
