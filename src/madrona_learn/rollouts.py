import jax
from jax import lax, random, numpy as jnp
import flax
from flax import linen as nn
from flax.core import frozen_dict, FrozenDict

from dataclasses import dataclass
from typing import List, Optional, Callable, Any
from functools import partial

from .cfg import TrainConfig
from .actor_critic import ActorCritic
from .algo_common import InternalConfig, compute_advantages, compute_returns
from .metrics import TrainingMetrics, Metric
from .profile import profile
from .train_state import TrainStateManager, PolicyState, PolicyTrainState
from .utils import TypedShape, convert_float_leaves

class RolloutState(flax.struct.PyTreeNode):
    step_fn: Callable = flax.struct.field(pytree_node=False)
    prng_key: random.PRNGKey
    rnn_states: Any
    sim_data: FrozenDict
    reorder_idxs: Optional[jax.Array]

    @staticmethod
    def create(
        step_fn,
        prng_key,
        rnn_states,
        init_sim_data,
    ):
        if 'policy_assignments' in init_sim_data:
            reorder_idxs = jnp.argsort(
                init_sim_data['policy_assignments'].squeeze(axis=-1))
        else:
            reorder_idxs = None

        sim_data = jax.tree_map(jnp.copy, init_sim_data)

        return RolloutState(
            step_fn = step_fn,
            prng_key = prng_key,
            rnn_states = rnn_states,
            sim_data = sim_data,
            reorder_idxs = reorder_idxs,
        )

    def update(
        self,
        prng_key=None,
        rnn_states=None,
        sim_data=None,
        reorder_idxs=None
    ):
        return RolloutState(
            step_fn = self.step_fn,
            prng_key = prng_key if prng_key != None else self.prng_key,
            rnn_states = rnn_states if rnn_states != None else self.rnn_states,
            sim_data = sim_data if sim_data != None else self.sim_data,
            reorder_idxs = (
                reorder_idxs if reorder_idxs != None else self.reorder_idxs),
        )


class RolloutData(flax.struct.PyTreeNode):
    data: FrozenDict[str, Any]

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


class RolloutExecutor:
    def __init__(
        self,
        cfg: TrainConfig,
        icfg: InternalConfig,
        train_state_mgr: TrainStateManager,
        init_rollout_state: RolloutState,
    ):
        cpu_dev = jax.devices('cpu')[0]

        self._num_bptt_chunks = cfg.num_bptt_chunks
        self._num_bptt_steps = icfg.num_bptt_steps
        self._num_train_policies = cfg.pbt_ensemble_size
        self._train_agents_per_policy = icfg.train_agents_per_policy
        self._num_rollout_policies = icfg.num_rollout_policies
        self._float_dtype = icfg.float_storage_type
        self._use_advantages = cfg.compute_advantages
        self._compute_advantages_fn = partial(compute_advantages, cfg)
        self._compute_returns_fn = partial(compute_returns, cfg)

        typed_shapes = {}

        sim_data = init_rollout_state.sim_data
        
        self._is_dynamic_policy_assignment = \
            init_rollout_state.reorder_idxs != None
        assert(sim_data['actions'].shape[0] == icfg.rollout_batch_size)

        def get_typed_shape(x):
            return TypedShape(x.shape, x.dtype)

        typed_shapes['obs'] = jax.tree_map(get_typed_shape, sim_data['obs'])

        typed_shapes['actions'] = TypedShape(
            sim_data['actions'].shape, sim_data['actions'].dtype)

        typed_shapes['log_probs'] = TypedShape(
            typed_shapes['actions'].shape, self._float_dtype)

        typed_shapes['rewards'] = TypedShape(
          sim_data['rewards'].shape, self._float_dtype)

        typed_shapes['dones'] = TypedShape(
            sim_data['dones'].shape, jnp.bool_)

        typed_shapes['values'] = TypedShape(
            typed_shapes['rewards'].shape, self._float_dtype)

        def expand_per_step_shapes(x):
            if jnp.issubdtype(x.dtype, jnp.floating):
                dtype = self._float_dtype
            else:
                dtype = x.dtype

            return TypedShape((
                    self._num_bptt_chunks,
                    self._num_bptt_steps,
                    self._num_train_policies,
                    self._train_agents_per_policy,
                    *x.shape[1:],
                ), dtype=dtype)

        typed_shapes = jax.tree_map(expand_per_step_shapes, typed_shapes)

        typed_shapes['bootstrap_values'] = TypedShape(
            typed_shapes['values'].shape[2:],
            self._float_dtype)

        typed_shapes['rnn_start_states'] = jax.tree_map(
            lambda x: TypedShape((
                    self._num_bptt_chunks,
                    self._num_train_policies,
                    self._train_agents_per_policy,
                    *x.shape[2:],
                ), x.dtype),
            init_rollout_state.rnn_states)

        self._store_typed_shape_tree = frozen_dict.freeze(typed_shapes)

    def add_metrics(
        self, 
        cfg: TrainConfig,
        metrics: FrozenDict[str, Metric],
    ):
        new_metrics = {
            'Rewards': Metric.init(True),
            'Returns': Metric.init(True),
            'Values': Metric.init(True),
        }

        if cfg.compute_advantages:
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
                    'rnn_start_states': self._slice_train_policy_data(
                        rollout_state.rnn_states),
                })

            rollout_state, rollout_store = rollout_loop(
                rollout_state, train_state_mgr.policy_states,
                self._num_rollout_policies, self._num_bptt_steps, 
                post_inference_cb, post_step_cb, rollout_store,
                self._float_dtype, sample_actions = True, return_debug = False)

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

    def _slice_train_policy_data(self, data):
        return jax.tree_map(lambda x: x[0:self._num_train_policies], data)

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
        prep_for_policy = _make_pbt_reorder_funcs(
            rollout_state.reorder_idxs != None, self._num_rollout_policies)[0]

        rnn_states = rollout_state.rnn_states
        obs = rollout_state.sim_data['obs']

        obs = convert_float_leaves(obs, self._float_dtype)
        obs = prep_for_policy(obs, rollout_state.reorder_idxs)

        rnn_states, obs = self._slice_train_policy_data((rnn_states, obs))
        policy_states = self._slice_train_policy_data(policy_states)

        @jax.vmap
        def critic_fn(state, rnn_states, obs):
            policy_out = state.apply_fn(
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
        ), metrics

    def _post_inference_cb(
        self,
        train_states: PolicyTrainState,
        bptt_chunk: int,
        bptt_step: int,
        policy_obs: FrozenDict[str, Any],
        policy_out: FrozenDict[str, Any],
        rollout_store: RolloutStore,
    ):
        with profile('Pre Step Rollout Store'):
            values = self._slice_train_policy_data(policy_out['values'])
            values = self._invert_value_normalization(train_states, values)

            obs, actions, log_probs = self._slice_train_policy_data(
                (policy_obs, policy_out['actions'], policy_out['log_probs']))

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
        rollout_store: RolloutStore,
    ):
        with profile('Post Step Rollout Store'):
            save_data = self._slice_train_policy_data({
                'dones': dones,
                'rewards': rewards,
            })
            return rollout_store.save(
                (bptt_chunk, bptt_step), save_data)

def rollout_loop(
    rollout_state: RolloutState,
    policy_states: PolicyState,
    num_policies: int,
    num_steps: int,
    post_inference_cb: Callable,
    post_step_cb: Callable,
    cb_state: Any,
    preferred_float_dtype: jnp.dtype,
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
    rnn_reset_fn = jax.vmap(policy_states.rnn_reset_fn)

    prep_for_policy, prep_for_sim = _make_pbt_reorder_funcs(
        rollout_state.reorder_idxs != None, num_policies)

    def rollout_iter(step_idx, iter_state):
        rollout_state, cb_state = iter_state

        prng_key = rollout_state.prng_key
        rnn_states = rollout_state.rnn_states
        sim_data = rollout_state.sim_data
        reorder_idxs = rollout_state.reorder_idxs

        with profile('Policy Inference'):
            prng_key, step_key = random.split(prng_key)
            step_keys = random.split(step_key, num_policies)

            policy_obs = convert_float_leaves(
                sim_data['obs'], preferred_float_dtype)

            policy_obs = prep_for_policy(policy_obs, reorder_idxs)
            policy_out = policy_fn(
                policy_states, step_keys, rnn_states, policy_obs)

            cb_state = post_inference_cb(
                step_idx, policy_obs, policy_out, cb_state)

        with profile('Rollout Step'):
            sim_data = sim_data.copy({
                'actions': prep_for_sim(policy_out['actions'], reorder_idxs),
            })

            sim_data = frozen_dict.freeze(rollout_state.step_fn(sim_data))

            if reorder_idxs != None:
                reorder_idxs = jnp.argsort(
                    sim_data['policy_assignments'].squeeze(axis=-1))

            dones = sim_data['dones'].astype(jnp.bool_)
            rewards = sim_data['rewards'].astype(preferred_float_dtype)

            dones, rewards = prep_for_policy((dones, rewards), reorder_idxs)

            rnn_states = rnn_reset_fn(rnn_states, dones)

            cb_state = post_step_cb(step_idx, dones, rewards, cb_state)

        rollout_state = rollout_state.update(
            prng_key = prng_key,
            rnn_states = rnn_states,
            sim_data = sim_data,
            reorder_idxs = reorder_idxs,
        )

        return rollout_state, cb_state

    return lax.fori_loop(0, num_steps, rollout_iter, (rollout_state, cb_state))

def _make_pbt_reorder_funcs(dyn_assignment, num_policies):
    def group_into_policy_batches(args):
        return jax.tree_map(
            lambda x: x.reshape(num_policies, -1, *x.shape[1:]), args)

    if dyn_assignment:
        def prep_for_policy(args, sort_idxs):
            reordered = jax.tree_map(lambda x: jnp.take(x, sort_idxs, 0), args)
            return group_into_policy_batches(reordered)

        def prep_for_sim(args, sort_idxs):
            args = jax.tree_map(lambda x: x.reshape(-1, *x.shape[2:]), args)

            unsort_idxs = jnp.arange(sort_idxs.shape[0])[sort_idxs]
            return jax.tree_map(lambda x: jnp.take(x, unsort_idxs, 0), args)
    else:
        def prep_for_policy(args, sort_idxs):
            assert(sort_idxs == None)
            return group_into_policy_batches(args)

        def prep_for_sim(args, sort_idxs):
            assert(sort_idxs == None)
            return jax.tree_map(lambda x: x.reshape(-1, *x.shape[2:]), args)

    return prep_for_policy, prep_for_sim
