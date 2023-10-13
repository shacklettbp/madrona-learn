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
from .algo_common import InternalConfig
from .profile import profile
from .train_state import TrainStateManager, PolicyTrainState
from .utils import TypedShape

class RolloutState(flax.struct.PyTreeNode):
    step_fn: Callable = flax.struct.field(pytree_node=False)
    prng_key: random.PRNGKey
    sim_data: FrozenDict
    reorder_idxs: Optional[jax.Array]
    rnn_states: Any

    @staticmethod
    def create(
        step_fn,
        prng_key,
        sim_data,
        rnn_states
    ):
        if 'policy_assignments' in sim_data:
            policy_assignments = sim_data['policy_assignments']
            reorder_idxs = jnp.argsort(policy_assignments)
        else:
            reorder_idxs = None

        return RolloutState(
            step_fn = step_fn,
            prng_key = prng_key,
            sim_data = frozen_dict.freeze(sim_data),
            reorder_idxs = reorder_idxs,
            rnn_states = rnn_states,
        )

    def update(
        self,
        prng_key=None,
        sim_data=None,
        rnn_states=None,
        reorder_idxs=None
    ):
        return RolloutState(
            step_fn = self.step_fn,
            prng_key = prng_key if prng_key != None else self.prng_key,
            sim_data = sim_data if sim_data != None else self.sim_data,
            reorder_idxs = (
                reorder_idxs if reorder_idxs != None else self.reorder_idxs),
            rnn_states = rnn_states if rnn_states != None else self.rnn_states,
        )


class RolloutStore(flax.struct.PyTreeNode):
    data : FrozenDict

    @staticmethod
    def create_from_tree(typed_shapes : FrozenDict):
        data = jax.tree_map(
            lambda x: jnp.empty(x.shape, x.dtype), typed_shapes)
        return RolloutStore(
            data = data,
        )

    def save(self, k, indices, values):
        def save_leaf(store, v): 
            return store.at[indices].set(v)

        updated = jax.tree_map(save_leaf, self.data[k], values)

        return RolloutStore(data = self.data.copy({k: updated}))


class RolloutExecutor:
    def __init__(
        self,
        cfg: TrainConfig,
        icfg: InternalConfig,
        policy: ActorCritic,
        init_rollout_state: RolloutState
    ):
        cpu_dev = jax.devices('cpu')[0]

        self._num_bptt_chunks = cfg.num_bptt_chunks
        self._num_bptt_steps = icfg.num_bptt_steps
        self._num_train_policies = cfg.pbt_ensemble_size
        self._num_rollout_policies = \
            cfg.pbt_ensemble_size * cfg.pbt_history_len
        self._float_dtype = icfg.float_storage_type

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

        def convert_per_step_shapes(x):
            if jnp.issubdtype(x.dtype, jnp.floating):
                dtype = self._float_dtype
            else:
                dtype = x.dtype

            return TypedShape((
                    self._num_bptt_chunks,
                    self._num_bptt_steps,
                    self._num_train_policies,
                    icfg.train_agents_per_policy,
                    *x.shape[1:],
                ), dtype=dtype)

        typed_shapes = jax.tree_map(convert_per_step_shapes, typed_shapes)

        typed_shapes['bootstrap_values'] = TypedShape(
            typed_shapes['values'].shape[2:],
            self._float_dtype)

        typed_shapes['rnn_start_states'] = jax.tree_map(
            lambda x: TypedShape((
                    self._num_bptt_chunks,
                    self._num_train_policies,
                    icfg.train_agents_per_policy,
                    *x.shape[2:],
                ), x.dtype),
            init_rollout_state.rnn_states)

        self._store_typed_shape_tree = frozen_dict.freeze(typed_shapes)

        def infer_wrapper(method, state, *args):
            return state.apply_fn(
                {
                    'params': state.params,
                    'batch_stats': state.batch_stats,
                },
                *args,
                train=False,
                method=method,
            )

        self._rollout_fn = jax.vmap(
            partial(infer_wrapper, 'rollout'))

        self._critic_fn = jax.vmap(
            partial(infer_wrapper, 'critic_only'))

        def rnn_reset_fn(rnn_states, should_clear):
            return policy.clear_recurrent_state(rnn_states, should_clear)

        self._rnn_reset_fn = jax.vmap(rnn_reset_fn)

    def _slice_train_policy_data(self, data):
        return jax.tree_map(lambda x: x[0:self._num_train_policies], data)

    def _group_into_policy_batches(self, args):
        def rebatch(x):
            return x.reshape(self._num_rollout_policies, -1, *x.shape[1:])

        return jax.tree_map(rebatch, args)

    def _reorder_into_policy_batches(self, args, sort_idxs):
        reordered = jax.tree_map(lambda x: jnp.take(x, sort_idxs, 0), args)
        return self._group_into_policy_batches(reordered)

    def _rollout_infer(self, train_states, rollout_state):
        next_key, step_key = random.split(rollout_state.prng_key)
        sample_keys = random.split(step_key, self._num_rollout_policies)

        rnn_states = rollout_state.rnn_states
        obs = rollout_state.sim_data['obs']

        def convert_float_ob(o):
            if jnp.issubdtype(o.dtype, jnp.floating):
                return jnp.asarray(o, dtype=self._float_dtype)
            else:
                return o

        obs = jax.tree_map(convert_float_ob, obs)

        if self._is_dynamic_policy_assignment:
            # Sort policy assignments
            sorted_obs = self._reorder_into_policy_batches(
                obs, rollout_state.reorder_idxs)

            actions, log_probs, values, rnn_states = \
                self._rollout_fn(train_states, sample_keys, rnn_states, obs)

            # flatten policy dim for simulator input & reverse sort
            sim_actions = actions.reshape(-1, *x.shape[2:])
            unsort_idxs = jnp.arange(sort_idxs.shape[0])[sort_idxs]
            sim_actions = jnp.take(sim_actions, unsort_idxs, 0)
        else:
            sorted_obs = self._group_into_policy_batches(obs)

            actions, log_probs, values, rnn_states = \
                self._rollout_fn(train_states, sample_keys, rnn_states, sorted_obs)

            # flatten policy dim for simulator input
            sim_actions = actions.reshape(-1, *actions.shape[2:])

        # FIXME
        sim_data_actions_set = rollout_state.sim_data.copy({
            'actions': rollout_state.sim_data['actions'].at[:].set(sim_actions)
        })

        rollout_state = rollout_state.update(
            prng_key = next_key,
            rnn_states = rnn_states,
            sim_data = sim_data_actions_set,
        )

        save_data = FrozenDict(
            obs = sorted_obs,
            actions = actions,
            log_probs =  log_probs,
            values = values,
        )

        save_data = self._slice_train_policy_data(save_data)

        return rollout_state, save_data

    def _critic_infer(self, train_states, rollout_state):
        rnn_states = rollout_state.rnn_states
        obs = rollout_state.sim_data['obs']

        if self._is_dynamic_policy_assignment:
            sorted_obs = self._reorder_into_policy_batches(
                obs, rollout_state.reorder_idxs)

            values, _ = self._critic_fn(train_states, rnn_states, sorted_obs)
        else:
            obs = self._group_into_policy_batches(obs)
            values, _= self._critic_fn(train_states, rnn_states, obs)

        return self._slice_train_policy_data(values)

    def _step_rollout_state(self, rollout_state):
        step_fn = rollout_state.step_fn
        sim_data = rollout_state.sim_data

        with profile('Simulator Step'):
            sim_data = step_fn(sim_data)

        rnn_states = rollout_state.rnn_states
        dones = rollout_state.sim_data['dones']
        rewards = rollout_state.sim_data['rewards']

        dones = jnp.asarray(dones, dtype=jnp.bool_)
        rewards = jnp.asarray(rewards, dtype=self._float_dtype)

        if self._is_dynamic_policy_assignment:
            reorder_idxs = jnp.argsort(sim_data['policy_assignments'])
            dones, rewards = self._reorder_into_policy_batches(
                (dones, rewards), reorder_idxs)
        else:
            reorder_idxs = None
            dones, rewards = self._group_into_policy_batches((dones, rewards))

        rnn_states = self._rnn_reset_fn(rnn_states, dones)

        rollout_state = rollout_state.update(
            sim_data = step_fn(sim_data),
            reorder_idxs = reorder_idxs,
            rnn_states = rnn_states,
        )

        save_data = FrozenDict(
            dones = dones,
            rewards = rewards,
        )

        save_data = self._slice_train_policy_data(save_data)

        return rollout_state, save_data

    def collect(
        self,
        rollout_state: RolloutState,
        train_state_mgr: TrainStateManager,
    ):
        def rollout_iter(bptt_step, inputs):
            rollout_state, rollout_store, bptt_chunk = inputs

            with profile('Policy Infer', gpu=True):
                rollout_state, pre_step_save_data = self._rollout_infer(
                    train_state_mgr.train_states, rollout_state)

            with profile('Pre Step Rollout Store'):
                for k in ['obs', 'actions', 'log_probs', 'values']:
                    rollout_store = rollout_store.save(
                        k, (bptt_chunk, bptt_step), pre_step_save_data[k])

            with profile('Rollout Step'):
                rollout_state, post_step_save_data = self._step_rollout_state(
                    rollout_state)

            with profile('Post Step Rollout Store'):
                for k in ['rewards', 'dones']:
                    rollout_store = rollout_store.save(
                        k, (bptt_chunk, bptt_step), post_step_save_data[k])

            return rollout_state, rollout_store, bptt_chunk

        def iter_bptt_chunk(bptt_chunk, inputs):
            rollout_state, rollout_store = inputs

            with profile("Cache RNN state"):
                rollout_store = rollout_store.save('rnn_start_states',
                    bptt_chunk, rollout_state.rnn_states)

            rollout_state, rollout_store, _ = lax.fori_loop(
                0, self._num_bptt_steps, rollout_iter,
                (rollout_state, rollout_store, bptt_chunk))

            return rollout_state, rollout_store

        rollout_store = RolloutStore.create_from_tree(
            self._store_typed_shape_tree)

        rollout_state, rollout_store = lax.fori_loop(
            0, self._num_bptt_chunks, iter_bptt_chunk,
            (rollout_state, rollout_store))

        with profile("Bootstrap Values"):
            bootstrap_values = self._critic_infer(
                train_state_mgr.train_states, rollout_state)

            rollout_store = rollout_store.save('bootstrap_values',
                slice(None), bootstrap_values)

        return rollout_state, rollout_store
