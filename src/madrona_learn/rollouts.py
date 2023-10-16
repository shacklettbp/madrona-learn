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
from .utils import TypedShape, make_pbt_reorder_funcs

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
            policy_assignments = init_sim_data['policy_assignments']
            reorder_idxs = jnp.argsort(policy_assignments)
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
        mb_per_step, bootstrap_values = mb.pop('bootstrap_values')

        mb = jax.tree_map(lambda x: jnp.swapaxes(x, 0, 1), mb)

        return mb.copy({
            'rnn_start_states': rnn_start_states,
            'bootstrap_values': bootstrap_values,
        })


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
        train_state_mgr: TrainStateManager,
        init_rollout_state: RolloutState,
    ):
        cpu_dev = jax.devices('cpu')[0]

        self._num_bptt_chunks = cfg.num_bptt_chunks
        self._num_bptt_steps = icfg.num_bptt_steps
        self._num_train_policies = cfg.pbt_ensemble_size
        self._train_agents_per_policy = icfg.train_agents_per_policy
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

        def rollout_fn_wrapper(state, sample_keys, rnn_states, obs):
            actions, log_probs, values, rnn_states = infer_wrapper(
                'rollout', state, sample_keys, rnn_states, obs)

            values = state.value_normalize_fn(
                { 'batch_stats': state.value_normalize_stats },
                mode='invert',
                update_stats=False,
                x=values,
            )

            return actions, log_probs, values, rnn_states

        def critic_fn_wrapper(state, rnn_states, obs):
            values, _ = infer_wrapper(
                'critic_only', state, rnn_states, obs)

            return state.value_normalize_fn(
                { 'batch_stats': state.value_normalize_stats },
                mode='invert',
                update_stats=False,
                x=values,
            )

        self._rollout_fn = jax.vmap(rollout_fn_wrapper)
        self._critic_fn = jax.vmap(critic_fn_wrapper)
        self._rnn_reset_fn = jax.vmap(
            train_state_mgr.train_states.rnn_reset_fn)
        self._finalize_rollouts_fn = partial(
            cfg.algo.finalize_rollouts_fn(), cfg)

        self._prep_data_for_policy, self._prep_data_for_sim = \
            make_pbt_reorder_funcs(self._is_dynamic_policy_assignment,
                                   self._num_rollout_policies)

    def _canonicalize_float_dtypes(self, xs):
        def canonicalize(x):
            if jnp.issubdtype(x.dtype, jnp.floating):
                return jnp.asarray(x, dtype=self._float_dtype)
            else:
                return x

        return jax.tree_map(canonicalize, xs)

    def _slice_train_policy_data(self, data):
        return jax.tree_map(lambda x: x[0:self._num_train_policies], data)

    def _rollout_infer(self, train_states, rollout_state):
        next_key, step_key = random.split(rollout_state.prng_key)
        sample_keys = random.split(step_key, self._num_rollout_policies)

        rnn_states = rollout_state.rnn_states
        obs = rollout_state.sim_data['obs']
        obs = self._canonicalize_float_dtypes(obs)

        policy_obs = self._prep_data_for_policy(
            obs, rollout_state.reorder_idxs)
        actions, log_probs, values, rnn_states = \
            self._rollout_fn(train_states, sample_keys, rnn_states, policy_obs)

        sim_actions = self._prep_data_for_sim(
            actions, rollout_state.reorder_idxs)

        sim_data_actions_set = rollout_state.sim_data.copy({
            'actions': sim_actions,
        })

        rollout_state = rollout_state.update(
            prng_key = next_key,
            rnn_states = rnn_states,
            sim_data = sim_data_actions_set,
        )

        save_data = FrozenDict(
            obs = policy_obs,
            actions = actions,
            log_probs = log_probs,
            values = values,
        )

        save_data = self._slice_train_policy_data(save_data)

        return rollout_state, save_data

    def _critic_infer(self, train_states, rollout_state):
        rnn_states = rollout_state.rnn_states
        obs = rollout_state.sim_data['obs']
        obs = self._canonicalize_float_dtypes(obs)

        policy_obs = self._prep_data_for_policy(obs, rollout_state.reorder_idxs)
        values = self._critic_fn(train_states, rnn_states, policy_obs)

        return self._slice_train_policy_data(values)

    def _step_rollout_state(self, rollout_state):
        step_fn = rollout_state.step_fn
        sim_data = rollout_state.sim_data
        rnn_states = rollout_state.rnn_states

        with profile('Simulator Step'):
            sim_data = frozen_dict.freeze(step_fn(sim_data))

        dones = sim_data['dones']
        rewards = sim_data['rewards']

        dones = jnp.asarray(dones, dtype=jnp.bool_)
        rewards = jnp.asarray(rewards, dtype=self._float_dtype)

        if self._is_dynamic_policy_assignment:
            reorder_idxs = jnp.argsort(sim_data['policy_assignments'])
        else:
            reorder_idxs = None

        dones, rewards = self._prep_data_for_policy(
            (dones, rewards), reorder_idxs)

        rnn_states = self._rnn_reset_fn(rnn_states, dones)

        rollout_state = rollout_state.update(
            sim_data = sim_data,
            reorder_idxs = reorder_idxs,
            rnn_states = rnn_states,
        )

        save_data = FrozenDict(
            dones = dones,
            rewards = rewards,
        )

        save_data = self._slice_train_policy_data(save_data)

        return rollout_state, save_data

    def _finalize_rollouts(self, rollouts):
        rollouts = self._finalize_rollouts_fn(rollouts)

        rollouts, rnn_start_states = rollouts.pop('rnn_start_states')
        rollouts, bootstrap_values = rollouts.pop('bootstrap_values')

        # Per Step rollouts reshaped / transposed as follows:
        # [C, T / C, P, B, ...] => [P, B * C, T / C, ...]

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
                'bootstrap_values': bootstrap_values,
            }),
        )

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

        with profile("Reshape Rollouts"):
            rollout_data = self._finalize_rollouts(rollout_store.data)

        return rollout_state, rollout_data
