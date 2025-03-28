import jax
from jax import lax, random, numpy as jnp
import flax
from flax import linen as nn
from flax.core import frozen_dict, FrozenDict

import dataclasses
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Callable, Any, Union
from functools import partial
import itertools
import math

from .cfg import TrainConfig, DiscreteActionsConfig, ContinuousActionsConfig
from .actor_critic import ActorCritic
from .algo_common import compute_advantages, compute_returns
from .metrics import TrainingMetrics, Metric
from .pbt import (
    PBTMatchmakeConfig,
    pbt_init_matchmaking,
    pbt_update_matchmaking,
)
from .profile import profile
from .train_state import TrainStateManager, PolicyState, PolicyTrainState
from .utils import TypedShape
from .observations import ObservationsPreprocess

@dataclass(frozen = True)
class RolloutConfig:
    sim_batch_size: int
    num_worlds: int
    actions_cfg: Dict[str, Union[DiscreteActionsConfig, ContinuousActionsConfig]]
    policy_chunk_size: int
    num_policy_chunks: int
    total_policy_batch_size: int
    reward_gamma: float
    policy_dtype: jnp.dtype
    reward_dtype: jnp.dtype
    prob_dtype: jnp.dtype
    pbt: PBTMatchmakeConfig

    @staticmethod
    def setup(
        num_current_policies: int,
        num_past_policies: int,
        num_teams: int,
        team_size: int,
        sim_batch_size: int,
        actions_cfg: Dict[str, Union[DiscreteActionsConfig, ContinuousActionsConfig]],
        self_play_portion: float,
        cross_play_portion: float,
        past_play_portion: float,
        static_play_portion: float,
        reward_gamma: float,
        custom_policy_ids: List[int],
        policy_dtype: jnp.dtype,
        reward_dtype: jnp.dtype = jnp.float32,
        prob_dtype: jnp.dtype = jnp.float32,
        policy_chunk_size_override: int = 0,
    ):
        pbt = PBTMatchmakeConfig.setup(
            num_current_policies = num_current_policies,
            num_past_policies = num_past_policies,
            num_teams = num_teams,
            team_size = team_size,
            sim_batch_size = sim_batch_size,
            self_play_portion = self_play_portion,
            cross_play_portion = cross_play_portion,
            past_play_portion = past_play_portion,
            static_play_portion = static_play_portion,
            custom_policy_ids = custom_policy_ids,
        )

        if pbt.complex_matchmaking:
            assert pbt.num_teams > 1
            assert pbt.num_current_policies > 1 or pbt.num_past_policies > 0

            min_policy_chunk_size = math.gcd(sim_batch_size, pbt.total_num_policies)

            if pbt.self_play_batch_size > 0:
                min_policy_chunk_size = min(min_policy_chunk_size,
                    pbt.self_play_batch_size // pbt.num_current_policies)

            if pbt.cross_play_batch_size > 0:
                min_policy_chunk_size = min(min_policy_chunk_size,
                    pbt.cross_play_batch_size // pbt.num_current_policies)

            if pbt.past_play_batch_size > 0:
                # FIXME: this doesn't make much sense, think more
                # about auto-picking policy batch size
                min_policy_chunk_size = min(min_policy_chunk_size,
                    pbt.past_play_batch_size // pbt.num_past_policies)

            if pbt.static_play_batch_size > 0:
                min_policy_chunk_size = min(min_policy_chunk_size,
                    pbt.static_play_batch_size // pbt.total_num_policies)

            assert min_policy_chunk_size > 0

            # Round to nearest power of 2
            policy_chunk_size = 1 << ((min_policy_chunk_size - 1).bit_length())
            policy_chunk_size = max(
                policy_chunk_size, min(64, sim_batch_size))
        else:
            assert num_past_policies == 0

            min_policy_chunk_size = 0
            policy_chunk_size = sim_batch_size // num_current_policies

        if policy_chunk_size_override != 0:
            policy_chunk_size = policy_chunk_size_override

        # Allocate enough policy sub-batches to evenly divide the full batch,
        # plus num_policies - 1 extras to handle worst case usage from unused
        # space in each subbatch
        num_policy_chunks = -(sim_batch_size // -policy_chunk_size)
        if pbt.complex_matchmaking:
            num_policy_chunks += pbt.total_num_policies - 1

        total_policy_batch_size = num_policy_chunks * policy_chunk_size

        return RolloutConfig(
            sim_batch_size = sim_batch_size,
            num_worlds = (sim_batch_size // (pbt.team_size * pbt.num_teams)),
            actions_cfg = actions_cfg,
            policy_chunk_size = policy_chunk_size,
            num_policy_chunks = num_policy_chunks,
            total_policy_batch_size = total_policy_batch_size,
            reward_gamma = reward_gamma,
            policy_dtype = policy_dtype,
            reward_dtype = reward_dtype,
            prob_dtype = prob_dtype,
            pbt = pbt,
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
    cfg: RolloutConfig = flax.struct.field(pytree_node=False)
    step_fn: Callable = flax.struct.field(pytree_node=False)
    load_ckpts_fn: Optional[Callable] = flax.struct.field(pytree_node=False)
    get_ckpts_fn: Optional[Callable] = flax.struct.field(pytree_node=False)
    sim_state: Any
    cur_obs: FrozenDict[str, Any]
    prng_key: random.PRNGKey
    rnn_states: Any
    reorder_state: PolicyBatchReorderState
    policy_assignments: jax.Array
    sim_ctrl: jax.Array
    env_returns: jax.Array

    @staticmethod
    def create(
        rollout_cfg,
        sim_fns,
        prng_key,
        rnn_states,
        init_sim_ctrl,
        static_play_assignments,
    ):
        if rollout_cfg.pbt.num_static_play_matches > 0.0:
            assert static_play_assignments != None
            assert (rollout_cfg.pbt.static_play_batch_size ==
                    static_play_assignments.shape[0])

        prng_key, assign_rnd = random.split(prng_key)
        policy_assignments = pbt_init_matchmaking(
            assign_rnd, rollout_cfg.pbt, static_play_assignments)
        assert policy_assignments.shape[0] == rollout_cfg.sim_batch_size

        reorder_state = _compute_reorder_state(policy_assignments, rollout_cfg)

        init_fn = sim_fns['init']
        step_fn = sim_fns['step']
        load_ckpts_fn = sim_fns.get('load_ckpts', None)
        get_ckpts_fn = sim_fns.get('get_ckpts', None)

        init_out = init_fn()
        init_out = frozen_dict.freeze(init_out)

        init_sim_state = init_out['state']
        init_obs = init_out['obs']

        init_env_returns = jnp.zeros(
            (rollout_cfg.sim_batch_size, 1),
            dtype=rollout_cfg.reward_dtype)

        return RolloutState(
            cfg = rollout_cfg,
            step_fn = step_fn,
            load_ckpts_fn = load_ckpts_fn,
            get_ckpts_fn = get_ckpts_fn,
            sim_state = init_sim_state,
            cur_obs = init_obs,
            prng_key = prng_key,
            rnn_states = rnn_states,
            reorder_state = reorder_state,
            policy_assignments = policy_assignments,
            sim_ctrl = init_sim_ctrl,
            env_returns = init_env_returns,
        )

    def update_matchmaking(
        self,
        self_play_portion: float,
        cross_play_portion: float,
        past_play_portion: float,
        static_play_portion: float,
        policy_assignments: jax.Array,
    ):
        new_pbt_cfg = PBTMatchmakeConfig.setup(
            self.cfg.pbt.num_current_policies,
            self.cfg.pbt.num_past_policies,
            self.cfg.pbt.num_teams,
            self.cfg.pbt.team_size,
            self.cfg.sim_batch_size,
            self_play_portion,
            cross_play_portion,
            past_play_portion,
            static_play_portion,
            self.cfg.pbt.custom_policy_ids,
        )

        new_cfg = dataclasses.replace(self.cfg, pbt=new_pbt_cfg)

        return self.replace(
            cfg = new_cfg,
            reorder_state = _compute_reorder_state(
                policy_assignments, new_cfg),
            policy_assignments = policy_assignments,
        )

    def update(
        self,
        sim_state=None,
        cur_obs=None,
        prng_key=None,
        rnn_states=None,
        reorder_state=None,
        policy_assignments=None,
        sim_ctrl=None,
        env_returns=None,
    ):
        return RolloutState(
            cfg = self.cfg,
            step_fn = self.step_fn,
            load_ckpts_fn = self.load_ckpts_fn,
            get_ckpts_fn = self.get_ckpts_fn,
            sim_state = (
                sim_state if sim_state is not None else self.sim_state),
            cur_obs = (
                cur_obs if cur_obs != None else self.cur_obs),
            prng_key = prng_key if prng_key != None else self.prng_key,
            rnn_states = rnn_states if rnn_states != None else self.rnn_states,
            reorder_state = (
                reorder_state if reorder_state != None else self.reorder_state),
            policy_assignments = (
                policy_assignments if policy_assignments != None else
                    self.policy_assignments),
            sim_ctrl = (
                sim_ctrl if sim_ctrl != None else
                    self.sim_ctrl),
            env_returns = (env_returns if env_returns != None else
                self.env_returns),
        )

    def get_current_checkpoints(self):
        return self.get_ckpts_fn()

    def load_checkpoints_into_sim(self, ckpts):
        assert ckpts.ndim == 2
        load_trigger = jnp.ones((ckpts.shape[0], 1), jnp.int32)
        new_obs = self.load_ckpts_fn(load_trigger, ckpts)
        new_obs = frozen_dict.freeze(new_obs)

        return self.update(cur_obs=new_obs)

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

    def flatten_time(self):
        flattened = jax.tree.map(lambda x: x.reshape(-1, 1, *x.shape[2:]), self.data)

        return self.replace(data=flattened)


class RolloutCollectState(flax.struct.PyTreeNode):
    store: FrozenDict[str, Any]
    obs_stats: FrozenDict[str, Any]
    env_returns_metric: Metric 

    @staticmethod
    def create(store_typed_shapes, init_obs_stats, num_train_policies):
        @partial(jax.vmap, in_axes=None, out_axes=0,
                 axis_size=num_train_policies)
        def expand_metric(x):
            return x

        return RolloutCollectState(
            store = jax.tree_map(
                lambda x: jnp.empty(x.shape, x.dtype), store_typed_shapes),
            obs_stats = init_obs_stats,
            env_returns_metric = expand_metric(Metric.init(True)),
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
        init_rollout_state: RolloutState,
        example_policy_states: PolicyState,
    ):
        cpu_dev = jax.devices('cpu')[0]

        self._cfg = init_rollout_state.cfg
        self._critic_outputs_distribution = (
            train_cfg.dreamer_v3_critic or train_cfg.hlgauss_critic)

        self._num_bptt_chunks = train_cfg.num_bptt_chunks
        assert train_cfg.steps_per_update % train_cfg.num_bptt_chunks == 0
        self._num_bptt_steps = (
            train_cfg.steps_per_update // train_cfg.num_bptt_chunks)

        self._num_train_policies = self._cfg.pbt.num_current_policies
        self._num_train_agents_per_policy = \
            _compute_num_train_agents_per_policy(self._cfg)

        self._num_train_seqs_per_policy = (
            self._num_train_agents_per_policy * self._num_bptt_chunks)

        def compute_sim_to_train_indices():
            return _compute_sim_to_train_indices(self._cfg)

        self._sim_to_train_idxs = jax.jit(compute_sim_to_train_indices)()
        assert (self._sim_to_train_idxs.shape[1] ==
                self._num_train_agents_per_policy)

        self._use_advantages = train_cfg.compute_advantages
        self._compute_advantages_fn = partial(compute_advantages, train_cfg)
        self._compute_returns_fn = partial(compute_returns, train_cfg)

        typed_shapes = {}

        def get_typed_shape(a):
            return TypedShape(a.shape, a.dtype)

        example_policy_state = jax.tree_map(
            lambda x: x[0], example_policy_states)

        # Preprocessed observations are stored in the rollouts. Figure
        # out observation shape / dtype after preprocessing
        def get_preprocessed_obs_abstract(policy_state, obs):
            return policy_state.obs_preprocess.preprocess(
                policy_state.obs_preprocess_state, obs, False)

        preprocessed_obs_abstract = jax.eval_shape(
            get_preprocessed_obs_abstract, example_policy_state,
            init_rollout_state.cur_obs)

        def get_actions_abstract(policy_state, rnn_states, preprocessed_obs):
            policy_out, rnn_states = policy_state.apply_fn(
                {
                    'params': policy_state.params,
                    'batch_stats': policy_state.batch_stats,
                },
                random.key(0),
                rnn_states,
                preprocessed_obs,
                train = False,
                method='rollout',
            )

            return policy_out['actions'], policy_out['log_probs']

        actions_abstract, log_probs_abstract = jax.eval_shape(
            get_actions_abstract, example_policy_state,
            init_rollout_state.rnn_states, preprocessed_obs_abstract)

        typed_shapes['obs'] = jax.tree_map(get_typed_shape, preprocessed_obs_abstract)
        typed_shapes['actions'] = jax.tree.map(get_typed_shape, actions_abstract)

        typed_shapes['log_probs'] = jax.tree.map(lambda a: TypedShape(a.shape, self._cfg.prob_dtype), log_probs_abstract)

        typed_shapes['rewards'] = TypedShape(
          (self._cfg.sim_batch_size, 1), self._cfg.reward_dtype)

        typed_shapes['dones'] = TypedShape(
            (self._cfg.sim_batch_size, 1), jnp.bool_)

        typed_shapes['values'] = TypedShape(
            (self._cfg.sim_batch_size, 1), self._cfg.reward_dtype)

        def expand_per_step_shapes(x):
            return TypedShape((
                    self._num_bptt_chunks,
                    self._num_bptt_steps,
                    self._num_train_policies,
                    self._num_train_agents_per_policy,
                    *x.shape[1:],
                ), dtype=x.dtype)

        typed_shapes = jax.tree_map(expand_per_step_shapes, typed_shapes)

        typed_shapes['rnn_start_states'] = jax.tree_map(
            lambda x: TypedShape((
                    self._num_bptt_chunks,
                    self._num_train_policies,
                    self._num_train_agents_per_policy,
                    *x.shape[1:],
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
            'Est Returns': Metric.init(True),
            'Env Returns': Metric.init(True),
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
        user_start_rollouts_hook: Callable,
        user_finish_rollouts_hook: Callable,
        user_metrics_hook: Callable,
    ):
        # This is the only mutable state used from train_state_mgr here.
        # Copied back into train_state_mgr at end of function.
        policy_states = train_state_mgr.policy_states
        user_state = train_state_mgr.user_state

        rollout_state, user_state = user_start_rollouts_hook(
            rollout_state, user_state)
        
        obs_preprocess = policy_states.obs_preprocess
        obs_preprocess_train_state = jax.tree_map(
            lambda s: s[0:self._num_train_policies],
            policy_states.obs_preprocess_state)

        def iter_bptt_chunk(bptt_chunk, inputs):
            rollout_state, policy_states, collect_state = inputs

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

            rollout_state, policy_states, collect_state = rollout_loop(
                rollout_state, policy_states,
                self._num_bptt_steps,  post_inference_cb, post_step_cb,
                collect_state, sample_actions = True, return_debug = False)

            return rollout_state, policy_states, collect_state

        collect_state = RolloutCollectState.create(
            self._store_typed_shape_tree,
            obs_preprocess.init_obs_stats(obs_preprocess_train_state, True),
            self._num_train_policies,
        )

        rollout_state, policy_states, collect_state = lax.fori_loop(
            0, self._num_bptt_chunks, iter_bptt_chunk,
            (rollout_state, policy_states, collect_state))

        metrics = metrics.update_metrics({
            'Env Returns': collect_state.env_returns_metric,
        })

        with profile("Bootstrap Values"):
            bootstrap_values = self._bootstrap_values(
                policy_states, train_state_mgr.train_states,
                rollout_state)

        with profile("Finalize Rollouts"):
            rollout_data, metrics, user_state = self._finalize_rollouts(
                train_state_mgr.train_states, collect_state.store,
                bootstrap_values, metrics, user_state,
                user_finish_rollouts_hook, user_metrics_hook)

        train_state_mgr = train_state_mgr.replace(
            policy_states = policy_states,
            user_state = user_state,
        )

        return (train_state_mgr, rollout_state, rollout_data,
                collect_state.obs_stats, metrics)

    def _sim_to_train(self, data, reorder_state):
        if self._cfg.pbt.complex_matchmaking:
            def to_train(x):
                return x[self._sim_to_train_idxs]
        else:
            def to_train(x):
                return x.reshape(self._num_train_policies, -1, *x.shape[1:])

        return jax.tree_map(to_train, data)

    def _policy_to_train(self, data, reorder_state):
        if not self._cfg.pbt.complex_matchmaking:
            # Already in train ordering!
            return data
        
        def to_train(x):
            # FIXME
            sim_ordering = reorder_state.to_sim(x)
            return sim_ordering[self._sim_to_train_idxs]

        return jax.tree_map(to_train, data)

    def _compute_value_estimate(self, critic_out):
        if not self._critic_outputs_distribution:
            return critic_out
        else:
            return critic_out.mean()

    def _bootstrap_values(self, policy_states, train_states, rollout_state):
        rnn_states = rollout_state.rnn_states
        obs = rollout_state.cur_obs
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

            return self._compute_value_estimate(policy_out['critic'])

        return critic_fn(policy_states, rnn_states, obs)

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
            values = self._compute_value_estimate(policy_out['critic'])
            values = self._policy_to_train(values, reorder_state)

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
        rollout_state: RolloutState,
        dones: jax.Array,
        rewards: jax.Array,
        episode_results: jax.Array,
        collect_state: RolloutCollectState,
    ):
        with profile('Post Step Rollout Store'):
            @jax.vmap
            def compute_env_returns_metrics(env_returns, dones):
                return Metric.init_from_data_masked(
                    per_policy=True,
                    data=env_returns,
                    mask=dones,
                )

            new_env_returns_metrics = compute_env_returns_metrics(
                self._sim_to_train(
                    rollout_state.env_returns, rollout_state.reorder_state),
                self._sim_to_train(dones, rollout_state.reorder_state))

            collect_state = collect_state.replace(env_returns_metric =
                collect_state.env_returns_metric.merge(new_env_returns_metrics))

            save_data = self._sim_to_train({
                'dones': dones,
                'rewards': rewards,
            }, rollout_state.reorder_state)
            return rollout_state, collect_state.save(
                (bptt_chunk, bptt_step), save_data)

    def _finalize_rollouts(
        self,
        train_states,
        rollouts,
        bootstrap_values,
        metrics,
        user_state,
        user_finish_rollouts_hook,
        user_metrics_hook
    ):
        if train_states.value_normalizer == None:
            unnormalized_values = rollouts['values']
            unnormalized_bootstrap_values = bootstrap_values
        else:
            def invert_value_norm(train_state, v):
                return train_state.value_normalizer.invert(
                    train_state.value_normalizer_state, v)

            unnormalized_values = jax.vmap(invert_value_norm,
                in_axes=(0, 2), out_axes=2)(train_states, rollouts['values'])

            unnormalized_bootstrap_values = jax.vmap(invert_value_norm)(
                train_states, bootstrap_values)

            assert unnormalized_values.dtype == self._cfg.reward_dtype
            assert unnormalized_bootstrap_values.dtype == self._cfg.reward_dtype

        rollouts, user_state = user_finish_rollouts_hook(
            rollouts, bootstrap_values, unnormalized_values,
            unnormalized_bootstrap_values, user_state)

        modified_rewards = rollouts['rewards']

        modify_rewards_based_on_filter_matches = False

        if modify_rewards_based_on_filter_matches:
            filters_state = rollouts['obs']['filters_state']
            has_filters_active = jnp.any(
                filters_state == 1.0, axis=1, keepdims=True)

            bonus_rewards = jnp.where(has_filters_active, 5.0, 0.0)
            modified_rewards = modified_rewards + bonus_rewards

            rollouts = rollouts.copy({'rewards': modified_rewards})

        if self._use_advantages:
            advantages = self._compute_advantages_fn(
                rollouts['rewards'],
                unnormalized_values,
                rollouts['dones'],
                unnormalized_bootstrap_values,
            )

            returns = advantages + unnormalized_values

            rollouts = rollouts.copy({
                'advantages': advantages.astype(self._cfg.prob_dtype),
            })

        else:
            returns = self._compute_returns_fn(
                rollouts['rewards'],
                rollouts['dones'],
                unnormalized_bootstrap_values,
            )

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
            'Values': reorder_seq_data(unnormalized_values),
            'Est Returns': rollouts['returns'],
            'Bootstrap Values': unnormalized_bootstrap_values,
        })

        if self._use_advantages:
            metrics = metrics.record({
                'Advantages': rollouts['advantages'],
            })

        metrics = user_metrics_hook(metrics, rollouts, user_state)

        return RolloutData(
            data = rollouts.copy({
                'rnn_start_states': rnn_start_states,
            }),
            num_train_seqs_per_policy = self._num_train_seqs_per_policy,
            num_train_policies = self._num_train_policies,
        ), metrics, user_state


def rollout_loop(
    rollout_state: RolloutState,
    policy_states: PolicyState,
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
        if not rollout_state.cfg.pbt.complex_matchmaking:
            return states 
        else:
            # FIXME: more efficient way to get this?
            state_idxs = reorder_state.to_policy(assignments)[:, 0]
            return jax.tree_map(lambda x: x[state_idxs], states)

    def rollout_iter(step_idx, iter_state):
        rollout_state, policy_states, cb_state = iter_state

        prng_key = rollout_state.prng_key
        rnn_states = rollout_state.rnn_states
        sim_state = rollout_state.sim_state
        sim_obs = rollout_state.cur_obs
        reorder_state = rollout_state.reorder_state
        policy_assignments = rollout_state.policy_assignments

        with profile('Policy Inference'):
            prng_key, step_key = random.split(prng_key)
            step_keys = random.split(
                step_key, rollout_state.cfg.num_policy_chunks)

            reordered_policy_states = reorder_policy_states(
                policy_states, policy_assignments, reorder_state)

            rnn_states, policy_obs = reorder_state.to_policy(
                (rnn_states, sim_obs))

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

            step_input = frozen_dict.freeze({
                'state': sim_state,
                'actions': reorder_state.to_sim(policy_out['actions']),
                'resets': jnp.zeros((rollout_state.cfg.num_worlds, 1), dtype=jnp.int32),
                'sim_ctrl': rollout_state.sim_ctrl,
            })

            pbt_inputs = FrozenDict({})

            pbt_inputs = pbt_inputs.copy({
                'policy_assignments': policy_assignments,
                #'world_curriculum': jnp.ones(
                #    (rollout_state.cfg.num_worlds, 1), dtype=jnp.int32),
            })

            if policy_states.reward_hyper_params != None:
                pbt_inputs = pbt_inputs.copy({
                    'reward_hyper_params': (
                        policy_states.reward_hyper_params),
                })

            if len(pbt_inputs) > 0:
                step_input = step_input.copy({'pbt': pbt_inputs})

            step_output = rollout_state.step_fn(step_input)
            step_output = frozen_dict.freeze(step_output)

            sim_state = step_output['state']
            dones = step_output['dones'].astype(jnp.bool_)
            rewards = step_output['rewards'].astype(
                rollout_state.cfg.reward_dtype)
            sim_obs = step_output['obs']

            env_returns = (rewards + rollout_state.cfg.reward_gamma *
                rollout_state.env_returns)

            # rnn_states kept in sim ordering
            rnn_states = rnn_reset_fn(rnn_states, dones)

            try:
                episode_results = step_output['pbt']['episode_results']
            except KeyError:
                episode_results = None


            policy_assignments, prng_key = pbt_update_matchmaking(
                policy_assignments, policy_states, dones, episode_results,
                prng_key, rollout_state.cfg.pbt)

            reorder_state = _compute_reorder_state(
                policy_assignments, rollout_state.cfg)

            rollout_state = rollout_state.update(
                prng_key = prng_key,
                rnn_states = rnn_states,
                sim_state = sim_state,
                cur_obs = sim_obs,
                reorder_state = reorder_state,
                policy_assignments = policy_assignments,
                env_returns = env_returns,
            )

            rollout_state, cb_state = post_step_cb(
                step_idx, rollout_state, dones, rewards,
                episode_results, cb_state)

            rollout_state = rollout_state.update(
                env_returns = jnp.where(dones, 0, rollout_state.env_returns),
            )

        return rollout_state, policy_states, cb_state

    return lax.fori_loop(0, num_steps, rollout_iter,
                         (rollout_state, policy_states, cb_state))


def rollouts_reset(
    rollout_state: RolloutState,
    policy_states: PolicyState,
):
    def gen_zero_action(action_cfg):
        if isinstance(action_cfg, DiscreteActionsConfig):
            return jnp.zeros(
                (rollout_state.cfg.sim_batch_size,
                    len(action_cfg.actions_num_buckets)),
                dtype=jnp.int32)
        elif isinstance(action_cfg, ContinuousActionsConfig):
            return jnp.zeros(
                (rollout_state.cfg.sim_batch_size,
                    1,
                    action_cfg.num_dims),
                dtype=jnp.float32)
        else:
            assert False

    step_input = frozen_dict.freeze({
        'state': rollout_state.sim_state,
        'actions': jax.tree.map(gen_zero_action, rollout_state.cfg.actions_cfg),
        'resets': jnp.ones(
            (rollout_state.cfg.sim_batch_size //
                (rollout_state.cfg.pbt.team_size *
                 rollout_state.cfg.pbt.num_teams), 1), 
            dtype=jnp.int32),
        'sim_ctrl': rollout_state.sim_ctrl,
    })

    pbt_inputs = FrozenDict({})

    pbt_inputs = pbt_inputs.copy({
        'policy_assignments': jnp.zeros(
            (rollout_state.cfg.sim_batch_size, 1), dtype=jnp.int32),
        #'world_curriculum': jnp.ones(
        #    (rollout_state.cfg.num_worlds, 1), dtype=jnp.int32),
    })

    if policy_states.reward_hyper_params != None:
        pbt_inputs = pbt_inputs.copy({
            'reward_hyper_params': (
                policy_states.reward_hyper_params),
        })

    if len(pbt_inputs) > 0:
        step_input = step_input.copy({'pbt': pbt_inputs})

    step_output = rollout_state.step_fn(step_input)
    step_output = frozen_dict.freeze(step_output)

    sim_state = step_output['state']
    dones = step_output['dones'].astype(jnp.bool_)
    rewards = step_output['rewards'].astype(rollout_state.cfg.reward_dtype)
    sim_obs = step_output['obs']

    env_returns = jnp.zeros_like(rollout_state.env_returns)

    rnn_states = policy_states.rnn_reset_fn(
        rollout_state.rnn_states, jnp.ones_like(dones))

    rollout_state = rollout_state.update(
        rnn_states = rnn_states,
        sim_state = sim_state,
        cur_obs = sim_obs,
        reorder_state = rollout_state.reorder_state,
        policy_assignments = rollout_state.policy_assignments,
        env_returns = env_returns,
    )

    return rollout_state


def _compute_num_train_agents_per_policy(rollout_cfg):
    assert rollout_cfg.pbt.cross_play_batch_size % rollout_cfg.pbt.num_teams == 0
    assert rollout_cfg.pbt.past_play_batch_size % rollout_cfg.pbt.num_teams == 0

    # FIXME: currently we only collect training data for team 1 in each
    # world in cross-play and past-play to avoid having variable amounts
    # of training data in each step
    total_num_train_agents = (
        rollout_cfg.pbt.self_play_batch_size +
        rollout_cfg.pbt.cross_play_batch_size // rollout_cfg.pbt.num_teams +
        rollout_cfg.pbt.past_play_batch_size // rollout_cfg.pbt.num_teams
    )

    assert total_num_train_agents % rollout_cfg.pbt.num_current_policies == 0
    return total_num_train_agents // rollout_cfg.pbt.num_current_policies


def _compute_sim_to_train_indices(rollout_cfg):
    global_indices = jnp.arange(rollout_cfg.sim_batch_size)

    def setup_match_indices(start, stop):
        return global_indices[start:stop].reshape(
            rollout_cfg.pbt.num_current_policies, -1,
            rollout_cfg.pbt.num_teams, rollout_cfg.pbt.team_size)

    self_play_indices = setup_match_indices(
        0, rollout_cfg.pbt.self_play_batch_size)

    cross_play_indices = setup_match_indices(
        rollout_cfg.pbt.self_play_batch_size,
        rollout_cfg.pbt.self_play_batch_size +
            rollout_cfg.pbt.cross_play_batch_size)

    past_play_indices = setup_match_indices(
        rollout_cfg.pbt.self_play_batch_size +
            rollout_cfg.pbt.cross_play_batch_size,
        rollout_cfg.pbt.self_play_batch_size +
            rollout_cfg.pbt.cross_play_batch_size +
            rollout_cfg.pbt.past_play_batch_size)

    self_play_gather = self_play_indices.reshape(
        rollout_cfg.pbt.num_current_policies, -1)

    cross_play_gather = cross_play_indices[:, :, 0, :].reshape(
        rollout_cfg.pbt.num_current_policies, -1)

    past_play_gather = past_play_indices[:, :, 0, :].reshape(
        rollout_cfg.pbt.num_current_policies, -1)

    return jnp.concatenate(
        [self_play_gather, cross_play_gather, past_play_gather], axis=1)


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
    if rollout_cfg.pbt.complex_matchmaking:
        to_policy_idxs, to_sim_idxs = _compute_reorder_chunks(
            assignments, rollout_cfg.pbt.total_num_policies, 
            rollout_cfg.policy_chunk_size, rollout_cfg.num_policy_chunks)
    else:
        to_policy_idxs = None
        to_sim_idxs = None

    return PolicyBatchReorderState(
        to_policy_idxs = to_policy_idxs,
        to_sim_idxs = to_sim_idxs,
        policy_dims = (
            rollout_cfg.pbt.total_num_policies, rollout_cfg.policy_chunk_size),
        sim_dims = (rollout_cfg.sim_batch_size,),
    )
