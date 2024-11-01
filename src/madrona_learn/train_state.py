import jax
from jax import lax, random, numpy as jnp
from jax.experimental import checkify
import numpy as np
import flax
from flax import linen as nn
from flax.core import FrozenDict, frozen_dict
from flax.training.dynamic_scale import DynamicScale
import flax.training.train_state
from flax.training import orbax_utils
import optax
import orbax.checkpoint

from dataclasses import dataclass
from typing import Optional, Any, Callable
from functools import partial

from .actor_critic import ActorCritic
from .algo_common import HyperParams, AlgoBase
from .cfg import TrainConfig
from .moving_avg import EMANormalizer
from .observations import ObservationsPreprocess, ObservationsPreprocessNoop
from .policy import Policy

class MovingEpisodeScore(flax.struct.PyTreeNode):
    mean: jax.Array
    var: jax.Array
    N: jax.Array


class MMR(flax.struct.PyTreeNode):
    elo: jax.Array


class PolicyState(flax.struct.PyTreeNode):
    apply_fn: Callable = flax.struct.field(pytree_node=False)
    rnn_reset_fn: Callable = flax.struct.field(pytree_node=False)

    params: FrozenDict[str, Any]
    batch_stats: FrozenDict[str, Any]

    obs_preprocess: ObservationsPreprocess = flax.struct.field(
        pytree_node=False)
    obs_preprocess_state: FrozenDict[str, Any]

    reward_hyper_params: jax.Array

    get_episode_scores_fn: Callable = flax.struct.field(pytree_node=False)
    episode_score: Optional[MovingEpisodeScore]
    mmr: Optional[MMR]

    def update(
        self,
        params = None,
        batch_stats = None,
        obs_preprocess_state = None,
        reward_hyper_params = None,
        episode_score = None,
        mmr = None,
    ):
        return PolicyState(
            apply_fn = self.apply_fn,
            rnn_reset_fn = self.rnn_reset_fn,
            params = params if params != None else self.params,
            batch_stats = (
                batch_stats if batch_stats != None else self.batch_stats
            ),
            obs_preprocess = self.obs_preprocess,
            obs_preprocess_state = (
                obs_preprocess_state if obs_preprocess_state != None else
                    self.obs_preprocess_state
            ),
            reward_hyper_params = (
                reward_hyper_params if reward_hyper_params != None else
                    self.reward_hyper_params
            ),
            get_episode_scores_fn = self.get_episode_scores_fn,
            episode_score = (
                episode_score if episode_score != None else
                    self.episode_score
            ),
            mmr = mmr if mmr != None else self.mmr,
        )


class PolicyTrainState(flax.struct.PyTreeNode):
    value_normalizer: Optional[EMANormalizer] = flax.struct.field(pytree_node=False)
    tx: optax.GradientTransformation = flax.struct.field(pytree_node=False)
    value_normalizer_state: Optional[FrozenDict[str, Any]]
    hyper_params: HyperParams
    opt_state: optax.OptState
    scheduler: Optional[optax.Schedule]
    scaler: Optional[DynamicScale]
    update_prng_key: jax.Array

    def update(
        self,
        tx=None,
        value_normalizer_state=None,
        hyper_params=None,
        opt_state=None,
        scheduler=None,
        scaler=None,
        update_prng_key=None,
    ):
        return PolicyTrainState(
            value_normalizer = self.value_normalizer,
            tx = tx if tx != None else self.tx,
            value_normalizer_state = (
                value_normalizer_state if value_normalizer_state != None else
                    self.value_normalizer_state
            ),
            hyper_params = (
                hyper_params if hyper_params != None else self.hyper_params
            ),
            opt_state = opt_state if opt_state != None else self.opt_state,
            scheduler = scheduler if scheduler != None else self.scheduler,
            scaler = scaler if scaler != None else self.scaler,
            update_prng_key = (
                update_prng_key if update_prng_key != None else 
                    self.update_prng_key
            ),
        )

    def gen_update_rnd(self):
        rnd, next_key = random.split(self.update_prng_key)
        return rnd, self.update(update_prng_key=next_key)


class TrainStateManager(flax.struct.PyTreeNode):
    policy_states: PolicyState
    train_states: PolicyTrainState
    pbt_rng: jax.Array
    user_state: Any

    def save(self, next_update, path):
        def prepare_for_ckpt(x):
            if jnp.issubdtype(x.dtype, jax.dtypes.prng_key):
                x = random.key_data(x)

            return np.asarray(x)

        prepared = jax.tree.map(prepare_for_ckpt, jax.device_get(self))

        ckpt = {
            'next_update': next_update,
            'policy_states': prepared.policy_states,
            'train_states': prepared.train_states,
            'pbt_rng': prepared.pbt_rng,
            'user_state': prepared.user_state,
        }

        checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(ckpt)
        checkpointer.save(path, ckpt, save_args=save_args)

    def load(self, path):
        checkpointer = orbax.checkpoint.PyTreeCheckpointer()

        restore_desc = {
            'next_update': 0,
            'policy_states': self.policy_states,
            'train_states': self.train_states,
            'pbt_rng': self.pbt_rng,
            'user_state': self.user_state,
        }
        
        loaded = checkpointer.restore(path, item=restore_desc)

        def to_jax_and_dtype(a, b):
            if jnp.issubdtype(b.dtype, jax.dtypes.prng_key):
                return jax.random.wrap_key_data(a)
            elif isinstance(a, np.ndarray) or isinstance(a, jax.Array):
                return jnp.asarray(a, dtype=b.dtype)
            else:
                return a

        return self.replace(
            policy_states = jax.tree.map(to_jax_and_dtype,
                loaded['policy_states'], self.policy_states),
            train_states = jax.tree.map(to_jax_and_dtype,
                loaded['train_states'], self.train_states),
            pbt_rng = jax.tree.map(to_jax_and_dtype,
                loaded['pbt_rng'], self.pbt_rng),
            user_state = jax.tree.map(to_jax_and_dtype,
                loaded['user_state'], self.user_state),
        ), loaded['next_update']

    @staticmethod
    def slice_checkpoint(src, dst, train_select, past_select):
        checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        loaded = checkpointer.restore(src)

        train_states = jax.tree.map(
            lambda x: x[train_select], loaded['train_states'])

        train_policy_states = jax.tree.map(
            lambda x: x[train_select], loaded['policy_states'])

        past_policy_states = jax.tree.map(
            lambda x: x[past_select], loaded['policy_states'])

        policy_states = jax.tree.map(
            lambda x, y: np.concatenate([x, y], axis=0),
            train_policy_states, past_policy_states)

        ckpt = {
            'next_update': loaded['next_update'],
            'policy_states': policy_states,
            'train_states': train_states,
            'pbt_rng': loaded['pbt_rng'],
            'user_state': loaded['user_state'],
        }

        save_args = orbax_utils.save_args_from_target(ckpt)
        checkpointer.save(dst, ckpt, save_args=save_args)

    @staticmethod
    def load_policies(policy, path):
        checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        loaded = checkpointer.restore(path)

        actor_critic = policy.actor_critic
        obs_preprocess = (
            policy.obs_preprocess or ObservationsPreprocessNoop.create())

        def to_jax(a):
            if isinstance(a, np.ndarray):
                return jnp.asarray(a)
            else:
                return a

        num_train_policies = loaded['train_states']['update_prng_key'].shape[0]

        reward_hyper_params = to_jax(
            loaded['policy_states']['reward_hyper_params'])

        if policy.get_episode_scores != None:
            get_episode_scores_fn = policy.get_episode_scores
        else:
            get_episode_scores_fn = lambda x: 0.0

        episode_score = to_jax(loaded['policy_states']['episode_score'])
        mmr = to_jax(loaded['policy_states']['mmr'])

        if episode_score != None:
            episode_score = MovingEpisodeScore(**episode_score)
            total_num_policies = episode_score.mean.shape[0]

        if mmr != None:
            mmr = MMR(**mmr)
            total_num_policies = mmr.elo.shape[0]

        return PolicyState(
            apply_fn = actor_critic.apply,
            rnn_reset_fn = actor_critic.clear_recurrent_state,
            params = jax.tree.map(to_jax, loaded['policy_states']['params']),
            batch_stats = jax.tree.map(to_jax, loaded['policy_states']['batch_stats']),
            obs_preprocess = obs_preprocess,
            obs_preprocess_state = frozen_dict.freeze(
                jax.tree.map(to_jax, loaded['policy_states']['obs_preprocess_state'])),
            reward_hyper_params = reward_hyper_params,
            get_episode_scores_fn = get_episode_scores_fn,
            episode_score = jax.tree.map(to_jax, episode_score),
            mmr = jax.tree.map(to_jax, mmr),
        ), num_train_policies, total_num_policies

    @staticmethod
    def create(
        policy: Policy,
        cfg: TrainConfig,
        algo: AlgoBase,
        init_user_state_cb: Callable,
        base_rng,
        example_obs,
        example_rnn_states,
        use_competitive_mmr,
        checkify_errors,
    ):
        base_init_rng, pbt_rng = random.split(base_rng)

        def make_policies(rnd, obs, rnn_states):
            return _make_policies(policy, cfg, algo, rnd, obs,
                rnn_states, use_competitive_mmr)

        make_policies = jax.jit(checkify.checkify(
            make_policies, errors=checkify_errors))

        err, (policy_states, train_states) = make_policies(
            base_init_rng, example_obs, example_rnn_states)
        err.throw()

        return TrainStateManager(
            policy_states = policy_states,
            train_states = train_states,
            pbt_rng = pbt_rng,
            user_state = init_user_state_cb(),
        )


def _setup_value_normalizer(hyper_params, fake_values):
    value_normalizer = EMANormalizer(
        decay = hyper_params.value_normalizer_decay,
        norm_dtype = fake_values.dtype,
        inv_dtype = jnp.float32,
        disable = not hyper_params.normalize_values,
    )

    value_normalizer_state = value_normalizer.init_estimates(fake_values)
    return value_normalizer, value_normalizer_state

def _setup_policy_state(
    policy,
    cfg,
    use_competitive_mmr,
    prng_key,
    rnn_states,
    obs,
):
    actor_critic = policy.actor_critic
    obs_preprocess = (
        policy.obs_preprocess or ObservationsPreprocessNoop.create())

    obs_preprocess_state = obs_preprocess.init_state(obs, False)
    preprocessed_obs = obs_preprocess.preprocess(
        obs_preprocess_state, obs, False)

    # The second prng key is passed as the key for sampling
    (fake_outs, rnn_states), variables = actor_critic.init_with_output(
        prng_key, random.PRNGKey(0), rnn_states, preprocessed_obs,
        method='rollout')

    params = variables['params']
    batch_stats = variables.get('batch_stats', {})

    num_reward_hyperparams = 0
    if cfg.pbt:
        num_reward_hyperparams = len(cfg.pbt.reward_hyper_params_explore)

    reward_hyper_params = jnp.zeros((num_reward_hyperparams,), jnp.float32)

    if not policy.get_episode_scores:
        get_episode_scores_fn = lambda x: 0.0
    else:
        get_episode_scores_fn = policy.get_episode_scores

    if use_competitive_mmr:
        mmr = MMR(elo = jnp.array(1500, jnp.float32))
        episode_score = None
    else:
        mmr = None
        episode_score = MovingEpisodeScore(
            mean = jnp.array(0, jnp.float32),
            var = jnp.array(0, jnp.float32),
            N = jnp.array(0, jnp.int32),
        )

    return PolicyState(
        apply_fn = actor_critic.apply,
        rnn_reset_fn = actor_critic.clear_recurrent_state,
        params = params,
        batch_stats = batch_stats,
        obs_preprocess = obs_preprocess,
        obs_preprocess_state = obs_preprocess_state,
        reward_hyper_params = reward_hyper_params,
        get_episode_scores_fn = get_episode_scores_fn,
        episode_score = episode_score,
        mmr = mmr,
    ), fake_outs, rnn_states

def _setup_train_state(
    cfg,
    algo,
    prng_key,
    policy_state,
    fake_policy_out,
):
    hyper_params = algo.init_hyperparams(cfg)
    optimizer = algo.make_optimizer(hyper_params)

    if cfg.normalize_values:
        assert fake_policy_out['critic'].shape[-1] == 1

        value_norm, value_norm_state = _setup_value_normalizer(
            hyper_params, fake_policy_out['critic'])
    else:
        value_norm = None
        value_norm_state = None

    opt_state = optimizer.init(policy_state.params)

    if cfg.compute_dtype == jnp.float16:
        scaler = DynamicScale()
    else:
        scaler = None

    return PolicyTrainState(
        value_normalizer = value_norm,
        tx = optimizer,
        value_normalizer_state = value_norm_state,
        hyper_params = hyper_params,
        opt_state = opt_state,
        scheduler = None,
        scaler = scaler,
        update_prng_key = prng_key,
    )

def _make_policies(
    policy,
    cfg,
    algo,
    base_init_rnd,
    example_obs,
    example_rnn_states,
    use_competitive_mmr,
):
    setup_policy_state = partial(
        _setup_policy_state, policy, cfg, use_competitive_mmr)
    setup_policy_states = jax.vmap(setup_policy_state)

    if cfg.pbt != None:
        num_make = cfg.pbt.num_train_policies
        num_past_copies = cfg.pbt.num_past_policies
    else:
        num_make = 1
        num_past_copies = 0

    obs = jax.tree_map(
        lambda x: x[:num_make, None, ...], example_obs)

    rnn_states = jax.tree_map(
        lambda x: x[:num_make, None, ...], example_rnn_states)

    policy_init_base_rnd, train_init_base_rnd = random.split(base_init_rnd)

    policy_init_rnds = random.split(policy_init_base_rnd, num_make)

    policy_states, fake_policy_outs, rnn_states = setup_policy_states(
        policy_init_rnds, rnn_states, obs)

    setup_train_state = partial(_setup_train_state, cfg, algo)
    setup_train_states = jax.vmap(setup_train_state)
    
    train_init_rnds = random.split(train_init_base_rnd, num_make)
    train_states = setup_train_states(
        train_init_rnds, policy_states, fake_policy_outs)

    if num_past_copies > 0:
        num_repeats = -(num_past_copies // -num_make)

        policy_states = jax.tree_map(
            lambda x: jnp.tile(
                x, (num_repeats + 1, *([1] * (len(x.shape) - 1))))[
                    0:num_make + num_past_copies],
            policy_states)

    return policy_states, train_states
