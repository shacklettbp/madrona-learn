import jax
from jax import lax, random, numpy as jnp
from jax.experimental import checkify
import flax
from flax import linen as nn
from flax.core import FrozenDict
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
from .observations import ObservationsPreprocess

class PolicyState(flax.struct.PyTreeNode):
    apply_fn: Callable = flax.struct.field(pytree_node=False)
    rnn_reset_fn: Callable = flax.struct.field(pytree_node=False)
    params: FrozenDict[str, Any]
    batch_stats: FrozenDict[str, Any]
    obs_preprocess: ObservationsPreprocess = flax.struct.field(
        pytree_node=False)
    obs_preprocess_state: FrozenDict[str, Any]

    def update(
        self,
        params = None,
        batch_stats = None,
        obs_preprocess_state = None,
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
        )


class PolicyTrainState(flax.struct.PyTreeNode):
    value_normalizer: EMANormalizer = flax.struct.field(pytree_node=False)
    tx: optax.GradientTransformation = flax.struct.field(pytree_node=False)
    value_normalizer_state: FrozenDict[str, Any]
    hyper_params: HyperParams
    opt_state: optax.OptState
    scheduler: Optional[optax.Schedule]
    scaler: Optional[DynamicScale]
    update_prng_key: random.PRNGKey

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


class ObsPreprocessNoop:
    def init(self, obs):
        return ObservationsPreprocess(
            preprocessors = jax.tree_map(lambda o: self, obs),
            init_state_fn = lambda _, ob: jnp.array(()),
            update_state_fn = lambda _, state, stats: state,
            init_obs_stats_fn = lambda _, stats: jnp.array(()),
            update_obs_stats_fn = \
                lambda _, state, stats, prev_updates, o: stats,
            preprocess_fn = lambda _, state, ob: ob,
        )


class TrainStateManager(flax.struct.PyTreeNode):
    policy_states: PolicyState
    train_states: PolicyTrainState
    pbt_rng: random.PRNGKey

    def save(self, update_idx, path):
        ckpt = {
            'next_update': update_idx + 1,
            'policy_states': self.policy_states,
            'train_states': self.train_states,
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
        }
        
        loaded = checkpointer.restore(path, item=restore_desc)

        return TrainStateManager(
            policy_states = loaded['policy_states'],
            train_states = loaded['train_states'],
        ), loaded['next_update']

    @staticmethod
    def load_policies(policy, obs_preprocess, path):
        checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        loaded = checkpointer.restore(path)

        obs_preprocess = obs_preprocess or ObsPreprocessNoop()

        return PolicyState(
            apply_fn = policy.apply,
            rnn_reset_fn = policy.clear_recurrent_state,
            params = loaded['policy_states']['params'],
            batch_stats = loaded['policy_states']['batch_stats'],
            obs_preprocess = obs_preprocess,
            obs_preprocess_state = (
                loaded['policy_states']['obs_preprocess_state']),
        )

    @staticmethod
    def create(
        policy: nn.Module,
        obs_preprocess: Optional[nn.Module],
        cfg: TrainConfig,
        algo: AlgoBase,
        base_rng,
        example_obs,
        example_rnn_states,
        checkify_errors,
    ):
        base_init_rng, pbt_rng = random.split(base_rng)

        obs_preprocess_builder = obs_preprocess or ObsPreprocessNoop()
        obs_preprocess = obs_preprocess_builder.init(example_obs)

        def make_policies(rnd, obs, rnn_states):
            return _make_policies(
                policy, obs_preprocess, cfg, algo, rnd, obs, rnn_states)

        make_policies = jax.jit(checkify.checkify(
            make_policies, errors=checkify_errors))

        err, (policy_states, train_states) = make_policies(
            base_init_rng, example_obs, example_rnn_states)
        err.throw()

        return TrainStateManager(
            policy_states = policy_states,
            train_states = train_states,
            pbt_rng = pbt_rng,
        )


def _setup_value_normalizer(hyper_params, fake_values):
    value_norm_decay = (hyper_params.value_normalizer_decay 
                        if hyper_params.normalize_values else 1.0)

    value_normalizer = EMANormalizer(
        decay = value_norm_decay,
        out_dtype = fake_values.dtype,
        disable = not hyper_params.normalize_values,
    )

    value_normalizer_state = value_normalizer.init_estimates(fake_values)
    return value_normalizer, value_normalizer_state

def _setup_policy_state(
    policy,
    obs_preprocess,
    prng_key,
    rnn_states,
    obs,
):
    obs_preprocess_state = obs_preprocess.init_state(obs)
    preprocessed_obs = obs_preprocess.preprocess(
        obs_preprocess_state, obs)

    # The second prng key is passed as the key for sampling
    (fake_outs, rnn_states), variables = policy.init_with_output(
        prng_key, random.PRNGKey(0), rnn_states, preprocessed_obs,
        method='rollout')

    params = variables['params']
    batch_stats = variables.get('batch_stats', {})

    return PolicyState(
        apply_fn = policy.apply,
        rnn_reset_fn = policy.clear_recurrent_state,
        params = params,
        batch_stats = batch_stats,
        obs_preprocess = obs_preprocess,
        obs_preprocess_state = obs_preprocess_state,
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

    value_norm, value_norm_state = _setup_value_normalizer(
        hyper_params, fake_policy_out['values'])

    opt_state = optimizer.init(policy_state.params)

    if cfg.mixed_precision:
        scaler = DynamicScale()
    else:
        scaler = None

    return PolicyTrainState(
        value_normalizer = value_norm_fn,
        tx = optimizer,
        value_normalizer_state = value_norm_stats,
        hyper_params = hyper_params,
        opt_state = opt_state,
        scheduler = None,
        scaler = scaler,
        update_prng_key = prng_key,
    )

def _make_policies(
    policy,
    obs_preprocess,
    cfg,
    algo,
    base_init_rnd,
    example_obs,
    example_rnn_states,
):
    setup_policy_state = partial(
        _setup_policy_state, policy, obs_preprocess)
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
            lambda x: jnp.tile(x, (num_repeats, *([1] * (len(x.shape) - 1))))[
                0:num_make + num_past_copies],
            policy_states)

    return policy_states, train_states
