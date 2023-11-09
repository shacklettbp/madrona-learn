import jax
from jax import lax, random, numpy as jnp
from jax.experimental import checkify
import flax
from flax import linen as nn
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

class PolicyState(flax.struct.PyTreeNode):
    apply_fn: Callable = flax.struct.field(pytree_node=False)
    rnn_reset_fn: Callable = flax.struct.field(pytree_node=False)
    params: flax.core.FrozenDict[str, Any]
    batch_stats: flax.core.FrozenDict[str, Any]

    def update(
        self,
        params = None,
        batch_stats = None,
    ):
        return PolicyState(
            apply_fn = self.apply_fn,
            rnn_reset_fn = self.rnn_reset_fn,
            params = params if params != None else self.params,
            batch_stats = (
                batch_stats if batch_stats != None else self.batch_stats
            ),
        )


class PolicyTrainState(flax.struct.PyTreeNode):
    value_normalize_fn: Callable = flax.struct.field(pytree_node=False)
    tx: optax.GradientTransformation = flax.struct.field(pytree_node=False)
    value_normalize_stats: flax.core.FrozenDict[str, Any]
    hyper_params: HyperParams
    opt_state: optax.OptState
    scheduler: Optional[optax.Schedule]
    scaler: Optional[DynamicScale]
    update_prng_key: random.PRNGKey

    def update(
        self,
        tx=None,
        value_normalize_stats=None,
        hyper_params=None,
        opt_state=None,
        scheduler=None,
        scaler=None,
        update_prng_key=None,
    ):
        return PolicyTrainState(
            value_normalize_fn = self.value_normalize_fn,
            tx = tx if tx != None else self.tx,
            value_normalize_stats = (
                value_normalize_stats if value_normalize_stats != None else
                    self.value_normalize_stats
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
    def load_policies(policy, path):
        checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        loaded = checkpointer.restore(path)

        return PolicyState(
            apply_fn = policy.apply,
            rnn_reset_fn = policy.clear_recurrent_state,
            params = loaded['policy_states']['params'],
            batch_stats = loaded['policy_states']['batch_stats'],
        )

    @staticmethod
    def create(
        policy: nn.Module,
        cfg: TrainConfig,
        algo: AlgoBase,
        base_rng,
        example_obs,
        example_rnn_states,
        checkify_errors,
    ):
        base_init_rng, pbt_rng = random.split(base_rng)

        def make_policies():
            return _make_policies(policy, cfg, algo, base_init_rng,
                example_obs, example_rnn_states)

        make_policies = jax.jit(checkify.checkify(
            make_policies, errors=checkify_errors))

        err, (policy_states, train_states) = make_policies()
        err.throw()

        return TrainStateManager(
            policy_states = policy_states,
            train_states = train_states,
            pbt_rng = pbt_rng,
        )


def _setup_value_normalizer(hyper_params, rng_key, fake_values):
    value_norm_decay = (hyper_params.value_normalizer_decay 
                        if hyper_params.normalize_values else 1.0)

    value_normalizer = EMANormalizer(
        value_norm_decay, disable=not hyper_params.normalize_values)

    value_normalizer_vars = value_normalizer.init(
        rng_key, 'normalize', False, fake_values)

    value_normalizer_stats = value_normalizer_vars.get('batch_stats', {})

    return value_normalizer.apply, value_normalizer_stats

def _setup_policy_state(
    policy,
    prng_key,
    rnn_states,
    obs,
):
    # The second prng key is passed as the key for sampling
    (fake_outs, rnn_states), variables = policy.init_with_output(
        prng_key, random.PRNGKey(0), rnn_states, obs,
        method='rollout')

    params = variables['params']

    if 'batch_stats' in variables:
        batch_stats = variables['batch_stats']
    else:
        batch_stats = {}

    return PolicyState(
        apply_fn = policy.apply,
        rnn_reset_fn = policy.clear_recurrent_state,
        params = params,
        batch_stats = batch_stats,
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

    value_norm_rng, update_rng = random.split(prng_key, 2)

    value_norm_fn, value_norm_stats = _setup_value_normalizer(
        hyper_params, value_norm_rng, fake_policy_out['values'])

    opt_state = optimizer.init(policy_state.params)

    if cfg.mixed_precision:
        scaler = DynamicScale()
    else:
        scaler = None

    return PolicyTrainState(
        value_normalize_fn = value_norm_fn,
        tx = optimizer,
        value_normalize_stats = value_norm_stats,
        hyper_params = hyper_params,
        opt_state = opt_state,
        scheduler = None,
        scaler = scaler,
        update_prng_key = update_rng,
    )

def _make_policies(
    policy,
    cfg,
    algo,
    base_init_rng,
    example_obs,
    example_rnn_states,
):
    setup_policy_state = partial(_setup_policy_state, policy)
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

    policy_init_base_rng, train_init_base_rng = random.split(base_init_rng)

    policy_init_rngs = random.split(policy_init_base_rng, num_make)

    policy_states, fake_policy_outs, rnn_states = setup_policy_states(
        policy_init_rngs, rnn_states, obs)

    setup_train_state = partial(_setup_train_state, cfg, algo)
    setup_train_states = jax.vmap(setup_train_state)
    
    train_init_rngs = random.split(train_init_base_rng, num_make)
    train_states = setup_train_states(
        train_init_rngs, policy_states, fake_policy_outs)

    if num_past_copies > 0:
        num_repeats = -(num_past_copies // -num_make)

        policy_states = jax.tree_map(
            lambda x: jnp.tile(x, (num_repeats, *([1] * (len(x.shape) - 1))))[
                0:num_make + num_past_copies],
            policy_states)

    return policy_states, train_states
