import jax
from jax import lax, random, numpy as jnp
import flax
from flax import linen as nn
from flax.core import FrozenDict
import optax

from dataclasses import dataclass
from functools import partial
from typing import List, Callable, Dict, Any

from .actor_critic import ActorCritic
from .cfg import AlgoConfig, TrainConfig
from .metrics import TrainingMetrics, Metric
from .profile import profile
from .train_state import HyperParams, PolicyTrainState
from .rollouts import RolloutData

from .algo_common import (
    AlgoBase,
    InternalConfig,
    zscore_data,
)

__all__ = [ "PPOConfig" ]

@dataclass(frozen=True)
class PPOConfig(AlgoConfig):
    num_epochs: int
    num_mini_batches: int
    clip_coef: float
    value_loss_coef: float
    entropy_coef: float
    max_grad_norm: float
    clip_value_loss: bool = False
    huber_value_loss: bool = True

    def name(self):
        return "ppo"

    def setup(self):
        return PPO()


class PPOHyperParams(HyperParams):
    clip_coef: float
    value_loss_coef: float
    entropy_coef: float
    max_grad_norm: float


class PPO(AlgoBase):
    def init_hyperparams(
        self,
        cfg: TrainConfig,
    ):
        return PPOHyperParams(
            # Common
            lr = cfg.lr,
            gamma = cfg.gamma,
            gae_lambda = cfg.gae_lambda,
            normalize_values = cfg.normalize_values,
            value_normalizer_decay = cfg.value_normalizer_decay,
            # PPO
            clip_coef = cfg.algo.clip_coef,
            value_loss_coef = cfg.algo.value_loss_coef,
            entropy_coef = cfg.algo.entropy_coef,
            max_grad_norm = cfg.algo.max_grad_norm,
        )

    def make_optimizer(
        self,
        hyper_params: HyperParams,
    ):
        return optax.chain(
            optax.clip_by_global_norm(hyper_params.max_grad_norm),
            optax.adam(learning_rate= hyper_params.lr))

    def update(self, *args, **kwargs):
        return _ppo(*args, **kwargs)

    def add_metrics(
        self,
        cfg: TrainConfig,
        metrics: FrozenDict[str, Metric],
    ):
        return metrics.copy({
            'Loss': Metric.init(True),
            'Action Loss': Metric.init(True),
            'Value Loss': Metric.init(True),
            'Entropy Loss': Metric.init(True),
        })


def _ppo_update(
    cfg: TrainConfig,
    mb: FrozenDict[str, Any],
    state: PolicyTrainState,
    metrics: TrainingMetrics,
):
    def fwd_pass(params):
        with profile('AC Forward'):
            return state.apply_fn(
                { 'params': params, 'batch_stats': state.batch_stats },
                mb['rnn_start_states'], mb['dones'], mb['actions'], mb['obs'],
                train=True,
                method='update',
                mutable=['batch_stats'],
            )

    def loss_fn(params):
        (new_log_probs, entropies, new_values_normalized), ac_mutable_out = \
            fwd_pass(params)

        if cfg.compute_advantages:
            advantages = mb['advantages']
            if cfg.normalize_advantages:
                advantages = zscore_data(advantages)
        else:
            # For simplicity below when computing the surrogate loss
            # just use the general name "advantages"
            advantages = mb['returns']
            if cfg.normalize_returns:
                advantages = zscore_data(advantages)

        ratio = jnp.exp(new_log_probs - mb['log_probs'])
        surr1 = advantages * ratio

        clipped_ratio = jnp.clip(ratio, 1.0 - state.hyper_params.clip_coef,
                                 1.0 + state.hyper_params.clip_coef)
        surr2 = advantages * clipped_ratio

        action_obj = jnp.minimum(surr1, surr2)


        jax.debug.print("Obs {}", mb['obs']['self'], ordered=True)
        jax.debug.print("Returns {}", mb['returns'], ordered=True)
        jax.debug.print("{}", mb['values'], ordered=True)
        jax.debug.print("NV: {}", new_values_normalized, ordered=True)
        jax.debug.print("{}", mb['advantages'], ordered=True)
        jax.debug.print("{}", mb['log_probs'], ordered=True)
        jax.debug.print("{} {} {}",
            advantages, jnp.mean(advantages), jnp.var(advantages), ordered=True)
        jax.debug.print("{}", ratio, ordered=True)
        jax.debug.print("{}", action_obj, ordered=True)

        if cfg.algo.clip_value_loss:
            old_values_normalized = state.value_normalize_fn(
                { 'batch_stats': state.value_normalize_stats },
                mode='normalize',
                update_stats=False,
                x=mb['values'],
            )

            low = old_values_normalized - state.hyper_params.clip_coef
            high = old_values_normalized + state.hyper_params.clip_coef

            new_values = jnp.clip(new_values, low, high)

        normalized_returns, value_norm_mutable_out = state.value_normalize_fn(
            { 'batch_stats': state.value_normalize_stats },
            mode='normalize',
            update_stats=True,
            x=mb['returns'],
            mutable=['batch_stats'],
        )

        jax.debug.print("Values, Returns: {} {}",
            jnp.mean(new_values_normalized), jnp.mean(normalized_returns),
            ordered=True)

        if cfg.algo.huber_value_loss:
            value_loss = optax.huber_loss(
                new_values_normalized, normalized_returns)
        else:
            value_loss = optax.l2_loss(
                new_values_normalized, normalized_returns)

        action_obj = jnp.mean(action_obj)
        value_loss = jnp.mean(value_loss)
        entropy_avg = jnp.mean(entropies)

        jax.debug.print("V loss: {}", value_loss, ordered=True)

        # Maximize the action objective function
        action_loss = -action_obj 
        value_loss = state.hyper_params.value_loss_coef * value_loss
        # Maximize entropy
        entropy_loss = - state.hyper_params.entropy_coef * entropy_avg

        loss = action_loss + value_loss + entropy_loss

        return loss, (
            ac_mutable_out['batch_stats'],
            value_norm_mutable_out['batch_stats'],
            loss,
            action_loss,
            value_loss,
            entropy_loss,
        )

    with profile('Optimize'):
        scaler = state.scaler
        params = state.params
        opt_state = state.opt_state

        if scaler != None:
            grad_fn = scaler.value_and_grad(loss_fn, has_aux=True)
            scaler, is_finite, aux, grads = grad_fn(params)
        else:
            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            aux, grads = grad_fn(state.params)

        flattened_grads, _ = jax.tree_util.tree_flatten_with_path(grads)
        jax.debug.print("\nGRADS: {}", aux[1][2], ordered=True)
        for k, v in flattened_grads:
            jax.debug.print(jax.tree_util.keystr(k) + ": {} {} {}", jnp.mean(v), jnp.min(v), jnp.max(v), ordered=True)
            jax.debug.print(" {}", v, ordered=True)

        with jax.numpy_dtype_promotion('standard'):
            param_updates, new_opt_state = state.tx.update(
                grads, opt_state, params)
        new_params = optax.apply_updates(params, param_updates)

        if scaler != None:
            where_finite = partial(jnp.where, is_finite)
            new_params = jax.tree_map(where_finite, new_params, params)
            new_opt_state = jax.tree_map(where_finite, new_opt_state, opt_state)

        (
            new_ac_batch_stats,
            new_vn_batch_stats,
            combined_loss,
            action_loss,
            value_loss,
            entropy_loss,
        ) = aux[1]

        flattened_new_params, _ = jax.tree_util.tree_flatten_with_path(new_params)
        jax.debug.print("\nPARAMS:", ordered=True)
        for k, v in flattened_new_params:
            jax.debug.print(jax.tree_util.keystr(k) + ": {}", v, ordered=True)

        state = state.update(
            params = new_params,
            batch_stats = new_ac_batch_stats,
            value_normalize_stats = new_vn_batch_stats,
            opt_state = new_opt_state,
            scaler = scaler,
        )

    metrics = metrics.record({
        'Loss': combined_loss,
        'Action Loss': action_loss,
        'Value Loss': value_loss,
        'Entropy Loss': entropy_loss,
    })

    return state, metrics

def _ppo(
    cfg: TrainConfig,
    icfg: InternalConfig,
    train_state: PolicyTrainState,
    rollout_data: RolloutData,
    metrics_cb: Callable,
    init_metrics: TrainingMetrics,
):
    def epoch_iter(epoch_i, inputs):
        train_state, metrics = inputs

        mb_rnd, train_state = train_state.gen_update_rnd()

        all_inds = random.permutation(mb_rnd, icfg.num_train_seqs_per_policy)
        jax.debug.print("{}", all_inds)
        all_inds = jnp.arange(icfg.num_train_seqs_per_policy)

        mb_inds = all_inds.reshape((cfg.algo.num_mini_batches, -1))

        def mb_iter(mb_i, inputs):
            train_state, metrics = inputs

            with profile('Gather Minibatch'):
                mb = rollout_data.minibatch(mb_inds[mb_i])

            train_state, metrics = _ppo_update(cfg, mb, train_state, metrics)

            metrics = metrics_cb(metrics, epoch_i, mb, train_state)

            return train_state, metrics

        return lax.fori_loop(
            0, cfg.algo.num_mini_batches, mb_iter, (train_state, metrics))

    train_state, metrics = lax.fori_loop(
        0, cfg.algo.num_epochs, epoch_iter, (train_state, init_metrics))

    return train_state, metrics
