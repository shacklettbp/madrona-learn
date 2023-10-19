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
from .metrics import TrainingMetrics
from .profile import profile
from .train_state import HyperParams, PolicyTrainState
from .rollouts import RolloutData

from .algo_common import (
    InternalConfig,
    compute_advantages, 
    compute_returns,
    normalize_advantages,
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
    use_advantage: bool = True

    def name(self):
        return "ppo"

    def update_fn(self):
        return _ppo

    def metrics(self):
        return [
            'Loss',
            'Action Loss',
            'Value Loss',
            'Entropy Loss',
            'Rewards',
            'Returns',
            'Values',
            'Advantages',
            'Bootstrap Values',
        ]

    def finalize_rollouts_fn(self):
        if self.use_advantage:
            return compute_advantages
        else:
            return compute_returns


class PPOHyperParams(HyperParams):
    clip_coef: float
    value_loss_coef: float
    entropy_coef: float
    max_grad_norm: float

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
        normalized_advantages = normalize_advantages(cfg, mb['advantages'])

        ratio = jnp.exp(new_log_probs - mb['log_probs'])
        surr1 = normalized_advantages * ratio

        clipped_ratio = jnp.clip(
            ratio, 1.0 - cfg.algo.clip_coef, 1.0 + cfg.algo.clip_coef)
        surr2 = normalized_advantages * clipped_ratio

        action_obj = jnp.minimum(surr1, surr2)

        returns = mb['advantages'] + mb['values']

        if cfg.algo.clip_value_loss:
            old_values_normalized = state.value_normalize_fn(
                { 'batch_stats': state.value_normalize_stats },
                mode='normalize',
                update_stats=False,
                x=mb.values,
            )

            low = old_values_normalized - cfg.algo.clip_coef
            high = old_values_normalized + cfg.algo.clip_coef

            new_values = jnp.clip(new_values, low, high)

        normalized_returns, value_norm_mutable_out = state.value_normalize_fn(
            { 'batch_stats': state.value_normalize_stats },
            mode='normalize',
            update_stats=True,
            x=returns,
            mutable=['batch_stats'],
        )

        if cfg.algo.huber_value_loss:
            value_loss = optax.huber_loss(
                new_values_normalized, normalized_returns)
        else:
            value_loss = optax.l2_loss(
                new_values_normalized, normalized_returns)

        action_obj = jnp.mean(action_obj)
        value_loss = jnp.mean(value_loss)
        entropy_avg = jnp.mean(entropies)

        # Maximize the action objective function
        action_loss = -action_obj 
        value_loss = cfg.algo.value_loss_coef * value_loss
        # Maximize entropy
        entropy_loss = - cfg.algo.entropy_coef * entropy_avg

        loss = action_loss + value_loss + entropy_loss

        return loss, (
            ac_mutable_out['batch_stats'],
            value_norm_mutable_out['batch_stats'],
            loss,
            action_loss,
            value_loss,
            entropy_loss,
            returns,
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
            returns,
        ) = aux[1]

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
        'Returns': returns,
        'Advantages': mb['advantages'],
        'Values': mb['values'],
        'Rewards': mb['rewards'],
        'Bootstrap Values': mb['bootstrap_values'],
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
        mb_inds = all_inds.reshape((cfg.algo.num_mini_batches, -1))

        def mb_iter(mb_i, inputs):
            train_state, metrics = inputs

            with profile('Gather Minibatch'):
                mb = rollout_data.minibatch(mb_inds[mb_i])

            metrics = metrics.increment_count()

            train_state, metrics = _ppo_update(cfg, mb, train_state, metrics)

            metrics = metrics_cb(metrics, epoch_i, mb, train_state)

            return train_state, metrics

        return lax.fori_loop(
            0, cfg.algo.num_mini_batches, mb_iter, (train_state, metrics))

    train_state, metrics = lax.fori_loop(
        0, cfg.algo.num_epochs, epoch_iter, (train_state, init_metrics))

    return train_state, metrics
