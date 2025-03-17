import jax
from jax import lax, random, numpy as jnp
import flax
from flax import linen as nn
from flax.core import FrozenDict
import optax
import operator

from dataclasses import dataclass
from functools import partial
from typing import List, Callable, Dict, Any, Union

from .actor_critic import ActorCritic
from .cfg import AlgoConfig, TrainConfig, ParamExplore
from .metrics import TrainingMetrics, Metric
from .profile import profile
from .train_state import HyperParams, PolicyState, PolicyTrainState
from .rollouts import RolloutData

from .algo_common import AlgoBase, zscore_data

__all__ = [ "PPOConfig" ]

@dataclass(frozen=True)
class PPOConfig(AlgoConfig):
    num_epochs: int
    minibatch_size: int
    clip_coef: float
    value_loss_coef: float
    entropy_coef: Union[float, ParamExplore]
    max_grad_norm: float
    clip_value_loss: bool = False
    huber_value_loss: bool = False

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
        if cfg.dreamer_v3_critic or cfg.hlgauss_critic:
            assert not cfg.algo.clip_value_loss
            assert not cfg.algo.huber_value_loss
            assert not cfg.normalize_values

        if isinstance(cfg.lr, ParamExplore):
            lr = cfg.lr.base
        else:
            lr = cfg.lr

        if isinstance(cfg.algo.entropy_coef, ParamExplore):
            entropy = cfg.algo.entropy_coef.base
        else:
            entropy = cfg.algo.entropy_coef

        return PPOHyperParams(
            # Common
            lr = lr,
            gamma = cfg.gamma,
            gae_lambda = cfg.gae_lambda,
            normalize_values = cfg.normalize_values,
            value_normalizer_decay = cfg.value_normalizer_decay,
            max_advantage_est_decay = cfg.max_advantage_est_decay,
            # PPO
            clip_coef = cfg.algo.clip_coef,
            value_loss_coef = cfg.algo.value_loss_coef,
            entropy_coef = entropy,
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
            'Action Obj': Metric.init(True),
            'Value Loss': Metric.init(True),
            'Value Errors': Metric.init(True),
            'Entropy': Metric.init(True),
        })


def _ppo_update(
    cfg: TrainConfig,
    mb: FrozenDict[str, Any],
    mb_weights: jax.Array,
    policy_state: PolicyState,
    train_state: PolicyTrainState,
    metrics: TrainingMetrics,
):
    value_norm = train_state.value_normalizer

    def fwd_pass(params):
        with profile('AC Forward'):
            return policy_state.apply_fn(
                { 'params': params, 'batch_stats': policy_state.batch_stats },
                mb['rnn_start_states'], mb['dones'], mb['actions'], mb['obs'],
                train=True,
                method='update',
                mutable=['batch_stats'],
            )

    def loss_fn(params):
        fwd_results, ac_mutable_new = fwd_pass(params)
        new_log_probs = fwd_results['log_probs']
        entropies = fwd_results['entropies']

        if cfg.compute_advantages:
            advantages = mb['advantages'].astype(jnp.float32)
            if cfg.normalize_advantages:
                advantages = zscore_data(advantages)
        else:
            # For simplicity below when computing the surrogate loss
            # just use the general name "advantages"
            advantages = mb['returns'].astype(jnp.float32)
            if cfg.normalize_returns:
                advantages = zscore_data(advantages)

        def compute_action_obj(new_log_probs, old_log_probs):
            old_log_probs = old_log_probs.astype(jnp.float32)
            ratio = jnp.exp(new_log_probs - old_log_probs)

            num_action_dims = len(ratio.shape) - 2

            scores = advantages
            if num_action_dims > 1:
                scores = scores[..., None]

            surr1 = scores * ratio

            clipped_ratio = jnp.clip(ratio,
                1.0 - train_state.hyper_params.clip_coef.astype(ratio.dtype),
                1.0 + train_state.hyper_params.clip_coef.astype(ratio.dtype))
            surr2 = scores * clipped_ratio

            action_objs = jnp.minimum(surr1, surr2)

            return action_objs

        action_objs = jax.tree.map(
            compute_action_obj, new_log_probs, mb['log_probs'])

        if cfg.dreamer_v3_critic:
            critic_distributions = fwd_results['critic']

            value_losses = critic_distributions.two_hot_cross_entropy_loss(
                mb['returns'])

            value_errs = critic_distributions.mean() - mb['returns']

            new_value_norm_state = None
        elif cfg.hlgauss_critic:
            critic_distributions = fwd_results['critic']

            value_losses = critic_distributions.loss(mb['returns'])

            value_errs = critic_distributions.mean() - mb['returns']

            new_value_norm_state = None
        else:
            assert fwd_results['critic'].shape[-1] == 1
            new_values_normalized = fwd_results['critic']

            if value_norm == None:
                value_errs = new_values_normalized - mb['returns']
            else:
                value_errs = (value_norm.invert(train_state.value_normalizer_state,
                                                new_values_normalized) - 
                              mb['returns'])

            if cfg.algo.clip_value_loss:
                old_values_normalized = mb['values']

                low = old_values_normalized - train_state.hyper_params.clip_coef
                high = old_values_normalized + train_state.hyper_params.clip_coef

                new_values_normalized = jnp.clip(new_values_normalized, low, high)

            if value_norm == None:
                normalized_returns = mb['returns']
                new_value_norm_state = None
            else:
                new_value_norm_state, normalized_returns = (
                    value_norm.normalize_and_update_estimates(
                        train_state.value_normalizer_state, mb['returns']))

            if cfg.algo.huber_value_loss:
                value_losses = optax.huber_loss(
                    new_values_normalized, normalized_returns)
            else:
                value_losses = optax.l2_loss(
                    new_values_normalized, normalized_returns)

        def reduce_action_objs(action_objs):
            def reduce_action_obj(action_obj):
                return jnp.mean(mb_weights * action_obj.astype(jnp.float32))

            return sum(reduce_action_obj(action_obj) for action_obj in jax.tree_leaves(action_objs))


        def reduce_values(values_losses):
            return jnp.mean(mb_weights * value_losses, dtype=jnp.float32)


        def reduce_entropies(entropies):
            def reduce_action_entropy(k):
                entropy_coef = cfg.algo.entropy_coef[k]
                action_entropy = entropies[k]

                return entropy_coef * jnp.mean(
                    mb_weights * action_entropy.astype(jnp.float32))

            return sum(reduce_action_entropy(k) for k in entropies.keys())

        action_obj_avg = reduce_action_objs(action_objs)
        value_loss = reduce_values(value_losses)
        entropy_avg = reduce_entropies(entropies)

        # Maximize the action objective function
        action_loss = -action_obj_avg
        value_loss = train_state.hyper_params.value_loss_coef * value_loss
        # Maximize entropy
        #entropy_loss = - train_state.hyper_params.entropy_coef * entropy_avg
        entropy_loss = - entropy_avg

        loss = action_loss + value_loss + entropy_loss

        return loss, (
            ac_mutable_new['batch_stats'],
            new_value_norm_state,
            loss,
            action_objs,
            value_losses,
            entropies,
            value_errs,
        )

    with profile('Optimize'):
        params = policy_state.params
        scaler = train_state.scaler
        opt_state = train_state.opt_state

        # FIXME: consider converting params to cfg.compute_dtype when using
        # bfloat16 before computing gradients. Gradients are currently being
        # computed and / or stored in fp32. Would bias params need to be
        # handled differently?
        #lp_params = jax.tree_map(
        #    lambda x: x.astype(cfg.compute_dtype), params)
        lp_params = params
        if scaler != None:
            grad_fn = scaler.value_and_grad(loss_fn, has_aux=True)
            scaler, is_finite, aux, grads = grad_fn(lp_params)
        else:
            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            aux, grads = grad_fn(lp_params)

        with jax.numpy_dtype_promotion('standard'):
            param_updates, new_opt_state = train_state.tx.update(
                grads, opt_state, params)
        new_params = optax.apply_updates(params, param_updates)

        if scaler != None:
            where_finite = partial(jnp.where, is_finite)
            new_params = jax.tree_map(where_finite, new_params, params)
            new_opt_state = jax.tree_map(where_finite, new_opt_state, opt_state)

        (
            new_ac_batch_stats,
            new_value_norm_state,
            combined_loss,
            action_objs,
            value_losses,
            entropies,
            value_errs,
        ) = aux[1]

        def normalize_params(param, init_norm):
            if init_norm == None:
                return param

            return init_norm * param / jnp.linalg.vector_norm(param, ord=2)

        new_params = jax.tree.map(
            normalize_params, new_params, train_state.initial_weight_norms)
        
        def normalize_layernorms(d):
            if not isinstance(d, dict):
                return d

            new = {}
            for k, v in d.items():
                if "LayerNorm" in k:
                    cur_bias = v['impl']['bias']
                    cur_scale = v['impl']['scale']
                    
                    num_features = cur_scale.shape[-1]

                    normalize_factor = jnp.sqrt(num_features / (
                        jnp.dot(cur_bias, cur_bias) + jnp.dot(cur_scale, cur_scale)))

                    new[k] = {
                        'impl': {
                            'bias': normalize_factor * cur_bias,
                            'scale': normalize_factor * cur_scale,
                        },
                    }
                else:
                    new[k] = normalize_layernorms(v)

            return new

        new_params = normalize_layernorms(new_params)

        policy_state = policy_state.update(
            params = new_params,
            batch_stats = new_ac_batch_stats,
        )

        train_state = train_state.update(
            value_normalizer_state = new_value_norm_state,
            opt_state = new_opt_state,
            scaler = scaler,
        )

    with profile('Record Metrics'):
        metrics = metrics.record({
            'Loss': combined_loss,
            'Action Obj': jnp.concatenate(
                [x.reshape(-1, x.shape[-1]) for x in jax.tree.leaves(action_objs)],
                axis=-1),
            'Value Loss': value_losses,
            'Value Errors': jnp.abs(value_errs),
            'Entropy': jnp.concatenate(
                [x.reshape(-1, x.shape[-1]) for x in jax.tree.leaves(entropies)],
                axis=-1),
        })

    return policy_state, train_state, metrics

def _ppo(
    cfg: TrainConfig,
    policy_state: PolicyState,
    train_state: PolicyTrainState,
    rollout_data: RolloutData,
    user_metrics_cb: Callable,
    init_metrics: TrainingMetrics,
):
    if cfg.filter_advantages:
        rollout_data = rollout_data.flatten_time()

        advantages = rollout_data.all()['advantages']
        advantages_abs = jnp.abs(advantages)
        max_advantages = jnp.max(advantages_abs)

        max_advantage_est_state = train_state.max_advantage_est_state
        old_max_adv_mu = max_advantage_est_state['mu']
        max_advantage_est_state = train_state.max_advantage_est.update_estimates(
            max_advantage_est_state, max_advantages)

        train_state = train_state.update(
            max_advantage_est_state = max_advantage_est_state,
        )

        cur_max_advantage_est = max_advantage_est_state['mu']

        advantages_abs_flat = advantages_abs.reshape(-1) 

        advantage_indices_sort = jnp.argsort(advantages_abs_flat, descending=True)
        num_above_threshold = jnp.sum(jnp.where(
            advantages_abs_flat >= 0.01 * cur_max_advantage_est,
            1, 0))

        num_minibatches = jnp.minimum((num_above_threshold + (cfg.algo.minibatch_size - 1)) // cfg.algo.minibatch_size, advantages_abs_flat.size // cfg.algo.minibatch_size)

        num_datapoints = num_minibatches * cfg.algo.minibatch_size
        valid_inds = jnp.where(jnp.arange(advantages_abs_flat.size) < num_datapoints,
                               advantage_indices_sort,
                               -1)

        traj_weights = jnp.ones((advantages.shape[0],), dtype=jnp.float32)
    elif cfg.importance_sample_trajectories:
        advantages = rollout_data.all()['advantages'].astype(jnp.float32)
        values = rollout_data.all()['values'].astype(jnp.float32)
        returns = rollout_data.all()['returns'].astype(jnp.float32)

        num_total_trajectories = advantages.shape[0]

        num_minibatches = cfg.importance_sample_num_minibatches
        num_sampled_trajectories = num_minibatches * cfg.algo.minibatch_size
        assert num_sampled_trajectories < num_total_trajectories
        assert num_minibatches > 0

        advantages_abs = jnp.abs(advantages)
        traj_avg_advantage_magnitude = jnp.mean(advantages_abs, axis=1)

        value_errs = jnp.abs(values - returns)
        traj_avg_value_err = jnp.mean(value_errs, axis=1)

        traj_scores = traj_avg_advantage_magnitude + traj_avg_value_err
        traj_probs = jax.nn.softmax(traj_scores, axis=0)
        traj_weights = (1.0 / num_total_trajectories) / traj_probs

        sample_rnd, train_state = train_state.gen_update_rnd()

        sampled_traj_indices = random.choice(
            sample_rnd, num_total_trajectories,
            shape=(num_sampled_trajectories,), replace=False, p=traj_probs.reshape(-1))

        valid_inds = sampled_traj_indices
    else:
        advantages = rollout_data.all()['advantages']
        num_minibatches = advantages.shape[0] // cfg.algo.minibatch_size
        assert advantages.shape[0] % cfg.algo.minibatch_size == 0

        valid_inds = jnp.arange(advantages.shape[0])

        traj_weights = jnp.ones((advantages.shape[0],1), dtype=jnp.float32)

    def epoch_iter(epoch_i, inputs):
        policy_state, train_state, metrics = inputs

        mb_rnd, train_state = train_state.gen_update_rnd()

        with profile('Compute Minibatch Indices'):
            rnd_inds = random.permutation(mb_rnd, valid_inds)

            def filter_valid_inds(x):
                keys = jnp.where(x == -1, 1, 0)
                valid = jnp.argsort(keys, stable=True)
                return x[valid]

            rnd_inds = filter_valid_inds(rnd_inds)

        def mb_iter(mb_i, inputs):
            policy_state, train_state, metrics = inputs

            with profile('Gather Minibatch'):
                mb_inds = lax.dynamic_slice(
                    rnd_inds, (mb_i * cfg.algo.minibatch_size,), 
                    (cfg.algo.minibatch_size,))
                #jax.debug.print("Um {}", mb_inds)
                mb = rollout_data.minibatch(mb_inds)
                mb_weights = traj_weights[mb_inds]

            policy_state, train_state, metrics = _ppo_update(
                cfg, mb, mb_weights, policy_state, train_state, metrics)

            with profile('Metrics Callback'):
                metrics = user_metrics_cb(
                    metrics, epoch_i, mb, policy_state, train_state)

            return policy_state, train_state, metrics

        return lax.fori_loop(
            0, num_minibatches, mb_iter,
            (policy_state, train_state, metrics))

    policy_state, train_state, metrics = lax.fori_loop(
        0, cfg.algo.num_epochs, epoch_iter,
        (policy_state, train_state, init_metrics))

    return policy_state, train_state, metrics
