import jax
from jax import lax, random, numpy as jnp
import flax
from flax import linen as nn
import flax.training.dynamic_scale
import flax.training.train_state
import optax

from dataclasses import dataclass
from typing import Optional, Any, Callable

from .actor_critic import ActorCritic
from .moving_avg import EMANormalizer

class HyperParams(flax.struct.PyTreeNode):
    lr: float
    gamma: float
    gae_lambda: float


class PolicyTrainState(flax.struct.PyTreeNode):
    apply_fn: Callable = flax.struct.field(pytree_node=False)
    params: flax.core.FrozenDict[str, Any]
    batch_stats: flax.core.FrozenDict[str, Any]
    value_normalize_fn: Callable = flax.struct.field(pytree_node=False)
    value_normalize_vars: flax.core.FrozenDict[str, Any]
    hyper_params: HyperParams
    tx: optax.GradientTransformation = flax.struct.field(pytree_node=False)
    opt_state: optax.OptState
    scheduler: Optional[optax.Schedule]
    scaler: Optional[flax.training.dynamic_scale.DynamicScale]
    update_prng_key: random.PRNGKey

    def update(
        self,
        params=None,
        batch_stats=None,
        value_normalize_vars=None,
        hyper_params=None,
        tx=None,
        opt_state=None,
        scheduler=None,
        scaler=None,
        update_prng_key=None,
    ):
        return PolicyTrainState(
            apply_fn = self.apply_fn,
            params = params if params != None else self.params,
            batch_stats = (
                batch_stats if batch_stats != None else self.batch_stats),
            value_normalize_fn = self.value_normalize_fn,
            value_normalize_vars = (
                value_normalize_vars if value_normalize_vars != None else
                    self.value_normalize_vars),
            hyper_params = (
                hyper_params if hyper_params != None else self.hyper_params),
            tx = tx if tx != None else self.tx,
            opt_state = opt_state if opt_state != None else self.opt_state,
            scheduler = scheduler if scheduler != None else self.scheduler,
            scaler = scaler if scaler != None else self.scaler,
            update_prng_key = (
                update_prng_key if update_prng_key != None else 
                    self.update_prng_key),
        )

    def gen_update_rnd(self):
        rnd, next_key = random.split(self.update_prng_key)
        return rnd, self.update(update_prng_key=next_key)


class TrainStateManager(flax.struct.PyTreeNode):
    train_states: PolicyTrainState

    def save(self, update_idx, path):
        def get_scheduler_state(learning_state):
            if learning_state.scheduler != None:
                scheduler_state_dict = learning_state.scheduler.state_dict()
            else:
                scheduler_state_dict = None

            return scheduler_state_dict

        def get_scaler_state(learning_state):
            if amp.scaler != None:
                scaler_state_dict = amp.scaler.state_dict()
            else:
                scaler_state_dict = None

            return scaler_state_dict

        policies = [ state.policy.state_dict() for state in self.policy_states ]
        optimizers = [ state.optimizer.state_dict() for state in self.policy_states ]
        schedulers = [ get_scheduler_state(state) for state in self.policy_states ]
        scalers = [ get_scaler_state(state) for state in self.policy_states ]
        value_norms = [ state.value_normalizer for state in self.policy_states ]

        torch.save({
            'next_update': update_idx + 1,
            'policies': policies,
            'optimizers': optimizers,
            'schedulers': schedulers,
            'scalers': scalers,
            'value_normalizers': value_norms,
            'amp': {
                'device_type': amp.device_type,
                'enabled': amp.enabled,
                'compute_dtype': amp.compute_dtype,
            },
        }, path)

    def load(self, path):
        loaded = torch.load(path)

        policies = loaded['policies']
        optimizers = loaded['optimizers']
        schedulers = loaded['schedulers']
        scalers = loaded['scalers']
        value_normalizers = loaded['value_normalizers']

        for i, state in enumerate(self.policy_states):
            state.policy.load_state_dict(policies[i])
            state.optimizer.load_state_dict(optimizers[i])
            sched_state = schedulers[i]
            if sched_state:
                state.scheduler.load_state_dict(sched_state)
            else:
                assert(state.scheduler == None)

            scaler_state = scalers[i]
            if scaler_state:
                state.scaler.load_state_dict(scaler_state)
            else:
                assert(state.scaler == None)

            state.value_normalizer.load_state_dict(value_normalizers[i])

        amp_dict = loaded['amp']
        assert(
            amp.device_type == amp_dict['device_type'] and
            amp.enabled == amp_dict['enabled'] and
            amp.compute_dtype == amp_dict['compute_dtype'])

        return loaded['next_update']

    @staticmethod
    def load_policy_weights(path):
        loaded = torch.load(path)
        return loaded['policies']
