import torch
from time import time
from dataclasses import dataclass
from typing import List, Optional
from .amp import AMPInfo
from .cfg import SimInterface
from .actor_critic import ActorCritic, RecurrentStateConfig

@dataclass(frozen = True)
class Rollouts:
    obs: List[torch.Tensor]
    actions: torch.Tensor
    log_probs: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor
    values: torch.Tensor
    bootstrap_values: torch.Tensor
    rnn_hidden_starts: tuple[torch.Tensor, ...]

class RolloutManager:
    def __init__(
            self,
            dev : torch.device,
            sim : SimInterface,
            steps_per_update : int,
            amp : AMPInfo,
            recurrent_cfg : RecurrentStateConfig,
        ):
        self.need_obs_copy = dev != sim.obs[0].device

        if dev.type == 'cuda':
            float_storage_type = torch.float16
        else:
            float_storage_type = torch.bfloat16

        self.actions = torch.zeros(
            (steps_per_update, *sim.actions.shape),
            dtype=sim.actions.dtype, device=dev)

        self.log_probs = torch.zeros(
            (steps_per_update, *sim.actions.shape),
            dtype=float_storage_type, device=dev)

        self.dones = torch.zeros(
            (steps_per_update, *sim.dones.shape),
            dtype=float_storage_type, device=dev)

        self.rewards = torch.zeros(
            (steps_per_update, *sim.rewards.shape),
            dtype=float_storage_type, device=dev)

        self.values = torch.zeros(
            (steps_per_update, *sim.rewards.shape),
            dtype=float_storage_type, device=dev)

        # FIXME: seems like this could be combined into self.values by
        # making self.values one longer, but that breaks torch.compile
        self.bootstrap_values = torch.zeros(
            sim.rewards.shape, dtype=amp.compute_dtype, device=dev)

        self.obs = []

        for obs_tensor in sim.obs:
            self.obs.append(torch.zeros(
                (steps_per_update, *obs_tensor.shape),
                dtype=obs_tensor.dtype, device=dev))

        if self.need_obs_copy:
            self.final_obs = []

            for obs_tensor in sim.obs:
                self.final_obs.append(torch.zeros(
                    obs_tensor.shape, dtype=obs_tensor.dtype, device=dev))

        self.steps_per_update = steps_per_update

        self.rnn_start_states = []
        self.rnn_end_states = []
        self.rnn_alt_states = []
        for rnn_state_shape in recurrent_cfg.shapes:
            # expand shape to batch size
            rnn_batch_shape = (*rnn_state_shape[0:2],
                sim.actions.shape[0], rnn_state_shape[2])

            rnn_start_state = torch.zeros(
                rnn_batch_shape, dtype=amp.compute_dtype, device=dev)
            rnn_end_state = torch.zeros_like(rnn_start_state)
            rnn_alt_state = torch.zeros_like(rnn_start_state)

            self.rnn_start_states.append(rnn_start_state)
            self.rnn_end_states.append(rnn_end_state)
            self.rnn_alt_states.append(rnn_alt_state)

        self.rnn_start_states = tuple(self.rnn_start_states)
        self.rnn_end_states = tuple(self.rnn_end_states)
        self.rnn_alt_states = tuple(self.rnn_alt_states)

    def collect(
            self,
            amp : AMPInfo,
            sim : SimInterface,
            actor_critic : ActorCritic,
        ):
        step_total = 0

        for start_state, end_state in zip(
                self.rnn_start_states, self.rnn_end_states):
            start_state.copy_(end_state)

        rnn_states_cur_in = self.rnn_end_states
        rnn_states_cur_out = self.rnn_alt_states

        for slot in range(0, self.steps_per_update):
            cur_obs_buffers = [obs[slot] for obs in self.obs]

            for obs_idx, step_obs in enumerate(sim.obs):
                cur_obs_buffers[obs_idx].copy_(step_obs, non_blocking=True)

            with amp.enable():
                actor_critic.rollout_infer(
                    self.actions[slot], self.log_probs[slot],
                    self.values[slot], rnn_hidden_cur_out,
                    rnn_hidden_cur_in, *cur_obs_buffers)

            rnn_states_cur_in, rnn_states_cur_out = \
                rnn_states_cur_out, rnn_states_cur_in

            sim.actions.copy_(self.actions[slot], non_blocking=True)

            step_start_time = time()
            sim.step()
            step_total += time() - step_start_time

            self.dones[slot].copy_(sim.dones, non_blocking=True)
            self.rewards[slot].copy_(sim.rewards, non_blocking=True)

        if self.need_obs_copy:
            final_obs = self.final_obs
            for obs_idx, step_obs in enumerate(sim.obs):
                final_obs[obs_idx].copy_(step_obs, non_blocking=True)
        else:
            final_obs = sim.obs

        with amp.enable():
            # rnn_hidden_cur_in and rnn_hidden_cur_out are flipped after each
            # iter so rnn_hidden_cur_in is the final output
            self.rnn_end_states = rnn_states_cur_in
            self.rnn_alt_states = rnn_states_cur_out

            actor_critic.critic_infer(
                self.bootstrap_values, None, self.rnn_hidden_end, *final_obs)

        # Right now this just returns the rollout manager's pointers,
        # but in the future could return only one set of buffers from a
        # double buffered store, etc

        return Rollouts(
            obs = self.obs,
            actions = self.actions,
            log_probs = self.log_probs,
            dones = self.dones,
            rewards = self.rewards,
            values = self.values,
            bootstrap_values = self.bootstrap_values,
            rnn_hidden_starts =
                self.rnn_hidden_start if self.recurrent_policy else None,
        )
