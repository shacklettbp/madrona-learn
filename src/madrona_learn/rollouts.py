import torch
from torch.func import stack_module_state, functional_call
import numpy as np
from time import time
from dataclasses import dataclass
from typing import List, Optional
from .amp import amp
from .cfg import SimInterface, TrainConfig
from .actor_critic import ActorCritic, RecurrentStateConfig
from .algo_common import InternalConfig
from .profile import profile

@dataclass(frozen = True)
class Rollouts:
    obs: List[torch.Tensor]
    actions: torch.Tensor
    log_probs: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor
    values: torch.Tensor
    bootstrap_values: torch.Tensor
    rnn_start_states: tuple[torch.Tensor, ...]


class RolloutManager:
    def __init__(
            self,
            dev : torch.device,
            sim : SimInterface,
            cfg : TrainConfig,
            icfg : InternalConfig,
            recurrent_cfg : RecurrentStateConfig,
        ):
        cpu_dev = torch.device('cpu')

        self.dev = dev
        self.need_sim_copy = sim.obs[0].device != dev

        elems_per_policy = icfg.num_train_agents // cfg.pbt_ensemble_size
        gather_indices_tmp = []
        for i in range(icfg.num_train_agents):
            policy_idx = i // cfg.pbt_ensemble_size
            policy_offset = i % cfg.pbt_ensemble_size
            policy_base = elems_per_policy * policy_idx
            gather_indices_tmp.append(policy_base + policy_offset)

        self.gather_indices = torch.tensor(gather_indices_tmp,
            dtype=torch.long, device=dev)

        self.vmap_in_dims_rollout = (
                [ 0, 0, 0, 0, 2, 2 ] + [ 0 ] * len(self.obs_in)
            )

        self.vmap_in_dims_critic = (
                [ 0, 0, None, 2 ] + [ 0 ] * len(self.obs_in)
            )

        if dev.type == 'cuda':
            float_storage_type = torch.float16
        else:
            float_storage_type = torch.bfloat16

        self.actions = torch.zeros(
            (num_bptt_chunks, num_bptt_steps,
             icfg.num_train_agents, *sim.actions.shape[1:]),
            dtype=sim.actions.dtype, device=cpu_dev)

        self.actions_gather = torch.zeros(
            self.actions.shape[2:],
            dtype=sim.actions.dtype, device=dev)

        self.log_probs = torch.zeros(
            self.actions.shape,
            dtype=float_storage_type, device=cpu_dev)

        self.log_probs_gather = torch.zeros(
            self.actions_gather.shape,
            dtype=float_storage_type, device=dev)

        self.dones = torch.zeros(
            (num_bptt_chunks, num_bptt_steps,
             icfg.num_train_agents, 1),
            dtype=torch.bool, device=cpu_dev)

        self.rewards = torch.zeros(self.dones.shape,
            dtype=float_storage_type, device=cpu_dev)

        if self.need_sim_copy:
            self.dones_gather = torch.zeros(
                self.dones.shape[2:],
                dtype=torch.bool, device=dev)

            self.rewards_gather = torch.zeros(
                self.dones_gather.shape,
                dtype=float_storage_type, device=dev)

        self.values = torch.zeros(self.dones.shape,
            dtype=float_storage_type, device=cpu_dev)

        self.bootstrap_values = torch.zeros((icfg.num_train_agents, 1),
            dtype=float_storage_type, device=cpu_dev)

        self.values_gather = torch.zeros(
            self.values.shape[2:],
            dtype=float_storage_type, device=dev)

        self.bootstrap_values_gather = torch.zeros(
            self.bootstrap_values.shape,
            dtype=float_storage_type, device=dev)

        self.values_out = torch.zeros(sim.rewards.shape,
            dtype=amp.compute_dtype, device=cpu_dev)

        self.log_probs_out = torch.zeros(self.actions_out.shape,
            dtype=float_storage_type, device=dev)

        if self.need_sim_copy:
            self.actions_out = torch.zeros(sim.actions.shape,
                dtype=sim.actions.dtype, device=dev)
        else:
            self.actions_out = sim.actions

        self.obs = []
        self.obs_in = []
        self.obs_gather = []

        for obs_tensor in sim.obs:
            self.obs.append(torch.zeros(
                (num_bptt_chunks, num_bptt_steps,
                 icfg.num_train_agents, *obs_tensor.shape[1:]),
                dtype=obs_tensor.dtype, device=cpu_dev))

            if self.need_sim_copy:
                self.obs_in.append(torch.zeros(obs_tensor.shape, 
                    dtype=obs_tensor.dtype, device=dev))

                self.obs_gather.append(torch.zeros(
                    (icfg.num_train_agents, *obs_tensor.shape[1:]),
                    dtype=obs_tensor.dtype, device=dev))
            else:
                self.obs_in.append(obs_tensor)

        self.rnn_end_states = []
        self.rnn_alt_states = []
        self.rnn_start_states = []
        for rnn_state_shape in recurrent_cfg.shapes:
            # expand shape to batch size
            batched_state_shape = (*rnn_state_shape[0:2],
                icfg.rollout_batch_size, rnn_state_shape[2])

            rnn_end_state = torch.zeros(
                batched_state_shape, dtype=amp.compute_dtype, device=dev)
            rnn_alt_state = torch.zeros_like(rnn_end_state)

            self.rnn_end_states.append(rnn_end_state)
            self.rnn_alt_states.append(rnn_alt_state)

            bptt_starts_shape = (num_bptt_chunks, *rnn_state_shape[0:2], icfg.num_train_agents,
                                 rnn_state_shape[2])

            rnn_start_state = torch.zeros(
                bptt_starts_shape, dtype=amp.compute_dtype, device=dev)

            self.rnn_start_states.append(rnn_start_state)

        self.rnn_end_states = tuple(self.rnn_end_states)
        self.rnn_alt_states = tuple(self.rnn_alt_states)
        self.rnn_start_states = tuple(self.rnn_start_states)

    def collect(
            self,
            sim : SimInterface,
            ac_functional,
            policies : List[ActorCritic],
            policy_assignments : np.array, # num_envs * num_teams
        ):

        policy_params, policy_buffers = stack_module_state(policies)

        def fpolicy_rollout(policy_idx, *args):
            unsqueezed = [a.unsqueeze(dim=0) for a in args]

            return functional_call(ac_functional,
                (policy_params[policy_idx], policy_buffers[policy_idx]),
                ('rollout', *unsqueezed))

        def fpolicy_critic(policy_idx, *args):
            unsqueezed = [a.unsqueeze(dim=0) for a in args]

            return functional_call(ac_functional,
                (policy_params[policy_idx], policy_buffers[policy_idx]),
                ('critic', *unsqueezed))

        rnn_states_cur_in = self.rnn_end_states
        rnn_states_cur_out = self.rnn_alt_states

        for bptt_chunk in range(0, self.actions.shape[0]):
            with profile("Cache RNN state"):
                # Cache starting RNN state for this chunk
                for start_state, end_state in zip(
                        self.rnn_start_states, rnn_states_cur_in):
                    torch.index_select(end_state, 2,
                                       self.gather_indices, out=start_state[bptt_chunk])

            for slot in range(0, self.actions.shape[1]):
                with profile('Policy Infer', gpu=True):
                    if self.need_sim_copy:
                        for sim_ob, policy_in in zip(sim.obs, self.obs_in):
                            policy_in.copy_(sim_ob, non_blocking=True)

                    with amp.enable():
                        torch.vmap(fpolicy_rollout,
                                   in_dims=self.vmap_in_dims_rollout)(
                            policy_assignments,
                            self.actions_out,
                            self.log_probs_out,
                            self.values_out,
                            rnn_states_cur_out,
                            rnn_states_cur_in,
                            *self.obs_in,
                        )


                    rnn_states_cur_in, rnn_states_cur_out = \
                        rnn_states_cur_out, rnn_states_cur_in

                with profile('Pre Step Rollout Store')
                    cur_obs_store = [obs[bptt_chunk, slot] for obs in self.obs]
                    cur_actions_store = self.actions[bptt_chunk, slot]
                    cur_log_probs_store = self.log_probs[bptt_chunk, slot]
                    cur_values_store = self.values[bptt_chunk, slot]
                    
                    if self.need_sim_copy:
                        for cur_ob, gather_ob, store_ob in zip(
                                self.obs_in, self.obs_gather, cur_obs_store):
                            torch.index_select(cur_ob, 0, self.gather_indices,
                                               out=gather_ob)
                            store_ob.copy_(gather_ob, non_blocking=True)
                    else:
                        for sim_ob, store_ob in zip(sim.obs, cur_obs_store):
                            torch.index_select(cur_ob, 0, self.gather_indices,
                                               out=gather_ob)

                    torch.index_select(self.actions_out, 0, self.gather_indices,
                                       out=self.actions_gather)
                    cur_actions_store.copy_(self.actions_gather, non_blocking=True)

                    torch.index_select(self.log_probs_out, 0, self.gather_indices,
                                       out=self.log_probs_gather)
                    cur_log_probs_store.copy_(self.log_probs_gather, non_blocking=True)

                    torch.index_select(self.values_out, 0, self.gather_indices,
                                       out=self.values_gather)
                    cur_values_store.copy_(self.values_gather, non_blocking=True)

                with profile('Simulator Step'):
                    if self.need_sim_copy:
                        # This isn't non-blocking because if the sim is running in
                        # CPU mode, the copy needs to be finished before sim.step()
                        # FIXME: proper pytorch <-> madrona cuda stream integration

                        sim.actions.copy_(self.actions_out)

                    sim.step()

                with profile('Post Step Rollout Store'):
                    cur_rewards_store = self.rewards[bptt_chunk, slot]
                    cur_dones_store = self.dones[bptt_chunk, slot]

                    if self.need_sim_copy:
                        torch.index_select(sim.rewards, 0, self.gather_indices,
                                           out=self.rewards_gather)
                        cur_rewards_store.copy_(self.rewards_gather)

                        torch.index_select(sim.dones, 0, self.gather_indices,
                                           out=self.dones_gather)
                        cur_dones_store.copy_(self.dones_gather)
                    else:
                        torch.index_select(sim.rewards, 0, self.gather_indices,
                                           out=cur_rewards_store)

                        torch.index_select(sim.dones, 0, self.gather_indices,
                                           out=cur_dones_store)

                    for rnn_states in rnn_states_cur_in:
                        rnn_states.masked_fill_(sim.dones, 0)

                profile.gpu_measure(sync=True)

        if self.need_sim_copy:
            for sim_ob, policy_in in zip(sim.obs, self.obs_in):
                policy_in.copy_(sim_ob, non_blocking=True)

        # rnn_hidden_cur_in and rnn_hidden_cur_out are flipped after each
        # iter so rnn_hidden_cur_in is the final output
        self.rnn_end_states = rnn_states_cur_in
        self.rnn_alt_states = rnn_states_cur_out

        with amp.enable(), profile("Bootstrap Values"):
            # FIXME: this only needs to call the trained policy critics,
            # would eliminate second gather
            torch.vmap(fpolicy_critic, in_dims=self.vmap_in_dims_critic)(
                       policy_assignments,
                       self.values_out,
                       None,
                       self.rnn_end_states,
                       *self.obs_in,
                   )

            if self.need_sim_copy:
                torch.index_select(self.values_out, 0, self.gather_indices,
                                   out=self.bootstrap_values_gather)
                self.bootstrap_values.copy_(self.bootstrap_values_gather)
            else:
                torch.index_select(self.values_out, 0, self.gather_indices,
                                   out=self.bootstrap_values)

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
            rnn_start_states = self.rnn_start_states,
        )
