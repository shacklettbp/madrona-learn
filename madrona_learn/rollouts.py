import torch
from time import time

class RolloutManager:
    def __init__(self, dev, sim, steps_per_update, gamma,
                 rnn_hidden_shape=None):
        self.need_obs_copy = dev != sim.obs[0].device

        self.actions = torch.zeros(
            (steps_per_update, *sim.actions.shape),
            dtype=sim.actions.dtype, device=dev)

        self.log_probs = torch.zeros(
            (steps_per_update, *sim.actions.shape),
            dtype=torch.float16, device=dev)

        self.dones = torch.zeros(
            (steps_per_update, *sim.dones.shape),
            dtype=torch.uint8, device=dev)

        self.rewards = torch.zeros(
            (steps_per_update, *sim.rewards.shape),
            dtype=torch.float16, device=dev)

        self.values = torch.zeros(
            (steps_per_update, *sim.rewards.shape),
            dtype=torch.float16, device=dev)

        # FIXME: seems like this could be combined into self.values by
        # making self.values one longer, but that breaks torch.compile
        self.bootstrap_values = torch.zeros(
            sim.rewards.shape, dtype=torch.float16, device=dev)

        self.returns = torch.zeros_like(self.rewards)

        self.obs = []

        obs_buffer_seq_len = steps_per_update

        if self.need_obs_copy:
            obs_buffer_seq_len += 1

        for obs_tensor in sim.obs:
            self.obs.append(torch.zeros(
                (obs_buffer_seq_len, *obs_tensor.shape),
                dtype=obs_tensor.dtype, device=dev))

        self.steps_per_update = steps_per_update
        self.gamma = gamma

        if rnn_hidden_shape != None:
            self.recurrent_policy = True

            hidden_batch_shape = (*rnn_hidden_shape[0:2],
                sim.actions.shape[0], rnn_hidden_shape[2])

            self.rnn_hidden_start = torch.zeros(
                hidden_batch_shape, dtype=torch.float16, device=dev)
            self.rnn_hidden_end = torch.zeros_like(self.rnn_hidden_start)
            self.rnn_hidden_alt = torch.zeros_like(self.rnn_hidden_start)
        else:
            self.recurrent_policy = False

    def collect(self, sim, policy_infer_fn, policy_infer_values_fn):
        step_total = 0

        if self.recurrent_policy:
            self.rnn_hidden_start.copy_(self.rnn_hidden_end)
            rnn_hidden_cur_in = self.rnn_hidden_end
            rnn_hidden_cur_out = self.rnn_hidden_alt

        for slot in range(0, self.steps_per_update):
            cur_obs_buffers = [obs[slot] for obs in self.obs]

            for obs_idx, step_obs in enumerate(sim.obs):
                cur_obs_buffers[obs_idx].copy_(step_obs, non_blocking=True)

            if self.recurrent_policy:
                policy_infer_fn(self.actions[slot], self.log_probs[slot],
                                self.values[slot], rnn_hidden_cur_out,
                                rnn_hidden_cur_in, *cur_obs_buffers)

                rnn_hidden_cur_in, rnn_hidden_cur_out = \
                    rnn_hidden_cur_out, rnn_hidden_cur_in
            else:
                policy_infer_fn(self.actions[slot], self.log_probs[slot],
                                self.values[slot], *cur_obs_buffers)

            sim.actions.copy_(self.actions[slot], non_blocking=True)

            step_start_time = time()
            sim.step()
            step_total += time() - step_start_time

            self.dones[slot].copy_(sim.dones, non_blocking=True)
            self.rewards[slot].copy_(sim.rewards, non_blocking=True)

        if self.need_obs_copy:
            final_obs = [obs[self.steps_per_update] for obs in self.obs]
            for obs_idx, step_obs in enumerate(sim.obs):
                final_obs[obs_idx].copy_(step_obs, non_blocking=True)
        else:
            final_obs = sim.obs

        if self.recurrent_policy:
            # rnn_hidden_cur_in and rnn_hidden_cur_out are flipped after each
            # iter so rnn_hidden_cur_in is the final output
            self.rnn_hidden_end = rnn_hidden_cur_in
            self.rnn_hidden_alt = rnn_hidden_cur_out

            policy_infer_values_fn(self.bootstrap_values, self.rnn_hidden_end,
                                   *final_obs)
        else:
            policy_infer_values_fn(self.bootstrap_values, *final_obs)

        self._compute_returns()

    def _compute_returns(self):
        discounted_sum = self.bootstrap_values

        for i in reversed(range(self.steps_per_update)):
            discounted_sum = self.rewards[i] + \
                self.gamma * (1.0 - self.dones[i].half()) * discounted_sum

            self.returns[i] = discounted_sum
