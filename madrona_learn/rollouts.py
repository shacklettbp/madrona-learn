import torch

class RolloutManager:
    def __init__(self, dev, sim, steps_per_update, gamma):
        self.actions = torch.zeros(
            (steps_per_update, *sim.actions.shape),
            dtype=actions_type, device=dev)

        self.dones = torch.zeros(
            (steps_per_update, *sim.dones.shape),
            dtype=torch.uint8, device=dev)

        self.rewards = torch.zeros(
            (steps_per_update, *sim.rewards.shape),
            dtype=torch.float16, device=dev)

        self.values = torch.zeros(
            (steps_per_update, *sim.rewards.shape),
            dtype=torch.float16, device=dev)

        self.returns = torch.zeros(
            (steps_per_update, *sim.rewards.shape),
            dtype=torch.float16, device=dev)

        self.obs = []

        for obs_tensor in sim.obs:
            self.obs.append(torch.zeros((steps_per_update, *obs_tensor.shape),
                                        dtype=obs_tensor.dtype, device=dev))


        self.steps_per_update = steps_per_update
        self.gamma = gamma

    def collect(self, sim, policy):
        with torch.no_grad():
            step_total = 0
            for slot in range(0, self.steps_per_update):
                step_start_time = time()

                sim.step()
                processed_obs = sim.process_obs(*sim.obs)
                policy.forward_inplace(processed_obs, sim.actions,
                                       self.values[slot])

                step_total += time() - step_start_time

                self.actions[slot].copy_(sim.actions)
                self.dones[slot].copy_(sim.dones)
                self.rewards[slot].copy_(sim.rewards)

                for obs_idx, step_obs in enumerate(sim.obs):
                    self.obs[obs_idx][slot].copy_(step_obs)

            self._compute_returns()

    def _compute_returns(self):
        discounted_sum = self.values[-1]

        for i in reversed(range(self.steps_per_update)):
            discounted_sum = rewards[i] + \
                self.gamma * (1.0 - dones[i].half()) * discounted_sum

            self.returns[i] = discounted_sum
