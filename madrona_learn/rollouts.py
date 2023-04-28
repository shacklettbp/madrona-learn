import torch

class RolloutBuffer:
    def __init__(self,
                 dev,
                 num_agents,
                 action_shape,
                 actions_type,
                 steps_per_update):
        self.actions = torch.zeros(
            (steps_per_update, num_agents, *action_shape),
            dtype=actions_type, device=dev)
        self.rewards = torch.zeros(
            (steps_per_update, num_agents, 1),
            dtype=torch.float16, device=dev)
        self.dones = torch.zeros(
                (steps_per_update, num_agents, 1),
            dtype=torch.float16, device=dev)
        self.values = torch.zeros(
            (steps_per_update, num_agents, 1),
            dtype=torch.float16, device=dev)

    def save(self, slot, step_actions, step_rewards, step_dones, step_values):
        self.actions[slot].copy_(step_actions)
        self.rewards[slot].copy_(step_rewards)
        self.rewards[slot].copy_(step_dones)
        self.values[slot].copy_(step_values)
