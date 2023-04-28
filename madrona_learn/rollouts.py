import torch

class RolloutBuffer:
    def __init__(self,
                 dev,
                 actions_shape,
                 actions_type,
                 rewards_shape,
                 steps_per_update):
        self.actions_store = torch.zeros((steps_per_update, *actions_shape),
            dtype=actions_type, device=dev)
        self.rewards_store = torch.zeros((steps_per_update, *rewards_shape),
            dtype=torch.float16, device=dev)

    def save(self, slot, actions, rewards):
        self.actions_store[slot].copy_(actions)
        self.rewards_store[slot].copy_(rewards)
