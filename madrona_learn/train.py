import torch

from .cfg import TrainConfig
from .rollouts import RolloutBuffer

def train(step_fn, dev, actions, rewards, **kwargs):
    cfg = TrainConfig(**kwargs)

    rollouts = RolloutBuffer(dev, actions.shape, actions.dtype,
                             rewards.shape, cfg.steps_per_update)

    print(cfg)

    for epoch_idx in range(cfg.num_epochs):
        for rollout_slot in range(0, cfg.steps_per_update):
            step_fn()
            rollouts.save(rollout_slot, actions, rewards)
