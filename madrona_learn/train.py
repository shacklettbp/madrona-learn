import torch
from torch.func import vmap
from time import time

from .cfg import TrainConfig
from .rollouts import RolloutManager


def ppo_train(sim, cfg, policy, dev):
    num_agents = sim.actions.shape[0]

    rollouts = RolloutManager(dev, sim, cfg.steps_per_update, cfg.gamma)

    values = torch.zeros((num_agents, 1),
                         dtype=torch.float16, device=dev)

    # Force compile of compute_returns
    compute_returns(rollouts.rewards, rollouts.dones, rollouts.values[-1],
                    returns, cfg.gamma, cfg.steps_per_update)

    for update_idx in range(cfg.num_updates):
        update_start_time = time()

        if update_idx % 1 == 0:
            print(f'Update: {update_idx}')

        rollouts.collect(sim, policy)

        update_end_time = time()

        print(update_end_time - update_start_time,
              step_total)

def train(sim, cfg, policy, dev):
    print(cfg)

    train_compiled = torch.compile(ppo_train, dynamic=False)

    train_compiled(sim, cfg, policy, dev)
