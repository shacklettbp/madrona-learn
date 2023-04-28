import torch
from torch.func import vmap
from time import time

from .cfg import TrainConfig
from .rollouts import RolloutBuffer

@torch.compile(dynamic=False, fullgraph=True)
def compute_returns(rewards, dones, final_values, returns, gamma, num_steps):
    discounted_sum = final_values
    for i in reversed(range(num_steps)):
        discounted_sum = \
            rewards[i] + gamma * (1.0 - dones[i].half()) * discounted_sum
        returns[i] = discounted_sum

def train(sim, cfg, dev):
    print(cfg)

    num_agents = sim.actions.shape[0]

    rollouts = RolloutBuffer(dev, num_agents,
                             sim.actions.shape[1:], sim.actions.dtype,
                             cfg.steps_per_update)

    returns = torch.zeros((cfg.steps_per_update, num_agents, 1),
                          dtype=torch.float16, device=dev)

    values = torch.zeros((num_agents, 1),
                         dtype=torch.float16, device=dev)

    # Force compile of compute_returns
    compute_returns(rollouts.rewards, rollouts.dones, rollouts.values[-1],
                    returns, cfg.gamma, cfg.steps_per_update)

    for update_idx in range(cfg.num_updates):
        update_start_time = time()

        if update_idx % 1 == 0:
            print(f'Update: {update_idx}')

        step_total = 0
        for rollout_slot in range(0, cfg.steps_per_update):
            step_start_time = time()

            sim.step()

            step_total += time() - step_start_time

            rollouts.save(rollout_slot, sim.actions, sim.rewards,
                          sim.dones, values)

        compute_returns(rollouts.rewards, rollouts.dones, rollouts.values[-1],
                        returns, cfg.gamma, cfg.steps_per_update)

        update_end_time = time()

        print(update_end_time - update_start_time,
              step_total)
