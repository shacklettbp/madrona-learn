import torch
from torch import optim
from torch.func import vmap
from time import time

from .cfg import TrainConfig
from .rollouts import RolloutManager

def ppo_train(sim, cfg, policy, dev):
    num_agents = sim.actions.shape[0]

    policy = policy.to(dev)

    optimizer = optim.Adam(policy.parameters(), lr=cfg.lr)

    if dev.type == 'cuda':
        scaler = torch.cuda.amp.GradScaler()

    rollouts = RolloutManager(dev, sim, cfg.steps_per_update, cfg.gamma)

    values = torch.zeros((num_agents, 1),
                         dtype=torch.float16, device=dev)

    if dev.type == 'cuda':
        def policy_infer_fn(actions_out, *obs):
            with torch.cuda.amp.autocast(dtype=torch.float16):
                policy.infer(actions_out, *obs)
    else:
        def policy_infer_fn(actions_out, *obs):
            policy.infer(actions_out, *obs)

    for update_idx in range(cfg.num_updates):
        update_start_time = time()

        if update_idx % 1 == 0:
            print(f'Update: {update_idx}')

        with torch.no_grad():
            rollouts.collect(sim, policy_infer_fn)

        optimizer.zero_grad()

        update_end_time = time()

        print(update_end_time - update_start_time)

def train(sim, cfg, policy, dev):
    print(cfg)

    #train_compiled = torch.compile(ppo_train, dynamic=False)
    train_compiled = ppo_train

    train_compiled(sim, cfg, policy, dev)
