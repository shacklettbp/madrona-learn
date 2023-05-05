import torch
from torch import optim
from torch.func import vmap
from time import time

from .cfg import TrainConfig, SimData
from .rollouts import RolloutManager

def _update_loop(cfg : TrainConfig,
                 sim : SimData,
                 rollouts : RolloutManager,
                 policy,
                 policy_infer_fn,
                 optimizer,
                 scaler):
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

    num_agents = sim.actions.shape[0]

    policy = policy.to(dev)

    optimizer = optim.Adam(policy.parameters(), lr=cfg.lr)

    if dev.type == 'cuda':
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    rollouts = RolloutManager(dev, sim, cfg.steps_per_update, cfg.gamma,
                              policy.rnn_hidden_shape)

    if dev.type == 'cuda':
        def policy_infer_wrapper(*args, **kwargs):
            with torch.cuda.amp.autocast(dtype=torch.float16):
                policy.rollout_infer(*args, **kwargs)
    else:
        def policy_infer_wrapper(*args, **kwargs):
            policy.rollout_infer(*args, **kwargs)

    #update_loop = torch.compile(_update_loop, dynamic=False)
    update_loop = _update_loop

    update_loop(
        cfg=cfg,
        sim=sim,
        rollouts=rollouts,
        policy=policy,
        policy_infer_fn=policy_infer_wrapper,
        optimizer=optimizer,
        scaler=scaler)
