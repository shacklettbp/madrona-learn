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
                 policy_rollout_infer,
                 policy_rollout_infer_values,
                 optimizer,
                 scaler):
    for update_idx in range(cfg.num_updates):
        update_start_time = time()

        if update_idx % 1 == 0:
            print(f'Update: {update_idx}')

        with torch.no_grad():
            rollouts.collect(sim, policy_rollout_infer,
                             policy_rollout_infer_values)

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

    if dev.type == 'cuda':
        def autocast_wrapper(fn):
            def autocast_wrapped_fn(*args, **kwargs):
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    return fn(*args, **kwargs)

            return autocast_wrapped_fn
    else:
        def autocast_wrapper(fn):
            return fn

    def policy_rollout_infer(*args, **kwargs):
        return policy.rollout_infer(*args, **kwargs)

    def policy_rollout_infer_values(*args, **kwargs):
        return policy.rollout_infer_values(*args, **kwargs)

    policy_rollout_infer = autocast_wrapper(policy_rollout_infer)
    policy_rollout_infer_values = autocast_wrapper(policy_rollout_infer_values)

    rollouts = RolloutManager(dev, sim, cfg.steps_per_update, cfg.gamma,
                              policy.rnn_hidden_shape)

    #update_loop = torch.compile(_update_loop, dynamic=False)
    update_loop = _update_loop

    update_loop(
        cfg=cfg,
        sim=sim,
        rollouts=rollouts,
        policy=policy,
        policy_rollout_infer=policy_rollout_infer,
        policy_rollout_infer_values=policy_rollout_infer_values,
        optimizer=optimizer,
        scaler=scaler)
