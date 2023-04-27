import torch

def load_cfg(**kwargs):
    return kwargs

def train(sim, **kwargs):
    cfg = load_cfg(kwargs)
    print(cfg)
