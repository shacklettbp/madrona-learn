import torch
from madrona_learn.moving_avg import EMANormalizer
from madrona_learn.amp import AMPState
torch.manual_seed(0)

dev = torch.device('cuda')

normalizer = EMANormalizer(0.99999)
normalizer = normalizer.to(dev)
normalizer.train()

batch_size = 1024

amp = AMPState(dev, True)

for i in range(10):
    print()
    values = torch.randn([batch_size, 1], device=dev) * 150 + 200000
    pre_var, pre_mean = torch.var_mean(values)
    pre_std = torch.sqrt(pre_var)
    pre_min = torch.min(values)
    pre_max = torch.max(values)

    normalized = normalizer(amp, values)
    post_var, post_mean = torch.var_mean(normalized)
    post_std = torch.sqrt(post_var)
    post_min = torch.min(normalized)
    post_max = torch.max(normalized)

    print("Pre:", pre_mean.cpu().item(), pre_std.cpu().item(), pre_min.cpu().item(), pre_max.cpu().item())
    print("Post:", post_mean.cpu().item(), post_std.cpu().item(), post_min.cpu().item(), post_max.cpu().item())
    print("Params:", normalizer.mu.cpu().item(), normalizer.sigma.cpu().item(), normalizer.inv_sigma.cpu().item(), normalizer.N.cpu().item())
    print(normalizer.mu_biased.cpu().item(), torch.sqrt(normalizer.sigma_sq_biased).cpu().item())
