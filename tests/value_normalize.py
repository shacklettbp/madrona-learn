import torch
from madrona_learn.moving_avg import EMANormalizer
from madrona_learn.amp import AMPState
torch.manual_seed(0)

dev = torch.device('cuda')

decay = 0.99999
normalizer = EMANormalizer(decay)
normalizer = normalizer.to(dev)
normalizer.train()

batch_size = 1024

amp = AMPState(dev, True)

log = []
naive_mean = torch.zeros(1, device=dev, dtype=torch.float64)
naive_mean_sqs = torch.zeros(1, device=dev, dtype=torch.float64)
naive_decay = torch.tensor([decay], device=dev, dtype=torch.float64)
naive_one_minus_decay = 1 - naive_decay

num_iters = 10

means = torch.rand(num_iters) * 10 - 5
stddevs = torch.randn(num_iters) * 2

means[-1] = -20
stddevs[-1] = 0.01

for i in range(num_iters):
    print()
    mean = means[i]
    stddev = stddevs[i]

    #stddev = 2 * (1 + i)
    #if i == 3:
    #    stddev = 0.1

    values = torch.randn([batch_size, 1], device=dev) * stddev + mean
    naive_mean = naive_decay * naive_mean + naive_one_minus_decay * torch.mean(values.to(dtype=torch.float64))
    naive_mean_sqs = naive_decay * naive_mean_sqs + naive_one_minus_decay * torch.mean(values.to(dtype=torch.float64) * values.to(dtype=torch.float64))
    #naive_debias = 1 - naive_decay ** (i + 1)
    naive_debias = -torch.expm1((i + 1) * torch.log(naive_decay))

    log.append(values)
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
    print("Biased Params:", normalizer.mu_biased.cpu().item(), torch.sqrt(normalizer.sigma_sq_biased).cpu().item())
    print("Params:", normalizer.mu.cpu().item(), normalizer.sigma.cpu().item(), normalizer.inv_sigma.cpu().item(), normalizer.N.cpu().item())

    naive_mean_debiased = naive_mean / naive_debias
    naive_mean_sqs_debiased = naive_mean_sqs / naive_debias
    print("Naive: ", naive_mean_debiased.cpu().item(), torch.sqrt(naive_mean_sqs_debiased - naive_mean_debiased * naive_mean_debiased).cpu().item())
    print("Naive biased params: ", naive_mean.cpu().item(), naive_mean_sqs.cpu().item(), naive_debias.cpu().item())

print()
total_var, total_mean = torch.var_mean(torch.stack(log))
print(total_mean, torch.sqrt(total_var))
