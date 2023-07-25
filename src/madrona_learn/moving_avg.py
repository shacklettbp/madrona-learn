import torch
import torch.nn as nn

from .amp import AMPState

# Exponential Moving Average mean and variance estimator for
# values and observations
class EMANormalizer(nn.Module):
    def __init__(self, decay, eps=1e-5, disable=False):
        super().__init__()

        self.disable = disable
        if disable:
            return

        self.eps = eps

        # Current parameter estimates
        self.register_buffer("mu", torch.zeros(1, dtype=torch.float32))
        self.register_buffer("inv_sigma", torch.zeros(1, dtype=torch.float32))
        self.register_buffer("sigma", torch.zeros(1, dtype=torch.float32))

        # Intermediate values used to compute the moving average
        # decay and one_minus_decay don't strictly need to be tensors, but it's
        # critically important for floating point precision that
        # one_minus_decay is computed in fp32 rather than fp64 to 
        # match the bias_correction computation below
        self.register_buffer("decay",
                             torch.tensor([decay], dtype=torch.float32))
        self.register_buffer("one_minus_decay", 1 - self.decay)

        self.register_buffer("mu_biased",
                             torch.zeros(1, dtype=torch.float32))
        self.register_buffer("sigma_sq_biased",
                             torch.zeros(1, dtype=torch.float32))
        self.register_buffer("N",
                             torch.zeros(1, dtype=torch.int64))

        nn.init.constant_(self.mu , 0)
        nn.init.constant_(self.inv_sigma, 0)
        nn.init.constant_(self.sigma, 0)

        nn.init.constant_(self.mu_biased, 0)
        nn.init.constant_(self.sigma_sq_biased, 0)
        nn.init.constant_(self.N, 0)

    def forward(self, amp, x):
        if self.disable:
            return x 

        with amp.disable():
            if self.training:
                x_f32 = x.to(dtype=torch.float32)
                x_mean = x_f32.mean()

                self.N.add_(1)
                bias_correction = -torch.expm1(self.N * torch.log(self.decay))

                self.mu_biased.add_(self.one_minus_decay *
                                    (x_f32.mean() - self.mu_biased))


                new_mu = self.mu_biased / bias_correction

                # Running variance estimate with Welford's algorithm
                # adapted to EMA. Need this hack for N == 1 as otherwise
                # the first estimate of variance is biased by the incorrect
                # self.mu (in addition to the bias from the EMA with 0)
                if self.N == 1:
                    var_contrib = x_f32 - new_mu
                    var_contrib = torch.mean(var_contrib * var_contrib)
                else:
                    var_contrib = torch.mean((x_f32 - self.mu) * (x_f32 - new_mu))

                print(var_contrib, new_mu, x_mean, self.mu_biased)
                self.sigma_sq_biased.add_(self.one_minus_decay * var_contrib)

                sigma_sq = self.sigma_sq_biased / bias_correction

                # Write out new unbiased params
                self.mu = new_mu
                self.inv_sigma = torch.rsqrt(torch.clamp(sigma_sq, min=self.eps))
                self.sigma = torch.reciprocal(self.inv_sigma)

            return torch.addcmul(
                -self.mu * self.inv_sigma,
                x,
                self.inv_sigma,
            ).to(dtype=x.dtype)

    def invert(self, amp, normalized_x):
        if self.disable:
            return normalized_x

        with amp.disable():
            return torch.addcmul(
                self.mu,
                normalized_x.to(dtype=torch.float32),
                self.sigma,
            ).to(dtype=normalized_x.dtype)
