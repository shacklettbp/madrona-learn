import torch
import torch.nn as nn

from .amp import AMPState

# Based on tensorflow probability moving_stats.py

# Exponential Moving Average mean and variance estimator for
# values and observations
class EMANormalizer(nn.Module):
    def __init__(self, decay, eps=1e-5, disable=False):
        super().__init__()

        self.disable = disable
        if disable:
            return

        self.decay = decay
        self.one_minus_decay = 1 - decay
        self.eps = eps

        # Current parameter estimates
        self.register_buffer("mu", torch.zeros(1, dtype=torch.float32))
        self.register_buffer("inv_sigma", torch.zeros(1, dtype=torch.float32))
        self.register_buffer("sigma", torch.zeros(1, dtype=torch.float32))

        # Intermediate values used to compute the moving average
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

                self.mu_biased.add_(self.one_minus_decay *
                                    (x_f32.mean() - self.mu_biased))

                self.N.add_(1)

                # Note this diff is against unbiased mu which hasn't been
                # updated yet
                mu_diffs = x_f32 - self.mu
                sq_mu_diffs = mu_diffs * mu_diffs

                self.sigma_sq_biased.add_(
                    self.one_minus_decay * (
                        self.decay * sq_mu_diffs.mean() - self.sigma_sq_biased))

                bias_correction = 1.0 - self.decay ** self.N

                sigma_sq = self.sigma_sq_biased / bias_correction

                # Write out new unbiased params
                self.mu = self.mu_biased / bias_correction
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
