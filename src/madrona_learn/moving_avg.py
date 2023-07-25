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
        self.register_buffer("mu", torch.zeros([], dtype=torch.float32))
        self.register_buffer("inv_sigma", torch.zeros([], dtype=torch.float32))
        self.register_buffer("sigma", torch.zeros([], dtype=torch.float32))

        # Intermediate values used to compute the moving average
        # decay and one_minus_decay don't strictly need to be tensors, but it's
        # critically important for floating point precision that
        # one_minus_decay is computed in fp32 rather than fp64 to 
        # match the bias_correction computation below
        self.register_buffer("decay",
                             torch.tensor(decay, dtype=torch.float32))
        self.register_buffer("one_minus_decay", 1 - self.decay)

        self.register_buffer("mu_biased",
                             torch.zeros([], dtype=torch.float32))
        self.register_buffer("sigma_sq_biased",
                             torch.zeros([], dtype=torch.float32))
        self.register_buffer("N",
                             torch.zeros([], dtype=torch.int64))

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
                x_sigma_sq, x_mu = torch.var_mean(x_f32)

                self.N.add_(1)
                bias_correction = -torch.expm1(self.N * torch.log(self.decay))

                self.mu_biased.add_(self.one_minus_decay *
                                    (x_mu - self.mu_biased))

                new_mu = self.mu_biased / bias_correction

                # prev_mu needs to be unbiased (bias_correction only accounts
                # for the initial EMA with 0), since otherwise variance would
                # be off by a squared factor.
                # On the first iteration, simply zero out the delta term
                # since there is no previous unbiased mean
                if self.N == 1:
                    prev_mu = x_mu
                else:
                    prev_mu = self.mu

                delta = new_mu - prev_mu
                print(prev_mu, x_mu, delta)
                print(x_sigma_sq, torch.sqrt(x_sigma_sq))
                print("delta", delta)
                print("delta * delta", delta * delta * self.decay * self.one_minus_decay)
                print("M_b", self.one_minus_decay * x_sigma_sq)
                print("M_a", self.decay * self.sigma_sq_biased)

                # The below code is Chan's algorithm for combining the
                # variance of two sets, with the sample sizes replaced with
                # the EMA weights. M2 = decay * M_a + (1 - decay) * M_b +
                # delta * delta * (1 - decay) * decay
                # The algorithm has been rearranged to reduce operations.

                #x_sigma_sq.addcmul_(
                #    delta,
                #    delta * self.decay
                #).sub_(self.sigma_sq_biased)

                #self.sigma_sq_biased.addcmul_(
                #    x_sigma_sq,
                #    self.one_minus_decay)

                M_a = self.decay * self.sigma_sq_biased
                M_b = self.one_minus_decay * x_sigma_sq
                delta_squared = delta * delta * self.decay * self.one_minus_decay

                self.sigma_sq_biased = M_a + M_b + delta_squared

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
