import torch
import torch.nn as nn
import numpy as np


class UniformDistribution(nn.Module):
    """
    Standard Normal Likelihood
    """
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.dim = int(np.prod(size))
        self.register_buffer('empty', torch.zeros(1))

    def forward(self, input, context=None):
        return self.log_prob(input, context)

    def log_prob(self, input, context=None):
        above_0 = input >= 0
        below_1 = input <= 1.

        minus_infty = -1e30

        log_px = torch.where(
            above_0 & below_1,
            torch.zeros_like(input),
            torch.ones_like(input) * minus_infty)

        log_px = log_px.view(log_px.size(0), self.dim).sum(-1)

        return log_px

    def sample(self, n_samples, context=None):
        x = torch.rand((n_samples, *self.size)).to(device=self.empty.device)
        log_px = 0.
        return x, log_px
