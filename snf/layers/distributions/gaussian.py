import torch
import torch.nn as nn
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal


class GaussianDistribution(nn.Module):
    """
    Standard Normal Likelihood
    """
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.dim = dim = int(np.prod(size))
        self.N = MultivariateNormal(torch.zeros(dim, device='cuda'),
                                    torch.eye(dim, device='cuda'))

    def forward(self, input, context=None):
        return self.log_prob(input, context).sum(-1)

    def log_prob(self, input, context=None, sum=True):
        return self.N.log_prob(input.view(-1, self.dim))

    def sample(self, n_samples, context=None):
        x = self.N.sample((n_samples,)).view(n_samples, *self.size)
        log_px = self.log_prob(x, context)
        return x, log_px
