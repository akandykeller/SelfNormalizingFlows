import torch
import torch.nn as nn
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal


class ConditionalDistributionWithContext(nn.Module):
    def __init__(self, distribution, model_context):
        super().__init__()
        self.distribution = distribution
        self.model_context = model_context

    def forward(self, input, context=None):
        return self.log_prob(input, context)

    def log_prob(self, input, context=None):
        h_context = self.model_context(context)
        return self.distribution.log_prob(input, h_context)

    def sample(self, n_samples, context=None):
        h_context = self.model_context(context)
        x, log_px = self.distribution.sample(n_samples, h_context)
        return x, log_px
