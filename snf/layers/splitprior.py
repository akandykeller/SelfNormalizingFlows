import torch

from .flowlayer import FlowLayer
from .coupling import Coupling


class SplitPrior(FlowLayer):
    def __init__(self, input_size, distribution, width=512):
        super().__init__()
        assert len(input_size) == 3
        self.n_channels = input_size[0]

        self.transform = Coupling(input_size, width=width)

        self.base = distribution(
            (self.n_channels // 2, input_size[1], input_size[2]))

    def forward(self, input, context=None):
        x, ldj = self.transform(input, context)

        x1 = x[:, :self.n_channels // 2, :, :]
        x2 = x[:, self.n_channels // 2:, :, :]

        log_pz2 = self.base.log_prob(x2)
        log_px2 = log_pz2 + ldj

        return x1, log_px2

    def reverse(self, input, context=None):
        x1 = input
        x2, log_px2 = self.base.sample(x1.shape[0], context)

        x = torch.cat([x1, x2], dim=1)
        x = self.transform.reverse(x, context)

        return x

    def logdet(self, input, context=None):
        x1, ldj = self.forward(input, context)
        return ldj


class SplitPriorFC(SplitPrior):
    def __init__(self, n_dims, distribution):
        assert type(n_dims) == int
        self.n_dims = n_dims
        self.half_dims = n_dims // 2
        input_size = (n_dims, 1, 1)
        super().__init__(input_size, distribution)

    def forward(self, input, context=None):
        input = input.view(-1, self.n_dims, 1, 1)
        output, ldj = super().forward(input, context)
        return output.view(-1, self.half_dims), ldj

    def reverse(self, input, context=None):
        input = input.view(-1, self.half_dims, 1, 1)
        output = super().reverse(input, context)
        return output.view(-1, self.n_dims)

    def logdet(self, input, context=None):
        input = input.view(-1, self.n_dims, 1, 1)
        return super().logdet(input)