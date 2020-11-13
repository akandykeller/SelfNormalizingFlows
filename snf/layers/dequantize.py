import torch

from .flowlayer import PreprocessingFlowLayer


class Dequantization(PreprocessingFlowLayer):
    def __init__(self, deq_distribution):
        super(Dequantization, self).__init__()
        # deq_distribution should be a distribution with support on [0, 1]^d
        self.distribution = deq_distribution

    def forward(self, input, context=None):

        # Note, input is the context for distribution model.
        noise, log_qnoise = self.distribution.sample(input.size(0), input.float())
        return input + noise, -log_qnoise

    def reverse(self, input, context=None):
        return input.floor()

    def logdet(self, input, context=None):
        raise NotImplementedError
