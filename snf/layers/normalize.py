import torch

from .flowlayer import PreprocessingFlowLayer


class Normalization(PreprocessingFlowLayer):

    def __init__(self, translation, scale, learnable=False):
        super().__init__()

        if learnable:
            self.translation = torch.nn.Parameter(torch.Tensor([translation]))
            self.scale = torch.nn.Parameter(torch.Tensor([scale]))
        else:
            self.register_buffer('translation', torch.Tensor([translation]))
            self.register_buffer('scale', torch.Tensor([scale]))

    def forward(self, input, context=None):
        return (input - self.translation) / self.scale, \
               self.logdet(input, context)

    def reverse(self, input, context=None):
        return (input * self.scale) + self.translation

    def logdet(self, input, context=None):
        N, C, H, W = input.size()
        logdet = -C * H * W * torch.log(self.scale)
        return logdet.expand(N)
