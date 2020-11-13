import torch
import torch.nn.functional as F
from .flowlayer import PreprocessingFlowLayer


class LogitTransform(PreprocessingFlowLayer):

    def __init__(self):
        super().__init__()

    def forward(self, input, context=None):
        return torch.log(input) - torch.log(1-input), self.logdet(input, context)

    def reverse(self, input, context=None):
        return torch.sigmoid(input)

    def logdet(self, input, context=None):
        return (-torch.log(input) - torch.log(1-input)).flatten(start_dim=1).sum(-1)


class SigmoidTransform(PreprocessingFlowLayer):

    def __init__(self):
        super().__init__()

    def forward(self, input, context=None):
        return torch.sigmoid(input), self.logdet(input, context)

    def reverse(self, input, context=None):
        return torch.log(input) - torch.log(1 - input)

    def logdet(self, input, context=None):
        log_derivative = F.logsigmoid(input) + F.logsigmoid(-input)
        return log_derivative.flatten(start_dim=1).sum(1)
