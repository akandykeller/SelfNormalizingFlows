import torch
from .flowlayer import FlowLayer


def space_to_depth(x):
    xs = x.size()
    # Pick off every second element
    x = x.view(xs[0], xs[1], xs[2] // 2, 2, xs[3] // 2, 2)
    # Transpose picked elements next to channels.
    x = x.permute((0, 1, 3, 5, 2, 4)).contiguous()
    # Combine with channels.
    x = x.view(xs[0], xs[1] * 4, xs[2] // 2, xs[3] // 2)
    return x


def depth_to_space(x):
    xs = x.size()
    # Pick off elements from channels
    x = x.view(xs[0], xs[1] // 4, 2, 2, xs[2], xs[3])
    # Transpose picked elements next to HW dimensions.
    x = x.permute((0, 1, 4, 2, 5, 3)).contiguous()
    # Combine with HW dimensions.
    x = x.view(xs[0], xs[1] // 4, xs[2] * 2, xs[3] * 2)
    return x


class Squeeze(FlowLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input, context=None):
        return space_to_depth(input), self.logdet(input, context)

    def reverse(self, input, context=None):
        return depth_to_space(input)

    def logdet(self, input, context=None):
        return input.new_zeros(len(input))


class UnSqueeze(FlowLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input, context=None):
        return depth_to_space(input), self.logdet(input, context)

    def reverse(self, input, context=None):
        return space_to_depth(input)

    def logdet(self, input, context=None):
        return input.new_zeros(len(input))
