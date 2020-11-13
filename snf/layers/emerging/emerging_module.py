import torch
import numpy as np
import torch.nn.functional as F
from snf.layers.flowlayer import FlowLayer
from snf.layers.conv1x1 import Conv1x1
from snf.layers.emerging.masks import get_conv_square_ar_mask
from snf.layers.emerging.inverse_triang_conv import Inverse


class SquareAutoRegressiveConv2d(FlowLayer):
    def __init__(self, n_channels):
        super(SquareAutoRegressiveConv2d, self).__init__()
        self.n_channels = n_channels
        kernel_size = [n_channels, n_channels, 2, 2]

        weight = torch.randn(kernel_size) / np.sqrt(np.prod(kernel_size))
        weight[torch.arange(n_channels), torch.arange(n_channels), -1, -1] += 1.
        self.weight = torch.nn.Parameter(weight)
        self.bias = torch.nn.Parameter(torch.zeros(n_channels))

        mask_np = get_conv_square_ar_mask(n_channels, n_channels, 2, 2)
        self.register_buffer('mask', torch.from_numpy(mask_np))

        self.inverse_op = Inverse()

    def delta_ldj(self, x):
        log_abs_diag = torch.log(torch.abs(self.weight[torch.arange(
            self.n_channels), torch.arange(self.n_channels), -1, -1]))

        delta_ldj = torch.sum(log_abs_diag) * x.size(2) * x.size(3)

        return delta_ldj.expand(x.size(0))

    def forward(self, x, context=None):
        weight = self.weight * self.mask
        z = F.conv2d(x, weight, self.bias, stride=1, padding=1)

        # Slice off last dimensions.
        z = z[:, :, :-1, :-1]

        delta_ldj = self.delta_ldj(x)

        return z, delta_ldj

    def reverse(self, z, context=None):
        weight = self.weight * self.mask

        bias = self.bias.view(1, self.n_channels, 1, 1)

        with torch.no_grad():
            x_np = self.inverse_op(
                z.detach().cpu().numpy(),
                weight.detach().cpu().numpy(),
                bias.detach().cpu().numpy())
            x = torch.from_numpy(x_np).to(z.device, z.dtype)

        return x

    def logdet(self, input, context=None):
        raise ValueError('Should not be called.')
    

class Flip2d(FlowLayer):
    def __init__(self):
        super(Flip2d, self).__init__()

    def forward(self, x, context=None):
        height = x.size(2)
        width = x.size(3)

        x = x[:, :, torch.arange(height-1, -1, -1)]
        x = x[:, :, :, torch.arange(width - 1, -1, -1)]

        return x, 0.

    def reverse(self, x, context=None):
        height = x.size(2)
        width = x.size(3)

        x = x[:, :, torch.arange(height - 1, -1, -1)]
        x = x[:, :, :, torch.arange(width - 1, -1, -1)]

        return x

    def logdet(self, input, context=None):
        raise ValueError('Should not be called.')


class Emerging(FlowLayer):
    def __init__(self, n_channels):
        super(Emerging, self).__init__()

        self.transformations = torch.nn.ModuleList([
            Conv1x1(n_channels),
            SquareAutoRegressiveConv2d(n_channels),
            Flip2d(),
            SquareAutoRegressiveConv2d(n_channels),
            Flip2d(),
        ])

    def forward(self, x, context=None):
        logdet = x.new_zeros(size=(x.size(0),))
        for transform in self.transformations:
            x, layer_logdet = transform(x, context)
            logdet = logdet + layer_logdet

        return x, logdet

    def reverse(self, x, context=None):
        for transform in reversed(self.transformations):
            x = transform.reverse(x, context)

        return x

    def logdet(self, input, context=None):
        raise ValueError('Should not be called.')


if __name__ == '__main__':
    x = torch.randn(1, 8, 4, 4)
    ldj = torch.zeros(8)
    layer = Emerging(8)

    z, _ = layer(x, None)

    x_recon = layer.reverse(z, None)

    print(torch.mean((x - x_recon)**2))