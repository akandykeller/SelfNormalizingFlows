import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from .flowlayer import FlowLayer


class Conv2dZero(nn.Module):
    """
    From https://github.com/ehoogeboom/emerging/blob/9545da2f87d5507a506b68e6f4a261086a4e2c47/tfops.py#L294
    """
    def __init__(self, in_channels, out_channels, bias=True, 
                 kernel_size=(3,3), stride=(1,1), padding=(1,1),
                 dilation=1, groups=1, logscale_factor=3):
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.logscale_factor = logscale_factor

        w_shape = (out_channels, in_channels, *self.kernel_size)
        w_init = nn.init.zeros_(torch.empty(w_shape))

        self.weight = nn.Parameter(w_init)

        zeros = torch.nn.init.zeros_(torch.empty(out_channels))
        self.bias = nn.Parameter(zeros) if bias else None

        # for ReZero Trick
        self.logs = nn.Parameter(zeros)

    def forward(self, input):
        output = F.conv2d(input, self.weight, self.bias, 
                          self.stride, self.padding, self.dilation, self.groups)
        output = output * torch.exp(self.logs * self.logscale_factor).view(1, -1, 1, 1)
        return output


class Coupling(FlowLayer):
    def __init__(self, input_size, width=512, n_context=None):
        super().__init__()
        self.n_channels = n_channels = input_size[0]
        self.half_channels = n_channels // 2
        self.width = width

        if n_context is not None:
            in_channels = self.half_channels + n_context
            self.uses_context = True
        else:
            in_channels = self.half_channels
            self.uses_context = False

        self.net = nn.Sequential(nn.Conv2d(in_channels, width,
                                           kernel_size=(3,3), padding=(1,1)),
                                 nn.ReLU(),
                                 nn.Conv2d(width, width, (1,1)),
                                 nn.ReLU(),
                                 Conv2dZero(width, n_channels))

    def get_xs_logs_t(self, x, context=None):
        assert (context is not None) == self.uses_context
        x1 = x[:, :self.half_channels, :, :]
        x2 = x[:, self.half_channels:, :, :]

        if context is not None:
            h = self.net(torch.cat([x1, context], dim=1))
        else:
            h = self.net(x1)
  
        h_s, t = h[:, ::2], h[:, 1::2]
  
        logs_range = 2.
        log_s = logs_range * torch.tanh(h_s / logs_range)

        return x1, x2, log_s, t

    def forward(self, input, context=None):
        x1, x2, log_s, t = self.get_xs_logs_t(input, context)

        z2 = x2 * torch.exp(log_s) + t
        input = torch.cat([x1, z2], dim=1)
 
        return input, log_s.flatten(start_dim=1).sum(-1)


    def reverse(self, input, context=None):
        x1, x2, log_s, t = self.get_xs_logs_t(input, context)

        z2 = (x2 - t) * torch.exp(-log_s)
        z = torch.cat([x1, z2], dim=1)

        return z

    def logdet(self, input, context=None):
        z, ldj = self.forward(input, context)
        return ldj
