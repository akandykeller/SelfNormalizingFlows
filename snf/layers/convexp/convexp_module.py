import torch
import snf.layers.convexp.functional as F_convexp
from snf.layers.flowlayer import FlowLayer
from snf.layers.conv1x1 import Conv1x1, Conv1x1Householder
from .spectral import spectral_norm_conv
import numpy as np


class ConvExp(FlowLayer):
    def __init__(self, input_size, convexp_coeff=0.9):
        super(ConvExp, self).__init__()
        kernel_size = [input_size[0], input_size[0], 3, 3]

        self.kernel = torch.nn.Parameter(
            torch.randn(kernel_size) / np.prod(kernel_size[1:]))

        self.stride = (1, 1)
        self.padding = (1, 1)

        # Probably not useful.
        self.pre_transform_bias = torch.nn.Parameter(
            torch.zeros((1, *input_size))
        )

        # Again probably not useful.
        self.post_transform_bias = torch.nn.Parameter(
            torch.zeros((1, *input_size))
        )

        if input_size[0] <= 64:
            self.conv1x1 = Conv1x1(input_size[0])
        else:
            self.conv1x1 = Conv1x1Householder(input_size[0], 64)

        spectral_norm_conv(
            self, coeff=convexp_coeff, input_dim=input_size, name='kernel',
            n_power_iterations=1, eps=1e-12)

        self.n_terms_train = 6
        self.n_terms_eval = self.n_terms_train * 2 + 1

    def forward(self, x, context, reverse=False):
        B, C, H, W = x.size()
        logdet = 0.
        kernel = self.kernel

        n_terms = self.n_terms_train if self.training else self.n_terms_eval

        if not reverse:
            x = x + self.pre_transform_bias
            if hasattr(self, 'conv1x1'):
                x, logdet = self.conv1x1(x, context)

            z = F_convexp.conv_exp(x, kernel, terms=n_terms)
            logdet = logdet + F_convexp.log_det(kernel) * H * W

            z = z + self.post_transform_bias
        else:
            x = x - self.post_transform_bias
            if x.device != kernel.device:
                print('Warning, x.device is not kernel.device')
                kernel = kernel.to(device=x.device)

            z = F_convexp.inv_conv_exp(x, kernel, terms=n_terms, verbose=True)

            if hasattr(self, 'conv1x1'):
                z = self.conv1x1.reverse(z, context)

            z = z - self.pre_transform_bias
        return z, logdet

    def reverse(self, x, context):
        # For this particular reverse it is important that forward is called,
        # as it activates the pre-forward hook for spectral normalization.
        # This situation occurs when a flow is used to sample, for instance
        # in the case of variational dequantization.
        return self(x, context, reverse=True)[0]

    def logdet(self, input, context=None):
        raise ValueError('Should not be called')
