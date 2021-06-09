import torch
import numpy as np
import torch.nn.functional as F

from .flowlayer import FlowLayer
from .splines.rational_quadratic import unconstrained_rational_quadratic_spline


class FlowActivationLayer(FlowLayer):
    def __init__(self):
        super().__init__()
        self._last_logdet_value = None

    def forward(self, input, context=None):
        act = self.activation(input, context)
        return act, self.logdet(input, context)

    def act_prime(self, input, context=None):
        raise NotImplementedError()

    def logdet(self, input, context=None):
        logderiv = torch.log(torch.abs(self.act_prime(input, context)))
        return logderiv.flatten(start_dim=1).sum(dim=-1)


def newton_raphson_inverse(f, y, x0, context=None, n_iter=100):

    x = x0
    for _ in range(n_iter):
        fprime = torch.clamp(f.act_prime(x, context), min=1e-2)
        x = x - (f.activation(x, context) - y) / fprime

    return x


class SmoothLeakyRelu(FlowActivationLayer):

    def __init__(self, alpha=0.3):
        super().__init__()
        self.alpha = alpha

    def activation(self, input, context=None):
        alpha = self.alpha

        stacked = torch.stack((torch.zeros_like(input), input))
        return alpha * input + (1-alpha) * torch.logsumexp(stacked, dim=0)

    def act_prime(self, input, context=None):
        alpha = self.alpha
        return alpha + (1-alpha) * torch.sigmoid(input)

    def reverse(self, input, context=None):
        y, x0 = input, input
        return newton_raphson_inverse(self, y, x0, context)


class LeakyRelu(FlowActivationLayer):

    def __init__(self, alpha=0.1):
        super().__init__()
        self.alpha = alpha

    def activation(self, input, context=None):
        out = torch.where(input < 0, 
                            self.alpha * input, 
                            input)
        return out

    def act_prime(self, input, context=None):
        derivative = torch.where(input < 0, 
                                    self.alpha, 
                                    1.0)
        return derivative

    def reverse(self, input, context=None):
        out = torch.where(input < 0, 
                            input / self.alpha, 
                            input)
        return out


class LearnableLeakyRelu(FlowActivationLayer):

    def __init__(self):
        super().__init__()

        self.alpha_logit = torch.nn.Parameter(torch.zeros([1]))

    def get_alpha(self):
        return torch.sigmoid(self.alpha_logit) + .5

    def activation(self, input, context=None):
        alpha = self.get_alpha()
        out = torch.where(input < 0, alpha * input, input)
        return out

    def act_prime(self, input, context=None):
        alpha = self.get_alpha()
        derivative = torch.where(input < 0, alpha, torch.ones_like(alpha))
        return derivative

    def reverse(self, input, context=None):
        alpha = self.get_alpha()
        out = torch.where(input < 0, input / alpha, input)
        return out


class SmoothTanh(FlowActivationLayer):

    def __init__(self, alpha=1.0, beta=0.1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def activation(self, input, context=None):
        return torch.tanh(self.alpha * input) + self.beta * input

    def act_prime(self, input, context=None):
        return self.beta + self.alpha / torch.pow(torch.cosh(self.alpha * input), 2)

    def reverse(self, input, context=None):
        y, x0 = input, input
        return newton_raphson_inverse(self, y, x0, context)


class SplineActivation(FlowActivationLayer):
    def __init__(self, input_size, n_bins=5, tail_bound=10., individual_weights=False):
        super().__init__()

        self.n_bins = n_bins
        self.tail_bound = tail_bound
        self.individual_weights = individual_weights

        if individual_weights:
            self.unnormalized_widths = torch.nn.Parameter(
                torch.randn(1, *input_size, n_bins) * 0.01)
            self.unnormalized_heights = torch.nn.Parameter(
                torch.randn(1, *input_size, n_bins) * 0.01)
            self.unnormalized_derivatives = torch.nn.Parameter(
                torch.randn(1, *input_size, n_bins - 1) * 0.01)
        else:
            self.unnormalized_widths = torch.nn.Parameter(torch.randn(n_bins) * 0.01)
            self.unnormalized_heights = torch.nn.Parameter(torch.randn(n_bins) * 0.01)
            self.unnormalized_derivatives = torch.nn.Parameter(torch.randn(n_bins - 1) * 0.01)

        # self.f1(nn_input).reshape(B, C, -1, H, W).permute(0, 1, 3, 4, 2)

    def forward(self, input, context=None):
        """
        Override forward to store the logdet more cheaply.
        """
        act, ldj = self.activation_and_logdet(input, context)
        return act, ldj

    def get_spline_params(self, input):
        if self.individual_weights:
            repeats = (input.size(0),) + (1,) * len(input.size())
            unnormalized_widths = self.unnormalized_widths.repeat(repeats)
            unnormalized_heights = self.unnormalized_heights.repeat(repeats)
            unnormalized_derivatives = self.unnormalized_derivatives.repeat(repeats)
        else:
            # Create a tuple with (B, ..., 1) with dimensions of input.
            repeats = input.size() + (1,)

            # Create ones tuple to reshape.
            ones = (1,) * len(input.size())

            unnormalized_widths = self.unnormalized_widths.view(*ones, -1).repeat(repeats)
            unnormalized_heights = self.unnormalized_heights.view(*ones, -1).repeat(repeats)
            unnormalized_derivatives = self.unnormalized_derivatives.view(*ones, -1).repeat(repeats)

        return unnormalized_widths, unnormalized_heights, \
            unnormalized_derivatives

    def activation_and_logdet(self, input, context=None):
        unnormalized_widths, unnormalized_heights, unnormalized_derivatives = \
            self.get_spline_params(input)

        out, ld = unconstrained_rational_quadratic_spline(
            input,
            unnormalized_widths,
            unnormalized_heights,
            unnormalized_derivatives,
            inverse=False,
            tails='linear',
            tail_bound=self.tail_bound)

        return out, ld.flatten(start_dim=1).sum(dim=-1)

    def reverse(self, input, context=None):
        unnormalized_widths, unnormalized_heights, unnormalized_derivatives = \
            self.get_spline_params(input)

        out, ld = unconstrained_rational_quadratic_spline(
            input,
            unnormalized_widths,
            unnormalized_heights,
            unnormalized_derivatives,
            inverse=True,
            tails='linear',
            tail_bound=self.tail_bound)
        return out

    def logdet(self, input, context=None):
        unnormalized_widths, unnormalized_heights, unnormalized_derivatives = \
            self.get_spline_params(input)

        out, ld = unconstrained_rational_quadratic_spline(
            input,
            unnormalized_widths,
            unnormalized_heights,
            unnormalized_derivatives,
            inverse=False,
            tails='linear',
            tail_bound=self.tail_bound)

        return ld.flatten(start_dim=1).sum(dim=-1)


class Identity(FlowActivationLayer):

    def __init__(self):
        super().__init__()

    def activation(self, input, context=None):
        return input

    def act_prime(self, input, context=None):
        return torch.ones_like(input)

    def reverse(self, input, context=None):
        return input