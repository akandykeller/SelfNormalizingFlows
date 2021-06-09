import torch
from .activations import FlowActivationLayer


class ActNorm(FlowActivationLayer):
    def __init__(self, n_dims):
        super().__init__()

        self.n_dims = n_dims
        self.translation = torch.nn.Parameter(torch.zeros(n_dims))
        self.log_scale = torch.nn.Parameter(torch.zeros(n_dims))
        self.register_buffer('initialized', torch.tensor(0))

    def forward(self, input, context=None):
        reduce_dims = [i for i in range(len(input.size())) if i != 1]

        if not self.initialized:
            with torch.no_grad():
                mean = torch.mean(input, dim=reduce_dims)
                log_stddev = torch.log(torch.std(input, dim=reduce_dims) + 1e-8)
                self.translation.data.copy_(mean)
                self.log_scale.data.copy_(log_stddev)
                self.initialized.fill_(1)

        if len(input.size()) == 4:
            _, C, H, W = input.size()
            translation = self.translation.view(1, C, 1, 1)
            log_scale = self.log_scale.view(1, C, 1, 1)
        else:
            _, D = input.size()
            translation = self.translation.view(1, D)
            log_scale = self.log_scale.view(1, D)

        out = (input - translation) * torch.exp(-log_scale)

        return out, self.logdet(input, context)

    def reverse(self, input, context=None):
        assert self.initialized

        if len(input.size()) == 4:
            _, C, H, W = input.size()
            translation = self.translation.view(1, C, 1, 1)
            log_scale = self.log_scale.view(1, C, 1, 1)
        else:
            _, D = input.size()
            translation = self.translation.view(1, D)
            log_scale = self.log_scale.view(1, D)

        output = input * torch.exp(log_scale) + translation
        return output

    def act_prime(self, input, context=None):
        return torch.exp(-self.log_scale)

    def logdet(self, input, context=None):
        B = input.size(0)
        if len(input.size()) == 2:
            ldj = -self.log_scale.sum().expand(B)
        elif len(input.size()) == 4:
            H, W = input.size()[2:]
            ldj = -self.log_scale.sum().expand(B) * H * W

        return ldj


class ActNormPlainLayer(ActNorm):
    def forward(self, *args, **kwargs):
        out, ldj = super().forward(*args, **kwargs)
        return out


class ActNormFC(ActNorm):
    def __init__(self, n_dims):
        super().__init__(n_dims)

    def forward(self, input, context=None):
        input = input.view(-1, self.n_dims, 1, 1)
        output, ldj = super().forward(input, context)
        return output.view(-1, self.n_dims), ldj

    def reverse(self, input, context=None):
        input = input.view(-1, self.n_dims, 1, 1)
        output = super().reverse(input, context)
        return output.view(-1, self.n_dims)

    def logdet(self, input, context=None):
        input = input.view(-1, self.n_dims, 1, 1)
        return super().logdet(input)