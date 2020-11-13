import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .flowlayer import FlowLayer


class Conv1x1(FlowLayer):
    def __init__(self, n_channels):
        super().__init__()
        self.n_channels = n_channels

        w_np = np.random.randn(n_channels, n_channels)
        q_np = np.linalg.qr(w_np)[0]

        self.W = torch.nn.Parameter(torch.from_numpy(q_np.astype('float32')))

    def forward(self, x, context=None):
        assert len(x.size()) == 4
        _, _, H, W = x.size()

        w = self.W
        ldj = H * W * torch.slogdet(w)[1]

        w = w.view(self.n_channels, self.n_channels, 1, 1)

        z = F.conv2d(
            x, w, bias=None, stride=1, padding=0, dilation=1,
            groups=1)

        return z, ldj

    def reverse(self, z, context=None):
        w_inv = torch.inverse(self.W)
        w_inv = w_inv.view(self.n_channels, self.n_channels, 1, 1)

        x = F.conv2d(
            z, w_inv, bias=None, stride=1, padding=0, dilation=1,
            groups=1)

        return x

    def logdet(self, input, context=None):
        raise NotImplementedError


class Conv1x1Householder(FlowLayer):
    def __init__(self, n_channels, n_reflections):
        super().__init__()
        self.n_channels = n_channels
        self.n_reflections = n_reflections

        v_np = np.random.randn(n_reflections, n_channels)

        self.V = torch.nn.Parameter(torch.from_numpy(v_np.astype('float32')))

    def contruct_Q(self):
        I = torch.eye(self.n_channels, dtype=self.V.dtype, device=self.V.device)
        Q = I

        for i in range(self.n_reflections):
            v = self.V[i].view(self.n_channels, 1)

            vvT = torch.matmul(v, v.t())
            vTv = torch.matmul(v.t(), v)
            Q = torch.matmul(Q, I - 2 * vvT / vTv)

        return Q

    def forward(self, x, context=None, reverse=False):
        _, _, H, W = x.size()

        ldj = 0.0
        Q = self.contruct_Q()

        if not reverse:
            Q = Q.view(self.n_channels, self.n_channels, 1, 1)

            z = F.conv2d(
                x, Q, bias=None, stride=1, padding=0, dilation=1,
                groups=1)
        else:
            Q_inv = Q.t()
            Q_inv = Q_inv.view(self.n_channels, self.n_channels, 1, 1)

            z = F.conv2d(
                x, Q_inv, bias=None, stride=1, padding=0, dilation=1,
                groups=1)

        return z, ldj

    def reverse(self, z, context=None):
        return self(z, context, reverse=True)[0]

    def logdet(self, input, context=None):
        raise NotImplementedError