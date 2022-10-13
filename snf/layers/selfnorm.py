from functools import lru_cache
from itertools import product

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import wandb

import math
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from snf.layers.flowlayer import FlowLayer, mark_expensive, \
    ModifiedGradFlowLayer
from snf.utils.toeplitz import get_sparse_toeplitz, get_toeplitz_idxs
from snf.utils.convbackward import conv2d_backward


@lru_cache(maxsize=128)
def _compute_weight_multiple(wshape, output, x, padding, stride, dilation, 
                                 groups, benchmark, deterministic):
    batch_multiple =  conv2d_backward.backward_weight(wshape, 
                                           torch.ones_like(output),
                                           torch.ones_like(x),
                                           padding, stride, dilation,
                                           groups, benchmark, deterministic)
    return batch_multiple / len(x)


def flip_kernel(W):
    return torch.flip(W, (2,3)).permute(1,0,2,3).clone()


class SelfNormConvFunc(autograd.Function):

    @staticmethod
    def forward(ctx, x, W, bw, R, stride, padding, dilation, groups):
        z = F.conv2d(x, W, bw, stride, padding, dilation, groups)

        ctx.save_for_backward(x, W, bw, R, z)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        return z

    @staticmethod
    def backward(ctx, output_grad):
        x, W, bw, R, output = ctx.saved_tensors

        stride = torch.Size(ctx.stride)
        padding = torch.Size(ctx.padding)
        dilation = torch.Size(ctx.dilation)
        groups = ctx.groups
        benchmark = False
        deterministic = False

        multiple = _compute_weight_multiple(W.shape, output, x, padding, stride, 
                            dilation, groups, benchmark, deterministic)

        # Grad_W LogP(x)
        delta_z_xt = conv2d_backward.backward_weight(W.shape, output_grad, x, 
                                                     padding, stride, dilation, 
                                                     groups, benchmark, deterministic)
        weight_grad_fwd = (delta_z_xt - flip_kernel(R) * multiple) / 2.0

        # Grad_R LogP(x)
        input_grad = conv2d_backward.backward_input(x.shape, output_grad, W,
                                                    padding, stride, dilation,
                                                    groups, benchmark,
                                                    deterministic)
        Wx = output - bw.view(1, -1, 1, 1) if bw is not None else output
        neg_delta_x_Wxt = conv2d_backward.backward_weight(R.shape, -1*input_grad, Wx,
                                                          padding, stride, dilation, 
                                                          groups, benchmark, 
                                                          deterministic)
        weight_grad_inv = (neg_delta_x_Wxt + flip_kernel(W) * flip_kernel(multiple)) / 2.0

        if bw is not None:
            # Sum over all except output channel
            bw_grad = output_grad.view(output_grad.shape[:-2] + (-1,)).sum(-1).sum(0) 
        else:
            bw_grad = None

        return input_grad, weight_grad_fwd, bw_grad, weight_grad_inv, None, None, None, None


def selfnorm_conv_2d(x, W, bw, R, stride, padding, dilation=1, groups=1):
    f = SelfNormConvFunc()
    return f.apply(x, W, bw, R, stride, padding, dilation, groups)


class SelfNormConv(ModifiedGradFlowLayer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 bias=True,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 sym_recon_grad=False,
                 only_R_recon=False,
                 recon_loss_weight=1.0,
                 recon_loss_lr=0.0,
                 recon_alpha=0.9):

        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.sym_recon_grad = sym_recon_grad
        self.only_R_recon = only_R_recon
        self.recon_loss_weight = recon_loss_weight
        self.recon_loss_lr = recon_loss_lr
        self.recon_loss_ema = None
        self.alpha = recon_alpha
        self.use_bias = bias

        self.reset_parameters()

    def reset_parameters(self):
        self.logabsdet_dirty = True
        self.T_idxs, self.f_idxs = None, None

        w_shape = (self.out_channels, self.in_channels, *self.kernel_size)
        w_eye = nn.init.dirac_(torch.empty(w_shape))
        w_noise = nn.init.xavier_normal_(torch.empty(w_shape), gain=0.01)

        if self.kernel_size[0] == 1 and self.kernel_size[1] == 1:
            # If 1x1 convolution, use a random orthogonal matrix
            w_np = np.random.randn(self.out_channels, self.in_channels)
            w_init = torch.tensor(np.linalg.qr(w_np)[0]).to(torch.float).view(w_shape)
        else:
            # Otherweise init with identity + noise
            w_init = w_eye + w_noise

        self.weight_fwd = nn.Parameter(w_init)
        self.weight_inv = nn.Parameter(flip_kernel(w_init))

        b_small = torch.nn.init.normal_(torch.empty(self.out_channels), 
                                        std=w_noise.std())
        self.bias_fwd = nn.Parameter(b_small) if self.use_bias else None

    def forward(self, input, context=None, compute_expensive=False):
        if self.training:
            self.logabsdet_dirty = True
        self.input = input

        if compute_expensive:
            self.output = F.conv2d(input, self.weight_fwd, self.bias_fwd, 
                self.stride, self.padding, self.dilation, self.groups)
            ldj = self.logdet(input, context)
        else:
            self.output = selfnorm_conv_2d(
                input, self.weight_fwd, self.bias_fwd, self.weight_inv,
                self.stride, self.padding, self.dilation, self.groups)
            ldj = 0.
        return self.output, ldj

    def reverse(self, input, context=None, compute_expensive=False):
        if self.bias_fwd is not None:
            input = input - self.bias_fwd.view(1, -1, 1, 1)

        if compute_expensive:
            # Use actual inverse
            T_sparse = self.sparse_toeplitz(input, context)
            rev = torch.matmul(T_sparse.to_dense().inverse().to(input.device),
                               input.flatten(start_dim=1).unsqueeze(-1))
            rev = rev.view(input.shape)
        else:
            ## Use inverse weights
            rev = F.conv2d(input, self.weight_inv, None, self.stride, 
                            self.padding, self.dilation, self.groups)
        return rev

    def add_recon_grad(self, recon_loss_weight_update=None):
        # Compute ||x - RWx||^2
        x = self.input.detach() # have to compute z again w/ detached x
        z = F.conv2d(x, self.weight_fwd, None, 
                self.stride, self.padding, self.dilation, self.groups)
        if self.only_R_recon:
            z = z.detach()
        x_hat = F.conv2d(z, self.weight_inv, None, self.stride, 
                         self.padding, self.dilation, self.groups)
        recon_loss = (x - x_hat).pow(2).flatten(start_dim=1).sum(-1)

        # Compute ||z - WRz||^2
        if self.sym_recon_grad:
            zsym = z.detach()
            xsym = F.conv2d(z, self.weight_inv, None, 
                    self.stride, self.padding, self.dilation, self.groups)
            z_hat_sym = F.conv2d(xsym, self.weight_fwd, None, self.stride, 
                             self.padding, self.dilation, self.groups)
            recon_loss_sym = (zsym - z_hat_sym).pow(2).flatten(start_dim=1).sum(-1)
            recon_loss = (recon_loss + recon_loss_sym) / 2.0

        if recon_loss_weight_update is not None:
            self.recon_loss_weight = recon_loss_weight_update

        # Set NaN values to 0 for stability
        recon_loss[recon_loss != recon_loss] = 0.0
        recon_loss_weighted = self.recon_loss_weight * recon_loss.mean()

        # Using .backward call to add recon gradient
        recon_loss_weighted.backward()

        # If using GECO (i.e. recon_loss_lr > 0.0) update recon_loss_weight from moving average
        if self.recon_loss_lr > 0.0:
            with torch.no_grad():
                if self.recon_loss_ema is None:
                    self.recon_loss_ema = recon_loss.mean()
                else:
                    self.recon_loss_ema = self.alpha * self.recon_loss_ema + (1 - self.alpha) * recon_loss.mean()
                C_t = recon_loss.mean() + (self.recon_loss_ema - recon_loss.mean()).detach()
                delta_rlw = torch.exp(self.recon_loss_lr * C_t)
                self.recon_loss_weight = self.recon_loss_weight * delta_rlw

        return recon_loss_weighted

    def sparse_toeplitz(self, input, context=None):
        if self.T_idxs is None or self.f_idxs is None:
            self.T_idxs, self.f_idxs = get_toeplitz_idxs(
                self.weight_fwd.shape, input.shape[1:], self.stride, self.padding)

        T_sparse = get_sparse_toeplitz(self.weight_fwd, input.shape[1:],
                                       self.T_idxs, self.f_idxs)
        return T_sparse

    @mark_expensive
    def logdet(self, input, context=None):
        if self.logabsdet_dirty:
            T_sparse = self.sparse_toeplitz(input, context)
            self.logabsdet = torch.slogdet(T_sparse.to_dense())[1].to(input.device)
            self.logabsdet_dirty = False
        return self.logabsdet.view(1).expand(len(input))

    def plot_filters(self, layer_idx, max_s=10):
        name = 'SNF_L{}_{}'

        weights = {'fwd': self.weight_fwd.detach().cpu().numpy(),
                   'inv': self.weight_inv.detach().cpu().numpy()}       
        
        c_out, c_in, h, w = weights['fwd'].shape
        s = min(max_s, int(np.ceil(np.sqrt(c_out))))
        
        empy_weight = np.zeros_like(weights['fwd'][0,0,:,:])

        for direction in ['fwd', 'inv']:
            for c in range(c_in):
                f, axarr = plt.subplots(s,s)
                f.set_size_inches(7, 7)
                for s_h in range(s):
                    for s_w in range(s):
                        w_idx = s_h * s + s_w
                        if w_idx < c_out:
                            img = axarr[s_h, s_w].imshow(weights[direction][w_idx, c, :, :], cmap='PuBu_r')
                            axarr[s_h, s_w].get_xaxis().set_visible(False)
                            axarr[s_h, s_w].get_yaxis().set_visible(False)
                            f.colorbar(img, ax=axarr[s_h, s_w])
                        else:
                            img = axarr[s_h, s_w].imshow(empy_weight, cmap='PuBu_r')
                            axarr[s_h, s_w].get_xaxis().set_visible(False)
                            axarr[s_h, s_w].get_yaxis().set_visible(False)
                            f.colorbar(img, ax=axarr[s_h, s_w])
                # f.colorbar(img, ax=axarr.ravel().tolist())
            wandb.log({name.format(layer_idx, direction): wandb.Image(plt)}, commit=True)
            plt.close('all')

class SelfNormFC(SelfNormConv):

    def __init__(self, in_features, out_features, bias=True, **kwargs):
        super().__init__(in_features, out_features, (1, 1), bias, **kwargs)

    def reset_parameters(self):
        self.logabsdet_dirty = True
        self.T_idxs, self.f_idxs = None, None

        w_shape = (self.out_channels, self.in_channels, *self.kernel_size)
        sq_c = min(self.out_channels, self.in_channels)
        w_eye = nn.init.dirac_(torch.empty((sq_c, sq_c, *self.kernel_size)))
        w_noise = nn.init.xavier_normal_(torch.empty(w_shape), gain=0.01)

        w_noise[:sq_c, :sq_c, :, :] = w_eye
        w_init = w_noise

        self.weight_fwd = nn.Parameter(w_init)
        self.weight_inv = nn.Parameter(flip_kernel(w_init))

        b_small = torch.nn.init.normal_(torch.empty(self.out_channels), 
                                        std=w_noise.std())
        self.bias_fwd = nn.Parameter(b_small) if self.use_bias else None

    def forward(self, input, context=None, compute_expensive=False):
        input = input.view(-1, self.in_channels, 1, 1)
        output, ldj = super().forward(input, context, compute_expensive)
        return output.view(-1, self.out_channels), ldj

    def reverse(self, input, context=None, compute_expensive=False):
        input = input.view(-1, self.out_channels, 1, 1)
        
        if self.bias_fwd is not None:
            input = input - self.bias_fwd.view(1, -1, 1, 1)

        if compute_expensive:
            # Use actual inverse
            rev = torch.matmul(self.weight_fwd[:,:,0,0].inverse(),
                               input.flatten(start_dim=1).unsqueeze(-1))
            rev = rev.view(-1, self.in_channels)
        else:
            # Use approximate inverse
            rev = torch.matmul(self.weight_inv[:,:,0,0],
                                input.flatten(start_dim=1).unsqueeze(-1))
            rev = rev.view(-1, self.in_channels)
        return rev.view(-1, self.in_channels)

    @mark_expensive
    def logdet(self, input, context=None):
        if self.in_channels != self.out_channels:
            return torch.tensor(0.0).expand(len(input)).to(input.device)
        if self.logabsdet_dirty:
            self.logabsdet = torch.slogdet(self.weight_fwd[:,:,0,0])[1]
            self.logabsdet_dirty = False
        return self.logabsdet.view(1).expand(len(input))
