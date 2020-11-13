import numpy as np
import torch
import torch.nn as nn
from snf.layers.activations import LearnableLeakyRelu, SplineActivation, \
    SmoothLeakyRelu, SmoothTanh, Identity
from snf.layers.actnorm import ActNorm
from snf.layers.conv1x1 import Conv1x1, Conv1x1Householder
from snf.layers.selfnorm import SelfNormConv, SelfNormFC
from snf.layers.coupling import Coupling
from snf.layers.normalize import Normalization
from snf.layers.squeeze import Squeeze, UnSqueeze

def check_inverse(module, data_dim, n_times=10, compute_expensive=False):
    for _ in range(n_times):
        if hasattr(module, 'reset_parameters'):
            module.reset_parameters()
            module.to('cuda')
        input = torch.randn(data_dim).to('cuda')

        if compute_expensive:
            forward, logdet = module(input, compute_expensive=compute_expensive)
            reverse = module.reverse(forward, compute_expensive=compute_expensive)
        else:
            forward, logdet = module(input)
            reverse = module.reverse(forward)

        inp = input.cpu().detach().numpy()
        outp = reverse.cpu().detach().view(data_dim).numpy()

        np.testing.assert_allclose(inp, outp, atol=1e-3)

def test_snf_layer_inverses(input_size=(12, 4, 16, 16)):
    c_in = c_out = input_size[1]
    f_in = f_out = input_size[1] * input_size[2] * input_size[3]
    conv_module = SelfNormConv(c_out, c_in, (3, 3), padding=1)
    fc_module = SelfNormFC(f_out, f_in)

    check_inverse(fc_module, input_size, compute_expensive=True)
    check_inverse(conv_module, input_size, compute_expensive=True)


def check_logdet(module, data_dim):
    if hasattr(module, 'reset_parameters'):
        module.reset_parameters()
        module.to('cuda')

    nfeats = np.prod(data_dim[1:])
    x = torch.randn(data_dim).to('cuda')

    module.to('cuda')
    _, _ = module(x)
    ldj_ours = module.logdet(x) 

    def func(*inputs):
        inp = torch.stack(inputs, dim=0)
        out, _ = module(inp)
        out = out.sum(dim=0)
        return out

    J = torch.autograd.functional.jacobian(
        func, tuple(x), create_graph=False,
        strict=False)
    J = torch.stack(J, dim=0)
    J = J.view(x.size(0), nfeats, nfeats)
    logdet_pytorch = torch.slogdet(J)[1]

    ldj_ours = ldj_ours.cpu().detach().numpy()
    ldj_pytorch = logdet_pytorch.cpu().detach().numpy()

    np.testing.assert_allclose(ldj_ours, ldj_pytorch, atol=1e-4)


def test_snf_logdet(input_size=(12, 4, 8, 8)):
    c_in = c_out = input_size[1]
    f_in = f_out = input_size[1] * input_size[2] * input_size[3]
    conv_module = SelfNormConv(c_out, c_in, (3, 3), padding=1)
    fc_module = SelfNormFC(f_out, f_in)

    check_logdet(fc_module, input_size)
    check_logdet(conv_module, input_size)


def test_inverses(input_size=(12, 4, 16, 16)):
    check_inverse(LearnableLeakyRelu().to('cuda'), input_size)
    check_inverse(SplineActivation(input_size).to('cuda'), input_size)
    check_inverse(SmoothLeakyRelu().to('cuda'), input_size)
    check_inverse(SmoothTanh().to('cuda'), input_size)
    check_inverse(Identity().to('cuda'), input_size)
    check_inverse(ActNorm(input_size[1]).to('cuda'), input_size)
    check_inverse(Conv1x1(input_size[1]).to('cuda'), input_size)
    check_inverse(Conv1x1Householder(input_size[1], 10).to('cuda'), input_size)
    check_inverse(Coupling(input_size[1:]).to('cuda'), input_size)
    check_inverse(Normalization(translation=-1e-6, scale=1 / (1 - 2 * 1e-6)).to('cuda'), input_size)
    check_inverse(Squeeze().to('cuda'), input_size)
    check_inverse(UnSqueeze().to('cuda'), input_size)
    test_snf_layer_inverses(input_size)

    print("All inverse tests passed")

def test_logdet(input_size=(12, 4, 8, 8)):
    check_logdet(LearnableLeakyRelu().to('cuda'), input_size)
    check_logdet(SplineActivation(input_size).to('cuda'), input_size)
    check_logdet(SmoothLeakyRelu().to('cuda'), input_size)
    check_logdet(SmoothTanh().to('cuda'), input_size)
    check_logdet(Identity().to('cuda'), input_size)
    check_logdet(ActNorm(input_size[1]).to('cuda'), input_size)
    check_logdet(Coupling(input_size[1:]).to('cuda'), input_size)
    check_logdet(Normalization(translation=-1e-6, scale=1 / (1 - 2 * 1e-6)).to('cuda'), input_size)
    check_logdet(Squeeze().to('cuda'), input_size)
    check_logdet(UnSqueeze().to('cuda'), input_size)
    test_snf_logdet(input_size)

    print("All log-det tests passed")


if __name__ == '__main__':
    test_inverses()
    test_logdet()