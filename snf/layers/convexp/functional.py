import torch
import numpy as np
import torch.nn.functional as F


def matrix_log(B, terms=10):
    assert B.size(0) == B.size(1)
    I = torch.eye(B.size(0))

    B_min_I = B - I

    # for k = 1.
    product = B_min_I
    result = B_min_I

    is_minus = -1
    for k in range(2, terms):
        # Reweighing with k term.
        product = torch.matmul(product, B_min_I) * (k - 1) / k
        result = result + is_minus * product

        is_minus *= -1

    return result


def matrix_exp(M, terms=10):
    assert M.size(0) == M.size(1)
    I = torch.eye(M.size(0))

    # for i = 0.
    result = I
    product = I

    for i in range(1, terms + 1):
        product = torch.matmul(product, M) / i
        result = result + product

    return result


def conv_exp(input, kernel, terms=10, dynamic_truncation=0, verbose=False):
    B, C, H, W = input.size()

    assert kernel.size(0) == kernel.size(1)
    assert kernel.size(0) == C, '{} != {}'.format(kernel.size(0), C)

    padding = (kernel.size(2) - 1) // 2, (kernel.size(3) - 1) // 2

    result = input
    product = input

    for i in range(1, terms + 1):
        product = F.conv2d(product, kernel, padding=padding, stride=(1, 1)) / i
        result = result + product

        if dynamic_truncation != 0 and i > 5:
            if product.abs().max().item() < dynamic_truncation:
                break

    # if verbose:
    #     print(
    #         'Maximum element size in term: {}'.format(
    #             torch.max(torch.abs(product))))

    return result


def inv_conv_exp(input, kernel, terms=10, dynamic_truncation=0, verbose=False):
    return conv_exp(input, -kernel, terms, dynamic_truncation, verbose)


def log_det(kernel):
    Cout, C, K1, K2 = kernel.size()
    assert Cout == C

    M1 = (K1 - 1) // 2
    M2 = (K2 - 1) // 2

    diagonal = kernel[torch.arange(C), torch.arange(C), M1, M2]

    trace = torch.sum(diagonal)

    return trace


def convergence_scale(c, kernel_size):
    C_out, C_in, K1, K2 = kernel_size

    d = C_in * K1 * K2

    return c / d


# def power_iteration(v, u, kernel)


if __name__ == '__main__':
    x = np.random.randn(2, 2)
    I = torch.eye(2)
    q = np.linalg.qr(x)[0].astype('float32')
    q = torch.from_numpy(q)
    print(torch.matmul(q, q.t()))

    W = I + q

    log_W = matrix_log(W, terms=10000)
    print(log_W)
    W_recon = matrix_exp(log_W, terms=100)

    print(W - W_recon)

    # x = torch.randn(127, 12, 16, 16)
    #
    # kernel = torch.randn(12, 12, 3, 3)
    #
    # allowed_range = convergence_scale(c=2., kernel_size=kernel.size())
    #
    # print('Allowed range:', allowed_range)
    #
    # kernel = torch.tanh(kernel) * allowed_range
    #
    # terms = 10
    #
    # z = conv_exp(x, kernel, terms=terms)
    #
    # x_recon = inv_conv_exp(z, kernel, terms=terms)
    #
    # print(torch.mean(torch.abs(x - x_recon)))
