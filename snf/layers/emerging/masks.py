import numpy as np


def get_linear_ar_mask(n_out, n_in, zerodiagonal=False):
    assert n_in % n_out == 0 or n_out % n_in == 0, "%d - %d" % (n_in, n_out)

    mask = np.ones([n_out, n_in], dtype=np.float32)
    if n_out >= n_in:
        k = n_out // n_in
        for i in range(n_in):
            mask[i * k:(i + 1) * k, i + 1:] = 0
            if zerodiagonal:
                mask[i * k:(i + 1) * k, i:i + 1] = 0
    else:
        k = n_in // n_out
        for i in range(n_out):
            mask[i:i + 1, (i + 1) * k:] = 0
            if zerodiagonal:
                mask[i:i + 1, i * k:(i + 1) * k] = 0
    return mask


# def get_conv_ar_mask(h, w, n_in, n_out, zerodiagonal=False):
#     """
#     Function to get autoregressive convolution
#     """
#     l = (h - 1) // 2
#     m = (w - 1) // 2
#     mask = np.ones([h, w, n_in, n_out], dtype=np.float32)
#     mask[:l, :, :, :] = 0
#     mask[l, :m, :, :] = 0
#     mask[l, m, :, :] = get_linear_ar_mask(n_in, n_out, zerodiagonal)
#     return mask


def get_conv_square_ar_mask(n_out, n_in, h, w, zerodiagonal=False):
    """
    Function to get autoregressive convolution with square shape.
    """
    mask = np.ones([n_out, n_in, h, w], dtype=np.float32)
    mask[:, :, -1, -1] = get_linear_ar_mask(n_out, n_in, zerodiagonal)

    return mask
