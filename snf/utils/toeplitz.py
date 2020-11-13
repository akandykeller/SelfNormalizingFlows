from itertools import product
import math
import torch
import torch.nn.functional as F

def im2toepidx(c, i, j, h, w):
    return c*h*w + i*w + j

def get_toeplitz_idxs(fshape, dshape, f_stride=(1,1), s_pad=(0,0)):
    assert fshape[1] == dshape[0], "data channels must match filters channels"
    fh, fw = fshape[-2:]
    ic, ih, iw = dshape
    oh = int(math.floor((ih + 2 * s_pad[0] - fh) / f_stride[0]) + 1)
    ow = int(math.floor((iw + 2 * s_pad[1] - fw) / f_stride[1]) + 1)
    oc = fshape[0]
    
    T_idxs = []
    f_idxs = []
    for outch, outh, outw in product(range(oc), range(oh), range(ow)):
        for fi, fj in product(range(0-s_pad[0], fh-s_pad[0]), range(0-s_pad[1], fw-s_pad[1])):
            readh, readw = (outh*f_stride[0]) + fi, (outw*f_stride[1]) + fj
            if readh < 0 or readw < 0 or readh >= ih or readw >= iw:
                # We don't want to go over the 'edges' of the image.
                continue
                
            for inch in range(ic):
                Mj = im2toepidx(inch, readh, readw, ih, iw)
                Mi = im2toepidx(outch, outh, outw, oh, ow)
                T_idxs.append([Mi, Mj])
                f_flat_idx = outch*(ic*fh*fw) + inch*(fh*fw) + (fi+s_pad[0])*fh + (fj+s_pad[1])
                f_idxs.append(f_flat_idx)

    T_idxs = torch.LongTensor(T_idxs).t()
    f_idxs = torch.LongTensor(f_idxs)
    return (T_idxs, f_idxs)

def get_filter_vals(f,  f_idxs):
    vals = torch.gather(f.view(-1).to('cpu'), dim=0, index=f_idxs)
    return vals

def get_sparse_toeplitz(f, dshape, T_idxs, f_idxs):
    t_size = (T_idxs[0].max() + 1, torch.prod(torch.tensor(dshape)))
    vals = get_filter_vals(f, f_idxs)
    return torch.sparse.FloatTensor(T_idxs, vals, t_size)

def apply_toeplitz_deriv_F(T, f, T_idxs, f_idxs):
    TF = torch.zeros_like(f)
    TF_flat = TF.view(-1)
    for f_i in range(f.view(-1).shape[0]):
        T_idxs_for_fi = T_idxs[:, torch.where(f_idxs == f_i)[0]]
        TF_flat[f_i] = torch.sum(T[T_idxs_for_fi[0], T_idxs_for_fi[1]])
    return TF

def DTdw(T_idxs, f_idxs):
    T0_max = (T_idxs[0].max().item() + 1)
    T1_max = (T_idxs[1].max().item() + 1)
    shape = (T0_max * T1_max, 
              f_idxs.max().item() + 1) 
    
    F = torch.zeros(shape).to('cuda')
    
    for t, f in zip(T_idxs.T, f_idxs):
        F[t[0]*T1_max + t[1], f] = 1.0
    return F

def test_sparse_toeplitz():
    # cin, cout, fh, fw, im_h, im_w, stride, pad 
    settings = [[1, 1, 3, 3, 8, 8, 1, 0],
                [1, 1, 3, 3, 8, 8, 1, 1],
                [1, 1, 3, 3, 8, 8, 2, 0],
                [1, 3, 3, 3, 8, 8, 1, 0],
                [3, 1, 3, 3, 8, 8, 1, 0],
                [3, 3, 3, 3, 8, 8, 1, 0],
                [1, 3, 1, 1, 8, 8, 1, 0],
                [1, 3, 6, 6, 8, 8, 1, 0],
                [1, 3, 3, 3, 8, 8, 2, 0],
                [1, 3, 6, 6, 8, 8, 3, 0],
                [1, 3, 3, 3, 8, 8, 1, 2],
                [1, 3, 3, 3, 3, 3, 1, 0],
                [1, 3, 3, 3, 3, 3, 1, 2]]
                 
    all_pass = True

    for s in settings:
        cin, cout, fh, fw, im_h, im_w, stride, pad = s
        print(f"Setting: {s}")

        filters = torch.randn(cout, cin, fh, fw)
        X = torch.randn(1, cin, im_h, im_w)

        conv_out = F.conv2d(X, filters, bias=None, stride=stride, padding=pad)
        conv_out = conv_out.view(-1)

        T_idxs, f_idxs = get_toeplitz_idxs(filters.shape, X.shape[1:],
                                            f_stride=(stride, stride), s_pad=(pad, pad))
        T_sparse = get_sparse_toeplitz(filters, X.shape[1:], T_idxs, f_idxs)

        toeplitz_out = torch.mm(T_sparse.to_dense(), X.view(-1, 1)).view(-1)
        close = torch.allclose(conv_out, toeplitz_out, atol=1e-5)

        print("Toeplitz and Convolution close: ", close)
        print("Maximum difference between conv and toeplitz:", 
                torch.abs(conv_out - toeplitz_out).max())

        if not close:
            all_pass = False

    print("********"*50)
    if all_pass:
        print("All tests passed")
    else:
        print("Test failed")


if __name__ == '__main__':
    test_sparse_toeplitz()