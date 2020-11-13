import numpy as np
cimport numpy as np
cimport cython
import cython.parallel as parallel
from libc.stdio cimport printf


DTYPE = np.float64


ctypedef np.float64_t DTYPE_t


#@cython.boundscheck(False)
#@cython.wraparound(False)
#@cython.nonecheck(False)
def inverse_conv(np.ndarray[DTYPE_t, ndim=4] z_np, np.ndarray[DTYPE_t, ndim=4] w_np):
    assert z_np.dtype == DTYPE and w_np.dtype == DTYPE

    cdef int batchsize = z_np.shape[0]
    cdef int n_channels = z_np.shape[1]
    cdef int height = z_np.shape[2]
    cdef int width = z_np.shape[3]
    cdef int ksize = w_np.shape[2]
    cdef int kcenter = ksize - 1

    cdef np.ndarray[DTYPE_t, ndim=4] x_np = np.zeros([batchsize, n_channels, height, width], dtype=DTYPE)

    cdef int b, j, i, c_out, c_in, k, m, j_, i_

    # Single threaded
    for b in range(batchsize):

    # Multi-threaded. Set max number of threads to avoid mem crash.
    # for b in parallel.prange(batchsize, nogil=True, num_threads=8):

        # Debug multi-threaded
        # cdef int thread_id = parallel.threadid()
        # printf("Thread ID: %d\n", thread_id)

        for j in range(height):
            for i in range(width):
                for c_out in range(n_channels):
                    for c_in in range(n_channels):
                        for k in range(ksize):
                            for m in range(ksize):
                                if k == kcenter and m == kcenter and \
                                        c_in == c_out:
                                    continue

                                j_ = j + (k - kcenter)
                                i_ = i + (m - kcenter)

                                if not ((j_ >= 0) and (j_ < height)):
                                    continue

                                if not ((i_ >= 0) and (i_ < width)):
                                    continue

                                x_np[b, c_out, j, i] -= w_np[c_out, c_in, k, m] * x_np[b, c_in, j_, i_]

                    # Compute value for x
                    x_np[b, c_out, j, i] += z_np[b, c_out, j, i]
                    x_np[b, c_out, j, i] /= w_np[c_out, c_out, kcenter, kcenter]

    return x_np
