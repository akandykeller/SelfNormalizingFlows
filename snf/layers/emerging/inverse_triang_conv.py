import numpy as np
# import time
# from Cython.Build import cythonize
import pyximport
pyximport.install(
    inplace=True,
    setup_args={"include_dirs": np.get_include()},
    )
import snf.layers.emerging.inverse_op_cython as inverse_op_cython


class Inverse():
    def __init__(self):
        pass

    def __call__(self, z, w, b):
        if np.isnan(z).any():
            return z

        z = z - b

        z_np = np.array(z, dtype='float64')
        w_np = np.array(w, dtype='float64')

        x_np = inverse_op_cython.inverse_conv(
            z_np, w_np)

        return x_np.astype('float32')
