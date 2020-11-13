import os
from pathlib import Path

from torch.utils.cpp_extension import load

# Note this requires GCC > 5.0
base_path = Path(os.path.dirname(os.path.realpath(__file__)))
cpp_path =  base_path / 'conv2d_backward.cpp'
build_path = base_path / 'torch_cpp_extensions/'
conv2d_backward = load(name="conv2d_backward",
                       sources=[str(cpp_path)],
                       verbose=True,
                       build_directory=str(build_path))


def conv_bias_map(b, shape):
    return b.unsqueeze(-1).unsqueeze(-1).unsqueeze(0).repeat(
        shape[0], 1, shape[2], shape[3])
