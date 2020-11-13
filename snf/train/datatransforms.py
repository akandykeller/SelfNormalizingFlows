import torch
import numpy as np


class ToTensorNoNorm():
    def __call__(self, X_i):
        X_i = np.array(X_i, copy=False)
        if len(X_i.shape) == 2:
            # Add channel dim.
            X_i = X_i[:, :, None]
        return torch.from_numpy(np.array(X_i, copy=True)).permute(2, 0, 1)