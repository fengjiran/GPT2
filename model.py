import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """
    def __init__(self, ndim, bias):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(ndim))
        self.beta = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x) -> torch.Tensor:
        return F.layer_norm(x, self.gamma.shape, self.gamma, self.beta, 1e-5)


if __name__ == "__main__":
    N, C, H, W = 20, 3, 10, 10
    x = torch.randn(N, C, H, W)
    layer_norm = nn.LayerNorm([C, H, W])
    y = layer_norm(x)
    print('done')
