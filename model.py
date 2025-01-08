import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster


class LayerNorm(nn.Module):
    """
    LayerNorm but with an optional bias.
    PyTorch doesn't support simply bias=False
    """

    def __init__(self, ndim, bias):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(ndim))
        self.beta = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x) -> torch.Tensor:
        return F.layer_norm(x, self.gamma.shape, self.gamma, self.beta, 1e-5)


if __name__ == "__main__":
    # N, C, H, W = 20, 3, 10, 10
    # input = torch.randn(N, C, H, W)
    # layer_norm = nn.LayerNorm([C, H, W])
    # output = layer_norm(input)

    b, s, d = 10, 5, 512
    input = torch.randn(b, s, d)
    layer_norm = LayerNorm(d, True)
    output = layer_norm(input)
    print('done')
