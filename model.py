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
        self.gelu = nn.GELU()

    def forward(self, x) -> torch.Tensor:
        return F.layer_norm(x, self.gamma.shape, self.gamma, self.beta, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super(CausalSelfAttention, self).__init__()
        assert config.n_embd % config.n_head == 0

        # input projection
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)

        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.residual_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = nn.Dropout(config.dropout)
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                 .view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # batch size, sequence length, embedding dimensionality (n_embd)
        B, N, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        query, key, value = self.c_attn(x).split(self.n_embd, dim = 2)


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.fc2 = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x) -> torch.Tensor:
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return self.dropout(x)


if __name__ == "__main__":
    # N, C, H, W = 20, 3, 10, 10
    # input = torch.randn(N, C, H, W)
    # layer_norm = nn.LayerNorm([C, H, W])
    # output = layer_norm(input)

    b, s, d = 10, 5, 64 * 3
    input = torch.randn(b, s, d)
    layer_norm = LayerNorm(d, True)
    output = layer_norm(input)
    assert output.is_contiguous()

    q, k, v = output.split(64, dim=-1)
    print('done')
