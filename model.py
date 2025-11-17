import math
from dataclasses import dataclass
import inspect
from typing import Tuple, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class GPTConfig:
    block_size: int = 1024  # sequence length
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
        # self.gelu = nn.GELU()

    def forward(self, x) -> torch.Tensor:
        return F.layer_norm(x, self.gamma.shape, self.gamma, self.beta, 1e-5)


# MHA
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
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                 .view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor, cache=None):
        # batch size, sequence length, embedding dimensionality (n_embd)
        bs, seq_len, dim = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        query, key, value = self.c_attn(x).split(self.n_embd, dim=2)
        if cache is not None:
            pk, pv = cache
            key = torch.concat([pk, key], dim=-2)
            value = torch.concat([pv, value], dim=-2)
            cache = (key, value)
        real_seq_len = key.shape[-2]

        # (BS, n_head, seq_len, head_dim)
        query = query.view(bs, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        key = key.view(bs, real_seq_len, self.n_head, self.head_dim).transpose(1, 2)
        value = value.view(bs, real_seq_len, self.n_head, self.head_dim).transpose(1, 2)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True
            )
        else:
            # manual implementation of attention
            attn = (query @ key.transpose(-2, -1)) * (1.0 / math.sqrt(query.size(-1)))
            attn = attn.masked_fill(self.bias[:, :, real_seq_len - seq_len:real_seq_len, :real_seq_len] == 0,
                                    float("-inf"))
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_dropout(attn)
            y = attn @ value

        y = y.transpose(1, 2).contiguous().view(bs, seq_len, self.n_embd)
        y = self.residual_dropout(self.c_proj(y))
        return y, cache


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


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super(Block, self).__init__()
        # self.ln_1 = LayerNorm(config.n_embd, config.bias)
        self.mha = CausalSelfAttention(config)
        # self.ln_2 = LayerNorm(config.n_embd, config.bias)
        self.mlp = MLP(config)
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)

    def forward(self, x) -> torch.Tensor:
        x = x + self.mha(self.ln_1(x))[0] # LN + MHA
        x = x + self.mlp(self.ln_2(x))[0] # LN + FFN
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super(GPT, self).__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(wte=nn.Embedding(config.vocab_size, config.n_embd),
                 wpe=nn.Embedding(config.block_size, config.n_embd),
                 dropout=nn.Dropout(config.dropout),
                 h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                 ln_f=nn.LayerNorm(config.n_embd, bias=config.bias)
                 )
        )

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight  # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for name, param in self.named_parameters():
            if name.endswith("c_proj.weight"):
                torch.nn.init.normal_(param, mean=0, std=0.02 / math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def forward(self, idx: torch.Tensor, targets=None) -> Tuple[Any, Optional[Tensor]]:
        dev = idx.device
        bs, seq_len = idx.size()
        assert seq_len <= self.config.block_size, \
            f"Cannot forward sequence of length {seq_len}, block size is only {self.config.block_size}"
        pos = torch.arange(0, seq_len, device=dev)

        token_embed = self.transformer.wte(idx)  # token embeddings of shape (bs, seq_len, n_embd)
        pos_embed = self.transformer.wpe(pos)  # position embeddings of shape (seq_len, n_embd)
        x = self.transformer.dropout(token_embed + pos_embed)

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)

    def crop_block_size(self, block_size):
        """
        model surgery to decrease the block size if necessary.
        e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        but want to use a smaller block size for some smaller, simpler model
        """
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        for block in self.transformer.h:
            if hasattr(block.mha, "bias"):
                block.mha.bias = block.mha.bias[:, :, :block_size, :block_size]

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}

        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layer norms don't.
        decay_params = [p for pn, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for pn, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        print(f"using fused AdamW: {use_fused}")
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        return optimizer


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
    q1 = q.view(10, 5, 4, 16).transpose(1, 2)
    assert q1.is_contiguous() is False
    assert q.untyped_storage().data_ptr() == q1.untyped_storage().data_ptr()
    assert hasattr(torch.nn.functional, "scaled_dot_product_attention")
    # assert torch.cuda.is_bf16_supported()

    config = GPTConfig()
    batch_size = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randint(config.vocab_size, (batch_size, config.block_size), device=device)
    y = torch.randint(config.vocab_size, (batch_size, config.block_size), device=device)

    gpt2 = GPT(config).to(device)
    # gpt2.crop_block_size(1024)

    logits, loss = gpt2(x, y)
    print('done')
    # for name, param in gpt2.named_parameters():
    #     print(name, param.size())
