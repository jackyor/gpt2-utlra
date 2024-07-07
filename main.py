import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F


# TODO : write gradient accumulation
@dataclass
class GptConfig:
    block_size: int = 128  # context window size, set it smaller for testing
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GptConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def foward(self, x):
        B, T, C = x.size()  # batch size, sequence length, n_embd

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        # view to (B, nh, T, hs)
        k = k.view(B, T, self.n_embd, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_embd, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_embd, C // self.n_head).transpose(1, 2)
        #  flash attention 2
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config: GptConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd * 4)
        # gpt paper for approximate gelu, not necessary
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config: GptConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __int__(self, config: GptConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(
                    config.vocab_size, config.n_embd
                ),  # weight token embedding
                wpe=nn.Embedding(
                    config.block_size, config.n_embd
                ),  # weight position embedding
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
