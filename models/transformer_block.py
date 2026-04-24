"""Causal Transformer block used for the 2 layers near the student's output."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mamba_block import RMSNorm


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, L, D = x.shape
        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=attn_mask is None,
        )
        y = y.transpose(1, 2).contiguous().view(B, L, D)
        return self.o_proj(y)


class FeedForward(nn.Module):
    """SwiGLU FFN (Llama-style) so dtype/shape align with the teacher."""

    def __init__(self, d_model: int, mult: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = int(mult * d_model * 2 / 3)
        hidden = ((hidden + 63) // 64) * 64   # round to multiple of 64
        self.w1 = nn.Linear(d_model, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d_model, bias=False)
        self.w3 = nn.Linear(d_model, hidden, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        ffn_mult: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = FeedForward(d_model, ffn_mult, dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), attn_mask=attn_mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x
