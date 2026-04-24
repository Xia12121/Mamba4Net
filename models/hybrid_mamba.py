"""Hybrid Mamba student model (10 Mamba + 2 Transformer blocks)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .mamba_block import MambaResidualBlock, RMSNorm
from .transformer_block import TransformerBlock


@dataclass
class StudentConfig:
    embed_dim: int = 512
    num_mamba_layers: int = 10
    num_transformer_layers: int = 2
    transformer_position: str = "tail"   # tail = last N blocks
    num_heads: int = 8
    ffn_mult: int = 4
    dropout: float = 0.1
    ssm_d_state: int = 16
    ssm_d_conv: int = 4
    ssm_expand: int = 2


class HybridMambaStudent(nn.Module):
    """Layer ordering (paper §III): 10 Mamba blocks followed by 2 Transformer
    blocks positioned ``near the output layer``. The final norm is RMSNorm
    matching Llama-style architectures."""

    def __init__(self, cfg: StudentConfig) -> None:
        super().__init__()
        self.cfg = cfg

        blocks: list[nn.Module] = []
        mamba_layers = [
            MambaResidualBlock(
                cfg.embed_dim,
                d_state=cfg.ssm_d_state,
                d_conv=cfg.ssm_d_conv,
                expand=cfg.ssm_expand,
            )
            for _ in range(cfg.num_mamba_layers)
        ]
        tfm_layers = [
            TransformerBlock(cfg.embed_dim, cfg.num_heads, cfg.ffn_mult, cfg.dropout)
            for _ in range(cfg.num_transformer_layers)
        ]
        if cfg.transformer_position == "tail":
            blocks = mamba_layers + tfm_layers
        elif cfg.transformer_position == "head":
            blocks = tfm_layers + mamba_layers
        elif cfg.transformer_position == "interleave":
            # Place Transformers at equal spacing, remainder Mamba.
            blocks = list(mamba_layers)
            step = max(1, len(blocks) // (cfg.num_transformer_layers + 1))
            for i, t in enumerate(tfm_layers):
                blocks.insert((i + 1) * step + i, t)
        else:
            raise ValueError(cfg.transformer_position)

        self.blocks = nn.ModuleList(blocks)
        self.norm_f = RMSNorm(cfg.embed_dim)

    @property
    def hidden_size(self) -> int:
        return self.cfg.embed_dim

    def forward(
        self,
        hidden_states: torch.Tensor,
        return_hidden_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Run the stack.

        Args:
            hidden_states: (B, L, D) token embeddings from the multi-modal
                projector.
            return_hidden_states: also return every block's output — used
                by the feature-matching term of DKO.
        """
        h = hidden_states
        collected = [] if return_hidden_states else None
        for blk in self.blocks:
            h = blk(h)
            if return_hidden_states:
                collected.append(h)
        h = self.norm_f(h)
        if return_hidden_states:
            return h, collected
        return h
