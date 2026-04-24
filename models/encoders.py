"""Multi-modal encoders (Eq. 1-2).

Each encoder ``ℰₘ`` turns a raw modality into a fixed-width embedding
``ℛₘ``; :class:`ModalityProjector` then linearly maps every modality to
the student / teacher embedding dimension and concatenates the resulting
tokens into the unified sequence ``E``."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Viewport Prediction — past viewports (time-series) + video frames (ViT)
# ---------------------------------------------------------------------------
class TimeSeriesEncoder(nn.Module):
    """1-D Transformer encoder for short viewport histories."""

    def __init__(self, input_dim: int, hidden: int = 256, num_layers: int = 2, n_heads: int = 4) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads, dim_feedforward=hidden * 4,
            batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.out_dim = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, C) -> (B, T, H)
        return self.encoder(self.proj(x))


class ViTEncoder(nn.Module):
    """Thin wrapper around timm's ViT. We take patch tokens (without cls)
    as the visual token sequence."""

    def __init__(self, model_name: str = "vit_tiny_patch16_224", pretrained: bool = True) -> None:
        super().__init__()
        import timm

        self.vit = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool="")
        self.out_dim = self.vit.num_features

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: (B, T, 3, H, W) or (B, 3, H, W)
        if frames.ndim == 5:
            B, T, C, H, W = frames.shape
            x = frames.view(B * T, C, H, W)
            tokens = self.vit.forward_features(x)               # (B*T, P, D)
            return tokens.view(B, T * tokens.size(1), tokens.size(-1))
        return self.vit.forward_features(frames)                # (B, P, D)


# ---------------------------------------------------------------------------
# ABR — throughput / delay / chunk-sizes / buffer length (all scalars)
# ---------------------------------------------------------------------------
class ABRScalarEncoder(nn.Module):
    """Mirrors the Pensieve-style feature extractor. Each scalar/vector input
    goes through a small 1-D conv or MLP and the resulting per-feature
    tokens are stacked into a sequence of length 4."""

    def __init__(
        self,
        throughput_dim: int,
        delay_dim: int,
        chunk_size_dim: int,
        buffer_dim: int = 1,
        hidden: int = 256,
    ) -> None:
        super().__init__()
        self.throughput = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=3, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(128, hidden),
        )
        self.delay = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=3, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(128, hidden),
        )
        self.chunk = nn.Sequential(
            nn.Linear(chunk_size_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden),
        )
        self.buffer = nn.Sequential(nn.Linear(buffer_dim, hidden), nn.GELU())
        self.out_dim = hidden
        self.num_tokens = 4

    def forward(
        self,
        throughput: torch.Tensor,      # (B, T)
        delay: torch.Tensor,           # (B, T)
        chunk_sizes: torch.Tensor,     # (B, C)
        buffer: torch.Tensor,          # (B, 1)
    ) -> torch.Tensor:
        t = self.throughput(throughput.unsqueeze(1))
        d = self.delay(delay.unsqueeze(1))
        c = self.chunk(chunk_sizes)
        b = self.buffer(buffer)
        return torch.stack([t, d, c, b], dim=1)                # (B, 4, H)


# ---------------------------------------------------------------------------
# Cluster Job Scheduling — DAG via GCN
# ---------------------------------------------------------------------------
class GNNEncoder(nn.Module):
    """Message-passing encoder over the scheduling DAG. Accepts a batch of
    graphs in ``torch_geometric`` format (node features + edge index) and
    returns per-node embeddings of shape ``(N, H)``."""

    def __init__(
        self,
        node_feat_dim: int,
        hidden: int = 256,
        num_layers: int = 3,
        gnn_type: str = "gcn",
    ) -> None:
        super().__init__()
        from torch_geometric.nn import GCNConv, GATConv

        conv_cls = {"gcn": GCNConv, "gat": GATConv}[gnn_type]
        self.in_lin = nn.Linear(node_feat_dim, hidden)
        self.convs = nn.ModuleList(
            [conv_cls(hidden, hidden) for _ in range(num_layers)]
        )
        self.out_dim = hidden

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.in_lin(x)
        for conv in self.convs:
            h = F.gelu(conv(h, edge_index)) + h
        return h


# ---------------------------------------------------------------------------
# Unified projector (Eq. 2): E = concat[linear(ℛₘ)]
# ---------------------------------------------------------------------------
class ModalityProjector(nn.Module):
    """Projects a dict of modality token tensors to a common ``d_out`` and
    concatenates along the sequence axis, preserving ordering."""

    def __init__(self, modality_dims: dict[str, int], d_out: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.projs = nn.ModuleDict(
            {name: nn.Linear(d_in, d_out) for name, d_in in modality_dims.items()}
        )
        self.norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(dropout)
        self.d_out = d_out

    def forward(self, feats: dict[str, torch.Tensor]) -> torch.Tensor:
        tokens: list[torch.Tensor] = []
        for name, proj in self.projs.items():
            if name not in feats:
                continue
            f = feats[name]
            if f.ndim == 2:                # (B, D) -> (B, 1, D)
                f = f.unsqueeze(1)
            tokens.append(proj(f))
        e = torch.cat(tokens, dim=1)
        return self.drop(self.norm(e))
