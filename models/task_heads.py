"""Task-specific networking heads. These replace the LLM's LM head so that
the final output logits map directly to the task's answer space (§III-A)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _PooledMLP(nn.Module):
    """Take the last token of the sequence, apply an MLP to the requested
    output dimension."""

    def __init__(self, d_model: int, out_dim: int, hidden: int | None = None) -> None:
        super().__init__()
        hidden = hidden or d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, out_dim)
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.mlp(hidden[:, -1])


class ViewportHead(nn.Module):
    """Regression head producing (yaw, pitch) for the next K frames."""

    def __init__(self, d_model: int, output_dim: int = 2, future_horizon: int = 1) -> None:
        super().__init__()
        self.horizon = future_horizon
        self.head = _PooledMLP(d_model, output_dim * future_horizon)
        self.output_dim = output_dim

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        y = self.head(hidden)
        return y.view(hidden.size(0), self.horizon, self.output_dim)


class BitratePolicyHead(nn.Module):
    """Categorical policy over `num_bitrates` actions + a value head."""

    def __init__(self, d_model: int, num_bitrates: int = 6) -> None:
        super().__init__()
        self.pi = _PooledMLP(d_model, num_bitrates)
        self.v = _PooledMLP(d_model, 1)

    def forward(self, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = self.pi(hidden)
        value = self.v(hidden).squeeze(-1)
        return {"logits": logits, "value": value, "probs": F.softmax(logits, dim=-1)}


class SchedulingPolicyHead(nn.Module):
    """Two-headed policy for CJS: per-node stage logits (sized by the current
    frontier, masked externally) plus a discrete distribution over executor
    buckets."""

    def __init__(self, d_model: int, num_executor_choices: int = 16) -> None:
        super().__init__()
        self.stage_score = nn.Linear(d_model, 1)
        self.exec_pi = _PooledMLP(d_model, num_executor_choices)
        self.value = _PooledMLP(d_model, 1)

    def forward(
        self,
        node_hidden: torch.Tensor,   # (N, D) per-node embedding
        global_hidden: torch.Tensor, # (B, L, D) from hybrid backbone
        node_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        stage_logits = self.stage_score(node_hidden).squeeze(-1)   # (N,)
        if node_mask is not None:
            stage_logits = stage_logits.masked_fill(~node_mask, float("-inf"))
        exec_logits = self.exec_pi(global_hidden)
        value = self.value(global_hidden).squeeze(-1)
        return {
            "stage_logits": stage_logits,
            "exec_logits": exec_logits,
            "value": value,
        }


class ValueHead(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.v = _PooledMLP(d_model, 1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.v(hidden).squeeze(-1)
