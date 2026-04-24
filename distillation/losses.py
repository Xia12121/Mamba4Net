"""Auxiliary losses: feature-matching (between teacher/student hidden
states) and PPO surrogate used by RL tasks."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
def feature_matching_loss(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    projector: nn.Module | None = None,
) -> torch.Tensor:
    """MSE(π(hₛ), hₜ) over the final token only — cheap proxy for the
    "DKO filters domain-irrelevant knowledge" requirement. The optional
    ``projector`` maps the student hidden size to the teacher's."""
    hs = student_hidden[:, -1]
    ht = teacher_hidden[:, -1]
    if projector is not None:
        hs = projector(hs)
    return F.mse_loss(hs, ht)


# ---------------------------------------------------------------------------
def ppo_surrogate_loss(
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    clip_ratio: float = 0.2,
) -> torch.Tensor:
    """Standard clipped-surrogate PPO objective (used as ℒᵣₗ in Eq. 6)."""
    ratio = torch.exp(new_logprobs - old_logprobs)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
    return -torch.min(unclipped, clipped).mean()


def gae_advantages(
    rewards: torch.Tensor,        # (T,)
    values: torch.Tensor,         # (T+1,) — bootstrap included
    dones: torch.Tensor,          # (T,)
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (advantages, returns) of shape (T,)."""
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    last = 0.0
    for t in reversed(range(T)):
        nonterminal = 1.0 - dones[t].float()
        delta = rewards[t] + gamma * values[t + 1] * nonterminal - values[t]
        last = delta + gamma * lam * nonterminal * last
        adv[t] = last
    returns = adv + values[:-1]
    return adv, returns
