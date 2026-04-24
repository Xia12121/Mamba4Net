"""Domain Knowledge-Oriented (DKO) distillation losses.

Supervised case (VP, Eq. 3):
    ℒₛ = ℒₛₗ(y, ŷₛ) + α · T² · KL(softmax(ŷₜ/T) ‖ softmax(ŷₛ/T))

RL case (ABR/CJS, Eqs. 6-8):
    ℒ_RL-total = ℒᵣₗ(πₛ, τ) + β · 𝔼[KL(πₜ ‖ πₛ)]

where ``ℒᵣₗ`` is supplied externally (PPO-clipped surrogate, see
``losses.ppo_surrogate_loss``).
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
def kl_divergence(
    logits_p: torch.Tensor,
    logits_q: torch.Tensor,
    temperature: float = 1.0,
    reduction: str = "batchmean",
) -> torch.Tensor:
    """KL(P ‖ Q) with temperature scaling (knowledge-distillation form)."""
    p = F.log_softmax(logits_p / temperature, dim=-1)
    q = F.log_softmax(logits_q / temperature, dim=-1)
    return F.kl_div(q, p, reduction=reduction, log_target=True) * (temperature ** 2)


# ---------------------------------------------------------------------------
def supervised_distillation_loss(
    student_out: torch.Tensor,
    teacher_out: torch.Tensor,
    target: torch.Tensor,
    base_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = F.mse_loss,
    alpha: float = 0.5,
    temperature: float = 2.0,
    task: str = "regression",
) -> dict[str, torch.Tensor]:
    """Eq. 3. Combines the task loss with a KL-matching term.

    For regression tasks (VP) we replace the KL term with a
    feature-matching MSE between student and teacher outputs (which have
    different dimensions and hence cannot be softmaxed). This preserves
    the *intent* of Eq. 3 — push ŷₛ towards ŷₜ — while being well-defined
    for continuous outputs.
    """
    base = base_loss_fn(student_out, target)
    if teacher_out is None:
        return {"loss": base, "base": base.detach(), "distill": torch.tensor(0.0)}

    if task == "classification":
        distill = kl_divergence(teacher_out, student_out, temperature=temperature)
    else:  # regression — align on scaled teacher target
        if teacher_out.shape != student_out.shape:
            # Use a linear projection if shapes disagree (e.g. teacher
            # logits vs student regression). Fall back to broadcasting
            # MSE on the overlapping leading dims.
            min_dim = min(teacher_out.shape[-1], student_out.shape[-1])
            distill = F.mse_loss(student_out[..., :min_dim], teacher_out[..., :min_dim])
        else:
            distill = F.mse_loss(student_out, teacher_out)

    total = base + alpha * distill
    return {"loss": total, "base": base.detach(), "distill": distill.detach()}


# ---------------------------------------------------------------------------
def rl_policy_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    """Eq. 7: ℒ_dis = 𝔼_{s∼𝒟}[KL(πₜ(·|s) ‖ πₛ(·|s))]."""
    return kl_divergence(teacher_logits, student_logits, temperature=temperature)
