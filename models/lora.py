"""Low-Rank Adaptation module (Eqs. 9–10).

We implement LoRA manually so the teacher's frozen weights and the
low-rank deltas are both visible to the CWR routine (which needs the
original φ₀ matrix). For real fine-tuning on top of Hugging Face
checkpoints, pass ``use_hf_peft=True`` to :func:`inject_lora` to delegate
to the ``peft`` library.
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Wraps a frozen ``nn.Linear`` with a trainable low-rank delta.

    Eq. 9:  φΔ = A · Σ · B  (we fold Σ into A so the update is just A·B)
    Eq. 10: ŷ = φ₀·x + (A·(B·x)) · α/r
    """

    def __init__(
        self,
        base: nn.Linear,
        rank: int = 16,
        alpha: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert isinstance(base, nn.Linear)
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)

        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        in_f = base.in_features
        out_f = base.out_features
        self.lora_B = nn.Parameter(torch.zeros(rank, in_f))
        self.lora_A = nn.Parameter(torch.zeros(out_f, rank))
        nn.init.kaiming_uniform_(self.lora_B, a=5 ** 0.5)
        # lora_A stays zero so the initial delta is 0.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        delta = F.linear(self.dropout(x), self.lora_B)        # (*, rank)
        delta = F.linear(delta, self.lora_A) * self.scale      # (*, out)
        return y + delta


def inject_lora(
    model: nn.Module,
    target_modules: Iterable[str] = ("q_proj", "k_proj", "v_proj", "o_proj"),
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    use_hf_peft: bool = False,
) -> nn.Module:
    """Replace every ``nn.Linear`` whose name matches ``target_modules`` in
    place with a :class:`LoRALinear`. Returns the mutated model.

    When ``use_hf_peft`` is True we defer to ``peft.LoraConfig`` — this is
    required to reuse HF's Llama checkpoints with standard adapters.
    """
    if use_hf_peft:
        from peft import LoraConfig, get_peft_model

        cfg = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=list(target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        )
        return get_peft_model(model, cfg)

    targets = set(target_modules)
    for parent_name, parent in model.named_modules():
        for child_name, child in list(parent.named_children()):
            if child_name in targets and isinstance(child, nn.Linear):
                setattr(parent, child_name, LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))
    return model
