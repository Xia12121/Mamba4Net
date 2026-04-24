"""Teacher wrapper — loads Llama2-7B and exposes the hooks the distiller
needs (hidden-state capture + per-token logits)."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.nn as nn

from .lora import inject_lora


@dataclass
class TeacherConfig:
    name_or_path: str = "meta-llama/Llama-2-7b-hf"
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    torch_dtype: str = "bfloat16"
    attn_impl: str = "sdpa"
    load_in_4bit: bool = False


class TeacherLLM(nn.Module):
    """Thin wrapper around ``transformers`` Llama. The forward pass returns
    the last-layer hidden states; the LM head is exposed separately so a
    networking-specific head can be swapped in."""

    def __init__(self, cfg: TeacherConfig) -> None:
        super().__init__()
        from transformers import AutoConfig, AutoModelForCausalLM

        dtype = getattr(torch, cfg.torch_dtype)
        hf_cfg = AutoConfig.from_pretrained(cfg.name_or_path)
        kwargs = dict(torch_dtype=dtype, attn_implementation=cfg.attn_impl)
        if cfg.load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        self.llm = AutoModelForCausalLM.from_pretrained(cfg.name_or_path, **kwargs)
        self.hidden_size = hf_cfg.hidden_size
        self.num_hidden_layers = hf_cfg.num_hidden_layers

        if cfg.use_lora:
            inject_lora(
                self.llm,
                target_modules=cfg.lora_target_modules,
                rank=cfg.lora_rank,
                alpha=cfg.lora_alpha,
                dropout=cfg.lora_dropout,
                use_hf_peft=True,
            )

    # ------------------------------------------------------------------
    @property
    def attention_weights(self) -> dict[str, torch.Tensor]:
        """Return a dict mapping layer index → {q,k,v,o} weight matrices of
        the base model (not the LoRA deltas). Used by CWR."""
        out: dict[str, torch.Tensor] = {}
        model = self.llm.base_model if hasattr(self.llm, "base_model") else self.llm
        for i, layer in enumerate(model.model.layers):
            attn = layer.self_attn
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                mod = getattr(attn, name)
                w = mod.base.weight if hasattr(mod, "base") else mod.weight
                out[f"layer{i}.{name}"] = w.detach().float()
        return out

    # ------------------------------------------------------------------
    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = True,
    ) -> dict[str, torch.Tensor]:
        out = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            use_cache=False,
            return_dict=True,
        )
        return {
            "logits": out.logits,              # (B, L, V) - unused if we use a task head
            "hidden_states": out.hidden_states if output_hidden_states else None,
            "last_hidden_state": out.hidden_states[-1] if output_hidden_states else None,
        }
