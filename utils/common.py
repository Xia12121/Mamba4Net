"""Generic training utilities (seeding, optim/sched builders, meters)."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(module: torch.nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)


def build_optimizer(params, cfg: dict[str, Any]) -> Optimizer:
    name = cfg.get("optimizer", "adamw").lower()
    lr = cfg["lr"]
    wd = cfg.get("weight_decay", 0.0)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd, betas=(0.9, 0.95))
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
    raise ValueError(name)


def build_scheduler(optimizer: Optimizer, total_steps: int, cfg: dict[str, Any]) -> LambdaLR:
    warmup = int(total_steps * cfg.get("warmup_ratio", 0.03))
    kind = cfg.get("scheduler", "cosine")

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        if kind == "linear":
            return max(0.0, (total_steps - step) / max(1, total_steps - warmup))
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.n = 0

    def update(self, val: float, n: int = 1) -> None:
        self.sum += float(val) * n
        self.n += n

    @property
    def avg(self) -> float:
        return self.sum / max(1, self.n)
