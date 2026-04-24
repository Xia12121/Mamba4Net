"""Unified training entrypoint for the three networking tasks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running `python train.py` from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import load_yaml, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Mamba4Net training")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-teacher", action="store_true",
                        help="Train Mamba4Net-S ablation (no teacher)")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg.get("seed", 42))
    task = cfg["task"]

    if args.no_teacher:
        cfg["teacher"]["use_lora"] = False
        cfg["distillation"]["alpha"] = 0.0
        cfg["distillation"]["beta"] = 0.0
        cfg["cwr"]["enabled"] = False

    if task == "vp":
        from tasks.viewport import ViewportTrainer
        trainer = ViewportTrainer(cfg, device=args.device)
    elif task == "abr":
        from tasks.abr import ABRTrainer
        trainer = ABRTrainer(cfg, device=args.device)
    elif task == "cjs":
        from tasks.cjs import CJSTrainer
        trainer = CJSTrainer(cfg, device=args.device)
    else:
        raise ValueError(f"Unknown task: {task}")

    trainer.train()


if __name__ == "__main__":
    main()
