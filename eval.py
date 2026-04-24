"""Evaluation entrypoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from utils import load_yaml, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Mamba4Net evaluation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg.get("seed", 42))
    task = cfg["task"]
    ckpt = torch.load(args.ckpt, map_location="cpu")

    if task == "vp":
        from tasks.viewport import ViewportTrainer
        trainer = ViewportTrainer(cfg, device=args.device)
        trainer.agent.load_state_dict(ckpt["agent"])
        metric = trainer.evaluate()
        print(f"VP validation MAE = {metric:.4f}")
    elif task == "abr":
        from tasks.abr import ABRTrainer
        trainer = ABRTrainer(cfg, device=args.device)
        trainer.agent.load_state_dict(ckpt["agent"])
        metric = trainer.evaluate(episodes=30)
        print(f"ABR QoE per episode = {metric:.3f}")
    elif task == "cjs":
        from tasks.cjs import CJSTrainer
        trainer = CJSTrainer(cfg, device=args.device)
        trainer.agent.load_state_dict(ckpt["agent"])
        metric = trainer.evaluate(episodes=20)
        print(f"CJS mean JCT = {metric:.3f}")
    else:
        raise ValueError(task)


if __name__ == "__main__":
    main()
