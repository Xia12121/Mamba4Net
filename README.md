# Mamba4Net

PyTorch reproduction of **Mamba4Net: Distilled Hybrid Mamba Large Language Models for Networking**
(Xia et al., 2025, arXiv:2510.17147).

The framework transfers networking-specific knowledge from a Transformer-based
teacher (Llama2-7B) into a hybrid-Mamba student (10 Mamba + 2 Transformer
layers, `d_model=512`) via two components:

- **DKO** — Domain Knowledge-Oriented cross-heterogeneous distillation
  (supervised KL loss for VP, policy distillation for ABR/CJS).
- **CWR** — Cross-heterogeneous Weight Reusing: SVD of the teacher's
  Q/K/V/O projections is used to warm-start the student's Mamba input
  projections.

Three networking tasks are supported:

| Task | Paradigm | Input modality | Output |
| ---- | -------- | -------------- | ------ |
| Viewport Prediction (VP)       | Supervised   | time-series + RGB (ViT) | next (yaw, pitch) |
| Adaptive Bitrate Streaming (ABR) | RL (PPO)   | time-series scalars     | bitrate index     |
| Cluster Job Scheduling (CJS)   | RL (PPO)     | DAG (GNN)               | (stage, #executors) |

## Repository Layout

```
Mamba4Net/
├── configs/                # YAML configs per task
├── models/                 # Mamba, hybrid student, encoders, heads, LoRA
├── distillation/           # DKO losses and CWR SVD initialisation
├── tasks/                  # Task-specific datasets/envs/trainers
│   ├── viewport/
│   ├── abr/
│   └── cjs/
├── utils/                  # Common helpers
├── train.py                # Training entrypoint
├── eval.py                 # Evaluation entrypoint
└── requirements.txt
```

## Quick Start

```bash
pip install -r requirements.txt

# 1. (once) cache teacher model from Hugging Face
huggingface-cli login

# 2. Train + distil a task
python train.py --config configs/vp.yaml
python train.py --config configs/abr.yaml
python train.py --config configs/cjs.yaml

# 3. Evaluate a checkpoint
python eval.py --config configs/vp.yaml --ckpt checkpoints/vp/best.pt
```

## Algorithmic Notes

- **Hybrid student** — layers 0..9 are Mamba-v1 SSM blocks, layers 10..11
  are pre-norm causal Transformer blocks (Eq. 3 in the paper's notion of
  "a limited number of transformer blocks near the output layer").
- **LoRA teacher fine-tune** — Eq. 9/10: `ŷ = φ₀x + A(Bx)` applied to
  `q_proj, k_proj, v_proj, o_proj`.
- **CWR** — Algorithm 1: each teacher attention projection φ(T) is SVD
  decomposed `φ(T) = UₜΣₜVₜᵀ`, truncated to rank `r`, and the first
  `dₛ` rows of Uₜ re-assembled into the student-dim projection
  `φ(S) = UₛΣₜVₜᵀ` (Eqs. 11–14). These matrices initialise the
  Mamba block's in_proj / x_proj / dt_proj / out_proj.
- **DKO-SL** — Eq. 3: `ℒₛ = CE(y,ŷₛ) + α · Tᵀ · KL(ŷₜ ∥ ŷₛ)`.
- **DKO-RL** — Eqs. 6–8: PPO policy loss + `β · 𝔼[KL(πₜ ∥ πₛ)]`.
- **Multi-modal fusion** — Eq. 1/2: per-modality encoders `ℛₘ =
  ℰₘ(xₘ)` followed by a trainable linear projection and concatenation
  into the unified token sequence.

## Licence

Code released under CC-BY-4.0 (matching the paper).

## Citation

If you find this work useful, please cite:

```bibtex
@article{xia2025mamba4net,
  title   = {Mamba4Net: Distilled Hybrid Mamba Large Language Models For Networking},
  author  = {Xia, Linhan and Yang, Mingzhan and Wang, Jingjing and Yan, Ziwei and Ren, Yakun and Yu, Guo and Lei, Kai},
  journal = {arXiv preprint arXiv:2510.17147},
  year    = {2025},
  url     = {https://arxiv.org/abs/2510.17147}
}
```
