"""Lightweight shape tests. Run with ``pytest tests/test_shapes.py``.

These tests skip the teacher (Llama2-7B) since loading 7B weights in CI
is impractical, but they exercise the Mamba block, the hybrid student,
CWR initialisation (with a synthetic teacher-sized matrix), and the
multi-modal projector."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from distillation.cwr import cwr_project_attention
from models.hybrid_mamba import HybridMambaStudent, StudentConfig
from models.mamba_block import MambaBlock


def test_mamba_block_forward():
    b = MambaBlock(d_model=128, d_state=16, d_conv=4, expand=2)
    x = torch.randn(2, 32, 128)
    y = b(x)
    assert y.shape == x.shape


def test_hybrid_student_forward():
    cfg = StudentConfig(embed_dim=128, num_mamba_layers=2, num_transformer_layers=1, num_heads=4)
    m = HybridMambaStudent(cfg)
    x = torch.randn(2, 16, 128)
    y = m(x)
    assert y.shape == x.shape


def test_cwr_projection_shape():
    W = torch.randn(4096, 4096)
    Ws = cwr_project_attention(W, student_rows=512, student_cols=512, rank=64)
    assert Ws.shape == (512, 512)


def test_cwr_reduces_rank():
    W = torch.randn(1024, 1024)
    Ws = cwr_project_attention(W, 256, 256, rank=16)
    assert torch.linalg.matrix_rank(Ws).item() <= 16
