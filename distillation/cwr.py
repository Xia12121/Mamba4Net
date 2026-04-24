"""Cross-Heterogeneous Weight Reusing (CWR) — Algorithm 1 + Eqs. 11-14.

For every Transformer attention matrix φ(T) ∈ ℝ^{d_T × d_T} we compute

    φ(T) = Uₜ Σₜ Vₜᵀ                         (Eq. 11, reduced SVD)

and construct a student-dim projection by taking the first dₛ rows of
Uₜ and the first r singular components:

    Uₛ = Uₜ[:dₛ, :r]                         (Eq. 12)
    φ(S) = Uₛ · Σₜ[:r, :r] · Vₜᵀ[:r, :]      (Eq. 14)

The student uses φ(S) to warm-start its Mamba block's input-side
projection ``in_proj`` (which serves as the linearised-attention Q/K/V
stand-in described in §III-C). The remaining Mamba-specific matrices
(``x_proj``, ``dt_proj``, ``out_proj``) are kept at their random
initialisation — CWR only narrows the representation gap of the
attention-like warm-start, not the SSM's internal dynamics.
"""

from __future__ import annotations

from typing import Iterable

import torch

from models.hybrid_mamba import HybridMambaStudent
from models.mamba_block import MambaBlock


# ---------------------------------------------------------------------------
def cwr_project_attention(
    teacher_matrix: torch.Tensor,     # (d_T, d_T)
    student_rows: int,
    student_cols: int,
    rank: int,
) -> torch.Tensor:
    """Return φ(S) ∈ ℝ^{student_rows × student_cols} from Eqs. 11-14."""
    W = teacher_matrix.float()
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)      # U:(d_T,k) S:(k,) Vh:(k,d_T)
    r = min(rank, S.shape[0])
    U, S, Vh = U[:, :r], S[:r], Vh[:r, :]

    # Student dimensions may be smaller (usual) or larger (rare).
    # Rows:
    if student_rows <= U.shape[0]:
        Us = U[:student_rows, :]
    else:
        pad = torch.zeros(student_rows - U.shape[0], U.shape[1], dtype=U.dtype, device=U.device)
        Us = torch.cat([U, pad], dim=0)
    # Cols:
    if student_cols <= Vh.shape[1]:
        Vhs = Vh[:, :student_cols]
    else:
        pad = torch.zeros(Vh.shape[0], student_cols - Vh.shape[1], dtype=Vh.dtype, device=Vh.device)
        Vhs = torch.cat([Vh, pad], dim=1)

    W_s = Us @ torch.diag(S) @ Vhs
    return W_s


# ---------------------------------------------------------------------------
def cwr_initialise(
    student: HybridMambaStudent,
    teacher_weights: dict[str, torch.Tensor],
    rank: int = 64,
) -> dict[str, int]:
    """Warm-start the Mamba blocks of ``student`` from the teacher's Q/K/V/O
    projections. Returns a small diagnostic dict with per-layer shapes.

    Teacher weights are expected in the format produced by
    :pyattr:`TeacherLLM.attention_weights`, i.e. keys like
    ``"layer3.q_proj"``.
    """
    d_student = student.cfg.embed_dim
    info: dict[str, int] = {}

    # Collect teacher layer indices sorted ascending and keep as a circular
    # pool if the student has fewer layers than the teacher.
    teacher_layers = sorted({int(k.split(".")[0][5:]) for k in teacher_weights})
    if not teacher_layers:
        return info

    mamba_blocks: list[tuple[int, MambaBlock]] = []
    for i, blk in enumerate(student.blocks):
        if hasattr(blk, "mixer") and isinstance(blk.mixer, MambaBlock):
            mamba_blocks.append((i, blk.mixer))

    for s_idx, (layer_pos, mamba) in enumerate(mamba_blocks):
        t_idx = teacher_layers[s_idx % len(teacher_layers)]

        Wq = teacher_weights[f"layer{t_idx}.q_proj"]
        Wk = teacher_weights[f"layer{t_idx}.k_proj"]
        Wv = teacher_weights[f"layer{t_idx}.v_proj"]
        Wo = teacher_weights[f"layer{t_idx}.o_proj"]

        # -- 1. in_proj: (2*d_inner, d_model). Concatenate Q- and V-derived
        # matrices as the (x, z) stream warm-starts.
        Wq_s = cwr_project_attention(Wq, mamba.d_inner, d_student, rank)
        Wv_s = cwr_project_attention(Wv, mamba.d_inner, d_student, rank)
        in_proj_w = torch.cat([Wq_s, Wv_s], dim=0)   # (2*d_inner, d_student)

        with torch.no_grad():
            mamba.in_proj.weight.copy_(in_proj_w.to(mamba.in_proj.weight.dtype))
            if mamba.in_proj.bias is not None:
                mamba.in_proj.bias.zero_()

            # -- 2. out_proj: (d_model, d_inner). Warm-start from o_proj.
            Wo_s = cwr_project_attention(Wo, d_student, mamba.d_inner, rank)
            mamba.out_proj.weight.copy_(Wo_s.to(mamba.out_proj.weight.dtype))
            if mamba.out_proj.bias is not None:
                mamba.out_proj.bias.zero_()

            # -- 3. x_proj first d_state columns get warm-started from k_proj
            # singular directions — this gives the "B" selective gate a
            # linearised-attention flavour. The dt and C portions stay
            # random.
            Wk_s = cwr_project_attention(Wk, mamba.d_state, mamba.d_inner, rank)  # (d_state, d_inner)
            # x_proj.weight is (dt_rank + 2 * d_state, d_inner)
            start = mamba.dt_rank
            end = start + mamba.d_state
            mamba.x_proj.weight[start:end, :].copy_(Wk_s.to(mamba.x_proj.weight.dtype))

        info[f"student_block{layer_pos}"] = t_idx

    return info
