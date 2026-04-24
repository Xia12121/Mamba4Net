"""Mamba (Selective State-Space Model) block.

Pure-PyTorch reference implementation following
Gu & Dao 2023, `Mamba: Linear-Time Sequence Modeling with Selective State
Spaces`. We deliberately use the parallel-scan-free recurrent formulation
so the block runs on any device (including Windows/CPU) without the
`mamba-ssm` / `causal-conv1d` CUDA kernels. When those kernels are
available the `mamba_ssm.Mamba` module can be swapped in as a drop-in
replacement - the public interface is kept identical (in_proj, x_proj,
dt_proj, A_log, D, out_proj).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class MambaBlock(nn.Module):
    """Selective SSM block as used in Mamba-v1."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int | str = "auto",
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        bias: bool = False,
        conv_bias: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else int(dt_rank)

        # Up-projection of the input into (x, z) streams
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=bias)

        # Depthwise causal convolution on x
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            bias=conv_bias,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        self.activation = nn.SiLU()

        # Project x into (dt, B, C) — the "selective" part.
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        # Project the learned dt_rank scalars up to inner dimension.
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # dt initialisation — follow mamba-ssm reference.
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise ValueError(dt_init)

        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True  # type: ignore[attr-defined]

        # State-space parameters A (negative real) and skip D.
        A = repeat(torch.arange(1, d_state + 1, dtype=torch.float32), "n -> d n", d=self.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True  # type: ignore[attr-defined]

        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True  # type: ignore[attr-defined]

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

    # ------------------------------------------------------------------
    # Selective scan (sequential, JIT-friendly)
    # ------------------------------------------------------------------
    @staticmethod
    def _selective_scan(
        u: torch.Tensor,   # (B, L, D)
        delta: torch.Tensor,  # (B, L, D)
        A: torch.Tensor,   # (D, N) — negative values expected
        B: torch.Tensor,   # (B, L, N)
        C: torch.Tensor,   # (B, L, N)
        D: torch.Tensor,   # (D,)
    ) -> torch.Tensor:
        """Reference selective scan producing y of shape (B, L, D)."""
        b, L, d = u.shape
        n = A.shape[1]
        # Discretise: deltaA (B,L,D,N), deltaB_u (B,L,D,N)
        deltaA = torch.exp(torch.einsum("bld,dn->bldn", delta, A))
        deltaB_u = torch.einsum("bld,bln,bld->bldn", delta, B, u)

        x = u.new_zeros(b, d, n)
        ys = []
        for t in range(L):
            x = deltaA[:, t] * x + deltaB_u[:, t]
            y_t = torch.einsum("bdn,bn->bd", x, C[:, t])
            ys.append(y_t)
        y = torch.stack(ys, dim=1)                              # (B,L,D)
        y = y + u * D                                           # skip connection
        return y

    # ------------------------------------------------------------------
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:  # (B, L, d_model)
        b, L, _ = hidden.shape

        xz = self.in_proj(hidden)                               # (B,L,2d_inner)
        x, z = xz.chunk(2, dim=-1)                              # each (B,L,d_inner)

        # Causal depthwise conv over sequence dimension
        x = rearrange(x, "b l d -> b d l")
        x = self.conv1d(x)[..., :L]
        x = rearrange(x, "b d l -> b l d")
        x = self.activation(x)

        # Selective parameters
        x_dbl = self.x_proj(x)                                  # (B,L,dt_rank+2N)
        dt, B_sel, C_sel = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        delta = F.softplus(self.dt_proj(dt))                    # (B,L,d_inner)

        A = -torch.exp(self.A_log.float())                      # (d_inner, N)
        y = self._selective_scan(x, delta, A, B_sel.float(), C_sel.float(), self.D.float())
        y = y.to(hidden.dtype)

        y = y * self.activation(z)
        y = self.out_proj(y)
        return y


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


class MambaResidualBlock(nn.Module):
    """Pre-norm + Mamba + residual. Matches the Mamba-v1 "Block" wrapper."""

    def __init__(self, d_model: int, **mamba_kwargs) -> None:
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.mixer = MambaBlock(d_model, **mamba_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mixer(self.norm(x))
