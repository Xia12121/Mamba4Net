"""Pensieve-style ABR simulator.

A minimal self-contained reimplementation used so the trainer can run
without requiring the external Pensieve/GENET simulator. The QoE reward
follows Pensieve:

    QoE = Σ ( q(bitrate) - μ · rebuffer - |q(bᵢ) − q(bᵢ₋₁)| )

where q(·) is bitrate utility (linear in Mbps), μ = 4.3, and all terms
are accumulated per chunk."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


VIDEO_BIT_RATES_KBPS = [300, 750, 1200, 1850, 2850, 4300]
BUFFER_THRESH = 60.0           # max buffer in seconds
CHUNK_LEN = 4.0                # seconds per chunk
HISTORY = 8


@dataclass
class ABRState:
    past_throughput: np.ndarray
    past_delay: np.ndarray
    chunk_sizes: np.ndarray
    buffer: float
    chunks_remaining: int


@dataclass
class ABRTrajectory:
    states: list[ABRState] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)


class ABREnv:
    def __init__(self, traces_dir: str | Path | None = None, chunks_per_video: int = 48, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)
        self.chunks_per_video = chunks_per_video
        self.traces: list[np.ndarray] = []
        if traces_dir and Path(traces_dir).is_dir():
            for f in sorted(Path(traces_dir).glob("*")):
                try:
                    arr = np.loadtxt(f, dtype=np.float32)
                    if arr.ndim == 1:
                        arr = arr[None, :]
                    self.traces.append(arr[:, -1] if arr.shape[1] >= 2 else arr.flatten())
                except Exception:
                    continue
        if not self.traces:
            # Synthetic LTE-like traces (Mbps)
            for _ in range(40):
                length = 300
                base = self.rng.uniform(0.5, 8.0)
                noise = self.rng.normal(0, 1.5, size=length).clip(-3, 3)
                trace = np.clip(base + noise, 0.2, 20.0).astype(np.float32)
                self.traces.append(trace)

    # ------------------------------------------------------------------
    def reset(self) -> ABRState:
        self.trace = random.choice(self.traces)
        self.trace_idx = self.rng.integers(0, max(1, len(self.trace) - self.chunks_per_video))
        self.buffer = 0.0
        self.chunk = 0
        self.last_bitrate = 1  # arbitrary start
        self.past_throughput = np.zeros(HISTORY, dtype=np.float32)
        self.past_delay = np.zeros(HISTORY, dtype=np.float32)
        return self._state()

    # ------------------------------------------------------------------
    def _state(self) -> ABRState:
        chunk_sizes = np.array(VIDEO_BIT_RATES_KBPS, dtype=np.float32) * CHUNK_LEN / 8000.0  # MB
        return ABRState(
            past_throughput=self.past_throughput.copy(),
            past_delay=self.past_delay.copy(),
            chunk_sizes=chunk_sizes,
            buffer=self.buffer,
            chunks_remaining=self.chunks_per_video - self.chunk,
        )

    # ------------------------------------------------------------------
    def step(self, action: int) -> tuple[ABRState, float, bool, dict]:
        bitrate_kbps = VIDEO_BIT_RATES_KBPS[action]
        chunk_bytes = bitrate_kbps * 1000 * CHUNK_LEN / 8.0
        tp_mbps = float(self.trace[min(self.trace_idx + self.chunk, len(self.trace) - 1)])
        download_time = chunk_bytes / (tp_mbps * 1_000_000 / 8.0 + 1e-6)
        rebuffer = max(0.0, download_time - self.buffer)
        self.buffer = max(0.0, self.buffer - download_time) + CHUNK_LEN
        self.buffer = min(self.buffer, BUFFER_THRESH)

        # QoE reward
        last_q = VIDEO_BIT_RATES_KBPS[self.last_bitrate] / 1000.0
        this_q = bitrate_kbps / 1000.0
        reward = this_q - 4.3 * rebuffer - abs(this_q - last_q)

        # Update histories
        self.past_throughput = np.roll(self.past_throughput, -1)
        self.past_throughput[-1] = tp_mbps
        self.past_delay = np.roll(self.past_delay, -1)
        self.past_delay[-1] = download_time

        self.last_bitrate = action
        self.chunk += 1
        done = self.chunk >= self.chunks_per_video
        return self._state(), float(reward), done, {"rebuffer": rebuffer}
