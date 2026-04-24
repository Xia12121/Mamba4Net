"""Viewport-prediction dataset.

Expected layout under ``cfg.data.root``::

    data/viewport/
      train/
        video_0001/
          viewports.npy      (T, 2)  — yaw/pitch in radians
          frames/%05d.jpg    (optional, used if cfg.encoder.use_vit)
        ...
      val/ ...

If the directory is missing we fall back to a synthetic random-walk
dataset so unit tests & smoke training can run end-to-end.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


class ViewportDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        window: int = 10,
        horizon: int = 1,
        fps: int = 30,
        use_vit: bool = False,
        image_size: int = 224,
        synthetic_if_empty: bool = True,
        synthetic_size: int = 2048,
    ) -> None:
        super().__init__()
        self.window = window
        self.horizon = horizon
        self.fps = fps
        self.use_vit = use_vit
        self.image_size = image_size

        self.samples: list[tuple[np.ndarray, Path | None, int]] = []
        split_dir = Path(root) / split
        if split_dir.is_dir() and any(split_dir.iterdir()):
            for vid_dir in sorted(split_dir.iterdir()):
                vp_file = vid_dir / "viewports.npy"
                if not vp_file.exists():
                    continue
                vp = np.load(vp_file).astype(np.float32)
                frames = (vid_dir / "frames") if (vid_dir / "frames").is_dir() else None
                for start in range(0, len(vp) - window - horizon + 1, max(1, window // 2)):
                    self.samples.append((vp, frames, start))
        elif synthetic_if_empty:
            self._build_synthetic(synthetic_size)
        else:
            raise FileNotFoundError(split_dir)

    # ------------------------------------------------------------------
    def _build_synthetic(self, n: int) -> None:
        rng = np.random.default_rng(0)
        for _ in range(n):
            # random walk viewports in radians
            T = self.window + self.horizon + rng.integers(0, 60)
            yaw = np.cumsum(rng.normal(0, 0.05, size=T)).astype(np.float32)
            pitch = np.cumsum(rng.normal(0, 0.03, size=T)).astype(np.float32)
            vp = np.stack([yaw, pitch], axis=-1)
            self.samples.append((vp, None, 0))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        vp, frames_dir, start = self.samples[idx]
        past = vp[start : start + self.window]
        future = vp[start + self.window : start + self.window + self.horizon]

        item: dict[str, torch.Tensor] = {
            "past_viewports": torch.from_numpy(past),
            "future_viewports": torch.from_numpy(future),
        }
        if self.use_vit:
            if frames_dir is not None:
                from PIL import Image
                import torchvision.transforms as T

                tf = T.Compose([
                    T.Resize(self.image_size), T.CenterCrop(self.image_size), T.ToTensor(),
                ])
                imgs = []
                for t in range(self.window):
                    fn = frames_dir / f"{start + t:05d}.jpg"
                    img = Image.open(fn).convert("RGB") if fn.exists() else Image.new("RGB", (self.image_size,) * 2)
                    imgs.append(tf(img))
                item["frames"] = torch.stack(imgs, dim=0)
            else:
                item["frames"] = torch.zeros(self.window, 3, self.image_size, self.image_size)
        return item


def collate_viewport(batch: Sequence[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for k in batch[0]:
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out
