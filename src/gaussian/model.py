"""Podatkovni model za oblak Gaussovih splattov.

GaussianCloud na enem mestu hrani položaje, barve, skale, rotacije in prosojnost,
zato ga lahko uporabljajo rendererji, LOD gradnja in bralniki PLY datotek.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class GaussianCloud:
    xyz: torch.Tensor
    scale: torch.Tensor
    color: torch.Tensor
    opacity: torch.Tensor
    rotation: torch.Tensor | None = None
    name: str = "lod"

    def to(self, device: torch.device | str) -> "GaussianCloud":
        return GaussianCloud(
            xyz=self.xyz.to(device),
            scale=self.scale.to(device),
            color=self.color.to(device),
            opacity=self.opacity.to(device),
            rotation=None if self.rotation is None else self.rotation.to(device),
            name=self.name,
        )

    @property
    def count(self) -> int:
        return int(self.xyz.shape[0])

    def memory_bytes(self) -> int:
        tensors = [self.xyz, self.scale, self.color, self.opacity]
        if self.rotation is not None:
            tensors.append(self.rotation)
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in tensors
        )

    def save_npz(self, path: str | Path) -> None:
        target = Path(path)
        np.savez_compressed(
            target,
            xyz=self.xyz.detach().cpu().numpy(),
            scale=self.scale.detach().cpu().numpy(),
            color=self.color.detach().cpu().numpy(),
            opacity=self.opacity.detach().cpu().numpy(),
            rotation=None if self.rotation is None else self.rotation.detach().cpu().numpy(),
            name=self.name,
        )

    @classmethod
    def load_npz(cls, path: str | Path, device: torch.device | str = "cpu") -> "GaussianCloud":
        data = np.load(Path(path))
        return cls(
            xyz=torch.as_tensor(data["xyz"], dtype=torch.float32, device=device),
            scale=torch.as_tensor(data["scale"], dtype=torch.float32, device=device),
            color=torch.as_tensor(data["color"], dtype=torch.float32, device=device),
            opacity=torch.as_tensor(data["opacity"], dtype=torch.float32, device=device),
            rotation=(
                torch.as_tensor(data["rotation"], dtype=torch.float32, device=device)
                if "rotation" in data and data["rotation"].shape
                else None
            ),
            name=str(data.get("name", Path(path).stem)),
        )
