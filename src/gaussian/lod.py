"""Gradnja nivojev lod za Gaussove splatte.

Modul izbira podmnožice splattov, da lahko primerjam goste in
redke predstavitve istega modela ter izvaja gladke prehode med njimi
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.gaussian.model import GaussianCloud


@dataclass
class GaussianLODBuilder:
    points: np.ndarray
    normals: np.ndarray
    colors: np.ndarray
    seed: int = 0
    low_lod_scale_boost: float = 1.8

    def build_nested(self, counts: list[int], device: torch.device | str = "cpu") -> dict[str, GaussianCloud]:
        max_count = max(counts)
        # Ne moremo zgraditi LOD-a z vec Gaussi, kot je vzorcenih povrsinskih
        if max_count > len(self.points):
            raise ValueError(f"Requested {max_count} Gaussians, but only {len(self.points)} points exist.")

        # Every LOD is a prefix of the same farthest-point order, which keeps
        # coarse and fine levels spatially related and reduces transition pops
        order = self._farthest_point_order(max_count)
        lods: dict[str, GaussianCloud] = {}
        for count in sorted(counts):
            indices = order[:count]
            xyz = self.points[indices]
            color = self.colors[indices]
            scale = self._estimate_scales(xyz, count=count, max_count=max_count)
            opacity = np.full((count, 1), 0.92, dtype=np.float32)
            lods[str(count)] = GaussianCloud(
                xyz=torch.as_tensor(xyz, dtype=torch.float32, device=device),
                scale=torch.as_tensor(scale, dtype=torch.float32, device=device),
                color=torch.as_tensor(color, dtype=torch.float32, device=device),
                opacity=torch.as_tensor(opacity, dtype=torch.float32, device=device),
                name=str(count),
            )
        return lods

    def _farthest_point_order(self, count: int) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        points = self.points
        order = np.empty(count, dtype=np.int64)
        first = int(rng.integers(0, len(points)))
        order[0] = first
        min_dist2 = np.sum((points - points[first]) ** 2, axis=1)

        for i in range(1, count):
            next_index = int(np.argmax(min_dist2))
            order[i] = next_index
            dist2 = np.sum((points - points[next_index]) ** 2, axis=1)
            min_dist2 = np.minimum(min_dist2, dist2)
        return order

    def _estimate_scales(self, xyz: np.ndarray, count: int, max_count: int) -> np.ndarray:
        # En sam splat nima soseda za oceno razdalje, zato dobi default size
        if len(xyz) <= 1:
            return np.full((len(xyz), 1), 0.18, dtype=np.float32)

        sample_count = min(len(xyz), 1024)
        sampled = xyz[np.linspace(0, len(xyz) - 1, sample_count, dtype=np.int64)]
        diff = xyz[:, None, :] - sampled[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        dist[dist < 1.0e-7] = np.inf
        nearest = np.min(dist, axis=1)
        median = float(np.median(nearest[np.isfinite(nearest)]))
        # Redkejsi LOD-i dobijo vecjo skalo, da zakrijejo luknje med manj "gostimi" splatti
        relative = (max_count / max(count, 1)) ** (1.0 / 3.0)
        scale = np.maximum(nearest, median) * self.low_lod_scale_boost * relative
        return np.clip(scale[:, None], 0.006, 0.35).astype(np.float32)
