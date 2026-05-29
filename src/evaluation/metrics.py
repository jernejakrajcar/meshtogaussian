"""Metrike za primerjavo renderiranih rezultatov

Vsebuje izračune MSE, PSNR in časovno razliko med frame-i,
rezultate LOD prehodov prikažemo tudi s številkami.
"""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from src.gaussian.model import GaussianCloud
from src.geometry.cameras import Camera
from src.render.gaussian_renderer import GaussianRenderer


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    value = mse(a, b)
    if value <= 1.0e-12:
        return float("inf")
    return float(10.0 * np.log10(1.0 / value))


def popping_score(frames: list[np.ndarray]) -> float:
    if len(frames) < 2:
        return 0.0
    diffs = [float(np.mean(np.abs(frames[i] - frames[i - 1]))) for i in range(1, len(frames))]
    return float(np.percentile(diffs, 95))


class Evaluator:
    def __init__(self, logger: Any | None = None):
        self.logger = logger

    def _items(self, lods: dict[str, GaussianCloud], description: str):
        items = lods.items()
        if self.logger is None:
            return items
        return self.logger.iter(items, description, total=len(lods))

    def render_performance(
        self,
        renderer: GaussianRenderer,
        lods: dict[str, GaussianCloud],
        camera: Camera,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, lod in self._items(lods, "LOD performance"):
            start = perf_counter()
            renderer.render(lod, camera)
            elapsed = perf_counter() - start
            result[name] = {
                "seconds": elapsed,
                "fps": 1.0 / elapsed if elapsed > 0.0 else None,
                "memory_bytes": lod.memory_bytes(),
                "gaussians": lod.count,
            }
        return result

    def quality_by_lod(
        self,
        reference_rgb: np.ndarray,
        renderer: GaussianRenderer,
        lods: dict[str, GaussianCloud],
        camera: Camera,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, lod in self._items(lods, "LOD quality"):
            rgb = renderer.render(lod, camera)
            result[name] = {
                "mse": mse(reference_rgb, rgb),
                "psnr": psnr(reference_rgb, rgb),
            }
        return result

    def save(self, path: str | Path, metrics: dict[str, Any]) -> None:
        target = Path(path)
        target.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
