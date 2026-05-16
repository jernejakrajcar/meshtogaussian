from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.gaussian.model import GaussianCloud


@dataclass
class GaussianTrainer:
    """Small placeholder refinement hook for seminar experiments.

    The default software splat renderer prioritizes readability and is not a full
    differentiable Gaussian rasterizer. This class keeps the pipeline API ready
    for a possible future `gsplat`/CUDA training path without pretending that
    this is the main project workflow. The current main path uses Mesh2Splat
    exports for practical Gaussian LODs.
    """

    cloud: GaussianCloud
    train_views: list[dict[str, Any]]
    cameras: list[Any]
    device: Any

    def optimize(
        self,
        steps: int = 50,
        learning_rate: float = 0.01,
        learn: list[str] | None = None,
        freeze: list[str] | None = None,
    ) -> GaussianCloud:
        return self.cloud
