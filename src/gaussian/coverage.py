from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.gaussian.model import GaussianCloud
from src.gaussian.trained_io import load_trained_gaussian_ply


@dataclass(frozen=True)
class GaussianCoverageStats:
    count: int
    file_size_bytes: int | None
    scale_percentiles: dict[str, float]
    nn_percentiles: dict[str, float]
    nn_to_scale_percentiles: dict[str, float]
    sample_count: int
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "file_size_bytes": self.file_size_bytes,
            "scale_percentiles": self.scale_percentiles,
            "nearest_neighbor_percentiles": self.nn_percentiles,
            "nearest_neighbor_to_scale_percentiles": self.nn_to_scale_percentiles,
            "sample_count": self.sample_count,
            "warnings": self.warnings,
        }


def analyze_gaussian_coverage(
    source: str | Path | GaussianCloud,
    sample_count: int = 80000,
    seed: int = 7,
) -> GaussianCoverageStats:
    path = Path(source) if isinstance(source, (str, Path)) else None
    cloud = load_trained_gaussian_ply(path) if path is not None else source
    xyz = cloud.xyz.detach().cpu().numpy()
    scale = cloud.scale.detach().cpu().numpy()
    visible_scale = _visible_scale(scale)
    if cloud.count <= 1:
        nn = np.zeros((cloud.count,), dtype=np.float32)
        sampled_scale = visible_scale
    else:
        indices = _sample_indices(cloud.count, min(sample_count, cloud.count), seed)
        sampled_xyz = xyz[indices]
        sampled_scale = visible_scale[indices]
        nn = _nearest_neighbor_distances(sampled_xyz)

    ratio = nn / np.maximum(sampled_scale, 1.0e-8)
    warnings = _coverage_warnings(ratio, sampled_scale)
    return GaussianCoverageStats(
        count=cloud.count,
        file_size_bytes=path.stat().st_size if path is not None and path.exists() else None,
        scale_percentiles=_percentiles(sampled_scale),
        nn_percentiles=_percentiles(nn),
        nn_to_scale_percentiles=_percentiles(ratio),
        sample_count=int(len(sampled_scale)),
        warnings=warnings,
    )


def _visible_scale(scale: np.ndarray) -> np.ndarray:
    if scale.ndim == 1 or scale.shape[1] == 1:
        return scale.reshape(-1).astype(np.float32)
    return np.maximum(scale[:, 0], scale[:, 1]).astype(np.float32)


def _sample_indices(count: int, sample_count: int, seed: int) -> np.ndarray:
    if sample_count >= count:
        return np.arange(count, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(count, size=sample_count, replace=False))


def _nearest_neighbor_distances(xyz: np.ndarray) -> np.ndarray:
    try:
        from scipy.spatial import cKDTree  # type: ignore

        distances, _ = cKDTree(xyz).query(xyz, k=2)
        return distances[:, 1].astype(np.float32)
    except Exception:
        return _nearest_neighbor_distances_numpy(xyz)


def _nearest_neighbor_distances_numpy(xyz: np.ndarray, chunk_size: int = 2048) -> np.ndarray:
    nn = np.full((len(xyz),), np.inf, dtype=np.float32)
    for start in range(0, len(xyz), chunk_size):
        chunk = xyz[start : start + chunk_size]
        diff = chunk[:, None, :] - xyz[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        rows = np.arange(len(chunk))
        dist2[rows, start + rows] = np.inf
        nn[start : start + len(chunk)] = np.sqrt(np.min(dist2, axis=1)).astype(np.float32)
    return nn


def _percentiles(values: np.ndarray) -> dict[str, float]:
    if len(values) == 0:
        return {key: 0.0 for key in ["p10", "p50", "p90", "p95", "p99"]}
    return {
        "p10": float(np.percentile(values, 10)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
    }


def _coverage_warnings(ratio: np.ndarray, visible_scale: np.ndarray) -> list[str]:
    warnings: list[str] = []
    if len(ratio) and float(np.percentile(ratio, 90)) > 2.0:
        warnings.append("High p90 nearest-neighbor/scale ratio; holes may be visible.")
    if len(ratio) and float(np.percentile(ratio, 99)) > 3.0:
        warnings.append("Very high p99 nearest-neighbor/scale ratio; sparse areas are likely.")
    if len(visible_scale) and float(np.mean(visible_scale <= 1.0e-3)) > 0.25:
        warnings.append("Many visible scales are at the loader minimum; consider a larger Gaussian scale/export setting.")
    return warnings
