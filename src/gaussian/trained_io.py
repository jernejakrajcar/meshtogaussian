from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.gaussian.model import GaussianCloud

SH_C0 = 0.28209479177387814


def load_trained_gaussian_ply(path: str | Path, device: torch.device | str = "cpu") -> GaussianCloud:
    data = _read_ascii_vertex_ply(path)
    required = {"x", "y", "z"}
    if not required.issubset(data):
        raise ValueError(f"{path} is missing required Gaussian position fields x/y/z.")

    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float32)
    color = _extract_color(data)
    opacity = _extract_opacity(data)
    scale = _extract_scale(data)

    return GaussianCloud(
        xyz=torch.as_tensor(xyz, dtype=torch.float32, device=device),
        scale=torch.as_tensor(scale, dtype=torch.float32, device=device),
        color=torch.as_tensor(color, dtype=torch.float32, device=device),
        opacity=torch.as_tensor(opacity, dtype=torch.float32, device=device),
        name=Path(path).stem,
    )


def build_trained_lods(
    cloud: GaussianCloud,
    counts: list[int],
    device: torch.device | str = "cpu",
) -> dict[str, GaussianCloud]:
    xyz = cloud.xyz.detach().cpu().numpy()
    opacity = cloud.opacity.detach().cpu().numpy().reshape(-1)
    scale = cloud.scale.detach().cpu().numpy().reshape(-1)
    color = cloud.color.detach().cpu().numpy()
    max_count = min(max(counts), cloud.count)
    order = _importance_spatial_order(xyz, opacity, scale, max_count)
    lods: dict[str, GaussianCloud] = {}
    for count in sorted(counts):
        if count > cloud.count:
            continue
        indices = order[:count]
        lods[str(count)] = GaussianCloud(
            xyz=torch.as_tensor(xyz[indices], dtype=torch.float32, device=device),
            scale=torch.as_tensor(scale[indices, None], dtype=torch.float32, device=device),
            color=torch.as_tensor(color[indices], dtype=torch.float32, device=device),
            opacity=torch.as_tensor(opacity[indices, None], dtype=torch.float32, device=device),
            name=str(count),
        )
    return lods


def _read_ascii_vertex_ply(path: str | Path) -> dict[str, np.ndarray]:
    target = Path(path)
    with target.open("rb") as handle:
        header_lines = []
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"{path} is not a valid PLY file.")
            text = line.decode("ascii", errors="strict").strip()
            header_lines.append(text)
            if text == "end_header":
                break
        if "format ascii 1.0" not in header_lines:
            raise ValueError("Only ASCII PLY is supported by the lightweight loader. Convert binary PLY to ASCII first.")

        vertex_count = 0
        properties: list[str] = []
        in_vertex = False
        for line in header_lines:
            parts = line.split()
            if len(parts) >= 3 and parts[:2] == ["element", "vertex"]:
                vertex_count = int(parts[2])
                in_vertex = True
                continue
            if len(parts) >= 2 and parts[0] == "element" and parts[1] != "vertex":
                in_vertex = False
            if in_vertex and len(parts) == 3 and parts[0] == "property":
                properties.append(parts[2])

        values = np.loadtxt(handle, max_rows=vertex_count, dtype=np.float32)
    if values.ndim == 1:
        values = values[None, :]
    return {name: values[:, index] for index, name in enumerate(properties)}


def _extract_color(data: dict[str, np.ndarray]) -> np.ndarray:
    if {"red", "green", "blue"}.issubset(data):
        return np.stack([data["red"], data["green"], data["blue"]], axis=1).astype(np.float32) / 255.0
    if {"r", "g", "b"}.issubset(data):
        return np.stack([data["r"], data["g"], data["b"]], axis=1).astype(np.float32)
    if {"f_dc_0", "f_dc_1", "f_dc_2"}.issubset(data):
        sh = np.stack([data["f_dc_0"], data["f_dc_1"], data["f_dc_2"]], axis=1)
        return np.clip(0.5 + SH_C0 * sh, 0.0, 1.0).astype(np.float32)
    return np.full((len(data["x"]), 3), [0.85, 0.68, 0.36], dtype=np.float32)


def _extract_opacity(data: dict[str, np.ndarray]) -> np.ndarray:
    if "opacity" not in data:
        return np.full((len(data["x"]), 1), 0.9, dtype=np.float32)
    raw = data["opacity"]
    # GraphDeco stores opacity in inverse-sigmoid/logit space.
    opacity = 1.0 / (1.0 + np.exp(-raw))
    return np.clip(opacity[:, None], 0.0, 1.0).astype(np.float32)


def _extract_scale(data: dict[str, np.ndarray]) -> np.ndarray:
    keys = [key for key in ["scale_0", "scale_1", "scale_2"] if key in data]
    if keys:
        raw = np.stack([data[key] for key in keys], axis=1)
        scale = np.exp(raw).mean(axis=1)
    elif "scale" in data:
        scale = data["scale"]
    else:
        scale = np.full(len(data["x"]), 0.015, dtype=np.float32)
    return np.clip(scale[:, None], 0.001, 0.35).astype(np.float32)


def _importance_spatial_order(
    xyz: np.ndarray,
    opacity: np.ndarray,
    scale: np.ndarray,
    count: int,
) -> np.ndarray:
    importance = np.clip(opacity, 0.0, 1.0) * np.maximum(scale, 1.0e-6)
    first = int(np.argmax(importance))
    order = np.empty(count, dtype=np.int64)
    order[0] = first
    min_dist2 = np.sum((xyz - xyz[first]) ** 2, axis=1)
    selected = np.zeros(len(xyz), dtype=bool)
    selected[first] = True
    for i in range(1, count):
        score = importance * np.sqrt(np.maximum(min_dist2, 1.0e-12))
        score[selected] = -np.inf
        next_index = int(np.argmax(score))
        order[i] = next_index
        selected[next_index] = True
        dist2 = np.sum((xyz - xyz[next_index]) ** 2, axis=1)
        min_dist2 = np.minimum(min_dist2, dist2)
    return order
