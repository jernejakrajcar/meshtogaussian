from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.gaussian.model import GaussianCloud

SH_C0 = 0.28209479177387814


def read_trained_gaussian_count(path: str | Path) -> int:
    target = Path(path)
    with target.open("rb") as handle:
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"{path} is not a valid PLY file.")
            text = line.decode("ascii", errors="strict").strip()
            parts = text.split()
            if len(parts) >= 3 and parts[:2] == ["element", "vertex"]:
                return int(parts[2])
            if text == "end_header":
                return 0


def load_trained_gaussian_ply(path: str | Path, device: torch.device | str = "cpu") -> GaussianCloud:
    data = _read_ascii_vertex_ply(path)
    required = {"x", "y", "z"}
    if not required.issubset(data):
        raise ValueError(f"{path} is missing required Gaussian position fields x/y/z.")

    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float32)
    color = _extract_color(data)
    opacity = _extract_opacity(data)
    scale = _extract_scale(data)
    rotation = _extract_rotation(data)

    return GaussianCloud(
        xyz=torch.as_tensor(xyz, dtype=torch.float32, device=device),
        scale=torch.as_tensor(scale, dtype=torch.float32, device=device),
        color=torch.as_tensor(color, dtype=torch.float32, device=device),
        opacity=torch.as_tensor(opacity, dtype=torch.float32, device=device),
        rotation=None if rotation is None else torch.as_tensor(rotation, dtype=torch.float32, device=device),
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
    scale_for_order = cloud.scale.detach().cpu().numpy()
    if scale_for_order.ndim == 2:
        scale = scale_for_order.max(axis=1)
    color = cloud.color.detach().cpu().numpy()
    rotation = None if cloud.rotation is None else cloud.rotation.detach().cpu().numpy()
    if cloud.count <= 0:
        raise ValueError("Cannot build LODs from an empty Gaussian cloud.")
    max_count = min(max(counts), cloud.count)
    # One deterministic order is shared by all LODs, so each denser level is a
    # true superset of the previous one. This is the main anti-popping property.
    order = _importance_spatial_order(xyz, opacity, scale, max_count)
    lods: dict[str, GaussianCloud] = {}
    for count in sorted(counts):
        actual_count = min(int(count), cloud.count)
        indices = order[:actual_count]
        lods[str(count)] = GaussianCloud(
            xyz=torch.as_tensor(xyz[indices], dtype=torch.float32, device=device),
            scale=torch.as_tensor(scale_for_order[indices], dtype=torch.float32, device=device),
            color=torch.as_tensor(color[indices], dtype=torch.float32, device=device),
            opacity=torch.as_tensor(opacity[indices, None], dtype=torch.float32, device=device),
            rotation=None if rotation is None else torch.as_tensor(rotation[indices], dtype=torch.float32, device=device),
            name=str(count),
        )
    return lods


def _read_ascii_vertex_ply(path: str | Path) -> dict[str, np.ndarray]:
    return _read_vertex_ply(path)


def _read_vertex_ply(path: str | Path) -> dict[str, np.ndarray]:
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

        format_line = next((line for line in header_lines if line.startswith("format ")), "")
        vertex_count = 0
        properties: list[tuple[str, str]] = []
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
                properties.append((parts[1], parts[2]))

        if vertex_count <= 0:
            return {name: np.asarray([], dtype=np.float32) for _, name in properties}

        if format_line == "format ascii 1.0":
            values = np.loadtxt(handle, max_rows=vertex_count, dtype=np.float32)
            if values.ndim == 1:
                values = values[None, :]
            return {name: values[:, index] for index, (_, name) in enumerate(properties)}

        if format_line in {"format binary_little_endian 1.0", "format binary_big_endian 1.0"}:
            endian = "<" if "little" in format_line else ">"
            dtype = np.dtype([(name, endian + _ply_numpy_type(type_name)) for type_name, name in properties])
            values = np.fromfile(handle, dtype=dtype, count=vertex_count)
            return {name: values[name].astype(np.float32) for _, name in properties}

    raise ValueError(f"Unsupported PLY format in {path}: {format_line or 'missing format line'}")


def _ply_numpy_type(type_name: str) -> str:
    mapping = {
        "char": "i1",
        "int8": "i1",
        "uchar": "u1",
        "uint8": "u1",
        "short": "i2",
        "int16": "i2",
        "ushort": "u2",
        "uint16": "u2",
        "int": "i4",
        "int32": "i4",
        "uint": "u4",
        "uint32": "u4",
        "float": "f4",
        "float32": "f4",
        "double": "f8",
        "float64": "f8",
    }
    if type_name not in mapping:
        raise ValueError(f"Unsupported PLY property type: {type_name}")
    return mapping[type_name]


def _extract_color(data: dict[str, np.ndarray]) -> np.ndarray:
    if {"f_dc_0", "f_dc_1", "f_dc_2"}.issubset(data):
        sh = np.stack([data["f_dc_0"], data["f_dc_1"], data["f_dc_2"]], axis=1)
        return np.clip(0.5 + SH_C0 * sh, 0.0, 1.0).astype(np.float32)
    if {"red", "green", "blue"}.issubset(data):
        return np.stack([data["red"], data["green"], data["blue"]], axis=1).astype(np.float32) / 255.0
    if {"r", "g", "b"}.issubset(data):
        return np.stack([data["r"], data["g"], data["b"]], axis=1).astype(np.float32)
    return np.full((len(data["x"]), 3), [0.85, 0.68, 0.36], dtype=np.float32)


def _extract_opacity(data: dict[str, np.ndarray]) -> np.ndarray:
    if "opacity" not in data:
        return np.full((len(data["x"]), 1), 0.9, dtype=np.float32)
    raw = data["opacity"]
    if raw.size and float(np.nanmin(raw)) >= 0.0 and float(np.nanmax(raw)) <= 1.0:
        return np.clip(raw[:, None], 0.0, 1.0).astype(np.float32)
    # GraphDeco stores opacity in inverse-sigmoid/logit space.
    opacity = 1.0 / (1.0 + np.exp(-raw))
    return np.clip(opacity[:, None], 0.0, 1.0).astype(np.float32)


def _extract_scale(data: dict[str, np.ndarray]) -> np.ndarray:
    keys = [key for key in ["scale_0", "scale_1", "scale_2"] if key in data]
    if keys:
        raw = np.stack([data[key] for key in keys], axis=1)
        scale_values = np.exp(raw) if float(np.nanmedian(raw)) < 0.0 else raw
        scale = scale_values
    elif "scale" in data:
        scale = data["scale"][:, None]
    else:
        scale = np.full((len(data["x"]), 1), 0.015, dtype=np.float32)
    # Preserve thin surface splats from Mesh2Splat/GraphDeco exports. Raising
    # their smallest axis to 0.001 turns surface disks into blurry blobs.
    return np.clip(scale, 1.0e-8, 0.35).astype(np.float32)


def _extract_rotation(data: dict[str, np.ndarray]) -> np.ndarray | None:
    keys = [key for key in ["rot_0", "rot_1", "rot_2", "rot_3"] if key in data]
    if len(keys) != 4:
        return None
    rotation = np.stack([data[key] for key in keys], axis=1).astype(np.float32)
    norm = np.linalg.norm(rotation, axis=1, keepdims=True)
    norm = np.maximum(norm, 1.0e-8)
    return rotation / norm


def _importance_spatial_order(
    xyz: np.ndarray,
    opacity: np.ndarray,
    scale: np.ndarray,
    count: int,
) -> np.ndarray:
    total = len(xyz)
    target_count = min(max(int(count), 0), total)
    if target_count == 0:
        return np.asarray([], dtype=np.int64)

    importance = np.clip(opacity, 0.0, 1.0) * np.maximum(scale, 1.0e-6)
    importance = np.nan_to_num(importance, nan=0.0, posinf=0.0, neginf=0.0)
    candidate_count = min(total, max(target_count * 4, target_count + 128))
    candidate_indices = np.argsort(-importance, kind="mergesort")[:candidate_count]
    candidate_xyz = xyz[candidate_indices]
    candidate_importance = importance[candidate_indices]

    minimum = candidate_xyz.min(axis=0)
    extent = np.maximum(candidate_xyz.max(axis=0) - minimum, 1.0e-6)
    grid_size = int(np.clip(round(target_count ** (1.0 / 3.0) * 3.0), 4, 32))
    normalized = np.clip((candidate_xyz - minimum[None, :]) / extent[None, :], 0.0, 0.999999)
    voxel = np.floor(normalized * grid_size).astype(np.int32)
    bucket_ids = voxel[:, 0] + grid_size * voxel[:, 1] + grid_size * grid_size * voxel[:, 2]

    buckets: dict[int, list[int]] = {}
    bucket_scores: dict[int, float] = {}
    for local_index, bucket_id in enumerate(bucket_ids.tolist()):
        buckets.setdefault(bucket_id, []).append(int(candidate_indices[local_index]))
        bucket_scores[bucket_id] = max(bucket_scores.get(bucket_id, 0.0), float(candidate_importance[local_index]))

    active_buckets = sorted(buckets, key=lambda bucket_id: (-bucket_scores[bucket_id], bucket_id))
    cursors = {bucket_id: 0 for bucket_id in active_buckets}
    order: list[int] = []
    selected: set[int] = set()

    while len(order) < target_count and active_buckets:
        next_active = []
        for bucket_id in active_buckets:
            cursor = cursors[bucket_id]
            bucket = buckets[bucket_id]
            if cursor >= len(bucket):
                continue
            index = bucket[cursor]
            cursors[bucket_id] = cursor + 1
            if index not in selected:
                order.append(index)
                selected.add(index)
                if len(order) >= target_count:
                    break
            if cursors[bucket_id] < len(bucket):
                next_active.append(bucket_id)
        active_buckets = next_active

    if len(order) < target_count:
        for index in np.argsort(-importance, kind="mergesort"):
            int_index = int(index)
            if int_index not in selected:
                order.append(int_index)
                if len(order) >= target_count:
                    break
    return np.asarray(order, dtype=np.int64)
