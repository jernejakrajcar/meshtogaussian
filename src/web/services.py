from __future__ import annotations

import hashlib
import itertools
import math
import re
import shutil
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np
import torch

from src.core.progress import StageLogger
from src.gaussian.lod import GaussianLODBuilder
from src.gaussian.model import GaussianCloud
from src.gaussian.trained_io import build_trained_lods, load_trained_gaussian_ply, read_trained_gaussian_count
from src.geometry.mesh_loader import MeshAsset

SUPPORTED_MESH_EXTENSIONS = {".obj", ".ply", ".gltf", ".glb"}
SUPPORTED_GAUSSIAN_EXTENSIONS = {".ply"}


def is_supported_mesh(path: str | Path) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_MESH_EXTENSIONS


def safe_upload_name(filename: str) -> str:
    raw = Path(filename).name.replace(" ", "_")
    if not is_supported_mesh(raw):
        raise ValueError(f"Unsupported mesh extension for {filename!r}.")
    return raw


def _best_axis_permutation(
    cloud_xyz: np.ndarray,
    mesh_vertices: np.ndarray,
    cloud_center: np.ndarray,
    mesh_center: np.ndarray,
    sample_count: int = 2500,
) -> tuple[np.ndarray, np.ndarray]:
    cloud_sample = _even_sample(cloud_xyz, sample_count)
    mesh_sample = _even_sample(mesh_vertices, sample_count)
    best_rotation = np.eye(3, dtype=np.float32)
    best_scale = np.ones(3, dtype=np.float32)
    best_score = float("inf")
    mesh_min, mesh_max = _robust_bounds(mesh_sample)
    mesh_extent = np.maximum(mesh_max - mesh_min, 1.0e-6)
    for permutation in itertools.permutations(range(3)):
        base = np.zeros((3, 3), dtype=np.float32)
        for row, column in enumerate(permutation):
            base[row, column] = 1.0
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            rotation = (np.asarray(signs, dtype=np.float32)[:, None] * base).astype(np.float32)
            rotated_cloud = (cloud_sample - cloud_center[None, :]) @ rotation.T
            cloud_min, cloud_max = _robust_bounds(rotated_cloud)
            cloud_extent = np.maximum(cloud_max - cloud_min, 1.0e-6)
            scale_factor = np.clip(mesh_extent / cloud_extent, 0.01, 100.0).astype(np.float32)
            transformed_cloud = rotated_cloud * scale_factor[None, :] + mesh_center[None, :]
            score = _mean_nearest_distance(transformed_cloud, mesh_sample)
            if score < best_score:
                best_score = score
                best_rotation = rotation
                best_scale = scale_factor
    return best_rotation, best_scale


def _even_sample(points: np.ndarray, sample_count: int) -> np.ndarray:
    if len(points) <= sample_count:
        return points.astype(np.float32, copy=False)
    indices = np.linspace(0, len(points) - 1, sample_count, dtype=np.int64)
    return points[indices].astype(np.float32, copy=False)


def _mean_nearest_distance(points: np.ndarray, targets: np.ndarray) -> float:
    try:
        from scipy.spatial import cKDTree

        distances, _ = cKDTree(targets).query(points, k=1)
        return float(np.mean(distances))
    except Exception:
        total = 0.0
        batch_size = 512
        for start in range(0, len(points), batch_size):
            chunk = points[start : start + batch_size]
            dist2 = np.sum((chunk[:, None, :] - targets[None, :, :]) ** 2, axis=2)
            total += float(np.sqrt(np.min(dist2, axis=1)).sum())
        return total / max(len(points), 1)


def _robust_bounds(
    points: np.ndarray,
    weights: np.ndarray | None = None,
    low: float = 0.01,
    high: float = 0.99,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(points, dtype=np.float32)
    finite = np.isfinite(values).all(axis=1)
    values = values[finite]
    if len(values) == 0:
        zeros = np.zeros(3, dtype=np.float32)
        return zeros, zeros
    if len(values) < 20:
        return values.min(axis=0), values.max(axis=0)

    if weights is None:
        return (
            np.quantile(values, low, axis=0).astype(np.float32),
            np.quantile(values, high, axis=0).astype(np.float32),
        )

    clean_weights = np.asarray(weights, dtype=np.float32).reshape(-1)[finite]
    if clean_weights.shape[0] != len(values) or not np.isfinite(clean_weights).all() or float(clean_weights.sum()) <= 1.0e-8:
        return (
            np.quantile(values, low, axis=0).astype(np.float32),
            np.quantile(values, high, axis=0).astype(np.float32),
        )
    mins = []
    maxs = []
    for axis in range(3):
        mins.append(_weighted_quantile(values[:, axis], clean_weights, low))
        maxs.append(_weighted_quantile(values[:, axis], clean_weights, high))
    return np.asarray(mins, dtype=np.float32), np.asarray(maxs, dtype=np.float32)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> np.float32:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    cutoff = quantile * cumulative[-1]
    return np.float32(sorted_values[np.searchsorted(cumulative, cutoff, side="left")])


def _quaternion_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float32)
    norm = max(float(np.linalg.norm(q)), 1.0e-8)
    w, x, y, z = (q / norm).tolist()
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _matrix_to_quaternion_wxyz(matrix: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=np.float32)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    else:
        axis = int(np.argmax(np.diag(m)))
        if axis == 0:
            s = math.sqrt(max(1.0 + float(m[0, 0] - m[1, 1] - m[2, 2]), 1.0e-8)) * 2.0
            qw = (m[2, 1] - m[1, 2]) / s
            qx = 0.25 * s
            qy = (m[0, 1] + m[1, 0]) / s
            qz = (m[0, 2] + m[2, 0]) / s
        elif axis == 1:
            s = math.sqrt(max(1.0 + float(m[1, 1] - m[0, 0] - m[2, 2]), 1.0e-8)) * 2.0
            qw = (m[0, 2] - m[2, 0]) / s
            qx = (m[0, 1] + m[1, 0]) / s
            qy = 0.25 * s
            qz = (m[1, 2] + m[2, 1]) / s
        else:
            s = math.sqrt(max(1.0 + float(m[2, 2] - m[0, 0] - m[1, 1]), 1.0e-8)) * 2.0
            qw = (m[1, 0] - m[0, 1]) / s
            qx = (m[0, 2] + m[2, 0]) / s
            qy = (m[1, 2] + m[2, 1]) / s
            qz = 0.25 * s
    quat = np.asarray([qw, qx, qy, qz], dtype=np.float32)
    return quat / max(float(np.linalg.norm(quat)), 1.0e-8)


def _matrices_to_quaternion_wxyz(matrices: np.ndarray) -> np.ndarray:
    m = np.asarray(matrices, dtype=np.float32)
    quaternions = np.empty((len(m), 4), dtype=np.float32)
    traces = np.trace(m, axis1=1, axis2=2)

    positive = traces > 0.0
    if np.any(positive):
        items = m[positive]
        s = np.sqrt(traces[positive] + 1.0) * 2.0
        quaternions[positive, 0] = 0.25 * s
        quaternions[positive, 1] = (items[:, 2, 1] - items[:, 1, 2]) / s
        quaternions[positive, 2] = (items[:, 0, 2] - items[:, 2, 0]) / s
        quaternions[positive, 3] = (items[:, 1, 0] - items[:, 0, 1]) / s

    diagonal_axis = np.argmax(np.diagonal(m, axis1=1, axis2=2), axis=1)
    for axis in range(3):
        mask = ~positive & (diagonal_axis == axis)
        if not np.any(mask):
            continue
        items = m[mask]
        if axis == 0:
            s = np.sqrt(np.maximum(1.0 + items[:, 0, 0] - items[:, 1, 1] - items[:, 2, 2], 1.0e-8)) * 2.0
            quaternions[mask, 0] = (items[:, 2, 1] - items[:, 1, 2]) / s
            quaternions[mask, 1] = 0.25 * s
            quaternions[mask, 2] = (items[:, 0, 1] + items[:, 1, 0]) / s
            quaternions[mask, 3] = (items[:, 0, 2] + items[:, 2, 0]) / s
        elif axis == 1:
            s = np.sqrt(np.maximum(1.0 + items[:, 1, 1] - items[:, 0, 0] - items[:, 2, 2], 1.0e-8)) * 2.0
            quaternions[mask, 0] = (items[:, 0, 2] - items[:, 2, 0]) / s
            quaternions[mask, 1] = (items[:, 0, 1] + items[:, 1, 0]) / s
            quaternions[mask, 2] = 0.25 * s
            quaternions[mask, 3] = (items[:, 1, 2] + items[:, 2, 1]) / s
        else:
            s = np.sqrt(np.maximum(1.0 + items[:, 2, 2] - items[:, 0, 0] - items[:, 1, 1], 1.0e-8)) * 2.0
            quaternions[mask, 0] = (items[:, 1, 0] - items[:, 0, 1]) / s
            quaternions[mask, 1] = (items[:, 0, 2] + items[:, 2, 0]) / s
            quaternions[mask, 2] = (items[:, 1, 2] + items[:, 2, 1]) / s
            quaternions[mask, 3] = 0.25 * s

    norm = np.maximum(np.linalg.norm(quaternions, axis=1, keepdims=True), 1.0e-8)
    return quaternions / norm


def _transform_gaussian_covariances(
    scale: np.ndarray,
    rotation: np.ndarray | None,
    alignment_rotation: np.ndarray,
    scale_factor: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    scale_np = np.asarray(scale, dtype=np.float32)
    if scale_np.ndim == 1 or scale_np.shape[1] == 1:
        scale_np = np.repeat(scale_np.reshape(-1, 1), 3, axis=1)
    rotation_np = (
        np.tile(np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (len(scale_np), 1))
        if rotation is None
        else np.asarray(rotation, dtype=np.float32)
    )
    linear = np.diag(np.asarray(scale_factor, dtype=np.float32)) @ np.asarray(alignment_rotation, dtype=np.float32)
    transformed_scale = np.empty_like(scale_np)
    transformed_rotation = np.empty_like(rotation_np)
    batch_size = 131072
    for start in range(0, len(scale_np), batch_size):
        end = min(start + batch_size, len(scale_np))
        item_rotation = rotation_np[start:end]
        item_rotation = item_rotation / np.maximum(np.linalg.norm(item_rotation, axis=1, keepdims=True), 1.0e-8)
        w, x, y, z = item_rotation.T
        local_rotation = np.empty((end - start, 3, 3), dtype=np.float32)
        local_rotation[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
        local_rotation[:, 0, 1] = 2.0 * (x * y - z * w)
        local_rotation[:, 0, 2] = 2.0 * (x * z + y * w)
        local_rotation[:, 1, 0] = 2.0 * (x * y + z * w)
        local_rotation[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
        local_rotation[:, 1, 2] = 2.0 * (y * z - x * w)
        local_rotation[:, 2, 0] = 2.0 * (x * z - y * w)
        local_rotation[:, 2, 1] = 2.0 * (y * z + x * w)
        local_rotation[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)

        item_scale_squared = np.square(scale_np[start:end])
        covariance = (local_rotation * item_scale_squared[:, None, :]) @ local_rotation.transpose(0, 2, 1)
        transformed_covariance = (linear[None, :, :] @ covariance) @ linear.T[None, :, :]
        eigenvalues, eigenvectors = np.linalg.eigh(transformed_covariance)
        eigenvalues = np.maximum(eigenvalues[:, ::-1], 1.0e-12)
        eigenvectors = eigenvectors[:, :, ::-1]
        negative_determinant = np.linalg.det(eigenvectors) < 0.0
        eigenvectors[negative_determinant, :, 2] *= -1.0
        transformed_scale[start:end] = np.sqrt(eigenvalues).astype(np.float32)
        transformed_rotation[start:end] = _matrices_to_quaternion_wxyz(eigenvectors)
    return transformed_scale.astype(np.float32), transformed_rotation.astype(np.float32)


@dataclass
class LazyGaussianLOD:
    path: Path
    count: int


@dataclass
class PreparedModel:
    model_id: str
    mesh: MeshAsset
    lods: dict[str, GaussianCloud | LazyGaussianLOD]
    source: str
    gaussian_source: str | None = None
    representation: str = "initialized"
    alignment: GaussianAlignment | None = None
    alignment_reference_key: str | None = None
    materialize_lock: threading.RLock = field(default_factory=threading.RLock, repr=False)


@dataclass(frozen=True)
class GaussianAlignment:
    cloud_center: np.ndarray
    mesh_center: np.ndarray
    rotation: np.ndarray
    scale_factor: np.ndarray


class ModelStore:
    def __init__(
        self,
        source_dirs: list[str | Path] | None = None,
        upload_dir: str | Path = "data/source/uploads",
        trained_dirs: list[str | Path] | None = None,
        mesh2splat_lod_dirs: list[str | Path] | None = None,
        logger: StageLogger | None = None,
    ):
        self.source_dirs = [Path(path) for path in (source_dirs or ["data/source", "data/meshes"])]
        self.upload_dir = Path(upload_dir)
        self.trained_dirs = [Path(path) for path in (trained_dirs or ["data/trained_gaussians"])]
        self.mesh2splat_lod_dirs = [Path(path) for path in (mesh2splat_lod_dirs or ["data/mesh2splats"])]
        self.logger = logger or StageLogger(enabled=True, verbose=True)
        self.prepared: dict[str, PreparedModel] = {}

    def ensure_dirs(self) -> None:
        for directory in [*self.source_dirs, self.upload_dir, *self.trained_dirs, *self.mesh2splat_lod_dirs]:
            directory.mkdir(parents=True, exist_ok=True)

    def list_models(self) -> list[dict[str, str]]:
        self.ensure_dirs()
        models = [
            {"id": "demo:procedural-sphere", "name": "Procedural demo sphere", "source": "generated"}
        ]
        seen: set[Path] = set()
        for directory in self.source_dirs:
            if not directory.exists():
                continue
            for path in sorted(directory.rglob("*")):
                if path.is_file() and is_supported_mesh(path) and path.resolve() not in seen:
                    seen.add(path.resolve())
                    models.append(
                        {
                            "id": self.path_to_id(path),
                            "name": path.name,
                            "source": str(path),
                        }
                    )
        return models

    def list_trained_gaussians(self) -> list[dict[str, str]]:
        self.ensure_dirs()
        models = []
        seen: set[Path] = set()
        for directory in self.trained_dirs:
            if not directory.exists():
                continue
            for path in sorted(directory.rglob("*.ply")):
                resolved = path.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                models.append({"id": self.path_to_id(path), "name": self._display_path_name(path), "source": str(path)})
        return models

    def list_mesh2splat_gaussians(self) -> list[dict[str, str]]:
        self.ensure_dirs()
        models = []
        for directory in self.mesh2splat_lod_dirs:
            if not directory.exists():
                continue
            for path in sorted(directory.rglob("*.ply")):
                models.append({"id": self.path_to_id(path), "name": self._display_path_name(path), "source": str(path)})
        return models

    def list_mesh2splat_lod_sets(self) -> list[dict[str, Any]]:
        self.ensure_dirs()
        grouped: dict[str, list[tuple[int, Path]]] = {}
        for directory in self.mesh2splat_lod_dirs:
            if not directory.exists():
                continue
            for path in sorted(directory.rglob("*.ply")):
                count = self._count_from_lod_filename(path)
                if count is None:
                    continue
                key = self._lod_group_key(path)
                grouped.setdefault(key, []).append((count, path))
        sets = []
        for key, entries in sorted(grouped.items()):
            unique_counts = sorted({count for count, _ in entries})
            if not unique_counts:
                continue
            sets.append(
                {
                    "id": f"mesh2splat-lods:{key}",
                    "name": f"{key} ({len(unique_counts)} LODs, {unique_counts[0]}-{unique_counts[-1]})",
                    "mesh_key": key,
                    "counts": unique_counts,
                    "source": ", ".join(str(path) for _, path in sorted(entries)),
                }
            )
        return sets

    def save_upload(self, filename: str, fileobj: Any) -> dict[str, str]:
        self.ensure_dirs()
        safe_name = safe_upload_name(filename)
        target = self.upload_dir / safe_name
        with target.open("wb") as handle:
            shutil.copyfileobj(fileobj, handle)
        return {"id": self.path_to_id(target), "name": target.name, "source": str(target)}

    def prepare(
        self,
        model_id: str | None,
        lod_counts: list[int],
        seed: int = 7,
        fallback_color: list[float] | None = None,
        representation: str = "initialized",
        trained_ply_id: str | None = None,
    ) -> PreparedModel:
        if representation == "mesh2splat_lods":
            raise ValueError(
                "Automatic Mesh2Splat LOD sets are disabled. Select one Mesh2Splat .ply file as the Gaussian source."
            )
        with self.logger.stage("web model preparation"):
            source = "generated"
            mesh_path: Path | None = None
            if model_id and not model_id.startswith("demo:"):
                mesh_path = self.id_to_path(model_id)
                source = str(mesh_path)

            mesh = MeshAsset.load(mesh_path, fallback_color=fallback_color)
            mesh_center, mesh_radius = mesh.normalize_to_unit_sphere()
            gaussian_source = None
            if representation in {"trained", "mesh2splat"}:
                if representation == "mesh2splat":
                    if not trained_ply_id:
                        raise ValueError("Select one Mesh2Splat .ply file before loading the setup.")
                    gaussian_path = self.id_to_mesh2splat_path(trained_ply_id)
                    source_cloud = load_trained_gaussian_ply(gaussian_path)
                    trained_cloud = self._normalize_gaussian_to_mesh(source_cloud, mesh_center, mesh_radius)
                else:
                    gaussian_path = self.id_to_trained_path(trained_ply_id) if trained_ply_id else self.find_trained_for_mesh(mesh.name)
                if gaussian_path is None:
                    raise FileNotFoundError(
                        "No trained Gaussian .ply was provided or found. Generate one with train_gaussians_from_mesh.py first."
                    )
                if representation == "trained":
                    trained_cloud = self._align_gaussian_to_normalized_mesh(load_trained_gaussian_ply(gaussian_path), mesh.vertices)
                trained_counts = sorted({count for count in lod_counts if count < trained_cloud.count} | {trained_cloud.count})
                lods = build_trained_lods(trained_cloud, trained_counts)
                gaussian_source = str(gaussian_path)
            else:
                points, normals, colors = mesh.sample_surface(max(lod_counts), seed=seed)
                lods = GaussianLODBuilder(points, normals, colors, seed=seed).build_nested(lod_counts)
                alignment_reference_key = None
            prepared_id = self._prepared_id(f"{model_id or 'demo'}:{representation}:{gaussian_source}", lod_counts, source)
            prepared = PreparedModel(
                model_id=prepared_id,
                mesh=mesh,
                lods=lods,
                source=source,
                gaussian_source=gaussian_source,
                representation=representation,
                alignment_reference_key=alignment_reference_key if representation == "mesh2splat_lods" else None,
            )
            self.prepared[prepared_id] = prepared
            return prepared

    def get_prepared(self, model_id: str) -> PreparedModel:
        if model_id not in self.prepared:
            raise KeyError(f"Model {model_id!r} has not been prepared yet.")
        return self.prepared[model_id]

    def serialize_model(self, prepared: PreparedModel) -> dict[str, Any]:
        mesh = prepared.mesh
        source_path = None if prepared.source == "generated" else Path(prepared.source)
        center = mesh.center if mesh.center is not None else np.zeros(3, dtype=np.float32)
        radius = mesh.radius if math.isfinite(mesh.radius) and mesh.radius > 0 else 1.0
        return {
            "id": prepared.model_id,
            "name": mesh.name,
            "source": prepared.source,
            "gaussian_source": prepared.gaussian_source,
            "representation": prepared.representation,
            "mesh": {
                "vertices": mesh.vertices.astype(float).tolist(),
                "faces": mesh.faces.astype(int).tolist(),
                "colors": mesh.vertex_colors.astype(float).tolist(),
                "center": center.astype(float).tolist(),
                "radius": float(radius),
                "source_url": f"/api/model/{prepared.model_id}/source/{quote(source_path.name)}" if source_path else None,
                "source_extension": source_path.suffix.lower() if source_path else None,
            },
            "lods": [
                {
                    "count": lod.count,
                    "name": name,
                    "memory_bytes": lod.memory_bytes() if isinstance(lod, GaussianCloud) else 0,
                    "loaded": isinstance(lod, GaussianCloud),
                }
                for name, lod in sorted(prepared.lods.items(), key=lambda item: int(item[0]))
            ],
        }

    def serialize_lod(self, prepared: PreparedModel, count: str | int) -> dict[str, Any]:
        key = str(count)
        if key not in prepared.lods:
            raise KeyError(f"LOD {key!r} does not exist for model {prepared.model_id!r}.")
        lod = self._materialize_lod(prepared, key)
        return {
            "model_id": prepared.model_id,
            "count": lod.count,
            "xyz": lod.xyz.detach().cpu().numpy().astype(float).tolist(),
            "scale": lod.scale.detach().cpu().numpy().astype(float).tolist(),
            "color": lod.color.detach().cpu().numpy().astype(float).tolist(),
            "opacity": lod.opacity.detach().cpu().numpy().astype(float).reshape(-1).tolist(),
            "rotation": (
                None
                if lod.rotation is None
                else lod.rotation.detach().cpu().numpy().astype(float).tolist()
            ),
        }

    def serialize_lod_binary(self, prepared: PreparedModel, count: str | int) -> bytes:
        key = str(count)
        if key not in prepared.lods:
            raise KeyError(f"LOD {key!r} does not exist for model {prepared.model_id!r}.")
        lod = self._materialize_lod(prepared, key)
        scale = lod.scale.detach().cpu().numpy().astype("<f4", copy=False)
        if scale.ndim == 1 or scale.shape[1] == 1:
            scale = np.repeat(scale.reshape(-1, 1), 3, axis=1)
        rotation = (
            np.tile(np.asarray([1.0, 0.0, 0.0, 0.0], dtype="<f4"), (lod.count, 1))
            if lod.rotation is None
            else lod.rotation.detach().cpu().numpy().astype("<f4", copy=False)
        )
        fields = [
            np.asarray([lod.count], dtype="<u4").tobytes(),
            lod.xyz.detach().cpu().numpy().astype("<f4", copy=False).tobytes(),
            scale.tobytes(),
            lod.color.detach().cpu().numpy().astype("<f4", copy=False).tobytes(),
            lod.opacity.detach().cpu().numpy().astype("<f4", copy=False).reshape(-1).tobytes(),
            rotation.tobytes(),
        ]
        return b"".join(fields)

    def _materialize_lod(self, prepared: PreparedModel, key: str) -> GaussianCloud:
        with prepared.materialize_lock:
            item = prepared.lods[key]
            if isinstance(item, GaussianCloud):
                return item
            if prepared.representation != "mesh2splat_lods":
                raise TypeError(f"Unexpected lazy LOD for representation {prepared.representation!r}.")

            reference_key = prepared.alignment_reference_key
            if reference_key is None:
                raise ValueError("Mesh2Splat preparation is missing an alignment reference LOD.")

            requested_raw: GaussianCloud | None = None
            if prepared.alignment is None:
                reference = prepared.lods[reference_key]
                if isinstance(reference, GaussianCloud):
                    reference_raw = reference
                else:
                    reference_raw = load_trained_gaussian_ply(reference.path)
                    if reference_key == key:
                        requested_raw = reference_raw
                prepared.alignment = self._gaussian_alignment_to_normalized_mesh(reference_raw, prepared.mesh.vertices)

            if requested_raw is None:
                requested_raw = load_trained_gaussian_ply(item.path)
            transformed = self._apply_gaussian_alignment(requested_raw, prepared.alignment)
            prepared.lods[key] = transformed
            return transformed

    @staticmethod
    def path_to_id(path: str | Path) -> str:
        normalized = str(Path(path).resolve())
        digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]
        return f"file:{digest}"

    @staticmethod
    def _normalize_gaussian_to_mesh(cloud: GaussianCloud, center: np.ndarray, radius: float) -> GaussianCloud:
        if radius <= 0.0:
            return cloud
        device = cloud.xyz.device
        center_tensor = torch.as_tensor(center, dtype=cloud.xyz.dtype, device=device)
        return GaussianCloud(
            xyz=(cloud.xyz - center_tensor[None, :]) / radius,
            scale=cloud.scale / radius,
            color=cloud.color,
            opacity=cloud.opacity,
            rotation=cloud.rotation,
            name=cloud.name,
        )

    @staticmethod
    def _align_gaussian_to_normalized_mesh(cloud: GaussianCloud, mesh_vertices: np.ndarray) -> GaussianCloud:
        alignment = ModelStore._gaussian_alignment_to_normalized_mesh(cloud, mesh_vertices)
        return ModelStore._apply_gaussian_alignment(cloud, alignment)

    @staticmethod
    def _gaussian_alignment_to_normalized_mesh(cloud: GaussianCloud, mesh_vertices: np.ndarray) -> GaussianAlignment:
        if cloud.count <= 0 or len(mesh_vertices) == 0:
            return GaussianAlignment(
                cloud_center=np.zeros(3, dtype=np.float32),
                mesh_center=np.zeros(3, dtype=np.float32),
                rotation=np.eye(3, dtype=np.float32),
                scale_factor=np.ones(3, dtype=np.float32),
            )
        mesh_np = np.asarray(mesh_vertices, dtype=np.float32)
        cloud_np = cloud.xyz.detach().cpu().numpy().astype(np.float32)
        opacity_np = cloud.opacity.detach().cpu().numpy().reshape(-1)
        scale_np = cloud.scale.detach().cpu().numpy()
        scale_weight = scale_np.max(axis=1) if scale_np.ndim == 2 else scale_np.reshape(-1)
        weights = np.clip(opacity_np, 0.0, 1.0) * np.maximum(scale_weight, 1.0e-6)

        mesh_min, mesh_max = _robust_bounds(mesh_np)
        cloud_min, cloud_max = _robust_bounds(cloud_np, weights=weights)
        mesh_center_np = (mesh_min + mesh_max) * 0.5
        cloud_center_np = (cloud_min + cloud_max) * 0.5
        rotation_np, scale_factor_np = _best_axis_permutation(cloud_np, mesh_np, cloud_center_np, mesh_center_np)
        return GaussianAlignment(
            cloud_center=cloud_center_np.astype(np.float32),
            mesh_center=mesh_center_np.astype(np.float32),
            rotation=rotation_np.astype(np.float32),
            scale_factor=scale_factor_np.astype(np.float32),
        )

    @staticmethod
    def _apply_gaussian_alignment(cloud: GaussianCloud, alignment: GaussianAlignment) -> GaussianCloud:
        device = cloud.xyz.device
        rotation = torch.as_tensor(alignment.rotation, dtype=cloud.xyz.dtype, device=device)
        mesh_center = torch.as_tensor(alignment.mesh_center, dtype=cloud.xyz.dtype, device=device)
        cloud_center = torch.as_tensor(alignment.cloud_center, dtype=cloud.xyz.dtype, device=device)
        scale_factor = torch.as_tensor(alignment.scale_factor, dtype=cloud.xyz.dtype, device=device)
        transformed_scale, transformed_rotation = _transform_gaussian_covariances(
            cloud.scale.detach().cpu().numpy(),
            None if cloud.rotation is None else cloud.rotation.detach().cpu().numpy(),
            alignment.rotation,
            alignment.scale_factor,
        )
        return GaussianCloud(
            xyz=((cloud.xyz - cloud_center[None, :]) @ rotation.T) * scale_factor[None, :] + mesh_center[None, :],
            scale=torch.as_tensor(transformed_scale, dtype=cloud.xyz.dtype, device=device),
            color=cloud.color,
            opacity=cloud.opacity,
            rotation=torch.as_tensor(transformed_rotation, dtype=cloud.xyz.dtype, device=device),
            name=cloud.name,
        )

    def id_to_path(self, model_id: str) -> Path:
        for model in self.list_models():
            if model["id"] == model_id:
                return Path(model["source"])
        raise KeyError(f"Unknown model id {model_id!r}.")

    def id_to_trained_path(self, model_id: str | None) -> Path:
        if not model_id:
            raise KeyError("No trained Gaussian model id was provided.")
        for model in self.list_trained_gaussians():
            if model["id"] == model_id:
                return Path(model["source"])
        path = Path(model_id)
        if path.exists() and path.suffix.lower() in SUPPORTED_GAUSSIAN_EXTENSIONS:
            return path
        raise KeyError(f"Unknown trained Gaussian id {model_id!r}.")

    def id_to_mesh2splat_path(self, model_id: str) -> Path:
        for model in self.list_mesh2splat_gaussians():
            if model["id"] == model_id:
                return Path(model["source"])
        raise KeyError(f"Unknown Mesh2Splat Gaussian id {model_id!r}.")

    def find_trained_for_mesh(self, mesh_name: str) -> Path | None:
        key = mesh_name.lower()
        for model in self.list_trained_gaussians():
            path = Path(model["source"])
            if key in path.stem.lower() or key in str(path.parent).lower():
                return path
        return None

    def find_mesh2splat_lod_set(self, mesh_name: str) -> dict[int, Path]:
        mesh_key = mesh_name.lower()
        candidates: dict[int, Path] = {}
        for directory in self.mesh2splat_lod_dirs:
            if not directory.exists():
                continue
            for path in sorted(directory.rglob("*.ply")):
                count = self._count_from_lod_filename(path)
                if count is None:
                    continue
                group_key = self._lod_group_key(path)
                if group_key == mesh_key:
                    candidates[count] = path
        return candidates

    @staticmethod
    def _count_from_lod_filename(path: Path) -> int | None:
        matches = re.findall(r"(\d+)", path.stem)
        return int(matches[-1]) if matches else None

    @staticmethod
    def _lod_group_key(path: Path) -> str:
        stem = re.sub(r"[-_]?(\d+)$", "", path.stem.lower())
        stem = stem.replace("-trained", "").replace("_trained", "")
        return stem or path.parent.name.lower()

    def _display_path_name(self, path: Path) -> str:
        for root in [*self.trained_dirs, *self.mesh2splat_lod_dirs, *self.source_dirs]:
            try:
                relative = path.relative_to(root)
            except ValueError:
                continue
            if len(relative.parts) > 1:
                return " / ".join(relative.parts)
            return path.name
        return f"{path.parent.name} / {path.name}"

    @staticmethod
    def _prepared_id(model_id: str, lod_counts: list[int], source: str) -> str:
        payload = f"{model_id}|{source}|{','.join(str(v) for v in lod_counts)}"
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
