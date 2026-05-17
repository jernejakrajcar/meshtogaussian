from __future__ import annotations

import hashlib
import itertools
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np
import torch

from src.core.progress import StageLogger
from src.gaussian.lod import GaussianLODBuilder
from src.gaussian.model import GaussianCloud
from src.gaussian.trained_io import build_trained_lods, load_trained_gaussian_ply
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
    scale_factor: np.float32,
    sample_count: int = 2500,
) -> np.ndarray:
    cloud_sample = _even_sample(cloud_xyz, sample_count)
    mesh_sample = _even_sample(mesh_vertices, sample_count)
    best_rotation = np.eye(3, dtype=np.float32)
    best_score = float("inf")
    for permutation in itertools.permutations(range(3)):
        base = np.zeros((3, 3), dtype=np.float32)
        for row, column in enumerate(permutation):
            base[row, column] = 1.0
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            rotation = (np.asarray(signs, dtype=np.float32)[:, None] * base).astype(np.float32)
            transformed_cloud = ((cloud_sample - cloud_center[None, :]) @ rotation.T) * scale_factor + mesh_center[None, :]
            score = _mean_nearest_distance(transformed_cloud, mesh_sample)
            if score < best_score:
                best_score = score
                best_rotation = rotation
    return best_rotation


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


@dataclass
class PreparedModel:
    model_id: str
    mesh: MeshAsset
    lods: dict[str, GaussianCloud]
    source: str
    gaussian_source: str | None = None
    representation: str = "initialized"


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
                models.append({"id": self.path_to_id(path), "name": path.name, "source": str(path)})
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
        with self.logger.stage("web model preparation"):
            source = "generated"
            mesh_path: Path | None = None
            if model_id and not model_id.startswith("demo:"):
                mesh_path = self.id_to_path(model_id)
                source = str(mesh_path)

            mesh = MeshAsset.load(mesh_path, fallback_color=fallback_color)
            mesh_center, mesh_radius = mesh.normalize_to_unit_sphere()
            gaussian_source = None
            if representation == "trained":
                trained_path = self.id_to_trained_path(trained_ply_id) if trained_ply_id else self.find_trained_for_mesh(mesh.name)
                if trained_path is None:
                    raise FileNotFoundError(
                        "No trained Gaussian .ply was provided or found. Generate one with train_gaussians_from_mesh.py first."
                    )
                trained_cloud = self._align_gaussian_to_normalized_mesh(load_trained_gaussian_ply(trained_path), mesh.vertices)
                trained_counts = sorted({count for count in lod_counts if count < trained_cloud.count} | {trained_cloud.count})
                lods = build_trained_lods(trained_cloud, trained_counts)
                gaussian_source = str(trained_path)
            elif representation == "mesh2splat_lods":
                lod_paths = self.find_mesh2splat_lod_set(mesh.name)
                if not lod_paths:
                    raise FileNotFoundError(
                        f"No Mesh2Splat LOD .ply set was found for mesh {mesh.name!r} under data/mesh2splats."
                    )
                lods = {
                    str(count): self._normalize_gaussian_to_mesh(load_trained_gaussian_ply(path), mesh_center, mesh_radius)
                    for count, path in sorted(lod_paths.items())
                }
                gaussian_source = "; ".join(str(path) for _, path in sorted(lod_paths.items()))
            else:
                points, normals, colors = mesh.sample_surface(max(lod_counts), seed=seed)
                lods = GaussianLODBuilder(points, normals, colors, seed=seed).build_nested(lod_counts)
            prepared_id = self._prepared_id(f"{model_id or 'demo'}:{representation}:{gaussian_source}", lod_counts, source)
            prepared = PreparedModel(
                model_id=prepared_id,
                mesh=mesh,
                lods=lods,
                source=source,
                gaussian_source=gaussian_source,
                representation=representation,
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
                    "memory_bytes": lod.memory_bytes(),
                }
                for name, lod in sorted(prepared.lods.items(), key=lambda item: int(item[0]))
            ],
        }

    def serialize_lod(self, prepared: PreparedModel, count: str | int) -> dict[str, Any]:
        key = str(count)
        if key not in prepared.lods:
            raise KeyError(f"LOD {key!r} does not exist for model {prepared.model_id!r}.")
        lod = prepared.lods[key]
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
        if cloud.count <= 0 or len(mesh_vertices) == 0:
            return cloud
        device = cloud.xyz.device
        mesh_np = np.asarray(mesh_vertices, dtype=np.float32)
        cloud_np = cloud.xyz.detach().cpu().numpy().astype(np.float32)

        mesh_min = mesh_np.min(axis=0)
        mesh_max = mesh_np.max(axis=0)
        cloud_min = cloud_np.min(axis=0)
        cloud_max = cloud_np.max(axis=0)
        mesh_center_np = (mesh_min + mesh_max) * 0.5
        cloud_center_np = (cloud_min + cloud_max) * 0.5
        mesh_extent = max(float(np.max(mesh_max - mesh_min)), 1.0e-6)
        cloud_extent = max(float(np.max(cloud_max - cloud_min)), 1.0e-6)
        scale_factor_np = np.clip(mesh_extent / cloud_extent, 0.01, 100.0).astype(np.float32)
        rotation_np = _best_axis_permutation(cloud_np, mesh_np, cloud_center_np, mesh_center_np, scale_factor_np)

        rotation = torch.as_tensor(rotation_np, dtype=cloud.xyz.dtype, device=device)
        mesh_center = torch.as_tensor(mesh_center_np, dtype=cloud.xyz.dtype, device=device)
        cloud_center = torch.as_tensor(cloud_center_np, dtype=cloud.xyz.dtype, device=device)
        scale_factor = torch.as_tensor(float(scale_factor_np), dtype=cloud.xyz.dtype, device=device)
        return GaussianCloud(
            xyz=((cloud.xyz - cloud_center[None, :]) @ rotation.T) * scale_factor + mesh_center[None, :],
            scale=cloud.scale * scale_factor,
            color=cloud.color,
            opacity=cloud.opacity,
            rotation=cloud.rotation,
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
        fallback: dict[int, Path] = {}
        for directory in self.mesh2splat_lod_dirs:
            if not directory.exists():
                continue
            for path in sorted(directory.rglob("*.ply")):
                count = self._count_from_lod_filename(path)
                if count is None:
                    continue
                group_key = self._lod_group_key(path)
                if mesh_key in group_key or group_key in mesh_key:
                    candidates[count] = path
                fallback[count] = path
        return candidates or fallback

    @staticmethod
    def _count_from_lod_filename(path: Path) -> int | None:
        matches = re.findall(r"(\d+)", path.stem)
        return int(matches[-1]) if matches else None

    @staticmethod
    def _lod_group_key(path: Path) -> str:
        stem = re.sub(r"[-_]?(\d+)$", "", path.stem.lower())
        stem = stem.replace("-trained", "").replace("_trained", "")
        return stem or path.parent.name.lower()

    @staticmethod
    def _prepared_id(model_id: str, lod_counts: list[int], source: str) -> str:
        payload = f"{model_id}|{source}|{','.join(str(v) for v in lod_counts)}"
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
