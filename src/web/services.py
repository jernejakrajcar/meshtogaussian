from __future__ import annotations

import hashlib
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

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
        for directory in [*self.trained_dirs, *self.mesh2splat_lod_dirs]:
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
            mesh.normalize_to_unit_sphere()
            gaussian_source = None
            if representation == "trained":
                trained_path = self.id_to_trained_path(trained_ply_id) if trained_ply_id else self.find_trained_for_mesh(mesh.name)
                if trained_path is None:
                    raise FileNotFoundError(
                        "No trained Gaussian .ply was provided or found. Generate one with train_gaussians_from_mesh.py first."
                    )
                trained_cloud = load_trained_gaussian_ply(trained_path)
                lods = build_trained_lods(trained_cloud, lod_counts)
                gaussian_source = str(trained_path)
            elif representation == "mesh2splat_lods":
                lod_paths = self.find_mesh2splat_lod_set(mesh.name)
                if not lod_paths:
                    raise FileNotFoundError(
                        f"No Mesh2Splat LOD .ply set was found for mesh {mesh.name!r} under data/mesh2splats."
                    )
                lods = {
                    str(count): load_trained_gaussian_ply(path)
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
            "scale": lod.scale.detach().cpu().numpy().astype(float).reshape(-1).tolist(),
            "color": lod.color.detach().cpu().numpy().astype(float).tolist(),
            "opacity": lod.opacity.detach().cpu().numpy().astype(float).reshape(-1).tolist(),
        }

    @staticmethod
    def path_to_id(path: str | Path) -> str:
        normalized = str(Path(path).resolve())
        digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]
        return f"file:{digest}"

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
