from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class MeshAsset:
    vertices: np.ndarray
    faces: np.ndarray
    vertex_colors: np.ndarray
    uvs: np.ndarray | None = None
    texture_image: np.ndarray | None = None
    name: str = "mesh"
    center: np.ndarray | None = None
    radius: float = 1.0

    @classmethod
    def load(
        cls,
        path: str | Path | None,
        fallback_color: list[float] | None = None,
        demo_shape: str = "uv_sphere",
    ) -> "MeshAsset":
        if path is None or str(path).lower() in {"", "none", "null"}:
            return cls.create_demo_shape(demo_shape, color=fallback_color)

        try:
            import trimesh  # type: ignore
        except Exception as exc:
            if Path(path).suffix.lower() == ".obj":
                return cls._load_simple_obj(Path(path), fallback_color=fallback_color)
            raise RuntimeError(
                "Loading external meshes requires trimesh. Install requirements.txt "
                "or use mesh.path: null for the procedural demo mesh."
            ) from exc

        loaded = trimesh.load(Path(path), force="mesh")
        if hasattr(loaded, "geometry"):
            loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))

        vertices = np.asarray(loaded.vertices, dtype=np.float32)
        faces = np.asarray(loaded.faces, dtype=np.int64)
        colors = cls._extract_vertex_colors(loaded, len(vertices), fallback_color)
        uvs, texture_image = cls._extract_texture_data(loaded, len(vertices))
        return cls(
            vertices=vertices,
            faces=faces,
            vertex_colors=colors,
            uvs=uvs,
            texture_image=texture_image,
            name=Path(path).stem,
        )

    @classmethod
    def _load_simple_obj(cls, path: Path, fallback_color: list[float] | None = None) -> "MeshAsset":
        vertices: list[list[float]] = []
        faces: list[list[int]] = []
        for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0] == "v" and len(parts) >= 4:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f" and len(parts) >= 4:
                polygon = [int(token.split("/")[0]) - 1 for token in parts[1:]]
                for index in range(1, len(polygon) - 1):
                    faces.append([polygon[0], polygon[index], polygon[index + 1]])
        if not vertices or not faces:
            raise ValueError(f"OBJ fallback loader could not read vertices/faces from {path}.")
        vertex_array = np.asarray(vertices, dtype=np.float32)
        face_array = np.asarray(faces, dtype=np.int64)
        color = np.asarray(fallback_color or [0.78, 0.64, 0.42], dtype=np.float32)
        colors = np.tile(color[None, :], (len(vertex_array), 1)).astype(np.float32)
        return cls(vertices=vertex_array, faces=face_array, vertex_colors=colors, name=path.stem)

    @classmethod
    def create_demo_shape(cls, shape: str, color: list[float] | None = None) -> "MeshAsset":
        shape_key = shape.lower().replace("-", "_")
        if shape_key in {"sphere", "uv_sphere"}:
            return cls.create_demo_sphere(color=color)
        if shape_key == "cube":
            return cls.create_demo_cube(color=color)
        raise ValueError(f"Unsupported demo_shape {shape!r}. Use 'uv_sphere' or 'cube'.")

    @staticmethod
    def _extract_vertex_colors(mesh: Any, count: int, fallback_color: list[float] | None) -> np.ndarray:
        default = np.asarray(fallback_color or [0.78, 0.64, 0.42], dtype=np.float32)
        colors = np.tile(default[None, :], (count, 1))
        visual = getattr(mesh, "visual", None)
        vertex_colors = getattr(visual, "vertex_colors", None)
        if vertex_colors is not None and len(vertex_colors) == count:
            arr = np.asarray(vertex_colors, dtype=np.float32)
            if arr.shape[1] >= 3:
                colors = arr[:, :3] / 255.0
        return np.clip(colors, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _extract_texture_data(mesh: Any, count: int) -> tuple[np.ndarray | None, np.ndarray | None]:
        visual = getattr(mesh, "visual", None)
        raw_uvs = getattr(visual, "uv", None)
        material = getattr(visual, "material", None)
        image = getattr(material, "baseColorTexture", None)
        if image is None:
            image = getattr(material, "image", None)
        if raw_uvs is None or image is None or len(raw_uvs) != count:
            return None, None

        uvs = np.asarray(raw_uvs, dtype=np.float32)
        texture = np.asarray(image.convert("RGBA") if hasattr(image, "convert") else image, dtype=np.float32)
        if texture.ndim != 3 or texture.shape[2] < 3:
            return None, None
        rgb = texture[:, :, :3] / 255.0
        if texture.shape[2] >= 4:
            alpha = texture[:, :, 3:4] / 255.0
            rgb = rgb * alpha + (1.0 - alpha)
        return np.clip(uvs, 0.0, 1.0).astype(np.float32), np.clip(rgb, 0.0, 1.0).astype(np.float32)

    @classmethod
    def create_demo_sphere(
        cls,
        segments: int = 48,
        rings: int = 24,
        color: list[float] | None = None,
    ) -> "MeshAsset":
        vertices = []
        colors = []
        base = np.asarray(color or [0.78, 0.64, 0.42], dtype=np.float32)
        for y in range(rings + 1):
            v = y / rings
            theta = np.pi * v
            for x in range(segments):
                u = x / segments
                phi = 2.0 * np.pi * u
                pos = np.array(
                    [np.sin(theta) * np.sin(phi), np.cos(theta), np.sin(theta) * np.cos(phi)],
                    dtype=np.float32,
                )
                vertices.append(pos)
                tint = 0.82 + 0.18 * pos[1]
                colors.append(np.clip(base * tint + np.array([0.08 * u, 0.04 * v, 0.0]), 0.0, 1.0))

        faces = []
        for y in range(rings):
            for x in range(segments):
                a = y * segments + x
                b = y * segments + (x + 1) % segments
                c = (y + 1) * segments + x
                d = (y + 1) * segments + (x + 1) % segments
                faces.append([a, c, b])
                faces.append([b, c, d])

        return cls(
            vertices=np.asarray(vertices, dtype=np.float32),
            faces=np.asarray(faces, dtype=np.int64),
            vertex_colors=np.asarray(colors, dtype=np.float32),
            name="demo_sphere",
        )

    @classmethod
    def create_demo_cube(cls, color: list[float] | None = None) -> "MeshAsset":
        vertices = np.asarray(
            [
                [-1.0, -1.0, -1.0],
                [1.0, -1.0, -1.0],
                [1.0, 1.0, -1.0],
                [-1.0, 1.0, -1.0],
                [-1.0, -1.0, 1.0],
                [1.0, -1.0, 1.0],
                [1.0, 1.0, 1.0],
                [-1.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        )
        faces = np.asarray(
            [
                [0, 2, 1],
                [0, 3, 2],
                [4, 5, 6],
                [4, 6, 7],
                [0, 1, 5],
                [0, 5, 4],
                [2, 3, 7],
                [2, 7, 6],
                [1, 2, 6],
                [1, 6, 5],
                [3, 0, 4],
                [3, 4, 7],
            ],
            dtype=np.int64,
        )
        base = np.asarray(color or [0.78, 0.64, 0.42], dtype=np.float32)
        tint = np.asarray(
            [
                [0.85, 0.90, 1.00],
                [1.00, 0.88, 0.82],
                [0.92, 1.00, 0.86],
                [0.86, 0.95, 1.00],
                [1.00, 0.92, 0.88],
                [0.95, 0.88, 1.00],
                [1.00, 1.00, 0.86],
                [0.88, 1.00, 0.96],
            ],
            dtype=np.float32,
        )
        colors = np.clip(base[None, :] * tint, 0.0, 1.0)
        return cls(vertices=vertices, faces=faces, vertex_colors=colors, name="demo_cube")

    def normalize_to_unit_sphere(self) -> tuple[np.ndarray, float]:
        center = (self.vertices.min(axis=0) + self.vertices.max(axis=0)) * 0.5
        self.vertices = self.vertices - center[None, :]
        radius = np.linalg.norm(self.vertices, axis=1).max()
        if radius > 0.0:
            self.vertices = self.vertices / radius
        self.center = center.astype(np.float32)
        self.radius = float(radius)
        return self.center, self.radius

    def sample_surface(self, n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        triangles = self.vertices[self.faces]
        face_colors = self.vertex_colors[self.faces].mean(axis=1)
        face_uvs = None if self.uvs is None or self.texture_image is None else self.uvs[self.faces]
        face_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        areas = np.linalg.norm(face_normals, axis=1) * 0.5
        valid = areas > 1.0e-12
        triangles = triangles[valid]
        face_colors = face_colors[valid]
        if face_uvs is not None:
            face_uvs = face_uvs[valid]
        face_normals = face_normals[valid]
        areas = areas[valid]
        normals = face_normals / np.maximum(np.linalg.norm(face_normals, axis=1, keepdims=True), 1.0e-8)

        # Sampling faces proportional to triangle area avoids over-representing
        # tiny triangles and gives each part of the surface a fair density.
        probabilities = areas / areas.sum()
        face_indices = rng.choice(len(triangles), size=n, replace=True, p=probabilities)
        chosen = triangles[face_indices]
        r1 = np.sqrt(rng.random(n, dtype=np.float32))
        r2 = rng.random(n, dtype=np.float32)
        weights = np.stack([1.0 - r1, r1 * (1.0 - r2), r1 * r2], axis=1)
        points = (chosen * weights[:, :, None]).sum(axis=1)
        colors = face_colors[face_indices].astype(np.float32)
        if face_uvs is not None and self.texture_image is not None:
            sampled_uvs = (face_uvs[face_indices] * weights[:, :, None]).sum(axis=1)
            colors = self.sample_texture(sampled_uvs)
        return (
            points.astype(np.float32),
            normals[face_indices].astype(np.float32),
            colors.astype(np.float32),
        )

    def sample_texture(self, uvs: np.ndarray) -> np.ndarray:
        if self.texture_image is None:
            raise ValueError("Mesh does not have a base-color texture.")
        texture = self.texture_image
        height, width = texture.shape[:2]
        wrapped = np.clip(np.asarray(uvs, dtype=np.float32), 0.0, 1.0)
        x = np.clip(wrapped[:, 0] * (width - 1), 0, width - 1)
        y = np.clip((1.0 - wrapped[:, 1]) * (height - 1), 0, height - 1)
        x0 = np.floor(x).astype(np.int64)
        y0 = np.floor(y).astype(np.int64)
        x1 = np.minimum(x0 + 1, width - 1)
        y1 = np.minimum(y0 + 1, height - 1)
        tx = (x - x0)[:, None]
        ty = (y - y0)[:, None]
        top = texture[y0, x0] * (1.0 - tx) + texture[y0, x1] * tx
        bottom = texture[y1, x0] * (1.0 - tx) + texture[y1, x1] * tx
        return np.clip(top * (1.0 - ty) + bottom * ty, 0.0, 1.0).astype(np.float32)
