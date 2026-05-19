from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.geometry.cameras import Camera
from src.geometry.mesh_loader import MeshAsset


@dataclass
class SyntheticViewRenderer:
    image_size: tuple[int, int]
    background: tuple[float, float, float] = (0.04, 0.045, 0.055)
    backend: str = "software"

    def render_batch(self, mesh: MeshAsset, cameras: list[Camera], outputs: list[str] | None = None) -> list[dict[str, np.ndarray]]:
        return [self.render(mesh, camera, outputs=outputs) for camera in cameras]

    def render(self, mesh: MeshAsset, camera: Camera, outputs: list[str] | None = None) -> dict[str, np.ndarray]:
        if self.backend not in {"software", "auto"}:
            raise ValueError(f"Unsupported mesh backend for this prototype: {self.backend}")
        outputs = outputs or ["rgb", "depth"]
        rgb, depth = self._software_render(mesh, camera)
        result: dict[str, np.ndarray] = {}
        if "rgb" in outputs:
            result["rgb"] = rgb
        if "depth" in outputs:
            result["depth"] = depth
        return result

    def _software_render(self, mesh: MeshAsset, camera: Camera) -> tuple[np.ndarray, np.ndarray]:
        width, height = self.image_size
        rgb = np.tile(np.asarray(self.background, dtype=np.float32)[None, None, :], (height, width, 1))
        depth = np.full((height, width), np.inf, dtype=np.float32)

        points_cam = self._transform(mesh.vertices, camera.view_matrix)
        projected, valid = self._project(points_cam, camera)
        face_depth = points_cam[mesh.faces, 2].mean(axis=1)
        order = np.argsort(face_depth)[::-1]

        for face_index in order:
            face = mesh.faces[face_index]
            if not np.all(valid[face]):
                continue
            pts = projected[face]
            z = points_cam[face, 2]
            if np.any(z >= -0.01):
                continue

            vertex_colors = mesh.vertex_colors[face]
            face_uvs = None if mesh.uvs is None or mesh.texture_image is None else mesh.uvs[face]
            self._rasterize_triangle(rgb, depth, pts, z, vertex_colors, 1.0, face_uvs, mesh.texture_image)

        depth[~np.isfinite(depth)] = 0.0
        return rgb, depth

    @staticmethod
    def _transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        homogeneous = np.concatenate([points, np.ones((len(points), 1), dtype=np.float32)], axis=1)
        return (matrix @ homogeneous.T).T[:, :3]

    @staticmethod
    def _project(points_cam: np.ndarray, camera: Camera) -> tuple[np.ndarray, np.ndarray]:
        fx, fy, cx, cy = camera.intrinsics
        z = points_cam[:, 2]
        valid = z < -0.01
        x = fx * (points_cam[:, 0] / -z) + cx
        y = fy * (-points_cam[:, 1] / -z) + cy
        projected = np.stack([x, y], axis=1).astype(np.float32)
        valid &= np.isfinite(projected).all(axis=1)
        return projected, valid

    @staticmethod
    def _rasterize_triangle(
        rgb: np.ndarray,
        depth: np.ndarray,
        pts: np.ndarray,
        z_values: np.ndarray,
        vertex_colors: np.ndarray,
        shade: float,
        uvs: np.ndarray | None = None,
        texture: np.ndarray | None = None,
    ) -> None:
        height, width = depth.shape
        min_x = max(0, int(np.floor(np.min(pts[:, 0]))))
        max_x = min(width - 1, int(np.ceil(np.max(pts[:, 0]))))
        min_y = max(0, int(np.floor(np.min(pts[:, 1]))))
        max_y = min(height - 1, int(np.ceil(np.max(pts[:, 1]))))
        if min_x > max_x or min_y > max_y:
            return

        a, b, c = pts
        area = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(float(area)) < 1.0e-8:
            return

        for py in range(min_y, max_y + 1):
            for px in range(min_x, max_x + 1):
                p = np.array([px + 0.5, py + 0.5], dtype=np.float32)
                w0 = ((b[1] - c[1]) * (p[0] - c[0]) + (c[0] - b[0]) * (p[1] - c[1])) / area
                w1 = ((c[1] - a[1]) * (p[0] - c[0]) + (a[0] - c[0]) * (p[1] - c[1])) / area
                w2 = 1.0 - w0 - w1
                if w0 < 0.0 or w1 < 0.0 or w2 < 0.0:
                    continue
                z = w0 * z_values[0] + w1 * z_values[1] + w2 * z_values[2]
                positive_depth = -z
                if positive_depth < depth[py, px]:
                    depth[py, px] = positive_depth
                    weights = np.asarray([w0, w1, w2], dtype=np.float32)
                    rgb[py, px] = SyntheticViewRenderer._surface_color(vertex_colors, weights, shade, uvs, texture)

    @staticmethod
    def _surface_color(
        vertex_colors: np.ndarray,
        weights: np.ndarray,
        shade: float,
        uvs: np.ndarray | None,
        texture: np.ndarray | None,
    ) -> np.ndarray:
        if uvs is not None and texture is not None:
            uv = (uvs * weights[:, None]).sum(axis=0)
            color = SyntheticViewRenderer._sample_texture(texture, uv)
        else:
            color = (vertex_colors * weights[:, None]).sum(axis=0)
        return np.clip(color * shade, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _sample_texture(texture: np.ndarray, uv: np.ndarray) -> np.ndarray:
        height, width = texture.shape[:2]
        wrapped = np.clip(np.asarray(uv, dtype=np.float32), 0.0, 1.0)
        x = float(np.clip(wrapped[0] * (width - 1), 0, width - 1))
        y = float(np.clip((1.0 - wrapped[1]) * (height - 1), 0, height - 1))
        x0 = int(np.floor(x))
        y0 = int(np.floor(y))
        x1 = min(x0 + 1, width - 1)
        y1 = min(y0 + 1, height - 1)
        tx = x - x0
        ty = y - y0
        top = texture[y0, x0] * (1.0 - tx) + texture[y0, x1] * tx
        bottom = texture[y1, x0] * (1.0 - tx) + texture[y1, x1] * tx
        return np.clip(top * (1.0 - ty) + bottom * ty, 0.0, 1.0).astype(np.float32)
