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

        light_dir = np.array([0.2, 0.7, 0.5], dtype=np.float32)
        light_dir /= np.linalg.norm(light_dir)

        for face_index in order:
            face = mesh.faces[face_index]
            if not np.all(valid[face]):
                continue
            pts = projected[face]
            z = points_cam[face, 2]
            if np.any(z >= -0.01):
                continue

            world_tri = mesh.vertices[face]
            normal = np.cross(world_tri[1] - world_tri[0], world_tri[2] - world_tri[0])
            normal /= max(float(np.linalg.norm(normal)), 1.0e-8)
            shade = 0.35 + 0.65 * max(0.0, float(np.dot(normal, light_dir)))
            color = np.clip(mesh.vertex_colors[face].mean(axis=0) * shade, 0.0, 1.0)
            self._rasterize_triangle(rgb, depth, pts, z, color)

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
        color: np.ndarray,
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
                    rgb[py, px] = color
