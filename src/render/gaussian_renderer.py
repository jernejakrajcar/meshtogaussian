from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.gaussian.model import GaussianCloud
from src.geometry.cameras import Camera


@dataclass
class GaussianRenderer:
    image_size: tuple[int, int]
    device: torch.device | str = "cpu"
    background: tuple[float, float, float] = (0.04, 0.045, 0.055)
    backend: str = "torch"

    def render(self, cloud: GaussianCloud, camera: Camera) -> np.ndarray:
        if self.backend not in {"torch", "software", "auto"}:
            raise ValueError(f"Unsupported Gaussian backend for this prototype: {self.backend}")
        return self._render_torch(cloud.to(self.device), camera)

    def _render_torch(self, cloud: GaussianCloud, camera: Camera) -> np.ndarray:
        width, height = self.image_size
        device = cloud.xyz.device
        dtype = cloud.xyz.dtype

        view = torch.as_tensor(camera.view_matrix, dtype=dtype, device=device)
        ones = torch.ones((cloud.count, 1), dtype=dtype, device=device)
        pts_h = torch.cat([cloud.xyz, ones], dim=1)
        pts_cam = (view @ pts_h.T).T[:, :3]
        z = pts_cam[:, 2]
        visible = z < -0.01
        if int(visible.sum().item()) == 0:
            return np.tile(np.asarray(self.background, dtype=np.float32)[None, None, :], (height, width, 1))

        pts_cam = pts_cam[visible]
        color = cloud.color[visible]
        opacity = cloud.opacity[visible].clamp(0.0, 1.0)
        scale = cloud.scale[visible].clamp_min(1.0e-4)
        z = pts_cam[:, 2]

        fx, fy, cx, cy = camera.intrinsics
        px = fx * (pts_cam[:, 0] / -z) + cx
        py = fy * (-pts_cam[:, 1] / -z) + cy
        sigma = (fx * scale[:, 0] / -z).clamp(0.75, 32.0)
        depth_order = torch.argsort(-z, descending=True)

        accum_rgb = torch.zeros((height, width, 3), dtype=dtype, device=device)
        accum_alpha = torch.zeros((height, width, 1), dtype=dtype, device=device)

        # Back-to-front alpha compositing approximates splat transparency while
        # keeping this fallback renderer readable and dependency-light.
        for idx in depth_order.tolist():
            x = float(px[idx].item())
            y = float(py[idx].item())
            s = float(sigma[idx].item())
            if x < -3.0 * s or x > width + 3.0 * s or y < -3.0 * s or y > height + 3.0 * s:
                continue

            radius = max(1, int(3.0 * s))
            min_x = max(0, int(x) - radius)
            max_x = min(width - 1, int(x) + radius)
            min_y = max(0, int(y) - radius)
            max_y = min(height - 1, int(y) + radius)
            if min_x > max_x or min_y > max_y:
                continue

            yy = torch.arange(min_y, max_y + 1, dtype=dtype, device=device)[:, None]
            xx = torch.arange(min_x, max_x + 1, dtype=dtype, device=device)[None, :]
            dist2 = (xx - x) ** 2 + (yy - y) ** 2
            alpha = torch.exp(-0.5 * dist2 / (s * s))[:, :, None] * opacity[idx]
            alpha = alpha.clamp(0.0, 0.98)

            patch_alpha = accum_alpha[min_y : max_y + 1, min_x : max_x + 1]
            patch_rgb = accum_rgb[min_y : max_y + 1, min_x : max_x + 1]
            contribution = (1.0 - patch_alpha) * alpha
            patch_rgb += contribution * color[idx][None, None, :]
            patch_alpha += contribution

        background = torch.as_tensor(self.background, dtype=dtype, device=device).view(1, 1, 3)
        image = accum_rgb + (1.0 - accum_alpha) * background
        return image.clamp(0.0, 1.0).detach().cpu().numpy()
