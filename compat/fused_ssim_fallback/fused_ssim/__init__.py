"""Lokalni fallback za fused_ssim - dodano zaradi kompatibilnosti za gspla env

omogoči, da gsplat runna tudi, ko CUDA implementacija SSIM ni nameščena
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def fused_ssim(
    x: torch.Tensor,
    y: torch.Tensor,
    padding: str = "valid",
    data_range: float = 1.0,
    size_average: bool = True,
    **_: object,
) -> torch.Tensor:
    """Small pure-PyTorch SSIM fallback for gsplat's example trainer.

    The real fused_ssim package is a CUDA optimization. This keeps the same
    import/function shape so training can run when that optional extension cannot
    be compiled locally.
    """
    if x.shape != y.shape:
        raise ValueError(f"fused_ssim fallback expected matching shapes, got {x.shape} and {y.shape}")
    if x.ndim != 4:
        raise ValueError(f"fused_ssim fallback expected NCHW tensors, got {x.shape}")

    kernel_size = 11 if min(x.shape[-2:]) >= 11 else 3
    pad = 0 if padding == "valid" else kernel_size // 2

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    mu_x = F.avg_pool2d(x, kernel_size, stride=1, padding=pad)
    mu_y = F.avg_pool2d(y, kernel_size, stride=1, padding=pad)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x = F.avg_pool2d(x * x, kernel_size, stride=1, padding=pad) - mu_x2
    sigma_y = F.avg_pool2d(y * y, kernel_size, stride=1, padding=pad) - mu_y2
    sigma_xy = F.avg_pool2d(x * y, kernel_size, stride=1, padding=pad) - mu_xy

    score = ((2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)) / ((mu_x2 + mu_y2 + c1) * (sigma_x + sigma_y + c2))
    return score.mean() if size_average else score.flatten(1).mean(dim=1)
