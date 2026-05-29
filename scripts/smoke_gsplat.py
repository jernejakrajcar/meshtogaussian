"""krajši Python smoke test za gsplat rasterizacijo (test)

Datoteka ni del končnega pipeline-a, ampak preveri, ali lahko gsplat v trenutnem
okolju nariše majhen primer brez napake.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from gsplat.rendering import rasterization


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in PyTorch. Install a CUDA-enabled PyTorch build first.")

    device = torch.device("cuda")
    width, height = 160, 120
    count = 64

    torch.manual_seed(7)
    means = torch.randn((count, 3), device=device) * 0.35
    means[:, 2] += 2.0

    quats = torch.zeros((count, 4), device=device)
    quats[:, 0] = 1.0
    scales = torch.full((count, 3), 0.045, device=device)
    opacities = torch.full((count,), 0.82, device=device)
    colors = torch.rand((count, 3), device=device)

    viewmats = torch.eye(4, device=device)[None, :, :]
    intrinsics = torch.tensor(
        [[120.0, 0.0, width / 2.0], [0.0, 120.0, height / 2.0], [0.0, 0.0, 1.0]],
        device=device,
    )[None, :, :]

    rendered, alphas, meta = rasterization(
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmats,
        intrinsics,
        width,
        height,
        backgrounds=torch.tensor([0.02, 0.025, 0.03], device=device),
    )

    image = rendered[0].detach().clamp(0.0, 1.0).cpu().numpy()
    image_u8 = (image * 255.0).astype(np.uint8)

    output_dir = Path("data/gsplat_smoke")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "gsplat_smoke.ppm"
    with output_path.open("wb") as handle:
        handle.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        handle.write(image_u8.tobytes())

    print("gsplat smoke test OK")
    print(f"torch: {torch.__version__}")
    print(f"cuda: {torch.version.cuda}")
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"rendered: {tuple(rendered.shape)}")
    print(f"alphas: {tuple(alphas.shape)}")
    print(f"visible gaussians: {int(meta['radii'].gt(0).sum().item())}")
    print(f"wrote: {output_path}")


if __name__ == "__main__":
    main()
