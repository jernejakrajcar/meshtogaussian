"""Skripta za generiranje sintetičnih pogledov iz mesha.

Uporabimo jo za pripravo slik in kamer, ki potem služijo kot vhod za trening
Gaussovih splattov oziroma za preverjanje renderiranja.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import imageio.v2 as imageio

from src.core.config import ensure_dir, load_config
from src.pipeline.run_pipeline import build_scene, image_size_from_config, tuple3
from src.render.mesh_renderer import SyntheticViewRenderer


def main() -> None:
    parser = argparse.ArgumentParser(description="Render synthetic mesh training views.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out", default="data/outputs/synthetic_views")
    args = parser.parse_args()

    cfg = load_config(args.config)
    mesh, train_cameras, _ = build_scene(cfg)
    render_cfg = cfg.get("render", {})
    renderer = SyntheticViewRenderer(
        image_size=image_size_from_config(cfg),
        background=tuple3(render_cfg.get("background", [0.04, 0.045, 0.055])),
    )
    out_dir = ensure_dir(Path(args.out))
    for index, view in enumerate(renderer.render_batch(mesh, train_cameras, outputs=["rgb"])):
        imageio.imwrite(out_dir / f"view_{index:04d}.png", (view["rgb"] * 255.0).astype("uint8"))
    print(f"Saved {len(train_cameras)} synthetic views to {out_dir}")


if __name__ == "__main__":
    main()
