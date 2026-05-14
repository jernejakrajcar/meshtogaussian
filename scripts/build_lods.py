from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.config import ensure_dir, load_config
from src.core.device import DeviceManager
from src.pipeline.run_pipeline import build_lods, build_scene


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Gaussian LOD npz files.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out", default="data/outputs")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = DeviceManager(cfg.get("device", {})).resolve()
    mesh, _, _ = build_scene(cfg)
    lods = build_lods(cfg, mesh, device.torch_device)
    out_dir = ensure_dir(Path(args.out))
    for name, lod in lods.items():
        lod.save_npz(out_dir / f"lod_{name}.npz")
    print(f"Saved {len(lods)} LOD files to {out_dir}")


if __name__ == "__main__":
    main()
