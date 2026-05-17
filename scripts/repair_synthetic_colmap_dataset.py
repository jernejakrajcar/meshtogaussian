from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair synthetic COLMAP text files for gsplat's pycolmap loader.")
    parser.add_argument("dataset", help="Dataset root, e.g. data/generated_datasets/plant")
    args = parser.parse_args()

    root = Path(args.dataset)
    manifest_path = root / "manifest.json"
    sparse_dir = root / "sparse" / "0"
    images_path = sparse_dir / "images.txt"
    points_path = sparse_dir / "points3D.txt"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames = manifest.get("images", [])
    if not frames:
        raise RuntimeError(f"No frames found in {manifest_path}")

    lines = ["# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"]
    for frame in frames:
        q = frame["qvec"]
        t = frame["tvec"]
        name = Path(frame["file_path"]).name
        lines.append(
            f"{frame['image_id']} {q[0]:.10f} {q[1]:.10f} {q[2]:.10f} {q[3]:.10f} "
            f"{t[0]:.10f} {t[1]:.10f} {t[2]:.10f} 1 {name}\n"
        )
        lines.append(f"{frame['width'] * 0.5:.3f} {frame['height'] * 0.5:.3f} 1\n")

    images_path.write_text("".join(lines), encoding="utf-8")
    _repair_points3d(points_path)
    print(f"Repaired {images_path}")
    print(f"Repaired {points_path}")
    print(f"Images: {len(frames)}")


def _repair_points3d(path: Path) -> None:
    repaired = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            repaired.append(line + "\n")
            continue
        parts = line.split()
        if len(parts) == 8:
            parts.extend(["1", "0"])
        repaired.append(" ".join(parts) + "\n")
    path.write_text("".join(repaired), encoding="utf-8")


if __name__ == "__main__":
    main()
