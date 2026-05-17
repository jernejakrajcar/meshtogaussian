from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a real-photo COLMAP dataset for gsplat training.")
    parser.add_argument("--images", required=True, help="Folder containing real photos.")
    parser.add_argument("--name", required=True, help="Dataset/model name, for example plant_real.")
    parser.add_argument("--out-root", default="data/real_datasets", help="Output root for COLMAP datasets.")
    parser.add_argument("--colmap-exe", default="colmap", help="Path to colmap.exe, or 'colmap' if it is in PATH.")
    parser.add_argument("--camera-model", default="OPENCV", help="COLMAP camera model. OPENCV is a good default for phone photos.")
    parser.add_argument("--matcher", choices=["exhaustive", "sequential"], default="exhaustive")
    parser.add_argument("--single-camera", action="store_true", default=True, help="Assume all photos use the same camera.")
    parser.add_argument("--no-single-camera", dest="single_camera", action="store_false")
    parser.add_argument("--cpu", action="store_true", help="Disable COLMAP GPU SIFT extraction/matching.")
    parser.add_argument("--overwrite", action="store_true", help="Delete and rebuild the output dataset directory.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running COLMAP.")
    args = parser.parse_args()

    image_source = Path(args.images).resolve()
    dataset_root = (Path(args.out_root) / args.name).resolve()
    if not image_source.exists() or not image_source.is_dir():
        raise FileNotFoundError(f"Image folder was not found: {image_source}")
    if dataset_root.exists() and args.overwrite:
        shutil.rmtree(dataset_root)

    images_dir = dataset_root / "images"
    sparse_dir = dataset_root / "sparse"
    database_path = dataset_root / "database.db"
    images_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    copied = copy_images(image_source, images_dir)
    if len(copied) < 8:
        raise ValueError(f"Only {len(copied)} image(s) found. Use at least 20-40 overlapping photos for a useful 3DGS dataset.")

    gpu = "0" if args.cpu else "1"
    commands = [
        [
            args.colmap_exe,
            "feature_extractor",
            "--database_path",
            str(database_path),
            "--image_path",
            str(images_dir),
            "--ImageReader.camera_model",
            args.camera_model,
            "--ImageReader.single_camera",
            "1" if args.single_camera else "0",
            "--SiftExtraction.use_gpu",
            gpu,
        ],
        [
            args.colmap_exe,
            "exhaustive_matcher" if args.matcher == "exhaustive" else "sequential_matcher",
            "--database_path",
            str(database_path),
            "--SiftMatching.use_gpu",
            gpu,
        ],
        [
            args.colmap_exe,
            "mapper",
            "--database_path",
            str(database_path),
            "--image_path",
            str(images_dir),
            "--output_path",
            str(sparse_dir),
            "--Mapper.ba_refine_focal_length",
            "1",
            "--Mapper.ba_refine_principal_point",
            "1",
            "--Mapper.ba_refine_extra_params",
            "1",
        ],
    ]

    summary = {
        "dataset_root": str(dataset_root),
        "images": len(copied),
        "image_source": str(image_source),
        "camera_model": args.camera_model,
        "matcher": args.matcher,
        "single_camera": args.single_camera,
        "commands": commands,
    }
    (dataset_root / "real_dataset_command.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.dry_run:
        print_summary(summary)
        return

    for command in commands:
        print("\n[run] " + " ".join(quote_arg(part) for part in command), flush=True)
        subprocess.run(command, cwd=dataset_root, check=True)

    if not (sparse_dir / "0").exists():
        reconstructions = sorted(path for path in sparse_dir.iterdir() if path.is_dir())
        raise RuntimeError(f"COLMAP did not create sparse/0. Reconstructions found: {[path.name for path in reconstructions]}")

    print_summary(summary)
    print("\nDataset ready for gsplat:")
    print(dataset_root)


def copy_images(source: Path, target: Path) -> list[Path]:
    copied: list[Path] = []
    for index, path in enumerate(sorted(source.rglob("*")), start=1):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        name = f"{index:04d}{path.suffix.lower()}"
        destination = target / name
        if not destination.exists() or path.stat().st_size != destination.stat().st_size:
            shutil.copy2(path, destination)
        copied.append(destination)
    return copied


def print_summary(summary: dict) -> None:
    print("\nReal COLMAP dataset summary:")
    print(f"- dataset: {summary['dataset_root']}")
    print(f"- images: {summary['images']}")
    print(f"- camera_model: {summary['camera_model']}")
    print(f"- matcher: {summary['matcher']}")


def quote_arg(value: str) -> str:
    return f'"{value}"' if " " in value else value


if __name__ == "__main__":
    main()
