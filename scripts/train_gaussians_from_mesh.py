"""Celoten ukazni workflow od mesha do treniranih Gaussov

Skripta pripravi sintetične poglede, COLMAP strukturo in gsplat ukaz,
glavna pot za ustvarjanje rezultatov iz izbranega 3D modela
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.config import load_config
from src.core.progress import StageLogger
from src.geometry.mesh_loader import MeshAsset
from src.pipeline.run_pipeline import image_size_from_config, tuple3
from src.render.mesh_renderer import SyntheticViewRenderer
from src.training.dataset_export import SyntheticDatasetManifest, export_synthetic_colmap_dataset
from src.training.gsplat_runner import (
    build_gsplat_command,
    check_cuda_training_environment,
    find_latest_trained_ply,
    run_gsplat_training,
)
from src.geometry.cameras import CameraRig


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Experimental future-work path: generate mesh training views and optionally run gsplat. "
            "The main project workflow uses Mesh2Splat-exported PLY LODs."
        )
    )
    parser.add_argument("--mesh", required=True, help="Input mesh path, or 'demo' for procedural demo.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset-out", default=None)
    parser.add_argument("--result-out", default=None)
    parser.add_argument("--run-trainer", action="store_true", help="Actually execute gsplat after generating the dataset.")
    parser.add_argument("--skip-cuda-check", action="store_true", help="Do not fail early when CUDA is unavailable.")
    parser.add_argument("--reuse-dataset", action="store_true", help="Use an existing synthetic COLMAP dataset if it already exists.")
    parser.add_argument("--gsplat-repo", default=None, help="Path to a cloned gsplat repository.")
    parser.add_argument("--gsplat-python", default=None, help="Python executable from the CUDA/gsplat virtual environment.")
    parser.add_argument("--research-defaults", action="store_true", help="Use denser object-centric camera and seed-point defaults.")
    parser.add_argument("--views-per-elevation", type=int, default=None)
    parser.add_argument("--elevations", default=None, help="Comma-separated elevation angles in degrees.")
    parser.add_argument("--initial-point-count", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument(
        "--trainer-arg",
        action="append",
        default=[],
        help="Extra argument forwarded to gsplat simple_trainer.py. Repeat for multiple args; use --trainer-arg=--flag for flags.",
    )
    parser.add_argument("--quiet", action="store_true", help="Hide progress output while generating the dataset.")
    args = parser.parse_args()
    logger = StageLogger(enabled=not args.quiet, verbose=True)

    def info(message: str) -> None:
        if not args.quiet:
            print(message, flush=True)

    with logger.stage("configuration and mesh loading"):
        cfg = load_config(args.config)
        mesh_cfg = cfg.get("mesh", {})
        # 'demo' je poseben vhod za proceduralni model; vse ostalo obravnavam
        # kot pot do prave mesh datoteke.
        mesh_path = None if args.mesh.lower() in {"demo", "none", "null"} else args.mesh
        mesh = MeshAsset.load(mesh_path, fallback_color=mesh_cfg.get("color"), demo_shape=mesh_cfg.get("demo_shape", "uv_sphere"))
        _, mesh_radius = mesh.normalize_to_unit_sphere()
        info(f"[info] mesh: {mesh.name} | vertices: {len(mesh.vertices):,} | faces: {len(mesh.faces):,} | radius: {mesh_radius:.4f}")

    image_size = image_size_from_config(cfg)
    camera_cfg = cfg.get("camera", {})
    training_cfg = cfg.get("training_dataset", {})
    default_elevations = (
        training_cfg.get("research_elevations", [-45.0, -30.0, -15.0, 0.0, 15.0, 30.0, 45.0])
        if args.research_defaults
        else camera_cfg.get("elevations", [-20.0, 0.0, 20.0])
    )
    # Prioriteta je: ekspliciten CLI, nato raziskovalni defaulti, nato config.
    # Tako lahko hitro preglasim samo en parameter eksperimenta.
    if args.elevations:
        elevations = _float_list(args.elevations)
    elif args.research_defaults:
        elevations = [float(v) for v in default_elevations]
    else:
        elevations = [float(v) for v in training_cfg.get("elevations", default_elevations)]
    views_per_elevation = args.views_per_elevation or int(
        training_cfg.get("research_views_per_elevation", 36)
        if args.research_defaults
        else training_cfg.get("views_per_elevation", camera_cfg.get("train_views", 24))
    )
    initial_point_count = args.initial_point_count or int(
        training_cfg.get("research_initial_point_count", 100000)
        if args.research_defaults
        else training_cfg.get("initial_point_count", cfg.get("lod", {}).get("max_gaussians", 20000))
    )
    cameras = CameraRig.orbit(
        radius=float(training_cfg.get("radius", camera_cfg.get("train_radius", 2.8))),
        elevations=elevations,
        azimuth_count=views_per_elevation,
        image_size=image_size,
        fov_y_degrees=float(camera_cfg.get("fov_y_degrees", 45.0)),
    )
    info(
        "[info] training dataset: "
        f"{len(cameras):,} views ({len(elevations)} elevations x {views_per_elevation}), "
        f"image size {image_size[0]}x{image_size[1]}, seed points {initial_point_count:,}"
    )
    render_cfg = cfg.get("render", {})
    renderer = SyntheticViewRenderer(
        image_size=image_size,
        background=tuple3(render_cfg.get("background", [0.04, 0.045, 0.055])),
        backend=str(render_cfg.get("mesh_backend", "software")),
    )

    dataset_root = Path(args.dataset_out or training_cfg.get("root", "data/generated_datasets")) / mesh.name
    # Reuse je uporaben, ker je renderiranje datasetov pocasno; vseeno preverim,
    # da manifest obstaja in kasneje tudi osnovne COLMAP datoteke.
    if args.reuse_dataset and (dataset_root / "manifest.json").exists():
        with logger.stage("synthetic COLMAP dataset reuse"):
            manifest = _load_existing_manifest(dataset_root)
        print(f"Reused dataset: {manifest.root}")
    else:
        with logger.stage("synthetic COLMAP dataset export"):
            manifest = export_synthetic_colmap_dataset(
                mesh=mesh,
                cameras=cameras,
                renderer=renderer,
                output_root=dataset_root,
                point_count=initial_point_count,
                seed=int(cfg.get("lod", {}).get("seed", 7)),
                logger=logger,
            )
        print(f"Generated dataset: {manifest.root}")
    info(f"[info] images: {manifest.image_count:,} | initial points: {manifest.point_count:,}")

    gsplat_cfg = cfg.get("gsplat", {})
    gsplat_repo = _resolve_existing_path(args.gsplat_repo or gsplat_cfg.get("repo", "../gsplat"))
    gsplat_python_raw = args.gsplat_python or gsplat_cfg.get("python_executable")
    gsplat_python = str(_resolve_existing_path(gsplat_python_raw)) if gsplat_python_raw else None
    result_dir = Path(args.result_out or gsplat_cfg.get("result_dir", "data/trained_gaussians")) / mesh.name
    command = build_gsplat_command(
        gsplat_repo=gsplat_repo,
        data_dir=manifest.root,
        result_dir=result_dir,
        python_executable=gsplat_python,
        steps=int(args.steps or gsplat_cfg.get("steps", 7000 if args.research_defaults else 3000)),
        data_factor=int(gsplat_cfg.get("data_factor", 1)),
        extra_args=[str(arg) for arg in gsplat_cfg.get("extra_args", [])] + [str(arg) for arg in args.trainer_arg],
    )
    # CUDA preflight je samo za dejanski trening. Pri dry-run pripravi ukaza ni
    # treba zahtevati GPU okolja.
    if args.run_trainer and not args.skip_cuda_check:
        with logger.stage("CUDA/gsplat preflight"):
            status = check_cuda_training_environment(gsplat_repo, python_executable=gsplat_python)
            if status["problems"]:
                print("CUDA/gsplat preflight failed:")
                for key in ["python_executable", "torch_version", "cuda_version", "cuda_available", "cuda_device_count", "devices"]:
                    if key in status:
                        print(f"- {key}: {status[key]}")
                for problem in status["problems"]:
                    print(f"- {problem}")
                raise SystemExit(1)

    with logger.stage("gsplat command preparation" if not args.run_trainer else "gsplat training"):
        summary = run_gsplat_training(command, execute=args.run_trainer)
    print("gsplat command:")
    print(command.as_shell_string())
    # Privzeto skripta ostane varna in samo pripravi ukaz/dataset; training se
    # zazene sele z eksplicitnim --run-trainer.
    if not args.run_trainer:
        print("Training was not executed. This gsplat route is experimental future work; the main workflow uses Mesh2Splat-exported PLY LODs.")
        print("Add --run-trainer after installing/configuring gsplat if you want to test it.")
    else:
        trained_ply = find_latest_trained_ply(result_dir)
        if trained_ply is None:
            print("Training finished, but no .ply was found under the result directory.")
            print("If your gsplat version saves only checkpoints, export/convert the checkpoint to PLY before using the viewer.")
        else:
            print(f"Latest trained PLY: {trained_ply}")
    print(f"Command summary: {result_dir / 'gsplat_command.json'}")


def _float_list(raw: str) -> list[float]:
    return [float(value.strip()) for value in raw.split(",") if value.strip()]


def _resolve_existing_path(raw: str | Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else (ROOT / path).resolve()


def _load_existing_manifest(root: Path) -> SyntheticDatasetManifest:
    manifest_path = root / "manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    images = data.get("images", [])
    point_count = int(data.get("point_count", 0))
    required = [
        root / "images",
        root / "sparse" / "0" / "cameras.txt",
        root / "sparse" / "0" / "images.txt",
        root / "sparse" / "0" / "points3D.txt",
        root / "initial_points.ply",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Existing dataset is incomplete:\n" + "\n".join(missing))
    return SyntheticDatasetManifest(
        root=root,
        images_dir=root / "images",
        sparse_dir=root / "sparse" / "0",
        manifest_path=manifest_path,
        image_count=len(images),
        point_count=point_count,
    )


if __name__ == "__main__":
    main()
