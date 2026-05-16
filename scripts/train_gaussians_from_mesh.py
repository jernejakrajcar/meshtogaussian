from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.config import load_config
from src.geometry.mesh_loader import MeshAsset
from src.pipeline.run_pipeline import image_size_from_config, tuple3
from src.render.mesh_renderer import SyntheticViewRenderer
from src.training.dataset_export import export_synthetic_colmap_dataset
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
    args = parser.parse_args()

    cfg = load_config(args.config)
    mesh_cfg = cfg.get("mesh", {})
    mesh_path = None if args.mesh.lower() in {"demo", "none", "null"} else args.mesh
    mesh = MeshAsset.load(mesh_path, fallback_color=mesh_cfg.get("color"), demo_shape=mesh_cfg.get("demo_shape", "uv_sphere"))
    mesh.normalize_to_unit_sphere()

    image_size = image_size_from_config(cfg)
    camera_cfg = cfg.get("camera", {})
    training_cfg = cfg.get("training_dataset", {})
    cameras = CameraRig.orbit(
        radius=float(training_cfg.get("radius", camera_cfg.get("train_radius", 2.8))),
        elevations=[float(v) for v in training_cfg.get("elevations", camera_cfg.get("elevations", [-20.0, 0.0, 20.0]))],
        azimuth_count=int(training_cfg.get("views_per_elevation", camera_cfg.get("train_views", 24))),
        image_size=image_size,
        fov_y_degrees=float(camera_cfg.get("fov_y_degrees", 45.0)),
    )
    render_cfg = cfg.get("render", {})
    renderer = SyntheticViewRenderer(image_size=image_size, background=tuple3(render_cfg.get("background", [0.04, 0.045, 0.055])))

    dataset_root = Path(args.dataset_out or training_cfg.get("root", "data/generated_datasets")) / mesh.name
    manifest = export_synthetic_colmap_dataset(
        mesh=mesh,
        cameras=cameras,
        renderer=renderer,
        output_root=dataset_root,
        point_count=int(training_cfg.get("initial_point_count", cfg.get("lod", {}).get("max_gaussians", 20000))),
        seed=int(cfg.get("lod", {}).get("seed", 7)),
    )
    print(f"Generated dataset: {manifest.root}")

    gsplat_cfg = cfg.get("gsplat", {})
    result_dir = Path(args.result_out or gsplat_cfg.get("result_dir", "data/trained_gaussians")) / mesh.name
    command = build_gsplat_command(
        gsplat_repo=gsplat_cfg.get("repo", "../gsplat"),
        data_dir=manifest.root,
        result_dir=result_dir,
        python_executable=gsplat_cfg.get("python_executable"),
        steps=int(gsplat_cfg.get("steps", 3000)),
        data_factor=int(gsplat_cfg.get("data_factor", 1)),
        extra_args=[str(arg) for arg in gsplat_cfg.get("extra_args", [])],
    )
    if args.run_trainer and not args.skip_cuda_check:
        status = check_cuda_training_environment(gsplat_cfg.get("repo", "../gsplat"))
        if status["problems"]:
            print("CUDA/gsplat preflight failed:")
            for problem in status["problems"]:
                print(f"- {problem}")
            raise SystemExit(1)

    summary = run_gsplat_training(command, execute=args.run_trainer)
    print("gsplat command:")
    print(command.as_shell_string())
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


if __name__ == "__main__":
    main()
