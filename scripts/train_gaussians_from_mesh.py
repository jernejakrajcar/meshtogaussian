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
from src.training.gsplat_runner import build_gsplat_command, run_gsplat_training
from src.geometry.cameras import CameraRig


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate mesh training views and optionally run gsplat.")
    parser.add_argument("--mesh", required=True, help="Input mesh path, or 'demo' for procedural demo.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset-out", default=None)
    parser.add_argument("--result-out", default=None)
    parser.add_argument("--run-trainer", action="store_true", help="Actually execute gsplat after generating the dataset.")
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
    )
    summary = run_gsplat_training(command, execute=args.run_trainer)
    print("gsplat command:")
    print(command.as_shell_string())
    if not args.run_trainer:
        print("Training was not executed. Add --run-trainer after installing/configuring gsplat.")
    print(f"Command summary: {result_dir / 'gsplat_command.json'}")


if __name__ == "__main__":
    main()
