from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np

from src.core.config import ensure_dir, load_config
from src.core.device import DeviceManager
from src.core.progress import StageLogger
from src.evaluation.metrics import Evaluator, popping_score
from src.gaussian.lod import GaussianLODBuilder
from src.gaussian.train import GaussianTrainer
from src.geometry.cameras import CameraRig
from src.geometry.mesh_loader import MeshAsset
from src.render.gaussian_renderer import GaussianRenderer
from src.render.mesh_renderer import SyntheticViewRenderer
from src.transition.blending import LODTransitionController


def tuple3(values: list[float]) -> tuple[float, float, float]:
    return (float(values[0]), float(values[1]), float(values[2]))


def image_size_from_config(cfg: dict[str, Any]) -> tuple[int, int]:
    size = cfg["render"].get("image_size", [192, 192])
    return int(size[0]), int(size[1])


def build_scene(cfg: dict[str, Any]):
    image_size = image_size_from_config(cfg)
    mesh_cfg = cfg.get("mesh", {})
    mesh = MeshAsset.load(
        mesh_cfg.get("path"),
        fallback_color=mesh_cfg.get("color"),
        demo_shape=str(mesh_cfg.get("demo_shape", "uv_sphere")),
    )
    mesh.normalize_to_unit_sphere()

    camera_cfg = cfg.get("camera", {})
    train_cameras = CameraRig.orbit(
        radius=float(camera_cfg.get("train_radius", 2.8)),
        elevations=[float(v) for v in camera_cfg.get("elevations", [-20.0, 0.0, 20.0])],
        azimuth_count=int(camera_cfg.get("train_views", 24)),
        image_size=image_size,
        fov_y_degrees=float(camera_cfg.get("fov_y_degrees", 45.0)),
    )

    demo_cfg = cfg.get("demo", {})
    demo_cameras = CameraRig.transition_path(
        far_radius=float(demo_cfg.get("far_radius", 4.0)),
        near_radius=float(demo_cfg.get("near_radius", 1.25)),
        frames=int(demo_cfg.get("frames", 32)),
        image_size=image_size,
        azimuth_degrees=float(demo_cfg.get("azimuth_degrees", 35.0)),
        elevation_degrees=float(demo_cfg.get("elevation_degrees", 10.0)),
        fov_y_degrees=float(camera_cfg.get("fov_y_degrees", 45.0)),
    )
    return mesh, train_cameras, demo_cameras


def build_lods(cfg: dict[str, Any], mesh: MeshAsset, device) -> dict:
    lod_cfg = cfg.get("lod", {})
    counts = [int(value) for value in lod_cfg.get("counts", [10, 100, 500, 5000, 20000])]
    points, normals, colors = mesh.sample_surface(n=max(counts), seed=int(lod_cfg.get("seed", 0)))
    builder = GaussianLODBuilder(
        points=points,
        normals=normals,
        colors=colors,
        seed=int(lod_cfg.get("seed", 0)),
        low_lod_scale_boost=float(lod_cfg.get("low_lod_scale_boost", 1.8)),
    )
    return builder.build_nested(counts=counts, device=device)


def blend_frame(mesh_rgb: np.ndarray, gaussian_rgbs: dict[str, np.ndarray], weights) -> np.ndarray:
    rgb = weights.mesh * mesh_rgb
    for name, weight in weights.gaussian_lods.items():
        if weight > 0.0 and name in gaussian_rgbs:
            rgb = rgb + weight * gaussian_rgbs[name]
    return np.clip(rgb, 0.0, 1.0)


def save_video(frames: list[np.ndarray], path: Path, fps: int) -> Path:
    uint8_frames = [(frame * 255.0).astype(np.uint8) for frame in frames]
    try:
        imageio.mimsave(path, uint8_frames, fps=fps, macro_block_size=1)
        return path
    except Exception:
        gif_path = path.with_suffix(".gif")
        imageio.mimsave(gif_path, uint8_frames, duration=1.0 / max(fps, 1))
        return gif_path


def apply_overrides(
    cfg: dict[str, Any],
    mesh_path: str | None = None,
    demo_shape: str | None = None,
    output_root: str | None = None,
) -> dict[str, Any]:
    if mesh_path is not None:
        cfg.setdefault("mesh", {})["path"] = None if mesh_path.lower() in {"demo", "none", "null"} else mesh_path
    if demo_shape is not None:
        cfg.setdefault("mesh", {})["demo_shape"] = demo_shape
        if mesh_path is None:
            cfg.setdefault("mesh", {})["path"] = None
    if output_root is not None:
        cfg.setdefault("outputs", {})["root"] = output_root
    return cfg


def run_pipeline(
    config_path: str | Path,
    mesh_path: str | None = None,
    demo_shape: str | None = None,
    output_root: str | None = None,
) -> dict[str, Any]:
    cfg = load_config(config_path)
    cfg = apply_overrides(cfg, mesh_path=mesh_path, demo_shape=demo_shape, output_root=output_root)
    progress_cfg = cfg.get("progress", {})
    logger = StageLogger(
        enabled=bool(progress_cfg.get("enabled", True)),
        verbose=bool(progress_cfg.get("verbose", True)),
    )

    with logger.stage("device setup"):
        device_info = DeviceManager(cfg.get("device", {})).resolve()

    with logger.stage("output setup"):
        output_root = ensure_dir(cfg.get("outputs", {}).get("root", "data/outputs"))
        frame_dir = ensure_dir(output_root / "frames")

    with logger.stage("mesh loading and camera generation"):
        mesh, train_cameras, demo_cameras = build_scene(cfg)
        image_size = image_size_from_config(cfg)
        render_cfg = cfg.get("render", {})
        background = tuple3(render_cfg.get("background", [0.04, 0.045, 0.055]))

    with logger.stage("renderer setup"):
        mesh_renderer = SyntheticViewRenderer(
            image_size=image_size,
            background=background,
            backend=str(render_cfg.get("mesh_backend", "software")),
        )
        gaussian_renderer = GaussianRenderer(
            image_size=image_size,
            device=device_info.torch_device,
            background=background,
            backend=str(render_cfg.get("gaussian_backend", "torch")),
        )
        transition = LODTransitionController(cfg.get("transition", {}))

    with logger.stage("synthetic view rendering"):
        train_views = [
            mesh_renderer.render(mesh, camera, outputs=["rgb", "depth"])
            for camera in logger.iter(train_cameras, "synthetic views", total=len(train_cameras))
        ]

    with logger.stage("surface sampling and LOD building"):
        lods = build_lods(cfg, mesh, device_info.torch_device)
        if cfg.get("outputs", {}).get("save_lods", True):
            for name, lod in logger.iter(lods.items(), "saving LOD files", total=len(lods)):
                lod.save_npz(output_root / f"lod_{name}.npz")

    if cfg.get("training", {}).get("enabled", False):
        with logger.stage("optional Gaussian refinement"):
            for lod in logger.iter(lods.values(), "training LODs", total=len(lods)):
                GaussianTrainer(lod, train_views, train_cameras, device_info.torch_device).optimize(
                    steps=int(cfg["training"].get("steps", 50)),
                    learning_rate=float(cfg["training"].get("learning_rate", 0.01)),
                    learn=["color", "opacity", "scale"],
                    freeze=["xyz"],
                )

    frames: list[np.ndarray] = []
    weights_log: list[dict[str, Any]] = []
    with logger.stage("transition rendering"):
        for frame_index, camera in logger.iter(
            enumerate(demo_cameras),
            "transition frames",
            total=len(demo_cameras),
        ):
            mesh_rgb = mesh_renderer.render(mesh, camera, outputs=["rgb"])["rgb"]
            weights = transition.weights(camera.distance_to_origin())

            gaussian_rgbs = {}
            for name, weight in weights.gaussian_lods.items():
                if weight > 1.0e-5:
                    gaussian_rgbs[name] = gaussian_renderer.render(lods[name], camera)

            rgb = blend_frame(mesh_rgb, gaussian_rgbs, weights)
            frames.append(rgb)
            weights_log.append(
                {
                    "frame": frame_index,
                    "distance": camera.distance_to_origin(),
                    "mesh": weights.mesh,
                    "gaussian_lods": weights.gaussian_lods,
                }
            )

            if cfg.get("outputs", {}).get("save_frames", True):
                imageio.imwrite(frame_dir / f"frame_{frame_index:04d}.png", (rgb * 255.0).astype(np.uint8))

    with logger.stage("video saving"):
        fps = int(cfg.get("outputs", {}).get("video_fps", 12))
        video_path = save_video(frames, output_root / "transition.mp4", fps=fps)

    with logger.stage("metrics evaluation and writing"):
        evaluator = Evaluator(logger=logger)
        reference = mesh_renderer.render(mesh, demo_cameras[len(demo_cameras) // 2], outputs=["rgb"])["rgb"]
        metrics = {
            "device": {
                "name": device_info.name,
                "backend": device_info.backend,
                "description": device_info.description,
            },
            "mesh": {
                "name": mesh.name,
                "vertices": int(len(mesh.vertices)),
                "faces": int(len(mesh.faces)),
            },
            "lods": {name: {"gaussians": lod.count, "memory_bytes": lod.memory_bytes()} for name, lod in lods.items()},
            "render_performance": evaluator.render_performance(gaussian_renderer, lods, demo_cameras[0]),
            "quality_by_lod": evaluator.quality_by_lod(
                reference,
                gaussian_renderer,
                lods,
                demo_cameras[len(demo_cameras) // 2],
            ),
            "transition": {
                "frames": len(frames),
                "popping_score": popping_score(frames),
                "weights": weights_log,
            },
            "outputs": {
                "video": str(video_path),
                "frames": str(frame_dir),
            },
        }
        evaluator.save(output_root / "metrics.json", metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run mesh-to-Gaussian LOD transition prototype.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--mesh",
        default=None,
        help="Path to OBJ/PLY/GLTF/GLB. Use 'demo' to force the procedural demo model.",
    )
    parser.add_argument(
        "--demo-shape",
        choices=["uv_sphere", "sphere", "cube"],
        default=None,
        help="Procedural model to generate when --mesh is omitted or set to 'demo'.",
    )
    parser.add_argument("--output", default=None, help="Override outputs.root from the config.")
    args = parser.parse_args()
    metrics = run_pipeline(
        args.config,
        mesh_path=args.mesh,
        demo_shape=args.demo_shape,
        output_root=args.output,
    )
    print(f"Done. Device: {metrics['device']['name']} | video: {metrics['outputs']['video']}")


if __name__ == "__main__":
    main()
