from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from src.core.config import ensure_dir
from src.core.progress import StageLogger
from src.geometry.cameras import Camera
from src.geometry.mesh_loader import MeshAsset
from src.render.mesh_renderer import SyntheticViewRenderer


@dataclass(frozen=True)
class SyntheticDatasetManifest:
    root: Path
    images_dir: Path
    sparse_dir: Path
    manifest_path: Path
    image_count: int
    point_count: int


def rotation_matrix_to_quaternion_wxyz(rotation: np.ndarray) -> tuple[float, float, float, float]:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (rotation[2, 1] - rotation[1, 2]) / s
        qy = (rotation[0, 2] - rotation[2, 0]) / s
        qz = (rotation[1, 0] - rotation[0, 1]) / s
    else:
        axis = int(np.argmax(np.diag(rotation)))
        if axis == 0:
            s = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            qw = (rotation[2, 1] - rotation[1, 2]) / s
            qx = 0.25 * s
            qy = (rotation[0, 1] + rotation[1, 0]) / s
            qz = (rotation[0, 2] + rotation[2, 0]) / s
        elif axis == 1:
            s = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            qw = (rotation[0, 2] - rotation[2, 0]) / s
            qx = (rotation[0, 1] + rotation[1, 0]) / s
            qy = 0.25 * s
            qz = (rotation[1, 2] + rotation[2, 1]) / s
        else:
            s = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            qw = (rotation[1, 0] - rotation[0, 1]) / s
            qx = (rotation[0, 2] + rotation[2, 0]) / s
            qy = (rotation[1, 2] + rotation[2, 1]) / s
            qz = 0.25 * s
    quat = np.asarray([qw, qx, qy, qz], dtype=np.float64)
    quat /= max(float(np.linalg.norm(quat)), 1.0e-12)
    return tuple(float(v) for v in quat)


def export_synthetic_colmap_dataset(
    mesh: MeshAsset,
    cameras: list[Camera],
    renderer: SyntheticViewRenderer,
    output_root: str | Path,
    point_count: int = 20000,
    seed: int = 7,
    logger: StageLogger | None = None,
) -> SyntheticDatasetManifest:
    root = ensure_dir(output_root)
    images_dir = ensure_dir(root / "images")
    sparse_dir = ensure_dir(root / "sparse" / "0")
    progress = logger or StageLogger(enabled=False)

    frames = []
    for index, camera in progress.iter(enumerate(cameras, start=1), "synthetic training views", total=len(cameras)):
        image_name = f"{index:04d}.png"
        rgb = renderer.render(mesh, camera, outputs=["rgb"])["rgb"]
        imageio.imwrite(images_dir / image_name, (rgb * 255.0).astype(np.uint8))
        view = camera.view_matrix
        qvec = rotation_matrix_to_quaternion_wxyz(view[:3, :3])
        tvec = tuple(float(v) for v in view[:3, 3])
        fx, fy, cx, cy = camera.intrinsics
        frames.append(
            {
                "image_id": index,
                "file_path": f"images/{image_name}",
                "width": camera.width,
                "height": camera.height,
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
                "qvec": qvec,
                "tvec": tvec,
                "transform_matrix": np.linalg.inv(view).astype(float).tolist(),
            }
        )

    with progress.stage(f"seed point sampling ({point_count:,} points)"):
        points, _, colors = mesh.sample_surface(point_count, seed=seed)
    with progress.stage("COLMAP text export"):
        _write_colmap_text(sparse_dir, frames, points, colors)
        _write_seed_points_ply(root / "initial_points.ply", points, colors)

    manifest = {
        "format": "synthetic_colmap_text",
        "images": frames,
        "initial_point_cloud": "initial_points.ply",
        "sparse_colmap": "sparse/0",
        "point_count": int(point_count),
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (root / "transforms.json").write_text(json.dumps({"frames": frames}, indent=2), encoding="utf-8")
    return SyntheticDatasetManifest(root, images_dir, sparse_dir, manifest_path, len(cameras), int(point_count))


def _write_colmap_text(
    sparse_dir: Path,
    frames: list[dict],
    points: np.ndarray,
    colors: np.ndarray,
) -> None:
    first = frames[0]
    camera_line = (
        f"1 PINHOLE {first['width']} {first['height']} "
        f"{first['fx']:.8f} {first['fy']:.8f} {first['cx']:.8f} {first['cy']:.8f}"
    )
    (sparse_dir / "cameras.txt").write_text(
        "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n" + camera_line + "\n",
        encoding="utf-8",
    )

    image_lines = ["# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"]
    for frame in frames:
        q = frame["qvec"]
        t = frame["tvec"]
        name = Path(frame["file_path"]).name
        image_lines.append(
            f"{frame['image_id']} {q[0]:.10f} {q[1]:.10f} {q[2]:.10f} {q[3]:.10f} "
            f"{t[0]:.10f} {t[1]:.10f} {t[2]:.10f} 1 {name}\n"
        )
        image_lines.append(f"{first['width'] * 0.5:.3f} {first['height'] * 0.5:.3f} 1\n")
    (sparse_dir / "images.txt").write_text("".join(image_lines), encoding="utf-8")

    point_lines = ["# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n"]
    rgb = np.clip(colors * 255.0, 0.0, 255.0).astype(int)
    for idx, (point, color) in enumerate(zip(points, rgb), start=1):
        point_lines.append(
            f"{idx} {point[0]:.8f} {point[1]:.8f} {point[2]:.8f} "
            f"{color[0]} {color[1]} {color[2]} 0.0 1 0\n"
        )
    (sparse_dir / "points3D.txt").write_text("".join(point_lines), encoding="utf-8")


def _write_seed_points_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    rgb = np.clip(colors * 255.0, 0.0, 255.0).astype(int)
    lines = [
        "ply\n",
        "format ascii 1.0\n",
        f"element vertex {len(points)}\n",
        "property float x\n",
        "property float y\n",
        "property float z\n",
        "property uchar red\n",
        "property uchar green\n",
        "property uchar blue\n",
        "end_header\n",
    ]
    for point, color in zip(points, rgb):
        lines.append(f"{point[0]:.8f} {point[1]:.8f} {point[2]:.8f} {color[0]} {color[1]} {color[2]}\n")
    path.write_text("".join(lines), encoding="utf-8")
