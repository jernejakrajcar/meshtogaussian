from __future__ import annotations

from pathlib import Path

import numpy as np

from src.gaussian.trained_io import build_trained_lods, load_trained_gaussian_ply
from src.geometry.cameras import CameraRig
from src.geometry.mesh_loader import MeshAsset
from src.render.mesh_renderer import SyntheticViewRenderer
from src.training.dataset_export import export_synthetic_colmap_dataset
from src.training.gsplat_runner import build_gsplat_command, check_cuda_training_environment


def test_synthetic_dataset_export_writes_colmap_files(tmp_path: Path) -> None:
    mesh = MeshAsset.create_demo_cube()
    mesh.normalize_to_unit_sphere()
    cameras = CameraRig.orbit(radius=2.0, elevations=[0.0], azimuth_count=2, image_size=(32, 32))
    manifest = export_synthetic_colmap_dataset(
        mesh=mesh,
        cameras=cameras,
        renderer=SyntheticViewRenderer((32, 32)),
        output_root=tmp_path / "dataset",
        point_count=12,
        seed=2,
    )
    assert manifest.image_count == 2
    assert (manifest.images_dir / "0001.png").exists()
    assert (manifest.sparse_dir / "cameras.txt").exists()
    assert (manifest.sparse_dir / "images.txt").exists()
    assert (manifest.sparse_dir / "points3D.txt").exists()
    assert manifest.manifest_path.exists()


def test_trained_gaussian_ascii_ply_loader_and_lods(tmp_path: Path) -> None:
    ply = tmp_path / "trained.ply"
    ply.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 4",
                "property float x",
                "property float y",
                "property float z",
                "property float opacity",
                "property float scale_0",
                "property float scale_1",
                "property float scale_2",
                "property float f_dc_0",
                "property float f_dc_1",
                "property float f_dc_2",
                "end_header",
                "0 0 0 2.0 -4 -4 -4 1 0 0",
                "1 0 0 1.0 -4 -4 -4 0 1 0",
                "0 1 0 0.5 -4 -4 -4 0 0 1",
                "0 0 1 0.0 -4 -4 -4 1 1 1",
            ]
        ),
        encoding="utf-8",
    )
    cloud = load_trained_gaussian_ply(ply)
    assert cloud.count == 4
    assert cloud.xyz.shape == (4, 3)
    assert cloud.opacity.detach().cpu().numpy().max() > 0.8
    assert np.isfinite(cloud.color.detach().cpu().numpy()).all()
    lods = build_trained_lods(cloud, [2, 4])
    assert {name: lod.count for name, lod in lods.items()} == {"2": 2, "4": 4}


def test_gsplat_command_points_at_simple_trainer(tmp_path: Path) -> None:
    command = build_gsplat_command(
        gsplat_repo=tmp_path / "gsplat",
        data_dir=tmp_path / "dataset",
        result_dir=tmp_path / "result",
        python_executable="python",
        steps=100,
        extra_args=["--save_ply"],
    )
    assert "simple_trainer.py" in command.argv[1]
    assert "--data_dir" in command.argv
    assert "--max_steps" in command.argv
    assert "--save_ply" in command.argv


def test_cuda_preflight_reports_missing_repo(tmp_path: Path) -> None:
    status = check_cuda_training_environment(tmp_path / "missing_gsplat")
    assert "cuda_available" in status
    assert status["gsplat_repo_ok"] is False
    assert any("simple_trainer.py" in problem for problem in status["problems"])
