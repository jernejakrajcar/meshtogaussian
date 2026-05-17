from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from src.conversion.mesh2splat_runner import (
    Mesh2SplatConfig,
    build_mesh2splat_command,
    convert_mesh_to_glb,
)
from src.gaussian.trained_io import build_trained_lods, load_trained_gaussian_ply
from src.geometry.cameras import CameraRig
from src.geometry.mesh_loader import MeshAsset
from src.render.mesh_renderer import SyntheticViewRenderer
from src.training.dataset_export import export_synthetic_colmap_dataset
from src.training.gsplat_runner import build_gsplat_command, check_cuda_training_environment
from src.web.services import ModelStore
from src.gaussian.model import GaussianCloud
import torch


def _write_tiny_textured_glb(path: Path) -> None:
    trimesh = pytest.importorskip("trimesh")
    Image = pytest.importorskip("PIL.Image")
    from trimesh.visual.material import PBRMaterial
    from trimesh.visual.texture import TextureVisuals

    vertices = np.asarray(
        [
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    uvs = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    texture = Image.fromarray(
        np.asarray(
            [
                [[255, 0, 0], [0, 255, 0]],
                [[0, 0, 255], [255, 255, 0]],
            ],
            dtype=np.uint8,
        ),
        "RGB",
    )
    visual = TextureVisuals(uv=uvs, material=PBRMaterial(baseColorTexture=texture, metallicFactor=0.0))
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, visual=visual, process=False)
    mesh.export(path)


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


def test_textured_glb_load_render_and_surface_sampling(tmp_path: Path) -> None:
    glb = tmp_path / "textured.glb"
    _write_tiny_textured_glb(glb)

    mesh = MeshAsset.load(glb)
    assert mesh.uvs is not None
    assert mesh.texture_image is not None
    assert mesh.texture_image.shape[:2] == (2, 2)

    mesh.normalize_to_unit_sphere()
    cameras = CameraRig.orbit(radius=2.0, elevations=[0.0], azimuth_count=1, image_size=(48, 48))
    rgb = SyntheticViewRenderer((48, 48)).render(mesh, cameras[0], outputs=["rgb"])["rgb"]
    foreground = rgb[np.linalg.norm(rgb - np.asarray([0.04, 0.045, 0.055], dtype=np.float32), axis=2) > 0.05]
    assert foreground.size > 0
    assert float(foreground.std()) > 0.02

    _, _, colors = mesh.sample_surface(64, seed=3)
    assert colors.shape == (64, 3)
    assert float(colors.std()) > 0.05


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
    capped = build_trained_lods(cloud, [10])
    assert capped["10"].count == 4


def test_trained_gaussian_binary_ply_loader(tmp_path: Path) -> None:
    ply = tmp_path / "trained_binary.ply"
    header = "\n".join(
        [
            "ply",
            "format binary_little_endian 1.0",
            "element vertex 2",
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
            "",
        ]
    ).encode("ascii")
    rows = [
        (0.0, 0.0, 0.0, 0.9, 0.02, 0.02, 0.02, 1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0, 0.5, 0.03, 0.03, 0.03, 0.0, 1.0, 0.0),
    ]
    ply.write_bytes(header + b"".join(struct.pack("<10f", *row) for row in rows))
    cloud = load_trained_gaussian_ply(ply)
    assert cloud.count == 2
    assert np.allclose(
        cloud.scale.detach().cpu().numpy(),
        [[0.02, 0.02, 0.02], [0.03, 0.03, 0.03]],
        atol=1.0e-6,
    )


def test_trained_gaussian_loader_uses_visible_anisotropic_scale(tmp_path: Path) -> None:
    ply = tmp_path / "anisotropic.ply"
    ply.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 1",
                "property float x",
                "property float y",
                "property float z",
                "property float scale_0",
                "property float scale_1",
                "property float scale_2",
                "property float rot_0",
                "property float rot_1",
                "property float rot_2",
                "property float rot_3",
                "end_header",
                "0 0 0 -6 -6 -22 2 0 0 0",
            ]
        ),
        encoding="utf-8",
    )
    cloud = load_trained_gaussian_ply(ply)
    scale = cloud.scale.detach().cpu().numpy()
    assert np.allclose(scale, [[np.exp(-6), np.exp(-6), 0.001]], atol=1.0e-6)
    assert cloud.rotation is not None
    assert np.allclose(cloud.rotation.detach().cpu().numpy(), [[1, 0, 0, 0]], atol=1.0e-6)


def test_trained_gaussian_alignment_corrects_signed_axis_permutation() -> None:
    mesh_vertices = np.asarray(
        [
            [-2.0, -1.0, -0.5],
            [2.0, -1.0, -0.5],
            [2.0, 1.0, -0.5],
            [-2.0, 1.0, -0.5],
            [-1.5, -0.5, 0.5],
            [1.0, 0.7, 0.5],
        ],
        dtype=np.float32,
    )
    rotation = np.asarray([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    cloud_xyz = (mesh_vertices @ rotation.T) * 0.35 + np.asarray([0.4, -0.2, 0.1], dtype=np.float32)
    cloud = GaussianCloud(
        xyz=torch.as_tensor(cloud_xyz, dtype=torch.float32),
        scale=torch.full((len(cloud_xyz), 3), 0.02),
        color=torch.ones((len(cloud_xyz), 3)),
        opacity=torch.ones((len(cloud_xyz), 1)),
        name="trained",
    )
    aligned = ModelStore._align_gaussian_to_normalized_mesh(cloud, mesh_vertices)
    aligned_np = aligned.xyz.detach().cpu().numpy()
    assert np.allclose(aligned_np.min(axis=0), mesh_vertices.min(axis=0), atol=1.0e-4)
    assert np.allclose(aligned_np.max(axis=0), mesh_vertices.max(axis=0), atol=1.0e-4)


def test_mesh2splat_command_uses_headless_contract(tmp_path: Path) -> None:
    config = Mesh2SplatConfig(
        executable=tmp_path / "Mesh2Splat.exe",
        output_dir=tmp_path / "out",
        glb_cache_dir=tmp_path / "glb",
        density=1.25,
    )
    command = build_mesh2splat_command(config, tmp_path / "input.glb", tmp_path / "output.ply")
    assert command[:2] == [str(config.executable), "--headless"]
    assert "--input" in command
    assert "--output" in command
    assert "--density" in command
    assert "1.25" in command


def test_mesh2splat_glb_input_bypasses_conversion(tmp_path: Path) -> None:
    glb = tmp_path / "model.glb"
    glb.write_bytes(b"glTF")
    assert convert_mesh_to_glb(glb, tmp_path / "cache") == glb


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
