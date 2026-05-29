"""Osnovni testi za glavni pipeline.

preverja geometrijo, kamere, LOD štetje, mešanje prehodov in kratek CPU smoke test
"""

from __future__ import annotations

import numpy as np

from src.gaussian.lod import GaussianLODBuilder
from src.geometry.cameras import CameraRig
from src.geometry.mesh_loader import MeshAsset
from src.pipeline.run_pipeline import apply_overrides, build_scene
from src.render.gaussian_renderer import GaussianRenderer
from src.render.mesh_renderer import SyntheticViewRenderer
from src.transition.blending import LODTransitionController


def test_mesh_normalization_fits_unit_sphere() -> None:
    mesh = MeshAsset.create_demo_sphere(segments=12, rings=6)
    mesh.vertices *= 4.0
    mesh.vertices += np.array([3.0, -2.0, 1.0], dtype=np.float32)
    mesh.normalize_to_unit_sphere()
    assert np.linalg.norm(mesh.vertices, axis=1).max() <= 1.0001
    assert np.abs((mesh.vertices.min(axis=0) + mesh.vertices.max(axis=0)) * 0.5).max() < 1.0e-5


def test_camera_points_toward_origin() -> None:
    camera = CameraRig.orbit(2.0, [0.0], 1, (64, 64))[0]
    forward = camera.target - camera.eye
    forward = forward / np.linalg.norm(forward)
    view_forward = -camera.view_matrix[2, :3]
    assert np.allclose(forward, view_forward, atol=1.0e-5)


def test_lod_counts_are_exact() -> None:
    mesh = MeshAsset.create_demo_sphere(segments=16, rings=8)
    points, normals, colors = mesh.sample_surface(500, seed=3)
    lods = GaussianLODBuilder(points, normals, colors, seed=3).build_nested([10, 100, 500])
    assert {name: lod.count for name, lod in lods.items()} == {"10": 10, "100": 100, "500": 500}


def test_transition_weights_sum_to_one() -> None:
    transition = LODTransitionController(
        {
            "mesh_fade_start": 3.6,
            "mesh_fade_end": 2.2,
            "lod_ranges": {"10": [4.2, 3.1], "100": [3.6, 2.5], "500": [3.0, 1.9]},
        }
    )
    for distance in np.linspace(1.0, 4.5, 20):
        weights = transition.weights(float(distance))
        assert abs(weights.total() - 1.0) < 1.0e-5
        assert 0.0 <= weights.mesh <= 1.0


def test_short_cpu_render_smoke() -> None:
    mesh = MeshAsset.create_demo_sphere(segments=10, rings=5)
    mesh.normalize_to_unit_sphere()
    camera = CameraRig.transition_path(2.5, 1.5, 1, (48, 48), 30.0, 10.0)[0]
    mesh_rgb = SyntheticViewRenderer((48, 48)).render(mesh, camera, outputs=["rgb"])["rgb"]
    assert mesh_rgb.shape == (48, 48, 3)

    points, normals, colors = mesh.sample_surface(25, seed=1)
    cloud = GaussianLODBuilder(points, normals, colors, seed=1).build_nested([25])["25"]
    gaussian_rgb = GaussianRenderer((48, 48)).render(cloud, camera)
    assert gaussian_rgb.shape == (48, 48, 3)
    assert np.isfinite(gaussian_rgb).all()


def test_pipeline_cli_demo_shape_override() -> None:
    cfg = {
        "mesh": {"path": "some-file.obj", "demo_shape": "uv_sphere"},
        "render": {"image_size": [32, 32]},
        "camera": {"train_views": 1, "elevations": [0.0]},
        "demo": {"frames": 1},
    }
    apply_overrides(cfg, mesh_path="demo", demo_shape="cube")
    mesh, _, _ = build_scene(cfg)
    assert mesh.name == "demo_cube"
    assert len(mesh.faces) == 12
