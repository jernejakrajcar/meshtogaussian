from __future__ import annotations

from pathlib import Path

import pytest

from src.core.progress import StageError, StageLogger
from src.geometry.mesh_loader import MeshAsset
from src.web.services import ModelStore, is_supported_mesh, safe_upload_name


def test_stage_logger_failure_names_stage() -> None:
    logger = StageLogger(enabled=False, verbose=False)
    try:
        with logger.stage("important step"):
            raise ValueError("broken")
    except StageError as exc:
        assert "important step" in str(exc)
        assert "broken" in str(exc)
    else:
        raise AssertionError("StageError was not raised")


def test_supported_upload_extensions() -> None:
    assert is_supported_mesh("model.obj")
    assert is_supported_mesh("model.GLB")
    assert safe_upload_name("../my mesh.ply") == "my_mesh.ply"
    try:
        safe_upload_name("notes.txt")
    except ValueError:
        pass
    else:
        raise AssertionError("Unsupported extension was accepted")


def test_model_discovery_from_source_dir(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.obj").write_text("# demo", encoding="utf-8")
    (source / "ignore.txt").write_text("no", encoding="utf-8")
    store = ModelStore(source_dirs=[source], upload_dir=tmp_path / "uploads", logger=StageLogger(False, False))
    models = store.list_models()
    assert any(model["name"] == "model.obj" for model in models)
    assert not any(model["name"] == "ignore.txt" for model in models)


def test_web_serialization_shapes() -> None:
    store = ModelStore(logger=StageLogger(False, False))
    mesh = MeshAsset.create_demo_sphere(segments=8, rings=4)
    prepared = store.prepare(model_id=None, lod_counts=[10], seed=1)
    prepared.mesh = mesh
    serialized = store.serialize_model(prepared)
    lod = store.serialize_lod(prepared, 10)
    assert len(serialized["mesh"]["vertices"]) == len(mesh.vertices)
    assert serialized["lods"][0]["count"] == 10
    assert len(lod["xyz"]) == 10
    assert len(lod["color"]) == 10


def test_fastapi_app_import_when_available() -> None:
    pytest.importorskip("fastapi")
    from src.web.app import create_app

    app = create_app()
    assert app.title == "Mesh-to-Gaussian Visualizer"


def test_fastapi_prepare_returns_viewer_transition_config() -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from src.web.app import create_app

    client = TestClient(create_app("configs/smoke.yaml"))
    response = client.post(
        "/api/prepare",
        json={"model_id": "demo:procedural-sphere", "lod_counts": [10], "seed": 1},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["viewer"]["far_radius"] == 3.0
    assert "transition" in payload["viewer"]
    assert payload["viewer"]["transition"]["lod_ranges"]["10"] == [3.0, 2.3]
