from __future__ import annotations

from pathlib import Path

import pytest

from src.conversion.mesh2splat_runner import Mesh2SplatResult
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


def test_fastapi_prepare_returns_mesh_transform_and_source_asset(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from src.web.app import create_app

    meshes = tmp_path / "meshes"
    meshes.mkdir(parents=True)
    mesh = meshes / "textured shell.obj"
    mesh.write_text(
        "\n".join(["v 0 0 0", "v 2 0 0", "v 0 2 0", "f 1 2 3"]),
        encoding="utf-8",
    )

    client = TestClient(create_app("configs/smoke.yaml", data_dir=tmp_path))
    model_id = next(model["id"] for model in client.get("/api/models").json()["models"] if model["name"] == mesh.name)
    response = client.post("/api/prepare", json={"model_id": model_id, "lod_counts": [10]})
    assert response.status_code == 200
    payload = response.json()
    assert payload["mesh"]["center"] == [1.0, 1.0, 0.0]
    assert payload["mesh"]["radius"] > 0
    assert payload["mesh"]["source_url"].endswith("/textured%20shell.obj")

    asset_response = client.get(payload["mesh"]["source_url"])
    assert asset_response.status_code == 200
    assert "v 2 0 0" in asset_response.text


def test_fastapi_mesh2splat_conversion_endpoint_uses_trained_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    import src.web.app as web_app

    meshes = tmp_path / "meshes"
    meshes.mkdir(parents=True)
    mesh = meshes / "tiny.obj"
    mesh.write_text(
        "\n".join(
            [
                "v 0 0 0",
                "v 1 0 0",
                "v 0 1 0",
                "f 1 2 3",
            ]
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "trained_gaussians" / "mesh2splat" / "tiny"
    output_dir.mkdir(parents=True)
    output_ply = output_dir / "tiny_mesh2splat.ply"
    output_ply.write_text(
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
                "0 0 0 0.9 0.02 0.02 0.02 1 0 0",
                "1 0 0 0.9 0.02 0.02 0.02 0 1 0",
                "0 1 0 0.9 0.02 0.02 0.02 0 0 1",
                "0 0 1 0.9 0.02 0.02 0.02 1 1 1",
            ]
        ),
        encoding="utf-8",
    )

    def fake_convert(mesh_path, config, density=None):
        return Mesh2SplatResult(
            input_mesh=Path(mesh_path),
            glb_mesh=Path(mesh_path),
            output_ply=output_ply,
            command=["fake"],
            returncode=0,
            stdout="",
            stderr="",
            manifest_path=output_dir / "mesh2splat_command.json",
        )

    monkeypatch.setattr(web_app, "convert_mesh_to_gaussians", fake_convert)
    client = TestClient(web_app.create_app("configs/smoke.yaml", data_dir=tmp_path))
    model_id = next(model["id"] for model in client.get("/api/models").json()["models"] if model["name"] == "tiny.obj")
    response = client.post("/api/convert-mesh2splat", json={"model_id": model_id, "density": 1.0})
    assert response.status_code == 200
    payload = response.json()
    assert payload["representation"] == "trained"
    assert payload["gaussian_source"].endswith("tiny_mesh2splat.ply")
    assert payload["lods"][0]["name"] == "4"
    assert payload["lods"][0]["count"] == 4


def test_fastapi_prepare_uses_mesh2splat_lod_set(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from src.web.app import create_app

    meshes = tmp_path / "meshes"
    lods = tmp_path / "mesh2splats"
    meshes.mkdir(parents=True)
    lods.mkdir(parents=True)
    (meshes / "sourdough.obj").write_text(
        "\n".join(["v 0 0 0", "v 1 0 0", "v 0 1 0", "f 1 2 3"]),
        encoding="utf-8",
    )
    for count in [300, 800]:
        (lods / f"sourdough-trained-{count}.ply").write_text(
            "\n".join(
                [
                    "ply",
                    "format ascii 1.0",
                    "element vertex 2",
                    "property float x",
                    "property float y",
                    "property float z",
                    "property float opacity",
                    "property float scale",
                    "property uchar red",
                    "property uchar green",
                    "property uchar blue",
                    "end_header",
                    "0 0 0 0.9 0.02 255 0 0",
                    "1 0 0 0.9 0.02 0 255 0",
                ]
            ),
            encoding="utf-8",
        )

    client = TestClient(create_app("configs/smoke.yaml", data_dir=tmp_path))
    sets = client.get("/api/mesh2splat-lod-sets").json()["sets"]
    assert sets[0]["counts"] == [300, 800]
    model_id = next(model["id"] for model in client.get("/api/models").json()["models"] if model["name"] == "sourdough.obj")
    response = client.post(
        "/api/prepare",
        json={"model_id": model_id, "representation": "mesh2splat_lods"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["representation"] == "mesh2splat_lods"
    assert [lod["name"] for lod in payload["lods"]] == ["300", "800"]
    assert sorted(payload["viewer"]["transition"]["lod_ranges"]) == ["300", "800"]


def test_fastapi_prepare_trained_returns_representation_metadata(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from src.web.app import create_app

    meshes = tmp_path / "meshes"
    trained = tmp_path / "trained_gaussians"
    meshes.mkdir(parents=True)
    trained.mkdir(parents=True)
    (meshes / "tiny.obj").write_text(
        "\n".join(["v 0 0 0", "v 1 0 0", "v 0 1 0", "f 1 2 3"]),
        encoding="utf-8",
    )
    (trained / "tiny_trained.ply").write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 2",
                "property float x",
                "property float y",
                "property float z",
                "property float opacity",
                "property float scale",
                "property uchar red",
                "property uchar green",
                "property uchar blue",
                "end_header",
                "0 0 0 0.9 0.02 255 0 0",
                "1 0 0 0.9 0.02 0 255 0",
            ]
        ),
        encoding="utf-8",
    )

    client = TestClient(create_app("configs/smoke.yaml", data_dir=tmp_path))
    model_id = next(model["id"] for model in client.get("/api/models").json()["models"] if model["name"] == "tiny.obj")
    trained_id = next(model["id"] for model in client.get("/api/trained-gaussians").json()["models"] if model["name"] == "tiny_trained.ply")
    response = client.post(
        "/api/prepare",
        json={"model_id": model_id, "representation": "trained", "trained_ply_id": trained_id},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["representation"] == "trained"
    assert payload["gaussian_source"].endswith("tiny_trained.ply")
    assert [lod["name"] for lod in payload["lods"]] == ["2"]
    assert sorted(payload["viewer"]["transition"]["lod_ranges"]) == ["2"]


def test_frontend_debug_ui_sections_and_hints() -> None:
    root = Path(__file__).resolve().parents[1]
    html = (root / "web" / "index.html").read_text(encoding="utf-8")
    main = (root / "web" / "main.js").read_text(encoding="utf-8")
    css = (root / "web" / "styles.css").read_text(encoding="utf-8")

    for heading in ["Input", "Gaussian Data", "View", "Camera / Rendering"]:
        assert f"<h2>{heading}</h2>" in html
    assert '<aside class="log-panel">' in html
    assert '<pre id="status">Loading models...</pre>' in html
    assert "Load Selected Setup" in html
    assert "Gaussian data source" in html
    assert "Mesh2Splat LOD files" in html
    assert "Single trained PLY" in html
    assert "Mesh-sampled preview" in html
    assert "Only used for Single trained PLY." in html
    assert "Only for Transition mode; depth-sorts splats for current camera." in html
    assert "For Gaussian/Both view with trained splats." in html
    assert 'id="gaussianYOffset"' in html
    assert 'id="gaussianScale"' in html
    assert "splat-render-3-raw-webgl" in main
    assert 'id="refreshModelsButton"' in html
    assert 'id="lockCameraButton"' in html
    assert "body {" in css
    assert "overflow: hidden;" in css
    assert ".log-panel" in css
    assert "overscroll-behavior: contain;" in css
    assert "statusBox.scrollTop = statusBox.scrollHeight" in main
    assert "DEFAULT_TRAINED_LOD_COUNTS" in main
    assert "function applyGaussianTransform(object)" in main
    assert "createRawGaussianRenderer" in main
    assert "new THREE.ShaderMaterial" not in main
    assert "raw_gaussian_renderer.js" in main
    raw_renderer = (root / "web" / "raw_gaussian_renderer.js").read_text(encoding="utf-8")
    assert "drawElementsInstanced" in raw_renderer
    assert "gl.POINTS" not in raw_renderer
    assert "AUTO_SORT_THRESHOLD" in main


def test_trained_gaussian_dropdown_names_include_parent_folder(tmp_path: Path) -> None:
    from src.core.progress import StageLogger
    from src.web.services import ModelStore

    trained = tmp_path / "trained_gaussians"
    (trained / "plant").mkdir(parents=True)
    (trained / "sourdough").mkdir(parents=True)
    for folder in ["plant", "sourdough"]:
        (trained / folder / "point-cloud-29999.ply").write_text(
            "\n".join(
                [
                    "ply",
                    "format ascii 1.0",
                    "element vertex 1",
                    "property float x",
                    "property float y",
                    "property float z",
                    "end_header",
                    "0 0 0",
                ]
            ),
            encoding="utf-8",
        )

    store = ModelStore(
        source_dirs=[tmp_path / "meshes"],
        upload_dir=tmp_path / "uploads",
        trained_dirs=[trained],
        logger=StageLogger(False, False),
    )
    names = [model["name"] for model in store.list_trained_gaussians()]
    assert "plant / point-cloud-29999.ply" in names
    assert "sourdough / point-cloud-29999.ply" in names


def test_frontend_control_state_rules_are_centralized() -> None:
    root = Path(__file__).resolve().parents[1]
    main = (root / "web" / "main.js").read_text(encoding="utf-8")

    assert "function updateControlAvailability()" in main
    assert 'const usesTrainedPly = representationSelect.value === "trained"' in main
    assert 'setControlAvailability(trainedSelect, usesTrainedPly, "Only used for Single trained PLY.")' in main
    assert 'setControlAvailability(transitionStyleSelect, inTransition, "Only used in Transition mode.")' in main
    assert 'setControlAvailability(transitionSlider, inTransition, "Only used in Transition mode.")' in main
    assert "setControlAvailability(lodSelect, hasPrepared && inLodView" in main
    assert "Disabled in Transition mode because the transition blends LODs automatically." in main
    assert "setControlAvailability(lockTransitionViewButton, hasPrepared && inTransition" in main
    assert "setControlAvailability(lockCameraButton, hasPrepared && inLodView" in main
    assert "function buildStatusContext()" in main
    assert 'state.prepared?.representation === "trained"' in main
    assert 'modeSelect.value === "gaussian" || modeSelect.value === "both"' in main
    assert 'prepared.representation === "trained"' in main
    assert "lodSelect.selectedIndex = lodSelect.options.length - 1" in main
    assert "function lockTransitionView()" in main
    assert 'lod_counts: representationSelect.value === "trained" ? DEFAULT_TRAINED_LOD_COUNTS : undefined' in main
