from __future__ import annotations

from pathlib import Path
from typing import Any

from src.core.config import load_config
from src.core.progress import StageLogger, StageError
from src.web.services import ModelStore


def create_app(config_path: str | Path = "configs/default.yaml", data_dir: str | Path | None = None):
    try:
        from fastapi import Body, FastAPI, File, HTTPException, UploadFile
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import FileResponse
        from fastapi.staticfiles import StaticFiles
    except Exception as exc:
        raise RuntimeError(
            "The web visualizer requires fastapi, uvicorn, and python-multipart. "
            "Install them with: pip install -r requirements.txt"
        ) from exc

    cfg = load_config(config_path)
    web_cfg = cfg.get("web", {})
    root = Path(data_dir) if data_dir is not None else None
    source_dirs = [Path(path) for path in web_cfg.get("source_dirs", ["data/source", "data/meshes"])]
    upload_dir = Path(web_cfg.get("upload_dir", "data/source/uploads"))
    trained_dirs = [Path(path) for path in web_cfg.get("trained_dirs", ["data/trained_gaussians"])]
    if root is not None:
        source_dirs = [root / "source", root / "meshes"]
        upload_dir = root / "source" / "uploads"
        trained_dirs = [root / "trained_gaussians"]

    logger = StageLogger(**cfg.get("progress", {"enabled": True, "verbose": True}))
    store = ModelStore(source_dirs=source_dirs, upload_dir=upload_dir, trained_dirs=trained_dirs, logger=logger)

    def with_viewer_config(payload: dict[str, Any]) -> dict[str, Any]:
        demo_cfg = cfg.get("demo", {})
        payload["viewer"] = {
            "far_radius": float(demo_cfg.get("far_radius", 4.0)),
            "near_radius": float(demo_cfg.get("near_radius", 1.25)),
            "azimuth_degrees": float(demo_cfg.get("azimuth_degrees", 35.0)),
            "elevation_degrees": float(demo_cfg.get("elevation_degrees", 10.0)),
            "transition": cfg.get("transition", {}),
        }
        return payload

    app = FastAPI(title="Mesh-to-Gaussian Visualizer")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    static_dir = Path("web").resolve()
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/")
    def index():
        return FileResponse(static_dir / "index.html")

    @app.get("/api/models")
    def models():
        return {"models": store.list_models()}

    @app.get("/api/trained-gaussians")
    def trained_gaussians():
        return {"models": store.list_trained_gaussians()}

    @app.post("/api/upload")
    async def upload(file: UploadFile = File(...)):
        try:
            saved = store.save_upload(file.filename or "model.obj", file.file)
            return {"model": saved}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/prepare")
    def prepare(request: dict[str, Any] = Body(default_factory=dict)):
        try:
            counts = request.get("lod_counts") or [int(v) for v in web_cfg.get("preview_lod_counts", [10, 100, 500])]
            prepared = store.prepare(
                model_id=request.get("model_id"),
                lod_counts=counts,
                seed=int(request.get("seed", 7)),
                fallback_color=cfg.get("mesh", {}).get("color"),
                representation=str(request.get("representation", "initialized")),
                trained_ply_id=request.get("trained_ply_id"),
            )
            return with_viewer_config(store.serialize_model(prepared))
        except StageError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/model/{model_id}")
    def model(model_id: str):
        try:
            return with_viewer_config(store.serialize_model(store.get_prepared(model_id)))
        except Exception as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/model/{model_id}/lod/{count}")
    def lod(model_id: str, count: int):
        try:
            return store.serialize_lod(store.get_prepared(model_id), count)
        except Exception as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    app.state.model_store = store
    return app
