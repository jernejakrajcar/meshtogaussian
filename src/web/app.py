from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import unquote

from src.conversion.mesh2splat_runner import Mesh2SplatConfig, convert_mesh_to_gaussians
from src.core.config import load_config
from src.core.progress import StageLogger, StageError
from src.web.services import ModelStore


def create_app(config_path: str | Path = "configs/default.yaml", data_dir: str | Path | None = None):
    try:
        from fastapi import Body, FastAPI, File, HTTPException, UploadFile
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import FileResponse, Response
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
    mesh2splat_lod_dirs = [Path(path) for path in web_cfg.get("mesh2splat_lod_dirs", ["data/mesh2splats"])]
    if root is not None:
        source_dirs = [root / "source", root / "meshes"]
        upload_dir = root / "source" / "uploads"
        trained_dirs = [root / "trained_gaussians"]
        mesh2splat_lod_dirs = [root / "mesh2splats"]
        cfg.setdefault("mesh2splat", {})["output_dir"] = str(root / "trained_gaussians" / "mesh2splat")
        cfg.setdefault("mesh2splat", {})["glb_cache_dir"] = str(root / "converted_glb")

    logger = StageLogger(**cfg.get("progress", {"enabled": True, "verbose": True}))
    store = ModelStore(
        source_dirs=source_dirs,
        upload_dir=upload_dir,
        trained_dirs=trained_dirs,
        mesh2splat_lod_dirs=mesh2splat_lod_dirs,
        logger=logger,
    )

    def with_viewer_config(payload: dict[str, Any]) -> dict[str, Any]:
        demo_cfg = cfg.get("demo", {})
        payload["viewer"] = {
            "far_radius": float(demo_cfg.get("far_radius", 4.0)),
            "near_radius": float(demo_cfg.get("near_radius", 1.25)),
            "azimuth_degrees": float(demo_cfg.get("azimuth_degrees", 35.0)),
            "elevation_degrees": float(demo_cfg.get("elevation_degrees", 10.0)),
            "transition": cfg.get("transition", {}),
        }
        if payload.get("representation") in {"mesh2splat_lods", "mesh2splat", "trained"}:
            payload["viewer"]["transition"] = _proportional_transition_for_lods(
                [str(lod["name"]) for lod in payload.get("lods", [])],
                cfg.get("transition", {}),
                payload["viewer"]["far_radius"],
                payload["viewer"]["near_radius"],
            )
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

    @app.get("/api/mesh2splat-gaussians")
    def mesh2splat_gaussians():
        return {"models": store.list_mesh2splat_gaussians()}

    @app.get("/api/mesh2splat-lod-sets")
    def mesh2splat_lod_sets():
        return {"sets": store.list_mesh2splat_lod_sets()}

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

    @app.post("/api/convert-mesh2splat")
    def convert_mesh2splat(request: dict[str, Any] = Body(default_factory=dict)):
        try:
            model_id = request.get("model_id")
            if not model_id or str(model_id).startswith("demo:"):
                raise ValueError("Mesh2Splat conversion requires a real mesh file, not the procedural demo.")
            mesh_path = store.id_to_path(str(model_id))
            converter_config = Mesh2SplatConfig.from_dict(cfg.get("mesh2splat", {}))
            result = convert_mesh_to_gaussians(
                mesh_path,
                converter_config,
                density=float(request["density"]) if request.get("density") is not None else None,
            )
            counts = request.get("lod_counts") or [int(v) for v in web_cfg.get("preview_lod_counts", [10, 100, 500])]
            prepared = store.prepare(
                model_id=str(model_id),
                lod_counts=counts,
                seed=int(request.get("seed", 7)),
                fallback_color=cfg.get("mesh", {}).get("color"),
                representation="trained",
                trained_ply_id=str(result.output_ply),
            )
            payload = with_viewer_config(store.serialize_model(prepared))
            payload["conversion"] = {
                "input_mesh": str(result.input_mesh),
                "glb_mesh": str(result.glb_mesh),
                "output_ply": str(result.output_ply),
                "manifest": str(result.manifest_path),
                "returncode": result.returncode,
            }
            return payload
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

    @app.get("/api/model/{model_id}/lod/{count}/binary")
    def binary_lod(model_id: str, count: int):
        try:
            return Response(
                content=store.serialize_lod_binary(store.get_prepared(model_id), count),
                media_type="application/octet-stream",
            )
        except Exception as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/model/{model_id}/source/{asset_path:path}")
    def source_asset(model_id: str, asset_path: str):
        try:
            prepared = store.get_prepared(model_id)
            if prepared.source == "generated":
                raise FileNotFoundError("Generated demo models do not have source assets.")
            source_path = Path(prepared.source).resolve()
            requested = (source_path.parent / unquote(asset_path)).resolve()
            if source_path.parent not in [requested, *requested.parents]:
                raise PermissionError("Source asset path must stay next to the source model.")
            if not requested.exists() or not requested.is_file():
                raise FileNotFoundError(f"Source asset was not found: {asset_path}")
            return FileResponse(requested)
        except Exception as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    app.state.model_store = store
    return app


def _proportional_transition_for_lods(
    lod_names: list[str],
    base_transition: dict[str, Any],
    far_radius: float,
    near_radius: float,
) -> dict[str, Any]:
    counts = sorted(int(name) for name in lod_names if str(name).isdigit())
    if not counts:
        return base_transition
    mesh_fade_start = float(base_transition.get("mesh_fade_start", far_radius * 0.9))
    mesh_fade_end = float(base_transition.get("mesh_fade_end", (far_radius + near_radius) * 0.5))
    start = mesh_fade_start
    end = max(float(near_radius), 0.05)
    if len(counts) == 1:
        ranges = {str(counts[0]): [start, end]}
    else:
        step = (start - end) / len(counts)
        ranges = {}
        for index, count in enumerate(counts):
            far = start - index * step
            near = start - (index + 1) * step
            ranges[str(count)] = [round(float(far), 4), round(float(max(near, end)), 4)]
    return {
        "mesh_fade_start": mesh_fade_start,
        "mesh_fade_end": mesh_fade_end,
        "lod_mode": "progressive",
        "lod_ranges": ranges,
    }
