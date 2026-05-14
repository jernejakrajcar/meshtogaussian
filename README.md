# Mesh-to-Gaussian LOD Prototype

Practical seminar prototype for smooth transitions between mesh rendering and 3D Gaussian Splatting-style level of detail.

The first version is intentionally dependency-light at runtime:

- Uses PyTorch tensors and selects CUDA when available.
- Falls back to CPU automatically.
- Tries DirectML only when requested and installed.
- Can run a demo without an input mesh by generating a procedural sphere.
- Uses a software mesh renderer and a simple PyTorch Gaussian splat renderer so the pipeline is inspectable.
- Supports `trimesh` for OBJ/PLY/GLTF/GLB loading when installed.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For a CPU-only quick test, PyTorch from your existing environment is enough. For CUDA, install the matching PyTorch build from the official PyTorch selector.

## Run The Full Demo

```powershell
python -m src.pipeline.run_pipeline --config configs/default.yaml
```

For a fast validation run:

```powershell
python -m src.pipeline.run_pipeline --config configs/smoke.yaml
```

Choose a model directly from the command line:

```powershell
python -m src.pipeline.run_pipeline --config configs/default.yaml --mesh data/source/my_model.obj
python -m src.pipeline.run_pipeline --config configs/smoke.yaml --mesh demo --demo-shape cube --output data/outputs_cube
```

Outputs are written to `data/outputs/`:

- `transition.mp4`
- optional per-frame PNGs
- `metrics.json`
- `lod_*.npz`

By default the config uses a procedural sphere. To use your own model, either pass `--mesh` or set:

```yaml
mesh:
  path: data/meshes/my_model.obj
```

## Useful Scripts

```powershell
python scripts/generate_views.py --config configs/default.yaml
python scripts/build_lods.py --config configs/default.yaml
python scripts/render_transition.py --config configs/default.yaml
python scripts/evaluate.py --config configs/default.yaml
```

## Train Real Gaussian Splats From A Mesh

Generate a synthetic COLMAP-style dataset from a mesh and create the `gsplat` training command:

```powershell
python scripts/train_gaussians_from_mesh.py --mesh data/meshes/my_model.glb --config configs/default.yaml
```

After installing and configuring `gsplat`, run training directly:

```powershell
python scripts/train_gaussians_from_mesh.py --mesh data/meshes/my_model.glb --config configs/default.yaml --run-trainer
```

Expected flow:

1. Synthetic images and camera files are written to `data/generated_datasets/<model_name>/`.
2. `gsplat` writes trained output under `data/trained_gaussians/<model_name>/`.
3. Put or keep the trained Gaussian `.ply` under `data/trained_gaussians/`.
4. The web viewer can use `Trained splats` mode to build LODs from that `.ply`.

## Web Visualizer

Install the web dependencies from `requirements.txt`, then run:

```powershell
python scripts/serve_visualizer.py --config configs/default.yaml
```

Open `http://127.0.0.1:8000`. The viewer can use the procedural demo model, list meshes from `data/source/` and `data/meshes/`, or upload OBJ/PLY/GLTF/GLB files into `data/source/uploads/`. It also lists trained Gaussian `.ply` files from `data/trained_gaussians/`.

The viewer has two Gaussian sources: `Trained splats` for real trained 3DGS PLY files, and `Initialized preview` for the old mesh-sampled baseline. The default `Pipeline transition` mode includes a slider that moves the camera from far to near and updates mesh/Gaussian LOD weights with the same transition model as the offline pipeline.

## Tests

```powershell
pytest
```

## Notes

This is a seminar prototype, not a production Gaussian Splatting system. The software renderer is deliberately simple and can be slow for `20_000` Gaussians on CPU. The architecture leaves room for `gsplat` or another CUDA rasterizer later through `GaussianRenderer`.
