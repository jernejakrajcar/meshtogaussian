# Alternative And Experimental Workflows

This file keeps the older or heavier project paths out of 
the main setup guide. The recommended workflow is still the 
Mesh2Splat PLY viewer path described in [README.md](README.md).

## Offline Demo Pipeline

The repository still contains an offline, 
inspectable pipeline that renders a synthetic transition 
video and metrics. It is useful for quick smoke tests and earlier 
experiments, but it is not the main Mesh2Splat viewer workflow.

Run the default demo:

```powershell
python -m src.pipeline.run_pipeline --config configs/default.yaml
```

Run the smoke configuration:

```powershell
python -m src.pipeline.run_pipeline --config configs/smoke.yaml
```

Choose a model directly:

```powershell
python -m src.pipeline.run_pipeline --config configs/default.yaml --mesh data/source/my_model.obj
python -m src.pipeline.run_pipeline --config configs/smoke.yaml --mesh demo --demo-shape cube --output data/outputs_cube
```

Outputs are written to `data/outputs/`:

- `transition.mp4`
- optional per-frame PNGs
- `metrics.json`
- `lod_*.npz`

## Utility Scripts

These scripts are mostly from the older offline pipeline:

```powershell
python scripts/generate_views.py --config configs/default.yaml
python scripts/build_lods.py --config configs/default.yaml
python scripts/render_transition.py --config configs/default.yaml
python scripts/evaluate.py --config configs/default.yaml
```

## Mesh2Splat LOD Sets

Older documentation described a workflow with several 
Mesh2Splat `.ply` files grouped as an automatic LOD set:

```text
data/mesh2splats/sourdough-trained-300.ply
data/mesh2splats/sourdough-trained-800.ply
data/mesh2splats/sourdough-trained-1700.ply
data/mesh2splats/sourdough-trained-284000.ply
```

The current viewer no longer uses this as the main path. 
It expects one selected Mesh2Splat `.ply` and then builds 
viewer-side nested LODs from that file. 
Keeping several exports can still be useful for manual comparison, 
but the active setup is `Single Mesh2Splat PLY`.

## Experimental NVIDIA/CUDA gsplat Training

This path generates synthetic COLMAP-style data from a mesh and 
prepares a `gsplat` training command. It requires a separate 
NVIDIA/CUDA environment and is not needed for the current 
Mesh2Splat workflow.

Generate the dataset and command:

```powershell
python scripts/train_gaussians_from_mesh.py --mesh data/meshes/my_model.glb --config configs/default.yaml
```

Check the CUDA/gsplat environment:

```powershell
python scripts/check_cuda_training_env.py --config configs/default.yaml
```

Run training only after CUDA-enabled PyTorch and 
`gsplat` are installed and configured:

```powershell
python scripts/train_gaussians_from_mesh.py --mesh data/meshes/my_model.glb --config configs/default.yaml --run-trainer
```

Expected experimental flow:

1. Synthetic images and camera files are written to `data/generated_datasets/<model_name>/`.
2. `gsplat` writes trained output under `data/trained_gaussians/<model_name>/`.
3. Put or keep the trained Gaussian `.ply` under `data/trained_gaussians/`.
4. The web viewer can use `Single trained PLY` mode to build LODs from that `.ply`.

## Mesh-Sampled Preview

The viewer still has `Mesh-sampled preview`. This creates 
simple Gaussian-like splats by sampling the mesh surface directly. 
It is useful for debugging UI and transition logic without an 
external Gaussian file, but it is not a real 3DGS export.

## Notes

This is a seminar prototype, not a production 
Gaussian Splatting system. The browser renderer is a 
simplified WebGL2 splat renderer. It is good enough for 
comparing transition behavior, but it can still show blur, 
imperfect sorting, and alpha blending artifacts.
