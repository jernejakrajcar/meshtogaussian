# Future Improvement: CUDA Training Workflow

The current main project workflow uses Mesh2Splat-exported PLY files for Gaussian LODs. This document is kept as a future-improvement path for experimenting with direct `gsplat` training on a separate NVIDIA/CUDA machine.

## 1. Prepare The NVIDIA Machine

Recommended setup:

- Windows or Linux with an NVIDIA GPU.
- Recent NVIDIA driver.
- CUDA-enabled PyTorch.
- A cloned `gsplat` repo.

Example environment:

```powershell
conda create -n meshtogaussian-cuda python=3.11 -y
conda activate meshtogaussian-cuda
```

Install CUDA PyTorch using the command from the official PyTorch selector for that machine. Then install this project dependencies:

```powershell
pip install -r requirements.txt
```

Install/configure `gsplat` according to its documentation, then set the path in `configs/default.yaml`:

```yaml
gsplat:
  repo: C:/path/to/gsplat
  steps: 3000
  data_factor: 1
```

Check readiness:

```powershell
python scripts/check_cuda_training_env.py --config configs/default.yaml
```

This must report `cuda_available: true` and find `examples/simple_trainer.py` in the configured `gsplat.repo`.

## 2. Generate Training Data From A Mesh

```powershell
python scripts/train_gaussians_from_mesh.py --mesh data/meshes/sourdough.glb --config configs/default.yaml
```

This creates:

```text
data/generated_datasets/sourdough/
  images/*.png
  sparse/0/cameras.txt
  sparse/0/images.txt
  sparse/0/points3D.txt
  initial_points.ply
  manifest.json
```

It also writes the exact gsplat command to:

```text
data/trained_gaussians/sourdough/gsplat_command.json
```

## 3. Run Training

After CUDA and gsplat are ready:

```powershell
python scripts/train_gaussians_from_mesh.py --mesh data/meshes/sourdough.glb --config configs/default.yaml --run-trainer
```

If training succeeds and a `.ply` is exported, place or keep it under:

```text
data/trained_gaussians/sourdough/
```

The viewer scans `data/trained_gaussians/` recursively.

## 4. View The Result

```powershell
python scripts/serve_visualizer.py --config configs/default.yaml
```

Open:

```text
http://127.0.0.1:8000
```

Select:

- source mesh: your mesh
- Gaussian source: `Trained splats`
- trained Gaussian PLY: auto-match or the specific `.ply`

## Notes

- The AMD/CPU laptop path is for dataset generation, debugging, and viewing.
- Real `gsplat` training should run on NVIDIA CUDA.
- If your `gsplat` version saves checkpoints but no `.ply`, export/convert the trained checkpoint to a GraphDeco-compatible Gaussian PLY before loading it in the viewer.
- The lightweight PLY loader currently supports ASCII PLY. If your trained PLY is binary, convert it to ASCII or add the `plyfile` parser path.
