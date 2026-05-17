# Hybrid Mesh2Splat and Gaussian Training Workflow

This project keeps Mesh2Splat as the stable conversion path and treats CUDA
training as the research-correct path.

## Mesh2Splat Sweep

Mesh2Splat is fast, but one density value is not enough to know whether the
export has enough coverage. Run a sweep and inspect the generated manifest:

```powershell
python scripts/sweep_mesh2splat.py `
  --mesh data/meshes/sourdough.glb `
  --densities 0.5,1.0,1.5,2.0,3.0
```

The script writes converted PLY files into `data/mesh2splats/` and a manifest
named `<mesh>_mesh2splat_sweep.json`. Each result includes:

- splat count and file size,
- visible scale percentiles,
- nearest-neighbor distance percentiles,
- nearest-neighbor / visible-scale percentiles,
- warnings when the export is likely to show holes.

If the Mesh2Splat command line build exposes a Gaussian scale option, pass it
with `--scale-values` and, if needed, `--scale-arg`:

```powershell
python scripts/sweep_mesh2splat.py `
  --mesh data/meshes/sourdough.glb `
  --densities 1.0,2.0,3.0 `
  --scale-values 0.75,1.0,1.25 `
  --scale-arg --GaussianScale
```

This is useful because density changes the internal conversion resolution, while
exported scale is affected by both the original Gaussian scale and that
resolution. More density alone can still leave visible holes if the exported
scales become too small.

## Research Training Path

For a more correct Gaussian result, generate synthetic views from the mesh and
train with `gsplat`:

```powershell
python scripts/train_gaussians_from_mesh.py `
  --mesh data/meshes/sourdough.glb `
  --config configs/default.yaml `
  --research-defaults
```

After the CUDA/gsplat environment is ready, execute the trainer:

```powershell
python scripts/train_gaussians_from_mesh.py `
  --mesh data/meshes/sourdough.glb `
  --config configs/default.yaml `
  --research-defaults `
  --run-trainer
```

This path creates a synthetic COLMAP-style dataset with known mesh camera poses,
then points `gsplat/examples/simple_trainer.py` at that dataset. The trained PLY
should be loaded in the viewer as `Trained splats`.

## Barycentric Coordinates

Barycentric coordinates would be useful for a future mesh-bound Gaussian method:
each Gaussian could store which triangle it belongs to and its position inside
that triangle. That would make progressive/incremental mesh-to-Gaussian
transitions more geometrically stable. Current Mesh2Splat PLY files do not store
the triangle identity or barycentric coordinates, so the current implementation
does not use them directly.
