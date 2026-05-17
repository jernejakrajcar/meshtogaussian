# Mesh-to-Gaussian LOD Prototype

Practical seminar prototype for smooth transitions between mesh rendering and 3D Gaussian Splatting-style level of detail.

The current practical workflow uses Mesh2Splat-exported `.ply` files for the Gaussian representation. 
The repository also includes an experimental NVIDIA/CUDA `gsplat` training path for generating trained splats from 
synthetic mesh views. The runtime viewer remains dependency-light:

- Uses Mesh2Splat LOD exports as the primary practical Gaussian source.
- Uses PyTorch tensors for the inspectable fallback renderer.
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

For the Mesh2Splat workflow, the Python app does not require CUDA. CUDA-enabled PyTorch and `gsplat` are only needed for the 
experimental NVIDIA training path.

### Windows Setup Script

On a fresh Windows laptop, run the setup script from the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_windows.ps1
```

If Mesh2Splat is already built, pass the executable path so the script can configure the app:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_windows.ps1 `
  -Mesh2SplatExe "C:\path\to\mesh2splat\bin\Release\Mesh2Splat.exe"
```

The script creates `.venv`, installs Python dependencies, creates the local data folders, configures `configs/default.yaml` and `configs/smoke.yaml`, 
runs the Mesh2Splat preflight when an executable is available, and runs the tests. 
Add `-StartViewer` to start the web viewer after setup.

Useful setup flags:

- `-SkipInstall` skips dependency installation when `.venv` is already ready.
- `-SkipTests` skips the test run.
- `-SkipMesh2SplatConfig` prepares the Python app without editing Mesh2Splat paths.

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

## Mesh2Splat Workflow

This is the main project workflow. Electronic Arts' Mesh2Splat is used as an external Windows converter to create Gaussian `.ply` files from mesh assets. Mesh2Splat is not imported as a Python package; the Python app either reads `.ply` files exported from the Mesh2Splat GUI or calls a compatible headless executable.

### 1. Get And Build Mesh2Splat

Clone Mesh2Splat next to this repository or anywhere on disk:

```powershell
git clone https://github.com/electronicarts/mesh2splat.git ..\mesh2splat
```

Build the Release executable with Visual Studio or CMake according to the Mesh2Splat repository instructions.
After building, you should have something like:

```text
C:\path\to\mesh2splat\bin\Release\Mesh2Splat.exe
```

Configure this project by running:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_windows.ps1 `
  -Mesh2SplatExe "C:\path\to\mesh2splat\bin\Release\Mesh2Splat.exe"
```

or edit the config manually:

```yaml
mesh2splat:
  executable: C:/path/to/mesh2splat/bin/Release/Mesh2Splat.exe
  working_dir: C:/path/to/mesh2splat/bin/Release
```

### 2. Export Several Mesh2Splat LOD Files

The current reliable workflow is the Mesh2Splat GUI:

1. Open `Mesh2Splat.exe`.
2. Load a `.glb` mesh.
3. Convert the mesh to 3DGS.
4. Export multiple `.ply` files with different splat counts.
5. Put the files in `data/mesh2splats/`.

Name the files so the splat count is the last number in the filename:

```text
data/mesh2splats/sourdough-trained-300.ply
data/mesh2splats/sourdough-trained-800.ply
data/mesh2splats/sourdough-trained-1700.ply
data/mesh2splats/sourdough-trained-284000.ply
```

The viewer groups files by the shared mesh name (`sourdough`) and uses the
final number as the LOD key. These files are ignored by git because they are large generated assets.

### 3. View Proportional Mesh2Splat LOD Transitions

Start the viewer:

```powershell
.\.venv\Scripts\python.exe scripts\serve_visualizer.py --config configs/default.yaml
```

Open `http://127.0.0.1:8000`, then:

1. Choose the matching source mesh, for example `sourdough.glb`.
2. Set `Gaussian source` to `Mesh2Splat LOD set`.
3. Click `Prepare Viewer Data`.
4. Set `View mode` to `Pipeline transition`.
5. Move the transition slider.

The viewer uses the smallest Mesh2Splat `.ply` at far Gaussian distances and progressively switches to
denser `.ply` files as the camera moves closer. For a single exported `.ply`, use `Gaussian source: Trained splats` instead.

### 4. Optional Headless Conversion

The `Convert with Mesh2Splat` button calls the configured executable with this command contract:

```powershell
Mesh2Splat.exe --headless --input model.glb --output model_mesh2splat.ply --density 1.0 --quit
```

Use this only if your Mesh2Splat build supports
those arguments or you have patched Mesh2Splat to expose a headless export path.
Otherwise, export from the GUI and place the `.ply` files in `data/mesh2splats/`.

## Experimental NVIDIA/CUDA gsplat Training

Generate a synthetic COLMAP-style dataset from a mesh and create the `gsplat` training command:

```powershell
python scripts/train_gaussians_from_mesh.py --mesh data/meshes/my_model.glb --config configs/default.yaml
```

After installing and configuring CUDA-enabled PyTorch and `gsplat` on an NVIDIA machine, training can be run directly. 
This is not the main Mesh2Splat workflow; it is an experimental path for producing trained Gaussian splats 
without manual Mesh2Splat export.

```powershell
python scripts/train_gaussians_from_mesh.py --mesh data/meshes/my_model.glb --config configs/default.yaml --run-trainer
```

On the NVIDIA/CUDA machine, first run the environment check:

```powershell
python scripts/check_cuda_training_env.py --config configs/default.yaml
```

Expected experimental flow:

1. Synthetic images and camera files are written to `data/generated_datasets/<model_name>/`.
2. `gsplat` writes trained output under `data/trained_gaussians/<model_name>/`.
3. Put or keep the trained Gaussian `.ply` under `data/trained_gaussians/`.
4. The web viewer can use `Trained splats` mode to build LODs from that `.ply`.

## Web Visualizer

Install the web dependencies from `requirements.txt`, then run:

```powershell
python scripts/serve_visualizer.py --config configs/default.yaml
```

Open `http://127.0.0.1:8000`. The viewer can use the procedural demo model, 
list meshes from `data/source/` and `data/meshes/`, or upload OBJ/PLY/GLTF/GLB 
files into `data/source/uploads/`. It also lists trained Gaussian 
`.ply` files from `data/trained_gaussians/` and Mesh2Splat LOD sets from `data/mesh2splats/`.

The viewer has three Gaussian sources: `Mesh2Splat LOD set` for multiple exported 
Mesh2Splat PLY levels, `Trained splats` for one real trained 3DGS PLY file, 
and `Initialized preview` for the old mesh-sampled baseline. 
The default `Pipeline transition` mode includes a slider that moves the camera from far 
to near and updates mesh/Gaussian LOD weights with the same transition model as the offline pipeline.

## Tests

```powershell
pytest
```

## Seminar Report

The short LaTeX report is located at:

```text
docs/report/report.tex
```

Export it to PDF with `latexmk` from the repository root:

```powershell
latexmk -pdf docs/report/report.tex -outdir=docs/report
```

The generated PDF will be written next to the source file as `docs/report/report.pdf`.

## License

This repository is published under the MIT License. See [LICENSE](LICENSE).

## Notes

This is a seminar prototype, not a production Gaussian Splatting system.
The current main path is Mesh2Splat export plus viewer-side LOD transitions.
The software renderer is deliberately simple and can be slow for `20_000` Gaussians on CPU.
The NVIDIA/CUDA `gsplat` training path is experimental and separate from the Mesh2Splat workflow.
GPU rasterization remains future work.
