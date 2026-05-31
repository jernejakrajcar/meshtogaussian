# Mesh-to-Gaussian LOD Prototype

Seminar prototype for interactive transitions 
between a mesh and Gaussian splats.

The current working pipeline:

- load a source mesh in the web viewer,
- load one Mesh2Splat-exported `.ply` file,
- let the viewer build nested LOD prefixes from that `.ply`,
- inspect transition modes in the browser.

This path does not require CUDA, `gsplat`, or a training setup. 
Mesh2Splat is used as an external converter/export tool, 
not as a Python dependency.

Older and experimental paths are documented in [alternatives.md](alternatives.md).

## Install

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On Windows you can also use the setup helper:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_windows.ps1
```

If you already have a Mesh2Splat executable and want the 
viewer's convert button to use it, pass the path:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_windows.ps1 `
  -Mesh2SplatExe "C:\path\to\mesh2splat\bin\Release\Mesh2Splat.exe"
```

Useful flags:

- `-SkipInstall` skips Python dependency installation.
- `-SkipTests` skips the test run.
- `-StartViewer` starts the web viewer after setup.

## Prepare Input Files

Put source meshes in one of these folders:

```text
data/source/
data/meshes/
```

Supported mesh formats include OBJ, PLY, GLTF, and GLB.

Export a Gaussian `.ply` from Mesh2Splat and put it in:

```text
data/mesh2splats/
```

Example:

```text
data/source/sourdough.glb
data/mesh2splats/sourdough-trained-284000.ply
```

The current viewer workflow expects one selected Mesh2Splat PLY. 
If you exported several density versions, choose the one you want 
to use from the dropdown.

## Run The Viewer

Start the web app:

```powershell
.\.venv\Scripts\python.exe scripts\serve_visualizer.py --config configs/default.yaml
```

Open:

```text
http://127.0.0.1:8000
```

In the viewer:

1. Choose the source mesh.
2. Set `Gaussian data source` to `Single Mesh2Splat PLY`.
3. Choose the exported `.ply` file.
4. Click `Load Selected Setup`.
5. Set `View mode` to `Transition`.
6. Move the transition slider and compare transition styles.

The most useful transition styles for the current implementation are:

- `Detail over mesh (optimized)`
- `Coverage-scaled detail over mesh (experimental)`
- `Keep mesh + add detail`

## Optional Mesh2Splat Convert Button

The viewer includes `Convert with Mesh2Splat`. It only works 
when `configs/default.yaml` points to a compatible Mesh2Splat 
executable:

```yaml
mesh2splat:
  executable: C:/path/to/mesh2splat/bin/Release/Mesh2Splat.exe
  working_dir: C:/path/to/mesh2splat/bin/Release
```

The expected command contract is:

```powershell
Mesh2Splat.exe --headless --input model.glb --output model_mesh2splat.ply --density 1.0 --quit
```

If your Mesh2Splat build does not support this headless interface, 
export from the Mesh2Splat GUI and place the `.ply` file 
in `data/mesh2splats/`.

## Tests

```powershell
pytest
```

## License

This repository is published under the MIT License. See [LICENSE](LICENSE).
