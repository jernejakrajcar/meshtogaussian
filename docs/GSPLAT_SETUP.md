# Future Improvement: gsplat Smoke-Test Setup

This is a minimal setup for trying `gsplat` without changing the main project
environment. It is not the current main workflow; the project currently uses
Mesh2Splat-exported PLY files as the practical Gaussian source.

## Current Machine Status

The machine has an NVIDIA RTX 3060 and a recent NVIDIA driver, but the current
project `.venv` uses CPU-only PyTorch. `gsplat` needs CUDA-enabled PyTorch.

The setup script also expects:

- Python 3.11, 3.12, or 3.13 through the Windows `py` launcher.
- Visual Studio C++ tools.
- CUDA Toolkit with `nvcc` available.

On this machine, Visual Studio C++ tools and Python 3.13/3.11 are available.
The remaining missing prerequisite is `nvcc`, which means the CUDA Toolkit is
not installed or is not on `PATH`.

## Install Prerequisites

Install Python 3.13 or 3.12 from:

```text
https://www.python.org/downloads/windows/
```

Install NVIDIA CUDA Toolkit from:

```text
https://developer.nvidia.com/cuda-downloads
```

For the default setup in this repository, prefer CUDA Toolkit 13.0 because the
script installs the CUDA 13.0 PyTorch wheels. The NVIDIA driver may report CUDA
13.1 in `nvidia-smi`; that is the maximum driver-supported CUDA version, not
necessarily the installed compiler toolkit version.

After installing CUDA Toolkit, open a new terminal and check:

```powershell
py -0p
nvcc --version
nvidia-smi
```

## Create The Test Environment

From the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_gsplat_windows.ps1
```

The script creates `.venv-gsplat`, installs CUDA-enabled PyTorch, installs
`gsplat`, and runs a small render smoke test.

If you installed Python 3.12 instead of 3.13:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_gsplat_windows.ps1 -PythonVersion 3.12
```

If Codex or an older terminal does not see the newly installed Python through
`py -0p`, pass the executable path directly:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_gsplat_windows.ps1 `
  -PythonExe "C:\Users\Uporabnik\AppData\Local\Programs\Python\Python313\python.exe"
```

## Run Only The Smoke Test

After the environment is installed:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_gsplat_smoke_windows.ps1
```

If a previous attempt failed while compiling `gsplat_cuda`, clean the cached
failed extension and run again:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_gsplat_smoke_windows.ps1 -CleanTorchExtensionCache
```

The smoke test renders a tiny synthetic Gaussian cloud and writes:

```text
data/gsplat_smoke/gsplat_smoke.ppm
```

The image is deliberately simple. It only checks that `gsplat` imports, CUDA is
available through PyTorch, and rasterization returns an image and alpha buffer.

## CUDA 13.2 MSVC Preprocessor Error

CUDA 13.2 may fail with:

```text
MSVC/cl.exe with traditional preprocessor is used
```

The helper scripts set `CL=/Zc:preprocessor` before running `gsplat`, which
passes the standard-conforming MSVC preprocessor flag required by CUDA's CCCL
headers.
