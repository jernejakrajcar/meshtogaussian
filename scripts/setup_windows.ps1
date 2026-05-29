<#
  Glavna Windows pripravljalna skripta za projekt.
  Namesti odvisnosti, po potrebi preveri Mesh2Splat in lahko zažene teste ali
  viewer - za zagon okolja za initial installation
#>

param(
    [string]$Python = "python",
    [string]$Mesh2SplatExe = "",
    [int]$Port = 8000,
    [switch]$SkipInstall,
    [switch]$SkipTests,
    [switch]$SkipMesh2SplatConfig,
    [switch]$StartViewer
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

function Info($Message) {
    Write-Host "[setup] $Message"
}

function Check-LastExit($Step) {
    if ($LASTEXITCODE -ne 0) {
        throw "$Step failed with exit code $LASTEXITCODE"
    }
}

function Resolve-Mesh2SplatExe {
    param([string]$ExplicitPath)
    if ($ExplicitPath) {
        return (Resolve-Path $ExplicitPath).Path
    }

    $Candidate = Join-Path $RepoRoot "..\mesh2splat\bin\Release\Mesh2Splat.exe"
    if (Test-Path $Candidate) {
        return (Resolve-Path $Candidate).Path
    }
    return ""
}

Info "Preparing data folders"
$Folders = @(
    "data\mesh2splats",
    "data\trained_gaussians\mesh2splat",
    "data\converted_glb",
    "data\source\uploads",
    "data\meshes",
    "data\outputs",
    "data\outputs_smoke"
)
foreach ($Folder in $Folders) {
    New-Item -ItemType Directory -Force -Path $Folder | Out-Null
}

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Info "Creating Python virtual environment"
    & $Python -m venv .venv
    Check-LastExit "Creating Python virtual environment"
}

$VenvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not $SkipInstall) {
    Info "Installing Python dependencies"
    & $VenvPython -m pip install --upgrade pip
    Check-LastExit "Upgrading pip"
    & $VenvPython -m pip install -r requirements.txt
    Check-LastExit "Installing Python dependencies"
} else {
    Info "Skipping dependency install"
}

$ResolvedMesh2SplatExe = Resolve-Mesh2SplatExe $Mesh2SplatExe
if ($SkipMesh2SplatConfig) {
    Info "Skipping Mesh2Splat config update"
} elseif ($ResolvedMesh2SplatExe) {
    Info "Configuring Mesh2Splat executable: $ResolvedMesh2SplatExe"
    $Env:MESH2SPLAT_EXE = $ResolvedMesh2SplatExe
    @'
import os
from pathlib import Path

import yaml

exe = Path(os.environ["MESH2SPLAT_EXE"]).resolve()
working_dir = exe.parent
for config_path in [Path("configs/default.yaml"), Path("configs/smoke.yaml")]:
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    cfg.setdefault("mesh2splat", {})
    cfg["mesh2splat"]["executable"] = str(exe).replace("\\", "/")
    cfg["mesh2splat"]["working_dir"] = str(working_dir).replace("\\", "/")
    cfg["mesh2splat"].setdefault("output_dir", "data/trained_gaussians/mesh2splat")
    cfg["mesh2splat"].setdefault("glb_cache_dir", "data/converted_glb")
    cfg["mesh2splat"].setdefault("density", 1.0)
    cfg["mesh2splat"].setdefault("timeout_seconds", 300)
    cfg["mesh2splat"].setdefault("extra_args", [])
    cfg.setdefault("web", {})
    cfg["web"].setdefault("mesh2splat_lod_dirs", ["data/mesh2splats"])
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
'@ | & $VenvPython -
    Check-LastExit "Configuring Mesh2Splat"

    Info "Running Mesh2Splat preflight"
    @'
from src.core.config import load_config
from src.conversion.mesh2splat_runner import Mesh2SplatConfig, check_mesh2splat_environment

cfg = Mesh2SplatConfig.from_dict(load_config("configs/default.yaml").get("mesh2splat", {}))
status = check_mesh2splat_environment(cfg)
print(status)
if status["problems"]:
    raise SystemExit(1)
'@ | & $VenvPython -
    Check-LastExit "Mesh2Splat preflight"
} else {
    Write-Warning "Mesh2Splat.exe was not found. Pass -Mesh2SplatExe C:\path\to\Mesh2Splat.exe after building Mesh2Splat."
}

if (-not $SkipTests) {
    Info "Running tests"
    & $VenvPython -m pytest
    Check-LastExit "Tests"
} else {
    Info "Skipping tests"
}

Info "Setup complete"
Info "Put manually exported Mesh2Splat LOD .ply files in: data\mesh2splats"
Info "Use names like: sourdough-trained-300.ply, sourdough-trained-800.ply, sourdough-trained-284000.ply"
Info "Run viewer with: .\.venv\Scripts\python.exe scripts\serve_visualizer.py --config configs/default.yaml --port $Port"

if ($StartViewer) {
    Info "Starting viewer at http://127.0.0.1:$Port"
    & $VenvPython scripts\serve_visualizer.py --config configs/default.yaml --host 127.0.0.1 --port $Port
    Check-LastExit "Viewer"
}
