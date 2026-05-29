<#
  Nastavitev lokalnega gsplat okolja na Windows.
  Skripta poišče Python, Visual Studio orodja in pripravi virtualno okolje, kar
  je predvsem praktična pomoč za ponovljiv zagon treninga na novem računalniku.
#>

param(
  [string]$PythonVersion = "3.11",
  [string]$PythonExe = "",
  [string]$VenvPath = ".venv-gsplat",
  [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu130",
  [string]$TorchVersion = "2.11.0",
  [string]$GsplatIndexUrl = "",
  [string]$GsplatSourcePath = "",
  [string]$GsplatRef = "v1.5.3",
  [switch]$SkipCudaCheck,
  [switch]$SkipSmokeTest
)

$ErrorActionPreference = "Stop"

function Find-CommandPath {
  param([string]$Name)
  $cmd = Get-Command $Name -ErrorAction SilentlyContinue
  if ($cmd) { return $cmd.Source }
  return $null
}

function Fail {
  param([string]$Message)
  Write-Host "ERROR: $Message" -ForegroundColor Red
  exit 1
}

function Find-VcVars64 {
  $candidates = @(
    "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat",
    "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
    "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
  )
  foreach ($candidate in $candidates) {
    if (Test-Path -LiteralPath $candidate) { return $candidate }
  }
  $found = Get-ChildItem "C:\Program Files\Microsoft Visual Studio" -Recurse -Filter vcvars64.bat -ErrorAction SilentlyContinue |
    Select-Object -First 1 -ExpandProperty FullName
  return $found
}

function Invoke-NativeCapture {
  param([scriptblock]$Command)
  $oldPreference = $ErrorActionPreference
  $ErrorActionPreference = "Continue"
  try {
    $output = & $Command 2>&1
    $exitCode = $LASTEXITCODE
  } finally {
    $ErrorActionPreference = $oldPreference
  }
  return [PSCustomObject]@{
    Output = $output
    ExitCode = $exitCode
  }
}

function Invoke-Checked {
  param(
    [scriptblock]$Command,
    [string]$ErrorMessage
  )
  & $Command
  if ($LASTEXITCODE -ne 0) {
    Fail $ErrorMessage
  }
}

function Ensure-GsplatSubmodules {
  param([string]$RepoPath)
  $glmHeader = Join-Path $RepoPath "gsplat\cuda\csrc\third_party\glm\glm\gtc\type_ptr.hpp"
  if (Test-Path -LiteralPath $glmHeader) {
    return
  }
  Write-Host "gsplat third-party files are missing. Initializing git submodules..."
  $git = Find-CommandPath "git"
  if (-not $git) {
    Fail "git was not found. Run this manually first: git -C `"$RepoPath`" submodule update --init --recursive"
  }
  Invoke-Checked { & $git -C $RepoPath submodule update --init --recursive } "Could not initialize gsplat submodules."
  if (-not (Test-Path -LiteralPath $glmHeader)) {
    Fail "gsplat GLM headers are still missing after submodule init: $glmHeader"
  }
}

function Ensure-GsplatRef {
  param(
    [string]$RepoPath,
    [string]$Ref
  )
  if (-not $Ref) {
    return
  }
  $git = Find-CommandPath "git"
  if (-not $git) {
    Fail "git was not found. Install git or re-run without -GsplatRef."
  }
  Write-Host "Checking out gsplat ref $Ref..."
  Invoke-Checked { & $git -C $RepoPath -c advice.detachedHead=false checkout $Ref } "Could not checkout gsplat ref $Ref."
}

if ($PythonExe) {
  Write-Host "Checking Python executable $PythonExe..."
  if (-not (Test-Path -LiteralPath $PythonExe)) {
    Fail "Python executable was not found: $PythonExe"
  }
  $pythonLauncher = $PythonExe
  $pythonLauncherArgs = @()
} else {
  Write-Host "Checking Python $PythonVersion..."
  $pyList = Invoke-NativeCapture { py -0p }
  if ($pyList.ExitCode -ne 0) {
    Fail "The Windows py launcher is not available. Install Python 3.13 or 3.12 with the py launcher enabled."
  }
  if ((($pyList.Output | Out-String) -notmatch [regex]::Escape("-V:$PythonVersion"))) {
    Write-Host "Detected Python runtimes:"
    Write-Host ($pyList.Output | Out-String)
    Fail "Python $PythonVersion was not found through the py launcher. Install that Python version, choose one listed above with -PythonVersion, or pass -PythonExe C:\path\to\python.exe."
  }
  $pythonLauncher = "py"
  $pythonLauncherArgs = @("-$PythonVersion")
}

$pythonCheck = Invoke-NativeCapture { & $pythonLauncher @pythonLauncherArgs -c "import sys; print(sys.executable); print(sys.version)" }
if ($pythonCheck.ExitCode -ne 0) {
  if ($pythonCheck.Output) { Write-Host ($pythonCheck.Output | Out-String) }
  Fail "Python check failed."
}
Write-Host ($pythonCheck.Output | Out-String)

Write-Host "Checking NVIDIA driver..."
$nvidiaSmi = Find-CommandPath "nvidia-smi"
if (-not $nvidiaSmi) {
  Fail "nvidia-smi was not found. Install/update the NVIDIA driver before trying gsplat."
}
& nvidia-smi

if (-not $SkipCudaCheck) {
  Write-Host "Checking CUDA Toolkit..."
  $nvcc = Find-CommandPath "nvcc"
  if (-not $nvcc) {
    Fail "nvcc was not found. Install NVIDIA CUDA Toolkit, open a new terminal, and re-run this script. Use -SkipCudaCheck only if you know a compatible precompiled gsplat wheel will be used."
  }
  & nvcc --version
}

$vcvars64 = Find-VcVars64
if (-not $vcvars64) {
  Fail "Visual Studio C++ vcvars64.bat was not found. Install Visual Studio Build Tools with the Desktop development with C++ workload."
}
Write-Host "Visual Studio environment: $vcvars64"

if (-not (Test-Path -LiteralPath $VenvPath)) {
  Write-Host "Creating $VenvPath..."
  & $pythonLauncher @pythonLauncherArgs -m venv $VenvPath
}

$python = Join-Path $VenvPath "Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
  Fail "Could not find $python after creating the virtual environment."
}

Write-Host "Upgrading packaging tools..."
Invoke-Checked { & $python -m pip install --upgrade pip "setuptools<82" wheel ninja numpy } "Could not install packaging tools."

Write-Host "Installing CUDA-enabled PyTorch $TorchVersion from $TorchIndexUrl..."
Invoke-Checked { & $python -m pip install --force-reinstall "torch==$TorchVersion" --index-url $TorchIndexUrl } "Could not install CUDA-enabled PyTorch."

Write-Host "Ensuring build helper packages are still available..."
Invoke-Checked { & $python -m pip install --upgrade "setuptools<82" wheel ninja numpy } "Could not install build helper packages."

Write-Host "Installing gsplat..."
if ($GsplatSourcePath) {
  if (-not (Test-Path -LiteralPath $GsplatSourcePath)) {
    Fail "gsplat source path was not found: $GsplatSourcePath"
  }
  Ensure-GsplatRef $GsplatSourcePath $GsplatRef
  Ensure-GsplatSubmodules $GsplatSourcePath
  Invoke-Checked { & $python -m pip install --force-reinstall --no-build-isolation --no-deps $GsplatSourcePath } "Could not build/install gsplat from source."
} elseif ($GsplatIndexUrl) {
  Invoke-Checked { & $python -m pip install --upgrade gsplat --index-url $GsplatIndexUrl } "Could not install gsplat."
} else {
  Invoke-Checked { & $python -m pip install --upgrade gsplat } "Could not install gsplat."
}

Write-Host "Checking PyTorch CUDA..."
Invoke-Checked { & $python -c "import torch; print('torch', torch.__version__); print('torch cuda', torch.version.cuda); print('cuda available', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')" } "PyTorch CUDA check failed."

if (-not $SkipSmokeTest) {
  Write-Host "Running gsplat smoke test inside the Visual Studio developer environment..."
  & powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\run_gsplat_smoke_windows.ps1" -VenvPath $VenvPath -CleanTorchExtensionCache
  if ($LASTEXITCODE -ne 0) {
    Fail "gsplat smoke test failed."
  }
}

Write-Host ""
Write-Host "Done. To run the smoke test again:"
Write-Host "$python scripts\smoke_gsplat.py"
