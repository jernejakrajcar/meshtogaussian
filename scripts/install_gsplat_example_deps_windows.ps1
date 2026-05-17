param(
  [string]$VenvPath = ".venv-gsplat310",
  [string]$GsplatRepo = "C:\Users\Uporabnik\Documents\faks\magisterij\1.letnik\nrg\gsplat",
  [string]$TorchVersion = "2.4.1",
  [string]$TorchVisionVersion = "0.19.1",
  [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu124",
  [string]$GsplatVersion = "1.5.3+pt24cu124",
  [string]$GsplatWheelIndexUrl = "https://docs.gsplat.studio/whl"
)

$ErrorActionPreference = "Stop"

function Fail {
  param([string]$Message)
  Write-Host "ERROR: $Message" -ForegroundColor Red
  exit 1
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

$python = Join-Path $VenvPath "Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
  Fail "Python was not found in $VenvPath."
}

$requirements = Join-Path $GsplatRepo "examples\requirements.txt"
if (-not (Test-Path -LiteralPath $requirements)) {
  Fail "gsplat example requirements were not found: $requirements"
}

Write-Host "Installing gsplat example dependencies without optional CUDA extensions..."
Invoke-Checked { & $python -m pip install --upgrade pip setuptools wheel ninja } "Could not update packaging tools."

$repoRoot = (Get-Location).Path
$filteredRequirements = Join-Path $env:TEMP "gsplat-example-requirements-no-fused.txt"
Get-Content -LiteralPath $requirements |
  Where-Object {
    ($_ -notmatch "fused-ssim") -and
    ($_ -notmatch "fused-bilagrid")
  } |
  Set-Content -LiteralPath $filteredRequirements -Encoding UTF8

Invoke-Checked { & $python -m pip install --no-build-isolation -r $filteredRequirements } "Could not install gsplat example dependencies."

$fallback = Join-Path $repoRoot "compat\fused_ssim_fallback"
Invoke-Checked { & $python -m pip install --force-reinstall --no-build-isolation --no-deps $fallback } "Could not install fused_ssim fallback."

Write-Host "Restoring CUDA PyTorch and matching gsplat wheel..."
Invoke-Checked { & $python -m pip install --force-reinstall "torch==$TorchVersion" --index-url $TorchIndexUrl } "Could not restore CUDA-enabled PyTorch."
Invoke-Checked { & $python -m pip install --force-reinstall --no-deps "torchvision==$TorchVisionVersion" --index-url $TorchIndexUrl } "Could not restore matching CUDA torchvision."
Invoke-Checked { & $python -m pip install --force-reinstall --no-deps "gsplat==$GsplatVersion" --index-url $GsplatWheelIndexUrl } "Could not restore matching gsplat wheel."

Write-Host "Checking PyTorch CUDA..."
Invoke-Checked { & $python -c "import torch, torchvision; print('torch', torch.__version__); print('torchvision', torchvision.__version__); print('torch cuda', torch.version.cuda); print('cuda available', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')" } "PyTorch CUDA check failed."

Write-Host "Done."
