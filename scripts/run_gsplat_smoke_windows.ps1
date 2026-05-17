param(
  [string]$VenvPath = ".venv-gsplat",
  [string]$LogPath = "data/gsplat_smoke/build.log",
  [switch]$CleanTorchExtensionCache,
  [string]$MaxJobs = "1"
)

$ErrorActionPreference = "Stop"

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
  return Get-ChildItem "C:\Program Files\Microsoft Visual Studio" -Recurse -Filter vcvars64.bat -ErrorAction SilentlyContinue |
    Select-Object -First 1 -ExpandProperty FullName
}

$repoRoot = (Get-Location).Path
$python = Join-Path $repoRoot (Join-Path $VenvPath "Scripts\python.exe")
$smokeScript = Join-Path $repoRoot "scripts\smoke_gsplat.py"
$vcvars64 = Find-VcVars64

if (-not (Test-Path -LiteralPath $python)) {
  Fail "Python was not found in $VenvPath. Run scripts/setup_gsplat_windows.ps1 first."
}
if (-not (Test-Path -LiteralPath $smokeScript)) {
  Fail "Smoke test script was not found: $smokeScript"
}
if (-not $vcvars64) {
  Fail "Visual Studio C++ vcvars64.bat was not found."
}

if ($CleanTorchExtensionCache) {
  $cacheRoot = Join-Path $env:LOCALAPPDATA "torch_extensions"
  if (Test-Path -LiteralPath $cacheRoot) {
    Write-Host "Removing cached gsplat CUDA extensions under $cacheRoot..."
    Get-ChildItem -LiteralPath $cacheRoot -Recurse -Directory -Filter "gsplat_cuda" -ErrorAction SilentlyContinue |
      ForEach-Object { Remove-Item -LiteralPath $_.FullName -Recurse -Force }
  }
}

Write-Host "Running gsplat smoke test with MSVC /Zc:preprocessor enabled..."
Write-Host "Full log: $LogPath"
$logFullPath = Join-Path $repoRoot $LogPath
$logDir = Split-Path -Parent $logFullPath
if ($logDir -and -not (Test-Path -LiteralPath $logDir)) {
  New-Item -ItemType Directory -Force -Path $logDir | Out-Null
}

$cmd = "`"$vcvars64`" && set CL=/Zc:preprocessor %CL% && set MAX_JOBS=$MaxJobs && set TORCH_CUDA_ARCH_LIST=8.6 && cd /d `"$repoRoot`" && `"$python`" `"$smokeScript`" && exit /b 0 || exit /b 1"
$output = & cmd.exe /c $cmd 2>&1
$exitCode = $LASTEXITCODE
$output | Tee-Object -FilePath $logFullPath
"ExitCode: $exitCode" | Add-Content -LiteralPath $logFullPath -Encoding UTF8
if ($exitCode -ne 0) {
  Write-Host "Smoke test failed. Re-run this command to capture a full log:"
  Write-Host "powershell -ExecutionPolicy Bypass -File scripts/run_gsplat_smoke_windows.ps1 -VenvPath $VenvPath -CleanTorchExtensionCache *>&1 | Tee-Object $LogPath"
  Fail "gsplat smoke test failed."
}
