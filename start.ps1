param(
    [switch]$Lan,
    [string]$HostAddr = "",
    [int]$Port = 8765
)

$ErrorActionPreference = "Stop"
$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RootDir

# Keep all model weights, caches, and temporary files on this drive (D:)
$ModelsDir = Join-Path $RootDir "models"
$CacheDir = Join-Path $RootDir ".cache"
$HfCacheDir = Join-Path $ModelsDir "hf-cache"
$TmpDir = Join-Path $CacheDir "tmp"

foreach ($dir in @($ModelsDir, $CacheDir, $HfCacheDir, (Join-Path $HfCacheDir "hub"), (Join-Path $CacheDir "torch"), (Join-Path $CacheDir "pip"), (Join-Path $CacheDir "xdg"), $TmpDir)) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}

$env:STUDIO_MODELS_DIR = $ModelsDir
$env:STUDIO_OUTPUTS_DIR = Join-Path $RootDir "outputs"
$env:STUDIO_CACHE_DIR = $CacheDir
$env:HF_HOME = $HfCacheDir
$env:HF_HUB_CACHE = Join-Path $HfCacheDir "hub"
$env:HUGGINGFACE_HUB_CACHE = Join-Path $HfCacheDir "hub"
$env:HF_XET_CACHE = Join-Path $HfCacheDir "xet"
$env:TORCH_HOME = Join-Path $CacheDir "torch"
$env:PIP_CACHE_DIR = Join-Path $CacheDir "pip"
$env:XDG_CACHE_HOME = Join-Path $CacheDir "xdg"
$env:TMP = $TmpDir
$env:TEMP = $TmpDir

if ($Lan) {
    $env:STUDIO_HOST = "0.0.0.0"
} elseif ($HostAddr) {
    $env:STUDIO_HOST = $HostAddr
} elseif (-not $env:STUDIO_HOST) {
    $env:STUDIO_HOST = "127.0.0.1"
}

if (-not $env:STUDIO_PORT) {
    $env:STUDIO_PORT = "$Port"
}

$PythonExe = Join-Path $RootDir ".venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    Write-Error "Virtual environment not found at $PythonExe"
    exit 1
}

Write-Host "Starting YuE2 Studio (CUDA) at http://$($env:STUDIO_HOST):$($env:STUDIO_PORT)" -ForegroundColor Cyan
Write-Host "  Models dir : $env:STUDIO_MODELS_DIR" -ForegroundColor DarkGray
Write-Host "  HF cache   : $env:HF_HOME" -ForegroundColor DarkGray
Write-Host "  Temp/cache : $env:STUDIO_CACHE_DIR" -ForegroundColor DarkGray

& $PythonExe -m studio.server
