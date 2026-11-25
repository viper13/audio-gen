param(
    [switch]$Lan,
    [string]$HostAddr = "",
    [int]$Port = 8765
)

$ErrorActionPreference = "Stop"
$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RootDir

# Keep all model weights, caches, logs, and temporary files strictly on D:
$ModelsDir = Join-Path $RootDir "models"
$CacheDir = Join-Path $RootDir ".cache"
$HfCacheDir = Join-Path $ModelsDir "hf-cache"
$TmpDir = Join-Path $CacheDir "tmp"
$LogsDir = Join-Path $RootDir "logs"

foreach ($dir in @($ModelsDir, $CacheDir, $HfCacheDir, (Join-Path $HfCacheDir "hub"), (Join-Path $CacheDir "torch"), (Join-Path $CacheDir "pip"), (Join-Path $CacheDir "xdg"), $TmpDir, $LogsDir)) {
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
$env:PYTHONUNBUFFERED = "1"

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
    throw "Virtual environment not found at $PythonExe"
}

$StdOutLog = Join-Path $LogsDir "studio.log"
$StdErrLog = Join-Path $LogsDir "studio.err.log"
$PidFile = Join-Path $LogsDir "studio.pid"

# Rotate logs > 5 MB so disk usage stays bounded
foreach ($logPath in @($StdOutLog, $StdErrLog)) {
    if ((Test-Path $logPath) -and ((Get-Item $logPath).Length -gt 5MB)) {
        Move-Item -Path $logPath -Destination "$logPath.old" -Force
    }
}

$stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Add-Content -Path $StdOutLog -Value "[$stamp] Starting YuE2 Studio service at http://$($env:STUDIO_HOST):$($env:STUDIO_PORT)"

$proc = Start-Process -FilePath $PythonExe `
    -ArgumentList "-u", "-m", "studio.server" `
    -WorkingDirectory $RootDir `
    -WindowStyle Hidden `
    -RedirectStandardOutput $StdOutLog `
    -RedirectStandardError $StdErrLog `
    -PassThru

Set-Content -Path $PidFile -Value $proc.Id -Encoding ascii

try {
    Wait-Process -Id $proc.Id
    exit $proc.ExitCode
} finally {
    if (Test-Path $PidFile) {
        $recorded = (Get-Content -Path $PidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
        if ("$recorded" -eq "$($proc.Id)") {
            Remove-Item -Path $PidFile -Force -ErrorAction SilentlyContinue
        }
    }
}
