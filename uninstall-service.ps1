param(
    [string]$TaskName = "YuE2Studio",
    [switch]$Quiet
)

$ErrorActionPreference = "Continue"
$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PidFile = Join-Path $RootDir "logs\studio.pid"

if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
}

# Stop recorded Python server process tree (venv shim + child Python process)
if (Test-Path $PidFile) {
    $pidVal = (Get-Content -Path $PidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
    if ($pidVal -match "^\d+$") {
        & taskkill.exe /PID $pidVal /T /F 2>$null | Out-Null
    }
    Remove-Item -Path $PidFile -Force -ErrorAction SilentlyContinue
}

# Clean up any remaining studio.server python process
Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -like "*studio.server*" } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }

if (-not $Quiet) {
    Write-Host "Stopped and removed automatic startup service '$TaskName'." -ForegroundColor Yellow
}
