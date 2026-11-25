param(
    [switch]$Lan,
    [int]$Port = 8765,
    [string]$TaskName = "YuE2Studio"
)

$ErrorActionPreference = "Stop"
$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$LauncherVbs = Join-Path $RootDir "service-launcher.vbs"

if (-not (Test-Path $LauncherVbs)) {
    throw "Missing launcher script: $LauncherVbs"
}

# Stop existing instance cleanly before re-registering
$UninstallScript = Join-Path $RootDir "uninstall-service.ps1"
if ((Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) -and (Test-Path $UninstallScript)) {
    & $UninstallScript -TaskName $TaskName -Quiet
}

$extraArgs = "-Port $Port"
if ($Lan) {
    $extraArgs = "-Lan $extraArgs"
}

$action = New-ScheduledTaskAction `
    -Execute "wscript.exe" `
    -Argument "//B //Nologo `"$LauncherVbs`" $extraArgs" `
    -WorkingDirectory $RootDir

$currentUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $currentUser
$trigger.Delay = "PT10S"

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -MultipleInstances IgnoreNew

$principal = New-ScheduledTaskPrincipal `
    -UserId $currentUser `
    -LogonType Interactive `
    -RunLevel Limited

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Principal $principal `
    -Description "YuE2 Studio local AI music generation & analysis web server (http://127.0.0.1:$Port)" `
    -Force | Out-Null

Start-ScheduledTask -TaskName $TaskName

Write-Host "Registered and started automatic startup service '$TaskName'." -ForegroundColor Green
Write-Host "  Trigger  : At Windows logon ($currentUser, 10s delay)" -ForegroundColor DarkGray
Write-Host "  Web UI   : http://127.0.0.1:$Port" -ForegroundColor Cyan
Write-Host "  Logs     : $(Join-Path $RootDir 'logs\studio.log')" -ForegroundColor DarkGray
