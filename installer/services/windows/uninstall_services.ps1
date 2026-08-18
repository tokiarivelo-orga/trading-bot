<#
.SYNOPSIS
    Removes every Windows Scheduled Task registered by install_services.ps1,
    and (if -RepoRoot is given) the generated launcher scripts alongside
    them.

.DESCRIPTION
    *** UNVERIFIED *** -- see the header comment in install_services.ps1;
    the same caveat applies here: written against documented
    `ScheduledTasks` PowerShell module behavior, not run against a real
    Windows Task Scheduler as part of this change.

    Every task this installer creates is named "TradingBot-<something>", so
    they're enumerated and removed as a group rather than needing an exact
    per-account list.

.PARAMETER RepoRoot
    Optional. Path to the installed repo root -- if given, also deletes
    <RepoRoot>\installer\services\windows\generated\ (the per-task launcher
    .cmd files install_services.ps1 wrote, some of which embed a gateway
    secret value -- see that script's header comment).

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File uninstall_services.ps1 -RepoRoot "C:\trading-bot"
#>
[CmdletBinding()]
param(
    [string]$RepoRoot
)

$ErrorActionPreference = "Stop"
$TaskPrefix = "TradingBot-"

$tasks = Get-ScheduledTask -TaskName "$TaskPrefix*" -ErrorAction SilentlyContinue
if (-not $tasks) {
    Write-Host "No '$TaskPrefix*' scheduled tasks found -- nothing to remove."
} else {
    foreach ($task in $tasks) {
        Write-Host "Stopping + removing scheduled task '$($task.TaskName)'..."
        try { Stop-ScheduledTask -TaskName $task.TaskName -ErrorAction SilentlyContinue } catch {}
    }
    Get-ScheduledTask -TaskName "$TaskPrefix*" | Unregister-ScheduledTask -Confirm:$false
    Write-Host "Removed $($tasks.Count) scheduled task(s)."
}

if ($RepoRoot) {
    $LauncherDir = Join-Path $RepoRoot "installer\services\windows\generated"
    if (Test-Path $LauncherDir) {
        Remove-Item -Recurse -Force $LauncherDir
        Write-Host "Removed generated launcher scripts at $LauncherDir"
    }
}
