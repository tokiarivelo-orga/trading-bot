<#
.SYNOPSIS
    Registers Windows Scheduled Tasks for the AI Trading Bot -- backend,
    frontend, and one *native* (no Wine) gateway + terminal pair per
    enabled MT5 account -- so the stack restarts on crash and comes back
    after a reboot. The Windows-native counterpart to
    installer/services_linux.py's systemd --user units on Linux.

.DESCRIPTION
    *** UNVERIFIED ***
    This script was written against documented Windows Task Scheduler /
    `ScheduledTasks` PowerShell module behavior (Register-ScheduledTask,
    New-ScheduledTaskTrigger, New-ScheduledTaskAction,
    New-ScheduledTaskSettingsSet, New-TimeSpan). It has NOT been run on a
    real Windows machine as part of this change -- no Windows environment
    was available to test it here. Review it and test on a disposable
    VPS/VM before relying on it for a live account. It supersedes the
    manual NSSM steps in gateway/README.md's "Option B -- Windows VPS"
    section with a scriptable equivalent; that manual documentation is
    left in place for a later docs pass to reconcile.

    Every task name is prefixed "TradingBot-" so they're easy to find and
    remove as a group (see uninstall_services.ps1, and
    installer/manage.py's Windows path).

    Each task restarts up to 999 times, 1 minute apart, if the process it
    runs exits (`-RestartCount 999 -RestartInterval (New-TimeSpan -Minutes
    1)`), and triggers both `AtStartup` (survives a reboot) and `AtLogOn`
    (comes up immediately in an interactive session too, rather than
    waiting for the next reboot).

    PowerShell's scheduled-task actions have no native "set an environment
    variable" the way a systemd unit's `Environment=` line does, so each
    task's action points at a small generated `.cmd` launcher (written under
    <RepoRoot>\installer\services\windows\generated\) that sets whatever
    env vars that process needs with `set NAME=value` before running the
    real command. The gateway launcher's secret value is copied out of
    `.env` at generation time (same reasoning as
    installer/services_linux.py's gateway unit -- see that file's
    docstring) -- the generated launcher directory's ACL is restricted to
    the current user for that reason.

.PARAMETER RepoRoot
    Path to the installed repo root -- the same directory installer/
    install.py wrote `.env` and `configs\accounts.yaml` into (i.e.
    Answers.install_dir from installer/wizard.py).

.PARAMETER StateJsonPath
    Path to installer/state.json, written by installer/install.py after a
    successful run (see wizard.answers_to_state() for its exact shape).
    Reading this file directly (via ConvertFrom-Json) is preferred over a
    hand-rolled -Accounts CLI argument per account -- it's already on disk
    and its shape can't drift from what the wizard actually collected.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File install_services.ps1 `
        -RepoRoot "C:\trading-bot" -StateJsonPath "C:\trading-bot\installer\state.json"
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$RepoRoot,

    [Parameter(Mandatory = $true)]
    [string]$StateJsonPath
)

$ErrorActionPreference = "Stop"

$TaskPrefix = "TradingBot-"
$DefaultBackendPort = 8000
$DefaultFrontendPort = 3000
$DefaultGatewayHost = "127.0.0.1"
$DefaultGatewayPort = 8787
$DefaultTerminalPath = "C:\Program Files\MetaTrader 5\terminal64.exe"

if (-not (Test-Path $StateJsonPath)) {
    throw "No installer state found at $StateJsonPath -- run installer/install.py first."
}
$state = Get-Content -Raw -Path $StateJsonPath | ConvertFrom-Json

$BackendDir = Join-Path $RepoRoot "backend"
$FrontendDir = Join-Path $RepoRoot "frontend"
$GatewayDir = Join-Path $RepoRoot "gateway"
$EnvPath = Join-Path $RepoRoot ".env"
$LauncherDir = Join-Path $RepoRoot "installer\services\windows\generated"
New-Item -ItemType Directory -Force -Path $LauncherDir | Out-Null

function Read-DotEnvValue {
    # Minimal `.env` line reader: first uncommented "Name=value" line.
    # Mirrors installer/env_writer.py's read_existing() well enough for the
    # handful of values this script needs (frontend port, gateway secrets).
    param([string]$Name)
    if (-not (Test-Path $EnvPath)) { return "" }
    $pattern = "^$([regex]::Escape($Name))="
    $line = Select-String -Path $EnvPath -Pattern $pattern | Select-Object -First 1
    if (-not $line) { return "" }
    return $line.Line.Substring($Name.Length + 1)
}

function New-TradingBotSettings {
    New-ScheduledTaskSettingsSet -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) `
        -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
}

function New-TradingBotTriggers {
    @(
        (New-ScheduledTaskTrigger -AtStartup),
        (New-ScheduledTaskTrigger -AtLogOn)
    )
}

function Write-LauncherScript {
    param(
        [string]$Path,
        [string]$Content
    )
    Set-Content -Path $Path -Value $Content -Encoding ASCII
    # Restrict the launcher to the current user only -- gateway launchers
    # embed a real secret value (see the .DESCRIPTION block above).
    icacls $Path /inheritance:r /grant:r "$($env:USERNAME):(R,W)" | Out-Null
}

function Register-TradingBotTask {
    param(
        [string]$Name,
        [string]$LauncherPath
    )
    $taskName = "$TaskPrefix$Name"
    $action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c `"$LauncherPath`"" -WorkingDirectory (Split-Path $LauncherPath -Parent)
    $trigger = New-TradingBotTriggers
    $settings = New-TradingBotSettings
    Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -Force | Out-Null
    Write-Host "Registered scheduled task '$taskName' -> $LauncherPath"
}

# ── Backend ──────────────────────────────────────────────────────────────
$backendLauncher = Join-Path $LauncherDir "backend.cmd"
Write-LauncherScript -Path $backendLauncher -Content @"
@echo off
cd /d "$BackendDir"
uv run uvicorn src.main:socket_app --port $DefaultBackendPort
"@
Register-TradingBotTask -Name "Backend" -LauncherPath $backendLauncher

# ── Frontend ─────────────────────────────────────────────────────────────
$frontendPort = Read-DotEnvValue -Name "TB_FRONTEND_PORT"
if ([string]::IsNullOrWhiteSpace($frontendPort)) { $frontendPort = "$DefaultFrontendPort" }
$frontendLauncher = Join-Path $LauncherDir "frontend.cmd"
Write-LauncherScript -Path $frontendLauncher -Content @"
@echo off
cd /d "$FrontendDir"
pnpm dev --port $frontendPort
"@
Register-TradingBotTask -Name "Frontend" -LauncherPath $frontendLauncher

# ── Per-account gateway + terminal (native, no Wine -- see
#    gateway/README.md "Option B -- Windows VPS") ─────────────────────────
foreach ($account in $state.accounts) {
    $accountId = $account.id

    $secretVarName = $account.gateway_secret_env
    if ([string]::IsNullOrWhiteSpace($secretVarName)) { $secretVarName = "TB_GATEWAY_SHARED_SECRET" }
    $secretValue = Read-DotEnvValue -Name $secretVarName

    $gatewayHost = $account.gateway_host
    if ([string]::IsNullOrWhiteSpace($gatewayHost)) { $gatewayHost = $DefaultGatewayHost }
    $gatewayPort = $account.gateway_port
    if (-not $gatewayPort) { $gatewayPort = $DefaultGatewayPort }

    $terminalPath = $account.terminal_path
    if ([string]::IsNullOrWhiteSpace($terminalPath)) { $terminalPath = $DefaultTerminalPath }

    # ── Terminal ────────────────────────────────────────────────────────
    $terminalLauncher = Join-Path $LauncherDir "terminal-$accountId.cmd"
    Write-LauncherScript -Path $terminalLauncher -Content @"
@echo off
start "" "$terminalPath"
"@
    Register-TradingBotTask -Name "Terminal-$accountId" -LauncherPath $terminalLauncher

    # ── Gateway ─────────────────────────────────────────────────────────
    # Value copied from .env's $secretVarName at generation time -- there's
    # no way for a scheduled task action to shell out to read .env on every
    # start the way `make dev-gateway` does. Re-run this script after
    # rotating the secret in .env.
    $gatewayLauncher = Join-Path $LauncherDir "gateway-$accountId.cmd"
    Write-LauncherScript -Path $gatewayLauncher -Content @"
@echo off
cd /d "$GatewayDir"
set GATEWAY_SHARED_SECRET=$secretValue
set GATEWAY_HOST=$gatewayHost
set GATEWAY_PORT=$gatewayPort
set MT5_TERMINAL_PATH=$terminalPath
python run_gateway.py
"@
    Register-TradingBotTask -Name "Gateway-$accountId" -LauncherPath $gatewayLauncher
}

Write-Host ""
Write-Host "Done. Manage with:"
Write-Host "  Get-ScheduledTask -TaskName '$TaskPrefix*' | Format-Table TaskName,State -AutoSize"
Write-Host "  python installer\manage.py status"
