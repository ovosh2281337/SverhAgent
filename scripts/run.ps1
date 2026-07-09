param(
    [switch]$PreflightOnly,
    [int]$EmbedReadyTimeoutSec = 0
)

# Start the local bot stack from the project root:
#   powershell -File scripts\run.ps1
#
# This launcher is intentionally strict:
# - it never falls back to global Python;
# - it reads .env through scripts.preflight, not PowerShell regexes;
# - it starts bundled embeddings only when EMBED_MODE=bundled;
# - it waits for real /health readiness before starting the bot;
# - it refuses a second launcher for the same project path.

$ErrorActionPreference = "Stop"
Set-StrictMode -Version 2.0

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root

function Invoke-Native {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [string[]]$Arguments = @(),
        [string]$Label = ""
    )
    if ($Label -eq "") { $Label = $FilePath }
    & $FilePath @Arguments
    $Code = $LASTEXITCODE
    if ($Code -ne 0) {
        throw "$Label failed with exit code $Code"
    }
}

function Invoke-CaptureNative {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [string[]]$Arguments = @(),
        [string]$Label = ""
    )
    if ($Label -eq "") { $Label = $FilePath }
    $Output = & $FilePath @Arguments 2>&1
    $Code = $LASTEXITCODE
    $Text = ($Output | Out-String).Trim()
    if ($Code -ne 0) {
        if ($Text -eq "") {
            throw "$Label failed with exit code $Code"
        }
        throw "$Label failed with exit code $Code`n$Text"
    }
    return $Text
}

function Get-ProjectMutexName {
    param([Parameter(Mandatory = $true)][string]$Path)
    $Bytes = [System.Text.Encoding]::UTF8.GetBytes($Path.ToLowerInvariant())
    $Sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        $HashBytes = $Sha.ComputeHash($Bytes)
    } finally {
        $Sha.Dispose()
    }
    $Hash = -join ($HashBytes | ForEach-Object { $_.ToString("x2") })
    return "Local\agentForSverh-run-$($Hash.Substring(0, 16))"
}

function Show-LogTail {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [int]$Lines = 80
    )
    if (Test-Path -LiteralPath $Path) {
        Write-Host "[run] log tail: $Path" -ForegroundColor DarkGray
        Get-Content -LiteralPath $Path -Tail $Lines -ErrorAction SilentlyContinue
    }
}

function Stop-ProcessTree {
    param([Parameter(Mandatory = $true)][int]$TargetPid)

    $Children = @()
    try {
        $Children = @(Get-CimInstance Win32_Process -Filter "ParentProcessId=$TargetPid" -ErrorAction SilentlyContinue)
    } catch {
        $Children = @()
    }

    foreach ($Child in $Children) {
        Stop-ProcessTree -TargetPid ([int]$Child.ProcessId)
    }

    $Proc = Get-Process -Id $TargetPid -ErrorAction SilentlyContinue
    if ($null -ne $Proc) {
        Stop-Process -Id $TargetPid -Force -ErrorAction SilentlyContinue
    }
}

function Wait-EmbedReady {
    param(
        [Parameter(Mandatory = $true)][string]$Url,
        [Parameter(Mandatory = $true)][int]$ExpectedDim,
        [Parameter(Mandatory = $true)][int]$TimeoutSec,
        [Parameter(Mandatory = $true)]$Process,
        [Parameter(Mandatory = $true)][string]$OutLog,
        [Parameter(Mandatory = $true)][string]$ErrLog
    )

    $Deadline = (Get-Date).AddSeconds($TimeoutSec)
    $LastStatus = "not checked yet"
    $NextProgress = Get-Date

    while ((Get-Date) -lt $Deadline) {
        $Process.Refresh()
        if ($Process.HasExited) {
            Show-LogTail -Path $OutLog
            Show-LogTail -Path $ErrLog
            throw "embed server exited before readiness, exit code $($Process.ExitCode)"
        }

        try {
            $Response = Invoke-WebRequest -UseBasicParsing -Uri $Url -TimeoutSec 5
            $Health = $Response.Content | ConvertFrom-Json
            if (($Health.status -eq "ok") -and ([int]$Health.dim -eq $ExpectedDim)) {
                Write-Host "[run] embed ready: $Url dim=$($Health.dim) device=$($Health.device)" -ForegroundColor Green
                return
            }
            $LastStatus = "bad health payload: $($Response.Content)"
        } catch {
            $LastStatus = $_.Exception.Message
        }

        if ((Get-Date) -ge $NextProgress) {
            Write-Host "[run] waiting for embed health: $LastStatus" -ForegroundColor DarkGray
            $NextProgress = (Get-Date).AddSeconds(5)
        }
        Start-Sleep -Seconds 2
    }

    Show-LogTail -Path $OutLog
    Show-LogTail -Path $ErrLog
    throw "embed server did not become ready within $TimeoutSec seconds: $LastStatus"
}

$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Py)) {
    throw "Project venv is missing. Run: powershell -File scripts\setup.ps1"
}

$MutexName = Get-ProjectMutexName -Path $Root
$CreatedNew = $false
$Mutex = New-Object System.Threading.Mutex($true, $MutexName, [ref]$CreatedNew)
$MutexOwned = $false
if (-not $CreatedNew) {
    $Mutex.Dispose()
    throw "Another scripts\run.ps1 is already running for this project: $Root"
}
$MutexOwned = $true

$EmbedProcess = $null

try {
    Invoke-Native $Py @("-m", "scripts.preflight", "runtime") "runtime preflight"
    Invoke-Native $Py @("-m", "scripts.preflight", "config") "config preflight"
    $LauncherJson = Invoke-CaptureNative $Py @("-m", "scripts.preflight", "launcher-env", "--json") "launcher env"
    $Launcher = $LauncherJson | ConvertFrom-Json

    if ($PreflightOnly) {
        Write-Host "[run] preflight ok" -ForegroundColor Green
        return
    }

    if ([bool]$Launcher.start_local_embed) {
        Invoke-Native $Py @("-m", "scripts.preflight", "model") "embedding model preflight"
        Invoke-Native $Py @(
            "-m", "scripts.preflight", "port-free",
            "--host", [string]$Launcher.embed_host,
            "--port", [string]$Launcher.embed_port
        ) "embedding port preflight"

        $RunDir = Join-Path $Root ".run"
        New-Item -ItemType Directory -Force -Path $RunDir | Out-Null
        $OutLog = Join-Path $RunDir "embed.out.log"
        $ErrLog = Join-Path $RunDir "embed.err.log"
        Remove-Item -LiteralPath $OutLog, $ErrLog -ErrorAction SilentlyContinue

        Write-Host "[run] starting bundled embed server on $($Launcher.embed_base_url)" -ForegroundColor Cyan
        $EmbedProcess = Start-Process `
            -FilePath $Py `
            -ArgumentList @("-m", "scripts.serve_embed") `
            -WorkingDirectory $Root `
            -RedirectStandardOutput $OutLog `
            -RedirectStandardError $ErrLog `
            -WindowStyle Hidden `
            -PassThru

        $TimeoutSec = [int]$Launcher.embed_ready_timeout_sec
        if ($EmbedReadyTimeoutSec -gt 0) {
            $TimeoutSec = $EmbedReadyTimeoutSec
        }
        Wait-EmbedReady `
            -Url ([string]$Launcher.embed_health_url) `
            -ExpectedDim ([int]$Launcher.embed_expected_dim) `
            -TimeoutSec $TimeoutSec `
            -Process $EmbedProcess `
            -OutLog $OutLog `
            -ErrLog $ErrLog
    } elseif ($Launcher.embed_mode -eq "external") {
        Write-Host "[run] external embeddings: $($Launcher.embed_base_url)" -ForegroundColor Cyan
    } else {
        Write-Host "[run] embeddings disabled" -ForegroundColor DarkGray
    }

    $Delay = [int]$Launcher.bot_restart_initial_delay_sec
    $MaxDelay = [int]$Launcher.bot_restart_max_delay_sec
    $MaxFastFailures = [int]$Launcher.bot_restart_max_fast_failures
    $FastFailures = 0

    while ($true) {
        $Start = Get-Date
        Write-Host "[watchdog] starting bot $(Get-Date -Format s)" -ForegroundColor Cyan
        & $Py -m src.bot
        $ExitCode = $LASTEXITCODE
        $Uptime = (New-TimeSpan -Start $Start -End (Get-Date)).TotalSeconds

        if ($ExitCode -eq 0) {
            Write-Host "[watchdog] bot exited normally" -ForegroundColor DarkGray
            break
        }

        if ($Uptime -lt 30) {
            $FastFailures += 1
        } else {
            $FastFailures = 0
            $Delay = [int]$Launcher.bot_restart_initial_delay_sec
        }

        if ($FastFailures -ge $MaxFastFailures) {
            throw "bot failed quickly $FastFailures times; not restarting deterministic failure"
        }

        Write-Host "[watchdog] bot exited code=$ExitCode uptime=$([int]$Uptime)s; restart in $Delay s" -ForegroundColor Yellow
        Start-Sleep -Seconds $Delay
        $Delay = [Math]::Min([int]($Delay * 2), $MaxDelay)
    }
} finally {
    if ($null -ne $EmbedProcess) {
        $EmbedProcess.Refresh()
        if (-not $EmbedProcess.HasExited) {
            Write-Host "[run] stopping bundled embed server pid=$($EmbedProcess.Id)" -ForegroundColor DarkGray
            Stop-ProcessTree -TargetPid ([int]$EmbedProcess.Id)
        }
    }
    if ($MutexOwned) {
        $Mutex.ReleaseMutex()
        $Mutex.Dispose()
    }
}
