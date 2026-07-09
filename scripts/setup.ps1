param(
    [switch]$WithModels,
    [switch]$WithEmbeddings,
    [switch]$WithStt,
    [switch]$SkipDocker,
    [int]$DockerCheckTimeoutSec = 60,
    [int]$DockerTimeoutSec = 300
)

# One-time setup from the project root:
#   powershell -File scripts\setup.ps1
#
# Optional bundled embeddings:
#   powershell -File scripts\setup.ps1 -WithEmbeddings
# Same, friendlier alias:
#   powershell -File scripts\setup.ps1 -WithModels
# Optional voice/STT models:
#   powershell -File scripts\setup.ps1 -WithStt

$ErrorActionPreference = "Stop"
Set-StrictMode -Version 2.0

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root

if ($WithModels) {
    $WithEmbeddings = $true
    $WithStt = $true
}

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

function Invoke-NativeWithTimeout {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [string[]]$Arguments = @(),
        [Parameter(Mandatory = $true)][int]$TimeoutSec,
        [string]$Label = ""
    )
    if ($Label -eq "") { $Label = $FilePath }
    $Psi = New-Object System.Diagnostics.ProcessStartInfo
    $Psi.FileName = $FilePath
    $Psi.Arguments = ($Arguments -join " ")
    $Psi.WorkingDirectory = $Root
    $Psi.UseShellExecute = $false
    $Proc = [System.Diagnostics.Process]::Start($Psi)
    if (-not $Proc.WaitForExit($TimeoutSec * 1000)) {
        Stop-ProcessTree -TargetPid ([int]$Proc.Id)
        throw "$Label timed out after $TimeoutSec seconds"
    }
    if ($Proc.ExitCode -ne 0) {
        throw "$Label failed with exit code $($Proc.ExitCode)"
    }
}

Write-Host "== 1/4 Python venv + deps ==" -ForegroundColor Cyan
Invoke-Native "python" @(
    "-c",
    "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 'Python 3.12 is required, current is ' + sys.version.split()[0])"
) "Python 3.12 check"

if (-not (Test-Path -LiteralPath ".venv\Scripts\python.exe")) {
    Invoke-Native "python" @("-m", "venv", ".venv") "create venv"
}

$Py = Join-Path $Root ".venv\Scripts\python.exe"
Invoke-Native $Py @("-m", "pip", "install", "--quiet", "--upgrade", "pip") "upgrade pip"
Invoke-Native $Py @("-m", "pip", "install", "--quiet", "-r", "requirements.txt") "install requirements"
Invoke-Native $Py @("-m", "scripts.preflight", "runtime") "runtime preflight"
Write-Host "   venv ready"

Write-Host "== 2/4 .env ==" -ForegroundColor Cyan
if (-not (Test-Path -LiteralPath ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "   created .env; edit TELEGRAM_BOT_TOKEN and OPENAI_BASE_URL" -ForegroundColor Yellow
} else {
    Write-Host "   .env already exists, left untouched"
}

Write-Host "== 3/4 Postgres (Docker) ==" -ForegroundColor Cyan
if ($SkipDocker) {
    Write-Host "   skipped by -SkipDocker" -ForegroundColor Yellow
} else {
    Invoke-NativeWithTimeout "docker" @("version") $DockerCheckTimeoutSec "docker daemon check"
    Invoke-NativeWithTimeout "docker" @("compose", "up", "-d", "db") $DockerTimeoutSec "docker compose up db"
    Write-Host "   Postgres requested on localhost:5432"
}

Write-Host "== 4/4 Optional model weights ==" -ForegroundColor Cyan
if ($WithEmbeddings) {
    Invoke-Native $Py @("-m", "pip", "install", "--quiet", "--upgrade", "-r", "requirements-embed.txt") "install embedding requirements"
    Invoke-Native $Py @("-m", "scripts.download_model") "download embedding model"
    Invoke-Native $Py @("-m", "scripts.preflight", "model") "embedding model preflight"
    Write-Host "   bundled embedding model ready"
} else {
    Write-Host "   embeddings skipped; run setup.ps1 -WithEmbeddings before EMBED_MODE=bundled"
}
if ($WithStt) {
    Invoke-Native $Py @("-m", "pip", "uninstall", "--quiet", "-y", "gigaam") "remove old GigaAM package"
    Invoke-Native $Py @(
        "-m", "pip", "install", "--quiet", "--upgrade",
        "--upgrade-strategy", "eager", "-r", "requirements-stt.txt"
    ) "install STT requirements"
    Invoke-Native $Py @("-m", "scripts.download_stt_model") "download STT models"
    Write-Host "   STT models ready"
} else {
    Write-Host "   STT skipped; run setup.ps1 -WithStt before enabling voice messages"
}

Write-Host ""
Write-Host "Setup done. Next:" -ForegroundColor Green
Write-Host "  1. Edit .env: TELEGRAM_BOT_TOKEN and OPENAI_BASE_URL"
Write-Host "  2. Optional local embeddings: set EMBED_MODE=bundled"
Write-Host "  3. Optional voice: set STT_BASE_URL=http://localhost:8301 and start scripts.serve_stt"
Write-Host "  4. Start: powershell -File scripts\run.ps1"
