<#
.SYNOPSIS
    Start the AI OCR API locally, without Docker.

.DESCRIPTION
    - Uses a project-local .venv when one exists, otherwise the `python` on PATH.
    - Creates .venv and installs requirements.txt only when imports are missing.
    - Runs a single uvicorn worker: PaddleOCR models are cached per process and the
      rate limiter/quota windows are in-process, so multiple workers would each
      load their own models and enforce their own limits.
    - Configuration is read from .env (see .env.example); no container involved.

.PARAMETER Port
    TCP port to listen on. Default 8000.

.PARAMETER BindAddress
    Interface to bind. Default 127.0.0.1 (this machine only).
    Use 0.0.0.0 to expose it on the LAN - put it behind a TLS reverse proxy
    (IIS/nginx/Caddy) before making it reachable from the internet.

.PARAMETER Restart
    Stop whatever already listens on the port (if it looks like a python process)
    before starting. Without it a busy port is reported and nothing is killed.

.EXAMPLE
    .\run.ps1
    .\run.ps1 -Port 8080
    .\run.ps1 -BindAddress 0.0.0.0 -Restart
#>
[CmdletBinding()]
param(
    [int]$Port = 8000,
    [string]$BindAddress = "127.0.0.1",
    [switch]$Restart
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

function Get-Interpreter {
    if (Test-Path -LiteralPath $venvPython) { return $venvPython }
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if (-not $cmd) {
        throw "Python 3.11+ not found on PATH. Install it from python.org (tick 'Add python.exe to PATH') and re-run this script."
    }
    return $cmd.Source
}

function Get-MissingDependencies {
    # Returns a comma-separated list of missing imports, or $null when all present.
    # Single quotes only: PowerShell 5.1 mangles embedded double quotes when it
    # builds the native command line, which would break the Python source.
    $probe = "import importlib.util,sys; mods=('fastapi','uvicorn','paddleocr','paddle','fitz','docx','openpyxl','PIL','openai','bs4','dotenv','numpy'); missing=[m for m in mods if importlib.util.find_spec(m) is None]; print(','.join(missing)); sys.exit(1 if missing else 0)"
    $probeOut = & $python -c $probe 2>$null
    if ($LASTEXITCODE -eq 0) { return $null }
    return (($probeOut | Out-String).Trim())
}

$python = Get-Interpreter
Write-Host "Using Python: $python"

$version = (& $python -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null | Out-String).Trim()
if ($LASTEXITCODE -ne 0 -or -not $version) {
    throw "Could not run $python. If that is the Microsoft Store alias stub, install Python 3.11 from python.org (tick 'Add python.exe to PATH')."
}
$major, $minor = $version.Split(".")
if ([int]$major -lt 3 -or ([int]$major -eq 3 -and [int]$minor -lt 10)) {
    Write-Host "WARNING: Python $version detected. PaddleOCR 3.x expects 3.10-3.12 (3.11 recommended)." -ForegroundColor Yellow
}

if (Get-MissingDependencies) {
    Write-Host "Installing dependencies (first run only, this can take a few minutes)..."
    if (-not (Test-Path -LiteralPath $venvPython)) {
        & $python -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw "Could not create .venv. Install the 'venv' module and retry." }
        $python = $venvPython
        Write-Host "Created .venv -> $python"
    }
    & $python -m pip install --upgrade pip
    & $python -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw "pip install failed. Check your network / proxy settings and retry." }
    $stillMissing = Get-MissingDependencies
    if ($stillMissing) { throw "Dependencies are still missing after pip install: $stillMissing" }
}

# --- make sure the port is free ------------------------------------------------
$listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
    Select-Object -First 1
if ($listener) {
    $ownerId = $listener.OwningProcess
    $proc = Get-Process -Id $ownerId -ErrorAction SilentlyContinue
    $procName = if ($proc) { $proc.ProcessName } else { "unknown" }

    if ($Restart -and $procName -like "python*") {
        Write-Host "Stopping $procName (PID $ownerId) that is holding port $Port..."
        # taskkill is used instead of Stop-Process: a sandboxed Stop-Process can
        # fail silently and leave the listener alive (the new worker then cannot bind).
        & taskkill /F /PID $ownerId 2>&1 | Out-Null
        Start-Sleep -Seconds 2
    }
    if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
        $hint = if ($Restart) { "Only python processes are stopped automatically." }
                else { "Re-run with -Restart to stop a stale python/uvicorn instance." }
        throw "Port $Port is already in use by $procName (PID $ownerId). $hint"
    }
}

# --- advisory checks (local runs default to demo-friendly, not internet-safe) ---
$envPath = Join-Path $PSScriptRoot ".env"
if (Test-Path -LiteralPath $envPath) {
    $envText = Get-Content -LiteralPath $envPath -Raw
    if ($envText -match "(?m)^\s*OCR_DEMO_MODE\s*=\s*(1|true|yes|on)\s*$") {
        Write-Host "WARNING: OCR_DEMO_MODE is on - the portal offers one-click admin login. Set it to false before exposing this server." -ForegroundColor Yellow
    }
    if ($BindAddress -ne "127.0.0.1" -and $BindAddress -ne "localhost" -and
        $envText -notmatch "(?m)^\s*OCR_COOKIE_SECURE\s*=\s*(1|true|yes|on)\s*$") {
        Write-Host "WARNING: bound to $BindAddress without OCR_COOKIE_SECURE=true - session cookies would travel over plain HTTP." -ForegroundColor Yellow
    }
}

$shown = if ($BindAddress -eq "0.0.0.0") { "localhost" } else { $BindAddress }
Write-Host ""
Write-Host "AI OCR listening on http://${shown}:${Port}" -ForegroundColor Green
Write-Host "  portal   http://${shown}:${Port}/"
Write-Host "  demo     http://${shown}:${Port}/demo"
Write-Host "  API docs http://${shown}:${Port}/docs"
Write-Host "Press Ctrl+C to stop."
Write-Host ""

& $python -m uvicorn app.main:app --host $BindAddress --port $Port
exit $LASTEXITCODE
