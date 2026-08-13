# modelo - single-file installer for Windows (PowerShell 5+)
#
#   Usage (PowerShell):   irm <base-url>/install.ps1 | iex
#   Usage (curl.exe):     curl -fsSL <base-url>/install.ps1 -o install.ps1; powershell -ExecutionPolicy Bypass -File install.ps1
#
# Downloads the one-file "modelo" program, adds a "modelo.cmd" launcher so you
# can type "modelo" from any folder, and installs the Python dependencies.
param(
    [string]$InstallDir = "$env:USERPROFILE\.local\bin"
)

$ErrorActionPreference = "Stop"

# Override the raw base URL that hosts install.ps1 and modelo.py
#   - GitHub repo:   https://raw.githubusercontent.com/USER/REPO/main
#   - Gist:          https://gist.githubusercontent.com/USER/GIST_ID/raw
if ($env:MODELO_URL) { $BaseUrl = $env:MODELO_URL }
else                  { $BaseUrl = "https://raw.githubusercontent.com/Merunism19/modelo/main" }

# --- Check prerequisites -----------------------------------------------------
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "error: python is required (install it from https://www.python.org)" -ForegroundColor Red
    exit 1
}

# --- Download the program -----------------------------------------------------
New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
Write-Host ">> Downloading modelo to $InstallDir\modelo.py"
Invoke-WebRequest -Uri "$BaseUrl/modelo.py" -OutFile (Join-Path $InstallDir "modelo.py") -UseBasicParsing

# Launcher so typing "modelo" works from any directory.
$launcher = @"
@echo off
python "%~dp0modelo.py" %*
"@
Set-Content -Path (Join-Path $InstallDir "modelo.cmd") -Value $launcher -Encoding ASCII

# --- Install Python dependencies ---------------------------------------------
Write-Host ">> Installing Python dependencies (typer, huggingface_hub, requests)"
python -m pip install --user typer huggingface_hub requests

# --- Done --------------------------------------------------------------------
Write-Host ""
Write-Host "modelo installed." -ForegroundColor Green
Write-Host "  - Add $InstallDir to your PATH (System Properties > Environment Variables),"
Write-Host "    or for the current session run:  `$env:Path = '$InstallDir;' + `$env:Path"
Write-Host "  - Then try:  modelo --help"
Write-Host "  - Optional (to serve models):  python -m pip install llama-cpp-python uvicorn"
