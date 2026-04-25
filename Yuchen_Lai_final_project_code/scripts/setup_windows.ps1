# Submission variant note:
# This packaged copy emphasizes experiment orchestration, documentation, and figure scripts.
# Package owner: Yuchen Lai
param(
    [switch]$RunSmokeTest,
    [switch]$CompileLatex
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$TectonicDir = Join-Path $ProjectRoot ".tools\tectonic"
$TectonicExe = Join-Path $TectonicDir "tectonic.exe"

function Write-Step {
    param([string]$Message)
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Assert-Python311 {
    try {
        $version = & py -3.11 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"
        Write-Step "Found Python $version via py -3.11"
    }
    catch {
        throw "Python 3.11 was not found. Install Python 3.11 and make sure the 'py' launcher is available."
    }
}

function Ensure-Venv {
    if (-not (Test-Path $VenvPython)) {
        Write-Step "Creating virtual environment in .venv"
        & py -3.11 -m venv (Join-Path $ProjectRoot ".venv")
    }
    else {
        Write-Step "Reusing existing virtual environment"
    }
}

function Install-PythonDependencies {
    Write-Step "Upgrading pip/wheel and pinning setuptools for torch compatibility"
    & $VenvPython -m pip install --upgrade pip wheel "setuptools<82"

    Write-Step "Installing PyTorch CPU wheels"
    & $VenvPython -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

    $requirementsPath = Join-Path $ProjectRoot "requirements.txt"
    if (-not (Test-Path $requirementsPath)) {
        throw "requirements.txt was not found at $requirementsPath"
    }

    Write-Step "Installing project dependencies from requirements.txt"
    & $VenvPython -m pip install -r $requirementsPath
}

function Ensure-Tectonic {
    if (Test-Path $TectonicExe) {
        Write-Step "Reusing local Tectonic install"
        return
    }

    Write-Step "Installing local Tectonic into .tools\\tectonic"
    New-Item -ItemType Directory -Force $TectonicDir | Out-Null
    powershell -ExecutionPolicy ByPass -Command "[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://drop-ps1.fullyjustified.net'))" | Out-Host
}

function Run-SmokeTest {
    Write-Step "Running smoke test"
    & $VenvPython (Join-Path $ProjectRoot "scripts\smoke_test.py")
}

function Build-Latex {
    Write-Step "Compiling preliminary_report.tex"
    Push-Location (Join-Path $ProjectRoot "latex")
    try {
        & $TectonicExe "preliminary_report.tex"
    }
    finally {
        Pop-Location
    }

    Copy-Item (Join-Path $ProjectRoot "latex\preliminary_report.pdf") (Join-Path $ProjectRoot "Shuaikang_Hou_preliminary_project_report.pdf") -Force
}

Assert-Python311
Ensure-Venv
Install-PythonDependencies
Ensure-Tectonic

if ($RunSmokeTest) {
    Run-SmokeTest
}

if ($CompileLatex) {
    Build-Latex
}

Write-Step "Setup complete"
Write-Host ""
Write-Host "Activate the environment with:" -ForegroundColor Green
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "Run the smoke test with:" -ForegroundColor Green
Write-Host "  .\.venv\Scripts\python scripts\smoke_test.py"
Write-Host ""
Write-Host "Compile the report with:" -ForegroundColor Green
Write-Host "  Push-Location .\latex"
Write-Host "  ..\.tools\tectonic\tectonic.exe preliminary_report.tex"
Write-Host "  Pop-Location"
