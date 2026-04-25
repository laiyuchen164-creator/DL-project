param(
    [int]$Epochs = 5,
    [int]$BatchSize = 16,
    [int]$NumWorkers = 4,
    [int]$ClipLength = 8,
    [int]$ClipStride = 8,
    [int]$FrameSize = 112,
    [int]$AudioFrames = 256,
    [string]$RunPrefix = "gpu_baselines_20260325",
    [string[]]$Methods = @("visual_only", "late_fusion", "dann")
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    throw "Python not found at $python"
}

Push-Location $projectRoot
try {
    foreach ($method in $Methods) {
        $outputDir = Join-Path $projectRoot ("runs\" + $RunPrefix + "_" + $method)
        New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

        $stdoutLog = Join-Path $outputDir "stdout.log"
        $stderrLog = Join-Path $outputDir "stderr.log"

        $args = @(
            "train.py",
            "--method", $method,
            "--device", "cuda",
            "--epochs", $Epochs,
            "--batch-size", $BatchSize,
            "--num-workers", $NumWorkers,
            "--clip-length", $ClipLength,
            "--clip-stride", $ClipStride,
            "--frame-size", $FrameSize,
            "--audio-frames", $AudioFrames,
            "--output-dir", $outputDir
        )

        Write-Host "[$(Get-Date -Format s)] Starting $method -> $outputDir"
        & $python @args 1> $stdoutLog 2> $stderrLog
        if ($LASTEXITCODE -ne 0) {
            throw "Training failed for method=$method. See $stderrLog"
        }
        Write-Host "[$(Get-Date -Format s)] Finished $method"
    }
}
finally {
    Pop-Location
}
