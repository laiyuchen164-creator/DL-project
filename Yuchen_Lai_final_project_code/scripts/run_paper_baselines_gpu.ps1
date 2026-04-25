param(
    [string]$RunPrefix = "paper_baselines_20260325",
    [int]$BatchSize = 16,
    [int]$NumWorkers = 4,
    [int]$Epochs = 5,
    [int]$ClipLength = 8,
    [int]$ClipStride = 8,
    [int]$FrameSize = 112,
    [int]$AudioFrames = 256,
    [string[]]$Methods = @("ortega_feature_svr", "ortega_decision_svr", "zhang_leader_follower")
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$sharedFeatureCache = Join-Path $projectRoot ("runs\" + $RunPrefix + "_shared_features")

Push-Location $projectRoot
try {
    foreach ($method in $Methods) {
        $outputDir = Join-Path $projectRoot ("runs\" + $RunPrefix + "_" + $method)
        New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

        $args = @(
            "scripts\train_paper_baseline.py",
            "--method", $method,
            "--device", "cuda",
            "--batch-size", $BatchSize,
            "--num-workers", $NumWorkers,
            "--epochs", $Epochs,
            "--clip-length", $ClipLength,
            "--clip-stride", $ClipStride,
            "--frame-size", $FrameSize,
            "--audio-frames", $AudioFrames,
            "--output-dir", $outputDir
        )

        if ($method -like "ortega_*") {
            $args += @("--feature-cache-dir", $sharedFeatureCache)
        }

        $stdoutLog = Join-Path $outputDir "stdout.log"
        $stderrLog = Join-Path $outputDir "stderr.log"

        Write-Host "[$(Get-Date -Format s)] Starting $method"
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
