# Submission variant note:
# This packaged copy emphasizes experiment orchestration, documentation, and figure scripts.
# Package owner: Yuchen Lai
param(
    [int]$TargetEpochs = 100
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"
$train = Join-Path $root "train.py"
$runsDir = Join-Path $root "runs"

function Get-LatestCheckpoint {
    param(
        [string]$RunDir
    )

    $checkpoints = Get-ChildItem -Path $RunDir -Filter "checkpoint_epoch_*.pt" -File -ErrorAction SilentlyContinue
    if (-not $checkpoints) {
        return $null
    }

    return $checkpoints |
        Sort-Object { [int]($_.BaseName -replace "^checkpoint_epoch_", "") } -Descending |
        Select-Object -First 1
}

function Get-CompletedEpochs {
    param(
        [string]$RunDir
    )

    $metricsPath = Join-Path $RunDir "metrics.jsonl"
    if (-not (Test-Path $metricsPath)) {
        return 0
    }

    return [int]((Get-Content $metricsPath | Measure-Object -Line).Lines)
}

function Invoke-Experiment {
    param(
        [string]$RunName,
        [string]$Method,
        [double]$LambdaAlign
    )

    $runDir = Join-Path $runsDir $RunName
    New-Item -ItemType Directory -Force -Path $runDir | Out-Null

    $completedEpochs = Get-CompletedEpochs -RunDir $runDir
    $checkpointTarget = Join-Path $runDir "checkpoint_epoch_$TargetEpochs.pt"
    if ($completedEpochs -ge $TargetEpochs -and (Test-Path $checkpointTarget)) {
        Write-Host "Skipping $RunName; already completed $TargetEpochs epochs."
        return
    }

    $stdoutPath = Join-Path $runDir "stdout.log"
    $stderrPath = Join-Path $runDir "stderr.log"

    $args = @(
        $train,
        "--device", "cuda",
        "--epochs", $TargetEpochs,
        "--method", $Method,
        "--temporal-model", "gru",
        "--batch-size", "16",
        "--num-workers", "4",
        "--lr", "5e-5",
        "--clip-length", "8",
        "--clip-stride", "8",
        "--frame-size", "112",
        "--audio-frames", "256",
        "--lambda-align", $LambdaAlign,
        "--visual-backbone", "video_r2plus1d_18_kinetics400",
        "--visual-train-fraction", "0.25",
        "--output-dir", $runDir,
        "--seed", "42"
    )

    $latestCheckpoint = Get-LatestCheckpoint -RunDir $runDir
    if ($completedEpochs -gt 0 -and $latestCheckpoint -ne $null) {
        $args += @("--resume-from", $latestCheckpoint.FullName)
        Write-Host "Resuming $RunName from $($latestCheckpoint.Name)."
    } else {
        Write-Host "Starting $RunName from scratch."
    }

    Push-Location $root
    try {
        & $python @args 1>> $stdoutPath 2>> $stderrPath
        if ($LASTEXITCODE -ne 0) {
            throw "Training failed for $RunName with exit code $LASTEXITCODE."
        }
    } finally {
        Pop-Location
    }
}

$experiments = @(
    @{ RunName = "proposal_r2plus1d_visual_only_f025_formal_e100_v1"; Method = "visual_only"; LambdaAlign = 0.0 },
    @{ RunName = "proposal_r2plus1d_align_l01_f025_formal_e100_v1"; Method = "proposed"; LambdaAlign = 0.1 }
)

foreach ($experiment in $experiments) {
    Invoke-Experiment @experiment
}
