param(
    [int]$TargetEpochs = 20
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
        [double]$VisualTrainFraction,
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
        "--visual-train-fraction", $VisualTrainFraction,
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
    @{ RunName = "proposal_r2plus1d_visual_only_f100_scout_e20_v1"; Method = "visual_only"; VisualTrainFraction = 1.0; LambdaAlign = 0.0 },
    @{ RunName = "proposal_r2plus1d_noalign_f100_scout_e20_v1"; Method = "proposed"; VisualTrainFraction = 1.0; LambdaAlign = 0.0 },
    @{ RunName = "proposal_r2plus1d_align_f100_scout_e20_v1"; Method = "proposed"; VisualTrainFraction = 1.0; LambdaAlign = 1.0 },
    @{ RunName = "proposal_r2plus1d_visual_only_f025_scout_e20_v1"; Method = "visual_only"; VisualTrainFraction = 0.25; LambdaAlign = 0.0 },
    @{ RunName = "proposal_r2plus1d_noalign_f025_scout_e20_v1"; Method = "proposed"; VisualTrainFraction = 0.25; LambdaAlign = 0.0 },
    @{ RunName = "proposal_r2plus1d_align_f025_scout_e20_v1"; Method = "proposed"; VisualTrainFraction = 0.25; LambdaAlign = 1.0 }
)

foreach ($experiment in $experiments) {
    Invoke-Experiment @experiment
}
