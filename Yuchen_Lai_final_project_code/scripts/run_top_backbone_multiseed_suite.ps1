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

    $lineCount = (Get-Content $metricsPath | Measure-Object -Line).Lines
    return [int]$lineCount
}

function Invoke-Experiment {
    param(
        [string]$RunName,
        [int]$Seed,
        [string]$Backbone,
        [int]$BatchSize,
        [int]$ClipLength,
        [int]$ClipStride,
        [int]$FrameSize
    )

    $runDir = Join-Path $runsDir $RunName
    New-Item -ItemType Directory -Force -Path $runDir | Out-Null

    $completedEpochs = Get-CompletedEpochs -RunDir $runDir
    $checkpoint100 = Join-Path $runDir "checkpoint_epoch_$TargetEpochs.pt"
    if ($completedEpochs -ge $TargetEpochs -and (Test-Path $checkpoint100)) {
        Write-Host "Skipping $RunName; already completed $TargetEpochs epochs."
        return
    }

    $stdoutPath = Join-Path $runDir "stdout.log"
    $stderrPath = Join-Path $runDir "stderr.log"

    $args = @(
        $train,
        "--device", "cuda",
        "--epochs", $TargetEpochs,
        "--method", "proposed",
        "--temporal-model", "gru",
        "--batch-size", $BatchSize,
        "--num-workers", "4",
        "--lr", "5e-5",
        "--clip-length", $ClipLength,
        "--clip-stride", $ClipStride,
        "--frame-size", $FrameSize,
        "--audio-frames", "256",
        "--lambda-align", "0",
        "--visual-backbone", $Backbone,
        "--output-dir", $runDir,
        "--seed", $Seed
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
    @{
        RunName = "video_mvit_v2_s_gru_noalign_formal_e100_seed7_v1"
        Seed = 7
        Backbone = "video_mvit_v2_s_kinetics400"
        BatchSize = 4
        ClipLength = 16
        ClipStride = 16
        FrameSize = 224
    },
    @{
        RunName = "video_mvit_v2_s_gru_noalign_formal_e100_seed123_v1"
        Seed = 123
        Backbone = "video_mvit_v2_s_kinetics400"
        BatchSize = 4
        ClipLength = 16
        ClipStride = 16
        FrameSize = 224
    },
    @{
        RunName = "video_mvit_v2_s_gru_noalign_formal_e100_seed3407_v1"
        Seed = 3407
        Backbone = "video_mvit_v2_s_kinetics400"
        BatchSize = 4
        ClipLength = 16
        ClipStride = 16
        FrameSize = 224
    },
    @{
        RunName = "video_r2plus1d18_gru_noalign_formal_e100_seed7_v1"
        Seed = 7
        Backbone = "video_r2plus1d_18_kinetics400"
        BatchSize = 16
        ClipLength = 8
        ClipStride = 8
        FrameSize = 112
    },
    @{
        RunName = "video_r2plus1d18_gru_noalign_formal_e100_seed123_v1"
        Seed = 123
        Backbone = "video_r2plus1d_18_kinetics400"
        BatchSize = 16
        ClipLength = 8
        ClipStride = 8
        FrameSize = 112
    },
    @{
        RunName = "video_r2plus1d18_gru_noalign_formal_e100_seed3407_v1"
        Seed = 3407
        Backbone = "video_r2plus1d_18_kinetics400"
        BatchSize = 16
        ClipLength = 8
        ClipStride = 8
        FrameSize = 112
    }
)

foreach ($experiment in $experiments) {
    Invoke-Experiment @experiment
}
