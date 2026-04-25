param(
    [string]$PrimaryOutputDir = "runs\alt_noalign_bs16_nw4_cached_e5_v2",
    [string]$AlignmentOutputDir = "runs\gru_align_bs16_nw4_cached_e5_v1",
    [int]$Epochs = 5,
    [int]$BatchSize = 16,
    [int]$NumWorkers = 4,
    [int]$ClipLength = 8,
    [int]$ClipStride = 8,
    [int]$FrameSize = 112,
    [int]$AudioFrames = 256,
    [int]$LogEvery = 100,
    [int]$PollSeconds = 60,
    [double]$MeaningfulDelta = 0.02
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$summaryDir = Join-Path $projectRoot "runs\experiment_summaries"
$summaryCsv = Join-Path $summaryDir "gru_alignment_suite_metrics.csv"
$summaryJson = Join-Path $summaryDir "gru_alignment_suite_summary.json"
$summaryMd = Join-Path $summaryDir "gru_alignment_suite_summary.md"
$suiteLog = Join-Path $summaryDir "gru_alignment_suite.log"

if (-not (Test-Path $python)) {
    throw "Python not found at $python"
}

New-Item -ItemType Directory -Force -Path $summaryDir | Out-Null

function Write-SuiteLog {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$timestamp] $Message"
    Write-Host $line
    Add-Content -Path $suiteLog -Value $line
}

function Get-MetricsRecords {
    param([string]$RunDir)
    $metricsPath = Join-Path $RunDir "metrics.jsonl"
    if (-not (Test-Path $metricsPath)) {
        return @()
    }

    $lines = Get-Content -Path $metricsPath | Where-Object { $_.Trim() -ne "" }
    if (-not $lines) {
        return @()
    }

    return @($lines | ForEach-Object { $_ | ConvertFrom-Json })
}

function Test-RunComplete {
    param(
        [string]$RunDir,
        [int]$ExpectedEpochs
    )

    $records = Get-MetricsRecords -RunDir $RunDir
    if ($records.Count -lt $ExpectedEpochs) {
        return $false
    }

    for ($epoch = 1; $epoch -le $ExpectedEpochs; $epoch++) {
        $checkpoint = Join-Path $RunDir ("checkpoint_epoch_{0}.pt" -f $epoch)
        if (-not (Test-Path $checkpoint)) {
            return $false
        }
    }

    return $true
}

function Get-TrainProcess {
    param([string]$RunDir)
    $leaf = Split-Path $RunDir -Leaf
    return @(Get-CimInstance Win32_Process | Where-Object {
        $_.Name -eq "python.exe" -and
        $_.CommandLine -like "*train.py*" -and
        $_.CommandLine -like "*$leaf*"
    })
}

function Wait-RunCompletion {
    param(
        [string]$RunDir,
        [int]$ExpectedEpochs,
        [int]$SleepSeconds
    )

    while (-not (Test-RunComplete -RunDir $RunDir -ExpectedEpochs $ExpectedEpochs)) {
        $procs = Get-TrainProcess -RunDir $RunDir
        if (-not $procs) {
            throw "Run '$RunDir' is incomplete and no matching train.py process is active."
        }

        $records = Get-MetricsRecords -RunDir $RunDir
        $lastEpoch = if ($records.Count -gt 0) { $records[-1].epoch } else { 0 }
        Write-SuiteLog "Waiting on $RunDir (completed epochs: $lastEpoch/$ExpectedEpochs, active processes: $($procs.Count))"
        Start-Sleep -Seconds $SleepSeconds
    }

    Write-SuiteLog "Run completed: $RunDir"
}

function Start-AlignmentRun {
    param([string]$RunDir)

    $resolvedDir = Join-Path $projectRoot $RunDir
    New-Item -ItemType Directory -Force -Path $resolvedDir | Out-Null

    $stdoutLog = Join-Path $resolvedDir "stdout.log"
    $stderrLog = Join-Path $resolvedDir "stderr.log"

    $args = @(
        "-u",
        "train.py",
        "--device", "cuda",
        "--epochs", $Epochs,
        "--temporal-model", "gru",
        "--batch-size", $BatchSize,
        "--num-workers", $NumWorkers,
        "--clip-length", $ClipLength,
        "--clip-stride", $ClipStride,
        "--frame-size", $FrameSize,
        "--audio-frames", $AudioFrames,
        "--log-every", $LogEvery,
        "--lambda-align", "1",
        "--output-dir", $RunDir
    )

    Write-SuiteLog "Starting alignment control: $RunDir"
    $process = Start-Process -FilePath $python -ArgumentList $args -WorkingDirectory $projectRoot -RedirectStandardOutput $stdoutLog -RedirectStandardError $stderrLog -PassThru
    Write-SuiteLog "Alignment process started with PID $($process.Id)"
}

function Flatten-Metrics {
    param(
        [string]$RunName,
        [string]$RunDir
    )

    $records = Get-MetricsRecords -RunDir $RunDir
    return @($records | ForEach-Object {
        [PSCustomObject]@{
            run = $RunName
            run_dir = $RunDir
            epoch = [int]$_.epoch
            eval_visual_loss = [double]$_.eval.visual_loss
            eval_mae = [double]$_.eval.mae
            eval_rmse = [double]$_.eval.rmse
            eval_ccc_valence = [double]$_.eval.ccc_valence
            eval_ccc_arousal = [double]$_.eval.ccc_arousal
            eval_ccc_mean = [double]$_.eval.ccc_mean
        }
    })
}

function Get-BestEpochRecord {
    param([object[]]$Rows)
    return $Rows | Sort-Object -Property @{ Expression = "eval_ccc_mean"; Descending = $true }, @{ Expression = "epoch"; Descending = $false } | Select-Object -First 1
}

Push-Location $projectRoot
try {
    Write-SuiteLog "Suite started. Primary run: $PrimaryOutputDir"
    Wait-RunCompletion -RunDir $PrimaryOutputDir -ExpectedEpochs $Epochs -SleepSeconds $PollSeconds

    if (-not (Test-RunComplete -RunDir $AlignmentOutputDir -ExpectedEpochs $Epochs)) {
        $alignmentProcs = Get-TrainProcess -RunDir $AlignmentOutputDir
        if (-not $alignmentProcs) {
            Start-AlignmentRun -RunDir $AlignmentOutputDir
        }
        else {
            Write-SuiteLog "Alignment run already active: $AlignmentOutputDir"
        }
    }
    else {
        Write-SuiteLog "Alignment run already complete: $AlignmentOutputDir"
    }

    Wait-RunCompletion -RunDir $AlignmentOutputDir -ExpectedEpochs $Epochs -SleepSeconds $PollSeconds

    $primaryRows = Flatten-Metrics -RunName "gru_no_alignment" -RunDir $PrimaryOutputDir
    $alignmentRows = Flatten-Metrics -RunName "gru_alignment" -RunDir $AlignmentOutputDir
    $allRows = @($primaryRows + $alignmentRows) | Sort-Object run, epoch

    if ($allRows.Count -eq 0) {
        throw "No metrics were collected for summary generation."
    }

    $allRows | Export-Csv -Path $summaryCsv -NoTypeInformation -Encoding UTF8
    $allRows | ConvertTo-Json -Depth 4 | Set-Content -Path $summaryJson -Encoding UTF8

    $primaryBest = Get-BestEpochRecord -Rows $primaryRows
    $alignmentBest = Get-BestEpochRecord -Rows $alignmentRows
    $delta = [double]$alignmentBest.eval_ccc_mean - [double]$primaryBest.eval_ccc_mean

    if ([Math]::Abs($delta) -lt $MeaningfulDelta) {
        $comparisonNote = "Difference in best eval.ccc_mean is below the configured threshold ($MeaningfulDelta). Do not claim a clear winner."
    }
    elseif ($delta -gt 0) {
        $comparisonNote = "Alignment run has the higher best eval.ccc_mean."
    }
    else {
        $comparisonNote = "No-alignment run has the higher best eval.ccc_mean."
    }

    $tableHeader = "| run | epoch | eval_visual_loss | eval_mae | eval_rmse | eval_ccc_valence | eval_ccc_arousal | eval_ccc_mean |"
    $tableDivider = "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    $tableRows = $allRows | ForEach-Object {
        "| {0} | {1} | {2:N4} | {3:N4} | {4:N4} | {5:N4} | {6:N4} | {7:N4} |" -f $_.run, $_.epoch, $_.eval_visual_loss, $_.eval_mae, $_.eval_rmse, $_.eval_ccc_valence, $_.eval_ccc_arousal, $_.eval_ccc_mean
    }

    $summaryLines = @(
        "# GRU Experiment Summary",
        "",
        "Primary run: $PrimaryOutputDir",
        "Alignment run: $AlignmentOutputDir",
        "",
        "## Best epochs",
        "",
        "- gru_no_alignment: epoch $($primaryBest.epoch), eval.ccc_mean=$('{0:N4}' -f $primaryBest.eval_ccc_mean), eval.mae=$('{0:N4}' -f $primaryBest.eval_mae), eval.rmse=$('{0:N4}' -f $primaryBest.eval_rmse)",
        "- gru_alignment: epoch $($alignmentBest.epoch), eval.ccc_mean=$('{0:N4}' -f $alignmentBest.eval_ccc_mean), eval.mae=$('{0:N4}' -f $alignmentBest.eval_mae), eval.rmse=$('{0:N4}' -f $alignmentBest.eval_rmse)",
        "",
        "## Comparison",
        "",
        "- Best-epoch delta (alignment - no_alignment) on eval.ccc_mean: $('{0:N4}' -f $delta)",
        "- $comparisonNote",
        "",
        "## Per-epoch metrics",
        "",
        $tableHeader,
        $tableDivider
    ) + $tableRows

    Set-Content -Path $summaryMd -Value $summaryLines -Encoding UTF8
    Write-SuiteLog "Summary written to $summaryMd"
    Write-SuiteLog "Suite finished successfully."
}
finally {
    Pop-Location
}
