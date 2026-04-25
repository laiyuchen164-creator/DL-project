param(
    [string]$TransformerOutputDir = "runs\transformer_noalign_bs16_nw4_formal_e50_v1",
    [string]$GruOutputDir = "runs\gru_noalign_bs16_nw4_formal_e50_v1",
    [int]$Epochs = 50,
    [int]$ExtendedEpochs = 100,
    [int]$BatchSize = 16,
    [int]$NumWorkers = 4,
    [int]$ClipLength = 8,
    [int]$ClipStride = 8,
    [int]$FrameSize = 112,
    [int]$AudioFrames = 256,
    [int]$LogEvery = 100,
    [int]$Seed = 42,
    [int]$PollSeconds = 60,
    [double]$MeaningfulDelta = 0.02,
    [double]$MinImprovementForExtension = 0.01,
    [double]$MaxMetricRegression = 0.005
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$summaryDir = Join-Path $projectRoot "runs\experiment_summaries"
$summaryCsv = Join-Path $summaryDir "temporal_model_suite_metrics.csv"
$summaryJson = Join-Path $summaryDir "temporal_model_suite_summary.json"
$summaryMd = Join-Path $summaryDir "temporal_model_suite_summary.md"
$suiteLog = Join-Path $summaryDir "temporal_model_suite.log"

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
    $metricsPath = Join-Path (Join-Path $projectRoot $RunDir) "metrics.jsonl"
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

    $epochs = @($records | ForEach-Object { [int]$_.epoch } | Sort-Object)
    if ($epochs.Count -ne $ExpectedEpochs) {
        return $false
    }

    for ($epoch = 1; $epoch -le $ExpectedEpochs; $epoch++) {
        if ($epochs[$epoch - 1] -ne $epoch) {
            return $false
        }
        $checkpoint = Join-Path (Join-Path $projectRoot $RunDir) ("checkpoint_epoch_{0}.pt" -f $epoch)
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
        $procs = @(Get-TrainProcess -RunDir $RunDir)
        if (-not $procs) {
            throw "Run '$RunDir' is incomplete and no matching train.py process is active."
        }

        $records = Get-MetricsRecords -RunDir $RunDir
        $lastEpoch = if ($records.Count -gt 0) { [int]($records | Sort-Object epoch | Select-Object -Last 1).epoch } else { 0 }
        Write-SuiteLog "Waiting on $RunDir (completed epochs: $lastEpoch/$ExpectedEpochs, active processes: $($procs.Count))"
        Start-Sleep -Seconds $SleepSeconds
    }

    Write-SuiteLog "Run completed: $RunDir"
}

function Start-Run {
    param(
        [string]$RunDir,
        [string]$TemporalModel,
        [int]$TargetEpochs,
        [string]$ResumeFrom = $null
    )

    $resolvedDir = Join-Path $projectRoot $RunDir
    New-Item -ItemType Directory -Force -Path $resolvedDir | Out-Null

    $stdoutLog = Join-Path $resolvedDir "stdout.log"
    $stderrLog = Join-Path $resolvedDir "stderr.log"

    $args = @(
        "-u",
        "train.py",
        "--device", "cuda",
        "--epochs", $TargetEpochs,
        "--method", "proposed",
        "--temporal-model", $TemporalModel,
        "--batch-size", $BatchSize,
        "--num-workers", $NumWorkers,
        "--clip-length", $ClipLength,
        "--clip-stride", $ClipStride,
        "--frame-size", $FrameSize,
        "--audio-frames", $AudioFrames,
        "--log-every", $LogEvery,
        "--lambda-align", "0",
        "--seed", $Seed,
        "--output-dir", $RunDir
    )

    if ($ResumeFrom) {
        $args += @("--resume-from", $ResumeFrom)
        Write-SuiteLog "Resuming $TemporalModel run to epoch ${TargetEpochs}: $RunDir"
    }
    else {
        Write-SuiteLog "Starting $TemporalModel run to epoch ${TargetEpochs}: $RunDir"
    }

    $process = Start-Process -FilePath $python -ArgumentList $args -WorkingDirectory $projectRoot -RedirectStandardOutput $stdoutLog -RedirectStandardError $stderrLog -PassThru
    Write-SuiteLog "$TemporalModel process started with PID $($process.Id)"
}

function Flatten-Metrics {
    param(
        [string]$RunName,
        [string]$RunDir
    )

    $records = Get-MetricsRecords -RunDir $RunDir
    return @($records | Sort-Object epoch | ForEach-Object {
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

function Get-WindowBest {
    param(
        [object[]]$Rows,
        [int]$StartEpoch,
        [int]$EndEpoch
    )

    $windowRows = @($Rows | Where-Object { $_.epoch -ge $StartEpoch -and $_.epoch -le $EndEpoch })
    if (-not $windowRows) {
        throw "Missing rows for window $StartEpoch-$EndEpoch."
    }
    return ($windowRows | Measure-Object -Property eval_ccc_mean -Maximum).Maximum
}

function Get-WindowSummary {
    param(
        [object[]]$Rows,
        [int]$StartEpoch,
        [int]$EndEpoch
    )

    $windowRows = @($Rows | Where-Object { $_.epoch -ge $StartEpoch -and $_.epoch -le $EndEpoch })
    if (-not $windowRows) {
        throw "Missing rows for window $StartEpoch-$EndEpoch."
    }

    return [PSCustomObject]@{
        best_ccc_mean = ($windowRows | Measure-Object -Property eval_ccc_mean -Maximum).Maximum
        best_mae = ($windowRows | Measure-Object -Property eval_mae -Minimum).Minimum
        best_rmse = ($windowRows | Measure-Object -Property eval_rmse -Minimum).Minimum
    }
}

function Test-ShouldExtend {
    param([object[]]$Rows)

    $window31to40 = Get-WindowSummary -Rows $Rows -StartEpoch 31 -EndEpoch 40
    $last10 = Get-WindowSummary -Rows $Rows -StartEpoch 41 -EndEpoch 50
    $improvement = [double]$last10.best_ccc_mean - [double]$window31to40.best_ccc_mean
    $mae_ok = ([double]$last10.best_mae -le ([double]$window31to40.best_mae + $MaxMetricRegression))
    $rmse_ok = ([double]$last10.best_rmse -le ([double]$window31to40.best_rmse + $MaxMetricRegression))

    return [PSCustomObject]@{
        should_extend = (($improvement -ge $MinImprovementForExtension) -and $mae_ok -and $rmse_ok)
        improvement = $improvement
        best_31_40_ccc_mean = [double]$window31to40.best_ccc_mean
        best_31_40_mae = [double]$window31to40.best_mae
        best_31_40_rmse = [double]$window31to40.best_rmse
        best_41_50_ccc_mean = [double]$last10.best_ccc_mean
        best_41_50_mae = [double]$last10.best_mae
        best_41_50_rmse = [double]$last10.best_rmse
        mae_ok = $mae_ok
        rmse_ok = $rmse_ok
    }
}

Push-Location $projectRoot
try {
    Write-SuiteLog "Temporal suite started. Formal transformer run: $TransformerOutputDir"

    if (-not (Test-RunComplete -RunDir $TransformerOutputDir -ExpectedEpochs $Epochs)) {
        $transformerProcs = @(Get-TrainProcess -RunDir $TransformerOutputDir)
        if (-not $transformerProcs) {
            Start-Run -RunDir $TransformerOutputDir -TemporalModel "transformer" -TargetEpochs $Epochs
        }
        else {
            Write-SuiteLog "Transformer run already active: $TransformerOutputDir"
        }
    }
    else {
        Write-SuiteLog "Transformer run already complete: $TransformerOutputDir"
    }

    Wait-RunCompletion -RunDir $TransformerOutputDir -ExpectedEpochs $Epochs -SleepSeconds $PollSeconds

    if (-not (Test-RunComplete -RunDir $GruOutputDir -ExpectedEpochs $Epochs)) {
        $gruProcs = @(Get-TrainProcess -RunDir $GruOutputDir)
        if (-not $gruProcs) {
            Start-Run -RunDir $GruOutputDir -TemporalModel "gru" -TargetEpochs $Epochs
        }
        else {
            Write-SuiteLog "GRU run already active: $GruOutputDir"
        }
    }
    else {
        Write-SuiteLog "GRU run already complete: $GruOutputDir"
    }

    Wait-RunCompletion -RunDir $GruOutputDir -ExpectedEpochs $Epochs -SleepSeconds $PollSeconds

    $transformerRows = Flatten-Metrics -RunName "transformer_no_alignment" -RunDir $TransformerOutputDir
    $gruRows = Flatten-Metrics -RunName "gru_no_alignment" -RunDir $GruOutputDir
    $transformerBest = Get-BestEpochRecord -Rows $transformerRows
    $gruBest = Get-BestEpochRecord -Rows $gruRows
    $bestDelta = [double]$transformerBest.eval_ccc_mean - [double]$gruBest.eval_ccc_mean

    $leaderName = if ($bestDelta -gt 0) { "transformer" } else { "gru" }
    $leaderRows = if ($leaderName -eq "transformer") { $transformerRows } else { $gruRows }
    $leaderRunDir = if ($leaderName -eq "transformer") { $TransformerOutputDir } else { $GruOutputDir }
    $leaderTemporalModel = $leaderName
    $extensionDecision = Test-ShouldExtend -Rows $leaderRows

    if ($extensionDecision.should_extend) {
        if (-not (Test-RunComplete -RunDir $leaderRunDir -ExpectedEpochs $ExtendedEpochs)) {
            $leaderProcs = @(Get-TrainProcess -RunDir $leaderRunDir)
            if (-not $leaderProcs) {
                $resumeFrom = Join-Path (Join-Path $projectRoot $leaderRunDir) "checkpoint_epoch_50.pt"
                Start-Run -RunDir $leaderRunDir -TemporalModel $leaderTemporalModel -TargetEpochs $ExtendedEpochs -ResumeFrom $resumeFrom
            }
            else {
                Write-SuiteLog "Leader run already active for extension: $leaderRunDir"
            }
        }
        else {
            Write-SuiteLog "Leader extension already complete: $leaderRunDir"
        }

        Wait-RunCompletion -RunDir $leaderRunDir -ExpectedEpochs $ExtendedEpochs -SleepSeconds $PollSeconds
    }
    else {
        Write-SuiteLog "Extension rule not met. Stopping formal comparison at 50 epochs."
    }

    $transformerRows = Flatten-Metrics -RunName "transformer_no_alignment" -RunDir $TransformerOutputDir
    $gruRows = Flatten-Metrics -RunName "gru_no_alignment" -RunDir $GruOutputDir
    $allRows = @($gruRows + $transformerRows) | Sort-Object run, epoch

    if ($allRows.Count -eq 0) {
        throw "No metrics were collected for summary generation."
    }

    $allRows | Export-Csv -Path $summaryCsv -NoTypeInformation -Encoding UTF8
    $allRows | ConvertTo-Json -Depth 4 | Set-Content -Path $summaryJson -Encoding UTF8

    $transformerBest = Get-BestEpochRecord -Rows $transformerRows
    $gruBest = Get-BestEpochRecord -Rows $gruRows
    $bestDelta = [double]$transformerBest.eval_ccc_mean - [double]$gruBest.eval_ccc_mean

    if ([Math]::Abs($bestDelta) -lt $MeaningfulDelta) {
        $comparisonNote = "Difference in best eval.ccc_mean is below the configured threshold ($MeaningfulDelta). Do not claim a clear winner."
    }
    elseif ($bestDelta -gt 0) {
        $comparisonNote = "Transformer run has the higher best eval.ccc_mean."
    }
    else {
        $comparisonNote = "GRU run has the higher best eval.ccc_mean."
    }

    $extensionNote = if ($extensionDecision.should_extend) {
        "Extended $leaderName to $ExtendedEpochs epochs because best eval.ccc_mean improved by $('{0:N4}' -f $extensionDecision.improvement) from epochs 31-40 to 41-50."
    }
    else {
        "Did not extend to $ExtendedEpochs epochs because best eval.ccc_mean improved by only $('{0:N4}' -f $extensionDecision.improvement) from epochs 31-40 to 41-50."
    }

    $tableHeader = "| run | epoch | eval_visual_loss | eval_mae | eval_rmse | eval_ccc_valence | eval_ccc_arousal | eval_ccc_mean |"
    $tableDivider = "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    $tableRows = $allRows | ForEach-Object {
        "| {0} | {1} | {2:N4} | {3:N4} | {4:N4} | {5:N4} | {6:N4} | {7:N4} |" -f $_.run, $_.epoch, $_.eval_visual_loss, $_.eval_mae, $_.eval_rmse, $_.eval_ccc_valence, $_.eval_ccc_arousal, $_.eval_ccc_mean
    }

    $summaryLines = @(
        "# Temporal Model Experiment Summary",
        "",
        "Transformer run: $TransformerOutputDir",
        "GRU run: $GruOutputDir",
        "",
        "## Best epochs",
        "",
        "- transformer_no_alignment: epoch $($transformerBest.epoch), eval.ccc_mean=$('{0:N4}' -f $transformerBest.eval_ccc_mean), eval.mae=$('{0:N4}' -f $transformerBest.eval_mae), eval.rmse=$('{0:N4}' -f $transformerBest.eval_rmse)",
        "- gru_no_alignment: epoch $($gruBest.epoch), eval.ccc_mean=$('{0:N4}' -f $gruBest.eval_ccc_mean), eval.mae=$('{0:N4}' -f $gruBest.eval_mae), eval.rmse=$('{0:N4}' -f $gruBest.eval_rmse)",
        "",
        "## Comparison",
        "",
        "- Best-epoch delta (transformer - gru) on eval.ccc_mean: $('{0:N4}' -f $bestDelta)",
        "- $comparisonNote",
        "",
        "## Extension Decision",
        "",
        "- Leader after 50 epochs: $leaderName",
        "- Best eval.ccc_mean in epochs 31-40: $('{0:N4}' -f $extensionDecision.best_31_40_ccc_mean)",
        "- Best eval.mae in epochs 31-40: $('{0:N4}' -f $extensionDecision.best_31_40_mae)",
        "- Best eval.rmse in epochs 31-40: $('{0:N4}' -f $extensionDecision.best_31_40_rmse)",
        "- Best eval.ccc_mean in epochs 41-50: $('{0:N4}' -f $extensionDecision.best_41_50_ccc_mean)",
        "- Best eval.mae in epochs 41-50: $('{0:N4}' -f $extensionDecision.best_41_50_mae)",
        "- Best eval.rmse in epochs 41-50: $('{0:N4}' -f $extensionDecision.best_41_50_rmse)",
        "- MAE extension check passed: $($extensionDecision.mae_ok)",
        "- RMSE extension check passed: $($extensionDecision.rmse_ok)",
        "- $extensionNote",
        "",
        "## Per-epoch metrics",
        "",
        $tableHeader,
        $tableDivider
    ) + $tableRows

    Set-Content -Path $summaryMd -Value $summaryLines -Encoding UTF8
    Write-SuiteLog "Summary written to $summaryMd"
    Write-SuiteLog "Temporal suite finished successfully."
}
finally {
    Pop-Location
}
