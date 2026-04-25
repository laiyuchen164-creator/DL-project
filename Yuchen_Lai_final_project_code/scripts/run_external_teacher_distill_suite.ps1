param(
    [string]$TeacherCalibrationDir = "runs\audeering_teacher_calibration_v1",
    [string]$SmokeRunDir = "runs\gru_noalign_external_teacher_smoke_v1",
    [string]$FormalRunDir = "runs\gru_noalign_external_teacher_formal_e50_v1",
    [string]$BaselineRunDir = "runs\gru_noalign_bs16_nw4_formal_e50_v1",
    [string]$SummaryDir = "runs\experiment_summaries\external_teacher_distill_suite",
    [int]$CalibrationEpochs = 20,
    [int]$SmokeEpochs = 2,
    [int]$FormalEpochs = 50,
    [int]$BatchSize = 16,
    [int]$NumWorkers = 4,
    [int]$ClipLength = 8,
    [int]$ClipStride = 8,
    [int]$FrameSize = 112,
    [int]$AudioFrames = 256,
    [double]$LambdaTs = 1.0,
    [int]$Seed = 42
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$summaryDirAbs = Join-Path $projectRoot $SummaryDir
$teacherCalibrationAbs = Join-Path $projectRoot $TeacherCalibrationDir
$smokeRunAbs = Join-Path $projectRoot $SmokeRunDir
$formalRunAbs = Join-Path $projectRoot $FormalRunDir
New-Item -ItemType Directory -Force -Path $summaryDirAbs | Out-Null
New-Item -ItemType Directory -Force -Path $teacherCalibrationAbs | Out-Null
New-Item -ItemType Directory -Force -Path $smokeRunAbs | Out-Null
New-Item -ItemType Directory -Force -Path $formalRunAbs | Out-Null

function Invoke-LoggedPythonCommand {
    param(
        [string[]]$Arguments,
        [string]$StdoutPath,
        [string]$StderrPath,
        [string]$FailureMessage
    )

    $process = Start-Process `
        -FilePath $python `
        -ArgumentList $Arguments `
        -WorkingDirectory $projectRoot `
        -RedirectStandardOutput $StdoutPath `
        -RedirectStandardError $StderrPath `
        -PassThru `
        -Wait

    if ($process.ExitCode -ne 0) {
        throw $FailureMessage
    }
}

Push-Location $projectRoot
try {
    Invoke-LoggedPythonCommand `
        -Arguments @(
            "scripts\calibrate_external_teacher.py",
            "--device", "cuda",
            "--epochs", $CalibrationEpochs,
            "--batch-size", "8",
            "--seed", $Seed,
            "--output-dir", $TeacherCalibrationDir
        ) `
        -StdoutPath (Join-Path $teacherCalibrationAbs "stdout.log") `
        -StderrPath (Join-Path $teacherCalibrationAbs "stderr.log") `
        -FailureMessage "Teacher calibration failed."

    $teacherCheckpoint = Join-Path $teacherCalibrationAbs "best_checkpoint.pt"

    Invoke-LoggedPythonCommand `
        -Arguments @(
            "train.py",
            "--device", "cuda",
            "--epochs", $SmokeEpochs,
            "--method", "proposed",
            "--temporal-model", "gru",
            "--batch-size", $BatchSize,
            "--num-workers", $NumWorkers,
            "--clip-length", $ClipLength,
            "--clip-stride", $ClipStride,
            "--frame-size", $FrameSize,
            "--audio-frames", $AudioFrames,
            "--lambda-align", "0",
            "--lambda-ts", $LambdaTs,
            "--teacher-backend", "external_audeering_dim",
            "--teacher-checkpoint", $teacherCheckpoint,
            "--max-audio-steps", "10",
            "--max-joint-steps", "10",
            "--max-eval-steps", "10",
            "--output-dir", $SmokeRunDir
        ) `
        -StdoutPath (Join-Path $smokeRunAbs "stdout.log") `
        -StderrPath (Join-Path $smokeRunAbs "stderr.log") `
        -FailureMessage "Teacher distillation smoke run failed."

    Invoke-LoggedPythonCommand `
        -Arguments @(
            "train.py",
            "--device", "cuda",
            "--epochs", $FormalEpochs,
            "--method", "proposed",
            "--temporal-model", "gru",
            "--batch-size", $BatchSize,
            "--num-workers", $NumWorkers,
            "--clip-length", $ClipLength,
            "--clip-stride", $ClipStride,
            "--frame-size", $FrameSize,
            "--audio-frames", $AudioFrames,
            "--lambda-align", "0",
            "--lambda-ts", $LambdaTs,
            "--teacher-backend", "external_audeering_dim",
            "--teacher-checkpoint", $teacherCheckpoint,
            "--output-dir", $FormalRunDir
        ) `
        -StdoutPath (Join-Path $formalRunAbs "stdout.log") `
        -StderrPath (Join-Path $formalRunAbs "stderr.log") `
        -FailureMessage "Formal teacher distillation run failed."

    Invoke-LoggedPythonCommand `
        -Arguments @(
            "scripts\summarize_distill_vs_baseline.py",
            "--baseline-run", (Join-Path $projectRoot $BaselineRunDir),
            "--candidate-run", $formalRunAbs,
            "--output-dir", $summaryDirAbs
        ) `
        -StdoutPath (Join-Path $summaryDirAbs "summary_stdout.log") `
        -StderrPath (Join-Path $summaryDirAbs "summary_stderr.log") `
        -FailureMessage "Summary generation failed."
}
finally {
    Pop-Location
}
