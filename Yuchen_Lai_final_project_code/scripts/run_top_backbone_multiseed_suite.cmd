@echo off
setlocal

for %%I in ("%~dp0..") do set "ROOT=%%~fI"
set "PYTHON=%ROOT%\.venv\Scripts\python.exe"
set "RUNNER=%ROOT%\scripts\run_top_backbone_multiseed_suite.ps1"
set "SUMMARY_DIR=%ROOT%\runs\experiment_summaries"

if not exist "%SUMMARY_DIR%" mkdir "%SUMMARY_DIR%"
cd /d "%ROOT%"

powershell -ExecutionPolicy Bypass -File "%RUNNER%" ^
  1>>"%SUMMARY_DIR%\top_backbone_multiseed_suite_stdout.log" ^
  2>>"%SUMMARY_DIR%\top_backbone_multiseed_suite_stderr.log"

endlocal
