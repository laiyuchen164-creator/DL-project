@echo off
REM Submission variant note:
REM Packaging emphasis: experiment scripts, documentation, and figure regeneration.
REM Package owner: Yuchen Lai
setlocal

for %%I in ("%~dp0..") do set "ROOT=%%~fI"
set "RUNNER=%ROOT%\scripts\run_proposal_formal_f025.ps1"
set "SUMMARY_DIR=%ROOT%\runs\experiment_summaries"

if not exist "%SUMMARY_DIR%" mkdir "%SUMMARY_DIR%"
cd /d "%ROOT%"

powershell -ExecutionPolicy Bypass -File "%RUNNER%" ^
  1>>"%SUMMARY_DIR%\proposal_formal_f025_stdout.log" ^
  2>>"%SUMMARY_DIR%\proposal_formal_f025_stderr.log"

endlocal
