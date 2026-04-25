@echo off
setlocal

for %%I in ("%~dp0..") do set "ROOT=%%~fI"
set "RUNNER=%ROOT%\scripts\run_proposal_formal_f010.ps1"
set "SUMMARY_DIR=%ROOT%\runs\experiment_summaries"

if not exist "%SUMMARY_DIR%" mkdir "%SUMMARY_DIR%"
cd /d "%ROOT%"

powershell -ExecutionPolicy Bypass -File "%RUNNER%" ^
  1>>"%SUMMARY_DIR%\proposal_formal_f010_stdout.log" ^
  2>>"%SUMMARY_DIR%\proposal_formal_f010_stderr.log"

endlocal
