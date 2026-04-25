@echo off
setlocal

for %%I in ("%~dp0..") do set "ROOT=%%~fI"
set "RUN_DIR=%ROOT%\runs\video_r2plus1d18_gru_noalign_formal_e100_v1"
set "PYTHON=%ROOT%\.venv\Scripts\python.exe"
set "TRAIN=%ROOT%\train.py"

if not exist "%RUN_DIR%" mkdir "%RUN_DIR%"
cd /d "%ROOT%"

"%PYTHON%" "%TRAIN%" ^
  --device cuda ^
  --epochs 100 ^
  --method proposed ^
  --temporal-model gru ^
  --batch-size 16 ^
  --num-workers 4 ^
  --clip-length 8 ^
  --clip-stride 8 ^
  --frame-size 112 ^
  --audio-frames 256 ^
  --lambda-align 0 ^
  --visual-backbone video_r2plus1d_18_kinetics400 ^
  --output-dir "%RUN_DIR%" ^
  1>"%RUN_DIR%\stdout.log" ^
  2>"%RUN_DIR%\stderr.log"

endlocal
