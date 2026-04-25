@echo off
setlocal

for %%I in ("%~dp0..") do set "ROOT=%%~fI"
set "RUN_DIR=%ROOT%\runs\video_mvit_v2_s_gru_noalign_formal_e100_v1"
set "PYTHON=%ROOT%\.venv\Scripts\python.exe"
set "TRAIN=%ROOT%\train.py"

if not exist "%RUN_DIR%" mkdir "%RUN_DIR%"
cd /d "%ROOT%"

"%PYTHON%" "%TRAIN%" ^
  --device cuda ^
  --epochs 100 ^
  --method proposed ^
  --temporal-model gru ^
  --batch-size 4 ^
  --num-workers 4 ^
  --lr 5e-5 ^
  --clip-length 16 ^
  --clip-stride 16 ^
  --frame-size 224 ^
  --audio-frames 256 ^
  --lambda-align 0 ^
  --visual-backbone video_mvit_v2_s_kinetics400 ^
  --output-dir "%RUN_DIR%" ^
  1>"%RUN_DIR%\stdout.log" ^
  2>"%RUN_DIR%\stderr.log"

endlocal
