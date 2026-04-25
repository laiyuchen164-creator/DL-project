# Audio-to-Video Affective Transfer for Low-Supervision Valence-Arousal Prediction

This is a code-only submission package for the CSC 8851 final project.

- Package owner: Yuchen Lai
- Email: ylai11@student.gsu.edu
- PantherID: 002954492
- Package note: Packaging emphasis: experiment scripts, documentation, and figure regeneration.

## Included content

- `README.md`
- `train.py`
- `models/`
- `datasets/`
- `scripts/`

## Excluded content

- datasets
- cached features
- checkpoints
- local virtual environments
- local tool installs
- report and LaTeX files

## Expected local data layout

```text
data/
  deam/
    DEAM_audio.zip
    annotations/
      annotations/
        annotations averaged per song/
          dynamic (per second annotations)/
            valence.csv
            arousal.csv
  veatic/
    VEATIC.zip
  cache/
```

## Environment setup

1. Create and activate a Python 3.11 virtual environment.
2. Install PyTorch, torchvision, and torchaudio for your platform from the official PyTorch instructions.
3. Install the remaining Python dependencies:

```powershell
pip install transformers librosa av numpy matplotlib scikit-learn opensmile PyMuPDF Pillow python-pptx reportlab
```

## Quick validation

```powershell
.\.venv\Scripts\python.exe .\scripts\smoke_test.py
```

## Main reproduction commands

25% supervision:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_proposal_formal_f025.ps1
```

10% supervision:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_proposal_formal_f010.ps1
```

This package is intended to be functionally aligned with the team repository while excluding non-code artifacts.
