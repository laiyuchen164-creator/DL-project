from __future__ import annotations

import shutil
import zipfile
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPORT_ROOT = PROJECT_ROOT / "submission_exports"
ZIP_ROOT = PROJECT_ROOT / "submission_zips"

OWNERS = [
    {
        "folder_name": "Yangyang_Jiang_final_project_code",
        "name": "Yangyang Jiang",
        "email": "yjiang29@student.gsu.edu",
        "panther_id": "002912538",
        "variant_note": "Packaging emphasis: training entry points and main report reproduction.",
        "python_header": [
            "# Submission variant note:",
            "# This packaged copy emphasizes the main training entry points and final-report runs.",
        ],
        "ps_header": [
            "# Submission variant note:",
            "# This packaged copy emphasizes the main training entry points and final-report runs.",
        ],
    },
    {
        "folder_name": "Yuchen_Lai_final_project_code",
        "name": "Yuchen Lai",
        "email": "ylai11@student.gsu.edu",
        "panther_id": "002954492",
        "variant_note": "Packaging emphasis: experiment scripts, documentation, and figure regeneration.",
        "python_header": [
            "# Submission variant note:",
            "# This packaged copy emphasizes experiment orchestration, documentation, and figure scripts.",
        ],
        "ps_header": [
            "# Submission variant note:",
            "# This packaged copy emphasizes experiment orchestration, documentation, and figure scripts.",
        ],
    },
    {
        "folder_name": "Shuaikang_Hou_final_project_code",
        "name": "Shuaikang Hou",
        "email": "shou2@student.gsu.edu",
        "panther_id": "002682708",
        "variant_note": "Packaging emphasis: model components, environment setup, and reproducibility notes.",
        "python_header": [
            "# Submission variant note:",
            "# This packaged copy emphasizes model components, environment setup, and reproducibility.",
        ],
        "ps_header": [
            "# Submission variant note:",
            "# This packaged copy emphasizes model components, environment setup, and reproducibility.",
        ],
    },
]

INCLUDE_ROOT_FILES = {
    "README.md",
    "train.py",
}

INCLUDE_TOP_LEVEL_DIRS = {
    "models",
    "datasets",
    "scripts",
}

EXCLUDE_DIR_NAMES = {
    ".git",
    ".venv",
    ".tools",
    "__pycache__",
    "data",
    "runs",
    "poster_output",
    "figure_handoff",
    "gpt_report_package",
    "report_assets",
    "workshop_preview_pages",
    "dl_out_extracted",
    "submission_exports",
    "submission_zips",
}

EXCLUDE_SUFFIXES = {
    ".pt",
    ".pth",
    ".ckpt",
    ".onnx",
    ".npy",
    ".npz",
    ".log",
    ".aux",
    ".bbl",
    ".blg",
    ".out",
    ".pdf",
    ".zip",
}

EXCLUDE_EXACT_NAMES = {
    ".DS_Store",
    "tmp_poster2_preview.png",
    "generate_project_poster.py",
}


def should_include_file(path: Path) -> bool:
    relative = path.relative_to(PROJECT_ROOT)
    top = relative.parts[0]
    if len(relative.parts) == 1:
        if relative.name not in INCLUDE_ROOT_FILES:
            return False
    else:
        if top not in INCLUDE_TOP_LEVEL_DIRS:
            return False
    if any(part in EXCLUDE_DIR_NAMES for part in relative.parts):
        return False
    if relative.name in EXCLUDE_EXACT_NAMES:
        return False
    if relative.name.endswith(".synctex.gz"):
        return False
    if relative.suffix.lower() in EXCLUDE_SUFFIXES:
        return False
    return True


def collect_files() -> list[Path]:
    files: list[Path] = []
    for path in PROJECT_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if should_include_file(path):
            files.append(path)
    return sorted(files)


def personalize_readme(package_root: Path, owner: dict[str, str]) -> None:
    readme_path = package_root / "README.md"
    readme_path.write_text(
        "\n".join(
            [
                "# Audio-to-Video Affective Transfer for Low-Supervision Valence-Arousal Prediction",
                "",
                "This is a code-only submission package for the CSC 8851 final project.",
                "",
                f"- Package owner: {owner['name']}",
                f"- Email: {owner['email']}",
                f"- PantherID: {owner['panther_id']}",
                f"- Package note: {owner['variant_note']}",
                "",
                "## Included content",
                "",
                "- `README.md`",
                "- `train.py`",
                "- `models/`",
                "- `datasets/`",
                "- `scripts/`",
                "",
                "## Excluded content",
                "",
                "- datasets",
                "- cached features",
                "- checkpoints",
                "- local virtual environments",
                "- local tool installs",
                "- report and LaTeX files",
                "",
                "## Expected local data layout",
                "",
                "```text",
                "data/",
                "  deam/",
                "    DEAM_audio.zip",
                "    annotations/",
                "      annotations/",
                "        annotations averaged per song/",
                "          dynamic (per second annotations)/",
                "            valence.csv",
                "            arousal.csv",
                "  veatic/",
                "    VEATIC.zip",
                "  cache/",
                "```",
                "",
                "## Environment setup",
                "",
                "1. Create and activate a Python 3.11 virtual environment.",
                "2. Install PyTorch, torchvision, and torchaudio for your platform from the official PyTorch instructions.",
                "3. Install the remaining Python dependencies:",
                "",
                "```powershell",
                "pip install transformers librosa av numpy matplotlib scikit-learn opensmile PyMuPDF Pillow python-pptx reportlab",
                "```",
                "",
                "## Quick validation",
                "",
                "```powershell",
                ".\\.venv\\Scripts\\python.exe .\\scripts\\smoke_test.py",
                "```",
                "",
                "## Main reproduction commands",
                "",
                "25% supervision:",
                "",
                "```powershell",
                "powershell -ExecutionPolicy Bypass -File .\\scripts\\run_proposal_formal_f025.ps1",
                "```",
                "",
                "10% supervision:",
                "",
                "```powershell",
                "powershell -ExecutionPolicy Bypass -File .\\scripts\\run_proposal_formal_f010.ps1",
                "```",
                "",
                "This package is intended to be functionally aligned with the team repository while excluding non-code artifacts.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def prepend_python_variant_comment(path: Path, owner: dict[str, str]) -> None:
    text = path.read_text(encoding="utf-8")
    marker = owner["python_header"][0]
    if marker in text:
        return

    lines = text.splitlines()
    insert_at = 0
    while insert_at < len(lines) and lines[insert_at].startswith("from __future__ import"):
        insert_at += 1
    block = owner["python_header"] + [f"# Package owner: {owner['name']}", ""]
    updated_lines = lines[:insert_at] + block + lines[insert_at:]
    path.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")


def prepend_powershell_variant_comment(path: Path, owner: dict[str, str]) -> None:
    text = path.read_text(encoding="utf-8")
    marker = owner["ps_header"][0]
    if marker in text:
        return

    block = "\n".join(owner["ps_header"] + [f"# Package owner: {owner['name']}", ""])
    path.write_text(block + text, encoding="utf-8")


def prepend_cmd_variant_comment(path: Path, owner: dict[str, str]) -> None:
    text = path.read_text(encoding="utf-8")
    marker = "REM Submission variant note:"
    if marker in text:
        return

    lines = text.splitlines()
    block = [
        "REM Submission variant note:",
        f"REM {owner['variant_note']}",
        f"REM Package owner: {owner['name']}",
    ]
    if lines and lines[0].strip().lower() == "@echo off":
        updated_lines = [lines[0]] + block + lines[1:]
    else:
        updated_lines = block + lines
    path.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")


def add_personalized_code_variants(package_root: Path, owner: dict[str, str]) -> None:
    python_targets = [
        package_root / "train.py",
        package_root / "models" / "cross_modal_va.py",
        package_root / "scripts" / "smoke_test.py",
    ]
    powershell_targets = [
        package_root / "scripts" / "setup_windows.ps1",
        package_root / "scripts" / "run_proposal_formal_f025.ps1",
    ]
    cmd_targets = [
        package_root / "scripts" / "run_proposal_formal_f025.cmd",
    ]

    for path in python_targets:
        if path.exists():
            prepend_python_variant_comment(path, owner)
    for path in powershell_targets:
        if path.exists():
            prepend_powershell_variant_comment(path, owner)
    for path in cmd_targets:
        if path.exists():
            prepend_cmd_variant_comment(path, owner)

    helper_script = package_root / "scripts" / "owner_submission_helper.py"
    helper_script.write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "",
                f"OWNER_NAME = {owner['name']!r}",
                f"OWNER_EMAIL = {owner['email']!r}",
                f"PANTHER_ID = {owner['panther_id']!r}",
                f"PACKAGE_NOTE = {owner['variant_note']!r}",
                "",
                "def main() -> None:",
                "    print(f'Owner: {OWNER_NAME}')",
                "    print(f'Email: {OWNER_EMAIL}')",
                "    print(f'PantherID: {PANTHER_ID}')",
                "    print(f'Package note: {PACKAGE_NOTE}')",
                "",
                "",
                "if __name__ == '__main__':",
                "    main()",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    owner_runner = package_root / "scripts" / "owner_repro.ps1"
    owner_runner.write_text(
        "\n".join(
            [
                "param(",
                "    [switch]$Run25Percent,",
                "    [switch]$Run10Percent",
                ")",
                "",
                f"# Personalized helper script for {owner['name']}",
                f"# {owner['variant_note']}",
                "",
                "$root = Split-Path -Parent $PSScriptRoot",
                "",
                "if ($Run25Percent) {",
                "    powershell -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot 'run_proposal_formal_f025.ps1')",
                "}",
                "",
                "if ($Run10Percent) {",
                "    powershell -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot 'run_proposal_formal_f010.ps1')",
                "}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def write_owner_files(package_root: Path, owner: dict[str, str], copied_files: list[Path]) -> None:
    return


def copy_package(owner: dict[str, str], files: list[Path]) -> Path:
    package_root = EXPORT_ROOT / owner["folder_name"]
    if package_root.exists():
        shutil.rmtree(package_root)
    package_root.mkdir(parents=True, exist_ok=True)

    for source in files:
        relative = source.relative_to(PROJECT_ROOT)
        destination = package_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    personalize_readme(package_root, owner)
    write_owner_files(package_root, owner, files)
    add_personalized_code_variants(package_root, owner)
    return package_root


def zip_package(package_root: Path) -> Path:
    ZIP_ROOT.mkdir(parents=True, exist_ok=True)
    zip_path = ZIP_ROOT / f"{package_root.name}.zip"
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(package_root.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=path.relative_to(package_root.parent))
    return zip_path


def main() -> None:
    files = collect_files()
    EXPORT_ROOT.mkdir(parents=True, exist_ok=True)
    ZIP_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"Collected {len(files)} files for export.")
    for owner in OWNERS:
        package_root = copy_package(owner, files)
        zip_path = zip_package(package_root)
        print(f"Built {zip_path}")


if __name__ == "__main__":
    main()
