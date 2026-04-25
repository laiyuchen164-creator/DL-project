from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict

import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import DEAMDataset
from models import (
    DEFAULT_AUDEERING_DIM_REPO,
    DEFAULT_TEACHER_SAMPLE_RATE,
    ExternalAudeeringDimTeacher,
    TeacherCalibrationHead,
    concordance_correlation_coefficient,
    regression_loss,
)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate an external speech emotion teacher on DEAM.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--audio-zip", default="data/deam/DEAM_audio.zip")
    parser.add_argument("--audio-annotations", default="data/deam/annotations/annotations")
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--teacher-repo", default=DEFAULT_AUDEERING_DIM_REPO)
    parser.add_argument("--teacher-audio-seconds", type=float, default=4.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--max-eval-steps", type=int, default=None)
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = DEAMDataset(
        audio_zip_path=args.audio_zip,
        annotations_dir=args.audio_annotations,
        cache_dir=Path(args.cache_dir) / "deam_teacher_calibration",
        transform=None,
        target_length_seconds=args.teacher_audio_seconds,
        sample_rate=DEFAULT_TEACHER_SAMPLE_RATE,
    )
    train_indices, val_indices = split_indices(dataset, seed=args.seed)
    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_raw_audio_batch,
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_raw_audio_batch,
    )

    teacher = ExternalAudeeringDimTeacher(repo_id=args.teacher_repo).to(device)
    calibration = TeacherCalibrationHead().to(device)
    optimizer = AdamW(calibration.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    baseline_metrics = evaluate_mapping(
        teacher=teacher,
        calibration=TeacherCalibrationHead().to(device),
        loader=val_loader,
        device=device,
        max_steps=args.max_eval_steps,
    )

    best_record: Dict[str, object] | None = None
    best_ccc = float("-inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(
            teacher=teacher,
            calibration=calibration,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            max_steps=args.max_train_steps,
        )
        eval_metrics = evaluate_mapping(
            teacher=teacher,
            calibration=calibration,
            loader=val_loader,
            device=device,
            max_steps=args.max_eval_steps,
        )
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "eval": eval_metrics,
        }
        history.append(record)
        print(json.dumps(record))

        if eval_metrics["ccc_mean"] > best_ccc:
            best_ccc = eval_metrics["ccc_mean"]
            best_record = record
            save_calibration_checkpoint(
                output_dir / "best_checkpoint.pt",
                calibration=calibration,
                args=args,
                metrics=record,
            )

    summary = {
        "teacher_repo": args.teacher_repo,
        "teacher_sample_rate": DEFAULT_TEACHER_SAMPLE_RATE,
        "teacher_audio_seconds": args.teacher_audio_seconds,
        "train_size": len(train_indices),
        "val_size": len(val_indices),
        "baseline_eval": baseline_metrics,
        "best_record": best_record,
        "history": history,
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def split_indices(dataset: DEAMDataset, seed: int) -> tuple[list[int], list[int]]:
    indices = list(range(len(dataset)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    split = max(1, int(len(indices) * 0.8))
    return indices[:split], indices[split:]


def collate_raw_audio_batch(batch: list[Dict[str, object]]) -> Dict[str, Tensor]:
    waveforms = []
    targets = []
    song_ids = []
    for sample in batch:
        waveform = sample["audio"]
        if not isinstance(waveform, torch.Tensor):
            raise TypeError("Expected raw audio tensor from DEAMDataset.")
        waveform = waveform.float()
        if waveform.ndim == 2:
            waveform = waveform.mean(dim=0)
        else:
            waveform = waveform.flatten()
        waveforms.append(waveform)
        targets.append(sample["target"])
        song_ids.append(sample["song_id"])

    return {
        "teacher_audio": torch.stack(waveforms, dim=0),
        "target": torch.stack(targets, dim=0),
        "song_id": song_ids,
    }


def train_epoch(
    teacher: ExternalAudeeringDimTeacher,
    calibration: nn.Module,
    loader: DataLoader,
    optimizer: AdamW,
    device: torch.device,
    max_steps: int | None,
) -> Dict[str, float]:
    calibration.train()
    meter = MetricAccumulator()

    for step, batch in enumerate(loader, start=1):
        teacher_audio = batch["teacher_audio"].to(device)
        targets = batch["target"].to(device)

        with torch.no_grad():
            teacher_dims = teacher.predict_raw_dimensions(teacher_audio)

        predictions = calibration(teacher_dims)
        loss = regression_loss(targets, predictions)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        meter.update(loss=loss.item(), predictions=predictions.detach(), targets=targets.detach())
        if max_steps is not None and step >= max_steps:
            break

    return meter.compute()


@torch.no_grad()
def evaluate_mapping(
    teacher: ExternalAudeeringDimTeacher,
    calibration: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_steps: int | None,
) -> Dict[str, float]:
    calibration.eval()
    meter = MetricAccumulator()

    for step, batch in enumerate(loader, start=1):
        teacher_audio = batch["teacher_audio"].to(device)
        targets = batch["target"].to(device)

        teacher_dims = teacher.predict_raw_dimensions(teacher_audio)
        predictions = calibration(teacher_dims)
        loss = regression_loss(targets, predictions)
        meter.update(loss=loss.item(), predictions=predictions, targets=targets)
        if max_steps is not None and step >= max_steps:
            break

    return meter.compute()


class MetricAccumulator:
    def __init__(self) -> None:
        self.losses: list[float] = []
        self.predictions: list[Tensor] = []
        self.targets: list[Tensor] = []

    def update(self, loss: float, predictions: Tensor, targets: Tensor) -> None:
        self.losses.append(float(loss))
        self.predictions.append(predictions.detach().cpu())
        self.targets.append(targets.detach().cpu())

    def compute(self) -> Dict[str, float]:
        predictions = torch.cat(self.predictions, dim=0)
        targets = torch.cat(self.targets, dim=0)
        mae = (predictions - targets).abs().mean()
        rmse = torch.sqrt((predictions - targets).pow(2).mean())
        ccc = concordance_correlation_coefficient(targets, predictions)
        return {
            "loss": sum(self.losses) / len(self.losses),
            "mae": float(mae),
            "rmse": float(rmse),
            "ccc_valence": float(ccc[0]),
            "ccc_arousal": float(ccc[1]),
            "ccc_mean": float(ccc.mean()),
        }


def save_calibration_checkpoint(
    path: Path,
    calibration: nn.Module,
    args: argparse.Namespace,
    metrics: Dict[str, object],
) -> None:
    torch.save(
        {
            "repo_id": args.teacher_repo,
            "teacher_sample_rate": DEFAULT_TEACHER_SAMPLE_RATE,
            "teacher_audio_seconds": args.teacher_audio_seconds,
            "calibration_state_dict": calibration.state_dict(),
            "args": vars(args),
            "metrics": metrics,
        },
        path,
    )


if __name__ == "__main__":
    main()
