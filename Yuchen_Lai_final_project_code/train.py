from __future__ import annotations
# Submission variant note:
# This packaged copy emphasizes experiment orchestration, documentation, and figure scripts.
# Package owner: Yuchen Lai


import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from datasets import AudioFeatureTransform, DEAMDataset, VEATICDataset, VideoFrameTransform
from models import (
    CrossModalVAConfig,
    CrossModalVAModel,
    DomainAdversarialVAModel,
    LateFusionVAModel,
    LossWeights,
    MViTV2SVideoBackbone,
    R2Plus1D18VideoBackbone,
    ResNet18FrameCNN,
    VideoMAEBaseVideoBackbone,
    build_external_teacher,
    extract_teacher_metadata,
    concordance_correlation_coefficient,
    cross_modal_training_loss,
    domain_adversarial_loss,
    regression_loss,
)


METHOD_CHOICES = ["proposed", "visual_only", "late_fusion", "dann"]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the cross-modal valence-arousal model.")
    parser.add_argument("--method", choices=METHOD_CHOICES, default="proposed")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--clip-length", type=int, default=32)
    parser.add_argument("--clip-stride", type=int, default=16)
    parser.add_argument("--frame-size", type=int, default=224)
    parser.add_argument("--audio-frames", type=int, default=512)
    parser.add_argument("--audio-zip", default="data/deam/DEAM_audio.zip")
    parser.add_argument("--audio-annotations", default="data/deam/annotations/annotations")
    parser.add_argument("--veatic-zip", default="data/veatic/VEATIC.zip")
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--temporal-model", choices=["gru", "transformer"], default="gru")
    parser.add_argument("--loss-alpha", type=float, default=0.5)
    parser.add_argument("--lambda-align", type=float, default=1.0)
    parser.add_argument("--lambda-ts", type=float, default=1.0)
    parser.add_argument("--lambda-visual", type=float, default=1.0)
    parser.add_argument("--lambda-domain", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--teacher-backend", choices=["external_audeering_dim"], default=None)
    parser.add_argument("--teacher-checkpoint", default=None)
    parser.add_argument(
        "--visual-backbone",
        choices=[
            "simple_cnn",
            "resnet18_imagenet",
            "video_r2plus1d_18_kinetics400",
            "video_videomae_base_k400",
            "video_mvit_v2_s_kinetics400",
        ],
        default="simple_cnn",
    )
    parser.add_argument("--freeze-visual-backbone", action="store_true")
    parser.add_argument("--videomae-trainable-blocks", type=int, default=0)
    parser.add_argument("--joint-fusion", choices=["none", "tagf_lite"], default="none")
    parser.add_argument("--late-fusion-mode", choices=["concat", "tagf_lite"], default="concat")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--visual-train-fraction", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--max-audio-steps", type=int, default=None)
    parser.add_argument("--max-joint-steps", type=int, default=None)
    parser.add_argument("--max-eval-steps", type=int, default=None)
    parser.add_argument("--output-dir", default="runs/default")
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = None
    start_epoch = 1
    if args.resume_from is not None:
        checkpoint = load_checkpoint(Path(args.resume_from), device)
        validate_resume_args(args, checkpoint)
        start_epoch = int(checkpoint["epoch"]) + 1
        if start_epoch > args.epochs:
            raise ValueError(
                f"Resume checkpoint epoch {checkpoint['epoch']} already meets or exceeds target epochs={args.epochs}."
            )

    torch.manual_seed(args.seed)
    teacher = build_external_teacher(args.teacher_backend, args.teacher_checkpoint)
    model = build_model(args, teacher=teacher).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    if checkpoint is not None:
        restore_training_state(model, optimizer, checkpoint)

    weights = LossWeights(
        alpha=args.loss_alpha,
        lambda_align=args.lambda_align,
        lambda_ts=args.lambda_ts,
        lambda_visual=args.lambda_visual,
        temperature=args.temperature,
    )

    audio_loader = build_deam_loader(args) if uses_audio_pretraining(args.method) else None
    veatic_train_loader = build_veatic_loader(args, split="train")
    veatic_test_loader = build_veatic_loader(args, split="test")

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs} [{args.method}]")
        audio_metrics: Dict[str, float] = {}
        if audio_loader is not None:
            audio_metrics = train_audio_epoch(
                model,
                audio_loader,
                optimizer,
                device,
                weights,
                max_steps=args.max_audio_steps,
                log_every=args.log_every,
            )

        joint_metrics = train_joint_epoch(
            method=args.method,
            model=model,
            loader=veatic_train_loader,
            optimizer=optimizer,
            device=device,
            weights=weights,
            lambda_domain=args.lambda_domain,
            max_steps=args.max_joint_steps,
            log_every=args.log_every,
        )
        eval_metrics = evaluate_epoch(
            method=args.method,
            model=model,
            loader=veatic_test_loader,
            device=device,
            weights=weights,
            max_steps=args.max_eval_steps,
            log_every=args.log_every,
        )

        print("  audio_train:", format_metrics(audio_metrics))
        print("  joint_train:", format_metrics(joint_metrics))
        print("  eval:", format_metrics(eval_metrics))

        epoch_record = {
            "epoch": epoch,
            "method": args.method,
            "audio_train": audio_metrics,
            "joint_train": joint_metrics,
            "eval": eval_metrics,
        }
        append_jsonl(output_dir / "metrics.jsonl", epoch_record)
        save_checkpoint(
            output_dir / f"checkpoint_epoch_{epoch}.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            args=args,
            metrics=epoch_record,
        )


def build_model(args: argparse.Namespace, teacher: Optional[nn.Module] = None) -> nn.Module:
    config = CrossModalVAConfig(temporal_model=args.temporal_model)
    frame_backbone, video_backbone = build_visual_backbones(args, config)
    if args.method in {"proposed", "visual_only"}:
        return CrossModalVAModel(
            config=config,
            frame_backbone=frame_backbone,
            video_backbone=video_backbone,
            joint_fusion=args.joint_fusion,
            teacher=teacher,
        )
    if args.method == "late_fusion":
        return LateFusionVAModel(
            config=config,
            frame_backbone=frame_backbone,
            fusion_mode=args.late_fusion_mode,
        )
    if args.method == "dann":
        return DomainAdversarialVAModel(config=config, frame_backbone=frame_backbone)
    raise ValueError(f"Unsupported method: {args.method}")


def build_visual_backbones(
    args: argparse.Namespace,
    config: CrossModalVAConfig,
) -> tuple[Optional[nn.Module], Optional[nn.Module]]:
    if args.visual_backbone == "simple_cnn":
        return None, None
    if args.visual_backbone == "resnet18_imagenet":
        return ResNet18FrameCNN(
            hidden_dim=config.visual_hidden_dim,
            output_dim=config.visual_hidden_dim,
            dropout=config.dropout,
            pretrained=True,
            freeze_backbone=args.freeze_visual_backbone,
        ), None
    if args.visual_backbone == "video_r2plus1d_18_kinetics400":
        return None, R2Plus1D18VideoBackbone(
            hidden_dim=config.visual_hidden_dim,
            output_dim=config.visual_hidden_dim,
            dropout=config.dropout,
            pretrained=True,
            freeze_backbone=args.freeze_visual_backbone,
        )
    if args.visual_backbone == "video_videomae_base_k400":
        return None, VideoMAEBaseVideoBackbone(
            hidden_dim=config.visual_hidden_dim,
            output_dim=config.visual_hidden_dim,
            dropout=config.dropout,
            pretrained_model_name="MCG-NJU/videomae-base-finetuned-kinetics",
            freeze_backbone=args.freeze_visual_backbone,
            trainable_blocks=args.videomae_trainable_blocks,
        )
    if args.visual_backbone == "video_mvit_v2_s_kinetics400":
        return None, MViTV2SVideoBackbone(
            hidden_dim=config.visual_hidden_dim,
            output_dim=config.visual_hidden_dim,
            dropout=config.dropout,
            pretrained=True,
            freeze_backbone=args.freeze_visual_backbone,
        )
    raise ValueError(f"Unsupported visual backbone: {args.visual_backbone}")


def uses_audio_pretraining(method: str) -> bool:
    return method in {"proposed", "dann"}


def build_deam_loader(args: argparse.Namespace) -> DataLoader:
    dataset = DEAMDataset(
        audio_zip_path=args.audio_zip,
        annotations_dir=args.audio_annotations,
        cache_dir=Path(args.cache_dir) / "deam",
        transform=AudioFeatureTransform(target_num_frames=args.audio_frames),
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_deam_batch,
    )


def build_veatic_loader(args: argparse.Namespace, split: str) -> DataLoader:
    teacher_audio_seconds = 4.0
    teacher_sample_rate = 16000
    if args.teacher_checkpoint is not None:
        teacher_metadata = extract_teacher_metadata(args.teacher_checkpoint)
        teacher_audio_seconds = float(teacher_metadata.get("teacher_audio_seconds") or teacher_audio_seconds)
        teacher_sample_rate = int(teacher_metadata.get("teacher_sample_rate") or teacher_sample_rate)

    dataset = VEATICDataset(
        zip_path=args.veatic_zip,
        cache_dir=Path(args.cache_dir) / "veatic",
        clip_length=args.clip_length,
        stride=args.clip_stride,
        split=split,
        transform=VideoFrameTransform(size=args.frame_size),
        audio_transform=AudioFeatureTransform(target_num_frames=args.audio_frames),
        include_teacher_audio=(args.teacher_backend is not None and split == "train"),
        teacher_audio_seconds=teacher_audio_seconds,
        teacher_sample_rate=teacher_sample_rate,
        train_fraction=args.visual_train_fraction if split == "train" else 1.0,
        sampling_seed=args.seed,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(split == "train"),
        num_workers=args.num_workers,
    )


def train_audio_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: AdamW,
    device: torch.device,
    weights: LossWeights,
    max_steps: Optional[int] = None,
    log_every: int = 100,
) -> Dict[str, float]:
    model.train()
    metrics = RunningAverage()
    regression_metrics = RegressionMetricTracker()
    for step, batch in enumerate(loader, start=1):
        audio = batch["audio"].to(device)
        targets = batch["target"].to(device)

        outputs = model(audio=audio)
        losses = cross_modal_training_loss(outputs, audio_targets=targets, weights=weights)

        optimizer.zero_grad(set_to_none=True)
        losses["audio_supervised"].backward()
        optimizer.step()
        metrics.update({"loss": losses["audio_supervised"].item()})
        regression_metrics.update(outputs["audio_prediction"], targets)
        maybe_log_progress("audio_train", step, metrics, log_every)
        if max_steps is not None and step >= max_steps:
            break
    return {**metrics.compute(), **regression_metrics.compute()}


def train_joint_epoch(
    method: str,
    model: nn.Module,
    loader: DataLoader,
    optimizer: AdamW,
    device: torch.device,
    weights: LossWeights,
    lambda_domain: float,
    max_steps: Optional[int] = None,
    log_every: int = 100,
) -> Dict[str, float]:
    if method == "proposed":
        return train_contrastive_epoch(model, loader, optimizer, device, weights, max_steps, log_every)
    if method == "visual_only":
        return train_visual_only_epoch(model, loader, optimizer, device, weights, max_steps, log_every)
    if method == "late_fusion":
        return train_late_fusion_epoch(model, loader, optimizer, device, weights, max_steps, log_every)
    if method == "dann":
        return train_dann_epoch(
            model,
            loader,
            optimizer,
            device,
            weights,
            lambda_domain=lambda_domain,
            max_steps=max_steps,
            log_every=log_every,
        )
    raise ValueError(f"Unsupported method: {method}")


def train_contrastive_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: AdamW,
    device: torch.device,
    weights: LossWeights,
    max_steps: Optional[int] = None,
    log_every: int = 100,
) -> Dict[str, float]:
    model.train()
    metrics = RunningAverage()
    regression_metrics = RegressionMetricTracker()
    for step, batch in enumerate(loader, start=1):
        video = batch["video"].to(device)
        paired_audio = batch["paired_audio"].to(device)
        targets = batch["target"].to(device)
        teacher_audio = batch.get("teacher_audio")
        if teacher_audio is not None:
            teacher_audio = teacher_audio.to(device)

        outputs = model(
            paired_audio=paired_audio,
            paired_video=video,
            teacher_audio=teacher_audio,
        )
        losses = cross_modal_training_loss(outputs, visual_targets=targets, weights=weights)

        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        optimizer.step()
        metrics.update(
            {
                "visual_loss": losses["visual_supervised"].item(),
                "alignment": losses["alignment"].item(),
                "distillation": losses["distillation"].item(),
                "total": losses["total"].item(),
            }
        )
        regression_metrics.update(outputs["paired_video_prediction"], targets)
        maybe_log_progress("joint_train", step, metrics, log_every)
        if max_steps is not None and step >= max_steps:
            break
    return {**metrics.compute(), **regression_metrics.compute()}


def train_visual_only_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: AdamW,
    device: torch.device,
    weights: LossWeights,
    max_steps: Optional[int] = None,
    log_every: int = 100,
) -> Dict[str, float]:
    model.train()
    metrics = RunningAverage()
    regression_metrics = RegressionMetricTracker()
    for step, batch in enumerate(loader, start=1):
        video = batch["video"].to(device)
        targets = batch["target"].to(device)
        outputs = model(video=video)
        losses = cross_modal_training_loss(outputs, visual_targets=targets, weights=weights)

        optimizer.zero_grad(set_to_none=True)
        losses["visual_supervised"].backward()
        optimizer.step()
        metrics.update({"visual_loss": losses["visual_supervised"].item()})
        regression_metrics.update(outputs["video_prediction"], targets)
        maybe_log_progress("visual_only_train", step, metrics, log_every)
        if max_steps is not None and step >= max_steps:
            break
    return {**metrics.compute(), **regression_metrics.compute()}


def train_late_fusion_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: AdamW,
    device: torch.device,
    weights: LossWeights,
    max_steps: Optional[int] = None,
    log_every: int = 100,
) -> Dict[str, float]:
    model.train()
    metrics = RunningAverage()
    regression_metrics = RegressionMetricTracker()
    for step, batch in enumerate(loader, start=1):
        video = batch["video"].to(device)
        paired_audio = batch["paired_audio"].to(device)
        targets = batch["target"].to(device)

        outputs = model(audio=paired_audio, video=video)
        loss = regression_loss(targets, outputs["fusion_prediction"], alpha=weights.alpha)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        metrics.update({"fusion_loss": loss.item()})
        regression_metrics.update(outputs["fusion_prediction"], targets)
        maybe_log_progress("late_fusion_train", step, metrics, log_every)
        if max_steps is not None and step >= max_steps:
            break
    return {**metrics.compute(), **regression_metrics.compute()}


def train_dann_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: AdamW,
    device: torch.device,
    weights: LossWeights,
    lambda_domain: float,
    max_steps: Optional[int] = None,
    log_every: int = 100,
) -> Dict[str, float]:
    model.train()
    metrics = RunningAverage()
    regression_metrics = RegressionMetricTracker()
    for step, batch in enumerate(loader, start=1):
        video = batch["video"].to(device)
        paired_audio = batch["paired_audio"].to(device)
        targets = batch["target"].to(device)

        outputs = model(
            paired_audio=paired_audio,
            paired_video=video,
            grl_lambda=lambda_domain,
        )
        visual_loss = regression_loss(targets, outputs["paired_video_prediction"], alpha=weights.alpha)
        domain_loss = domain_adversarial_loss(
            outputs["paired_audio_domain_logits"],
            outputs["paired_video_domain_logits"],
        )
        total = weights.lambda_visual * visual_loss + lambda_domain * domain_loss

        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()
        metrics.update(
            {
                "visual_loss": visual_loss.item(),
                "domain_loss": domain_loss.item(),
                "total": total.item(),
            }
        )
        regression_metrics.update(outputs["paired_video_prediction"], targets)
        maybe_log_progress("dann_train", step, metrics, log_every)
        if max_steps is not None and step >= max_steps:
            break
    return {**metrics.compute(), **regression_metrics.compute()}


def evaluate_epoch(
    method: str,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    weights: LossWeights,
    max_steps: Optional[int] = None,
    log_every: int = 100,
) -> Dict[str, float]:
    if method == "late_fusion":
        return evaluate_late_fusion_epoch(model, loader, device, weights, max_steps, log_every)
    return evaluate_visual_epoch(model, loader, device, weights, max_steps, log_every)


@torch.no_grad()
def evaluate_visual_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    weights: LossWeights,
    max_steps: Optional[int] = None,
    log_every: int = 100,
) -> Dict[str, float]:
    model.eval()
    metrics = RunningAverage()
    regression_metrics = RegressionMetricTracker()
    for step, batch in enumerate(loader, start=1):
        video = batch["video"].to(device)
        targets = batch["target"].to(device)
        outputs = model(video=video)
        losses = cross_modal_training_loss(outputs, visual_targets=targets, weights=weights)
        metrics.update({"visual_loss": losses["visual_supervised"].item()})
        regression_metrics.update(outputs["video_prediction"], targets)
        maybe_log_progress("eval", step, metrics, log_every)
        if max_steps is not None and step >= max_steps:
            break
    return {**metrics.compute(), **regression_metrics.compute()}


@torch.no_grad()
def evaluate_late_fusion_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    weights: LossWeights,
    max_steps: Optional[int] = None,
    log_every: int = 100,
) -> Dict[str, float]:
    model.eval()
    metrics = RunningAverage()
    regression_metrics = RegressionMetricTracker()
    for step, batch in enumerate(loader, start=1):
        video = batch["video"].to(device)
        paired_audio = batch["paired_audio"].to(device)
        targets = batch["target"].to(device)
        outputs = model(audio=paired_audio, video=video)
        loss = regression_loss(targets, outputs["fusion_prediction"], alpha=weights.alpha)
        metrics.update({"fusion_loss": loss.item()})
        regression_metrics.update(outputs["fusion_prediction"], targets)
        maybe_log_progress("eval", step, metrics, log_every)
        if max_steps is not None and step >= max_steps:
            break
    return {**metrics.compute(), **regression_metrics.compute()}


class RunningAverage:
    def __init__(self) -> None:
        self.values: Dict[str, list[float]] = {}

    def update(self, batch_metrics: Dict[str, float]) -> None:
        for key, value in batch_metrics.items():
            self.values.setdefault(key, []).append(float(value))

    def compute(self) -> Dict[str, float]:
        return {
            key: sum(values) / len(values)
            for key, values in self.values.items()
            if values
        }


class RegressionMetricTracker:
    def __init__(self) -> None:
        self.predictions: list[Tensor] = []
        self.targets: list[Tensor] = []

    def update(self, predictions: Tensor, targets: Tensor) -> None:
        self.predictions.append(predictions.detach().cpu())
        self.targets.append(targets.detach().cpu())

    def compute(self) -> Dict[str, float]:
        if not self.predictions:
            return {}

        predictions = torch.cat(self.predictions, dim=0)
        targets = torch.cat(self.targets, dim=0)

        mae = (predictions - targets).abs().mean()
        rmse = torch.sqrt((predictions - targets).pow(2).mean())
        ccc = concordance_correlation_coefficient(targets, predictions)

        return {
            "mae": float(mae),
            "rmse": float(rmse),
            "ccc_valence": float(ccc[0]),
            "ccc_arousal": float(ccc[1]),
            "ccc_mean": float(ccc.mean()),
        }


def format_metrics(metrics: Dict[str, float]) -> str:
    if not metrics:
        return "n/a"
    return ", ".join(f"{key}={value:.4f}" for key, value in metrics.items())


def maybe_log_progress(stage: str, step: int, metrics: "RunningAverage", log_every: int) -> None:
    if log_every <= 0:
        return
    if step == 1 or step % log_every == 0:
        print(f"    {stage} step {step}: {format_metrics(metrics.compute())}", flush=True)


def collate_deam_batch(batch: list[Dict[str, object]]) -> Dict[str, object]:
    collated = {
        "audio": default_collate([sample["audio"] for sample in batch]),
        "target": default_collate([sample["target"] for sample in batch]),
        "song_id": [sample["song_id"] for sample in batch],
    }
    if "dynamic_target" in batch[0]:
        collated["dynamic_target"] = [sample["dynamic_target"] for sample in batch]
    return collated


def append_jsonl(path: Path, record: Dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


RESUME_ARG_KEYS = [
    "method",
    "batch_size",
    "lr",
    "clip_length",
    "clip_stride",
    "frame_size",
    "audio_frames",
    "audio_zip",
    "audio_annotations",
    "veatic_zip",
    "cache_dir",
    "temporal_model",
    "loss_alpha",
    "lambda_align",
    "lambda_ts",
    "lambda_visual",
    "lambda_domain",
    "temperature",
    "teacher_backend",
    "teacher_checkpoint",
    "visual_backbone",
    "freeze_visual_backbone",
    "joint_fusion",
    "late_fusion_mode",
    "num_workers",
    "visual_train_fraction",
    "max_audio_steps",
    "max_joint_steps",
    "max_eval_steps",
    "seed",
]


def load_checkpoint(path: Path, device: torch.device) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=device, weights_only=False)


def restore_training_state(
    model: nn.Module,
    optimizer: AdamW,
    checkpoint: Dict[str, object],
) -> None:
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


def validate_resume_args(args: argparse.Namespace, checkpoint: Dict[str, object]) -> None:
    checkpoint_args = checkpoint.get("args")
    if not isinstance(checkpoint_args, dict):
        raise ValueError("Checkpoint is missing saved args for resume validation.")

    mismatches: list[str] = []
    for key in RESUME_ARG_KEYS:
        current_value = getattr(args, key)
        checkpoint_value = checkpoint_args.get(key)
        if current_value != checkpoint_value:
            mismatches.append(f"{key}: current={current_value!r}, checkpoint={checkpoint_value!r}")

    checkpoint_output_dir = checkpoint_args.get("output_dir")
    if args.output_dir != checkpoint_output_dir:
        mismatches.append(
            f"output_dir: current={args.output_dir!r}, checkpoint={checkpoint_output_dir!r}"
        )

    metrics_path = Path(args.output_dir) / "metrics.jsonl"
    if metrics_path.exists():
        existing_epochs = read_metrics_epochs(metrics_path)
        resume_epoch = int(checkpoint["epoch"])
        if existing_epochs and existing_epochs != list(range(1, resume_epoch + 1)):
            mismatches.append(
                f"metrics.jsonl epochs are not a clean prefix through checkpoint epoch {resume_epoch}: "
                f"found={existing_epochs}"
            )

    if mismatches:
        mismatch_text = "\n".join(f"- {item}" for item in mismatches)
        raise ValueError(f"Resume validation failed:\n{mismatch_text}")


def read_metrics_epochs(path: Path) -> list[int]:
    epochs: list[int] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            epochs.append(int(record["epoch"]))
    return epochs


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: AdamW,
    epoch: int,
    args: argparse.Namespace,
    metrics: Dict[str, object],
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
        "metrics": metrics,
    }
    torch.save(checkpoint, path)


if __name__ == "__main__":
    main()
