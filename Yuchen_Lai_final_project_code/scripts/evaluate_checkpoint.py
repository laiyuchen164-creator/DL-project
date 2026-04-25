from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import AudioFeatureTransform, VEATICDataset, VideoFrameTransform
from models import (
    CrossModalVAConfig,
    CrossModalVAModel,
    DomainAdversarialVAModel,
    LateFusionVAModel,
    LossWeights,
    concordance_correlation_coefficient,
    cross_modal_training_loss,
    regression_loss,
)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a training checkpoint on VEATIC.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    train_args = checkpoint["args"]
    method = train_args.get("method", "proposed")

    model = build_model(method, train_args).to(args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = VEATICDataset(
        zip_path=train_args["veatic_zip"],
        cache_dir=Path(train_args["cache_dir"]) / "veatic",
        clip_length=train_args["clip_length"],
        stride=train_args["clip_stride"],
        split=args.split,
        transform=VideoFrameTransform(size=train_args["frame_size"]),
        audio_transform=AudioFeatureTransform(target_num_frames=train_args["audio_frames"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    metrics = evaluate(model, loader, torch.device(args.device), method, max_steps=args.max_steps)
    metrics["checkpoint"] = str(args.checkpoint)
    metrics["split"] = args.split
    metrics["method"] = method

    print(json.dumps(metrics, indent=2))
    if args.output_json is not None:
        args.output_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def build_model(method: str, train_args: dict) -> torch.nn.Module:
    config = CrossModalVAConfig(temporal_model=train_args["temporal_model"])
    if method in {"proposed", "visual_only"}:
        return CrossModalVAModel(config=config)
    if method == "late_fusion":
        return LateFusionVAModel(config=config)
    if method == "dann":
        return DomainAdversarialVAModel(config=config)
    raise ValueError(f"Unsupported method: {method}")


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    method: str,
    max_steps: int | None = None,
) -> dict[str, float]:
    losses = []
    predictions = []
    targets_list = []
    weights = LossWeights()

    for step, batch in enumerate(loader, start=1):
        video = batch["video"].to(device)
        paired_audio = batch["paired_audio"].to(device)
        targets = batch["target"].to(device)

        if method == "late_fusion":
            outputs = model(audio=paired_audio, video=video)
            loss = regression_loss(targets, outputs["fusion_prediction"], alpha=weights.alpha)
            prediction = outputs["fusion_prediction"]
            loss_name = "fusion_loss"
        else:
            outputs = model(video=video)
            loss = cross_modal_training_loss(outputs, visual_targets=targets, weights=weights)["visual_supervised"]
            prediction = outputs["video_prediction"]
            loss_name = "visual_loss"

        losses.append(float(loss.detach().cpu()))
        predictions.append(prediction.detach().cpu())
        targets_list.append(targets.detach().cpu())
        if max_steps is not None and step >= max_steps:
            break

    predictions_tensor = torch.cat(predictions, dim=0)
    targets_tensor = torch.cat(targets_list, dim=0)
    ccc = concordance_correlation_coefficient(targets_tensor, predictions_tensor)
    mae = (predictions_tensor - targets_tensor).abs().mean()
    rmse = torch.sqrt((predictions_tensor - targets_tensor).pow(2).mean())

    return {
        loss_name: sum(losses) / len(losses),
        "mae": float(mae),
        "rmse": float(rmse),
        "ccc_valence": float(ccc[0]),
        "ccc_arousal": float(ccc[1]),
        "ccc_mean": float(ccc.mean()),
    }


if __name__ == "__main__":
    main()
