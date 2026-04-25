from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import opensmile
import torch
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import AudioFeatureTransform, VEATICDataset, VideoFrameTransform
from models import (
    LeaderFollowerAttentiveFusionModel,
    LossWeights,
    concordance_correlation_coefficient,
    regression_loss,
)


METHOD_CHOICES = [
    "ortega_feature_svr",
    "ortega_decision_svr",
    "zhang_leader_follower",
]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train paper-derived baselines on VEATIC.")
    parser.add_argument("--method", choices=METHOD_CHOICES, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--clip-length", type=int, default=8)
    parser.add_argument("--clip-stride", type=int, default=8)
    parser.add_argument("--frame-size", type=int, default=224)
    parser.add_argument("--audio-frames", type=int, default=256)
    parser.add_argument("--veatic-zip", default="data/veatic/VEATIC.zip")
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--feature-cache-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--svr-c", type=float, default=10.0)
    parser.add_argument("--svr-epsilon", type=float, default=0.1)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--max-eval-steps", type=int, default=None)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.method.startswith("ortega_"):
        train_ortega_baseline(args, output_dir)
        return
    if args.method == "zhang_leader_follower":
        train_zhang_baseline(args, output_dir)
        return
    raise ValueError(f"Unsupported method: {args.method}")


def train_ortega_baseline(args: argparse.Namespace, output_dir: Path) -> None:
    device = torch.device(args.device)
    feature_cache_dir = Path(args.feature_cache_dir) if args.feature_cache_dir else output_dir
    feature_cache_dir.mkdir(parents=True, exist_ok=True)
    train_loader = build_classical_loader(args, split="train")
    test_loader = build_classical_loader(args, split="test")

    extractor = OrtegaFeatureExtractor(device=device)
    train_data = extractor.extract_dataset(
        train_loader,
        cache_path=feature_cache_dir / "train_features.pt",
        max_steps=args.max_train_steps,
    )
    test_data = extractor.extract_dataset(
        test_loader,
        cache_path=feature_cache_dir / "test_features.pt",
        max_steps=args.max_eval_steps,
    )

    y_train = train_data["targets"]
    y_test = test_data["targets"]

    if args.method == "ortega_feature_svr":
        x_train = np.concatenate([train_data["audio_features"], train_data["video_features"]], axis=1)
        x_test = np.concatenate([test_data["audio_features"], test_data["video_features"]], axis=1)
        predictions = fit_multioutput_svr(x_train, y_train, x_test, c=args.svr_c, epsilon=args.svr_epsilon)
        metrics = compute_numpy_metrics(y_test, predictions)
        metrics["method"] = args.method
    else:
        predictions, fusion_info = fit_decision_level_svr(
            train_data=train_data,
            test_data=test_data,
            c=args.svr_c,
            epsilon=args.svr_epsilon,
        )
        metrics = compute_numpy_metrics(y_test, predictions)
        metrics["method"] = args.method
        metrics["fusion_alpha_valence"] = fusion_info["alpha_valence"]
        metrics["fusion_alpha_arousal"] = fusion_info["alpha_arousal"]

    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


def train_zhang_baseline(args: argparse.Namespace, output_dir: Path) -> None:
    device = torch.device(args.device)
    model = LeaderFollowerAttentiveFusionModel(audio_input_dim=513).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    weights = LossWeights()

    train_loader = build_deep_loader(args, split="train")
    test_loader = build_deep_loader(args, split="test")

    records = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_lfan_epoch(
            model,
            train_loader,
            optimizer,
            device,
            weights,
            max_steps=args.max_train_steps,
        )
        eval_metrics = eval_lfan_epoch(
            model,
            test_loader,
            device,
            weights,
            max_steps=args.max_eval_steps,
        )
        record = {
            "epoch": epoch,
            "method": args.method,
            "train": train_metrics,
            "eval": eval_metrics,
        }
        records.append(record)
        print(json.dumps(record))
        torch.save(
            {
                "epoch": epoch,
                "method": args.method,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": vars(args),
                "metrics": record,
            },
            output_dir / f"checkpoint_epoch_{epoch}.pt",
        )

    with (output_dir / "metrics.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def build_classical_loader(args: argparse.Namespace, split: str) -> DataLoader:
    dataset = VEATICDataset(
        zip_path=args.veatic_zip,
        cache_dir=Path(args.cache_dir) / "veatic_classical",
        clip_length=args.clip_length,
        stride=args.clip_stride,
        split=split,
        transform=VideoFrameTransform(size=224),
        audio_transform=None,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_classical_batch,
    )


def build_deep_loader(args: argparse.Namespace, split: str) -> DataLoader:
    dataset = VEATICDataset(
        zip_path=args.veatic_zip,
        cache_dir=Path(args.cache_dir) / "veatic_zhang",
        clip_length=args.clip_length,
        stride=args.clip_stride,
        split=split,
        transform=VideoFrameTransform(size=args.frame_size),
        audio_transform=AudioFeatureTransform(target_num_frames=args.audio_frames),
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(split == "train"),
        num_workers=args.num_workers,
    )


def collate_classical_batch(batch: list[Dict[str, object]]) -> Dict[str, object]:
    return {
        "video": torch.stack([sample["video"] for sample in batch], dim=0),
        "paired_audio": [sample["paired_audio"] for sample in batch],
        "audio_sample_rate": [int(sample["audio_sample_rate"]) for sample in batch],
        "target": torch.stack([sample["target"] for sample in batch], dim=0),
        "video_id": [sample["video_id"] for sample in batch],
        "start_index": [int(sample["start_index"]) for sample in batch],
        "end_index": [int(sample["end_index"]) for sample in batch],
    }


class OrtegaFeatureExtractor:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.visual_backbone = torch.nn.Sequential(*list(backbone.children())[:-1]).to(device)
        self.visual_backbone.eval()
        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 1, 3, 1, 1)
        self.visual_mean = mean
        self.visual_std = std

    @torch.no_grad()
    def extract_dataset(
        self,
        loader: DataLoader,
        cache_path: Path,
        max_steps: int | None = None,
    ) -> Dict[str, np.ndarray]:
        if cache_path.exists():
            cached = torch.load(cache_path, map_location="cpu", weights_only=False)
            return {key: np.asarray(value) if isinstance(value, list) else value for key, value in cached.items()}

        audio_features = []
        video_features = []
        targets = []
        video_ids = []
        start_indices = []

        for step, batch in enumerate(loader, start=1):
            videos = batch["video"].to(self.device)
            video_vectors = self._extract_video_features(videos)
            audio_vectors = self._extract_audio_features(batch["paired_audio"], batch["audio_sample_rate"])

            video_features.append(video_vectors)
            audio_features.append(audio_vectors)
            targets.append(batch["target"].cpu().numpy())
            video_ids.extend(batch["video_id"])
            start_indices.extend(batch["start_index"])
            if max_steps is not None and step >= max_steps:
                break

        payload = {
            "audio_features": np.concatenate(audio_features, axis=0),
            "video_features": np.concatenate(video_features, axis=0),
            "targets": np.concatenate(targets, axis=0),
            "video_ids": np.array(video_ids),
            "start_index": np.array(start_indices),
        }
        torch.save(payload, cache_path)
        return payload

    def _extract_video_features(self, videos: Tensor) -> np.ndarray:
        normalized = (videos - self.visual_mean) / self.visual_std
        batch_size, num_frames, channels, height, width = normalized.shape
        frame_batch = normalized.view(batch_size * num_frames, channels, height, width)
        features = self.visual_backbone(frame_batch).flatten(1)
        pooled = features.view(batch_size, num_frames, -1).mean(dim=1)
        return pooled.cpu().numpy().astype(np.float32)

    def _extract_audio_features(self, audio_batch: list[Tensor], sample_rates: list[int]) -> np.ndarray:
        features = []
        for audio_tensor, sample_rate in zip(audio_batch, sample_rates):
            waveform = audio_tensor.mean(dim=0).detach().cpu().numpy()
            smile_df = self.smile.process_signal(waveform, sample_rate)
            features.append(smile_df.to_numpy(dtype=np.float32).reshape(-1))
        return np.stack(features, axis=0)


def fit_multioutput_svr(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    c: float,
    epsilon: float,
) -> np.ndarray:
    predictions = []
    for dim in range(y_train.shape[1]):
        model = build_svr(c=c, epsilon=epsilon)
        model.fit(x_train, y_train[:, dim])
        predictions.append(model.predict(x_test))
    return np.stack(predictions, axis=1)


def fit_decision_level_svr(
    train_data: Dict[str, np.ndarray],
    test_data: Dict[str, np.ndarray],
    c: float,
    epsilon: float,
) -> tuple[np.ndarray, Dict[str, float]]:
    train_indices, val_indices = split_by_video(train_data["video_ids"])
    audio_train = train_data["audio_features"]
    video_train = train_data["video_features"]
    y_train = train_data["targets"]

    audio_models = []
    video_models = []
    alphas = []

    for dim in range(y_train.shape[1]):
        audio_model = build_svr(c=c, epsilon=epsilon)
        video_model = build_svr(c=c, epsilon=epsilon)
        audio_model.fit(audio_train[train_indices], y_train[train_indices, dim])
        video_model.fit(video_train[train_indices], y_train[train_indices, dim])

        audio_val_pred = audio_model.predict(audio_train[val_indices])
        video_val_pred = video_model.predict(video_train[val_indices])
        alpha = select_fusion_alpha(
            y_true=y_train[val_indices, dim],
            audio_pred=audio_val_pred,
            video_pred=video_val_pred,
        )
        alphas.append(alpha)

        audio_model.fit(audio_train, y_train[:, dim])
        video_model.fit(video_train, y_train[:, dim])
        audio_models.append(audio_model)
        video_models.append(video_model)

    predictions = []
    for dim, (audio_model, video_model, alpha) in enumerate(zip(audio_models, video_models, alphas)):
        audio_pred = audio_model.predict(test_data["audio_features"])
        video_pred = video_model.predict(test_data["video_features"])
        predictions.append(alpha * video_pred + (1.0 - alpha) * audio_pred)

    return (
        np.stack(predictions, axis=1),
        {
            "alpha_valence": float(alphas[0]),
            "alpha_arousal": float(alphas[1]),
        },
    )


def build_svr(c: float, epsilon: float) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("svr", SVR(kernel="rbf", C=c, epsilon=epsilon)),
        ]
    )


def split_by_video(video_ids: np.ndarray, val_ratio: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    unique_ids = sorted({str(video_id) for video_id in video_ids})
    if len(video_ids) < 4 or len(unique_ids) < 2:
        indices = np.arange(len(video_ids), dtype=np.int64)
        split = max(1, len(indices) - max(1, int(round(len(indices) * val_ratio))))
        split = min(split, len(indices) - 1)
        return indices[:split], indices[split:]
    val_count = max(1, int(round(len(unique_ids) * val_ratio)))
    val_count = min(val_count, len(unique_ids) - 1)
    val_ids = set(unique_ids[-val_count:])
    train_indices = [index for index, video_id in enumerate(video_ids) if str(video_id) not in val_ids]
    val_indices = [index for index, video_id in enumerate(video_ids) if str(video_id) in val_ids]
    if not train_indices or not val_indices:
        indices = np.arange(len(video_ids), dtype=np.int64)
        split = max(1, len(indices) - max(1, int(round(len(indices) * val_ratio))))
        split = min(split, len(indices) - 1)
        return indices[:split], indices[split:]
    return np.asarray(train_indices, dtype=np.int64), np.asarray(val_indices, dtype=np.int64)


def select_fusion_alpha(y_true: np.ndarray, audio_pred: np.ndarray, video_pred: np.ndarray) -> float:
    best_alpha = 0.5
    best_score = -float("inf")
    for alpha in np.linspace(0.0, 1.0, num=21):
        fused = alpha * video_pred + (1.0 - alpha) * audio_pred
        score = float(concordance_correlation_coefficient(
            torch.tensor(y_true, dtype=torch.float32),
            torch.tensor(fused, dtype=torch.float32),
        ).mean())
        if score > best_score:
            best_score = score
            best_alpha = float(alpha)
    return best_alpha


def train_lfan_epoch(
    model: LeaderFollowerAttentiveFusionModel,
    loader: DataLoader,
    optimizer: AdamW,
    device: torch.device,
    weights: LossWeights,
    max_steps: int | None = None,
) -> Dict[str, float]:
    model.train()
    tracker = NumpyMetricTracker()
    losses = []
    for step, batch in enumerate(loader, start=1):
        audio = batch["paired_audio"].to(device)
        video = batch["video"].to(device)
        targets = batch["target"].to(device)

        outputs = model(audio=audio, video=video)
        loss = regression_loss(targets, outputs["prediction"], alpha=weights.alpha)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        losses.append(float(loss.detach().cpu()))
        tracker.update(outputs["prediction"], targets)
        if max_steps is not None and step >= max_steps:
            break

    metrics = tracker.compute()
    metrics["loss"] = sum(losses) / len(losses)
    return metrics


@torch.no_grad()
def eval_lfan_epoch(
    model: LeaderFollowerAttentiveFusionModel,
    loader: DataLoader,
    device: torch.device,
    weights: LossWeights,
    max_steps: int | None = None,
) -> Dict[str, float]:
    model.eval()
    tracker = NumpyMetricTracker()
    losses = []
    for step, batch in enumerate(loader, start=1):
        audio = batch["paired_audio"].to(device)
        video = batch["video"].to(device)
        targets = batch["target"].to(device)

        outputs = model(audio=audio, video=video)
        loss = regression_loss(targets, outputs["prediction"], alpha=weights.alpha)

        losses.append(float(loss.detach().cpu()))
        tracker.update(outputs["prediction"], targets)
        if max_steps is not None and step >= max_steps:
            break

    metrics = tracker.compute()
    metrics["loss"] = sum(losses) / len(losses)
    return metrics


class NumpyMetricTracker:
    def __init__(self) -> None:
        self.predictions: list[np.ndarray] = []
        self.targets: list[np.ndarray] = []

    def update(self, predictions: Tensor, targets: Tensor) -> None:
        self.predictions.append(predictions.detach().cpu().numpy())
        self.targets.append(targets.detach().cpu().numpy())

    def compute(self) -> Dict[str, float]:
        predictions = np.concatenate(self.predictions, axis=0)
        targets = np.concatenate(self.targets, axis=0)
        return compute_numpy_metrics(targets, predictions)


def compute_numpy_metrics(targets: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
    target_tensor = torch.tensor(targets, dtype=torch.float32)
    pred_tensor = torch.tensor(predictions, dtype=torch.float32)
    ccc = concordance_correlation_coefficient(target_tensor, pred_tensor)
    mae = torch.mean(torch.abs(pred_tensor - target_tensor))
    rmse = torch.sqrt(torch.mean((pred_tensor - target_tensor) ** 2))
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "ccc_valence": float(ccc[0]),
        "ccc_arousal": float(ccc[1]),
        "ccc_mean": float(ccc.mean()),
    }


if __name__ == "__main__":
    main()
