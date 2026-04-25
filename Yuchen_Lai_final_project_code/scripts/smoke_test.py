from __future__ import annotations
# Submission variant note:
# This packaged copy emphasizes experiment orchestration, documentation, and figure scripts.
# Package owner: Yuchen Lai


import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import AudioFeatureTransform, DEAMDataset, VEATICDataset, VideoFrameTransform
from models import (
    CrossModalVAConfig,
    CrossModalVAModel,
    DomainAdversarialVAModel,
    LateFusionVAModel,
    LossWeights,
    cross_modal_training_loss,
    domain_adversarial_loss,
    regression_loss,
)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    audio_transform = AudioFeatureTransform(target_num_frames=512)
    video_transform = VideoFrameTransform(size=224)

    deam = DEAMDataset(
        audio_zip_path=PROJECT_ROOT / "data/deam/DEAM_audio.zip",
        annotations_dir=PROJECT_ROOT / "data/deam/annotations/annotations",
        cache_dir=PROJECT_ROOT / "data/cache/deam",
        transform=audio_transform,
        target_length_seconds=30.0,
    )
    print(f"deam_samples={len(deam)}")
    deam_sample = deam[0]
    print(
        "deam_shapes",
        {
            "audio": tuple(deam_sample["audio"].shape),
            "target": tuple(deam_sample["target"].shape),
            "dynamic_target": tuple(deam_sample["dynamic_target"].shape),
        },
    )

    veatic = VEATICDataset(
        zip_path=PROJECT_ROOT / "data/veatic/VEATIC.zip",
        cache_dir=PROJECT_ROOT / "data/cache/veatic",
        clip_length=16,
        stride=16,
        split="train",
        transform=video_transform,
        audio_transform=audio_transform,
    )
    print(f"veatic_windows={len(veatic)}")
    veatic_sample = veatic[0]
    print(
        "veatic_shapes",
        {
            "video": tuple(veatic_sample["video"].shape),
            "paired_audio": tuple(veatic_sample["paired_audio"].shape),
            "target": tuple(veatic_sample["target"].shape),
            "frame_targets": tuple(veatic_sample["frame_targets"].shape),
        },
    )

    model = CrossModalVAModel(
        config=CrossModalVAConfig(
            temporal_model="gru",
            shared_embedding_dim=128,
            audio_hidden_dim=128,
            visual_hidden_dim=128,
            temporal_hidden_dim=128,
            regressor_hidden_dim=64,
        )
    ).to(device)
    model.train()

    audio_batch = deam_sample["audio"].unsqueeze(0).to(device)
    audio_target = deam_sample["target"].unsqueeze(0).to(device)
    video_batch = veatic_sample["video"].unsqueeze(0).to(device)
    paired_audio_batch = veatic_sample["paired_audio"].unsqueeze(0).to(device)
    video_target = veatic_sample["target"].unsqueeze(0).to(device)

    outputs = model(
        audio=audio_batch,
        paired_audio=paired_audio_batch,
        paired_video=video_batch,
    )
    print("output_keys", sorted(outputs.keys()))

    losses = cross_modal_training_loss(
        outputs,
        audio_targets=audio_target,
        visual_targets=video_target,
        weights=LossWeights(lambda_align=1.0, lambda_visual=1.0, lambda_ts=0.0),
    )
    printable_losses = {key: float(value.detach().cpu()) for key, value in losses.items()}
    print("losses", printable_losses)

    losses["total"].backward()
    grad_norm = first_grad_norm(model)
    print(f"first_grad_norm={grad_norm:.6f}")

    fusion_model = LateFusionVAModel(
        config=CrossModalVAConfig(
            temporal_model="gru",
            shared_embedding_dim=128,
            audio_hidden_dim=128,
            visual_hidden_dim=128,
            temporal_hidden_dim=128,
            regressor_hidden_dim=64,
        )
    ).to(device)
    fusion_outputs = fusion_model(audio=paired_audio_batch, video=video_batch)
    fusion_loss = regression_loss(video_target, fusion_outputs["fusion_prediction"])
    fusion_loss.backward()
    print(f"late_fusion_loss={float(fusion_loss.detach().cpu()):.6f}")

    dann_model = DomainAdversarialVAModel(
        config=CrossModalVAConfig(
            temporal_model="gru",
            shared_embedding_dim=128,
            audio_hidden_dim=128,
            visual_hidden_dim=128,
            temporal_hidden_dim=128,
            regressor_hidden_dim=64,
        )
    ).to(device)
    dann_outputs = dann_model(paired_audio=paired_audio_batch, paired_video=video_batch, grl_lambda=1.0)
    dann_visual_loss = regression_loss(video_target, dann_outputs["paired_video_prediction"])
    dann_domain_loss = domain_adversarial_loss(
        dann_outputs["paired_audio_domain_logits"],
        dann_outputs["paired_video_domain_logits"],
    )
    dann_total = dann_visual_loss + dann_domain_loss
    dann_total.backward()
    print(f"dann_total_loss={float(dann_total.detach().cpu()):.6f}")
    print("smoke_test=passed")


def first_grad_norm(model: torch.nn.Module) -> float:
    for parameter in model.parameters():
        if parameter.grad is not None:
            return float(parameter.grad.norm().detach().cpu())
    return 0.0


if __name__ == "__main__":
    main()
