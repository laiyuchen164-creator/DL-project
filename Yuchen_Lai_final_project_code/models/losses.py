from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import Tensor
import torch.nn.functional as F


@dataclass
class LossWeights:
    alpha: float = 0.5
    lambda_align: float = 1.0
    lambda_ts: float = 1.0
    lambda_visual: float = 1.0
    temperature: float = 0.07


def concordance_correlation_coefficient(target: Tensor, prediction: Tensor, eps: float = 1e-8) -> Tensor:
    if target.ndim == 1:
        target = target.unsqueeze(-1)
    if prediction.ndim == 1:
        prediction = prediction.unsqueeze(-1)

    target_mean = target.mean(dim=0)
    prediction_mean = prediction.mean(dim=0)
    target_var = target.var(dim=0, unbiased=False)
    prediction_var = prediction.var(dim=0, unbiased=False)

    covariance = ((target - target_mean) * (prediction - prediction_mean)).mean(dim=0)
    numerator = 2.0 * covariance
    denominator = target_var + prediction_var + (target_mean - prediction_mean).pow(2) + eps
    return numerator / denominator


def ccc_loss(target: Tensor, prediction: Tensor) -> Tensor:
    return 1.0 - concordance_correlation_coefficient(target, prediction).mean()


def regression_loss(target: Tensor, prediction: Tensor, alpha: float = 0.5) -> Tensor:
    l1 = F.l1_loss(prediction, target)
    return alpha * l1 + (1.0 - alpha) * ccc_loss(target, prediction)


def symmetric_info_nce_loss(audio_embeddings: Tensor, video_embeddings: Tensor, temperature: float = 0.07) -> Tensor:
    logits = audio_embeddings @ video_embeddings.transpose(0, 1)
    logits = logits / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    audio_to_video = F.cross_entropy(logits, labels)
    video_to_audio = F.cross_entropy(logits.transpose(0, 1), labels)
    return 0.5 * (audio_to_video + video_to_audio)


def domain_adversarial_loss(audio_logits: Tensor, video_logits: Tensor) -> Tensor:
    audio_labels = torch.zeros(audio_logits.size(0), dtype=torch.long, device=audio_logits.device)
    video_labels = torch.ones(video_logits.size(0), dtype=torch.long, device=video_logits.device)
    audio_loss = F.cross_entropy(audio_logits, audio_labels)
    video_loss = F.cross_entropy(video_logits, video_labels)
    return 0.5 * (audio_loss + video_loss)


def cross_modal_training_loss(
    outputs: Dict[str, Tensor],
    audio_targets: Optional[Tensor] = None,
    visual_targets: Optional[Tensor] = None,
    weights: Optional[LossWeights] = None,
) -> Dict[str, Tensor]:
    weights = weights or LossWeights()
    device = next(iter(outputs.values())).device
    zero = torch.zeros((), device=device)

    losses: Dict[str, Tensor] = {
        "audio_supervised": zero,
        "visual_supervised": zero,
        "alignment": zero,
        "distillation": zero,
        "total": zero,
    }

    if audio_targets is not None and "audio_prediction" in outputs:
        losses["audio_supervised"] = regression_loss(
            audio_targets,
            outputs["audio_prediction"],
            alpha=weights.alpha,
        )

    visual_prediction = outputs.get("video_prediction", outputs.get("paired_video_prediction"))
    if visual_targets is not None and visual_prediction is not None:
        losses["visual_supervised"] = regression_loss(
            visual_targets,
            visual_prediction,
            alpha=weights.alpha,
        )

    if "paired_audio_embedding" in outputs and "paired_video_embedding" in outputs:
        losses["alignment"] = symmetric_info_nce_loss(
            outputs["paired_audio_embedding"],
            outputs["paired_video_embedding"],
            temperature=weights.temperature,
        )

    if "teacher_prediction" in outputs and "paired_video_prediction" in outputs:
        losses["distillation"] = F.l1_loss(
            outputs["paired_video_prediction"],
            outputs["teacher_prediction"],
        )

    losses["total"] = (
        losses["audio_supervised"]
        + weights.lambda_visual * losses["visual_supervised"]
        + weights.lambda_align * losses["alignment"]
        + weights.lambda_ts * losses["distillation"]
    )
    return losses
