from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .cross_modal_va import (
    AudioEncoder,
    AudioSequenceEncoder,
    CrossModalVAConfig,
    SharedVARegressor,
    TemporalGatedCrossAttention,
    VisualEncoder,
)


class LateFusionRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, fused_embedding: Tensor) -> Tensor:
        return self.mlp(fused_embedding)


class LateFusionVAModel(nn.Module):
    def __init__(
        self,
        config: Optional[CrossModalVAConfig] = None,
        audio_backbone: Optional[nn.Module] = None,
        frame_backbone: Optional[nn.Module] = None,
        fusion_mode: str = "concat",
    ) -> None:
        super().__init__()
        self.config = config or CrossModalVAConfig()
        self.fusion_mode = fusion_mode
        self.audio_encoder = AudioEncoder(
            backbone=audio_backbone,
            in_channels=self.config.audio_in_channels,
            hidden_dim=self.config.audio_hidden_dim,
            output_dim=self.config.shared_embedding_dim,
            dropout=self.config.dropout,
        )
        self.visual_encoder = VisualEncoder(
            frame_backbone=frame_backbone,
            in_channels=self.config.visual_in_channels,
            frame_dim=self.config.visual_hidden_dim,
            temporal_hidden_dim=self.config.temporal_hidden_dim,
            output_dim=self.config.shared_embedding_dim,
            temporal_model=self.config.temporal_model,
            temporal_layers=self.config.temporal_layers,
            transformer_heads=self.config.transformer_heads,
            dropout=self.config.dropout,
        )
        self.audio_sequence_encoder = None
        self.temporal_fusion = None
        if self.fusion_mode == "tagf_lite":
            self.audio_sequence_encoder = AudioSequenceEncoder(
                output_dim=self.config.visual_hidden_dim,
                dropout=self.config.dropout,
            )
            self.temporal_fusion = TemporalGatedCrossAttention(
                embedding_dim=self.config.visual_hidden_dim,
                heads=self.config.transformer_heads,
                dropout=self.config.dropout,
            )
        self.regressor = LateFusionRegressor(
            input_dim=self.config.shared_embedding_dim * 2,
            hidden_dim=self.config.regressor_hidden_dim,
            dropout=self.config.dropout,
        )

    def forward(self, audio: Tensor, video: Tensor) -> Dict[str, Tensor]:
        audio_raw = self.audio_encoder(audio)
        if self.fusion_mode == "tagf_lite":
            if self.audio_sequence_encoder is None or self.temporal_fusion is None:
                raise RuntimeError("Late-fusion TAGF-lite mode is missing sequence fusion modules.")
            video_sequence = self.visual_encoder.extract_frame_features(video)
            audio_sequence = self.audio_sequence_encoder(audio, target_steps=video_sequence.size(1))
            fused_video_sequence = self.temporal_fusion(video_sequence, audio_sequence)
            video_raw = self.visual_encoder.temporal_encoder(fused_video_sequence)
        else:
            video_raw = self.visual_encoder(video)
        fused = torch.cat([audio_raw, video_raw], dim=-1)
        return {
            "audio_embedding": F.normalize(audio_raw, dim=-1),
            "video_embedding": F.normalize(video_raw, dim=-1),
            "fusion_prediction": self.regressor(fused),
        }


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: Tensor, lambda_: float) -> Tensor:
        ctx.lambda_ = lambda_
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, None]:
        return -ctx.lambda_ * grad_output, None


def gradient_reverse(inputs: Tensor, lambda_: float = 1.0) -> Tensor:
    return GradientReversalFunction.apply(inputs, lambda_)


class DomainClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        output_dim: int = 2,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, embedding: Tensor) -> Tensor:
        return self.mlp(embedding)


class DomainAdversarialVAModel(nn.Module):
    def __init__(
        self,
        config: Optional[CrossModalVAConfig] = None,
        audio_backbone: Optional[nn.Module] = None,
        frame_backbone: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.config = config or CrossModalVAConfig()
        self.audio_encoder = AudioEncoder(
            backbone=audio_backbone,
            in_channels=self.config.audio_in_channels,
            hidden_dim=self.config.audio_hidden_dim,
            output_dim=self.config.shared_embedding_dim,
            dropout=self.config.dropout,
        )
        self.visual_encoder = VisualEncoder(
            frame_backbone=frame_backbone,
            in_channels=self.config.visual_in_channels,
            frame_dim=self.config.visual_hidden_dim,
            temporal_hidden_dim=self.config.temporal_hidden_dim,
            output_dim=self.config.shared_embedding_dim,
            temporal_model=self.config.temporal_model,
            temporal_layers=self.config.temporal_layers,
            transformer_heads=self.config.transformer_heads,
            dropout=self.config.dropout,
        )
        self.regressor = SharedVARegressor(
            input_dim=self.config.shared_embedding_dim,
            hidden_dim=self.config.regressor_hidden_dim,
            dropout=self.config.dropout,
        )
        self.domain_classifier = DomainClassifier(
            input_dim=self.config.shared_embedding_dim,
            hidden_dim=self.config.regressor_hidden_dim,
            dropout=self.config.dropout,
        )

    def forward(
        self,
        audio: Optional[Tensor] = None,
        video: Optional[Tensor] = None,
        paired_audio: Optional[Tensor] = None,
        paired_video: Optional[Tensor] = None,
        grl_lambda: float = 1.0,
    ) -> Dict[str, Tensor]:
        outputs: Dict[str, Tensor] = {}

        if audio is not None:
            audio_raw = self.audio_encoder(audio)
            outputs["audio_embedding"] = F.normalize(audio_raw, dim=-1)
            outputs["audio_prediction"] = self.regressor(audio_raw)

        if video is not None:
            video_raw = self.visual_encoder(video)
            outputs["video_embedding"] = F.normalize(video_raw, dim=-1)
            outputs["video_prediction"] = self.regressor(video_raw)

        if paired_audio is not None and paired_video is not None:
            paired_audio_raw = self.audio_encoder(paired_audio)
            paired_video_raw = self.visual_encoder(paired_video)
            outputs["paired_audio_embedding"] = F.normalize(paired_audio_raw, dim=-1)
            outputs["paired_video_embedding"] = F.normalize(paired_video_raw, dim=-1)
            outputs["paired_video_prediction"] = self.regressor(paired_video_raw)
            outputs["paired_audio_domain_logits"] = self.domain_classifier(
                gradient_reverse(paired_audio_raw, grl_lambda)
            )
            outputs["paired_video_domain_logits"] = self.domain_classifier(
                gradient_reverse(paired_video_raw, grl_lambda)
            )

        return outputs
