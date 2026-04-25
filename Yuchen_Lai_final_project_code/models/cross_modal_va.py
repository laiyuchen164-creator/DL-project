from __future__ import annotations
# Submission variant note:
# This packaged copy emphasizes experiment orchestration, documentation, and figure scripts.
# Package owner: Yuchen Lai


from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from transformers import VideoMAEForVideoClassification
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models.video import MViT_V2_S_Weights, R2Plus1D_18_Weights, mvit_v2_s, r2plus1d_18


@dataclass
class CrossModalVAConfig:
    audio_in_channels: int = 1
    visual_in_channels: int = 3
    audio_hidden_dim: int = 256
    visual_hidden_dim: int = 256
    shared_embedding_dim: int = 256
    temporal_hidden_dim: int = 256
    regressor_hidden_dim: int = 128
    temporal_model: str = "gru"
    temporal_layers: int = 1
    transformer_heads: int = 4
    dropout: float = 0.1


class SimpleAudioCNN(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        hidden_dim: int = 256,
        output_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.projection(self.features(x))


class AudioEncoder(nn.Module):
    def __init__(
        self,
        backbone: Optional[nn.Module] = None,
        in_channels: int = 1,
        hidden_dim: int = 256,
        output_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = backbone or SimpleAudioCNN(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
        )
        self.output_dim = output_dim

    def forward(self, audio: Tensor) -> Tensor:
        return self.backbone(audio)


class SimpleFrameCNN(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 256,
        output_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, frames: Tensor) -> Tensor:
        return self.projection(self.features(frames))


class ResNet18FrameCNN(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        output_dim: int = 256,
        dropout: float = 0.1,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.register_buffer(
            "pixel_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        if freeze_backbone:
            for parameter in self.features.parameters():
                parameter.requires_grad = False

    def forward(self, frames: Tensor) -> Tensor:
        normalized = (frames - self.pixel_mean) / self.pixel_std
        features = self.features(normalized)
        return self.projection(features)


class R2Plus1D18VideoBackbone(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        output_dim: int = 256,
        dropout: float = 0.1,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        weights = R2Plus1D_18_Weights.DEFAULT if pretrained else None
        backbone = r2plus1d_18(weights=weights)
        self.stem = backbone.stem
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool
        self.register_buffer(
            "pixel_mean",
            torch.tensor([0.43216, 0.394666, 0.37645], dtype=torch.float32).view(1, 3, 1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor([0.22803, 0.22145, 0.216989], dtype=torch.float32).view(1, 3, 1, 1, 1),
            persistent=False,
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        if freeze_backbone:
            for module in [self.stem, self.layer1, self.layer2, self.layer3, self.layer4]:
                for parameter in module.parameters():
                    parameter.requires_grad = False

    def forward(self, video: Tensor) -> Tensor:
        clip = video.permute(0, 2, 1, 3, 4).contiguous()
        clip = (clip - self.pixel_mean) / self.pixel_std
        features = self.stem(clip)
        features = self.layer1(features)
        features = self.layer2(features)
        features = self.layer3(features)
        features = self.layer4(features)
        pooled = self.avgpool(features)
        return self.projection(pooled)


class VideoMAEBaseVideoBackbone(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        output_dim: int = 256,
        dropout: float = 0.1,
        pretrained_model_name: str = "MCG-NJU/videomae-base-finetuned-kinetics",
        freeze_backbone: bool = True,
        trainable_blocks: int = 0,
    ) -> None:
        super().__init__()
        model = VideoMAEForVideoClassification.from_pretrained(pretrained_model_name)
        self.backbone = model.videomae
        self.fc_norm = model.fc_norm
        self.register_buffer(
            "pixel_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.num_frames = int(model.config.num_frames)
        self.image_size = int(model.config.image_size)
        self.trainable_blocks = max(0, int(trainable_blocks))
        self.projection = nn.Sequential(
            nn.Linear(model.config.hidden_size, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
            for parameter in self.fc_norm.parameters():
                parameter.requires_grad = False
            if self.trainable_blocks > 0:
                for layer in self.backbone.encoder.layer[-self.trainable_blocks :]:
                    for parameter in layer.parameters():
                        parameter.requires_grad = True
                for parameter in self.fc_norm.parameters():
                    parameter.requires_grad = True

    def forward(self, video: Tensor) -> Tensor:
        clip = (video - self.pixel_mean) / self.pixel_std
        fully_frozen = self.freeze_backbone and self.trainable_blocks == 0
        if fully_frozen:
            with torch.no_grad():
                outputs = self.backbone(pixel_values=clip)
                pooled = self.fc_norm(outputs.last_hidden_state.mean(dim=1))
        else:
            outputs = self.backbone(pixel_values=clip)
            pooled = self.fc_norm(outputs.last_hidden_state.mean(dim=1))
        return self.projection(pooled)


class MViTV2SVideoBackbone(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        output_dim: int = 256,
        dropout: float = 0.1,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        weights = MViT_V2_S_Weights.DEFAULT if pretrained else None
        backbone = mvit_v2_s(weights=weights)
        backbone.head = nn.Identity()
        self.backbone = backbone
        self.register_buffer(
            "pixel_mean",
            torch.tensor([0.45, 0.45, 0.45], dtype=torch.float32).view(1, 3, 1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor([0.225, 0.225, 0.225], dtype=torch.float32).view(1, 3, 1, 1, 1),
            persistent=False,
        )
        self.projection = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

    def forward(self, video: Tensor) -> Tensor:
        clip = video.permute(0, 2, 1, 3, 4).contiguous()
        clip = (clip - self.pixel_mean) / self.pixel_std
        features = self.backbone(clip)
        return self.projection(features)


class TemporalEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        model_type: str = "gru",
        layers: int = 1,
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.model_type = model_type.lower()
        if self.model_type == "gru":
            self.sequence_model: nn.Module = nn.GRU(
                input_dim,
                hidden_dim,
                num_layers=layers,
                batch_first=True,
                dropout=dropout if layers > 1 else 0.0,
                bidirectional=False,
            )
            temporal_out_dim = hidden_dim
        elif self.model_type == "transformer":
            layer = nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            self.sequence_model = nn.TransformerEncoder(layer, num_layers=layers)
            temporal_out_dim = input_dim
        else:
            raise ValueError(f"Unsupported temporal model: {model_type}")

        self.output = nn.Sequential(
            nn.Linear(temporal_out_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, frame_features: Tensor) -> Tensor:
        if self.model_type == "gru":
            _, hidden = self.sequence_model(frame_features)
            pooled = hidden[-1]
        else:
            encoded = self.sequence_model(frame_features)
            pooled = encoded.mean(dim=1)
        return self.output(pooled)


class AudioSequenceEncoder(nn.Module):
    def __init__(self, output_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.input_projection = nn.LazyLinear(output_dim)
        self.block = nn.Sequential(
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, audio: Tensor, target_steps: int) -> Tensor:
        sequence = audio.squeeze(1).transpose(1, 2)
        sequence = self.input_projection(sequence)
        sequence = self.block(sequence)
        if sequence.size(1) != target_steps:
            sequence = F.interpolate(
                sequence.transpose(1, 2),
                size=target_steps,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        return sequence


class TemporalGatedCrossAttention(nn.Module):
    def __init__(self, embedding_dim: int = 256, heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.video_queries_audio = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.audio_queries_video = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.delta = nn.Sequential(
            nn.Linear(embedding_dim * 3, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(embedding_dim * 3, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, video_sequence: Tensor, audio_sequence: Tensor) -> Tensor:
        audio_context, _ = self.video_queries_audio(
            query=video_sequence,
            key=audio_sequence,
            value=audio_sequence,
            need_weights=False,
        )
        video_feedback, _ = self.audio_queries_video(
            query=audio_sequence,
            key=video_sequence,
            value=video_sequence,
            need_weights=False,
        )
        fusion_features = torch.cat([video_sequence, audio_context, video_feedback], dim=-1)
        gated_delta = self.gate(fusion_features) * self.delta(fusion_features)
        return self.norm(video_sequence + gated_delta)


class VisualEncoder(nn.Module):
    def __init__(
        self,
        frame_backbone: Optional[nn.Module] = None,
        video_backbone: Optional[nn.Module] = None,
        in_channels: int = 3,
        frame_dim: int = 256,
        temporal_hidden_dim: int = 256,
        output_dim: int = 256,
        temporal_model: str = "gru",
        temporal_layers: int = 1,
        transformer_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.video_backbone = video_backbone
        self.frame_backbone = frame_backbone or SimpleFrameCNN(
            in_channels=in_channels,
            hidden_dim=frame_dim,
            output_dim=frame_dim,
            dropout=dropout,
        )
        self.temporal_encoder = None
        if self.video_backbone is None:
            self.temporal_encoder = TemporalEncoder(
                input_dim=frame_dim,
                hidden_dim=temporal_hidden_dim,
                output_dim=output_dim,
                model_type=temporal_model,
                layers=temporal_layers,
                heads=transformer_heads,
                dropout=dropout,
            )

    def extract_frame_features(self, video: Tensor) -> Tensor:
        if self.video_backbone is not None:
            raise RuntimeError("Frame features are not available when using a spatiotemporal video backbone.")
        batch_size, num_frames, channels, height, width = video.shape
        flattened_frames = video.view(batch_size * num_frames, channels, height, width)
        frame_features = self.frame_backbone(flattened_frames)
        return frame_features.view(batch_size, num_frames, -1)

    def forward(self, video: Tensor) -> Tensor:
        if self.video_backbone is not None:
            return self.video_backbone(video)
        frame_features = self.extract_frame_features(video)
        if self.temporal_encoder is None:
            raise RuntimeError("Temporal encoder is missing for frame-based visual encoding.")
        return self.temporal_encoder(frame_features)


class SharedVARegressor(nn.Module):
    def __init__(
        self,
        input_dim: int = 256,
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

    def forward(self, embedding: Tensor) -> Tensor:
        return self.mlp(embedding)


class CrossModalVAModel(nn.Module):
    def __init__(
        self,
        config: Optional[CrossModalVAConfig] = None,
        audio_backbone: Optional[nn.Module] = None,
        frame_backbone: Optional[nn.Module] = None,
        video_backbone: Optional[nn.Module] = None,
        joint_fusion: str = "none",
        teacher: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.config = config or CrossModalVAConfig()
        self.joint_fusion = joint_fusion
        self.audio_encoder = AudioEncoder(
            backbone=audio_backbone,
            in_channels=self.config.audio_in_channels,
            hidden_dim=self.config.audio_hidden_dim,
            output_dim=self.config.shared_embedding_dim,
            dropout=self.config.dropout,
        )
        self.visual_encoder = VisualEncoder(
            frame_backbone=frame_backbone,
            video_backbone=video_backbone,
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
        self.audio_sequence_encoder = None
        self.temporal_fusion = None
        if self.joint_fusion == "tagf_lite":
            if video_backbone is not None:
                raise ValueError("joint_fusion=tagf_lite is not supported with a spatiotemporal video backbone.")
            self.audio_sequence_encoder = AudioSequenceEncoder(
                output_dim=self.config.visual_hidden_dim,
                dropout=self.config.dropout,
            )
            self.temporal_fusion = TemporalGatedCrossAttention(
                embedding_dim=self.config.visual_hidden_dim,
                heads=self.config.transformer_heads,
                dropout=self.config.dropout,
            )
        self.teacher = teacher

    def encode_audio(self, audio: Tensor, normalize: bool = True) -> Tensor:
        embedding = self.audio_encoder(audio)
        return F.normalize(embedding, dim=-1) if normalize else embedding

    def encode_video(self, video: Tensor, normalize: bool = True) -> Tensor:
        embedding = self.visual_encoder(video)
        return F.normalize(embedding, dim=-1) if normalize else embedding

    def predict_from_audio(self, audio: Tensor) -> Dict[str, Tensor]:
        raw_embedding = self.audio_encoder(audio)
        return {
            "embedding": F.normalize(raw_embedding, dim=-1),
            "prediction": self.regressor(raw_embedding),
        }

    def predict_from_video(self, video: Tensor) -> Dict[str, Tensor]:
        raw_embedding = self.visual_encoder(video)
        return {
            "embedding": F.normalize(raw_embedding, dim=-1),
            "prediction": self.regressor(raw_embedding),
        }

    def forward(
        self,
        audio: Optional[Tensor] = None,
        video: Optional[Tensor] = None,
        paired_audio: Optional[Tensor] = None,
        paired_video: Optional[Tensor] = None,
        teacher_audio: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        outputs: Dict[str, Tensor] = {}

        if audio is not None:
            audio_out = self.predict_from_audio(audio)
            outputs["audio_embedding"] = audio_out["embedding"]
            outputs["audio_prediction"] = audio_out["prediction"]

        if video is not None:
            video_out = self.predict_from_video(video)
            outputs["video_embedding"] = video_out["embedding"]
            outputs["video_prediction"] = video_out["prediction"]

        if paired_audio is not None and paired_video is not None:
            paired_audio_raw = self.audio_encoder(paired_audio)
            if self.joint_fusion == "tagf_lite":
                if self.audio_sequence_encoder is None or self.temporal_fusion is None:
                    raise RuntimeError("TAGF-lite joint fusion was requested but fusion modules are missing.")
                video_sequence = self.visual_encoder.extract_frame_features(paired_video)
                audio_sequence = self.audio_sequence_encoder(
                    paired_audio,
                    target_steps=video_sequence.size(1),
                )
                fused_video_sequence = self.temporal_fusion(video_sequence, audio_sequence)
                paired_video_raw = self.visual_encoder.temporal_encoder(fused_video_sequence)
            else:
                paired_video_raw = self.visual_encoder(paired_video)
            outputs["paired_audio_embedding"] = F.normalize(paired_audio_raw, dim=-1)
            outputs["paired_video_embedding"] = F.normalize(paired_video_raw, dim=-1)
            outputs["paired_video_prediction"] = self.regressor(paired_video_raw)

            if self.teacher is not None and teacher_audio is not None:
                with torch.no_grad():
                    outputs["teacher_prediction"] = self.teacher(teacher_audio)

        return outputs
