from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import Tensor, nn

from .cross_modal_va import SimpleFrameCNN, SharedVARegressor


@dataclass
class LeaderFollowerConfig:
    visual_hidden_dim: int = 256
    audio_hidden_dim: int = 256
    fusion_hidden_dim: int = 256
    regressor_hidden_dim: int = 128
    dropout: float = 0.1
    audio_kernels: tuple[int, ...] = (3, 5, 7)
    attention_heads: int = 4


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, inputs: Tensor) -> Tensor:
        if self.chomp_size == 0:
            return inputs
        return inputs[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs: Tensor) -> Tensor:
        residual = inputs if self.downsample is None else self.downsample(inputs)
        outputs = self.net(inputs)
        return self.activation(outputs + residual)


class TemporalConvNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            TemporalBlock(input_dim, hidden_dim, kernel_size=kernel_size, dilation=1, dropout=dropout),
            TemporalBlock(hidden_dim, hidden_dim, kernel_size=kernel_size, dilation=2, dropout=dropout),
            TemporalBlock(hidden_dim, output_dim, kernel_size=kernel_size, dilation=4, dropout=dropout),
        )

    def forward(self, sequence: Tensor) -> Tensor:
        # [B, T, D] -> [B, D, T] -> [B, T, D]
        outputs = self.network(sequence.transpose(1, 2))
        return outputs.transpose(1, 2)


class VisualTCNEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.frame_backbone = SimpleFrameCNN(
            in_channels=3,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            dropout=dropout,
        )
        self.temporal = TemporalConvNet(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            kernel_size=3,
            dropout=dropout,
        )

    def forward(self, video: Tensor) -> Tensor:
        batch_size, num_frames, channels, height, width = video.shape
        frame_features = self.frame_backbone(video.view(batch_size * num_frames, channels, height, width))
        frame_features = frame_features.view(batch_size, num_frames, -1)
        return self.temporal(frame_features)


class AudioParallelTCNEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 256,
        kernels: tuple[int, ...] = (3, 5, 7),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.branches = nn.ModuleList(
            [
                TemporalConvNet(
                    input_dim=hidden_dim,
                    hidden_dim=hidden_dim,
                    output_dim=output_dim,
                    kernel_size=kernel,
                    dropout=dropout,
                )
                for kernel in kernels
            ]
        )
        self.output_projection = nn.Sequential(
            nn.Linear(output_dim * len(kernels), output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, audio: Tensor, target_steps: int) -> Tensor:
        # audio: [B, 1, F, T] -> [B, T, F]
        sequence = audio.squeeze(1).transpose(1, 2)
        sequence = self.input_projection(sequence)
        branch_outputs = [branch(sequence) for branch in self.branches]
        fused = torch.cat(branch_outputs, dim=-1)
        fused = self.output_projection(fused)
        if fused.size(1) == target_steps:
            return fused
        fused = fused.transpose(1, 2)
        fused = torch.nn.functional.interpolate(fused, size=target_steps, mode="linear", align_corners=False)
        return fused.transpose(1, 2)


class LeaderFollowerAttentiveFusionModel(nn.Module):
    def __init__(self, config: LeaderFollowerConfig | None = None, audio_input_dim: int = 513) -> None:
        super().__init__()
        self.config = config or LeaderFollowerConfig()
        self.visual_encoder = VisualTCNEncoder(
            hidden_dim=self.config.visual_hidden_dim,
            dropout=self.config.dropout,
        )
        self.audio_encoder = AudioParallelTCNEncoder(
            input_dim=audio_input_dim,
            hidden_dim=self.config.audio_hidden_dim,
            output_dim=self.config.fusion_hidden_dim,
            kernels=self.config.audio_kernels,
            dropout=self.config.dropout,
        )
        self.visual_projection = nn.Linear(self.config.visual_hidden_dim, self.config.fusion_hidden_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=self.config.fusion_hidden_dim,
            num_heads=self.config.attention_heads,
            dropout=self.config.dropout,
            batch_first=True,
        )
        self.fusion = nn.Sequential(
            nn.Linear(self.config.fusion_hidden_dim * 2, self.config.fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.config.dropout),
        )
        self.regressor = SharedVARegressor(
            input_dim=self.config.fusion_hidden_dim,
            hidden_dim=self.config.regressor_hidden_dim,
            dropout=self.config.dropout,
        )

    def forward(self, audio: Tensor, video: Tensor) -> Dict[str, Tensor]:
        visual_sequence = self.visual_encoder(video)
        visual_query = self.visual_projection(visual_sequence)
        audio_sequence = self.audio_encoder(audio, target_steps=visual_query.size(1))
        attended_audio, attention_weights = self.attention(
            query=visual_query,
            key=audio_sequence,
            value=audio_sequence,
        )
        fused_sequence = self.fusion(torch.cat([visual_query, attended_audio], dim=-1))
        pooled_embedding = fused_sequence.mean(dim=1)
        return {
            "prediction": self.regressor(pooled_embedding),
            "attention_weights": attention_weights,
            "embedding": pooled_embedding,
        }
