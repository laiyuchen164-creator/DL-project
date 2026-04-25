from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch
from torch import Tensor, nn
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)


DEFAULT_AUDEERING_DIM_REPO = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
DEFAULT_TEACHER_SAMPLE_RATE = 16000


class RegressionHead(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features: Tensor) -> Tensor:
        outputs = self.dropout(features)
        outputs = self.dense(outputs)
        outputs = torch.tanh(outputs)
        outputs = self.dropout(outputs)
        return self.out_proj(outputs)


class Wav2Vec2ForSpeechClassification(Wav2Vec2PreTrainedModel):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.post_init()

    def forward(
        self,
        input_values: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        outputs = self.wav2vec2(input_values=input_values, attention_mask=attention_mask)
        hidden_states = outputs[0]
        pooled_states = hidden_states.mean(dim=1)
        logits = self.classifier(pooled_states)
        return pooled_states, logits


class TeacherCalibrationHead(nn.Linear):
    def __init__(self) -> None:
        super().__init__(3, 2)
        self.reset_parameters_to_default_mapping()

    def reset_parameters_to_default_mapping(self) -> None:
        with torch.no_grad():
            self.weight.zero_()
            self.bias.zero_()
            # Teacher order: arousal, dominance, valence in ~[0, 1].
            # Project to project label order: valence, arousal in ~[-1, 1].
            self.weight[0, 2] = 2.0
            self.bias[0] = -1.0
            self.weight[1, 0] = 2.0
            self.bias[1] = -1.0


class ExternalAudeeringDimTeacher(nn.Module):
    def __init__(
        self,
        checkpoint_path: Optional[str | Path] = None,
        repo_id: str = DEFAULT_AUDEERING_DIM_REPO,
    ) -> None:
        super().__init__()
        self.repo_id = repo_id
        self.processor = Wav2Vec2Processor.from_pretrained(repo_id)
        self.backbone = Wav2Vec2ForSpeechClassification.from_pretrained(repo_id)
        self.calibration = TeacherCalibrationHead()
        self.freeze_backbone()
        if checkpoint_path is not None:
            self.load_calibration_checkpoint(checkpoint_path)
        self.freeze_calibration()

    def freeze_backbone(self) -> None:
        self.backbone.eval()
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def freeze_calibration(self) -> None:
        self.calibration.eval()
        for parameter in self.calibration.parameters():
            parameter.requires_grad = False

    def load_calibration_checkpoint(self, checkpoint_path: str | Path) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("calibration_state_dict")
        if state_dict is None:
            raise ValueError(f"Calibration checkpoint missing 'calibration_state_dict': {checkpoint_path}")
        self.calibration.load_state_dict(state_dict)

    @torch.no_grad()
    def predict_raw_dimensions(self, audio_batch: Tensor) -> Tensor:
        if audio_batch.ndim == 1:
            audio_batch = audio_batch.unsqueeze(0)
        if audio_batch.ndim == 3 and audio_batch.size(1) == 1:
            audio_batch = audio_batch[:, 0, :]
        if audio_batch.ndim != 2:
            raise ValueError(f"Expected audio batch with shape [B, T], got {tuple(audio_batch.shape)}")

        inputs = self.processor(
            [sample.detach().cpu().numpy() for sample in audio_batch],
            sampling_rate=DEFAULT_TEACHER_SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs["input_values"].to(audio_batch.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(audio_batch.device)

        _, logits = self.backbone(input_values=input_values, attention_mask=attention_mask)
        return logits

    @torch.no_grad()
    def forward(self, audio_batch: Tensor) -> Tensor:
        raw_dimensions = self.predict_raw_dimensions(audio_batch)
        return self.calibration(raw_dimensions)


def build_external_teacher(
    backend: Optional[str],
    checkpoint_path: Optional[str | Path],
) -> Optional[nn.Module]:
    if backend is None:
        return None
    if backend != "external_audeering_dim":
        raise ValueError(f"Unsupported teacher backend: {backend}")
    if checkpoint_path is None:
        raise ValueError("External teacher requires --teacher-checkpoint.")
    return ExternalAudeeringDimTeacher(checkpoint_path=checkpoint_path)


def load_calibration_head(checkpoint_path: str | Path) -> TeacherCalibrationHead:
    head = TeacherCalibrationHead()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("calibration_state_dict")
    if state_dict is None:
        raise ValueError(f"Calibration checkpoint missing 'calibration_state_dict': {checkpoint_path}")
    head.load_state_dict(state_dict)
    return head


def extract_teacher_metadata(checkpoint_path: str | Path) -> Dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return {
        "repo_id": checkpoint.get("repo_id", DEFAULT_AUDEERING_DIM_REPO),
        "teacher_sample_rate": checkpoint.get("teacher_sample_rate", DEFAULT_TEACHER_SAMPLE_RATE),
        "teacher_audio_seconds": checkpoint.get("teacher_audio_seconds"),
        "metrics": checkpoint.get("metrics"),
    }
