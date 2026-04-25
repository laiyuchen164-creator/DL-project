from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor
import torch.nn.functional as F


class AudioFeatureTransform:
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 64,
        n_fft: int = 1024,
        hop_length: int = 256,
        target_num_frames: Optional[int] = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.target_num_frames = target_num_frames

    def __call__(self, waveform: Tensor) -> Tensor:
        if waveform.ndim == 2:
            waveform = waveform.mean(dim=0)
        waveform = waveform.float()
        if waveform.numel() == 0:
            waveform = torch.zeros(self.n_fft, dtype=torch.float32)
        elif waveform.numel() < self.n_fft:
            waveform = F.pad(waveform, (0, self.n_fft - waveform.numel()))

        spectrogram = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            return_complex=True,
        )
        power = spectrogram.abs().pow(2)
        power = torch.log1p(power)

        if self.target_num_frames is not None:
            power = _pad_or_trim_2d(power, self.target_num_frames)

        return power.unsqueeze(0)

    def cache_key(self) -> str:
        target_frames = self.target_num_frames if self.target_num_frames is not None else "none"
        return (
            f"sr{self.sample_rate}_fft{self.n_fft}_hop{self.hop_length}"
            f"_mels{self.n_mels}_frames{target_frames}"
        )


class VideoFrameTransform:
    def __init__(self, size: int = 224) -> None:
        self.size = size

    def __call__(self, frames: Tensor) -> Tensor:
        if frames.dtype != torch.float32:
            frames = frames.float() / 255.0
        frames = F.interpolate(
            frames,
            size=(self.size, self.size),
            mode="bilinear",
            align_corners=False,
        )
        return frames

    def cache_key(self) -> str:
        return f"size{self.size}"


def _pad_or_trim_2d(features: Tensor, target_num_frames: int) -> Tensor:
    current_frames = features.size(-1)
    if current_frames == target_num_frames:
        return features
    if current_frames > target_num_frames:
        return features[..., :target_num_frames]
    pad_amount = target_num_frames - current_frames
    return F.pad(features, (0, pad_amount))
