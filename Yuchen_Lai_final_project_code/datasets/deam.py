from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .archive import ZipExtractor, load_tensor_cache, save_tensor_cache

try:
    import torchaudio
except ImportError:  # pragma: no cover
    torchaudio = None

try:
    import librosa
except ImportError:  # pragma: no cover
    librosa = None


@dataclass
class DEAMSample:
    song_id: str
    audio_path: str
    valence: List[float]
    arousal: List[float]


class DEAMDataset(Dataset[Dict[str, Tensor]]):
    def __init__(
        self,
        audio_root: Optional[str | Path] = None,
        audio_zip_path: Optional[str | Path] = None,
        annotations_dir: str | Path = "data/deam/annotations/annotations",
        transform=None,
        cache_dir: str | Path = "data/cache/deam",
        target_length_seconds: Optional[float] = 30.0,
        sample_rate: int = 16000,
    ) -> None:
        self.audio_root = Path(audio_root) if audio_root else None
        self.annotations_dir = Path(annotations_dir)
        self.transform = transform
        self.target_length_seconds = target_length_seconds
        self.sample_rate = sample_rate
        self.feature_cache_dir = Path(cache_dir) / "features"

        self.zip_extractor = None
        if self.audio_root is None and audio_zip_path is not None:
            self.zip_extractor = ZipExtractor(audio_zip_path, cache_dir)

        self.samples = self._build_samples()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        sample = self.samples[index]
        audio_tensor = self._load_or_build_audio_tensor(sample)
        target = torch.tensor(
            [
                float(sum(sample.valence) / len(sample.valence)),
                float(sum(sample.arousal) / len(sample.arousal)),
            ],
            dtype=torch.float32,
        )
        dynamic_target = torch.tensor(
            list(zip(sample.valence, sample.arousal)),
            dtype=torch.float32,
        )

        return {
            "audio": audio_tensor,
            "target": target,
            "dynamic_target": dynamic_target,
            "song_id": sample.song_id,
        }

    def _load_or_build_audio_tensor(self, sample: DEAMSample) -> Tensor:
        cache_path = self._audio_cache_path(sample.song_id)
        cached = load_tensor_cache(cache_path)
        if cached is not None:
            return cached["audio"]

        audio_path = self._resolve_audio_path(sample.audio_path)
        waveform = self._load_audio(audio_path)

        if self.target_length_seconds is not None:
            waveform = self._fit_waveform_length(waveform)

        audio_tensor = self.transform(waveform) if self.transform else waveform
        save_tensor_cache(cache_path, {"audio": audio_tensor.cpu()})
        return audio_tensor

    def _audio_cache_path(self, song_id: str) -> Path:
        transform_key = self.transform.cache_key() if self.transform is not None and hasattr(self.transform, "cache_key") else "raw"
        length_key = (
            f"len{self.target_length_seconds}".replace(".", "p")
            if self.target_length_seconds is not None
            else "lenfull"
        )
        return self.feature_cache_dir / f"{song_id}_{transform_key}_{length_key}.pt"

    def _build_samples(self) -> List[DEAMSample]:
        dynamic_dir = self.annotations_dir / "annotations averaged per song" / "dynamic (per second annotations)"
        valence_map = self._load_dynamic_annotations(dynamic_dir / "valence.csv")
        arousal_map = self._load_dynamic_annotations(dynamic_dir / "arousal.csv")
        audio_map = self._build_audio_index()

        samples: List[DEAMSample] = []
        for song_id in sorted(set(valence_map) & set(arousal_map) & set(audio_map)):
            valence = valence_map[song_id]
            arousal = arousal_map[song_id]
            length = min(len(valence), len(arousal))
            if length == 0:
                continue
            samples.append(
                DEAMSample(
                    song_id=song_id,
                    audio_path=audio_map[song_id],
                    valence=valence[:length],
                    arousal=arousal[:length],
                )
            )
        return samples

    def _build_audio_index(self) -> Dict[str, str]:
        if self.audio_root is not None:
            paths = list(self.audio_root.rglob("*.mp3"))
            return {path.stem: str(path) for path in paths}

        if self.zip_extractor is None:
            raise ValueError("Provide either audio_root or audio_zip_path for DEAMDataset.")

        import zipfile

        with zipfile.ZipFile(self.zip_extractor.zip_path) as archive:
            members = [name for name in archive.namelist() if name.lower().endswith(".mp3")]
        return {Path(name).stem: name for name in members}

    def _load_dynamic_annotations(self, csv_path: Path) -> Dict[str, List[float]]:
        annotations: Dict[str, List[float]] = {}
        with csv_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                song_id = row["song_id"]
                values = [
                    float(value)
                    for key, value in row.items()
                    if key != "song_id" and value not in (None, "")
                ]
                annotations[song_id] = values
        return annotations

    def _resolve_audio_path(self, stored_path: str) -> Path:
        if self.audio_root is not None:
            return Path(stored_path)
        assert self.zip_extractor is not None
        return self.zip_extractor.extract(stored_path)

    def _fit_waveform_length(self, waveform: Tensor) -> Tensor:
        target_samples = int(self.target_length_seconds * self.sample_rate)
        current_samples = waveform.size(-1)
        if current_samples == target_samples:
            return waveform
        if current_samples > target_samples:
            return waveform[..., :target_samples]

        pad_amount = target_samples - current_samples
        return torch.nn.functional.pad(waveform, (0, pad_amount))

    def _load_audio(self, audio_path: Path) -> Tensor:
        if torchaudio is not None:
            try:
                waveform, source_sample_rate = torchaudio.load(str(audio_path))
                if source_sample_rate != self.sample_rate:
                    waveform = torchaudio.functional.resample(
                        waveform,
                        orig_freq=source_sample_rate,
                        new_freq=self.sample_rate,
                    )
                return waveform
            except (RuntimeError, ImportError, OSError):
                pass

        if librosa is None:
            raise ImportError(
                "Unable to decode audio with torchaudio and librosa is not installed."
            )

        waveform_np, _ = librosa.load(
            str(audio_path),
            sr=self.sample_rate,
            mono=False,
        )
        waveform = torch.as_tensor(waveform_np, dtype=torch.float32)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        return waveform
