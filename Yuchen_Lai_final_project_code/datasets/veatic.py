from __future__ import annotations

import csv
import io
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import zipfile

import torch
from torch import Tensor
from torch.utils.data import Dataset
import torch.nn.functional as F

from .archive import ZipExtractor, load_tensor_cache, save_tensor_cache

try:
    from torchvision.io import read_video
except ImportError:  # pragma: no cover
    read_video = None

try:
    import av
except ImportError:  # pragma: no cover
    av = None

try:
    import torchaudio
except ImportError:  # pragma: no cover
    torchaudio = None


@dataclass
class VEATICWindow:
    video_id: str
    video_member: str
    start_index: int
    end_index: int
    valence: List[float]
    arousal: List[float]


class VEATICDataset(Dataset[Dict[str, Tensor]]):
    def __init__(
        self,
        video_root: Optional[str | Path] = None,
        zip_path: Optional[str | Path] = None,
        cache_dir: str | Path = "data/cache/veatic",
        clip_length: int = 32,
        stride: int = 16,
        split: str = "train",
        transform=None,
        audio_transform=None,
        include_teacher_audio: bool = False,
        teacher_audio_seconds: float = 4.0,
        teacher_sample_rate: int = 16000,
        train_fraction: float = 1.0,
        sampling_seed: int = 42,
    ) -> None:
        self.video_root = Path(video_root) if video_root else None
        self.clip_length = clip_length
        self.stride = stride
        self.split = split
        self.transform = transform
        self.audio_transform = audio_transform
        self.include_teacher_audio = include_teacher_audio
        self.teacher_audio_seconds = teacher_audio_seconds
        self.teacher_sample_rate = teacher_sample_rate
        self.train_fraction = float(train_fraction)
        self.sampling_seed = int(sampling_seed)
        self.window_cache_dir = Path(cache_dir) / "windows"

        if not (0.0 < self.train_fraction <= 1.0):
            raise ValueError(f"train_fraction must be in (0, 1], got {self.train_fraction!r}")

        self.zip_extractor = None
        self.archive_path = Path(zip_path) if zip_path else None
        if self.video_root is None and self.archive_path is not None:
            self.zip_extractor = ZipExtractor(self.archive_path, cache_dir)

        self.windows = self._apply_train_fraction(self._build_windows())

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        sample = self.windows[index]
        clip, paired_audio, audio_sample_rate, teacher_audio = self._load_or_build_window_tensors(sample)

        target = torch.tensor(
            [
                float(sum(sample.valence) / len(sample.valence)),
                float(sum(sample.arousal) / len(sample.arousal)),
            ],
            dtype=torch.float32,
        )
        frame_targets = torch.tensor(
            list(zip(sample.valence, sample.arousal)),
            dtype=torch.float32,
        )

        item = {
            "video": clip,
            "paired_audio": paired_audio,
            "audio_sample_rate": int(audio_sample_rate),
            "target": target,
            "frame_targets": frame_targets,
            "video_id": sample.video_id,
            "start_index": sample.start_index,
            "end_index": sample.end_index,
        }
        if teacher_audio is not None:
            item["teacher_audio"] = teacher_audio
        return item

    def _load_or_build_window_tensors(self, sample: VEATICWindow) -> tuple[Tensor, Tensor, int, Optional[Tensor]]:
        cache_path = self._window_cache_path(sample)
        cached = load_tensor_cache(cache_path)
        if cached is not None:
            audio_sample_rate = int(cached.get("audio_sample_rate", 16000))
            teacher_audio = cached.get("teacher_audio")
            if teacher_audio is not None:
                teacher_audio = teacher_audio.float()
            return cached["video"], cached["paired_audio"], audio_sample_rate, teacher_audio

        video_path = self._resolve_video_path(sample.video_member)
        frames, audio, info = self._read_video(
            video_path,
            start_index=sample.start_index,
            end_index=sample.end_index,
        )
        frames = frames.permute(0, 3, 1, 2)

        clip = frames[: self.clip_length]
        if clip.size(0) == 0:
            clip = frames[: self.clip_length]
        if clip.size(0) < self.clip_length:
            clip = self._pad_frames(clip)

        if self.transform is not None:
            clip = self.transform(clip)

        paired_audio = self._slice_audio_window(audio, info, sample.start_index, sample.end_index)
        if paired_audio.numel() == 0:
            paired_audio = audio
        paired_audio = self._prepare_audio(paired_audio)
        if self.audio_transform is not None:
            paired_audio = self.audio_transform(paired_audio)

        teacher_audio = None
        if self.include_teacher_audio:
            teacher_audio = self._slice_teacher_audio_window(audio, info, sample.start_index, sample.end_index)

        payload = {
            "video": clip.cpu(),
            "paired_audio": paired_audio.cpu(),
            "audio_sample_rate": int(info.get("audio_fps") or 16000),
        }
        if teacher_audio is not None:
            payload["teacher_audio"] = teacher_audio.half().cpu()
        save_tensor_cache(cache_path, payload)
        return clip, paired_audio, int(info.get("audio_fps") or 16000), teacher_audio

    def _window_cache_path(self, sample: VEATICWindow) -> Path:
        video_key = self.transform.cache_key() if self.transform is not None and hasattr(self.transform, "cache_key") else "rawvideo"
        audio_key = self.audio_transform.cache_key() if self.audio_transform is not None and hasattr(self.audio_transform, "cache_key") else "rawaudio"
        teacher_key = (
            f"teacher{str(self.teacher_audio_seconds).replace('.', 'p')}s_sr{self.teacher_sample_rate}"
            if self.include_teacher_audio
            else "noteacher"
        )
        return (
            self.window_cache_dir
            / self.split
            / sample.video_id
            / f"{sample.start_index}_{sample.end_index}_{video_key}_{audio_key}_{teacher_key}.pt"
        )

    def _read_video(
        self,
        video_path: Path,
        start_index: int = 0,
        end_index: Optional[int] = None,
    ) -> tuple[Tensor, Tensor, Dict[str, object]]:
        if read_video is not None:
            return read_video(str(video_path), pts_unit="sec")
        if av is None:
            raise ImportError("Install torchvision video support or PyAV to load VEATIC clips.")

        container = av.open(str(video_path))
        try:
            frames = []
            requested_end = end_index if end_index is not None else float("inf")
            for frame_index, frame in enumerate(container.decode(video=0)):
                if frame_index < start_index:
                    continue
                if frame_index >= requested_end:
                    break
                frames.append(torch.from_numpy(frame.to_ndarray(format="rgb24")))
            if frames:
                video = torch.stack(frames, dim=0)
            else:
                video = torch.empty((0, 0, 0, 3), dtype=torch.uint8)

            audio = torch.empty((0,), dtype=torch.float32)
            audio_rate = None
            if container.streams.audio:
                container.seek(0)
                audio_chunks = []
                for frame in container.decode(audio=0):
                    chunk = torch.as_tensor(frame.to_ndarray())
                    if chunk.ndim == 1:
                        chunk = chunk.unsqueeze(0)
                    audio_chunks.append(chunk)
                if audio_chunks:
                    audio = torch.cat(audio_chunks, dim=-1).to(torch.float32)
                audio_rate = container.streams.audio[0].rate

            video_rate = None
            if container.streams.video and container.streams.video[0].average_rate is not None:
                video_rate = float(container.streams.video[0].average_rate)

            return video, audio, {"video_fps": video_rate, "audio_fps": audio_rate}
        finally:
            container.close()

    def _build_windows(self) -> List[VEATICWindow]:
        rating_members = self._rating_members()
        video_members = self._video_members()
        windows: List[VEATICWindow] = []

        for video_id, video_member in sorted(video_members.items(), key=lambda item: int(item[0])):
            valence = self._load_rating_values(rating_members[f"{video_id}_valence.csv"])
            arousal = self._load_rating_values(rating_members[f"{video_id}_arousal.csv"])
            total_length = min(len(valence), len(arousal))
            if total_length == 0:
                continue

            split_point = max(1, int(total_length * 0.7))
            if self.split == "train":
                range_start, range_end = 0, split_point
            elif self.split == "test":
                range_start, range_end = split_point, total_length
            else:
                range_start, range_end = 0, total_length

            if range_end - range_start <= self.clip_length:
                segment_valence = valence[range_start:range_end]
                segment_arousal = arousal[range_start:range_end]
                windows.append(
                    VEATICWindow(
                        video_id=video_id,
                        video_member=video_member,
                        start_index=range_start,
                        end_index=range_end,
                        valence=segment_valence,
                        arousal=segment_arousal,
                    )
                )
                continue

            for start in range(range_start, range_end - self.clip_length + 1, self.stride):
                end = start + self.clip_length
                windows.append(
                    VEATICWindow(
                        video_id=video_id,
                        video_member=video_member,
                        start_index=start,
                        end_index=end,
                        valence=valence[start:end],
                        arousal=arousal[start:end],
                    )
                )
        return windows

    def _apply_train_fraction(self, windows: List[VEATICWindow]) -> List[VEATICWindow]:
        if self.split != "train" or self.train_fraction >= 1.0:
            return windows

        grouped: Dict[str, List[VEATICWindow]] = {}
        for window in windows:
            grouped.setdefault(window.video_id, []).append(window)

        selected: List[VEATICWindow] = []
        for video_id in sorted(grouped, key=lambda item: int(item)):
            video_windows = list(grouped[video_id])
            rng = random.Random(self.sampling_seed + int(video_id))
            rng.shuffle(video_windows)
            keep_count = max(1, int(round(len(video_windows) * self.train_fraction)))
            selected.extend(video_windows[:keep_count])

        selected.sort(key=lambda item: (int(item.video_id), item.start_index, item.end_index))
        return selected

    def _rating_members(self) -> Dict[str, str]:
        if self.video_root is not None:
            ratings = {}
            for path in (self.video_root / "rating_averaged").glob("*.csv"):
                ratings[path.name] = str(path)
            return ratings

        assert self.archive_path is not None
        with zipfile.ZipFile(self.archive_path) as archive:
            members = [name for name in archive.namelist() if name.startswith("rating_averaged/") and name.endswith(".csv")]
        return {Path(name).name: name for name in members}

    def _video_members(self) -> Dict[str, str]:
        if self.video_root is not None:
            videos = {}
            for path in (self.video_root / "videos").glob("*.mp4"):
                videos[path.stem] = str(path)
            return videos

        assert self.archive_path is not None
        with zipfile.ZipFile(self.archive_path) as archive:
            members = [name for name in archive.namelist() if name.startswith("videos/") and name.endswith(".mp4")]
        return {Path(name).stem: name for name in members}

    def _load_rating_values(self, member_name: str) -> List[float]:
        if self.video_root is not None:
            with open(member_name, newline="") as handle:
                return [float(row[1]) for row in csv.reader(handle)]

        assert self.archive_path is not None
        with zipfile.ZipFile(self.archive_path) as archive:
            raw = archive.read(member_name).decode("utf-8")
        handle = io.StringIO(raw)
        return [float(row[1]) for row in csv.reader(handle)]

    def _resolve_video_path(self, stored_path: str) -> Path:
        if self.video_root is not None:
            return Path(stored_path)
        assert self.zip_extractor is not None
        return self.zip_extractor.extract(stored_path)

    def _pad_frames(self, frames: Tensor) -> Tensor:
        if frames.size(0) >= self.clip_length:
            return frames[: self.clip_length]
        if frames.size(0) == 0:
            raise ValueError("Encountered empty frame tensor while loading VEATIC clip.")
        pad_count = self.clip_length - frames.size(0)
        pad = frames[-1:].repeat(pad_count, 1, 1, 1)
        return torch.cat([frames, pad], dim=0)

    def _slice_audio_window(
        self,
        audio: Tensor,
        info: Dict[str, object],
        start_index: int,
        end_index: int,
    ) -> Tensor:
        if audio.numel() == 0:
            return audio

        video_fps = info.get("video_fps")
        audio_fps = info.get("audio_fps")
        if video_fps in (None, 0) or audio_fps in (None, 0):
            return audio

        audio_cf = self._channels_first_audio(audio)
        start_sample = max(0, int(round((start_index / float(video_fps)) * float(audio_fps))))
        end_sample = max(start_sample + 1, int(round((end_index / float(video_fps)) * float(audio_fps))))
        end_sample = min(end_sample, audio_cf.size(-1))
        start_sample = min(start_sample, end_sample - 1)
        return audio_cf[..., start_sample:end_sample]

    def _channels_first_audio(self, audio: Tensor) -> Tensor:
        if audio.ndim == 1:
            return audio.unsqueeze(0).float()
        if audio.ndim != 2:
            return audio.reshape(1, -1).float()
        if audio.size(0) <= 8:
            return audio.float()
        if audio.size(1) <= 8:
            return audio.transpose(0, 1).float()
        return audio.float()

    def _prepare_audio(self, audio: Tensor) -> Tensor:
        if audio.numel() == 0:
            return torch.zeros(1, 16000, dtype=torch.float32)
        return self._channels_first_audio(audio)

    def _slice_teacher_audio_window(
        self,
        audio: Tensor,
        info: Dict[str, object],
        start_index: int,
        end_index: int,
    ) -> Tensor:
        target_num_samples = int(round(self.teacher_audio_seconds * self.teacher_sample_rate))
        if audio.numel() == 0:
            return torch.zeros(target_num_samples, dtype=torch.float32)

        video_fps = info.get("video_fps")
        audio_fps = info.get("audio_fps")
        audio_cf = self._channels_first_audio(audio)

        if video_fps in (None, 0) or audio_fps in (None, 0):
            mono = audio_cf.mean(dim=0)
            return self._resample_and_pad_teacher_audio(mono, int(self.teacher_sample_rate))

        clip_center_seconds = ((start_index + end_index) * 0.5) / float(video_fps)
        half_window_seconds = self.teacher_audio_seconds * 0.5
        start_seconds = max(0.0, clip_center_seconds - half_window_seconds)
        end_seconds = clip_center_seconds + half_window_seconds

        start_sample = int(round(start_seconds * float(audio_fps)))
        end_sample = int(round(end_seconds * float(audio_fps)))
        end_sample = max(start_sample + 1, min(end_sample, audio_cf.size(-1)))
        start_sample = min(start_sample, end_sample - 1)

        mono = audio_cf.mean(dim=0)[start_sample:end_sample]
        return self._resample_and_pad_teacher_audio(mono, int(audio_fps))

    def _resample_and_pad_teacher_audio(self, waveform: Tensor, source_sample_rate: int) -> Tensor:
        waveform = waveform.float().flatten()
        if waveform.numel() == 0:
            waveform = torch.zeros(1, dtype=torch.float32)

        if source_sample_rate != self.teacher_sample_rate:
            if torchaudio is not None:
                waveform = torchaudio.functional.resample(
                    waveform.unsqueeze(0),
                    orig_freq=source_sample_rate,
                    new_freq=self.teacher_sample_rate,
                ).squeeze(0)
            else:
                target_length = max(
                    1,
                    int(round(waveform.numel() * self.teacher_sample_rate / float(source_sample_rate))),
                )
                waveform = F.interpolate(
                    waveform.view(1, 1, -1),
                    size=target_length,
                    mode="linear",
                    align_corners=False,
                ).view(-1)

        target_num_samples = int(round(self.teacher_audio_seconds * self.teacher_sample_rate))
        if waveform.numel() > target_num_samples:
            center = waveform.numel() // 2
            start = max(0, center - target_num_samples // 2)
            end = start + target_num_samples
            waveform = waveform[start:end]
        elif waveform.numel() < target_num_samples:
            waveform = F.pad(waveform, (0, target_num_samples - waveform.numel()))
        return waveform
