from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import AudioFeatureTransform, DEAMDataset, VEATICDataset, VideoFrameTransform

FIGURE_DIR = PROJECT_ROOT / "icml2026_template" / "figures"


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    make_data_examples_figure()
    make_training_dynamics_figure()


def make_data_examples_figure() -> None:
    veatic_raw = VEATICDataset(
        zip_path=PROJECT_ROOT / "data" / "veatic" / "VEATIC.zip",
        cache_dir=PROJECT_ROOT / "data" / "cache" / "veatic_figures_raw",
        clip_length=8,
        stride=8,
        split="train",
        transform=VideoFrameTransform(size=224),
        audio_transform=None,
    )
    veatic_sample = pick_most_dynamic_veatic_window(veatic_raw)

    deam_dataset = DEAMDataset(
        audio_zip_path=PROJECT_ROOT / "data" / "deam" / "DEAM_audio.zip",
        annotations_dir=PROJECT_ROOT / "data" / "deam" / "annotations" / "annotations",
        transform=None,
        cache_dir=PROJECT_ROOT / "data" / "cache" / "deam_figures_raw",
        target_length_seconds=30.0,
    )
    deam_sample = pick_most_dynamic_deam_song(deam_dataset)

    fig = plt.figure(figsize=(7.2, 5.8), constrained_layout=True)
    subfigs = fig.subfigures(2, 1, height_ratios=[1.15, 1.0], hspace=0.06)

    top = subfigs[0]
    top.suptitle("VEATIC paired video-audio training sample", fontsize=10, y=1.02)
    top_axes = top.subplots(2, 4, gridspec_kw={"height_ratios": [1.0, 0.9]})
    clip = veatic_sample["video"].cpu()
    frame_targets = veatic_sample["frame_targets"].cpu().numpy()
    paired_audio = veatic_sample["paired_audio"].cpu()
    spectrogram = compute_log_spectrogram(paired_audio)

    for axis, frame_index in zip(top_axes[0], sample_indices(len(clip), 4)):
        frame = clip[frame_index].permute(1, 2, 0).numpy()
        axis.imshow(np.clip(frame, 0.0, 1.0))
        axis.set_title(f"Frame {frame_index + 1}", fontsize=8)
        axis.axis("off")

    top_axes[1, 0].plot(frame_targets[:, 0], color="#d14a61", linewidth=1.8, label="Valence")
    top_axes[1, 0].plot(frame_targets[:, 1], color="#2b6cb0", linewidth=1.8, label="Arousal")
    top_axes[1, 0].set_title("Window labels", fontsize=8)
    top_axes[1, 0].set_xlabel("Frame index")
    top_axes[1, 0].set_ylim(-1.0, 1.0)
    top_axes[1, 0].legend(frameon=False, fontsize=7, loc="lower right")

    top_axes[1, 1].imshow(spectrogram, aspect="auto", origin="lower", cmap="magma")
    top_axes[1, 1].set_title("Paired audio spectrogram", fontsize=8)
    top_axes[1, 1].set_xlabel("Time")
    top_axes[1, 1].set_ylabel("Freq.")

    top_axes[1, 2].plot(paired_audio.mean(dim=0).numpy(), color="#444444", linewidth=1.0)
    top_axes[1, 2].set_title("Waveform", fontsize=8)
    top_axes[1, 2].set_xlabel("Samples")
    top_axes[1, 2].set_yticks([])

    top_axes[1, 3].axis("off")
    top_axes[1, 3].text(
        0.0,
        0.92,
        "\n".join(
            [
                f"video id: {veatic_sample['video_id']}",
                f"window: {veatic_sample['start_index']} to {veatic_sample['end_index']}",
                f"mean valence: {frame_targets[:, 0].mean():.3f}",
                f"mean arousal: {frame_targets[:, 1].mean():.3f}",
            ]
        ),
        va="top",
        fontsize=8,
    )

    bottom = subfigs[1]
    bottom.suptitle("DEAM source audio supervision example", fontsize=10, y=1.04)
    bottom_axes = bottom.subplots(1, 3, gridspec_kw={"width_ratios": [1.45, 1.0, 1.0]})
    waveform = deam_sample["audio"].cpu()
    dynamic_target = deam_sample["dynamic_target"].cpu().numpy()
    audio_spec = compute_log_spectrogram(waveform)
    seconds = np.arange(min(audio_spec.shape[1], dynamic_target.shape[0]))

    bottom_axes[0].imshow(audio_spec[:, : len(seconds)], aspect="auto", origin="lower", cmap="viridis")
    bottom_axes[0].set_title("Log-power spectrogram", fontsize=8)
    bottom_axes[0].set_xlabel("Time")
    bottom_axes[0].set_ylabel("Freq.")

    bottom_axes[1].plot(seconds, dynamic_target[: len(seconds), 0], color="#d14a61", linewidth=1.7)
    bottom_axes[1].set_title("Valence trajectory", fontsize=8)
    bottom_axes[1].set_xlabel("Second")
    bottom_axes[1].set_ylim(-1.0, 1.0)

    bottom_axes[2].plot(seconds, dynamic_target[: len(seconds), 1], color="#2b6cb0", linewidth=1.7)
    bottom_axes[2].set_title("Arousal trajectory", fontsize=8)
    bottom_axes[2].set_xlabel("Second")
    bottom_axes[2].set_ylim(-1.0, 1.0)

    fig.savefig(FIGURE_DIR / "data_examples.pdf", bbox_inches="tight")


def make_training_dynamics_figure() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.55), constrained_layout=True)

    paper_baselines = {
        "Ortega feat.": 0.5166635513305664,
        "Ortega dec.": 0.5124934911727905,
        "Zhang LFAN": 0.5732549428939819,
    }
    axes[0].bar(
        np.arange(len(paper_baselines)),
        list(paper_baselines.values()),
        color=["#b9cdf7", "#d6e2fb", "#7ea0ff"],
        edgecolor="#3559a8",
        linewidth=0.8,
    )
    axes[0].axhline(0.6391928195953369, color="#c53b32", linestyle="--", linewidth=1.5, label="Ours (25%)")
    axes[0].axhline(0.6080424189567566, color="#ef8b42", linestyle=":", linewidth=1.6, label="Ours (10%)")
    axes[0].set_xticks(np.arange(len(paper_baselines)), list(paper_baselines.keys()))
    axes[0].set_ylim(0.48, 0.67)
    axes[0].set_title("Contextual literature baselines")
    axes[0].set_ylabel("CCC mean")
    axes[0].legend(frameon=False, fontsize=7, loc="upper left")

    plot_eval_curve(
        axes[1],
        PROJECT_ROOT / "runs" / "proposal_r2plus1d_visual_only_f025_formal_e100_v1" / "metrics.jsonl",
        PROJECT_ROOT / "runs" / "proposal_r2plus1d_align_l01_f025_formal_e100_v1" / "metrics.jsonl",
        "25% visual supervision",
    )
    plot_eval_curve(
        axes[2],
        PROJECT_ROOT / "runs" / "proposal_r2plus1d_visual_only_f010_formal_e100_v1" / "metrics.jsonl",
        PROJECT_ROOT / "runs" / "proposal_r2plus1d_align_l01_f010_formal_e100_v1" / "metrics.jsonl",
        "10% visual supervision",
    )

    fig.savefig(FIGURE_DIR / "training_dynamics.pdf", bbox_inches="tight")


def plot_eval_curve(axis, visual_path: Path, ours_path: Path, title: str) -> None:
    visual = load_jsonl_metrics(visual_path)
    ours = load_jsonl_metrics(ours_path)
    axis.plot(visual["epoch"], visual["ccc_mean"], color="#7a7a7a", linewidth=1.6, label="Visual-only")
    axis.plot(ours["epoch"], ours["ccc_mean"], color="#d66a1f", linewidth=1.8, label="Ours")
    axis.scatter(
        [visual["epoch"][int(np.argmax(visual["ccc_mean"]))]],
        [np.max(visual["ccc_mean"])],
        color="#7a7a7a",
        s=18,
    )
    axis.scatter(
        [ours["epoch"][int(np.argmax(ours["ccc_mean"]))]],
        [np.max(ours["ccc_mean"])],
        color="#d66a1f",
        s=18,
    )
    axis.set_title(title)
    axis.set_xlabel("Epoch")
    axis.set_ylim(0.48, 0.67)
    axis.grid(alpha=0.2, linewidth=0.5)
    axis.legend(frameon=False, fontsize=7, loc="lower right")


def pick_most_dynamic_veatic_window(dataset: VEATICDataset):
    best_index = 0
    best_score = float("-inf")
    for index, window in enumerate(dataset.windows):
        frame_targets = np.stack([window.valence, window.arousal], axis=1)
        score = frame_targets.std(axis=0).sum() + np.abs(frame_targets.mean(axis=0)).sum() * 0.2
        if score > best_score:
            best_index = index
            best_score = float(score)
    return dataset[best_index]


def pick_most_dynamic_deam_song(dataset: DEAMDataset):
    best_index = 0
    best_score = float("-inf")
    for index, sample in enumerate(dataset.samples):
        valence = np.asarray(sample.valence[:30], dtype=np.float32)
        arousal = np.asarray(sample.arousal[:30], dtype=np.float32)
        if len(valence) == 0 or len(arousal) == 0:
            continue
        score = valence.std() + arousal.std() + 0.1 * (np.abs(valence.mean()) + np.abs(arousal.mean()))
        if score > best_score:
            best_index = index
            best_score = float(score)
    return dataset[best_index]


def compute_log_spectrogram(waveform: torch.Tensor) -> np.ndarray:
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=0)
    waveform = waveform.float()
    if waveform.numel() < 1024:
        waveform = torch.nn.functional.pad(waveform, (0, 1024 - waveform.numel()))
    spec = torch.stft(
        waveform,
        n_fft=1024,
        hop_length=256,
        return_complex=True,
    )
    spec = torch.log1p(spec.abs().pow(2))
    return spec.cpu().numpy()


def sample_indices(length: int, count: int) -> Iterable[int]:
    if length <= count:
        return range(length)
    return np.linspace(0, length - 1, num=count, dtype=int)


def load_jsonl_metrics(path: Path) -> dict[str, np.ndarray]:
    epochs = []
    ccc_mean = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = eval_json_line(line)
            epochs.append(row["epoch"])
            ccc_mean.append(row["eval"]["ccc_mean"])
    return {
        "epoch": np.asarray(epochs, dtype=np.int32),
        "ccc_mean": np.asarray(ccc_mean, dtype=np.float32),
    }


def eval_json_line(line: str) -> dict:
    import json

    return json.loads(line)


if __name__ == "__main__":
    main()
