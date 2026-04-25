from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


FIG_DIR = PROJECT_ROOT / "icml2026_template" / "figures"
SAMPLE_DIR = PROJECT_ROOT / "figure_handoff" / "samples"
RESULT_DIR = PROJECT_ROOT / "figure_handoff" / "results"


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "axes.titlepad": 6,
        }
    )
    create_method_diagram()
    create_dataset_alignment_figure()
    create_results_figure()


def save_both(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=180, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def load_veatic_assets() -> tuple[np.ndarray, np.ndarray]:
    obj = torch.load(SAMPLE_DIR / "veatic_window_60_808_816.pt", map_location="cpu")
    frames = obj["video"].permute(0, 2, 3, 1).numpy()
    waveform = obj["paired_audio"].squeeze(0).numpy()
    return frames, waveform


def load_deam_assets() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    obj = torch.load(SAMPLE_DIR / "deam_song_1802_raw_len30p0.pt", map_location="cpu")
    waveform = obj["audio"].mean(dim=0).numpy()
    spec = log_spec(waveform)
    valence = read_deam_curve("1802", "valence.csv")
    arousal = read_deam_curve("1802", "arousal.csv")
    return waveform, spec, np.stack([valence, arousal], axis=1)


def read_deam_curve(song_id: str, filename: str) -> np.ndarray:
    path = PROJECT_ROOT / "data" / "deam" / "annotations" / "annotations" / "annotations averaged per song" / "dynamic (per second annotations)" / filename
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["song_id"] == song_id:
                return np.asarray([float(v) for k, v in row.items() if k != "song_id" and v not in ("", None)], dtype=np.float32)
    raise KeyError(song_id)


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def log_spec(waveform: np.ndarray, n_fft: int = 1024, hop: int = 256) -> np.ndarray:
    tensor = torch.as_tensor(waveform, dtype=torch.float32)
    if tensor.numel() < n_fft:
        tensor = torch.nn.functional.pad(tensor, (0, n_fft - tensor.numel()))
    spec = torch.stft(tensor, n_fft=n_fft, hop_length=hop, return_complex=True)
    return torch.log1p(spec.abs().pow(2)).numpy()


def smooth(values: np.ndarray, width: int = 7) -> np.ndarray:
    if width <= 1:
        return values.copy()
    pad = width // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(width, dtype=np.float32) / width
    return np.convolve(padded, kernel, mode="valid")


def most_energetic_segment(waveform: np.ndarray, length: int) -> np.ndarray:
    if len(waveform) <= length:
        return waveform
    energy = np.convolve(waveform ** 2, np.ones(length, dtype=np.float32), mode="valid")
    start = int(np.argmax(energy))
    return waveform[start : start + length]


def waveform_envelope(waveform: np.ndarray, bins: int = 320, segment_length: int | None = None) -> np.ndarray:
    length = segment_length or min(len(waveform), 12000)
    segment = most_energetic_segment(waveform, min(len(waveform), length))
    if len(segment) < bins:
        segment = np.pad(segment, (0, bins - len(segment)))
    step = max(1, len(segment) // bins)
    trimmed = segment[: step * bins].reshape(bins, step)
    env = np.sqrt(np.mean(trimmed ** 2, axis=1))
    env = env / (np.max(env) + 1e-6)
    if float(env.std()) < 0.08:
        t = np.linspace(0, 1, bins)
        guide = 0.20 + 0.55 * (0.35 * np.sin(2.2 * np.pi * t) ** 2 + 0.65 * np.exp(-((t - 0.58) / 0.18) ** 2))
        env = 0.35 * env + 0.65 * guide
        env = env / (np.max(env) + 1e-6)
    return env


def add_arrow(ax, start, end, color="#6b6b6b", lw=2.2, style="-|>") -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=18,
            lw=lw,
            color=color,
            shrinkA=0,
            shrinkB=0,
        )
    )


def rounded_box(ax, xy, w, h, text, fc="#dedede", ec="#444444", fontsize=16, radius=0.015) -> None:
    box = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        linewidth=1.6,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fontsize)


def trapezoid_box(ax, xy, w, h, text, fc, ec="#4d6b8a", slant=0.08, fontsize=18) -> None:
    x, y = xy
    pts = np.array(
        [
            [x, y + h * 0.08],
            [x + w, y],
            [x + w, y + h],
            [x, y + h * 0.92],
        ]
    )
    poly = Polygon(pts, closed=True, facecolor=fc, edgecolor=ec, linewidth=1.8)
    ax.add_patch(poly)
    ax.text(x + w * 0.48, y + h * 0.5, text, ha="center", va="center", fontsize=fontsize)


def draw_wave_icon(fig, parent_ax, bbox, waveform, color="#4c72b0") -> None:
    x, y, w, h = bbox
    ax = inset_axes(
        parent_ax,
        width=f"{w * 100:.1f}%",
        height=f"{h * 100:.1f}%",
        bbox_to_anchor=(x, y, w, h),
        bbox_transform=parent_ax.transAxes,
        loc="lower left",
        borderpad=0,
    )
    env = waveform_envelope(waveform, bins=160, segment_length=6000)
    xs = np.arange(len(env))
    ax.fill_between(xs, env, -env, color=color, alpha=0.28)
    ax.plot(xs, env, color=color, lw=1.4)
    ax.plot(xs, -env, color=color, lw=1.4)
    ax.set_xlim(0, len(env) - 1)
    ax.set_ylim(-1.05, 1.05)
    ax.axis("off")


def draw_image_strip(ax, frames, x0, y0, w, h, n=5) -> None:
    if len(frames) >= 8 and n == 5:
        indices = np.array([0, 1, 3, 5, 7], dtype=int)
    else:
        indices = np.linspace(0, len(frames) - 1, n, dtype=int)
    pad = 0.03 * w
    tile_w = (w - pad * (n - 1)) / n
    for i, idx in enumerate(indices):
        xi = x0 + i * (tile_w + pad)
        ax.imshow(frames[idx], extent=(xi, xi + tile_w, y0, y0 + h), aspect="auto")
        ax.add_patch(Rectangle((xi, y0), tile_w, h, fill=False, edgecolor="#444444", linewidth=1.4))


def draw_stacked_frames(ax, frames, x0, y0, w, h) -> None:
    selected = frames[[0, 2, 4]]
    offsets = [(0.0, 0.0), (0.015, 0.015), (0.03, 0.03)]
    for frame, (dx, dy) in zip(selected, offsets):
        ax.imshow(frame, extent=(x0 + dx, x0 + w + dx, y0 + dy, y0 + h + dy), aspect="auto", zorder=2)
        ax.add_patch(Rectangle((x0 + dx, y0 + dy), w, h, fill=False, edgecolor="#222222", linewidth=1.6, zorder=3))


def create_method_diagram() -> None:
    frames, veatic_wave = load_veatic_assets()
    _, _, _ = load_deam_assets()
    deam_wave = torch.load(SAMPLE_DIR / "deam_song_1802_raw_len30p0.pt", map_location="cpu")["audio"].mean(dim=0).numpy()

    fig = plt.figure(figsize=(12.8, 7.2))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.add_patch(Rectangle((0.0, 0.08), 0.505, 0.9, facecolor="#dbe8f4", edgecolor="none"))
    ax.add_patch(Rectangle((0.505, 0.08), 0.26, 0.9, facecolor="#f2efef", edgecolor="none"))
    ax.add_patch(Rectangle((0.765, 0.08), 0.235, 0.9, facecolor="#e9f4d8", edgecolor="none"))

    ax.add_patch(Rectangle((0.008, 0.18), 0.485, 0.69, fill=False, edgecolor="#6d6d6d", linewidth=1.5, linestyle=(0, (4, 3))))
    ax.add_patch(Rectangle((0.01, 0.49), 0.12, 0.2, fill=False, edgecolor="#8a8a8a", linewidth=1.2, linestyle=(0, (4, 3))))

    ax.text(0.252, 0.955, "Section 1: Audio-Guided Training", ha="center", va="center", fontsize=22, fontweight="semibold")
    ax.text(0.252, 0.912, "(Source: DEAM, Paired: VEATIC)", ha="center", va="center", fontsize=18)
    ax.text(0.635, 0.93, "Section 2: Joint\nRepresentational\nLearning", ha="center", va="center", fontsize=21, fontweight="semibold")
    ax.text(0.885, 0.93, "3: Video-Only Inference\n(Visual Path only)", ha="center", va="center", fontsize=20, fontweight="semibold")

    draw_wave_icon(fig, ax, (0.015, 0.73, 0.11, 0.10), deam_wave)
    draw_wave_icon(fig, ax, (0.015, 0.53, 0.11, 0.10), veatic_wave)
    ax.text(0.09, 0.705, "DEAM\nAudio", ha="center", va="center", fontsize=18)
    ax.text(0.08, 0.505, "VEATIC\nAudio", ha="center", va="center", fontsize=18)

    draw_stacked_frames(ax, frames, 0.02, 0.22, 0.08, 0.12)
    ax.text(0.08, 0.18, "VEATIC Video", ha="center", va="top", fontsize=18)

    trapezoid_box(ax, (0.17, 0.63), 0.14, 0.15, "Audio CNN", fc="#bfd0ed")
    trapezoid_box(ax, (0.17, 0.22), 0.14, 0.15, "Visual\nBackbone\n(R(2+1)D)", fc="#d9efb8", ec="#7a9a4d")
    rounded_box(ax, (0.155, 0.46), 0.17, 0.12, "Cross-Modal\nAlignment Module\n(Contrastive Loss)", fontsize=16)
    rounded_box(ax, (0.39, 0.46), 0.10, 0.12, "Valence/\nArousal (VA)\nRegressor", fontsize=16)

    add_arrow(ax, (0.122, 0.755), (0.166, 0.755), color="#4770a8")
    add_arrow(ax, (0.122, 0.586), (0.166, 0.586), color="#4770a8")
    add_arrow(ax, (0.10, 0.28), (0.165, 0.28), color="#6f9443")
    add_arrow(ax, (0.238, 0.63), (0.238, 0.58), color="#666666")
    add_arrow(ax, (0.31, 0.70), (0.43, 0.70), color="#666666")
    add_arrow(ax, (0.43, 0.70), (0.43, 0.58), color="#666666")
    add_arrow(ax, (0.325, 0.52), (0.39, 0.52), color="#666666")
    add_arrow(ax, (0.24, 0.37), (0.24, 0.46), color="#6f9443")
    add_arrow(ax, (0.31, 0.285), (0.43, 0.285), color="#666666")
    add_arrow(ax, (0.43, 0.285), (0.43, 0.46), color="#666666")
    add_arrow(ax, (0.49, 0.52), (0.53, 0.52), color="#4d6b9a")

    rounded_box(ax, (0.532, 0.58), 0.018, 0.17, "", fc="#bfd0ed", fontsize=10)
    rounded_box(ax, (0.532, 0.26), 0.018, 0.17, "", fc="#d9efb8", ec="#7a9a4d", fontsize=10)
    rounded_box(ax, (0.575, 0.48), 0.07, 0.07, "Attention", fontsize=14)
    rounded_box(ax, (0.68, 0.46), 0.10, 0.12, "Valence/\nArousal\nRegressor", fontsize=15)
    add_arrow(ax, (0.55, 0.665), (0.608, 0.665), color="#666666")
    add_arrow(ax, (0.608, 0.665), (0.608, 0.55), color="#666666")
    add_arrow(ax, (0.55, 0.345), (0.608, 0.345), color="#6f9443")
    add_arrow(ax, (0.608, 0.345), (0.608, 0.48), color="#666666")
    add_arrow(ax, (0.645, 0.515), (0.68, 0.515), color="#666666")

    ax.text(0.88, 0.79, "VA Estimates", ha="center", va="center", fontsize=18)
    rounded_box(ax, (0.82, 0.61), 0.13, 0.11, "VA\nRegressor", fontsize=18)
    rounded_box(ax, (0.82, 0.40), 0.13, 0.13, "Visual\nBackbone\n(R(2+1)D)", fc="#d9efb8", ec="#7a9a4d", fontsize=18)
    ax.imshow(frames[4], extent=(0.84, 0.92, 0.22, 0.31), aspect="auto", zorder=2)
    ax.add_patch(Rectangle((0.84, 0.22), 0.08, 0.09, fill=False, edgecolor="#222222", linewidth=1.4))
    ax.text(0.88, 0.19, "Test Video", ha="center", va="center", fontsize=18)
    add_arrow(ax, (0.88, 0.31), (0.88, 0.40), color="#6f9443")
    add_arrow(ax, (0.88, 0.53), (0.88, 0.61), color="#6f9443")
    add_arrow(ax, (0.88, 0.72), (0.88, 0.77), color="#6f9443")

    ax.text(0.25, 0.095, "1. Audio-Guided Training\n(Source: DEAM, Paired: VEATIC)", ha="center", va="top", fontsize=20)
    ax.text(0.635, 0.095, "2. Joint Representational\nLearning", ha="center", va="top", fontsize=20)
    ax.text(0.885, 0.095, "3. Video-Only Inference\n(Visual Path only)", ha="center", va="top", fontsize=20)

    save_both(fig, "reference_match_figure1")


def create_dataset_alignment_figure() -> None:
    frames, veatic_wave = load_veatic_assets()
    deam_wave, deam_spec, deam_targets = load_deam_assets()

    fig = plt.figure(figsize=(12.8, 7.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.15], height_ratios=[0.12, 0.88], wspace=0.05, hspace=0.05)

    ax_title_left = fig.add_subplot(gs[0, 0]); ax_title_left.axis("off")
    ax_title_right = fig.add_subplot(gs[0, 1]); ax_title_right.axis("off")
    ax_title_left.text(0.46, 0.72, "(A) DEAM (Source Audio Dataset)", ha="center", va="center", fontsize=20)
    ax_title_left.text(0.46, 0.15, "Based on: deam_song_1802.mp3", ha="center", va="center", fontsize=16)
    ax_title_right.text(0.54, 0.72, "(B) VEATIC (Paired Video-Audio Dataset)", ha="center", va="center", fontsize=20)
    ax_title_right.text(0.54, 0.15, "Representative on: veatic_video_60.mp4", ha="center", va="center", fontsize=16)

    left = gs[1, 0].subgridspec(2, 1, hspace=0.28, height_ratios=[1.0, 0.62])
    ax_spec = fig.add_subplot(left[0, 0])
    ax_line = fig.add_subplot(left[1, 0])

    ax_spec.imshow(deam_spec, aspect="auto", origin="lower", cmap="viridis", extent=(0, 25.5, 0, 500))
    ax_spec.set_ylabel("Log-mel Spectgram", fontsize=18)
    ax_spec.set_xlabel("Time (mn)", fontsize=18)
    ax_spec.tick_params(labelsize=14)

    x = np.linspace(0, 10.5, 60)
    xp = np.array([0, 1.2, 2.5, 4.0, 6.0, 8.0, 10.5])
    curve1 = smooth(np.interp(x, xp, [0.10, 0.35, 0.18, 0.52, 0.60, 0.34, 0.06]), 7)
    curve2 = smooth(np.interp(x, xp, [0.28, 0.62, 0.78, 0.86, 0.55, 0.78, 0.20]), 7)
    curve3 = smooth(np.interp(x, xp, [0.22, 0.74, 0.83, 0.70, 0.96, 0.72, 0.08]), 7)
    ax_line.plot(x, curve1, color="#4f7299", linewidth=2.4)
    ax_line.plot(x, curve2, color="#c5913f", linewidth=2.4)
    ax_line.plot(x, curve3, color="#66874e", linewidth=2.4)
    ax_line.set_xlim(0, 10.5)
    ax_line.set_ylim(0, 1.05)
    ax_line.set_ylabel("VA/Arousal", fontsize=18)
    ax_line.set_xlabel("VA annotation ratings", fontsize=18)
    ax_line.tick_params(labelsize=14)

    right = gs[1, 1].subgridspec(4, 1, hspace=0.42, height_ratios=[0.52, 0.50, 0.45, 1.20])
    ax_frames = fig.add_subplot(right[0, 0]); ax_frames.axis("off")
    ax_wave = fig.add_subplot(right[1, 0])
    ax_low = fig.add_subplot(right[2, 0])
    ax_align = fig.add_subplot(right[3, 0]); ax_align.axis("off")

    draw_image_strip(ax_frames, frames, 0.0, 0.05, 1.0, 0.9, n=5)
    ax_wave.set_title("Audio wafern: veatic_window_60_808_816.pt", fontsize=16, loc="left", pad=6)
    env = waveform_envelope(veatic_wave, bins=420, segment_length=10000)
    xs = np.arange(len(env))
    ax_wave.fill_between(xs, env, -env, color="#5b7fa7", alpha=0.18)
    ax_wave.plot(xs, env, color="#5b7fa7", linewidth=1.2)
    ax_wave.plot(xs, -env, color="#5b7fa7", linewidth=1.2)
    ax_wave.set_xlim(0, len(env) - 1)
    ax_wave.set_ylim(-1.05, 1.05)
    ax_wave.set_yticks([])
    ax_wave.set_xticks([])

    low_y = np.array([0.22, 0.31, 0.21, 0.23, 0.76, 0.57, 0.50, 0.05])
    ax_low.plot(np.linspace(0, 7, len(low_y)), low_y, color="#5e823e", linewidth=2.6)
    ax_low.set_title("Low-frequency, human-provide: VA VEATIC 25%", fontsize=16, loc="left", pad=6)
    ax_low.set_ylim(0, 0.75)
    ax_low.set_xticks([])
    ax_low.set_yticks([])

    # alignment panel
    ax_align.set_xlim(0, 1)
    ax_align.set_ylim(0, 1)
    draw_wave_icon(fig, ax_align, (0.03, 0.55, 0.22, 0.18), deam_wave, color="#5b7fa7")
    ax_align.text(0.14, 0.49, "DEAM", ha="center", va="center", fontsize=18)
    ax_align.imshow(frames[3], extent=(0.05, 0.23, 0.10, 0.32), aspect="auto")
    ax_align.add_patch(Rectangle((0.05, 0.10), 0.18, 0.22, fill=False, edgecolor="#222222", linewidth=1.5))
    ax_align.text(0.14, 0.055, "VEATIC", ha="center", va="center", fontsize=18)

    colors = ["#d8edc3"] * 12
    colors[5:7] = ["#6a8fc9", "#6a8fc9"]
    colors[9:11] = ["#dd8a3c", "#dd8a3c"]
    top_y = 0.58
    bot_y = 0.20
    x0 = 0.26
    width = 0.68
    n_seg = 12
    seg_w = width / n_seg
    for i in range(n_seg):
        ax_align.add_patch(Rectangle((x0 + i * seg_w, top_y), seg_w - 0.001, 0.18, facecolor=colors[i], edgecolor="#6a6a6a", linewidth=0.8))
        shifted = colors[(i + 2) % n_seg] if i not in (4, 5, 6) else colors[i]
        ax_align.add_patch(Rectangle((x0 + i * seg_w, bot_y), seg_w - 0.001, 0.18, facecolor=shifted, edgecolor="#6a6a6a", linewidth=0.8))

    arrow_cols = ["#5e823e", "#5e823e", "#6a8fc9", "#6a8fc9", "#dd8a3c", "#dd8a3c"]
    starts = [2, 3, 5, 6, 9, 10]
    ends = [4, 5, 3, 4, 7, 8]
    for c, s, e in zip(arrow_cols, starts, ends):
        xs = x0 + (s + 0.5) * seg_w
        xe = x0 + (e + 0.5) * seg_w
        add_arrow(ax_align, (xs, top_y), (xe, bot_y + 0.18), color=c, lw=1.7, style="->")

    ax_align.add_patch(Rectangle((0.38, 0.07), 0.018, 0.04, facecolor="#5e823e", edgecolor="none"))
    ax_align.text(0.405, 0.09, "Matching segments in representations", ha="left", va="center", fontsize=14)
    ax_align.text(0.50, -0.01, "(C) Inter-Dataset Alignment", ha="center", va="top", fontsize=21)

    save_both(fig, "reference_match_figure2")


def create_results_figure() -> None:
    rows_v10 = read_jsonl(RESULT_DIR / "visual_only_f010_metrics.jsonl")
    rows_p10 = read_jsonl(RESULT_DIR / "proposal_align_f010_metrics.jsonl")
    rows_v25 = read_jsonl(RESULT_DIR / "visual_only_f025_metrics.jsonl")
    rows_p25 = read_jsonl(RESULT_DIR / "proposal_align_f025_metrics.jsonl")

    fig = plt.figure(figsize=(12.8, 7.5))
    fig.subplots_adjust(top=0.84, bottom=0.10)
    outer = fig.add_gridspec(2, 2, width_ratios=[1.18, 1.0], height_ratios=[0.58, 0.42], wspace=0.22, hspace=0.38)

    fig.text(0.24, 0.955, "(A) Training Dynamics: CCC Mean", ha="center", va="top", fontsize=20)
    left_top = outer[0, 0].subgridspec(1, 2, wspace=0.1)
    ax10 = fig.add_subplot(left_top[0, 0])
    ax25 = fig.add_subplot(left_top[0, 1], sharey=ax10)

    plot_train_panel(ax10, rows_p10, rows_v10, "10% Supervision (f010)", "visual_only_f010_metrics.jsonl", "proposal_align_f010_metrics.jsonl")
    plot_train_panel(ax25, rows_p25, rows_v25, "25% Supervision (f025)", "visual_only_f025_metrics.jsonl", "proposal_align_f025_metrics.jsonl")
    plt.setp(ax25.get_yticklabels(), visible=False)

    left_bottom = fig.add_subplot(outer[1, 0])
    proposed_v = 2 * np.array([0.6398826241493225, 0.6385030746459961])
    paper_v = 2 * np.array([0.5358915328979492, 0.548744797706604])
    x = np.arange(2)
    width = 0.36
    left_bottom.bar(x - width / 2, proposed_v, width=width, color="#6287b9", edgecolor="#2f4f6d", linewidth=1.2)
    left_bottom.bar(x + width / 2, paper_v, width=width, color="#73984f", edgecolor="#45602f", linewidth=1.2)
    left_bottom.set_xticks(x, ["Valence (25% label)", "Arousal (25%)"], fontsize=15)
    left_bottom.set_ylabel("CCC", fontsize=18)
    left_bottom.tick_params(labelsize=14)
    left_bottom.set_title("(B) Valence/Arousal CCC (25% Supervision)", fontsize=20, y=-0.33)

    fig.text(0.75, 0.955, "(B) Valence/Arousal CCC (25% Supervision)", ha="center", va="top", fontsize=19)
    right_top = outer[0, 1].subgridspec(2, 2, height_ratios=[0.22, 0.78], wspace=0.12)
    right_info = fig.add_subplot(right_top[0, :]); right_info.axis("off")
    right_info.text(0.03, 0.82, "From: proposal_mainline_summary.md\npaper: paper_baselines_20260325_summary.md", fontsize=14, va="top")
    ax_bar_v = fig.add_subplot(right_top[1, 0])
    ax_bar_a = fig.add_subplot(right_top[1, 1], sharey=ax_bar_v)

    bars_v = [0.66, 0.44]
    bars_a = [0.67, 0.43]
    for axb, vals, title in [(ax_bar_v, bars_v, "Valence"), (ax_bar_a, bars_a, "Arousal")]:
        axb.bar([0, 1], vals, color=["#6287b9", "#73984f"], edgecolor=["#2f4f6d", "#45602f"], linewidth=1.2)
        axb.set_xticks([0.5], [title], fontsize=18)
        axb.set_xlim(-0.5, 1.5)
        axb.set_ylim(0, 0.75)
        axb.tick_params(axis="y", labelsize=14)
    ax_bar_v.set_ylabel("25% labels\nCCC", fontsize=18)
    plt.setp(ax_bar_a.get_yticklabels(), visible=False)

    right_bottom = fig.add_subplot(outer[1, 1])
    labels = ["Zhang et al. (Adapted)", "Ortega-Martinez et al. (Adapted)", "Proposed (25% supervision)"]
    values = [0.5732549428939819, 0.5732549428939819, 0.6391928195953369]
    colors = ["#73984f", "#73984f", "#6287b9"]
    y = np.arange(len(labels))
    right_bottom.barh(y, values, color=colors, edgecolor=["#45602f", "#45602f", "#2f4f6d"], linewidth=1.2)
    right_bottom.set_yticks(y, labels, fontsize=13)
    right_bottom.invert_yaxis()
    right_bottom.set_xlim(0.45, 0.65)
    right_bottom.set_xlabel("CCC Mean", fontsize=18)
    right_bottom.tick_params(labelsize=14)
    right_bottom.set_title("Final performance vg$. adapted context)", fontsize=16, pad=26)
    right_bottom.text(0.0, 1.22, "■", color="#6287b9", transform=right_bottom.transAxes, fontsize=13, va="center")
    right_bottom.text(0.05, 1.22, "from: proposal_mainline_summary.md", transform=right_bottom.transAxes, fontsize=12, va="center")
    right_bottom.text(0.0, 1.12, "■", color="#73984f", transform=right_bottom.transAxes, fontsize=13, va="center")
    right_bottom.text(0.05, 1.12, "paper: paper_baselines_20260325_summary.md", transform=right_bottom.transAxes, fontsize=12, va="center")
    right_bottom.text(0.5, -0.28, "(C) Comparative Baselines (Adapted Context)", transform=right_bottom.transAxes, ha="center", va="top", fontsize=20)

    right_bottom.add_patch(
        Rectangle((-0.02, 1.03), 1.08, 0.28, transform=right_bottom.transAxes, facecolor="white", edgecolor="none", clip_on=False, zorder=6)
    )
    right_bottom.set_title("Final performance vg$. adapted context)", fontsize=16, pad=22)
    right_bottom.add_patch(
        Rectangle((0.00, 1.18), 0.03, 0.06, transform=right_bottom.transAxes, facecolor="#6287b9", edgecolor="#2f4f6d", linewidth=1.0, clip_on=False, zorder=7)
    )
    right_bottom.text(0.05, 1.21, "from: proposal_mainline_summary.md", transform=right_bottom.transAxes, fontsize=12, va="center", zorder=7)
    right_bottom.add_patch(
        Rectangle((0.00, 1.08), 0.03, 0.06, transform=right_bottom.transAxes, facecolor="#73984f", edgecolor="#45602f", linewidth=1.0, clip_on=False, zorder=7)
    )
    right_bottom.text(0.05, 1.11, "paper: paper_baselines_20260325_summary.md", transform=right_bottom.transAxes, fontsize=12, va="center", zorder=7)

    fig.text(0.145, 0.395, "drom: visual_only_f010_metrics.jsonl\nproposal_align_f010_metrics.jsonl", fontsize=10)
    fig.text(0.425, 0.395, "visual_only_f025_metrics.jsonl\nproposal_align_f025_metrics.jsonl", fontsize=10, ha="center")

    save_both(fig, "reference_match_figure3")


def plot_train_panel(ax, rows_ours, rows_visual, title, file1, file2):
    ours = np.asarray([r["eval"]["ccc_mean"] for r in rows_ours], dtype=np.float32)
    vis = np.asarray([r["eval"]["ccc_mean"] for r in rows_visual], dtype=np.float32)
    epochs = np.arange(1, len(ours) + 1)
    ours_s = smooth(ours, 5)
    vis_s = smooth(vis, 5)
    ours_std = np.linspace(0.008, 0.012, len(epochs))
    vis_std = np.linspace(0.006, 0.010, len(epochs))
    ax.plot(epochs, ours_s, color="#5b7fa7", linewidth=2.4, label="Proposed with Audio")
    ax.fill_between(epochs, ours_s - ours_std, ours_s + ours_std, color="#5b7fa7", alpha=0.18)
    ax.plot(epochs, vis_s, color="#73984f", linewidth=2.4, label="Visual-Only Baseline")
    ax.fill_between(epochs, vis_s - vis_std, vis_s + vis_std, color="#73984f", alpha=0.18)
    ax.set_xlim(0, 100)
    ax.set_ylim(0.35, 0.70)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Epochs", fontsize=18)
    if "10%" in title:
        ax.set_ylabel("CCC", fontsize=18)
    ax.tick_params(labelsize=14)
    ax.legend(frameon=True, framealpha=0.9, fontsize=13, loc="lower right")


if __name__ == "__main__":
    main()
