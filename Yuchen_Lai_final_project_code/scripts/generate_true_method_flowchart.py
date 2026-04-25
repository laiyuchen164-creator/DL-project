from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT_ROOT / "icml2026_template" / "figures"
SAMPLE_DIR = PROJECT_ROOT / "figure_handoff" / "samples"


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "axes.titlepad": 6,
        }
    )
    create_flowchart()


def save_both(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def load_assets() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    veatic = torch.load(SAMPLE_DIR / "veatic_window_60_808_816.pt", map_location="cpu")
    deam = torch.load(SAMPLE_DIR / "deam_song_1802_raw_len30p0.pt", map_location="cpu")
    frames = veatic["video"].permute(0, 2, 3, 1).numpy()
    veatic_wave = veatic["paired_audio"].squeeze(0).numpy()
    deam_wave = deam["audio"].mean(dim=0).numpy()
    return frames, veatic_wave, deam_wave


def most_energetic_segment(waveform: np.ndarray, length: int) -> np.ndarray:
    if len(waveform) <= length:
        return waveform
    energy = np.convolve(waveform ** 2, np.ones(length, dtype=np.float32), mode="valid")
    start = int(np.argmax(energy))
    return waveform[start : start + length]


def waveform_envelope(waveform: np.ndarray, bins: int = 180, segment_length: int = 6000) -> np.ndarray:
    segment = most_energetic_segment(waveform, min(len(waveform), segment_length))
    if len(segment) < bins:
        segment = np.pad(segment, (0, bins - len(segment)))
    step = max(1, len(segment) // bins)
    trimmed = segment[: step * bins].reshape(bins, step)
    env = np.sqrt(np.mean(trimmed ** 2, axis=1))
    env = env / (np.max(env) + 1e-6)
    if float(env.std()) < 0.08:
        t = np.linspace(0, 1, bins)
        guide = 0.22 + 0.58 * (0.4 * np.sin(2.0 * np.pi * t) ** 2 + 0.6 * np.exp(-((t - 0.55) / 0.18) ** 2))
        env = 0.4 * env + 0.6 * guide
        env = env / (np.max(env) + 1e-6)
    return env


def add_arrow(ax, start, end, color="#666666", lw=2.1, style="-|>", ls="-", zorder=3) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=18,
            linewidth=lw,
            color=color,
            linestyle=ls,
            shrinkA=0,
            shrinkB=0,
            zorder=zorder,
        )
    )


def box(
    ax,
    xy,
    w,
    h,
    text,
    *,
    fc="#f3f3f3",
    ec="#444444",
    fontsize=15,
    weight="regular",
    radius=0.018,
    zorder=2,
):
    patch = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        linewidth=1.6,
        edgecolor=ec,
        facecolor=fc,
        zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + w / 2,
        xy[1] + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        zorder=zorder + 1,
    )


def draw_wave_icon(parent_ax, bbox, waveform, color="#4c72b0") -> None:
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
    env = waveform_envelope(waveform)
    xs = np.arange(len(env))
    ax.fill_between(xs, env, -env, color=color, alpha=0.24)
    ax.plot(xs, env, color=color, lw=1.2)
    ax.plot(xs, -env, color=color, lw=1.2)
    ax.set_xlim(0, len(env) - 1)
    ax.set_ylim(-1.05, 1.05)
    ax.axis("off")


def draw_frame_stack(ax, frames, x0, y0, w, h) -> None:
    selected = frames[[0, 2, 5]]
    offsets = [(0.0, 0.0), (0.015, 0.015), (0.03, 0.03)]
    for frame, (dx, dy) in zip(selected, offsets):
        ax.imshow(frame, extent=(x0 + dx, x0 + w + dx, y0 + dy, y0 + h + dy), aspect="auto", zorder=2)
        ax.add_patch(
            Rectangle((x0 + dx, y0 + dy), w, h, fill=False, edgecolor="#222222", linewidth=1.4, zorder=3)
        )


def create_flowchart() -> None:
    frames, veatic_wave, deam_wave = load_assets()

    fig = plt.figure(figsize=(15, 8.2))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Background panels.
    ax.add_patch(Rectangle((0.00, 0.06), 0.79, 0.90, facecolor="#eef3fb", edgecolor="none"))
    ax.add_patch(Rectangle((0.79, 0.06), 0.21, 0.90, facecolor="#edf6e4", edgecolor="none"))
    ax.add_patch(Rectangle((0.02, 0.55), 0.69, 0.33, fill=False, edgecolor="#73839a", linewidth=1.2, linestyle=(0, (4, 3))))
    ax.add_patch(Rectangle((0.02, 0.15), 0.69, 0.31, fill=False, edgecolor="#73839a", linewidth=1.2, linestyle=(0, (4, 3))))

    ax.text(0.37, 0.95, "Training-Time Audio Guidance (Actual Implementation)", ha="center", va="center", fontsize=24, fontweight="semibold")
    ax.text(0.895, 0.95, "Testing / Deployment", ha="center", va="center", fontsize=23, fontweight="semibold")

    ax.text(0.045, 0.89, "Step 1. DEAM audio-supervised update", ha="left", va="center", fontsize=18, fontweight="semibold")
    ax.text(0.045, 0.49, "Step 2. VEATIC paired update", ha="left", va="center", fontsize=18, fontweight="semibold")

    # Step 1: DEAM branch.
    draw_wave_icon(ax, (0.04, 0.70, 0.08, 0.10), deam_wave, color="#5b7fa7")
    ax.text(0.08, 0.66, "DEAM audio\n(30 s crop)", ha="center", va="center", fontsize=16)
    box(ax, (0.14, 0.68), 0.12, 0.10, "Log-power\nspectrogram", fc="#dde8f7", ec="#5579a3", fontsize=16)
    box(ax, (0.31, 0.67), 0.14, 0.12, "Audio Encoder $f_A$\n(SimpleAudioCNN)", fc="#c7d7f0", ec="#5579a3", fontsize=17)
    box(ax, (0.50, 0.68), 0.12, 0.10, "Shared VA\nRegressor $g$", fc="#ececec", ec="#4d4d4d", fontsize=16)
    box(ax, (0.65, 0.68), 0.05, 0.10, "$\\mathcal{L}_{audio}$", fc="#f8e8a8", ec="#9b8444", fontsize=16, weight="semibold")

    add_arrow(ax, (0.12, 0.73), (0.14, 0.73), color="#5579a3")
    add_arrow(ax, (0.26, 0.73), (0.31, 0.73), color="#5579a3")
    add_arrow(ax, (0.45, 0.73), (0.50, 0.73))
    add_arrow(ax, (0.62, 0.73), (0.65, 0.73), color="#9b8444")

    ax.text(0.675, 0.62, "$\\mathcal{L}_{reg}=0.5\\,\\ell_1+0.5\\,(1-\\mathrm{CCC})$", ha="center", va="center", fontsize=14)
    ax.text(0.38, 0.60, "Target: song-level mean valence / arousal", ha="center", va="center", fontsize=14, style="italic")

    # Step 2: VEATIC paired update.
    draw_wave_icon(ax, (0.04, 0.31, 0.08, 0.10), veatic_wave, color="#5b7fa7")
    ax.text(0.08, 0.27, "VEATIC paired\naudio window", ha="center", va="center", fontsize=16)
    draw_frame_stack(ax, frames, 0.04, 0.17, 0.08, 0.11)
    ax.text(0.08, 0.145, "VEATIC video clip\n(8 frames)", ha="center", va="top", fontsize=16)

    box(ax, (0.14, 0.29), 0.12, 0.10, "Log-power\nspectrogram", fc="#dde8f7", ec="#5579a3", fontsize=16)
    box(ax, (0.31, 0.28), 0.14, 0.12, "Audio Encoder $f_A$\n(shared weights)", fc="#c7d7f0", ec="#5579a3", fontsize=17)
    box(ax, (0.49, 0.30), 0.09, 0.08, "Audio\nembedding $z_A$", fc="#e9f0fb", ec="#5579a3", fontsize=15)

    box(ax, (0.14, 0.15), 0.12, 0.10, "Resize to\n$112\\times112$", fc="#e7f3df", ec="#6f9647", fontsize=16)
    box(ax, (0.31, 0.14), 0.14, 0.12, "Video Encoder $f_V$\nR(2+1)D-18", fc="#dcefc0", ec="#6f9647", fontsize=17)
    box(ax, (0.49, 0.16), 0.09, 0.08, "Video\nembedding $z_V$", fc="#eef7e1", ec="#6f9647", fontsize=15)
    box(ax, (0.61, 0.14), 0.11, 0.12, "Shared VA\nRegressor $g$\n(shared weights)", fc="#ececec", ec="#4d4d4d", fontsize=15)
    box(ax, (0.73, 0.15), 0.05, 0.10, "$\\mathcal{L}_{visual}$", fc="#f8e8a8", ec="#9b8444", fontsize=16, weight="semibold")

    box(ax, (0.60, 0.32), 0.11, 0.09, "Symmetric\nInfoNCE", fc="#f7f1d7", ec="#9b8444", fontsize=15)
    box(ax, (0.73, 0.32), 0.05, 0.09, "$\\mathcal{L}_{align}$", fc="#f8e8a8", ec="#9b8444", fontsize=15, weight="semibold")

    add_arrow(ax, (0.12, 0.34), (0.14, 0.34), color="#5579a3")
    add_arrow(ax, (0.26, 0.34), (0.31, 0.34), color="#5579a3")
    add_arrow(ax, (0.45, 0.34), (0.49, 0.34), color="#5579a3")

    add_arrow(ax, (0.12, 0.205), (0.14, 0.205), color="#6f9647")
    add_arrow(ax, (0.26, 0.205), (0.31, 0.205), color="#6f9647")
    add_arrow(ax, (0.45, 0.205), (0.49, 0.205), color="#6f9647")
    add_arrow(ax, (0.58, 0.205), (0.61, 0.205))
    add_arrow(ax, (0.72, 0.205), (0.73, 0.205), color="#9b8444")

    add_arrow(ax, (0.58, 0.34), (0.60, 0.365), color="#5579a3", style="->", ls=(0, (4, 3)))
    add_arrow(ax, (0.58, 0.205), (0.60, 0.355), color="#6f9647", style="->", ls=(0, (4, 3)))
    add_arrow(ax, (0.71, 0.365), (0.73, 0.365), color="#9b8444")

    ax.text(0.57, 0.295, "Target: window-level mean valence / arousal", ha="left", va="center", fontsize=14, style="italic")
    ax.text(0.41, 0.095, "Step 2 optimizes $\\mathcal{L}_{visual} + 0.1\\,\\mathcal{L}_{align}$ on VEATIC.", ha="center", va="center", fontsize=14)

    # Testing panel.
    draw_frame_stack(ax, frames, 0.835, 0.18, 0.09, 0.12)
    ax.text(0.88, 0.15, "Test video clip", ha="center", va="center", fontsize=17)
    box(ax, (0.81, 0.41), 0.15, 0.14, "Video Encoder $f_V$\nR(2+1)D-18\n(same weights)", fc="#dcefc0", ec="#6f9647", fontsize=18)
    box(ax, (0.81, 0.63), 0.15, 0.12, "Shared VA\nRegressor $g$\n(same weights)", fc="#ececec", ec="#4d4d4d", fontsize=18)
    box(ax, (0.82, 0.82), 0.13, 0.07, "Predicted\nvalence / arousal", fc="#f7f1d7", ec="#9b8444", fontsize=17)

    add_arrow(ax, (0.885, 0.30), (0.885, 0.41), color="#6f9647")
    add_arrow(ax, (0.885, 0.55), (0.885, 0.63), color="#4d4d4d")
    add_arrow(ax, (0.885, 0.75), (0.885, 0.82), color="#9b8444")

    ax.text(0.895, 0.34, "Audio branch is removed at test time.", ha="center", va="center", fontsize=15, style="italic")
    ax.text(0.895, 0.09, "Evaluation on VEATIC test uses only video input.", ha="center", va="center", fontsize=14)

    save_both(fig, "true_method_flowchart")


if __name__ == "__main__":
    main()
