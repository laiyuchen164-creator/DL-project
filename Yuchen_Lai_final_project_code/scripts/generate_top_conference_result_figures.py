from __future__ import annotations

import csv
import json
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "latex" / "figures" / "generated"

NAVY = "#1F4E79"
ORANGE = "#D55E00"
TEAL = "#009E73"
RED = "#C44E52"
GRAY = "#6C757D"
LIGHT_GRAY = "#D8DEE9"
GOLD = "#E69F00"


def main() -> None:
    configure_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = read_mainline_summary(PROJECT_ROOT / "runs" / "experiment_summaries" / "proposal_mainline_summary.csv")
    formal_runs = load_formal_runs()
    veatic = load_veatic_window_targets(PROJECT_ROOT / "data" / "veatic" / "VEATIC.zip")
    deam_targets = load_deam_dynamic_targets(
        PROJECT_ROOT
        / "data"
        / "deam"
        / "annotations"
        / "annotations"
        / "annotations averaged per song"
        / "dynamic (per second annotations)"
    )

    saved: list[Path] = []
    saved += make_formal_metric_comparison(summary)
    saved += make_contextual_baseline_figure()
    saved += make_training_curve_figure(formal_runs)
    saved += make_va_breakdown_figure(formal_runs)
    saved += make_alignment_lambda_scout_figure(summary)
    saved += make_model_ablation_figure()
    saved += make_data_scale_figure(deam_targets, veatic)
    saved += make_label_space_figure(deam_targets, veatic)
    saved += make_veatic_window_distribution_figure(veatic)

    print("Generated figures:")
    for path in saved:
        print(f"  {path}")


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 7.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "grid.color": "#E7EAF0",
            "grid.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 320,
        }
    )


def save_both(fig: plt.Figure, stem: str) -> list[Path]:
    pdf = OUT_DIR / f"{stem}.pdf"
    png = OUT_DIR / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=320)
    plt.close(fig)
    return [pdf, png]


def read_mainline_summary(path: Path) -> dict[tuple[str, str], dict[str, float | int | str]]:
    rows: dict[tuple[str, str], dict[str, float | int | str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            converted: dict[str, float | int | str] = dict(row)
            converted["best_epoch"] = int(row["best_epoch"])
            converted["best_ccc_mean"] = float(row["best_ccc_mean"])
            converted["mae"] = float(row["mae"])
            converted["rmse"] = float(row["rmse"])
            rows[(row["setting"], row["method"])] = converted
    return rows


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def eval_series(path: Path) -> dict[str, np.ndarray]:
    rows = read_jsonl(path)
    keys = ["visual_loss", "mae", "rmse", "ccc_valence", "ccc_arousal", "ccc_mean"]
    out: dict[str, list[float]] = {"epoch": []}
    for key in keys:
        out[key] = []
    for row in rows:
        out["epoch"].append(float(row["epoch"]))
        for key in keys:
            out[key].append(float(row["eval"][key]))
    return {key: np.asarray(value, dtype=float) for key, value in out.items()}


def best_eval(path: Path) -> dict[str, float]:
    rows = read_jsonl(path)
    best = max(rows, key=lambda row: float(row["eval"]["ccc_mean"]))
    out = {"epoch": float(best["epoch"])}
    out.update({key: float(value) for key, value in best["eval"].items()})
    return out


def load_formal_runs() -> dict[str, dict[str, dict[str, np.ndarray] | dict[str, float]]]:
    specs = {
        "25% labels": {
            "Visual-only": PROJECT_ROOT / "runs" / "proposal_r2plus1d_visual_only_f025_formal_e100_v1" / "metrics.jsonl",
            "Proposed": PROJECT_ROOT / "runs" / "proposal_r2plus1d_align_l01_f025_formal_e100_v1" / "metrics.jsonl",
        },
        "10% labels": {
            "Visual-only": PROJECT_ROOT / "runs" / "proposal_r2plus1d_visual_only_f010_formal_e100_v1" / "metrics.jsonl",
            "Proposed": PROJECT_ROOT / "runs" / "proposal_r2plus1d_align_l01_f010_formal_e100_v1" / "metrics.jsonl",
        },
    }
    loaded: dict[str, dict[str, dict[str, np.ndarray] | dict[str, float]]] = {}
    for regime, methods in specs.items():
        loaded[regime] = {}
        for method, path in methods.items():
            loaded[regime][method] = {
                "series": eval_series(path),
                "best": best_eval(path),
            }
    return loaded


def make_formal_metric_comparison(summary: dict[tuple[str, str], dict[str, float | int | str]]) -> list[Path]:
    regimes = [("25% labels", "f025_formal_e100"), ("10% labels", "f010_formal_e100")]
    methods = [("Visual-only", "visual_only"), ("Proposed", "proposed_lambda_0.1")]
    metrics = [
        ("best_ccc_mean", "CCC mean", "Higher is better", (0.59, 0.65)),
        ("mae", "MAE", "Lower is better", (0.185, 0.201)),
        ("rmse", "RMSE", "Lower is better", (0.254, 0.262)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.35), constrained_layout=True)
    x = np.arange(len(regimes))
    width = 0.34

    for ax, (metric, ylabel, title, ylim) in zip(axes, metrics):
        vals = np.asarray(
            [
                [float(summary[(setting, method_key)][metric]) for _, method_key in methods]
                for _, setting in regimes
            ],
            dtype=float,
        )
        ax.bar(
            x - width / 2,
            vals[:, 0],
            width,
            color="#E6E8EB",
            edgecolor=GRAY,
            linewidth=0.8,
            label="Visual-only",
        )
        ax.bar(
            x + width / 2,
            vals[:, 1],
            width,
            color="#F4A261",
            edgecolor=ORANGE,
            linewidth=0.8,
            label="Proposed",
        )
        ax.set_xticks(x, [label for label, _ in regimes])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(*ylim)
        ax.grid(axis="y")
        for idx in range(len(regimes)):
            for jdx, offset in enumerate([-width / 2, width / 2]):
                ax.text(
                    idx + offset,
                    vals[idx, jdx] + (ylim[1] - ylim[0]) * 0.025,
                    f"{vals[idx, jdx]:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.4,
                )
            if metric == "best_ccc_mean":
                delta = vals[idx, 1] - vals[idx, 0]
                ax.text(
                    idx,
                    max(vals[idx]) + (ylim[1] - ylim[0]) * 0.14,
                    f"Delta {delta:+.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.6,
                    color=TEAL if delta >= 0 else RED,
                    fontweight="bold",
                )
    axes[0].legend(frameon=False, loc="lower left")
    fig.suptitle("Formal low-supervision VEATIC test performance", y=1.05, fontsize=10.5, fontweight="bold")
    return save_both(fig, "results_formal_low_supervision_metrics")


def make_contextual_baseline_figure() -> list[Path]:
    baseline_labels = ["Ortega\nfeature", "Ortega\ndecision", "Zhang\nLFAN"]
    baseline_scores = np.asarray([0.5167, 0.5125, 0.5733], dtype=float)
    ours_labels = ["Ours\n25%", "Ours\n10%"]
    ours_scores = np.asarray([0.6392, 0.6080], dtype=float)

    fig, ax = plt.subplots(figsize=(4.9, 2.75), constrained_layout=True)
    baseline_x = np.arange(len(baseline_labels))
    ours_x = np.arange(len(ours_labels)) + len(baseline_labels) + 0.85
    ax.bar(
        baseline_x,
        baseline_scores,
        color=["#C8D7EB", "#DCE6F2", "#9DB9DD"],
        edgecolor=NAVY,
        linewidth=0.75,
        label="Contextual paper baselines",
    )
    ax.bar(
        ours_x,
        ours_scores,
        color=[ORANGE, "#F4A261"],
        edgecolor=ORANGE,
        linewidth=0.75,
        label="This project",
    )
    for xpos, score in zip(list(baseline_x) + list(ours_x), list(baseline_scores) + list(ours_scores)):
        ax.text(xpos, score + 0.006, f"{score:.3f}", ha="center", va="bottom", fontsize=7.7)
    ax.axvspan(len(baseline_labels) - 0.35, len(baseline_labels) + 0.35, color="#F1F3F5", zorder=-1)
    ax.text(
        len(baseline_labels) + 0.25,
        0.665,
        "not matched\ntraining protocol",
        ha="center",
        va="top",
        fontsize=7.2,
        color=GRAY,
    )
    ax.set_xticks(list(baseline_x) + list(ours_x), baseline_labels + ours_labels)
    ax.set_ylabel("CCC mean")
    ax.set_ylim(0.49, 0.68)
    ax.set_title("Contextual comparison against literature baselines")
    ax.grid(axis="y")
    ax.legend(frameon=False, loc="upper left")
    return save_both(fig, "results_contextual_baseline_comparison")


def make_training_curve_figure(
    formal_runs: dict[str, dict[str, dict[str, np.ndarray] | dict[str, float]]]
) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.55), constrained_layout=True, sharey=True)
    colors = {"Visual-only": GRAY, "Proposed": ORANGE}
    linestyles = {"Visual-only": "-", "Proposed": "-"}

    for ax, regime in zip(axes, ["25% labels", "10% labels"]):
        for method in ["Visual-only", "Proposed"]:
            payload = formal_runs[regime][method]
            series = payload["series"]  # type: ignore[index]
            best = payload["best"]  # type: ignore[index]
            epoch = series["epoch"]
            ccc = series["ccc_mean"]
            smoothed = smooth(ccc, 5)
            ax.plot(epoch, ccc, color=colors[method], alpha=0.16, linewidth=0.8)
            ax.plot(
                epoch,
                smoothed,
                color=colors[method],
                linewidth=1.75 if method == "Proposed" else 1.55,
                linestyle=linestyles[method],
                label=method,
            )
            ax.scatter(
                [best["epoch"]],
                [best["ccc_mean"]],
                s=26,
                color=colors[method],
                edgecolor="white",
                linewidth=0.7,
                zorder=4,
            )
        ax.set_title(regime)
        ax.set_xlabel("Epoch")
        ax.grid(axis="y")
        ax.set_xlim(1, 100)
        ax.set_ylim(0.50, 0.665)
    axes[0].set_ylabel("Eval CCC mean")
    axes[0].legend(frameon=False, loc="lower right")
    fig.suptitle("Evaluation dynamics across alternating training epochs", y=1.05, fontsize=10.5, fontweight="bold")
    return save_both(fig, "results_formal_training_curves")


def make_va_breakdown_figure(
    formal_runs: dict[str, dict[str, dict[str, np.ndarray] | dict[str, float]]]
) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4), constrained_layout=True, sharey=True)
    dims = ["Valence", "Arousal"]
    dim_keys = ["ccc_valence", "ccc_arousal"]
    x = np.arange(len(dims))
    width = 0.34

    for ax, regime in zip(axes, ["25% labels", "10% labels"]):
        visual = formal_runs[regime]["Visual-only"]["best"]  # type: ignore[index]
        proposed = formal_runs[regime]["Proposed"]["best"]  # type: ignore[index]
        visual_vals = np.asarray([visual[key] for key in dim_keys], dtype=float)
        proposed_vals = np.asarray([proposed[key] for key in dim_keys], dtype=float)
        ax.bar(x - width / 2, visual_vals, width, color="#E6E8EB", edgecolor=GRAY, linewidth=0.8, label="Visual-only")
        ax.bar(x + width / 2, proposed_vals, width, color="#F4A261", edgecolor=ORANGE, linewidth=0.8, label="Proposed")
        for idx, value in enumerate(visual_vals):
            ax.text(idx - width / 2, value + 0.006, f"{value:.3f}", ha="center", va="bottom", fontsize=7.2)
        for idx, value in enumerate(proposed_vals):
            ax.text(idx + width / 2, value + 0.006, f"{value:.3f}", ha="center", va="bottom", fontsize=7.2)
        ax.set_xticks(x, dims)
        ax.set_title(regime)
        ax.set_ylim(0.56, 0.665)
        ax.grid(axis="y")
    axes[0].set_ylabel("CCC at best mean-CCC epoch")
    axes[0].legend(frameon=False, loc="lower right")
    fig.suptitle("Valence/arousal breakdown for the selected checkpoints", y=1.06, fontsize=10.5, fontweight="bold")
    return save_both(fig, "results_va_dimension_breakdown")


def make_alignment_lambda_scout_figure(summary: dict[tuple[str, str], dict[str, float | int | str]]) -> list[Path]:
    rows = []
    for (setting, method), row in summary.items():
        if setting == "f025_scout_e20" and method.startswith("proposed_lambda_"):
            lam = float(method.replace("proposed_lambda_", ""))
            rows.append((lam, float(row["best_ccc_mean"]), float(row["mae"]), float(row["rmse"])))
    rows.sort(key=lambda item: item[0])
    lambdas = [item[0] for item in rows]
    ccc = np.asarray([item[1] for item in rows], dtype=float)
    labels = ["0" if value == 0 else f"{value:g}" for value in lambdas]
    x = np.arange(len(labels))
    visual = float(summary[("f025_scout_e20", "visual_only")]["best_ccc_mean"])

    fig, ax = plt.subplots(figsize=(4.25, 2.55), constrained_layout=True)
    ax.plot(x, ccc, color=ORANGE, linewidth=1.9, marker="o", markersize=5.2)
    ax.axhline(visual, color=GRAY, linestyle="--", linewidth=1.2)
    best_idx = int(np.argmax(ccc))
    ax.scatter([x[best_idx]], [ccc[best_idx]], s=58, color=TEAL, edgecolor="white", linewidth=0.8, zorder=4)
    label_y = visual - 0.0012
    ax.annotate(
        f"Best lambda={labels[best_idx]}\nCCC={ccc[best_idx]:.3f}",
        xy=(x[best_idx], ccc[best_idx]),
        xytext=(x[best_idx] - 1.1, label_y),
        arrowprops=dict(arrowstyle="->", lw=0.8, color=TEAL),
        fontsize=7.6,
        color=TEAL,
        va="top",
    )
    for idx, value in enumerate(ccc):
        ax.text(idx, value - 0.0065, f"{value:.3f}", ha="center", va="top", fontsize=7.1)
    ax.set_xticks(x, labels)
    ax.set_xlabel(r"Alignment weight $\lambda_{\mathrm{align}}$")
    ax.set_ylabel("Best eval CCC mean")
    ax.set_ylim(0.595, 0.647)
    ax.set_title("Alignment-weight scout on 25% VEATIC labels")
    ax.grid(axis="y")
    return save_both(fig, "results_alignment_lambda_scout")


def make_model_ablation_figure() -> list[Path]:
    temporal = best_by_run_from_csv(PROJECT_ROOT / "runs" / "experiment_summaries" / "temporal_model_suite_metrics.csv")
    temporal_labels = ["GRU", "Transformer"]
    temporal_values = [
        temporal.get("gru_no_alignment", np.nan),
        temporal.get("transformer_no_alignment", np.nan),
    ]

    backbone_specs = [
        ("R(2+1)D-18", PROJECT_ROOT / "runs" / "video_r2plus1d18_gru_noalign_formal_e100_v1" / "metrics.jsonl"),
        ("MViT-v2-S", PROJECT_ROOT / "runs" / "video_mvit_v2_s_gru_noalign_formal_e100_v1" / "metrics.jsonl"),
        (
            "VideoMAE\nfrozen",
            PROJECT_ROOT / "runs" / "video_videomae_base_k400_gru_noalign_frozen_formal_e100_v1" / "metrics.jsonl",
        ),
        (
            "VideoMAE\npartial",
            PROJECT_ROOT / "runs" / "video_videomae_base_k400_partial2_gru_noalign_formal_e100_v1" / "metrics.jsonl",
        ),
    ]
    backbone_values = [best_eval(path)["ccc_mean"] if path.exists() else np.nan for _, path in backbone_specs]

    fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.55), constrained_layout=True)
    draw_ablation_bars(
        axes[0],
        temporal_labels,
        temporal_values,
        [NAVY, "#86A6C9"],
        "Temporal head ablation",
        "Best eval CCC mean",
        (0.60, 0.67),
    )
    draw_ablation_bars(
        axes[1],
        [label for label, _ in backbone_specs],
        backbone_values,
        [NAVY, TEAL, "#B8C1CC", GOLD],
        "Video backbone ablation",
        "Best eval CCC mean",
        (0.59, 0.67),
    )
    fig.suptitle("Auxiliary visual-only ablations", y=1.06, fontsize=10.5, fontweight="bold")
    return save_both(fig, "results_auxiliary_model_ablations")


def best_by_run_from_csv(path: Path) -> dict[str, float]:
    best: dict[str, float] = defaultdict(lambda: -np.inf)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            value = float(row["eval_ccc_mean"])
            run = row["run"]
            if value > best[run]:
                best[run] = value
    return dict(best)


def draw_ablation_bars(
    ax: plt.Axes,
    labels: list[str],
    values: Iterable[float],
    colors: list[str],
    title: str,
    ylabel: str,
    ylim: tuple[float, float],
) -> None:
    values_arr = np.asarray(list(values), dtype=float)
    x = np.arange(len(labels))
    bars = ax.bar(x, values_arr, color=colors[: len(labels)], edgecolor="#2E3440", linewidth=0.75)
    ax.set_xticks(x, labels)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.grid(axis="y")
    for bar, value in zip(bars, values_arr):
        if np.isfinite(value):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + (ylim[1] - ylim[0]) * 0.025,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.3,
            )


def make_data_scale_figure(deam_targets: np.ndarray, veatic: dict[str, object]) -> list[Path]:
    train_count = int(len(veatic["train_targets"]))  # type: ignore[arg-type]
    test_count = int(len(veatic["test_targets"]))  # type: ignore[arg-type]
    train_counts_by_video = veatic["train_counts_by_video"]  # type: ignore[assignment]
    count_25 = fraction_count(train_counts_by_video, 0.25)  # type: ignore[arg-type]
    count_10 = fraction_count(train_counts_by_video, 0.10)  # type: ignore[arg-type]
    video_count = int(veatic["video_count"])  # type: ignore[arg-type]

    fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.65), constrained_layout=True)

    ax = axes[0]
    units = ["DEAM\nsongs", "VEATIC\nvideos"]
    values = [len(deam_targets), video_count]
    colors = [NAVY, ORANGE]
    bars = ax.bar(units, values, color=colors, edgecolor="#2E3440", linewidth=0.75)
    ax.set_title("Raw dataset units")
    ax.set_ylabel("Count")
    ax.grid(axis="y")
    ax.set_ylim(0, max(values) * 1.18)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + max(values) * 0.025, f"{value:,}", ha="center", fontsize=8)

    ax = axes[1]
    labels = ["Train\nwindows", "Test\nwindows", "25% labeled\ntrain", "10% labeled\ntrain"]
    values = [train_count, test_count, count_25, count_10]
    colors = [ORANGE, "#F6C28B", TEAL, "#A7D8C8"]
    bars = ax.bar(labels, values, color=colors, edgecolor="#2E3440", linewidth=0.75)
    ax.set_title("VEATIC supervision budget")
    ax.set_ylabel("8-frame windows")
    ax.grid(axis="y")
    ax.set_ylim(0, max(values) * 1.18)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + max(values) * 0.025, f"{value:,}", ha="center", fontsize=7.8)

    fig.suptitle("Dataset scale and low-label protocol", y=1.06, fontsize=10.5, fontweight="bold")
    return save_both(fig, "data_scale_and_supervision_protocol")


def make_label_space_figure(deam_targets: np.ndarray, veatic: dict[str, object]) -> list[Path]:
    train_targets = veatic["train_targets"]  # type: ignore[assignment]
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.0), constrained_layout=True, sharex=True, sharey=True)
    datasets = [
        ("DEAM song-level audio labels", deam_targets, "Blues"),
        ("VEATIC train-window labels", train_targets, "Oranges"),
    ]

    for ax, (title, targets, cmap) in zip(axes, datasets):
        hb = ax.hexbin(
            targets[:, 0],
            targets[:, 1],
            gridsize=34,
            extent=(-1, 1, -1, 1),
            mincnt=1,
            linewidths=0.0,
            cmap=cmap,
        )
        ax.axhline(0, color="#A0A7B3", linewidth=0.7, zorder=0)
        ax.axvline(0, color="#A0A7B3", linewidth=0.7, zorder=0)
        ax.set_title(title)
        ax.set_xlabel("Valence")
        ax.set_ylabel("Arousal")
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect("equal", adjustable="box")
        ax.text(
            -0.95,
            0.88,
            f"n={len(targets):,}",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#C9CED6", linewidth=0.6),
        )
        cbar = fig.colorbar(hb, ax=ax, fraction=0.045, pad=0.02)
        cbar.ax.tick_params(labelsize=7)
        cbar.set_label("Samples/bin", fontsize=7.5)

    fig.suptitle("Valence-arousal label-space coverage", y=1.03, fontsize=10.5, fontweight="bold")
    return save_both(fig, "data_va_label_space_coverage")


def make_veatic_window_distribution_figure(veatic: dict[str, object]) -> list[Path]:
    train_counts = np.asarray(list((veatic["train_counts_by_video"]).values()), dtype=float)  # type: ignore[union-attr]
    test_counts = np.asarray(list((veatic["test_counts_by_video"]).values()), dtype=float)  # type: ignore[union-attr]
    total_counts = train_counts + test_counts

    fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.6), constrained_layout=True)
    bins = np.linspace(0, max(total_counts) * 1.05, 14)

    axes[0].hist(train_counts, bins=bins, color=ORANGE, alpha=0.72, edgecolor="white", label="Train")
    axes[0].hist(test_counts, bins=bins, color=NAVY, alpha=0.55, edgecolor="white", label="Test")
    axes[0].axvline(np.median(train_counts), color=ORANGE, linestyle="--", linewidth=1.1)
    axes[0].axvline(np.median(test_counts), color=NAVY, linestyle="--", linewidth=1.1)
    axes[0].set_title("Per-video window counts")
    axes[0].set_xlabel("Windows per video")
    axes[0].set_ylabel("Number of videos")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y")

    sorted_counts = np.sort(total_counts)
    y = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts)
    axes[1].plot(sorted_counts, y, color=TEAL, linewidth=2.0)
    axes[1].fill_between(sorted_counts, 0, y, color=TEAL, alpha=0.10)
    axes[1].axvline(np.median(total_counts), color=TEAL, linestyle="--", linewidth=1.1)
    axes[1].text(np.median(total_counts) + 5, 0.1, f"median={np.median(total_counts):.0f}", color=TEAL, fontsize=7.8)
    axes[1].set_title("Cumulative window distribution")
    axes[1].set_xlabel("Total windows per video")
    axes[1].set_ylabel("Fraction of videos")
    axes[1].set_ylim(0, 1.02)
    axes[1].grid(axis="y")

    fig.suptitle("VEATIC temporal-window sampling profile", y=1.06, fontsize=10.5, fontweight="bold")
    return save_both(fig, "data_veatic_window_distribution")


def load_deam_dynamic_targets(dynamic_dir: Path) -> np.ndarray:
    valence = load_deam_dynamic_file(dynamic_dir / "valence.csv")
    arousal = load_deam_dynamic_file(dynamic_dir / "arousal.csv")
    targets: list[tuple[float, float]] = []
    for song_id in sorted(set(valence) & set(arousal), key=lambda item: int(item)):
        v = np.asarray(valence[song_id], dtype=float)
        a = np.asarray(arousal[song_id], dtype=float)
        length = min(len(v), len(a))
        if length:
            targets.append((float(v[:length].mean()), float(a[:length].mean())))
    return np.asarray(targets, dtype=float)


def load_deam_dynamic_file(path: Path) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            out[row["song_id"]] = [float(value) for key, value in row.items() if key != "song_id" and value not in ("", None)]
    return out


def load_veatic_window_targets(zip_path: Path, clip_length: int = 8, stride: int = 8) -> dict[str, object]:
    train_targets: list[tuple[float, float]] = []
    test_targets: list[tuple[float, float]] = []
    train_counts_by_video: dict[str, int] = {}
    test_counts_by_video: dict[str, int] = {}

    with zipfile.ZipFile(zip_path) as archive:
        video_ids = sorted(
            [Path(name).stem for name in archive.namelist() if name.startswith("videos/") and name.endswith(".mp4")],
            key=lambda item: int(item),
        )
        for video_id in video_ids:
            valence = read_veatic_rating(archive, f"rating_averaged/{video_id}_valence.csv")
            arousal = read_veatic_rating(archive, f"rating_averaged/{video_id}_arousal.csv")
            total_length = min(len(valence), len(arousal))
            split_point = max(1, int(total_length * 0.7))
            split_specs = {
                "train": (0, split_point, train_targets, train_counts_by_video),
                "test": (split_point, total_length, test_targets, test_counts_by_video),
            }
            for _, (start, end, target_store, count_store) in split_specs.items():
                count_before = len(target_store)
                if end - start <= clip_length:
                    if end > start:
                        target_store.append((float(np.mean(valence[start:end])), float(np.mean(arousal[start:end]))))
                else:
                    for window_start in range(start, end - clip_length + 1, stride):
                        window_end = window_start + clip_length
                        target_store.append(
                            (
                                float(np.mean(valence[window_start:window_end])),
                                float(np.mean(arousal[window_start:window_end])),
                            )
                        )
                count_store[video_id] = len(target_store) - count_before

    return {
        "video_count": len(train_counts_by_video),
        "train_targets": np.asarray(train_targets, dtype=float),
        "test_targets": np.asarray(test_targets, dtype=float),
        "train_counts_by_video": train_counts_by_video,
        "test_counts_by_video": test_counts_by_video,
    }


def read_veatic_rating(archive: zipfile.ZipFile, member: str) -> np.ndarray:
    raw = archive.read(member).decode("utf-8").splitlines()
    values = []
    for line in raw:
        if not line.strip():
            continue
        parts = line.split(",")
        values.append(float(parts[1]))
    return np.asarray(values, dtype=float)


def fraction_count(counts_by_video: dict[str, int], fraction: float) -> int:
    return int(sum(max(1, int(round(count * fraction))) for count in counts_by_video.values()))


def smooth(values: np.ndarray, width: int) -> np.ndarray:
    if width <= 1:
        return values.copy()
    pad = width // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(width, dtype=float) / width
    return np.convolve(padded, kernel, mode="valid")


if __name__ == "__main__":
    main()
