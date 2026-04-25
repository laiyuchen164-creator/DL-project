from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "latex" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "workshop_results.pdf"

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.6), constrained_layout=True)

    baseline_labels = ["Ortega\nfeat.", "Ortega\ndec.", "Zhang\nLFAN"]
    baseline_scores = np.array([0.5167, 0.5125, 0.5733], dtype=float)
    ours_25 = 0.6392
    ours_10 = 0.6080

    x0 = np.arange(len(baseline_labels))
    baseline_colors = ["#a8c6ff", "#c4d8ff", "#7ea0ff"]
    axes[0].bar(x0, baseline_scores, color=baseline_colors, edgecolor="#3559a8", linewidth=0.8)
    axes[0].axhline(ours_25, color="#c53b32", linestyle="--", linewidth=1.4)
    axes[0].axhline(ours_10, color="#ef8b42", linestyle=":", linewidth=1.6)
    for idx, score in enumerate(baseline_scores):
        axes[0].text(idx, score + 0.007, f"{score:.3f}", ha="center", va="bottom", fontsize=8)
    axes[0].set_title("Paper-derived baselines\n(full VEATIC train split)")
    axes[0].set_xticks(x0, baseline_labels)
    axes[0].set_ylim(0.45, 0.68)
    axes[0].set_xlim(-0.5, 2.7)
    axes[0].set_ylabel("CCC mean")
    axes[0].text(
        2.48,
        ours_25 + 0.003,
        "Ours (25%)",
        color="#c53b32",
        ha="right",
        va="bottom",
        fontsize=8,
    )
    axes[0].text(
        2.48,
        ours_10 + 0.003,
        "Ours (10%)",
        color="#ef8b42",
        ha="right",
        va="bottom",
        fontsize=8,
    )

    regimes = ["25% labels", "10% labels"]
    visual_only = np.array([0.6346, 0.6108], dtype=float)
    proposed = np.array([0.6392, 0.6080], dtype=float)
    x1 = np.arange(len(regimes))
    width = 0.34
    axes[1].bar(x1 - width / 2, visual_only, width, color="#d9d9d9", edgecolor="#666666", linewidth=0.8, label="Visual-only")
    axes[1].bar(x1 + width / 2, proposed, width, color="#ffb381", edgecolor="#c96b18", linewidth=0.8, label="Ours")
    for idx, (v_score, p_score) in enumerate(zip(visual_only, proposed)):
        axes[1].text(idx - width / 2, v_score + 0.006, f"{v_score:.3f}", ha="center", va="bottom", fontsize=8)
        axes[1].text(idx + width / 2, p_score + 0.006, f"{p_score:.3f}", ha="center", va="bottom", fontsize=8)
    axes[1].set_title("Matched low-supervision comparison")
    axes[1].set_xticks(x1, regimes)
    axes[1].set_ylim(0.56, 0.66)
    axes[1].legend(frameon=False, loc="upper right", fontsize=8)

    fig.savefig(output_path, bbox_inches="tight")
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
