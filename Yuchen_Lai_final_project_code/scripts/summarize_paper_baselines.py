from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize paper baseline results into markdown and csv.")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--prefix", default="paper_baselines_20260325")
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    runs_dir = args.runs_dir
    prefix = args.prefix

    ortega_feature = load_json(runs_dir / f"{prefix}_ortega_feature_svr" / "metrics.json")
    ortega_decision = load_json(runs_dir / f"{prefix}_ortega_decision_svr" / "metrics.json")
    zhang_records = load_jsonl(runs_dir / f"{prefix}_zhang_leader_follower" / "metrics.jsonl")
    zhang_best = max(zhang_records, key=lambda record: record["eval"]["ccc_mean"])
    zhang_final = zhang_records[-1]

    rows = [
        {
            "method": "Ortega et al. (feature-level fusion + SVR)",
            "source_id": "ortega_feature_svr",
            "epoch": "-",
            "mae": ortega_feature["mae"],
            "rmse": ortega_feature["rmse"],
            "ccc_valence": ortega_feature["ccc_valence"],
            "ccc_arousal": ortega_feature["ccc_arousal"],
            "ccc_mean": ortega_feature["ccc_mean"],
            "notes": "-",
        },
        {
            "method": "Ortega et al. (decision-level fusion + SVR)",
            "source_id": "ortega_decision_svr",
            "epoch": "-",
            "mae": ortega_decision["mae"],
            "rmse": ortega_decision["rmse"],
            "ccc_valence": ortega_decision["ccc_valence"],
            "ccc_arousal": ortega_decision["ccc_arousal"],
            "ccc_mean": ortega_decision["ccc_mean"],
            "notes": (
                f"alpha_v={ortega_decision['fusion_alpha_valence']:.1f}, "
                f"alpha_a={ortega_decision['fusion_alpha_arousal']:.1f}"
            ),
        },
        {
            "method": "Zhang et al. (leader-follower attentive fusion)",
            "source_id": "zhang_leader_follower_best",
            "epoch": zhang_best["epoch"],
            "mae": zhang_best["eval"]["mae"],
            "rmse": zhang_best["eval"]["rmse"],
            "ccc_valence": zhang_best["eval"]["ccc_valence"],
            "ccc_arousal": zhang_best["eval"]["ccc_arousal"],
            "ccc_mean": zhang_best["eval"]["ccc_mean"],
            "notes": "best eval epoch",
        },
        {
            "method": "Zhang et al. (leader-follower attentive fusion, final)",
            "source_id": "zhang_leader_follower_final",
            "epoch": zhang_final["epoch"],
            "mae": zhang_final["eval"]["mae"],
            "rmse": zhang_final["eval"]["rmse"],
            "ccc_valence": zhang_final["eval"]["ccc_valence"],
            "ccc_arousal": zhang_final["eval"]["ccc_arousal"],
            "ccc_mean": zhang_final["eval"]["ccc_mean"],
            "notes": "final epoch",
        },
    ]

    output_md = args.output_md or runs_dir / f"{prefix}_summary.md"
    output_csv = args.output_csv or runs_dir / f"{prefix}_summary.csv"

    output_md.write_text(render_markdown(rows), encoding="utf-8")
    write_csv(output_csv, rows)

    print(render_markdown(rows))
    print(f"\nWrote {output_md}")
    print(f"Wrote {output_csv}")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def render_markdown(rows: list[dict]) -> str:
    header = (
        "| Method | Epoch | MAE | RMSE | CCC-V | CCC-A | CCC-Mean | Notes |\n"
        "|---|---:|---:|---:|---:|---:|---:|---|\n"
    )
    body = "\n".join(
        (
            f"| {row['method']} | {row['epoch']} | {row['mae']:.4f} | {row['rmse']:.4f} | "
            f"{row['ccc_valence']:.4f} | {row['ccc_arousal']:.4f} | {row['ccc_mean']:.4f} | {row['notes']} |"
        )
        for row in rows
    )
    return header + body


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "source_id",
                "epoch",
                "mae",
                "rmse",
                "ccc_valence",
                "ccc_arousal",
                "ccc_mean",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
