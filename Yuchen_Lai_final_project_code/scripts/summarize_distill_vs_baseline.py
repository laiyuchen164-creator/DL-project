from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize distillation results against the frozen baseline.")
    parser.add_argument("--baseline-run", required=True)
    parser.add_argument("--candidate-run", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    baseline_records = load_metrics(Path(args.baseline_run) / "metrics.jsonl")
    candidate_records = load_metrics(Path(args.candidate_run) / "metrics.jsonl")

    baseline_rows = flatten_records("baseline_gru_noalign", baseline_records, args.baseline_run)
    candidate_rows = flatten_records("distill_gru_noalign", candidate_records, args.candidate_run)
    all_rows = baseline_rows + candidate_rows
    best_baseline = max(baseline_rows, key=lambda row: row["eval_ccc_mean"])
    best_candidate = max(candidate_rows, key=lambda row: row["eval_ccc_mean"])
    delta = best_candidate["eval_ccc_mean"] - best_baseline["eval_ccc_mean"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "distill_vs_baseline_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    summary = {
        "baseline_best": best_baseline,
        "candidate_best": best_candidate,
        "delta_ccc_mean": delta,
        "claim_clear_winner": abs(delta) >= 0.02,
    }
    (output_dir / "distill_vs_baseline_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Distillation vs Baseline Summary",
        "",
        f"- baseline best epoch: {best_baseline['epoch']}, eval.ccc_mean={best_baseline['eval_ccc_mean']:.4f}, eval.mae={best_baseline['eval_mae']:.4f}, eval.rmse={best_baseline['eval_rmse']:.4f}",
        f"- candidate best epoch: {best_candidate['epoch']}, eval.ccc_mean={best_candidate['eval_ccc_mean']:.4f}, eval.mae={best_candidate['eval_mae']:.4f}, eval.rmse={best_candidate['eval_rmse']:.4f}",
        f"- delta (candidate - baseline) on eval.ccc_mean: {delta:.4f}",
        (
            "- difference is >= 0.02, candidate can be treated as clearly better"
            if abs(delta) >= 0.02
            else "- difference is < 0.02, do not claim a clear winner"
        ),
        "",
        "| run | epoch | eval_visual_loss | eval_mae | eval_rmse | eval_ccc_valence | eval_ccc_arousal | eval_ccc_mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in all_rows:
        lines.append(
            "| {run} | {epoch} | {eval_visual_loss:.4f} | {eval_mae:.4f} | {eval_rmse:.4f} | "
            "{eval_ccc_valence:.4f} | {eval_ccc_arousal:.4f} | {eval_ccc_mean:.4f} |".format(**row)
        )
    (output_dir / "distill_vs_baseline_summary.md").write_text("\n".join(lines), encoding="utf-8")


def load_metrics(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def flatten_records(run_name: str, records: list[dict], run_dir: str) -> list[dict]:
    rows = []
    for record in records:
        eval_metrics = record["eval"]
        rows.append(
            {
                "run": run_name,
                "run_dir": run_dir,
                "epoch": int(record["epoch"]),
                "eval_visual_loss": float(eval_metrics["visual_loss"]),
                "eval_mae": float(eval_metrics["mae"]),
                "eval_rmse": float(eval_metrics["rmse"]),
                "eval_ccc_valence": float(eval_metrics["ccc_valence"]),
                "eval_ccc_arousal": float(eval_metrics["ccc_arousal"]),
                "eval_ccc_mean": float(eval_metrics["ccc_mean"]),
            }
        )
    return rows


if __name__ == "__main__":
    main()
