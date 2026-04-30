from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score the manual LLM evaluation sheet and build a summary."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="Filled manual_scores_template.csv file.",
    )
    parser.add_argument(
        "--group-by",
        type=str,
        default="model",
        choices=["model", "temperature"],
        help="Column used for aggregate summaries.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for scored outputs. Defaults to the input CSV folder.",
    )
    return parser.parse_args()


def parse_score(value: str) -> float | None:
    value = value.strip()
    if not value:
        return None
    return float(value)


def write_csv(output_path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_plot(summary_rows: list[dict], group_by: str, output_path: Path) -> None:
    x_values = [str(row[group_by]) for row in summary_rows]
    y_values = [float(row["average_manual_score"]) for row in summary_rows]
    plt.figure(figsize=(8, 5))
    plt.bar(x_values, y_values, color="#2563eb")
    plt.ylim(0, 5)
    plt.ylabel("Average Manual Score")
    plt.xlabel(group_by.capitalize())
    plt.title(f"Manual Score Summary by {group_by.capitalize()}")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.input_csv.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with args.input_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    scored_rows = []
    grouped_scores: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        scores = [
            parse_score(row.get("rater_1_score", "")),
            parse_score(row.get("rater_2_score", "")),
            parse_score(row.get("rater_3_score", "")),
        ]
        valid_scores = [score for score in scores if score is not None]
        average_score = (
            round(sum(valid_scores) / len(valid_scores), 4) if valid_scores else ""
        )
        scored_row = {**row, "average_score": average_score}
        scored_rows.append(scored_row)
        if valid_scores:
            grouped_scores[str(row[args.group_by])].append(float(average_score))

    summary_rows = []
    for group_value, scores in grouped_scores.items():
        summary_rows.append(
            {
                args.group_by: group_value,
                "average_manual_score": round(sum(scores) / len(scores), 4),
                "num_scored_rows": len(scores),
            }
        )

    scored_path = output_dir / "manual_scores_scored.csv"
    summary_path = output_dir / "manual_score_summary.csv"
    plot_path = output_dir / "manual_score_summary.png"
    write_csv(scored_path, scored_rows)
    write_csv(summary_path, summary_rows)
    if summary_rows:
        save_plot(summary_rows, args.group_by, plot_path)

    print(f"Saved scored sheet: {scored_path}")
    print(f"Saved summary: {summary_path}")
    if summary_rows:
        print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
