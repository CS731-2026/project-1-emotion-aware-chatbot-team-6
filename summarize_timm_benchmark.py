from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


DEFAULT_MODEL_ORDER = [
    "resnet50",
    "efficientnet_b0",
    "efficientnet_b3",
    "swin_tiny",
    "mobilenet_v2",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize timm benchmark runs into a comparison plot and table."
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path(r"G:\731\runs_timm"),
        help="Root directory that stores timm benchmark run folders.",
    )
    parser.add_argument(
        "--run-names",
        nargs="+",
        default=DEFAULT_MODEL_ORDER,
        help="Run folder names to compare.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"G:\731\runs_timm\comparison"),
        help="Directory for the comparison outputs.",
    )
    return parser.parse_args()


def load_history(history_path: Path) -> list[dict[str, str]]:
    with history_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_metrics(metrics_path: Path) -> dict:
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def write_summary_csv(output_path: Path, rows: list[dict[str, str]]) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "best_val_accuracy",
                "best_epoch",
                "avg_inference_ms_per_image",
                "images_per_second",
                "num_parameters",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_summary_markdown(output_path: Path, rows: list[dict[str, str]]) -> None:
    lines = [
        "| Model | Best Val Accuracy (%) | Best Epoch | Avg Inference (ms/image) | Images/sec | Parameters |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["model"],
                    row["best_val_accuracy"],
                    row["best_epoch"],
                    row["avg_inference_ms_per_image"],
                    row["images_per_second"],
                    row["num_parameters"],
                ]
            )
            + " |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    summary_rows: list[dict[str, str]] = []
    max_epoch = 0

    for run_name in args.run_names:
        run_dir = args.runs_root / run_name
        history_path = run_dir / "history.csv"
        metrics_path = run_dir / "metrics.json"
        if not history_path.exists() or not metrics_path.exists():
            raise FileNotFoundError(
                f"Missing benchmark files for run '{run_name}' under {run_dir}."
            )

        history_rows = load_history(history_path)
        metrics = load_metrics(metrics_path)
        epochs = [int(row["epoch"]) for row in history_rows]
        val_accuracies = [float(row["val_accuracy"]) for row in history_rows]
        max_epoch = max(max_epoch, max(epochs, default=0))

        plt.plot(epochs, val_accuracies, marker="o", linewidth=2, label=metrics["display_name"])
        summary_rows.append(
            {
                "model": metrics["display_name"],
                "best_val_accuracy": f"{metrics['best_val_accuracy']:.4f}",
                "best_epoch": str(metrics["best_epoch"]),
                "avg_inference_ms_per_image": f"{metrics['avg_inference_ms_per_image']:.4f}",
                "images_per_second": f"{metrics['images_per_second']:.4f}",
                "num_parameters": str(metrics["num_parameters"]),
            }
        )

    plt.title("Validation Accuracy Comparison Across timm Models")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy (%)")
    plt.xticks(range(1, max_epoch + 1))
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plot_path = args.output_dir / "validation_accuracy_comparison.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()

    csv_path = args.output_dir / "benchmark_summary.csv"
    markdown_path = args.output_dir / "benchmark_summary.md"
    write_summary_csv(csv_path, summary_rows)
    write_summary_markdown(markdown_path, summary_rows)

    print(f"Saved plot: {plot_path}")
    print(f"Saved CSV summary: {csv_path}")
    print(f"Saved Markdown summary: {markdown_path}")


if __name__ == "__main__":
    main()
