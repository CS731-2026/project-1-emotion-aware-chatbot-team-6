from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from drivesense.backend.chatbot import DriverAssistantChatbot
from drivesense.benchmarks.llm_benchmark import (
    SCENARIOS,
    build_manual_scores,
    write_csv,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEMPERATURES = [0.5, 1.0, 1.5]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a temperature sweep for the selected best LLM."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Best OpenRouter model id selected from the main benchmark.",
    )
    parser.add_argument(
        "--temperatures",
        nargs="+",
        type=float,
        default=DEFAULT_TEMPERATURES,
        help="Temperature values to compare.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "benchmark_results" / "temperature_sweep",
        help="Directory for sweep outputs.",
    )
    return parser.parse_args()


def save_plot(summary_rows: list[dict], output_path: Path) -> None:
    temperatures = [str(row["temperature"]) for row in summary_rows]
    latencies = [float(row["avg_latency_ms"]) for row in summary_rows]
    plt.figure(figsize=(8, 5))
    plt.plot(temperatures, latencies, marker="o", linewidth=2, color="#7c3aed")
    plt.xlabel("Temperature")
    plt.ylabel("Average Latency (ms)")
    plt.title("Temperature Sweep Latency")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    chatbot = DriverAssistantChatbot(app_title="Driver Assistant Temperature Sweep")

    response_rows: list[dict] = []
    grouped_latencies: dict[float, list[float]] = defaultdict(list)

    for temperature in args.temperatures:
        for scenario in SCENARIOS:
            response = chatbot.generate_reply(
                emotion=scenario["emotion"],
                user_message=scenario["user_message"],
                model=args.model,
                temperature=temperature,
                conversation_history=[],
                auto_trigger=scenario["auto_trigger"],
            )
            row = {
                "scenario_id": scenario["scenario_id"],
                "emotion": scenario["emotion"],
                "user_message": scenario["user_message"] or "",
                "auto_trigger": scenario["auto_trigger"],
                "model": args.model,
                "temperature": temperature,
                "reply": response.text,
                "latency_ms": round(response.latency_ms, 3),
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "total_tokens": response.total_tokens,
            }
            response_rows.append(row)
            grouped_latencies[temperature].append(response.latency_ms)
            print(json.dumps(row, ensure_ascii=False))

    summary_rows = []
    for temperature in args.temperatures:
        summary_rows.append(
            {
                "temperature": temperature,
                "avg_latency_ms": round(
                    sum(grouped_latencies[temperature])
                    / len(grouped_latencies[temperature]),
                    3,
                ),
            }
        )

    responses_json_path = args.output_dir / "responses.json"
    responses_json_path.write_text(
        json.dumps(response_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_csv(args.output_dir / "responses.csv", response_rows)
    write_csv(args.output_dir / "manual_scores_template.csv", build_manual_scores(response_rows))
    write_csv(args.output_dir / "temperature_summary.csv", summary_rows)
    save_plot(summary_rows, args.output_dir / "temperature_latency.png")

    print(f"Saved responses: {responses_json_path}")
    print(f"Saved summary: {args.output_dir / 'temperature_summary.csv'}")
    print(f"Saved manual score template: {args.output_dir / 'manual_scores_template.csv'}")


if __name__ == "__main__":
    main()
