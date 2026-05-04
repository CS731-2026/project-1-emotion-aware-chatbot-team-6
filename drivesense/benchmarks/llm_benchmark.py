from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from drivesense.backend.chatbot import DriverAssistantChatbot, SUPPORTED_LLM_MODELS


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SCENARIOS = [
    {
        "scenario_id": 1,
        "emotion": "anger",
        "user_message": "This traffic jam is infuriating! Why is everyone driving so slowly?",
        "auto_trigger": False,
    },
    {
        "scenario_id": 2,
        "emotion": "fear",
        "user_message": "I'm exhausted and the road conditions are getting worse. I'm worried about safety.",
        "auto_trigger": False,
    },
    {
        "scenario_id": 3,
        "emotion": "neutral",
        "user_message": "How much longer until we reach the destination?",
        "auto_trigger": False,
    },
    {
        "scenario_id": 4,
        "emotion": "sad",
        "user_message": "I had a terrible day at work. I just want to get home and relax.",
        "auto_trigger": False,
    },
    {
        "scenario_id": 5,
        "emotion": "anxiety",
        "user_message": "I'm running late for an important meeting. Can you help me find a faster route?",
        "auto_trigger": False,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the fixed LLM comparison benchmark for driver-support prompts."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=SUPPORTED_LLM_MODELS,
        help="OpenRouter model ids to compare.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Shared temperature used for all models in the benchmark.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "benchmark_results" / "llm_benchmark",
        help="Directory for benchmark outputs.",
    )
    parser.add_argument(
        "--pricing-json",
        type=Path,
        default=None,
        help="Optional JSON file with model pricing: {model: {input_per_million, output_per_million}}.",
    )
    return parser.parse_args()


def load_pricing(pricing_path: Path | None) -> dict[str, dict[str, float]]:
    if pricing_path is None or not pricing_path.exists():
        return {}
    return json.loads(pricing_path.read_text(encoding="utf-8"))


def estimate_cost(
    pricing: dict[str, float] | None,
    prompt_tokens: int | None,
    completion_tokens: int | None,
) -> float | None:
    if not pricing or prompt_tokens is None or completion_tokens is None:
        return None
    input_cost = pricing.get("input_per_million")
    output_cost = pricing.get("output_per_million")
    if input_cost is None or output_cost is None:
        return None
    return (prompt_tokens / 1_000_000.0) * input_cost + (
        completion_tokens / 1_000_000.0
    ) * output_cost


def write_csv(output_path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_manual_scores(rows: list[dict]) -> list[dict]:
    manual_rows = []
    for row in rows:
        manual_rows.append(
            {
                **row,
                "rater_1_score": "",
                "rater_2_score": "",
                "rater_3_score": "",
                "average_score": "",
            }
        )
    return manual_rows


def save_latency_plot(summary_rows: list[dict], output_path: Path) -> None:
    model_names = [row["model"] for row in summary_rows]
    latencies = [float(row["avg_latency_ms"]) for row in summary_rows]
    plt.figure(figsize=(8, 5))
    plt.bar(model_names, latencies, color=["#3b82f6", "#f59e0b", "#10b981"])
    plt.ylabel("Average Latency (ms)")
    plt.title("LLM Average Latency Comparison")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pricing_map = load_pricing(args.pricing_json)
    chatbot = DriverAssistantChatbot(app_title="Driver Assistant LLM Benchmark")

    response_rows: list[dict] = []
    grouped_latencies: dict[str, list[float]] = defaultdict(list)
    grouped_costs: dict[str, list[float]] = defaultdict(list)

    for model in args.models:
        for scenario in SCENARIOS:
            response = chatbot.generate_reply(
                emotion=scenario["emotion"],
                user_message=scenario["user_message"],
                model=model,
                temperature=args.temperature,
                conversation_history=[],
                auto_trigger=scenario["auto_trigger"],
            )
            estimated_cost = estimate_cost(
                pricing_map.get(model),
                response.prompt_tokens,
                response.completion_tokens,
            )
            row = {
                "scenario_id": scenario["scenario_id"],
                "emotion": scenario["emotion"],
                "user_message": scenario["user_message"] or "",
                "auto_trigger": scenario["auto_trigger"],
                "model": model,
                "temperature": args.temperature,
                "reply": response.text,
                "latency_ms": round(response.latency_ms, 3),
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "total_tokens": response.total_tokens,
                "estimated_cost_usd": (
                    round(estimated_cost, 8) if estimated_cost is not None else ""
                ),
            }
            response_rows.append(row)
            grouped_latencies[model].append(response.latency_ms)
            if estimated_cost is not None:
                grouped_costs[model].append(estimated_cost)
            print(json.dumps(row, ensure_ascii=False))

    summary_rows = []
    for model in args.models:
        costs = grouped_costs.get(model, [])
        summary_rows.append(
            {
                "model": model,
                "avg_latency_ms": round(
                    sum(grouped_latencies[model]) / len(grouped_latencies[model]), 3
                ),
                "avg_estimated_cost_usd": (
                    round(sum(costs) / len(costs), 8) if costs else ""
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
    write_csv(args.output_dir / "model_summary.csv", summary_rows)
    save_latency_plot(summary_rows, args.output_dir / "latency_comparison.png")

    print(f"Saved responses: {responses_json_path}")
    print(f"Saved summary: {args.output_dir / 'model_summary.csv'}")
    print(f"Saved manual score template: {args.output_dir / 'manual_scores_template.csv'}")


if __name__ == "__main__":
    main()
