from __future__ import annotations

import argparse


MODULE_CHOICES = {
    "gui": "drivesense.frontend.gui",
    "vision": "drivesense.backend.vision",
    "chatbot": "drivesense.backend.chatbot",
    "speech": "drivesense.backend.speech",
    "prepare-dataset": "drivesense.data.prepare_dataset",
    "repair-affectnet": "drivesense.data.repair_affectnet_labels",
    "train-emotion": "drivesense.training.train_emotion_timm",
    "train-eye": "drivesense.training.train_eye_timm",
    "benchmark-llm": "drivesense.benchmarks.llm_benchmark",
    "score-llm": "drivesense.benchmarks.score_llm_results",
    "summarize-timm": "drivesense.benchmarks.summarize_timm_benchmark",
    "temperature-sweep": "drivesense.benchmarks.temperature_sweep",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DriveSense package entry point."
    )
    parser.add_argument(
        "module",
        nargs="?",
        choices=sorted(MODULE_CHOICES),
        help="DriveSense subcommand to run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.module:
        print("DriveSense package entry point.")
        print("Use one of the following:")
        for command, module_path in sorted(MODULE_CHOICES.items()):
            print(f"  python -m {module_path}")
            print(f"  python -m drivesense {command}")
        return

    module_path = MODULE_CHOICES[args.module]
    print(f"Run: python -m {module_path}")


if __name__ == "__main__":
    main()
