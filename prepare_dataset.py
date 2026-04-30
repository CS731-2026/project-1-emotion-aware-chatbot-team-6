from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_EMOTION_CLASSES = {
    "anger",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
}
EMOTION_LABEL_MAP = {
    "angry": "anger",
    "anger": "anger",
    "disgust": "disgust",
    "fear": "fear",
    "happy": "happy",
    "neutral": "neutral",
    "sad": "sad",
    "surprise": "surprise",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare emotion and eye datasets for YOLOv8 classification."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(r"G:\731\dataset"),
        help="Source dataset root.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(r"G:\731\prepared_datasets"),
        help="Output dataset root in YOLOv8 classification format.",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["all", "emotion", "eye"],
        default="all",
        help="Which task to prepare.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of creating hard links.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing prepared task directory before rebuilding it.",
    )
    return parser.parse_args()


def normalize_class_name(name: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", name.strip().lower())
    return normalized.strip("_")


def ensure_task_root(path: Path, overwrite: bool) -> None:
    if path.exists() and overwrite:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def transfer_file(source_path: Path, target_path: Path, copy_files: bool) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        return

    if copy_files:
        shutil.copy2(source_path, target_path)
        return

    try:
        os.link(source_path, target_path)
    except OSError:
        shutil.copy2(source_path, target_path)


def iter_images(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def prepare_emotion_dataset(
    source_root: Path, output_root: Path, copy_files: bool, overwrite: bool
) -> dict:
    task_source = source_root / "emotion"
    task_output = output_root / "emotion"
    ensure_task_root(task_output, overwrite)

    split_map = {"train": "train", "valid": "val", "test": "test"}
    summary: dict[str, dict[str, int]] = {}

    for source_split, output_split in split_map.items():
        split_source = task_source / source_split
        if not split_source.exists():
            continue

        split_summary: dict[str, int] = {}
        for class_dir in sorted(path for path in split_source.iterdir() if path.is_dir()):
            class_name = EMOTION_LABEL_MAP.get(
                normalize_class_name(class_dir.name),
                normalize_class_name(class_dir.name),
            )
            if class_name not in ALLOWED_EMOTION_CLASSES:
                continue
            count = 0
            for index, image_path in enumerate(iter_images(class_dir)):
                target_path = (
                    task_output
                    / output_split
                    / class_name
                    / f"{index:06d}{image_path.suffix.lower()}"
                )
                transfer_file(image_path, target_path, copy_files)
                count += 1
            split_summary[class_name] = count
        summary[output_split] = split_summary

    affectnet_summary = append_affectnet_hq(
        source_root=source_root,
        task_output=task_output,
        copy_files=copy_files,
    )
    if affectnet_summary:
        summary["train_affectnet_hq"] = dict(sorted(affectnet_summary.items()))

    return summary


def append_affectnet_hq(
    source_root: Path, task_output: Path, copy_files: bool
) -> dict[str, int]:
    affectnet_root = source_root / "Affectnet-HQ"
    csv_path = affectnet_root / "labels.csv"
    if not csv_path.exists():
        return {}

    train_root = task_output / "train"
    counters: defaultdict[str, int] = defaultdict(int)
    for class_dir in sorted(path for path in train_root.iterdir() if path.is_dir()):
        counters[class_dir.name] = len(iter_images(class_dir))

    added: defaultdict[str, int] = defaultdict(int)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label = EMOTION_LABEL_MAP.get(normalize_class_name(row["label"]))
            if label is None:
                continue

            image_path = affectnet_root / Path(*row["pth"].split("/"))
            if not image_path.exists():
                continue

            index = counters[label]
            counters[label] += 1
            added[label] += 1
            target_path = (
                train_root
                / label
                / f"affectnet_{index:06d}{image_path.suffix.lower()}"
            )
            transfer_file(image_path, target_path, copy_files)

    return added


def resolve_eye_label(row: dict[str, str], label_columns: list[str]) -> str:
    positive_columns = [column for column in label_columns if row[column].strip() == "1"]
    if positive_columns:
        return normalize_class_name(positive_columns[0])

    best_column = max(label_columns, key=lambda column: float(row[column] or 0.0))
    return normalize_class_name(best_column)


def prepare_eye_dataset(
    source_root: Path, output_root: Path, copy_files: bool, overwrite: bool
) -> dict:
    task_source = source_root / "eye"
    task_output = output_root / "eye"
    ensure_task_root(task_output, overwrite)

    split_map = {"train": "train", "valid": "val", "test": "test"}
    summary: dict[str, dict[str, int]] = {}

    for source_split, output_split in split_map.items():
        split_source = task_source / source_split
        classes_csv = split_source / "_classes.csv"
        if not split_source.exists() or not classes_csv.exists():
            continue

        counters: defaultdict[str, int] = defaultdict(int)
        with classes_csv.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            label_columns = [column for column in reader.fieldnames or [] if column != "filename"]
            for row in reader:
                label = resolve_eye_label(row, label_columns)
                image_path = split_source / row["filename"]
                if not image_path.exists():
                    continue

                index = counters[label]
                counters[label] += 1
                target_path = (
                    task_output
                    / output_split
                    / label
                    / f"{index:06d}{image_path.suffix.lower()}"
                )
                transfer_file(image_path, target_path, copy_files)

        summary[output_split] = dict(sorted(counters.items()))

    return summary


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    summary = {
        "source_root": str(args.source_root),
        "output_root": str(args.output_root),
        "copy_files": args.copy,
        "tasks": {},
    }

    if args.task in {"all", "emotion"}:
        summary["tasks"]["emotion"] = prepare_emotion_dataset(
            args.source_root,
            args.output_root,
            args.copy,
            args.overwrite,
        )

    if args.task in {"all", "eye"}:
        summary["tasks"]["eye"] = prepare_eye_dataset(
            args.source_root,
            args.output_root,
            args.copy,
            args.overwrite,
        )

    summary_path = args.output_root / "dataset_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Prepared datasets:")
    for task_name, task_summary in summary["tasks"].items():
        print(f"  {task_name}:")
        for split_name, split_counts in task_summary.items():
            counts = ", ".join(
                f"{class_name}={count}" for class_name, count in split_counts.items()
            )
            print(f"    {split_name}: {counts}")
    print(f"Summary file: {summary_path}")


if __name__ == "__main__":
    main()
