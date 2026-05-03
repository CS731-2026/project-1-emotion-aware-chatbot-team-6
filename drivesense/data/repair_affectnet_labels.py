from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair Affectnet-HQ labels.csv by matching it to the files currently present on disk."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "Affectnet-HQ",
        help="Affectnet-HQ dataset root.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite labels.csv after creating a timestamped backup.",
    )
    return parser.parse_args()


def normalize_relpath(path: str) -> str:
    return path.replace("\\", "/").strip()


def iter_dataset_files(dataset_root: Path) -> list[str]:
    return sorted(
        normalize_relpath(str(path.relative_to(dataset_root)))
        for class_dir in sorted(path for path in dataset_root.iterdir() if path.is_dir())
        for path in sorted(class_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def rebuild_rows(dataset_root: Path, rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], dict]:
    all_files = iter_dataset_files(dataset_root)
    all_files_set = set(all_files)
    existing_by_path = {
        normalize_relpath(row.get("pth", "")): row
        for row in rows
        if normalize_relpath(row.get("pth", "")) in all_files_set
    }

    repaired_rows: list[dict[str, str]] = []
    added_paths: list[str] = []
    dropped_paths = sorted(
        normalize_relpath(row.get("pth", ""))
        for row in rows
        if normalize_relpath(row.get("pth", "")) not in all_files_set
    )

    for index, relpath in enumerate(all_files):
        if relpath in existing_by_path:
            row = dict(existing_by_path[relpath])
            row["pth"] = relpath
        else:
            label = Path(relpath).parent.name
            row = {
                "": str(index),
                "pth": relpath,
                "label": label,
                "relFCs": "",
            }
            added_paths.append(relpath)

        row[""] = str(index)
        repaired_rows.append(row)

    label_counts = Counter(row["label"] for row in repaired_rows)
    report = {
        "dataset_root": str(dataset_root),
        "original_row_count": len(rows),
        "repaired_row_count": len(repaired_rows),
        "added_count": len(added_paths),
        "dropped_count": len(dropped_paths),
        "added_examples": added_paths[:20],
        "dropped_examples": dropped_paths[:20],
        "label_counts": dict(label_counts),
    }
    return repaired_rows, report


def main() -> None:
    args = parse_args()
    csv_path = args.dataset_root / "labels.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"labels.csv not found under {args.dataset_root}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        original_rows = list(reader)
        fieldnames = reader.fieldnames or ["", "pth", "label", "relFCs"]

    repaired_rows, report = rebuild_rows(args.dataset_root, original_rows)
    report_path = args.dataset_root / "labels_repair_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.overwrite:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = args.dataset_root / f"labels_backup_before_repair_{timestamp}.csv"
        backup_path.write_text(csv_path.read_text(encoding="utf-8-sig"), encoding="utf-8-sig")
        with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(repaired_rows)
        print(f"Backup saved to: {backup_path}")
        print(f"Repaired labels written to: {csv_path}")
    else:
        preview_path = args.dataset_root / "labels_repaired_preview.csv"
        with preview_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(repaired_rows)
        print(f"Preview repaired labels written to: {preview_path}")

    print(f"Repair report: {report_path}")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
