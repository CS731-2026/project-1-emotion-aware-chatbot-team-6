from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a YOLOv8 classification model for facial expression recognition."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(r"G:\731\emotion_cls_data"),
        help="Prepared classification dataset root with train/val folders.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n-cls.pt",
        help="Base YOLOv8 classification checkpoint.",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs.")
    parser.add_argument("--imgsz", type=int, default=224, help="Training image size.")
    parser.add_argument("--batch", type=int, default=64, help="Batch size.")
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help='Training device, for example "0" or "cpu".',
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Data loader workers. Set 0 if Windows data loading is unstable.",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=Path(r"G:\731\runs"),
        help="Ultralytics runs directory.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="emotion_yolov8n_cls",
        help="Training run name.",
    )
    parser.add_argument("--patience", type=int, default=10, help="Early stop patience.")
    return parser.parse_args()


def validate_dataset_root(data_root: Path) -> None:
    required_dirs = [data_root / "train", data_root / "val"]
    missing = [str(path) for path in required_dirs if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Prepared dataset is missing required folders: " + ", ".join(missing)
        )


def main() -> None:
    args = parse_args()
    validate_dataset_root(args.data_root)

    model = YOLO(args.model)
    results = model.train(
        data=str(args.data_root),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(args.project),
        name=args.name,
        patience=args.patience,
        pretrained=True,
        verbose=True,
    )

    save_dir = Path(results.save_dir)
    print(f"Training finished. Best checkpoint: {save_dir / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
