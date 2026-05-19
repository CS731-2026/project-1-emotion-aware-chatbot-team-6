from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

from drivesense.backend.vision import apply_emotion_postprocess


PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a timm emotion checkpoint on the prepared test set and "
            "generate a confusion-matrix image."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=PROJECT_ROOT / "runs_timm" / "efficientnet_b0",
        help="Path to the timm run directory that contains metadata.json and the checkpoint.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "prepared_datasets" / "emotion",
        help="Prepared emotion dataset root containing train/val/test folders.",
    )
    parser.add_argument(
        "--model-file",
        type=str,
        default="best_model.pth",
        help="Checkpoint filename inside --run-dir.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help='Evaluation device, for example "cuda" or "cpu".',
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path. Defaults to runs_timm/<run>/confusion_matrix_normalized.png.",
    )
    parser.add_argument(
        "--counts-out",
        type=Path,
        default=None,
        help="Optional CSV output path for raw confusion counts.",
    )
    parser.add_argument(
        "--show-counts",
        action="store_true",
        help="Plot raw counts instead of row-normalized percentages.",
    )
    parser.add_argument(
        "--apply-runtime-postprocess",
        action="store_true",
        help=(
            "Apply the same runtime post-processing used by the GUI, for example "
            "downgrading low-confidence sad/anger predictions to neutral."
        ),
    )
    return parser.parse_args()


def build_eval_transform(img_size: int) -> transforms.Compose:
    resize_size = int(img_size * 256 / 224)
    return transforms.Compose(
        [
            transforms.Resize(resize_size, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def load_metadata(run_dir: Path) -> dict:
    metadata_path = run_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found: {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def load_model(run_dir: Path, model_file: str, metadata: dict, device: torch.device) -> torch.nn.Module:
    timm_name = metadata.get("timm_name")
    if not timm_name:
        raise RuntimeError("metadata.json does not contain 'timm_name'.")

    class_names = metadata.get("classes")
    num_classes = int(metadata.get("num_classes", len(class_names or [])))
    if num_classes <= 0:
        raise RuntimeError("metadata.json does not contain a valid class count.")

    model = timm.create_model(timm_name, pretrained=False, num_classes=num_classes)
    checkpoint_path = run_dir / model_file
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def evaluate_confusion_matrix(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: list[str],
    apply_runtime_postprocess: bool,
) -> np.ndarray:
    class_to_idx = {name: index for index, name in enumerate(class_names)}
    confusion = np.zeros((len(class_names), len(class_names)), dtype=np.int64)

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probabilities = torch.softmax(logits, dim=1)
        confidences, predicted_indices = probabilities.max(dim=1)

        predicted_indices_np = predicted_indices.cpu().numpy()
        confidences_np = confidences.cpu().numpy()
        targets_np = targets.cpu().numpy()

        for true_index, predicted_index, confidence in zip(
            targets_np,
            predicted_indices_np,
            confidences_np,
        ):
            final_predicted_index = int(predicted_index)
            if apply_runtime_postprocess:
                predicted_label, _ = apply_emotion_postprocess(
                    class_names[final_predicted_index],
                    float(confidence),
                )
                final_predicted_index = class_to_idx[predicted_label]
            confusion[int(true_index), final_predicted_index] += 1

    return confusion


def normalize_rows(confusion: np.ndarray) -> np.ndarray:
    row_sums = confusion.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = np.divide(
            confusion,
            row_sums,
            out=np.zeros_like(confusion, dtype=np.float64),
            where=row_sums != 0,
        )
    return normalized


def plot_confusion_matrix(
    confusion: np.ndarray,
    class_names: list[str],
    output_path: Path,
    normalized: bool,
) -> None:
    matrix = normalize_rows(confusion) if normalized else confusion
    fig, ax = plt.subplots(figsize=(10, 7.5))
    image = ax.imshow(matrix, interpolation="nearest", cmap=plt.cm.Blues, vmin=0)
    ax.figure.colorbar(image, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    threshold = float(matrix.max()) / 2.0 if matrix.size and matrix.max() > 0 else 0.0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            if normalized:
                text = f"{value:.1%}"
            else:
                text = str(int(value))
            ax.text(
                col,
                row,
                text,
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=11,
            )

    title_suffix = "Row-Normalized" if normalized else "Raw Counts"
    ax.set_title(f"Emotion Confusion Matrix ({title_suffix})")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_counts_csv(confusion: np.ndarray, class_names: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_label"] + [f"pred_{name}" for name in class_names])
        for label, row in zip(class_names, confusion):
            writer.writerow([label] + [int(value) for value in row])


def validate_dataset_classes(dataset_classes: list[str], metadata_classes: list[str]) -> None:
    if dataset_classes != metadata_classes:
        raise RuntimeError(
            "Dataset class order does not match metadata.json.\n"
            f"Dataset classes: {dataset_classes}\n"
            f"Metadata classes: {metadata_classes}\n"
            "Fix the prepared dataset or use a checkpoint with matching classes."
        )


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    data_root = args.data_root.resolve()
    test_dir = data_root / "test"
    if not test_dir.exists():
        raise FileNotFoundError(f"Test folder not found: {test_dir}")

    metadata = load_metadata(run_dir)
    class_names = list(metadata.get("classes") or [])
    if not class_names:
        raise RuntimeError("metadata.json does not contain class names.")

    device = torch.device(args.device)
    model = load_model(run_dir, args.model_file, metadata, device)
    transform = build_eval_transform(int(metadata.get("img_size", 224)))
    dataset = datasets.ImageFolder(test_dir, transform=transform)
    validate_dataset_classes(dataset.classes, class_names)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    print(f"Run directory: {run_dir}")
    print(f"Checkpoint: {run_dir / args.model_file}")
    print(f"Test data: {test_dir}")
    print(f"Classes: {class_names}")
    print(f"Test images: {len(dataset)}")
    print(f"Device: {device}")
    print(f"Runtime postprocess: {args.apply_runtime_postprocess}")

    confusion = evaluate_confusion_matrix(
        model=model,
        loader=loader,
        device=device,
        class_names=class_names,
        apply_runtime_postprocess=args.apply_runtime_postprocess,
    )

    accuracy = np.trace(confusion) / max(confusion.sum(), 1)
    print(f"Overall accuracy: {accuracy:.2%}")

    normalized = not args.show_counts
    default_name = "confusion_matrix_counts.png" if args.show_counts else "confusion_matrix_normalized.png"
    output_path = args.out or (run_dir / default_name)
    plot_confusion_matrix(confusion, class_names, output_path, normalized=normalized)
    print(f"Saved confusion matrix image: {output_path}")

    counts_output = args.counts_out or (run_dir / "confusion_matrix_counts.csv")
    write_counts_csv(confusion, class_names, counts_output)
    print(f"Saved raw counts CSV: {counts_output}")


if __name__ == "__main__":
    main()
