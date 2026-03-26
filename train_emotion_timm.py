from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import time
from contextlib import nullcontext
from pathlib import Path

import timm
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode


EXPECTED_CLASSES = {
    "anger",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
}

MODEL_SPECS = {
    "resnet50": {
        "timm_name": "resnet50",
        "display_name": "ResNet50",
    },
    "efficientnet_b0": {
        "timm_name": "efficientnet_b0",
        "display_name": "EfficientNet-B0",
    },
    "efficientnet_b3": {
        "timm_name": "efficientnet_b3",
        "display_name": "EfficientNet-B3",
    },
    "swin_tiny": {
        "timm_name": "swin_tiny_patch4_window7_224",
        "display_name": "Swin Transformer Tiny",
    },
    "mobilenet_v2": {
        "timm_name": "mobilenetv2_100",
        "display_name": "MobileNetV2",
    },
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a timm emotion classifier under a fixed benchmark setting."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(r"G:\731\prepared_datasets\emotion"),
        help="Prepared dataset root with train/val/test folders.",
    )
    parser.add_argument(
        "--model-key",
        type=str,
        choices=sorted(MODEL_SPECS),
        required=True,
        help="Benchmark model key.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=224,
        help="Shared input size for all models.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Shared batch size for all models.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for AdamW.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay for AdamW.",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.1,
        help="Label smoothing for cross-entropy loss.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help='Training device, for example "cuda", "cuda:0", or "cpu".',
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Data loader workers. Use 0 on Windows if multiprocess loading is unstable.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path(r"G:\731\runs_timm"),
        help="Directory for timm benchmark runs.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional run directory name. Defaults to the model key.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the existing run directory before training.",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Disable ImageNet pretrained weights.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transforms(img_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    resize_size = int(img_size * 256 / 224)
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                img_size,
                scale=(0.8, 1.0),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(resize_size, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform


def validate_class_names(class_names: list[str]) -> None:
    discovered = set(class_names)
    if discovered != EXPECTED_CLASSES:
        raise ValueError(
            "Dataset classes do not match the expected 6-class set. "
            f"Expected {sorted(EXPECTED_CLASSES)}, got {sorted(discovered)}."
        )


def build_dataloaders(
    data_root: Path, img_size: int, batch_size: int, workers: int, device: torch.device
) -> tuple[dict[str, datasets.ImageFolder], dict[str, DataLoader]]:
    train_transform, eval_transform = build_transforms(img_size)
    datasets_map = {
        "train": datasets.ImageFolder(data_root / "train", transform=train_transform),
        "val": datasets.ImageFolder(data_root / "val", transform=eval_transform),
        "test": datasets.ImageFolder(data_root / "test", transform=eval_transform),
    }
    validate_class_names(datasets_map["train"].classes)

    common_kwargs = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": workers > 0,
    }
    dataloaders = {
        "train": DataLoader(datasets_map["train"], shuffle=True, **common_kwargs),
        "val": DataLoader(datasets_map["val"], shuffle=False, **common_kwargs),
        "test": DataLoader(datasets_map["test"], shuffle=False, **common_kwargs),
    }
    return datasets_map, dataloaders


def resolve_device(device_arg: str) -> torch.device:
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(device_arg)


def create_run_dir(runs_root: Path, run_name: str, overwrite: bool) -> Path:
    run_dir = runs_root / run_name
    if run_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Run directory already exists: {run_dir}. Use --overwrite to replace it."
            )
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> int:
    predictions = logits.argmax(dim=1)
    return int((predictions == targets).sum().item())


def autocast_context(device: torch.device, amp_enabled: bool):
    if device.type != "cuda" or not amp_enabled:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.float16)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    amp_enabled: bool,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, amp_enabled):
            logits = model(images)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        correct += accuracy_from_logits(logits, targets)
        total += batch_size

    return running_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool,
) -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with autocast_context(device, amp_enabled):
            logits = model(images)
            loss = criterion(logits, targets)

        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        correct += accuracy_from_logits(logits, targets)
        total += batch_size

    return running_loss / total, 100.0 * correct / total


@torch.no_grad()
def benchmark_inference_speed(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
) -> tuple[float, float]:
    model.eval()

    warmup_batches = 3
    for batch_index, (images, _) in enumerate(loader):
        if batch_index >= warmup_batches:
            break
        images = images.to(device, non_blocking=True)
        with autocast_context(device, amp_enabled):
            _ = model(images)
    if device.type == "cuda":
        torch.cuda.synchronize()

    total_images = 0
    start_time = time.perf_counter()
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        with autocast_context(device, amp_enabled):
            _ = model(images)
        total_images += images.size(0)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time

    milliseconds_per_image = 1000.0 * elapsed / total_images
    images_per_second = total_images / elapsed
    return milliseconds_per_image, images_per_second


def save_history(history_path: Path, history_rows: list[dict[str, float | int]]) -> None:
    with history_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_accuracy",
                "val_loss",
                "val_accuracy",
                "learning_rate",
                "epoch_seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(history_rows)


def main() -> None:
    args = parse_args()
    spec = MODEL_SPECS[args.model_key]
    run_name = args.run_name or args.model_key
    device = resolve_device(args.device)
    amp_enabled = device.type == "cuda"

    set_seed(args.seed)
    run_dir = create_run_dir(args.runs_root, run_name, args.overwrite)
    datasets_map, dataloaders = build_dataloaders(
        args.data_root,
        args.img_size,
        args.batch_size,
        args.workers,
        device,
    )

    model = timm.create_model(
        spec["timm_name"],
        pretrained=not args.no_pretrained,
        num_classes=len(datasets_map["train"].classes),
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    metadata = {
        "model_key": args.model_key,
        "display_name": spec["display_name"],
        "timm_name": spec["timm_name"],
        "pretrained": not args.no_pretrained,
        "epochs": args.epochs,
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "seed": args.seed,
        "device": str(device),
        "classes": datasets_map["train"].classes,
        "num_parameters": count_parameters(model),
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    history_rows: list[dict[str, float | int]] = []
    best_val_accuracy = -1.0
    best_epoch = 0
    history_path = run_dir / "history.csv"
    best_checkpoint_path = run_dir / "best_model.pth"
    last_checkpoint_path = run_dir / "last_model.pth"

    print(f"Run directory: {run_dir}")
    print(f"Model: {spec['display_name']} ({spec['timm_name']})")
    print(f"Classes: {datasets_map['train'].classes}")
    print(f"Train images: {len(datasets_map['train'])}")
    print(f"Val images: {len(datasets_map['val'])}")
    print(f"Test images: {len(datasets_map['test'])}")

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_loss, train_accuracy = train_one_epoch(
            model,
            dataloaders["train"],
            criterion,
            optimizer,
            scaler,
            device,
            amp_enabled,
        )
        val_loss, val_accuracy = evaluate(
            model,
            dataloaders["val"],
            criterion,
            device,
            amp_enabled,
        )
        epoch_seconds = time.perf_counter() - epoch_start
        learning_rate = optimizer.param_groups[0]["lr"]

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "train_accuracy": round(train_accuracy, 4),
                "val_loss": round(val_loss, 6),
                "val_accuracy": round(val_accuracy, 4),
                "learning_rate": learning_rate,
                "epoch_seconds": round(epoch_seconds, 3),
            }
        )
        save_history(history_path, history_rows)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_accuracy": best_val_accuracy,
            "metadata": metadata,
        }
        torch.save(checkpoint, last_checkpoint_path)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            checkpoint["best_val_accuracy"] = best_val_accuracy
            torch.save(checkpoint, best_checkpoint_path)

        scheduler.step()

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"train_acc={train_accuracy:.2f}% | "
            f"val_acc={val_accuracy:.2f}% | "
            f"val_loss={val_loss:.4f} | "
            f"time={epoch_seconds:.1f}s"
        )

    best_checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    val_loss, val_accuracy = evaluate(
        model,
        dataloaders["val"],
        criterion,
        device,
        amp_enabled,
    )
    test_loss, test_accuracy = evaluate(
        model,
        dataloaders["test"],
        criterion,
        device,
        amp_enabled,
    )
    avg_ms_per_image, images_per_second = benchmark_inference_speed(
        model,
        dataloaders["val"],
        device,
        amp_enabled,
    )

    metrics = {
        "model_key": args.model_key,
        "display_name": spec["display_name"],
        "timm_name": spec["timm_name"],
        "best_epoch": best_epoch,
        "best_val_accuracy": round(best_val_accuracy, 4),
        "final_val_accuracy": round(val_accuracy, 4),
        "final_val_loss": round(val_loss, 6),
        "test_accuracy": round(test_accuracy, 4),
        "test_loss": round(test_loss, 6),
        "avg_inference_ms_per_image": round(avg_ms_per_image, 4),
        "images_per_second": round(images_per_second, 4),
        "num_parameters": metadata["num_parameters"],
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("Benchmark metrics:")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
