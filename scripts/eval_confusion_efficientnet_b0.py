from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(img_size: int):
    resize_size = int(img_size * 256 / 224)
    eval_transform = transforms.Compose(
        [
            transforms.Resize(resize_size, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return eval_transform


def load_metadata(run_dir: Path) -> dict:
    metadata_path = run_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {run_dir}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


@torch.no_grad()
def evaluate_confusion(model, loader, device, num_classes: int):
    model.eval()
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for images, targets in loader:
        images = images.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1).cpu().numpy()
        targets = targets.cpu().numpy()
        for t, p in zip(targets, preds):
            cm[int(t), int(p)] += 1
    return cm


def plot_confusion(cm: np.ndarray, class_names: list[str], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate confusion matrix for a timm run")
    parser.add_argument("--run-dir", type=Path, required=True, help="Path to runs_timm/<run> directory")
    parser.add_argument("--data-root", type=Path, required=True, help="Prepared dataset root containing test folder")
    parser.add_argument("--device", type=str, default="cpu", help='Device: "cpu" or "cuda"')
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--model-file", type=str, default="best_model.pth", help="Checkpoint filename in run dir")
    parser.add_argument("--out", type=Path, default=None, help="Output image path for confusion matrix")
    args = parser.parse_args()

    run_dir = args.run_dir
    metadata = load_metadata(run_dir)
    class_names = metadata.get("classes")
    if not class_names:
        raise RuntimeError("No classes found in metadata.json")
    num_classes = int(metadata.get("num_classes", len(class_names)))

    model = timm.create_model(metadata.get("timm_name"), pretrained=False, num_classes=num_classes)

    device = torch.device(args.device)
    ckpt_path = run_dir / args.model_file
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    # checkpoint may be raw state_dict or a dict with key 'model_state_dict'
    state = ckpt.get("model_state_dict") if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device)

    eval_transform = build_transforms(int(metadata.get("img_size", 224)))
    test_dir = args.data_root / "test"
    dataset = datasets.ImageFolder(test_dir, transform=eval_transform)
    if dataset.classes != class_names:
        print("Warning: dataset classes differ from metadata classes. Using dataset class ordering.")
        class_names = dataset.classes
        num_classes = len(class_names)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    cm = evaluate_confusion(model, loader, device, num_classes)

    out_path = args.out or (run_dir / "confusion_matrix.png")
    plot_confusion(cm, class_names, out_path)
    print(f"Saved confusion matrix to: {out_path}")


if __name__ == "__main__":
    main()
