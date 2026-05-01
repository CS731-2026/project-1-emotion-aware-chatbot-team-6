from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import timm
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from ultralytics import YOLO
import torch.nn as nn
from typing_extensions import TypedDict


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "runs_timm"
DEFAULT_WEIGHTS_ROOT = PROJECT_ROOT / "weights"
FACE_MODEL_URL = (
    "https://github.com/lindevs/yolov8-face/releases/latest/download/"
    "yolov8n-face-lindevs.pt"
)
EMOTION_CLASSES = {"anger", "disgust", "fear", "happy", "neutral", "sad", "surprise"}
EYE_CLASSES = {"closed_eye", "open_eye"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run face detection with timm-based emotion and eye-state recognition from a webcam."
    )
    parser.add_argument(
        "--emotion-model",
        type=Path,
        default=DEFAULT_RUNS_ROOT / "efficientnet_b0" / "best_model.pth",
        help="Path to a timm emotion checkpoint file or run directory.",
    )
    parser.add_argument(
        "--eye-model",
        type=Path,
        default=None,
        help="Path to a timm eye-state checkpoint file or run directory.",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=DEFAULT_RUNS_ROOT,
        help="Directory containing timm training runs.",
    )
    parser.add_argument(
        "--face-model",
        type=Path,
        default=DEFAULT_WEIGHTS_ROOT / "yolov8n-face-lindevs.pt",
        help="Path to the YOLO face detection model.",
    )
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam index.")
    parser.add_argument(
        "--capture-width",
        type=int,
        default=1280,
        help="Requested webcam capture width.",
    )
    parser.add_argument(
        "--capture-height",
        type=int,
        default=720,
        help="Requested webcam capture height.",
    )
    parser.add_argument(
        "--window-width",
        type=int,
        default=1280,
        help="Display window width.",
    )
    parser.add_argument(
        "--window-height",
        type=int,
        default=720,
        help="Display window height.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help='Inference device, for example "0", "cuda:0", or "cpu".',
    )
    parser.add_argument(
        "--face-imgsz",
        type=int,
        default=640,
        help="YOLO face detector input size.",
    )
    parser.add_argument(
        "--face-confidence",
        type=float,
        default=0.35,
        help="Minimum face detection confidence.",
    )
    parser.add_argument(
        "--classification-confidence",
        type=float,
        default=0.35,
        help="Minimum confidence for displayed classification labels.",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.15,
        help="Extra padding around the detected face box.",
    )
    parser.add_argument(
        "--max-faces",
        type=int,
        default=5,
        help="Maximum number of detected faces to process.",
    )
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=0,
        help="Skip N frames between model predictions to improve FPS.",
    )
    parser.add_argument(
        "--focus-seconds",
        type=float,
        default=2.0,
        help="Closed-eye duration before the focus warning is shown.",
    )
    parser.add_argument(
        "--driver-side",
        type=str,
        choices=["left", "center", "right", "largest"],
        default="right",
        help="Heuristic used to choose the driver's face when multiple faces are visible.",
    )
    parser.add_argument(
        "--print-emotion-top3",
        action="store_true",
        help="Print the driver's emotion top-3 probabilities for each frame.",
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


def normalize_checkpoint_path(model_path: Path | None) -> Path | None:
    if model_path is None:
        return None
    if model_path.is_dir():
        return model_path / "best_model.pth"
    return model_path


def infer_task_from_metadata(metadata: dict) -> str | None:
    task = metadata.get("task")
    if task in {"emotion", "eye"}:
        return task

    classes = set(metadata.get("classes", []))
    if classes == EMOTION_CLASSES:
        return "emotion"
    if classes == EYE_CLASSES:
        return "eye"
    return None


def resolve_latest_timm_model(runs_root: Path, task: str) -> Path:
    candidates: list[Path] = []
    for metadata_path in runs_root.glob("*/metadata.json"):
        checkpoint_path = metadata_path.parent / "best_model.pth"
        if not checkpoint_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if infer_task_from_metadata(metadata) == task:
            candidates.append(checkpoint_path)

    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(
            f"No {task} model matching runs under {runs_root}."
        )
    return candidates[0]


def resolve_classifier_paths(
    emotion_path: Path | None, eye_path: Path | None, runs_root: Path
) -> tuple[Path, Path]:
    resolved_emotion = normalize_checkpoint_path(emotion_path) or resolve_latest_timm_model(
        runs_root, "emotion"
    )
    resolved_eye = normalize_checkpoint_path(eye_path) or resolve_latest_timm_model(
        runs_root, "eye"
    )

    if not resolved_emotion.exists():
        raise FileNotFoundError(f"Emotion model not found: {resolved_emotion}")
    if not resolved_eye.exists():
        raise FileNotFoundError(f"Eye model not found: {resolved_eye}")
    return resolved_emotion, resolved_eye


def ensure_face_model(face_model_path: Path) -> Path:
    if face_model_path.exists():
        return face_model_path

    face_model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading face detector to {face_model_path} ...")
    urllib.request.urlretrieve(FACE_MODEL_URL, face_model_path)
    return face_model_path


def resolve_devices(device_arg: str) -> tuple[str, torch.device]:
    normalized = device_arg.strip().lower()
    if normalized.isdigit():
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device index was requested but CUDA is not available.")
        return normalized, torch.device(f"cuda:{normalized}")
    if normalized.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but CUDA is not available.")
        return normalized, torch.device(normalized)
    if normalized == "cpu":
        return "cpu", torch.device("cpu")
    raise ValueError(
        'Unsupported device. Use values like "0", "cuda:0", or "cpu".'
    )

from torchvision.transforms import InterpolationMode
from ultralytics import YOLO
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parent
from torchvision.transforms import InterpolationMode
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent
def load_timm_classifier(
    checkpoint_path: Path, device: torch.device
) -> ClassifierDict:
    metadata_path = checkpoint_path.parent / "metadata.json"
    metadata = None
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if metadata is None:
        metadata = checkpoint.get("metadata")
    if metadata is None:
        raise ValueError(f"Missing metadata for checkpoint: {checkpoint_path}")

    class_names = metadata.get("classes")
    timm_name = metadata.get("timm_name")
    img_size = int(metadata.get("img_size", 224))
    if not class_names or not timm_name:
        raise ValueError(f"Invalid metadata in {metadata_path}")

    model = timm.create_model(
        timm_name,
        pretrained=False,
        num_classes=len(class_names),
    )
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return {
        "checkpoint_path": checkpoint_path,
        "class_names": class_names,
        "metadata": metadata,
        "model": model,
        "transform": build_eval_transform(img_size),
    }


TransformType = transforms.Compose


class ClassifierDict(TypedDict):
    checkpoint_path: Path
    class_names: list[str]
    metadata: dict
    model: nn.Module
    transform: TransformType


def expand_box(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    frame_width: int,
    frame_height: int,
    padding: float,
) -> tuple[int, int, int, int]:
    width = x2 - x1
    height = y2 - y1
    pad_w = int(width * padding)
    pad_h = int(height * padding)
    return (
        max(0, x1 - pad_w),
        max(0, y1 - pad_h),
        min(frame_width, x2 + pad_w),
        min(frame_height, y2 + pad_h),
    )


def draw_tag(
    frame,
    text: str,
    anchor: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.65
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = anchor
    y_top = max(0, y - text_h - baseline - 8)
    cv2.rectangle(
        frame,
        (x, y_top),
        (x + text_w + 10, y_top + text_h + baseline + 8),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        frame,
        text,
        (x + 5, y_top + text_h + 2),
        font,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def normalize_label(label: str) -> str:
    return label.replace("_", " ")


def choose_driver_face(
    faces: list[dict],
    frame_width: int,
    driver_side: str,
    previous_center_x: float | None = None,
) -> dict | None:
    if not faces:
        return None
    if driver_side == "largest":
        return max(faces, key=lambda item: item["area"])

    preferred_x = {
        "left": 0.25,
        "center": 0.50,
        "right": 0.75,
    }.get(driver_side, 0.75)
    max_area = max(face["area"] for face in faces) or 1

    def score(face: dict) -> float:
        x1, _, x2, _ = face["bbox"]
        center_x = ((x1 + x2) / 2.0) / max(frame_width, 1)
        area_score = face["area"] / max_area
        position_score = 1.0 - min(abs(center_x - preferred_x), 1.0)
        tracking_score = 0.0
        if previous_center_x is not None:
            tracking_score = 1.0 - min(abs(center_x - previous_center_x), 1.0)
        return 0.50 * area_score + 0.35 * position_score + 0.15 * tracking_score

    return max(faces, key=score)


@torch.inference_mode()
def classify_crops(
    classifier: ClassifierDict, crops_rgb: list, device: torch.device
) -> list[tuple[str, float]]:
    predictions = classify_crops_with_topk(
        classifier=classifier,
        crops_rgb=crops_rgb,
        device=device,
        top_k=1,
    )
    return [(cast(str, item["label"]), cast(float, item["confidence"])) for item in predictions]


@torch.inference_mode()
def classify_crops_with_topk(
    classifier: ClassifierDict,
    crops_rgb: list[np.ndarray],
    device: torch.device,
    top_k: int = 3,
) -> list[dict[str, object]]:
    if not crops_rgb:
        return []

    transform = classifier["transform"]
    model = classifier["model"]
    class_names = classifier["class_names"]
    top_k = max(1, min(top_k, len(class_names)))

    tensors: list[torch.Tensor] = [
        transform(Image.fromarray(crop_rgb)) for crop_rgb in crops_rgb  # type: ignore[arg-type]
    ]
    batch = torch.stack(tensors).to(device)
    logits = model(batch)
    probabilities = torch.softmax(logits, dim=1)
    top_confidences, top_indices = probabilities.topk(k=top_k, dim=1)

    results: list[dict[str, object]] = []
    for confidence_row, index_row in zip(top_confidences, top_indices):
        topk: list[tuple[str, float]] = []
        for confidence, index in zip(confidence_row, index_row):
            label = str(class_names[int(index)])
            topk.append((label, float(confidence)))
        results.append(
            {
                "label": topk[0][0],
                "confidence": topk[0][1],
                "topk": topk,
            }
        )
    return results


def format_topk_prediction(topk: list[tuple[str, float]]) -> str:
    return ", ".join(f"{normalize_label(label)}={confidence:.3f}" for label, confidence in topk)


def estimate_eye_boxes(face_shape: tuple[int, int, int]) -> list[tuple[int, int, int, int]]:
    face_h, face_w = face_shape[:2]
    eye_y1 = int(face_h * 0.34)
    eye_y2 = int(face_h * 0.44)
    return [
        (
            int(face_w * 0.16),
            eye_y1,
            int(face_w * 0.40),
            eye_y2,
        ),
        (
            int(face_w * 0.60),
            eye_y1,
            int(face_w * 0.84),
            eye_y2,
        ),
    ]


def main() -> None:
    args = parse_args()
    face_device, torch_device = resolve_devices(args.device)
    emotion_model_path, eye_model_path = resolve_classifier_paths(
        args.emotion_model, args.eye_model, args.runs_root
    )
    face_model_path = ensure_face_model(args.face_model)

    face_detector = YOLO(str(face_model_path))
    emotion_classifier = load_timm_classifier(emotion_model_path, torch_device)
    eye_classifier = load_timm_classifier(eye_model_path, torch_device)

    cap = cv2.VideoCapture(args.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.capture_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.capture_height)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open webcam index {args.camera_index}.")

    window_name = "Focus Monitor"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, args.window_width, args.window_height)

    frame_index = 0
    fps = 0.0
    previous_time = time.perf_counter()
    closed_eye_start: float | None = None
    previous_driver_center_x: float | None = None
    cached_faces: list[dict] = []

    print(f"Face detector: {face_model_path}")
    print(f"Emotion model: {emotion_model_path}")
    print(f"Eye model: {eye_model_path}")
    print("Press q to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read a frame from the webcam.")
                break

            frame_h, frame_w = frame.shape[:2]
            should_predict = (
                args.skip_frames == 0
                or frame_index % (args.skip_frames + 1) == 0
                or not cached_faces
            )

            if should_predict:
                cached_faces = []
                face_result = face_detector.predict(
                    source=frame,
                    imgsz=args.face_imgsz,
                    device=face_device,
                    conf=args.face_confidence,
                    verbose=False,
                    max_det=args.max_faces,
                )[0]

                boxes = []
                crops_rgb = []
                if face_result.boxes is not None:
                    raw_boxes = face_result.boxes.xyxy.cpu().numpy().astype(int)  # type: ignore[union-attr]
                    raw_confidences = face_result.boxes.conf.cpu().numpy().tolist()  # type: ignore[union-attr]
                    detections = sorted(
                        zip(raw_boxes, raw_confidences),
                        key=lambda item: (item[0][2] - item[0][0]) * (item[0][3] - item[0][1]),
                        reverse=True,
                    )[: args.max_faces]

                    for raw_box, detection_conf in detections:
                        x1, y1, x2, y2 = expand_box(
                            int(raw_box[0]),
                            int(raw_box[1]),
                            int(raw_box[2]),
                            int(raw_box[3]),
                            frame_w,
                            frame_h,
                            args.padding,
                        )
                        face_crop = frame[y1:y2, x1:x2]
                        if face_crop.size == 0:
                            continue
                        boxes.append((x1, y1, x2, y2, float(detection_conf)))
                        crops_rgb.append(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))

                if crops_rgb:
                    emotion_predictions = classify_crops_with_topk(
                        emotion_classifier, crops_rgb, torch_device
                    )
                    eye_predictions = classify_crops(
                        eye_classifier, crops_rgb, torch_device
                    )

                    for box, face_rgb, emotion_pred, eye_pred in zip(
                        boxes,
                        crops_rgb,
                        emotion_predictions,
                        eye_predictions,
                    ):
                        x1, y1, x2, y2, detection_conf = box
                        eye_boxes = estimate_eye_boxes(face_rgb.shape)
                        cached_faces.append(
                            {
                                "bbox": (x1, y1, x2, y2),
                                "area": (x2 - x1) * (y2 - y1),
                                "detection_confidence": detection_conf,
                                "emotion_label": emotion_pred["label"],
                                "emotion_confidence": emotion_pred["confidence"],
                                "emotion_topk": emotion_pred["topk"],
                                "eye_label": eye_pred[0],
                                "eye_confidence": eye_pred[1],
                                "eye_boxes": eye_boxes,
                            }
                        )

            primary_face = choose_driver_face(
                cached_faces,
                frame_w,
                args.driver_side,
                previous_driver_center_x,
            )
            if primary_face is not None:
                x1, _, x2, _ = primary_face["bbox"]
                previous_driver_center_x = ((x1 + x2) / 2.0) / max(frame_w, 1)
            current_time = time.perf_counter()
            if (
                primary_face
                and primary_face["eye_label"] == "closed_eye"
                and primary_face["eye_confidence"] >= args.classification_confidence
            ):
                if closed_eye_start is None:
                    closed_eye_start = current_time
            else:
                closed_eye_start = None

            alert_visible = (
                closed_eye_start is not None
                and current_time - closed_eye_start >= args.focus_seconds
            )

            for face_info in cached_faces:
                x1, y1, x2, y2 = face_info["bbox"]
                emotion_conf = face_info["emotion_confidence"]
                eye_conf = face_info["eye_confidence"]
                is_driver_face = face_info is primary_face

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    3 if is_driver_face else 2,
                )
                if emotion_conf >= args.classification_confidence:
                    draw_tag(
                        frame,
                        f"{normalize_label(face_info['emotion_label'])} {emotion_conf:.2f}",
                        (x1, y1),
                        (0, 255, 0),
                    )
                else:
                    draw_tag(frame, "emotion uncertain", (x1, y1), (0, 255, 0))

                for eye_box in face_info["eye_boxes"]:
                    ex1, ey1, ex2, ey2 = eye_box
                    cv2.rectangle(
                        frame,
                        (x1 + ex1, y1 + ey1),
                        (x1 + ex2, y1 + ey2),
                        (255, 0, 0),
                        2,
                    )

                draw_tag(
                    frame,
                    f"{normalize_label(face_info['eye_label'])} {eye_conf:.2f}",
                    (x1, min(frame_h - 5, y2 + 28)),
                    (255, 0, 0),
                )
                if is_driver_face:
                    draw_tag(frame, "driver", (x1, y2 + 56), (255, 255, 255))

            delta = current_time - previous_time
            if delta > 0:
                fps = 1.0 / delta
            previous_time = current_time
            frame_index += 1

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

            if alert_visible:
                alert_text = "Please stay focused"
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1.0
                thickness = 3
                (text_w, text_h), baseline = cv2.getTextSize(
                    alert_text, font, scale, thickness
                )
                margin = 20
                x1 = frame_w - text_w - 40 - margin
                y1 = 20
                x2 = frame_w - margin
                y2 = y1 + text_h + baseline + 20
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)
                cv2.putText(
                    frame,
                    alert_text,
                    (x1 + 20, y2 - 12),
                    font,
                    scale,
                    (0, 0, 255),
                    thickness,
                    cv2.LINE_AA,
                )

            if args.print_emotion_top3:
                if primary_face is None:
                    print("Driver emotion top-3: no face", flush=True)
                else:
                    topk = primary_face.get("emotion_topk", [])
                    print(
                        f"Driver emotion top-3: {format_topk_prediction(topk)}",
                        flush=True,
                    )

            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
