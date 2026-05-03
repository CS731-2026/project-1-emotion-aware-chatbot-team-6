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
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from typing_extensions import TypedDict
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]
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


class ClassifierDict(TypedDict):
    model: nn.Module
    transform: transforms.Compose
    class_names: list[str]
    img_size: int
    timm_name: str


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
        action=argparse.BooleanOptionalAction,
        default=True,
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
        raise FileNotFoundError(f"No {task} model matching runs under {runs_root}.")
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
    raise ValueError('Unsupported device. Use values like "0", "cuda:0", or "cpu".')


def load_timm_classifier(checkpoint_path: Path, device: torch.device) -> ClassifierDict:
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
        "model": model,
        "transform": build_eval_transform(img_size),
        "class_names": list(class_names),
        "img_size": img_size,
        "timm_name": timm_name,
    }


def classify_crops(
    classifier: ClassifierDict, crops_bgr: list[np.ndarray], device: torch.device
) -> list[tuple[str, float]]:
    topk_predictions = classify_crops_with_topk(classifier, crops_bgr, device, topk=1)
    outputs: list[tuple[str, float]] = []
    for predictions in topk_predictions:
        if isinstance(predictions, dict):
            outputs.append((str(predictions["label"]), float(predictions["confidence"])))
        else:
            outputs.append((predictions[0][0], predictions[0][1]))
    return outputs


def classify_crops_with_topk(
    classifier: ClassifierDict,
    crops_bgr: list[np.ndarray],
    device: torch.device,
    topk: int = 3,
    top_k: int | None = None,
) -> list[list[tuple[str, float]]] | list[dict[str, object]]:
    if not crops_bgr:
        return []

    legacy_gui_mode = top_k is not None
    requested_topk = top_k if top_k is not None else topk

    pil_images = []
    for crop in crops_bgr:
        if legacy_gui_mode:
            pil_images.append(Image.fromarray(crop))
        else:
            pil_images.append(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
    batch = torch.stack(
        [classifier["transform"](image) for image in pil_images]
    ).to(device)

    with torch.inference_mode():
        logits = cast(torch.Tensor, classifier["model"](batch))
        probabilities = torch.softmax(logits, dim=1)

    k = min(requested_topk, probabilities.shape[1])
    top_probabilities, top_indices = torch.topk(probabilities, k=k, dim=1)

    outputs: list[list[tuple[str, float]]] = []
    for probs_row, indices_row in zip(top_probabilities, top_indices):
        row_predictions: list[tuple[str, float]] = []
        for score, class_index in zip(probs_row.cpu().tolist(), indices_row.cpu().tolist()):
            row_predictions.append((classifier["class_names"][class_index], float(score)))
        outputs.append(row_predictions)

    if legacy_gui_mode:
        return [
            {
                "label": predictions[0][0],
                "confidence": predictions[0][1],
                "topk": predictions,
            }
            for predictions in outputs
        ]
    return outputs


def format_topk_prediction(predictions: list[tuple[str, float]]) -> str:
    return ", ".join(f"{label}={score:.2f}" for label, score in predictions)


def expand_box(*args: object) -> tuple[int, int, int, int]:
    if len(args) == 4:
        box, frame_width, frame_height, padding_ratio = args
        x1, y1, x2, y2 = cast(tuple[int, int, int, int], box)
    elif len(args) == 7:
        x1, y1, x2, y2, frame_width, frame_height, padding_ratio = args
        x1 = int(cast(int, x1))
        y1 = int(cast(int, y1))
        x2 = int(cast(int, x2))
        y2 = int(cast(int, y2))
    else:
        raise TypeError("expand_box() expects 4 or 7 positional arguments.")

    frame_width = int(cast(int, frame_width))
    frame_height = int(cast(int, frame_height))
    padding_ratio = float(cast(float, padding_ratio))
    width = x2 - x1
    height = y2 - y1
    pad_x = int(width * padding_ratio)
    pad_y = int(height * padding_ratio)
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(frame_width, x2 + pad_x),
        min(frame_height, y2 + pad_y),
    )


def estimate_eye_boxes(
    face_box: tuple[int, int, int, int] | tuple[int, ...]
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    if len(face_box) == 4:
        x1, y1, x2, y2 = face_box
    elif len(face_box) >= 2:
        height = int(face_box[0])
        width = int(face_box[1])
        x1, y1, x2, y2 = 0, 0, width, height
    else:
        raise ValueError("estimate_eye_boxes() expected a face box or image shape.")
    face_w = x2 - x1
    face_h = y2 - y1

    eye_w = int(face_w * 0.28)
    eye_h = int(face_h * 0.14)
    eye_y1 = y1 + int(face_h * 0.32)
    eye_y2 = eye_y1 + eye_h

    left_eye_x1 = x1 + int(face_w * 0.12)
    left_eye_x2 = left_eye_x1 + eye_w
    right_eye_x2 = x2 - int(face_w * 0.12)
    right_eye_x1 = right_eye_x2 - eye_w

    return (
        (left_eye_x1, eye_y1, left_eye_x2, eye_y2),
        (right_eye_x1, eye_y1, right_eye_x2, eye_y2),
    )


def normalize_label(label: str) -> str:
    return label.replace("_", " ")


def draw_tag(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    text_color: tuple[int, int, int],
    bg_color: tuple[int, int, int] = (0, 0, 0),
    scale: float = 0.8,
    thickness: int = 2,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin
    top = max(0, y - text_height - baseline - 8)
    cv2.rectangle(
        image,
        (x, top),
        (x + text_width + 12, y + 4),
        bg_color,
        thickness=-1,
    )
    cv2.putText(
        image,
        text,
        (x + 6, y - 4),
        font,
        scale,
        text_color,
        thickness,
        lineType=cv2.LINE_AA,
    )


def choose_driver_face(
    face_boxes: list[tuple[int, int, int, int]] | list[dict[str, object]],
    frame_width: int,
    strategy: str,
    previous_driver_center_x: float | None = None,
) -> int | dict[str, object] | None:
    if not face_boxes:
        return None

    if isinstance(face_boxes[0], dict):
        faces = cast(list[dict[str, object]], face_boxes)
        box_list = [cast(tuple[int, int, int, int], face["bbox"]) for face in faces]
        selected = choose_driver_face(box_list, frame_width, strategy, previous_driver_center_x)
        if selected is None:
            return None
        return faces[cast(int, selected)]

    if strategy == "largest":
        return max(
            range(len(face_boxes)),
            key=lambda idx: (face_boxes[idx][2] - face_boxes[idx][0])
            * (face_boxes[idx][3] - face_boxes[idx][1]),
        )

    targets = {
        "left": frame_width * 0.25,
        "center": frame_width * 0.5,
        "right": frame_width * 0.75,
    }
    target_x = targets[strategy]
    return min(
        range(len(face_boxes)),
        key=lambda idx: abs(((face_boxes[idx][0] + face_boxes[idx][2]) / 2) - target_x),
    )


def main() -> None:
    args = parse_args()
    face_device, torch_device = resolve_devices(args.device)
    emotion_model_path, eye_model_path = resolve_classifier_paths(
        args.emotion_model,
        args.eye_model,
        args.runs_root,
    )
    face_model_path = ensure_face_model(args.face_model)

    face_detector = YOLO(str(face_model_path))
    emotion_classifier = load_timm_classifier(emotion_model_path, torch_device)
    eye_classifier = load_timm_classifier(eye_model_path, torch_device)

    print(f"Emotion model: {emotion_model_path}")
    print(f"Eye model: {eye_model_path}")
    print(f"Face model: {face_model_path}")

    cap = cv2.VideoCapture(args.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.capture_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.capture_height)
    if not cap.isOpened():
        raise RuntimeError("Could not open the webcam.")

    cv2.namedWindow("YOLO Focus Monitor", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLO Focus Monitor", args.window_width, args.window_height)

    frames_since_inference = 0
    cached_face_boxes: list[tuple[int, int, int, int]] = []
    cached_face_labels: list[tuple[str, float]] = []
    cached_face_top3: list[list[tuple[str, float]]] = []
    cached_eye_labels: list[tuple[str, float]] = []
    closed_eye_started_at: float | None = None
    last_frame_time = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_h, frame_w = frame.shape[:2]
            run_inference = frames_since_inference == 0 or args.skip_frames == 0

            if run_inference:
                face_result = face_detector.predict(
                    source=frame,
                    imgsz=args.face_imgsz,
                    conf=args.face_confidence,
                    device=face_device,
                    verbose=False,
                    max_det=args.max_faces,
                )[0]

                face_boxes: list[tuple[int, int, int, int]] = []
                face_crops: list[np.ndarray] = []
                eye_crops: list[np.ndarray] = []

                boxes = face_result.boxes
                if boxes is not None:
                    xyxy_array = cast(np.ndarray, boxes.xyxy.cpu().numpy())
                    for xyxy in xyxy_array:
                        x1, y1, x2, y2 = [int(v) for v in xyxy.tolist()]
                        expanded = expand_box((x1, y1, x2, y2), frame_w, frame_h, args.padding)
                        face_boxes.append(expanded)
                        fx1, fy1, fx2, fy2 = expanded
                        face_crops.append(frame[fy1:fy2, fx1:fx2].copy())

                        left_eye_box, right_eye_box = estimate_eye_boxes(expanded)
                        for ex1, ey1, ex2, ey2 in (left_eye_box, right_eye_box):
                            eye_crops.append(frame[ey1:ey2, ex1:ex2].copy())

                face_labels = classify_crops(emotion_classifier, face_crops, torch_device)
                face_top3 = classify_crops_with_topk(
                    emotion_classifier, face_crops, torch_device, topk=3
                )
                eye_labels = classify_crops(eye_classifier, eye_crops, torch_device)

                cached_face_boxes = face_boxes
                cached_face_labels = face_labels
                cached_face_top3 = face_top3
                cached_eye_labels = eye_labels
                frames_since_inference = args.skip_frames
            else:
                frames_since_inference -= 1

            driver_idx = choose_driver_face(cached_face_boxes, frame_w, args.driver_side)

            current_closed = False
            for idx, face_box in enumerate(cached_face_boxes):
                x1, y1, x2, y2 = face_box
                label, confidence = cached_face_labels[idx]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                draw_tag(
                    frame,
                    f"{normalize_label(label)} {confidence:.2f}",
                    (x1, max(20, y1)),
                    (0, 255, 0),
                    (0, 0, 0),
                )

                left_eye_box, right_eye_box = estimate_eye_boxes(face_box)
                eye_predictions = cached_eye_labels[idx * 2 : idx * 2 + 2]
                both_closed = (
                    len(eye_predictions) == 2
                    and all(pred[0] == "closed_eye" for pred in eye_predictions)
                )
                if idx == driver_idx:
                    current_closed = both_closed
                    if args.print_emotion_top3 and idx < len(cached_face_top3):
                        print(
                            "Driver emotion top-3:",
                            format_topk_prediction(cached_face_top3[idx]),
                        )

                for eye_box, eye_prediction in zip(
                    (left_eye_box, right_eye_box), eye_predictions
                ):
                    ex1, ey1, ex2, ey2 = eye_box
                    eye_label, eye_confidence = eye_prediction
                    cv2.rectangle(frame, (ex1, ey1), (ex2, ey2), (255, 0, 0), 2)
                    draw_tag(
                        frame,
                        f"{normalize_label(eye_label)} {eye_confidence:.2f}",
                        (ex1, min(frame_h - 8, ey2 + 26)),
                        (255, 0, 0),
                        (0, 0, 0),
                        scale=0.7,
                    )

            now = time.perf_counter()
            if current_closed:
                if closed_eye_started_at is None:
                    closed_eye_started_at = now
            else:
                closed_eye_started_at = None

            warning_active = (
                closed_eye_started_at is not None
                and (now - closed_eye_started_at) >= args.focus_seconds
            )
            if warning_active:
                draw_tag(
                    frame,
                    "Please stay focused",
                    (30, 110),
                    (0, 0, 255),
                    (0, 0, 0),
                    scale=1.6,
                    thickness=4,
                )

            fps = 1.0 / max(now - last_frame_time, 1e-6)
            last_frame_time = now
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.4,
                (255, 255, 0),
                3,
                lineType=cv2.LINE_AA,
            )

            cv2.imshow("YOLO Focus Monitor", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in {27, ord("q")}:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
