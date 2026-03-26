from __future__ import annotations

import argparse
import time
import urllib.request
from pathlib import Path

import cv2
from ultralytics import YOLO


FACE_MODEL_URL = (
    "https://github.com/lindevs/yolov8-face/releases/latest/download/"
    "yolov8n-face-lindevs.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO-based face detection, emotion recognition, and eye-state monitoring from a webcam."
    )
    parser.add_argument(
        "--emotion-model",
        type=Path,
        default=None,
        help="Path to a trained YOLOv8 emotion classification checkpoint.",
    )
    parser.add_argument(
        "--eye-model",
        type=Path,
        default=None,
        help="Path to a trained YOLOv8 eye-state classification checkpoint.",
    )
    parser.add_argument(
        "--face-model",
        type=Path,
        default=Path(r"G:\731\weights\yolov8n-face-lindevs.pt"),
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
        help='Inference device, for example "0" or "cpu".',
    )
    parser.add_argument(
        "--face-imgsz",
        type=int,
        default=640,
        help="YOLO face detector input size.",
    )
    parser.add_argument(
        "--cls-imgsz",
        type=int,
        default=224,
        help="Classification image size for emotion and eye-state models.",
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
        default=3.0,
        help="Closed-eye duration before the focus warning is shown.",
    )
    return parser.parse_args()


def resolve_latest_model(directory: Path, prefix: str) -> Path:
    candidates = sorted(
        directory.glob(f"{prefix}*/weights/best.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No model matching {prefix}* was found under {directory}."
        )
    return candidates[0]


def resolve_classifier_paths(
    emotion_path: Path | None, eye_path: Path | None
) -> tuple[Path, Path]:
    runs_dir = Path(r"G:\731\runs")
    resolved_emotion = emotion_path or resolve_latest_model(runs_dir, "emotion_")
    resolved_eye = eye_path or resolve_latest_model(runs_dir, "eye_")

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


def classify_crops(
    model: YOLO, crops_rgb: list, imgsz: int, device: str
) -> list[tuple[str, float]]:
    results = model.predict(source=crops_rgb, imgsz=imgsz, device=device, verbose=False)
    predictions = []
    for result in results:
        probs = result.probs
        if probs is None:
            predictions.append(("unknown", 0.0))
            continue
        top1 = int(probs.top1)
        predictions.append((str(model.names[top1]), float(probs.top1conf)))
    return predictions


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
    emotion_model_path, eye_model_path = resolve_classifier_paths(
        args.emotion_model, args.eye_model
    )
    face_model_path = ensure_face_model(args.face_model)

    face_detector = YOLO(str(face_model_path))
    emotion_model = YOLO(str(emotion_model_path))
    eye_model = YOLO(str(eye_model_path))

    cap = cv2.VideoCapture(args.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.capture_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.capture_height)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open webcam index {args.camera_index}.")

    window_name = "YOLO Focus Monitor"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, args.window_width, args.window_height)

    frame_index = 0
    fps = 0.0
    previous_time = time.perf_counter()
    closed_eye_start: float | None = None
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
                    device=args.device,
                    conf=args.face_confidence,
                    verbose=False,
                    max_det=args.max_faces,
                )[0]

                boxes = []
                crops_rgb = []
                if face_result.boxes is not None:
                    raw_boxes = face_result.boxes.xyxy.cpu().numpy().astype(int)
                    raw_confidences = face_result.boxes.conf.cpu().numpy().tolist()
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
                    emotion_predictions = classify_crops(
                        emotion_model, crops_rgb, args.cls_imgsz, args.device
                    )
                    eye_predictions = classify_crops(
                        eye_model, crops_rgb, args.cls_imgsz, args.device
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
                                "emotion_label": emotion_pred[0],
                                "emotion_confidence": emotion_pred[1],
                                "eye_label": eye_pred[0],
                                "eye_confidence": eye_pred[1],
                                "eye_boxes": eye_boxes,
                            }
                        )

            primary_face = max(cached_faces, key=lambda item: item["area"], default=None)
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

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
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

            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
