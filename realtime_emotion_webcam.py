from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run real-time facial expression recognition from a webcam."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to a trained YOLOv8 classification checkpoint. Defaults to the latest best.pt under G:\\731\\runs.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Webcam index passed to OpenCV.",
    )
    parser.add_argument("--imgsz", type=int, default=224, help="Classification image size.")
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help='Inference device, for example "0" or "cpu".',
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.15,
        help="Extra padding around the detected face box.",
    )
    parser.add_argument(
        "--min-face-size",
        type=int,
        default=80,
        help="Minimum face size in pixels.",
    )
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=0,
        help="Skip N frames between predictions to improve FPS.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.35,
        help="Do not draw a label below this confidence.",
    )
    return parser.parse_args()


def resolve_model_path(explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        if not explicit_path.exists():
            raise FileNotFoundError(f"Model file not found: {explicit_path}")
        return explicit_path

    candidates = sorted(
        Path(r"G:\731\runs").glob("**/weights/best.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            "No trained model found under G:\\731\\runs. Train the model first."
        )
    return candidates[0]


def expand_box(
    x: int,
    y: int,
    w: int,
    h: int,
    frame_width: int,
    frame_height: int,
    padding: float,
) -> tuple[int, int, int, int]:
    pad_w = int(w * padding)
    pad_h = int(h * padding)
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(frame_width, x + w + pad_w)
    y2 = min(frame_height, y + h + pad_h)
    return x1, y1, x2, y2


def draw_label(frame, text: str, x1: int, y1: int) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    y_top = max(0, y1 - text_h - baseline - 10)
    cv2.rectangle(frame, (x1, y_top), (x1 + text_w + 10, y_top + text_h + baseline + 10), (0, 0, 0), -1)
    cv2.putText(
        frame,
        text,
        (x1 + 5, y_top + text_h + 2),
        font,
        scale,
        (0, 255, 0),
        thickness,
        cv2.LINE_AA,
    )


def main() -> None:
    args = parse_args()
    model_path = resolve_model_path(args.model)
    model = YOLO(str(model_path))

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if face_cascade.empty():
        raise RuntimeError("Failed to load OpenCV Haar cascade for face detection.")

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open webcam index {args.camera_index}.")

    window_name = "YOLOv8 Facial Expression Recognition"
    frame_index = 0
    fps = 0.0
    previous_time = time.perf_counter()
    cached_predictions: list[tuple[tuple[int, int, int, int], str, float]] = []

    print(f"Using model: {model_path}")
    print("Press q to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read a frame from the webcam.")
                break

            frame_h, frame_w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detections = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(args.min_face_size, args.min_face_size),
            )

            should_predict = args.skip_frames == 0 or frame_index % (args.skip_frames + 1) == 0
            if should_predict:
                crops = []
                boxes = []
                for x, y, w, h in detections:
                    x1, y1, x2, y2 = expand_box(
                        x, y, w, h, frame_w, frame_h, args.padding
                    )
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    boxes.append((x1, y1, x2, y2))

                cached_predictions = []
                if crops:
                    results = model.predict(
                        source=crops,
                        imgsz=args.imgsz,
                        device=args.device,
                        verbose=False,
                    )
                    for box, result in zip(boxes, results):
                        probs = result.probs
                        if probs is None:
                            continue
                        top1 = int(probs.top1)
                        confidence = float(probs.top1conf)
                        label = str(model.names[top1])
                        cached_predictions.append((box, label, confidence))

            for (x1, y1, x2, y2), label, confidence in cached_predictions:
                color = (0, 255, 0) if confidence >= args.min_confidence else (0, 165, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                if confidence >= args.min_confidence:
                    draw_label(frame, f"{label} {confidence:.2f}", x1, y1)
                else:
                    draw_label(frame, f"uncertain {confidence:.2f}", x1, y1)

            current_time = time.perf_counter()
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
            cv2.imshow(window_name, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
