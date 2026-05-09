from __future__ import annotations

import argparse
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import cv2
import torch
from PyQt5.QtCore import QObject, Qt, QThread, pyqtSignal, pyqtSlot   
from PyQt5.QtGui import QCloseEvent, QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from ultralytics import YOLO

from drivesense.backend.chatbot import (
    DEFAULT_MODEL,
    DriverAssistantChatbot,
    SUPPORTED_LLM_MODELS,
    sanitize_driver_state,
)
from drivesense.backend.focus_monitor import FocusMonitor, FocusMonitorConfig
from drivesense.backend.vision import (
    apply_emotion_postprocess,
    build_voice_pipeline,
    classify_crops,
    classify_crops_with_topk,
    choose_driver_face,
    draw_tag,
    ensure_face_model,
    estimate_eye_boxes,
    expand_box,
    filter_primary_face_candidates,
    format_topk_prediction,
    load_timm_classifier,
    normalize_checkpoint_path,
    normalize_label,
    prepare_eye_debug_dir,
    resolve_devices,
    resolve_latest_timm_model,
    save_eye_debug_crop,
)
from drivesense.backend.speech import TextToSpeech, WhisperTranscriber, record_microphone_audio
from drivesense.backend.wake_word import WakeWordListener, WakeWordConfig, ContinuedConversationListener


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "runs_timm"
DEFAULT_WEIGHTS_ROOT = PROJECT_ROOT / "weights"


EMOTION_COLORS_BGR = {
    "anger": (0, 0, 255),
    "fear": (0, 165, 255),
    "sad": (255, 90, 0),
    "happy": (0, 200, 0),
    "surprise": (0, 255, 255),
    "disgust": (180, 0, 180),
    "neutral": (180, 180, 180),
}
EMOTION_COLORS_HEX = {
    "anger": "#e53935",
    "fear": "#fb8c00",
    "sad": "#1e88e5",
    "happy": "#43a047",
    "surprise": "#fdd835",
    "disgust": "#8e24aa",
    "neutral": "#607d8b",
}

EMOTION_TO_RISK = {
    "anger": "HIGH",
    "fear": "HIGH",
    "sad": "MED",
    "disgust": "LOW",
    "surprise": "MED",
    "happy": "OK",
    "neutral": "OK",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PyQt5 GUI for webcam emotion monitoring, OpenRouter chat, and local speech transcription."
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
        default=DEFAULT_RUNS_ROOT / "eye_efficientnet_b0" / "best_model.pth",
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
        help="YOLO face detector path.",
    )
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam index.")
    parser.add_argument("--capture-width", type=int, default=1280, help="Webcam capture width.")
    parser.add_argument("--capture-height", type=int, default=720, help="Webcam capture height.")
    parser.add_argument("--device", type=str, default="cuda", help='Inference device, for example "cuda" or "cpu".')
    parser.add_argument("--face-imgsz", type=int, default=640, help="YOLO face detector image size.")
    parser.add_argument("--face-confidence", type=float, default=0.35, help="Minimum face confidence.")
    parser.add_argument(
        "--classification-confidence",
        type=float,
        default=0.35,
        help="Minimum classification confidence for displayed labels.",
    )
    parser.add_argument("--padding", type=float, default=0.15, help="Face padding ratio.")
    parser.add_argument("--max-faces", type=int, default=5, help="Maximum number of faces to process.")
    parser.add_argument(
        "--min-face-area-ratio",
        type=float,
        default=0.015,
        help="Reject faces smaller than this fraction of the full frame area.",
    )
    parser.add_argument(
        "--relative-min-face-scale",
        type=float,
        default=0.35,
        help="Reject faces much smaller than the largest visible face.",
    )
    parser.add_argument(
        "--focus-seconds",
        type=float,
        default=2.0,
        help="Closed-eye duration before the focus warning appears.",
    )
    parser.add_argument(
        "--eye-warmup-seconds",
        type=float,
        default=5.0,
        help="Seconds after startup before eye-state results can affect focus alerts.",
    )
    parser.add_argument(
        "--cooldown-seconds",
        type=float,
        default=30.0,
        help="Minimum seconds between two voice interventions.",
    )
    parser.add_argument(
        "--enable-voice-dialogue",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable the full record/transcribe/LLM/speak loop after the beep.",
    )
    parser.add_argument(
        "--driver-side",
        type=str,
        choices=["left", "center", "right", "largest"],
        default="left",
        help="Heuristic used to choose the driver's face when multiple faces are visible.",
    )
    parser.add_argument("--default-llm-model", type=str, default=DEFAULT_MODEL, help="Default OpenRouter model.")
    parser.add_argument("--default-temperature", type=float, default=1.0, help="Default chatbot temperature.")
    parser.add_argument("--whisper-model-size", type=str, default="tiny", help="faster-whisper model size (tiny=fastest, base=more accurate).")
    parser.add_argument(
        "--max-recording-duration",
        type=float,
        default=8.0,
        help="Maximum recording duration in seconds (VAD may stop earlier).",
    )
    parser.add_argument(
        "--silence-threshold",
        type=float,
        default=1.0,
        help="Seconds of silence before auto-stopping recording (VAD).",
    )
    parser.add_argument(
        "--print-emotion-top3",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print the driver's emotion top-3 probabilities for each frame.",
    )
    parser.add_argument(
        "--save-eye-crops",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save the driver eye crops that are sent to the eye-state model.",
    )
    parser.add_argument(
        "--save-eye-crops-dir",
        type=Path,
        default=PROJECT_ROOT / "debug_exports" / "eye_crops",
        help="Directory for exported eye crops.",
    )
    parser.add_argument(
        "--save-eye-crops-limit",
        type=int,
        default=200,
        help="Maximum number of eye crops to save per run.",
    )
    return parser.parse_args()


def emotion_color_bgr(emotion: str) -> tuple[int, int, int]:
    return EMOTION_COLORS_BGR.get(emotion, EMOTION_COLORS_BGR["neutral"])


def emotion_color_hex(emotion: str) -> str:
    return EMOTION_COLORS_HEX.get(emotion, EMOTION_COLORS_HEX["neutral"])


def risk_color_hex(risk: str) -> str:
    normalized = risk.strip().upper()
    if normalized == "HIGH":
        return "#ef4444"
    if normalized in {"MED", "LOW"}:
        return "#f59e0b"
    return "#34d399"


def build_context_summary(driver_state: dict[str, Any]) -> str:
    if not driver_state.get("driver_detected", False):
        return "Current context: no confident driver detected"
    emotion = normalize_label(str(driver_state.get("emotion", "neutral")))
    if driver_state.get("eye_warmup_active", False):
        remaining = float(driver_state.get("eye_warmup_remaining", 0.0))
        eyes = f"eye warm-up {remaining:.1f}s"
    else:
        eyes = normalize_label(str(driver_state.get("eye_label", "open_eye")))
    risk = str(driver_state.get("risk", "OK")).upper()
    return f"Current context: {emotion}, {eyes}, {risk}"


class ChatBubble(QFrame):
    def __init__(self, text: str, is_user: bool) -> None:
        super().__init__()
        bubble = QLabel(text)
        bubble.setWordWrap(True)
        bubble.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        bubble.setStyleSheet(
            (
                "background-color: #0059b5; color: white; border-radius: 18px; "
                "padding: 14px 18px; font-size: 15px; line-height: 1.4;"
            )
            if is_user
            else (
                "background-color: #eeeeee; color: #1a1c1c; border-radius: 18px; "
                "padding: 14px 18px; font-size: 15px; line-height: 1.4;"
            )
        )
        bubble.setMaximumWidth(340)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 8, 0, 8)
        if is_user:
            layout.addStretch()
            layout.addWidget(bubble)
        else:
            layout.addWidget(bubble)
            layout.addStretch()


class VisionWorker(QObject):
    frame_ready = pyqtSignal(QImage)
    state_ready = pyqtSignal(object)
    voice_dialogue_ready = pyqtSignal(object)
    voice_dialogue_error = pyqtSignal(str)
    error = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.args = args
        self._running = True
        self.current_chat_model = args.default_llm_model
        self.current_temperature = float(args.default_temperature)

    def update_chat_settings(self, model: str, temperature: float) -> None:
        self.current_chat_model = model
        self.current_temperature = float(temperature)

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        cap = None
        try:
            face_device, torch_device = resolve_devices(self.args.device)
            emotion_model_path = normalize_checkpoint_path(
                self.args.emotion_model
            ) or resolve_latest_timm_model(self.args.runs_root, "emotion")
            eye_model_path = normalize_checkpoint_path(
                self.args.eye_model
            ) or resolve_latest_timm_model(self.args.runs_root, "eye")
            face_model_path = ensure_face_model(self.args.face_model)

            face_detector = YOLO(str(face_model_path))
            emotion_classifier = load_timm_classifier(emotion_model_path, torch_device)
            eye_classifier = load_timm_classifier(eye_model_path, torch_device)
            tts = TextToSpeech()
            voice_pipeline = build_voice_pipeline(self.args.enable_voice_dialogue)
            focus_monitor = FocusMonitor(
                config=FocusMonitorConfig(
                    closed_eye_seconds=self.args.focus_seconds,
                    cooldown_seconds=self.args.cooldown_seconds,
                ),
                tts=tts,
                voice_pipeline=voice_pipeline,
                on_voice_result=lambda result: self.voice_dialogue_ready.emit(asdict(result)),
                on_voice_error=self.voice_dialogue_error.emit,
            )

            cap = cv2.VideoCapture(self.args.camera_index)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.args.capture_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.args.capture_height)
            if not cap.isOpened():
                raise RuntimeError(f"Unable to open webcam index {self.args.camera_index}.")

            last_state = {
                "driver_detected": False,
                "driver_confident": False,
                "emotion": "neutral",
                "emotion_confidence": 0.0,
                "emotion_secondary": "",
                "emotion_secondary_confidence": 0.0,
                "eye_label": "warming_up",
                "eye_confidence": 0.0,
                "eye_warmup_active": True,
                "risk": "OK",
                "focus_alert": False,
                "focus_level": 0,
                "closed_eye_duration": 0.0,
                "trigger_reason": "",
                "driver_side": self.args.driver_side,
                "emotion_model_path": str(emotion_model_path),
                "eye_model_path": str(eye_model_path),
            }
            previous_time = time.perf_counter()
            vision_started_at = previous_time
            previous_driver_center_x: float | None = None
            saved_eye_crop_count = 0
            frame_index = 0
            run_prefix = time.strftime("%Y%m%d_%H%M%S")
            if self.args.save_eye_crops:
                prepare_eye_debug_dir(self.args.save_eye_crops_dir)

            while self._running:
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("Failed to read a frame from the webcam.")
                frame_index += 1
                elapsed_since_start = time.perf_counter() - vision_started_at
                eye_warmup_active = elapsed_since_start < self.args.eye_warmup_seconds
                eye_warmup_remaining = max(self.args.eye_warmup_seconds - elapsed_since_start, 0.0)

                frame_h, frame_w = frame.shape[:2]
                face_result = face_detector.predict(
                    source=frame,
                    imgsz=self.args.face_imgsz,
                    device=face_device,
                    conf=self.args.face_confidence,
                    verbose=False,
                    max_det=self.args.max_faces,
                )[0]

                detections = []
                face_crops_bgr = []
                eye_crops_bgr = []
                if face_result.boxes is not None:
                    raw_boxes_data: Any = face_result.boxes.xyxy
                    raw_conf_data: Any = face_result.boxes.conf
                    if hasattr(raw_boxes_data, "cpu"):
                        raw_boxes = raw_boxes_data.cpu().numpy().astype(int)
                    else:
                        raw_boxes = raw_boxes_data.astype(int)
                    if hasattr(raw_conf_data, "cpu"):
                        raw_confidences = raw_conf_data.cpu().numpy().tolist()
                    else:
                        raw_confidences = raw_conf_data.tolist()
                    for raw_box, detection_conf in zip(raw_boxes, raw_confidences):
                        x1, y1, x2, y2 = expand_box(
                            int(raw_box[0]),
                            int(raw_box[1]),
                            int(raw_box[2]),
                            int(raw_box[3]),
                            frame_w,
                            frame_h,
                            self.args.padding,
                        )
                        face_crop = frame[y1:y2, x1:x2]
                        if face_crop.size == 0:
                            continue
                        detections.append((x1, y1, x2, y2, float(detection_conf)))
                        face_crops_bgr.append(face_crop.copy())

                        left_eye_box, right_eye_box = estimate_eye_boxes((x1, y1, x2, y2))
                        for ex1, ey1, ex2, ey2 in (left_eye_box, right_eye_box):
                            eye_crop = frame[ey1:ey2, ex1:ex2]
                            if eye_crop.size == 0:
                                eye_crop = face_crop.copy()
                            eye_crops_bgr.append(eye_crop.copy())

                emotion_predictions = classify_crops_with_topk(
                    emotion_classifier,
                    face_crops_bgr,
                    torch_device,
                    topk=3,
                )
                if eye_warmup_active:
                    eye_predictions = [("open_eye", 0.0)] * len(eye_crops_bgr)
                else:
                    eye_predictions = classify_crops(eye_classifier, eye_crops_bgr, torch_device)
                primary_face = None
                if detections and emotion_predictions and eye_predictions:
                    candidate_indices = filter_primary_face_candidates(
                        [cast(tuple[int, int, int, int], det[:4]) for det in detections],
                        frame_w,
                        frame_h,
                        min_face_area_ratio=self.args.min_face_area_ratio,
                        relative_min_face_scale=self.args.relative_min_face_scale,
                    )
                    faces = []
                    for index, (detection, emotion_prediction) in enumerate(
                        zip(detections, emotion_predictions)
                    ):
                        x1, y1, x2, y2, detection_conf = detection
                        emotion_topk = cast(list[tuple[str, float]], emotion_prediction)
                        emotion_label = emotion_topk[0][0]
                        emotion_confidence = float(emotion_topk[0][1])
                        emotion_label, emotion_confidence = apply_emotion_postprocess(
                            emotion_label,
                            emotion_confidence,
                        )

                        per_face_eye_predictions = eye_predictions[index * 2 : index * 2 + 2]
                        avg_closed_conf = (
                            sum(score for label, score in per_face_eye_predictions if label == "closed_eye")
                            / max(len(per_face_eye_predictions), 1)
                        )
                        if eye_warmup_active:
                            eye_label = "warming_up"
                            eye_confidence = 0.0
                        elif avg_closed_conf > 0.80:
                            eye_label = "closed_eye"
                            eye_confidence = avg_closed_conf
                        else:
                            eye_label = "open_eye"
                            eye_confidence = 1.0 - avg_closed_conf

                        faces.append(
                            {
                                "bbox": (x1, y1, x2, y2),
                                "area": (x2 - x1) * (y2 - y1),
                                "emotion": emotion_label,
                                "emotion_confidence": emotion_confidence,
                                "emotion_topk": emotion_topk,
                                "eye_label": eye_label,
                                "eye_confidence": eye_confidence,
                                "eye_boxes": estimate_eye_boxes((x1, y1, x2, y2)),
                                "eye_predictions": per_face_eye_predictions,
                                "detection_confidence": detection_conf,
                                "is_candidate": index in candidate_indices,
                            }
                        )

                    candidate_faces = [faces[idx] for idx in candidate_indices]
                    primary_face = choose_driver_face(
                        candidate_faces,
                        frame_w,
                        self.args.driver_side,
                        previous_driver_center_x,
                    )
                    if primary_face is not None:
                        px1, _, px2, _ = primary_face["bbox"]
                        previous_driver_center_x = ((px1 + px2) / 2.0) / max(frame_w, 1)
                    for face in faces:
                        x1, y1, x2, y2 = face["bbox"]
                        emotion = face["emotion"]
                        is_driver_face = face is primary_face
                        color = emotion_color_bgr(emotion) if is_driver_face else (120, 120, 120)
                        cv2.rectangle(
                            frame,
                            (x1, y1),
                            (x2, y2),
                            color,
                            4 if is_driver_face else 1,
                        )
                        if is_driver_face:
                            draw_tag(
                                frame,
                                f"{normalize_label(emotion)} {face['emotion_confidence']:.2f}",
                                (x1, y1),
                                color,
                            )
                            draw_tag(
                                frame,
                                f"{normalize_label(face['eye_label'])} {face['eye_confidence']:.2f}",
                                (x1, min(frame_h - 5, y2 + 28)),
                                (255, 0, 0),
                            )
                            draw_tag(frame, "driver", (x1, y2 + 56), (255, 255, 255))
                        elif face.get("is_candidate", False):
                            draw_tag(frame, "other face", (x1, y1), (220, 220, 220), (40, 40, 40), scale=0.6)

                        if (
                            is_driver_face
                            and self.args.save_eye_crops
                            and saved_eye_crop_count < self.args.save_eye_crops_limit
                        ):
                            eye_boxes = cast(tuple[tuple[int, int, int, int], tuple[int, int, int, int]], face["eye_boxes"])
                            per_face_eye_predictions = cast(list[tuple[str, float]], face["eye_predictions"])
                            for eye_offset, (eye_box, eye_prediction) in enumerate(
                                zip(eye_boxes, per_face_eye_predictions)
                            ):
                                if saved_eye_crop_count >= self.args.save_eye_crops_limit:
                                    break
                                ex1, ey1, ex2, ey2 = eye_box
                                eye_crop = frame[ey1:ey2, ex1:ex2].copy()
                                saved_path = save_eye_debug_crop(
                                    self.args.save_eye_crops_dir,
                                    run_prefix,
                                    frame_index,
                                    eye_offset,
                                    eye_crop,
                                    eye_prediction[0],
                                    eye_prediction[1],
                                )
                                if saved_path is not None:
                                    saved_eye_crop_count += 1

                driver_detected = primary_face is not None
                if primary_face is not None:
                    current_emotion = primary_face["emotion"]
                    emotion_confidence = float(primary_face["emotion_confidence"])
                    emotion_topk = cast(list[tuple[str, float]], primary_face.get("emotion_topk", []))
                    secondary_emotion = emotion_topk[1][0] if len(emotion_topk) > 1 else ""
                    secondary_emotion_confidence = float(emotion_topk[1][1]) if len(emotion_topk) > 1 else 0.0
                    current_eye_label = primary_face["eye_label"]
                    eye_confidence = float(primary_face["eye_confidence"])
                else:
                    current_emotion = "neutral"
                    emotion_confidence = 0.0
                    secondary_emotion = ""
                    secondary_emotion_confidence = 0.0
                    current_eye_label = "open_eye"
                    eye_confidence = 0.0
                    driver_detected = False
                if eye_warmup_active:
                    current_eye_label = "warming_up"
                    eye_confidence = 0.0

                if self.args.print_emotion_top3:
                    if primary_face is None:
                        print("Driver emotion top-3: no face", flush=True)
                    else:
                        print(
                            "Driver emotion top-3: "
                            f"{format_topk_prediction(primary_face.get('emotion_topk', []))}",
                            flush=True,
                        )

                if (
                    primary_face is not None
                    and current_eye_label == "closed_eye"
                ):
                    current_closed = True
                else:
                    current_closed = False

                risk = EMOTION_TO_RISK.get(current_emotion, "OK")
                current_driver_state = {
                    "driver_detected": driver_detected,
                    "driver_confident": driver_detected,
                    "emotion": current_emotion,
                    "emotion_confidence": emotion_confidence,
                    "emotion_secondary": secondary_emotion,
                    "emotion_secondary_confidence": secondary_emotion_confidence,
                    "eye_label": current_eye_label,
                    "eye_confidence": eye_confidence,
                    "eye_warmup_active": eye_warmup_active,
                    "eye_warmup_remaining": eye_warmup_remaining,
                    "risk": risk,
                    "focus_alert": False,
                    "driver_side": self.args.driver_side,
                }
                focus_monitor.set_runtime_context(
                    chat_model=self.current_chat_model,
                    temperature=self.current_temperature,
                )
                focus_alert = focus_monitor.update(
                    eyes_closed=current_closed,
                    emotion=current_emotion,
                    driver_state=current_driver_state,
                )
                if focus_alert:
                    risk = "HIGH"
                current_driver_state["risk"] = risk
                current_driver_state["focus_alert"] = focus_alert
                current_driver_state.update(focus_monitor.get_state_snapshot())
                last_state = {
                    **current_driver_state,
                    "emotion_model_path": str(emotion_model_path),
                    "eye_model_path": str(eye_model_path),
                }

                now = time.perf_counter()
                fps = 1.0 / max(now - previous_time, 1e-6)
                previous_time = now
                cv2.putText(
                    frame,
                    f"FPS: {fps:.1f}",
                    (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    f"Eyes: {normalize_label(current_eye_label)}",
                    (10, frame_h - 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 0, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    f"Emotion: {normalize_label(current_emotion)}",
                    (10, frame_h - 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    emotion_color_bgr(current_emotion),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    f"Risk: {risk}",
                    (10, frame_h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    emotion_color_bgr(current_emotion),
                    2,
                    cv2.LINE_AA,
                )
                if last_state.get("focus_level", 0) >= 1:
                    alert_text = (
                        "Please stay focused"
                        if last_state.get("focus_level", 0) >= 2
                        else "Stay attentive"
                    )
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    scale = 1.0 if last_state.get("focus_level", 0) >= 2 else 0.8
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

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                qimage = QImage(
                    rgb_frame.data,
                    rgb_frame.shape[1],
                    rgb_frame.shape[0],
                    rgb_frame.strides[0],
                    QImage.Format_RGB888,
                ).copy()
                self.frame_ready.emit(qimage)
                self.state_ready.emit(last_state)
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            if cap is not None:
                cap.release()
            self.finished.emit()


class ChatWorker(QThread):
    result_ready = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(
        self,
        chatbot: DriverAssistantChatbot,
        emotion: str,
        user_message: str | None,
        conversation_history: list[dict[str, str]],
        model: str,
        temperature: float,
        auto_trigger: bool = False,
        driver_state: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.chatbot = chatbot
        self.emotion = emotion
        self.user_message = user_message
        self.conversation_history = conversation_history
        self.model = model
        self.temperature = temperature
        self.auto_trigger = auto_trigger
        self.driver_state = dict(driver_state) if driver_state else None

    def run(self) -> None:
        try:
            print(
                "ChatWorker start | "
                f"model={self.model} temperature={self.temperature:.2f} "
                f"emotion={self.emotion} auto_trigger={self.auto_trigger} "
                f"driver_state={sanitize_driver_state(self.driver_state)}",
                flush=True,
            )
            response = self.chatbot.generate_reply(
                emotion=self.emotion,
                user_message=self.user_message,
                model=self.model,
                temperature=self.temperature,
                conversation_history=self.conversation_history,
                auto_trigger=self.auto_trigger,
                driver_state=self.driver_state,
            )
            self.result_ready.emit(asdict(response))
        except Exception as exc:
            self.error.emit(
                f"model={self.model}, temperature={self.temperature:.2f}, error={exc}"
            )


class SpeechWorker(QThread):
    transcription_ready = pyqtSignal(str)
    error = pyqtSignal(str)

    _cache_lock = threading.Lock()
    _transcriber_cache: dict[tuple[str, str], WhisperTranscriber] = {}

    def __init__(
        self,
        model_size: str,
        duration_seconds: float = 5.0,
        vad_enabled: bool = True,
        min_duration_seconds: float = 0.5,
        silence_duration_seconds: float = 1.0,
    ) -> None:
        super().__init__()
        self.model_size = model_size
        self.duration_seconds = duration_seconds
        self.vad_enabled = vad_enabled
        self.min_duration_seconds = min_duration_seconds
        self.silence_duration_seconds = silence_duration_seconds
        self.stop_event = threading.Event()

    def stop_recording(self) -> None:
        self.stop_event.set()

    @classmethod
    def get_transcriber(cls, model_size: str) -> WhisperTranscriber:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        key = (model_size, device)
        with cls._cache_lock:
            if key not in cls._transcriber_cache:
                cls._transcriber_cache[key] = WhisperTranscriber(
                    model_size=model_size,
                    device=device,
                )
            return cls._transcriber_cache[key]

    def run(self) -> None:
        try:
            audio = record_microphone_audio(
                duration_seconds=self.duration_seconds,
                sample_rate=16000,
                stop_event=self.stop_event,
                vad_enabled=self.vad_enabled,
                min_duration_seconds=self.min_duration_seconds,
                silence_duration_seconds=self.silence_duration_seconds,
            )
            if audio.size == 0:
                raise RuntimeError("No audio was captured from the microphone.")

            transcriber = self.get_transcriber(self.model_size)
            result = transcriber.transcribe_audio(audio, sample_rate=16000)
            self.transcription_ready.emit(result.text)
        except Exception as exc:
            self.error.emit(str(exc))


class DriverAssistantWindow(QMainWindow):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.args = args
        self.setWindowTitle("DriveSense")
        self.resize(1500, 900)

        self.current_emotion = "neutral"
        self.current_risk = "OK"
        self.current_driver_state: dict[str, Any] = {
            "driver_detected": False,
            "driver_confident": False,
            "emotion": "neutral",
            "emotion_confidence": 0.0,
            "eye_label": "open_eye",
            "eye_confidence": 0.0,
            "eye_warmup_active": True,
            "eye_warmup_remaining": float(self.args.eye_warmup_seconds),
            "risk": "OK",
            "focus_alert": False,
            "focus_level": 0,
            "closed_eye_duration": 0.0,
            "trigger_reason": "",
            "driver_side": self.args.driver_side,
        }
        self.conversation_history: list[dict[str, str]] = []
        self.chat_worker: ChatWorker | None = None
        self.speech_worker: SpeechWorker | None = None
        self.last_used_model = self.args.default_llm_model
        self.last_selected_model = self.args.default_llm_model
        self.last_fallback_used = False
        self.wake_word_listening = False
        self.wake_word_listener: WakeWordListener | None = None
        self.continued_listener: ContinuedConversationListener | None = None
        self._in_continued_mode = False

        try:
            self.chatbot = DriverAssistantChatbot(app_title="DriveSense GUI")
        except Exception as exc:
            self.chatbot = None
            QMessageBox.warning(self, "OpenRouter", str(exc))

        central = QWidget()
        central.setStyleSheet("background-color: #f9f9f9; color: #1a1c1c; font-family: Inter, Segoe UI, Arial;")
        self.setCentralWidget(central)
        page_layout = QVBoxLayout(central)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(0)

        nav_bar = QFrame()
        nav_bar.setFixedHeight(72)
        nav_bar.setStyleSheet("background-color: #1a1c1c; color: #ffffff;")
        nav_layout = QHBoxLayout(nav_bar)
        nav_layout.setContentsMargins(72, 0, 72, 0)
        nav_layout.setSpacing(26)
        brand_label = QLabel("DriveSense")
        brand_label.setStyleSheet("color: #ffffff; font-size: 28px; font-weight: 800;")
        nav_layout.addWidget(brand_label)
        for item, active in [("Dashboard", True), ("Logs", False), ("Telemetry", False), ("Models", False)]:
            nav_item = QLabel(item)
            nav_item.setStyleSheet(
                "font-size: 15px; font-weight: 700; padding: 22px 0 17px 0; "
                + ("color: #ffffff; border-bottom: 2px solid #0071e3;" if active else "color: #c1c6d6;")
            )
            nav_layout.addWidget(nav_item)
        nav_layout.addStretch()
        for text in ["Settings", "Account"]:
            nav_button = QPushButton(text)
            nav_button.setCursor(Qt.CursorShape.PointingHandCursor)
            nav_button.setStyleSheet(
                "QPushButton { background: transparent; color: #ffffff; border: none; "
                "font-size: 13px; font-weight: 600; padding: 8px 10px; }"
                "QPushButton:hover { background-color: #414753; border-radius: 14px; }"
            )
            nav_layout.addWidget(nav_button)
        page_layout.addWidget(nav_bar)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(72, 40, 72, 28)
        content_layout.setSpacing(28)
        page_layout.addWidget(content, 1)

        title_label = QLabel("Driver Monitoring")
        title_label.setStyleSheet("font-size: 48px; font-weight: 800; color: #1a1c1c;")
        subtitle_label = QLabel("System Active")
        subtitle_label.setStyleSheet("font-size: 21px; color: #414753;")
        content_layout.addWidget(title_label)
        content_layout.addWidget(subtitle_label)

        main_layout = QHBoxLayout()
        main_layout.setSpacing(36)
        content_layout.addLayout(main_layout, 1)

        left_panel = QVBoxLayout()
        left_panel.setSpacing(30)
        right_panel = QVBoxLayout()
        right_panel.setSpacing(26)
        main_layout.addLayout(left_panel, 8)
        main_layout.addLayout(right_panel, 4)

        self.video_label = QLabel("Camera starting...")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(820, 520)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_label.setStyleSheet(
            "background-color: #2f3131; color: #ffffff; border-radius: 24px; "
            "border: 1px solid #e2e2e2; font-size: 16px; font-weight: 700;"
        )
        left_panel.addWidget(self.video_label)

        info_card = QFrame()
        info_card.setStyleSheet(
            "QFrame { background-color: #eeeeee; color: #1a1c1c; border-radius: 18px; "
            "border: 1px solid #e2e2e2; }"
        )
        info_layout = QVBoxLayout(info_card)
        info_layout.setContentsMargins(28, 26, 28, 26)
        info_layout.setSpacing(12)
        telemetry_title = QLabel("Telemetry Logs")
        telemetry_title.setStyleSheet("font-size: 26px; font-weight: 800; color: #1a1c1c; border: none;")
        info_layout.addWidget(telemetry_title)

        def add_metric_row(label_text: str, value_label: QLabel) -> None:
            row = QFrame()
            row.setStyleSheet("QFrame { border: none; border-bottom: 1px solid #dadada; background: transparent; }")
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 10, 0, 10)
            label = QLabel(label_text)
            label.setStyleSheet("color: #1a1c1c; font-size: 17px; border: none; background: transparent;")
            value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            value_label.setStyleSheet("color: #1a1c1c; font-size: 17px; font-weight: 700; border: none; background: transparent;")
            row_layout.addWidget(label)
            row_layout.addWidget(value_label, 1)
            info_layout.addWidget(row)

        self.driver_detected_label = QLabel("No")
        self.emotion_label = QLabel("neutral")
        self.eye_label = QLabel("open eye")
        self.risk_label = QLabel("OK")
        self.focus_label = QLabel("OK")
        self.reason_label = QLabel("none")
        add_metric_row("Driver detected", self.driver_detected_label)
        add_metric_row("Emotion", self.emotion_label)
        add_metric_row("Eyes", self.eye_label)
        add_metric_row("Risk", self.risk_label)
        add_metric_row("Focus", self.focus_label)
        add_metric_row("Reason", self.reason_label)

        meta_block = QVBoxLayout()
        meta_block.setContentsMargins(0, 14, 0, 0)
        meta_block.setSpacing(4)
        self.driver_label = QLabel("Driver heuristic: left")
        self.model_label = QLabel(f"Selected model: {self.args.default_llm_model}")
        self.last_model_label = QLabel(f"Last reply model: {self.args.default_llm_model}")
        self.model_path_label = QLabel("Emotion model: loading...")
        self.eye_model_path_label = QLabel("Eye model: loading...")
        self.status_label = QLabel("Status: ready")
        for label in [
            self.driver_label,
            self.model_label,
            self.last_model_label,
            self.model_path_label,
            self.eye_model_path_label,
            self.status_label,
        ]:
            label.setWordWrap(True)
            label.setStyleSheet("color: #1a1c1c; font-size: 14px; border: none; background: transparent;")
            meta_block.addWidget(label)
        info_layout.addLayout(meta_block)
        left_panel.addWidget(info_card)

        settings_card = QFrame()
        settings_card.setStyleSheet(
            "QFrame { background-color: #ffffff; border: 1px solid #dadada; border-radius: 18px; }"
        )
        settings_layout = QVBoxLayout(settings_card)
        settings_layout.setContentsMargins(20, 18, 20, 18)
        settings_layout.setSpacing(16)

        controls_row = QHBoxLayout()
        controls_row.setSpacing(12)
        self.model_combo = QComboBox()
        self.model_combo.addItems(SUPPORTED_LLM_MODELS)
        default_index = self.model_combo.findText(self.args.default_llm_model)
        if default_index >= 0:
            self.model_combo.setCurrentIndex(default_index)
        self.model_combo.currentTextChanged.connect(self.handle_model_selection_change)
        self.model_combo.setStyleSheet(
            "QComboBox { background-color: #eeeeee; color: #1a1c1c; border: none; border-radius: 8px; padding: 8px 12px; font-size: 15px; }"
            "QComboBox::drop-down { border: none; }"
            "QComboBox::down-arrow { image: none; }"
            "QComboBox QAbstractItemView { background-color: #ffffff; color: #1a1c1c; border: 1px solid #dadada; selection-background-color: #0071e3; }"
        )
        self.temperature_spin = QDoubleSpinBox()
        self.temperature_spin.setRange(0.0, 2.0)
        self.temperature_spin.setSingleStep(0.1)
        self.temperature_spin.setValue(self.args.default_temperature)
        self.temperature_spin.setStyleSheet("QDoubleSpinBox { background-color: #eeeeee; color: #1a1c1c; border: none; border-radius: 8px; padding: 8px 10px; font-size: 15px; }")
        label_model = QLabel("LLM Model")
        label_model.setStyleSheet("color: #1f2937; font-weight: 700; font-size: 13px; border: none; background: transparent;")
        label_temp = QLabel("Temperature")
        label_temp.setStyleSheet("color: #1f2937; font-weight: 700; font-size: 13px; border: none; background: transparent;")
        controls_row.addWidget(label_model)
        controls_row.addWidget(self.model_combo, 1)
        controls_row.addWidget(label_temp)
        controls_row.addWidget(self.temperature_spin)
        settings_layout.addLayout(controls_row)

        self.context_label = QLabel("Current context: neutral, open eye, OK")
        self.context_label.setWordWrap(True)
        self.context_label.setStyleSheet(
            "background-color: transparent; color: #1f2937; border-top: 1px solid #e5e5e5; "
            "padding: 12px 0 0 0; font-size: 14px;"
        )
        settings_layout.addWidget(self.context_label)
        right_panel.addWidget(settings_card)

        chat_card = QFrame()
        chat_card.setStyleSheet(
            "QFrame { background-color: #ffffff; border: 1px solid #dadada; border-radius: 18px; }"
        )
        chat_card_layout = QVBoxLayout(chat_card)
        chat_card_layout.setContentsMargins(0, 0, 0, 0)
        chat_card_layout.setSpacing(0)
        self.chat_scroll = QScrollArea()
        self.chat_scroll.setWidgetResizable(True)
        self.chat_scroll.setStyleSheet(
            "QScrollArea { background-color: #ffffff; border: none; border-radius: 18px; }"
            "QScrollBar:vertical { background-color: #f4f4f4; width: 8px; }"
            "QScrollBar::handle:vertical { background-color: #dadada; border-radius: 4px; }"
        )
        self.chat_container = QWidget()
        self.chat_container.setStyleSheet("background-color: #ffffff; border: none;")
        self.chat_layout = QVBoxLayout(self.chat_container)
        self.chat_layout.setContentsMargins(22, 22, 22, 22)
        self.chat_layout.setSpacing(10)
        self.chat_layout.addStretch()
        self.chat_scroll.setWidget(self.chat_container)
        chat_card_layout.addWidget(self.chat_scroll, 1)

        input_row = QHBoxLayout()
        input_row.setContentsMargins(18, 16, 18, 16)
        input_row.setSpacing(10)
        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("Type a short message...")
        self.input_edit.setStyleSheet(
            "QLineEdit { background-color: #eeeeee; color: #1a1c1c; border: none; border-radius: 20px; "
            "padding: 11px 16px; font-size: 14px; }"
            "QLineEdit:focus { border: 1px solid #0071e3; }"
        )
        self.input_edit.returnPressed.connect(self.send_text_message)
        self.mic_button = QPushButton("🎤 Wake-word Listening: OFF")
        self.mic_button.clicked.connect(self.toggle_wake_word_listening)
        self.mic_button.setStyleSheet(
            "QPushButton { background-color: #eeeeee; color: #1a1c1c; border: none; border-radius: 20px; "
            "padding: 10px 16px; font-weight: 700; font-size: 13px; }"
            "QPushButton:hover { background-color: #e1e1e1; }"
            "QPushButton:pressed { background-color: #d4d4d4; }"
        )
        self.send_button = QPushButton("Send")
        self.send_button.setStyleSheet(
            "QPushButton { background-color: #0059b5; color: white; border: none; border-radius: 20px; "
            "padding: 10px 20px; font-weight: 700; font-size: 13px; }"
            "QPushButton:hover { background-color: #0071e3; }"
            "QPushButton:pressed { background-color: #00458c; }"
        )
        self.send_button.clicked.connect(self.send_text_message)
        # Push-to-talk button (press & hold to speak)
        self.ptt_button = QPushButton("🎙 Hold to Talk")
        self.ptt_button.setStyleSheet(
            "QPushButton { background-color: #10b981; color: white; border: none; border-radius: 8px; "
            "padding: 8px 12px; font-weight: 600; font-size: 13px; }"
            "QPushButton:hover { background-color: #059669; }"
            "QPushButton:pressed { background-color: #047857; }"
        )
        # Press & hold behaviour: start in push-to-talk mode (no VAD), stop on release
        self.ptt_button.pressed.connect(lambda: self.start_recording(push_to_talk=True))
        self.ptt_button.released.connect(self.stop_recording)

        input_row.addWidget(self.input_edit, 1)
        input_row.addWidget(self.mic_button)
        input_row.addWidget(self.ptt_button)
        input_row.addWidget(self.send_button)
        chat_card_layout.addLayout(input_row)
        right_panel.addWidget(chat_card, 1)

        self.add_message(
            "Assistant ready. I will keep replies short and adjust tone to the detected emotion.",
            is_user=False,
        )

        self.vision_thread = QThread(self)
        self.vision_worker = VisionWorker(args)
        self.vision_worker.moveToThread(self.vision_thread)
        self.vision_thread.started.connect(self.vision_worker.run)
        self.vision_worker.frame_ready.connect(self.update_frame)
        self.vision_worker.state_ready.connect(self.update_emotion_state)
        self.vision_worker.voice_dialogue_ready.connect(self.handle_voice_dialogue_result)
        self.vision_worker.voice_dialogue_error.connect(self.handle_voice_dialogue_error)
        self.vision_worker.error.connect(self.handle_worker_error)
        self.vision_worker.finished.connect(self.vision_thread.quit)
        self.model_combo.currentTextChanged.connect(self.push_chat_settings_to_worker)
        self.temperature_spin.valueChanged.connect(self.push_chat_settings_to_worker)
        self.vision_thread.start()
        self.push_chat_settings_to_worker()

        # Initialize wake-word listener.
        self.wake_word_listener = WakeWordListener(
            config=WakeWordConfig(
                keywords=["hey moss", "hey", "moss"],
                chunk_duration_seconds=1.0,
                confidence_threshold=0.6,
                whisper_model_size="tiny",
            ),
            on_detected=self.on_wake_word_detected,
        )

        # Start wake-word listening automatically on app launch.
        try:
            self.wake_word_listener.start()
            self.wake_word_listening = True
            self.mic_button.setVisible(False)
            self.mic_button.setEnabled(False)
            self.status_label.setText("Status: listening for 'hey moss")
        except Exception as exc:
            print(f"Failed to start wake-word listener automatically: {exc}")

        # Initialize continued conversation listener (post-reply).
        self.continued_listener = ContinuedConversationListener(
            config=WakeWordConfig(
                keywords=["hey moss"],  # Not needed for continued mode, but kept for consistency.
                chunk_duration_seconds=1.0,
                confidence_threshold=0.6,
                whisper_model_size="tiny",
            ),
            on_voice_detected=self.on_continued_voice_detected,
            on_timeout=self.on_continued_timeout,
            timeout_seconds=10.0,
        )

    def add_message(self, text: str, is_user: bool) -> None:
        bubble = ChatBubble(text, is_user=is_user)
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, bubble)
        scrollbar = self.chat_scroll.verticalScrollBar()
        if scrollbar is not None:
            scrollbar.setValue(scrollbar.maximum())

    def update_frame(self, image: QImage) -> None:
        pixmap = QPixmap.fromImage(image).scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.video_label.setPixmap(pixmap)

    def update_emotion_state(self, state: dict) -> None:
        self.current_driver_state = sanitize_driver_state(state) or {}
        self.current_emotion = state["emotion"]
        self.current_risk = state["risk"]
        color = emotion_color_hex(self.current_emotion)
        driver_detected = bool(state.get("driver_detected", False))
        trigger_reason = str(state.get("trigger_reason", "")).strip()
        focus_level = int(state.get("focus_level", 0))
        closed_eye_duration = float(state.get("closed_eye_duration", 0.0))
        eye_warmup_active = bool(state.get("eye_warmup_active", False))
        eye_warmup_remaining = float(state.get("eye_warmup_remaining", 0.0))
        self.driver_detected_label.setText("Yes" if driver_detected else "No")
        self.driver_detected_label.setStyleSheet(
            "color: #00a86b; font-size: 17px; font-weight: 700; border: none; background: transparent;"
            if driver_detected
            else "color: #ff9500; font-size: 17px; font-weight: 700; border: none; background: transparent;"
        )
        self.emotion_label.setText(
            f"{normalize_label(self.current_emotion)} ({state['emotion_confidence']:.2f})"
        )
        self.emotion_label.setStyleSheet(
            f"color: {color}; font-size: 17px; font-weight: 700; border: none; background: transparent;"
        )
        if eye_warmup_active:
            self.eye_label.setText(f"warming up ({eye_warmup_remaining:.1f}s)")
        else:
            self.eye_label.setText(
                f"{normalize_label(state['eye_label'])} ({state['eye_confidence']:.2f})"
            )
        self.eye_label.setStyleSheet(
            "color: #ff9500; font-size: 17px; font-weight: 700; border: none; background: transparent;"
            if eye_warmup_active
            else "color: #0059b5; font-size: 17px; font-weight: 700; border: none; background: transparent;"
        )
        self.risk_label.setText(self.current_risk)
        self.risk_label.setStyleSheet(
            f"color: {risk_color_hex(self.current_risk)}; font-size: 17px; font-weight: 700; border: none; background: transparent;"
        )
        self.focus_label.setText(
            "Please stay focused"
            if state["focus_alert"]
            else ("Stay attentive" if focus_level == 1 else "OK")
        )
        self.focus_label.setStyleSheet(
            "color: #e53935; font-size: 17px; font-weight: 700; border: none; background: transparent;"
            if state["focus_alert"]
            else (
                "color: #ff9500; font-size: 17px; font-weight: 700; border: none; background: transparent;"
                if focus_level == 1
                else "color: #00a86b; font-size: 17px; font-weight: 700; border: none; background: transparent;"
            )
        )
        if trigger_reason:
            self.reason_label.setText(
                f"{trigger_reason} ({closed_eye_duration:.1f}s)"
            )
        else:
            self.reason_label.setText("none")
        self.driver_label.setText(f"Driver heuristic: {state['driver_side']}")
        self.model_label.setText(f"Selected model: {self.model_combo.currentText()}")
        fallback_suffix = " (fallback)" if self.last_fallback_used else ""
        self.last_model_label.setText(f"Last reply model: {self.last_used_model}{fallback_suffix}")
        self.model_path_label.setText(f"Emotion model: {state['emotion_model_path']}")
        self.eye_model_path_label.setText(f"Eye model: {state['eye_model_path']}")
        self.context_label.setText(build_context_summary(state))

    def handle_worker_error(self, message: str) -> None:
        self.status_label.setText(f"Status: error - {message}")

    def handle_model_selection_change(self, model_name: str) -> None:
        self.last_selected_model = model_name
        self.model_label.setText(f"Selected model: {model_name}")
        message = f"Selected chat model: {model_name}"
        self.status_label.setText(f"Status: {message}")
        print(message)

    def push_chat_settings_to_worker(self, *args: object) -> None:
        self.vision_worker.update_chat_settings(
            self.model_combo.currentText(),
            float(self.temperature_spin.value()),
        )

    def send_text_message(self) -> None:
        text = self.input_edit.text().strip()
        if not text or self.chatbot is None or self.chat_worker is not None:
            return

        self.input_edit.clear()
        self.add_message(text, is_user=True)
        history_snapshot = list(self.conversation_history)
        self.conversation_history.append({"role": "user", "content": text})
        self.start_chat_worker(
            user_message=text,
            history_snapshot=history_snapshot,
            auto_trigger=False,
        )

    def start_chat_worker(
        self,
        user_message: str | None,
        history_snapshot: list[dict[str, str]],
        auto_trigger: bool,
    ) -> None:
        assert self.chatbot is not None
        self.send_button.setEnabled(False)
        self.mic_button.setEnabled(False)
        selected_model = self.model_combo.currentText()
        self.status_label.setText(f"Status: contacting OpenRouter with {selected_model}...")
        print(f"Sending chat request via OpenRouter using model: {selected_model}")
        self.chat_worker = ChatWorker(
            chatbot=self.chatbot,
            emotion=self.current_emotion,
            user_message=user_message,
            conversation_history=history_snapshot,
            model=selected_model,
            temperature=self.temperature_spin.value(),
            auto_trigger=auto_trigger,
            driver_state=self.current_driver_state,
        )
        self.chat_worker.result_ready.connect(self.handle_chat_response)
        self.chat_worker.error.connect(self.handle_chat_error)
        self.chat_worker.finished.connect(self.clear_chat_worker)
        self.chat_worker.start()

    def handle_chat_response(self, payload: dict) -> None:
        self.add_message(payload["text"], is_user=False)
        self.conversation_history.append({"role": "assistant", "content": payload["text"]})
        self.speak_reply_async(payload["text"], payload.get("emotion", self.current_emotion))
        self.last_used_model = str(payload.get("model", self.model_combo.currentText()))
        self.last_fallback_used = bool(payload.get("fallback_used", False))
        self.last_model_label.setText(
            f"Last reply model: {self.last_used_model}"
            + (" (fallback)" if self.last_fallback_used else "")
        )
        print(
            f"OpenRouter reply received from {payload['model']} "
            f"in {payload['latency_ms']:.0f} ms"
        )
        self.status_label.setText(
            f"Status: OpenRouter reply in {payload['latency_ms']:.0f} ms via {payload['model']}"
            + (" (fallback)" if self.last_fallback_used else "")
        )

    def handle_chat_error(self, message: str) -> None:
        self.status_label.setText(f"Status: chatbot error - {message}")
        print(f"Chat error: {message}", flush=True)

    def handle_voice_dialogue_result(self, payload: dict[str, Any]) -> None:
        user_input = str(payload.get("user_input", "")).strip()
        bot_reply = str(payload.get("bot_reply", "")).strip()
        if user_input:
            self.add_message(user_input, is_user=True)
            self.conversation_history.append({"role": "user", "content": user_input})
        if bot_reply:
            self.add_message(bot_reply, is_user=False)
            self.conversation_history.append({"role": "assistant", "content": bot_reply})
        model = payload.get("model", self.model_combo.currentText())
        self.last_used_model = str(model)
        self.last_fallback_used = bool(payload.get("fallback_used", False))
        self.last_model_label.setText(
            f"Last reply model: {self.last_used_model}"
            + (" (fallback)" if self.last_fallback_used else "")
        )
        latency_ms = payload.get("latency_ms")
        if isinstance(latency_ms, (int, float)):
            self.status_label.setText(
                f"Status: voice dialogue reply in {float(latency_ms):.0f} ms via {model}"
                + (" (fallback)" if self.last_fallback_used else "")
            )
        else:
            self.status_label.setText(f"Status: voice dialogue completed via {model}")

    def handle_voice_dialogue_error(self, message: str) -> None:
        if message.strip() == "No speech detected.":
            self.status_label.setText("Status: no speech detected")
            print("Voice dialogue: no speech detected", flush=True)
            return
        self.status_label.setText(f"Status: voice dialogue error - {message}")
        print(f"Voice dialogue error: {message}", flush=True)

    def clear_chat_worker(self) -> None:
        self.send_button.setEnabled(True)
        # mic_button is hidden; no state toggle necessary.
        self.chat_worker = None

    def speak_reply_async(self, text: str, emotion: str | None = None) -> None:
        if not text.strip():
            # 没有 TTS 也要启动 continued listener
            self._on_tts_finished()
            return

        def worker() -> None:
            try:
                tts = TextToSpeech(rate=150, volume=1.0)
                tts.speak(text, emotion=emotion)
            except Exception as exc:
                print(f"TTS error: {exc}")
            finally:
                # 回到主线程再操作 Qt 对象
                from PyQt5.QtCore import QMetaObject, Qt as QtNS
                QMetaObject.invokeMethod(
                    self, "_on_tts_finished",
                    QtNS.ConnectionType.QueuedConnection,
                )

        threading.Thread(target=worker, daemon=True).start()

    @pyqtSlot()
    def _on_tts_finished(self) -> None:
        """TTS 播完后在主线程调用，此时才启动 continued listener。"""
        # 停 wake-word，避免两个 InputStream 同时开着
        if self.wake_word_listener is not None and self.wake_word_listening:
            self.wake_word_listener.stop()
            self.wake_word_listening = False

        self._in_continued_mode = True
        if self.continued_listener is not None:
            self.continued_listener.start()
            self.status_label.setText("Status: listening for follow-up (10 s)...")

    def start_recording(self, push_to_talk: bool = False) -> None:
        """Start recording.

        If `push_to_talk` is True, disable VAD and record until user releases button.
        Otherwise use VAD with configured silence threshold.
        """
        if self.speech_worker is not None:
            return
        if push_to_talk:
            self.status_label.setText("Status: recording (push-to-talk)...")
            self.mic_button.setText("Recording...")
            # Push-to-talk: disable VAD, stop on explicit release.
            self.speech_worker = SpeechWorker(
                self.args.whisper_model_size,
                duration_seconds=self.args.max_recording_duration,
                vad_enabled=False,
            )
        else:
            self.status_label.setText("Status: recording audio (auto-stop on silence)...")
            self.mic_button.setText("Recording...")
            # Use configured max duration and silence threshold with VAD.
            self.speech_worker = SpeechWorker(
                self.args.whisper_model_size,
                duration_seconds=self.args.max_recording_duration,
                vad_enabled=True,
                min_duration_seconds=0.5,
                silence_duration_seconds=self.args.silence_threshold,
            )

        self.speech_worker.transcription_ready.connect(self.handle_transcription)
        self.speech_worker.error.connect(self.handle_speech_error)
        self.speech_worker.finished.connect(self.clear_speech_worker)
        self.speech_worker.start()

    def stop_recording(self) -> None:
        if self.speech_worker is not None:
            self.speech_worker.stop_recording()

    def handle_transcription(self, text: str) -> None:
        self.input_edit.setText(text)
        if text.strip():
            self.status_label.setText("Status: transcription ready")
            self.send_text_message()
        else:
            # 空转录：如果还在 continued 窗口，重新等；否则恢复 wake-word
            if self._in_continued_mode:
                if self.continued_listener is not None:
                    self.continued_listener.start()
                    self.status_label.setText("Status: listening for follow-up (10 s)...")
            else:
                if self.wake_word_listener is not None and not self.wake_word_listening:
                    self.wake_word_listener.start()
                    self.wake_word_listening = True
                self.status_label.setText("Status: listening for 'hey moss'")

    def handle_speech_error(self, message: str) -> None:
        self.status_label.setText(f"Status: speech error - {message}")

    def clear_speech_worker(self) -> None:
        self.speech_worker = None
        # 如果还在 continued 窗口内（由 voice 触发了录音），
        # continued listener 已经自己 stop 了，不需要重启 wake-word
        if self._in_continued_mode:
            # continued listener 触发录音后自己把 _running 置 False 了，
            # 但线程可能还没退出；等录音结束后再决定下一步
            # → 不做任何事，等 on_continued_timeout 或下一次 handle_chat_response
            pass
        else:
            # 普通录音（wake-word / PTT）结束，恢复 wake-word
            if self.wake_word_listener is not None and not self.wake_word_listening:
                self.wake_word_listener.start()
                self.wake_word_listening = True
            self.status_label.setText("Status: listening for 'hey moss'")
    
    def toggle_wake_word_listening(self) -> None:
        """Toggle wake-word listening on/off."""
        if self.wake_word_listener is None:
            return
        # Wake-word listening runs in background by default. This toggle is a no-op.
        self.status_label.setText("Status: wake-word listening is always enabled in background")

    def on_wake_word_detected(self) -> None:
        """Callback when wake-word is detected; trigger 5-second recording."""
        if self.speech_worker is not None:
            return
        self.status_label.setText("Status: wake-word detected! Recording for 5 seconds...")
        print("[Wake-word] Triggered speech recording.")
        self.start_recording()

    def on_continued_voice_detected(self) -> None:
        """Continued listener 检测到能量，直接触发录音。"""
        if self.speech_worker is not None:
            return
        # continued listener 内部已把 _running 置 False，线程会自己退出
        self.status_label.setText("Status: follow-up voice detected, recording...")
        self.start_recording(push_to_talk=False)
        
    def on_continued_timeout(self) -> None:
        """10 s 无声，回到 idle wake-word 状态。"""
        self._in_continued_mode = False
        if self.wake_word_listener is not None:
            self.wake_word_listener.start()
            self.wake_word_listening = True
        self.status_label.setText("Status: listening for 'hey moss'")

    def closeEvent(self, a0: QCloseEvent | None) -> None:
        if self.wake_word_listener is not None and self.wake_word_listening:
            self.wake_word_listener.stop()
        if self.continued_listener is not None:
            self.continued_listener.stop()
        if self.speech_worker is not None:
            self.speech_worker.stop_recording()
            self.speech_worker.wait(2000)
        if self.chat_worker is not None:
            self.chat_worker.wait(2000)
        self.vision_worker.stop()
        self.vision_thread.quit()
        self.vision_thread.wait(3000)
        if a0 is not None:
            super().closeEvent(a0)


def main() -> None:
    args = parse_args()
    app = QApplication(sys.argv)
    window = DriverAssistantWindow(args)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
