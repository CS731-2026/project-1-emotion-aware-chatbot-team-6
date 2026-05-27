from __future__ import annotations

import argparse
import csv
import os
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import cv2


def configure_frozen_dll_search_paths() -> None:
    if not getattr(sys, "frozen", False):
        return

    meipass = getattr(sys, "_MEIPASS", None)
    if not meipass:
        return

    internal_root = Path(meipass)
    dll_dirs = [
        internal_root,
        internal_root / "torch" / "lib",
        internal_root / "ctranslate2",
        internal_root / "cv2",
    ]

    existing_dirs: list[str] = []
    for dll_dir in dll_dirs:
        if dll_dir.exists():
            existing_dirs.append(str(dll_dir))
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(str(dll_dir))

    if existing_dirs:
        os.environ["PATH"] = os.pathsep.join(existing_dirs + [os.environ.get("PATH", "")])


configure_frozen_dll_search_paths()

import torch
from PyQt5.QtCore import QObject, QMetaObject, Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QColor, QCloseEvent, QCursor, QGuiApplication, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
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
from drivesense.backend.focus_monitor import (
    EMOTION_FULL_DIALOGUE,
    EMOTION_TRIGGER_THRESHOLDS,
    EMOTION_TTS_ONLY,
    FocusMonitor,
    FocusMonitorConfig,
)
from drivesense.backend.vision import (
    EmotionMajorityWindow,
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
from drivesense.backend.speech import (
    TextToSpeech,
    TTS_PRIORITY_CHAT,
    VoiceIOGate,
    WhisperTranscriber,
    detect_text_language,
    is_supported_zh_en_text,
    record_microphone_audio,
)
from drivesense.backend.tts_queue import TTSQueue
from drivesense.backend.wake_word import WakeWordListener, WakeWordConfig, ContinuedConversationListener


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "runs_timm"
DEFAULT_WEIGHTS_ROOT = PROJECT_ROOT / "weights"
DEFAULT_LLM_BENCHMARK_DIR = PROJECT_ROOT / "benchmark_results" / "llm_benchmark"


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
EMOTION_HISTORY_LABELS = [
    "anger",
    "fear",
    "sad",
    "happy",
    "surprise",
    "disgust",
    "neutral",
]


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
        default=10.0,
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


def compute_attention_score(driver_state: dict[str, Any]) -> int:
    if not driver_state.get("driver_detected", False):
        return 25

    if driver_state.get("eye_warmup_active", False):
        return 65

    score = 100
    emotion = str(driver_state.get("emotion", "neutral")).lower()
    risk = str(driver_state.get("risk", "OK")).upper()
    eye_label = str(driver_state.get("eye_label", "open_eye")).lower()
    focus_level = int(driver_state.get("focus_level", 0))

    if eye_label == "closed_eye":
        score -= 45
    if focus_level >= 2 or driver_state.get("focus_alert", False):
        score -= 35
    elif focus_level == 1:
        score -= 18

    if risk == "HIGH":
        score -= 15
    elif risk == "MED":
        score -= 8
    elif risk == "LOW":
        score -= 4

    if emotion in {"anger", "fear"}:
        score -= 10
    elif emotion == "sad":
        score -= 6

    return max(0, min(100, score))


class AttentionHistoryChart(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._points: list[tuple[float, float]] = []
        self.setMinimumHeight(260)

    def set_points(self, points: list[tuple[float, float]]) -> None:
        self._points = points[-180:]
        self.update()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#ffffff"))

        rect = self.rect().adjusted(18, 16, -18, -24)
        if rect.width() <= 0 or rect.height() <= 0:
            return

        grid_pen = QPen(QColor("#e8e8e8"))
        grid_pen.setWidth(1)
        painter.setPen(grid_pen)
        for idx in range(5):
            y = rect.top() + int(rect.height() * idx / 4)
            painter.drawLine(rect.left(), y, rect.right(), y)

        axis_pen = QPen(QColor("#b8b8b8"))
        axis_pen.setWidth(1)
        painter.setPen(axis_pen)
        painter.drawRect(rect)

        label_pen = QPen(QColor("#6b7280"))
        painter.setPen(label_pen)
        for value in (100, 75, 50, 25, 0):
            y = rect.top() + int((100 - value) / 100 * rect.height())
            painter.drawText(rect.left() + 6, y - 4, str(value))

        if len(self._points) < 2:
            painter.setPen(QPen(QColor("#9aa1ad")))
            painter.drawText(rect.adjusted(0, 0, 0, -8), Qt.AlignmentFlag.AlignCenter, "Waiting for attention history...")
            return

        first_ts = self._points[0][0]
        last_ts = self._points[-1][0]
        span = max(last_ts - first_ts, 1e-6)

        line_pen = QPen(QColor("#0059b5"))
        line_pen.setWidth(3)
        painter.setPen(line_pen)

        prev_x = 0
        prev_y = 0
        for idx, (ts_value, score) in enumerate(self._points):
            x = rect.left() + int(((ts_value - first_ts) / span) * rect.width())
            y = rect.bottom() - int((score / 100.0) * rect.height())
            if idx > 0:
                painter.drawLine(prev_x, prev_y, x, y)
            prev_x, prev_y = x, y

        point_pen = QPen(QColor("#0071e3"))
        point_pen.setWidth(6)
        painter.setPen(point_pen)
        painter.drawPoint(prev_x, prev_y)

        painter.setPen(QPen(QColor("#1f2937")))
        painter.drawText(rect.left(), rect.bottom() + 18, "Oldest")
        painter.drawText(rect.right() - 42, rect.bottom() + 18, "Now")


class EmotionConfidenceHistoryChart(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._points: list[tuple[float, dict[str, float]]] = []
        self.setMinimumHeight(300)

    def set_points(self, points: list[tuple[float, dict[str, float]]]) -> None:
        self._points = points[-180:]
        self.update()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#ffffff"))

        rect = self.rect().adjusted(44, 18, -22, -52)
        if rect.width() <= 0 or rect.height() <= 0:
            return

        grid_pen = QPen(QColor("#e8e8e8"))
        grid_pen.setWidth(1)
        painter.setPen(grid_pen)
        for idx in range(5):
            y = rect.top() + int(rect.height() * idx / 4)
            painter.drawLine(rect.left(), y, rect.right(), y)

        axis_pen = QPen(QColor("#b8b8b8"))
        axis_pen.setWidth(1)
        painter.setPen(axis_pen)
        painter.drawRect(rect)

        painter.setPen(QPen(QColor("#6b7280")))
        for value in (1.0, 0.75, 0.5, 0.25, 0.0):
            y = rect.top() + int((1.0 - value) * rect.height())
            painter.drawText(rect.left() - 38, y + 4, f"{value:.2g}")

        if len(self._points) < 2:
            painter.setPen(QPen(QColor("#9aa1ad")))
            painter.drawText(
                rect.adjusted(0, 0, 0, -8),
                Qt.AlignmentFlag.AlignCenter,
                "Waiting for emotion confidence history...",
            )
            self._draw_legend(painter, rect)
            return

        first_ts = self._points[0][0]
        last_ts = self._points[-1][0]
        span = max(last_ts - first_ts, 1e-6)

        for emotion in EMOTION_HISTORY_LABELS:
            line_pen = QPen(QColor(emotion_color_hex(emotion)))
            line_pen.setWidth(2 if emotion != "neutral" else 3)
            painter.setPen(line_pen)

            prev_x = 0
            prev_y = 0
            for idx, (ts_value, confidences) in enumerate(self._points):
                confidence = max(0.0, min(1.0, float(confidences.get(emotion, 0.0))))
                x = rect.left() + int(((ts_value - first_ts) / span) * rect.width())
                y = rect.bottom() - int(confidence * rect.height())
                if idx > 0:
                    painter.drawLine(prev_x, prev_y, x, y)
                prev_x, prev_y = x, y

        painter.setPen(QPen(QColor("#1f2937")))
        painter.drawText(rect.left(), rect.bottom() + 18, "Oldest")
        painter.drawText(rect.right() - 42, rect.bottom() + 18, "Now")
        self._draw_legend(painter, rect)

    def _draw_legend(self, painter: QPainter, rect) -> None:
        x = rect.left()
        y = rect.bottom() + 32
        item_gap = 86
        for index, emotion in enumerate(EMOTION_HISTORY_LABELS):
            item_x = x + (index % 4) * item_gap
            item_y = y + (index // 4) * 18
            painter.setPen(QPen(QColor(emotion_color_hex(emotion)), 3))
            painter.drawLine(item_x, item_y - 5, item_x + 18, item_y - 5)
            painter.setPen(QPen(QColor("#374151")))
            painter.drawText(item_x + 24, item_y, normalize_label(emotion))


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
        bubble.setMaximumWidth(360)

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
    voice_status_ready = pyqtSignal(str)
    error = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.args = args
        self._running = True
        self.current_chat_model = args.default_llm_model
        self.current_temperature = float(args.default_temperature)
        self.voice_dialogue_enabled = bool(args.enable_voice_dialogue)
        self.eye_monitoring_enabled = True

    def update_chat_settings(self, model: str, temperature: float) -> None:
        self.current_chat_model = model
        self.current_temperature = float(temperature)

    def update_voice_dialogue_enabled(self, enabled: bool) -> None:
        self.voice_dialogue_enabled = enabled

    def update_eye_monitoring_enabled(self, enabled: bool) -> None:
        self.eye_monitoring_enabled = enabled

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
                on_voice_status=self.voice_status_ready.emit,
            )
            emotion_window = EmotionMajorityWindow()

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
                "eye_monitoring_enabled": True,
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
                        if isinstance(primary_face, dict):
                            bbox = cast(tuple[int, int, int, int], primary_face["bbox"])
                            px1, _, px2, _ = bbox
                        elif isinstance(primary_face, int):
                            face_dict = faces[primary_face]
                            px1, _, px2, _ = face_dict["bbox"]
                            primary_face = face_dict
                        else:
                            px1, _, px2, _ = primary_face  # type: ignore[misc]
                        previous_driver_center_x = ((px1 + px2) / 2.0) / max(frame_w, 1)
                        if isinstance(primary_face, dict):
                            smoothed_emotion, smoothed_confidence = emotion_window.update(
                                str(primary_face.get("emotion", "neutral")),
                                cast(float, primary_face.get("emotion_confidence", 0.0)),
                            )
                            primary_face["emotion"] = smoothed_emotion
                            primary_face["emotion_confidence"] = smoothed_confidence
                    else:
                        emotion_window.reset()
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
                if primary_face is not None and isinstance(primary_face, dict):
                    current_emotion = str(primary_face.get("emotion", "neutral"))
                    emotion_confidence = cast(float, primary_face.get("emotion_confidence", 0.0))
                    emotion_topk = cast(list[tuple[str, float]], primary_face.get("emotion_topk", []))
                    secondary_emotion = emotion_topk[1][0] if len(emotion_topk) > 1 else ""
                    secondary_emotion_confidence = float(emotion_topk[1][1]) if len(emotion_topk) > 1 else 0.0
                    current_eye_label = str(primary_face.get("eye_label", "open_eye"))
                    eye_confidence = cast(float, primary_face.get("eye_confidence", 0.0))
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
                            f"{format_topk_prediction(cast(list[tuple[str, float]], primary_face.get('emotion_topk', [])))}",
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
                    "eye_monitoring_enabled": self.eye_monitoring_enabled,
                    "risk": risk,
                    "focus_alert": False,
                    "driver_side": self.args.driver_side,
                }
                focus_monitor.set_voice_dialogue_enabled(self.voice_dialogue_enabled)
                focus_monitor.set_eye_monitoring_enabled(self.eye_monitoring_enabled)
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
                    emotion_warn = last_state.get("emotion_warning_active", False)
                    focus_lvl = last_state.get("focus_level", 0)
                    if emotion_warn and focus_lvl >= 2:
                        streak_label = str(last_state.get("emotion_streak_label", "")).capitalize()
                        streak_dur = float(last_state.get("emotion_streak_duration", 0.0))
                        alert_text = f"{streak_label} detected - Please calm down"
                        alert_bgr = {
                            "anger": (0, 0, 255),
                            "fear": (0, 165, 255),
                            "sad": (255, 90, 0),
                            "disgust": (180, 0, 180),
                            "surprise": (0, 255, 255),
                        }.get(streak_label.lower(), (0, 0, 255))
                    elif focus_lvl >= 2:
                        alert_text = "Please stay focused"
                        alert_bgr = (0, 0, 255)
                    else:
                        alert_text = "Stay attentive"
                        alert_bgr = (0, 165, 255)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    scale = 1.0 if focus_lvl >= 2 else 0.8
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
                        alert_bgr,
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
        device = "cpu"
        key = (model_size, device)
        with cls._cache_lock:
            if key not in cls._transcriber_cache:
                cls._transcriber_cache[key] = WhisperTranscriber(
                    model_size=model_size,
                    device=device,
                    compute_type="int8",
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
            if VoiceIOGate.is_tts_active():
                self.transcription_ready.emit("")
                return

            transcriber = self.get_transcriber(self.model_size)
            result = transcriber.transcribe_audio(audio, sample_rate=16000)
            text = result.text.strip()
            if text and not is_supported_zh_en_text(text):
                raise ValueError("Only Chinese and English voice input is supported.")
            self.transcription_ready.emit(text)
        except Exception as exc:
            self.error.emit(str(exc))


class DriverAssistantWindow(QMainWindow):
    wake_word_detected_signal = pyqtSignal()
    continued_voice_detected_signal = pyqtSignal()
    continued_timeout_signal = pyqtSignal()

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.args = args
        self.setWindowTitle("DriveSense")
        self.apply_initial_window_geometry()

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
        self.runtime_logs: list[str] = []
        self.attention_history: list[tuple[float, float]] = []
        self.emotion_history: list[tuple[float, dict[str, float]]] = []
        self._history_started_at = time.perf_counter()
        self._last_history_point_at = 0.0
        self._last_logged_snapshot: tuple[bool, str, str, int] | None = None
        self.nav_buttons: dict[str, QPushButton] = {}
        self.wake_word_listening = False
        self.voice_input_enabled = True
        self.eye_monitoring_enabled = True
        self.tts_muted = TTSQueue.instance().is_muted()
        self.wake_word_listener: WakeWordListener | None = None
        self.continued_listener: ContinuedConversationListener | None = None
        self._in_continued_mode = False
        self.voice_input_state = "starting..."
        self.voice_output_state = "idle"
        self.voice_dialogue_state = (
            "enabled" if bool(self.args.enable_voice_dialogue) else "disabled"
        )
        self.reply_tts = TextToSpeech(rate=150, volume=1.0)
        self.llm_benchmark_rows = self.load_llm_benchmark_rows()

        try:
            self.chatbot = DriverAssistantChatbot(app_title="DriveSense GUI")
            self.append_log("system", "OpenRouter chatbot initialized")
        except Exception as exc:
            self.chatbot = None
            QMessageBox.warning(self, "OpenRouter", str(exc))
            self.append_log("error", f"OpenRouter init failed: {exc}")

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
        for item in ["Dashboard", "Logs", "History", "Models"]:
            nav_button = QPushButton(item)
            nav_button.setCursor(Qt.CursorShape.PointingHandCursor)
            nav_button.setStyleSheet(self.nav_button_style(active=(item == "Dashboard")))
            nav_button.clicked.connect(
                lambda checked=False, name=item: self.switch_page(name)
            )
            self.nav_buttons[item] = nav_button
            nav_layout.addWidget(nav_button)
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

        self.title_label = QLabel("Driver Monitoring")
        self.title_label.setStyleSheet("font-size: 48px; font-weight: 800; color: #1a1c1c;")
        content_layout.addWidget(self.title_label)

        self.page_stack = QStackedWidget()
        self.page_stack.setStyleSheet("background: transparent; border: none;")
        content_layout.addWidget(self.page_stack, 1)

        dashboard_page = QWidget()
        dashboard_page.setStyleSheet("background: transparent;")
        main_layout = QHBoxLayout(dashboard_page)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(36)
        self.page_stack.addWidget(dashboard_page)

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
        telemetry_title = QLabel("Driver State")
        telemetry_title.setStyleSheet("font-size: 26px; font-weight: 800; color: #1a1c1c; border: none;")
        info_layout.addWidget(telemetry_title)

        def add_metric_row(label_text: str, value_label: QLabel) -> None:
            row = QFrame()
            row.setStyleSheet("QFrame { border: none; border-bottom: 1px solid #dadada; background: transparent; }")
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 10, 0, 10)
            label = QLabel(label_text)
            label.setStyleSheet("color: #1a1c1c; font-size: 17px; border: none; background: transparent;")
            value_label.setAlignment(
                cast(Any, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            )
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
        self.emotion_streak_label = QLabel("none")
        self.emotion_timer_label = QLabel("not active")
        self.emotion_alerts_label = QLabel("0")
        add_metric_row("Driver detected", self.driver_detected_label)
        add_metric_row("Emotion", self.emotion_label)
        add_metric_row("Eyes", self.eye_label)
        add_metric_row("Risk", self.risk_label)
        add_metric_row("Focus", self.focus_label)
        add_metric_row("Reason", self.reason_label)
        add_metric_row("Emotion streak", self.emotion_streak_label)
        add_metric_row("Emotion timer", self.emotion_timer_label)
        add_metric_row("Emotion alerts", self.emotion_alerts_label)

        meta_block = QVBoxLayout()
        meta_block.setContentsMargins(0, 14, 0, 0)
        meta_block.setSpacing(4)
        self.voice_input_label = QLabel("Voice input: starting...")
        self.voice_output_label = QLabel("Voice output: idle")
        self.voice_dialogue_label = QLabel(f"Auto dialogue: {self.voice_dialogue_state}")
        self.wake_word_status_label = QLabel("Wake word: starting...")
        self.tts_status_label = QLabel("TTS: unmuted")
        self.status_label = QLabel("Status: ready")
        for label in [
            self.voice_input_label,
            self.voice_output_label,
            self.voice_dialogue_label,
            self.wake_word_status_label,
            self.tts_status_label,
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

        audio_controls_row = QHBoxLayout()
        audio_controls_row.setSpacing(10)
        self.mute_button = QPushButton("Mute TTS")
        self.mute_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.mute_button.clicked.connect(self.toggle_tts_mute)
        self.mute_button.setStyleSheet(self.secondary_control_button_style(active=False))
        self.stop_listening_button = QPushButton("Stop Listening")
        self.stop_listening_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.stop_listening_button.clicked.connect(self.toggle_voice_input)
        self.stop_listening_button.setStyleSheet(self.secondary_control_button_style(active=False))
        self.eye_monitor_button = QPushButton("Disable Eye Monitor")
        self.eye_monitor_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.eye_monitor_button.clicked.connect(self.toggle_eye_monitoring)
        self.eye_monitor_button.setStyleSheet(self.secondary_control_button_style(active=False))
        audio_controls_row.addWidget(self.mute_button)
        audio_controls_row.addWidget(self.stop_listening_button)
        audio_controls_row.addWidget(self.eye_monitor_button)
        settings_layout.addLayout(audio_controls_row)
        right_panel.addWidget(settings_card)
        right_panel.addWidget(self.build_emotion_alert_rules_card())

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

        self.page_stack.addWidget(self.build_logs_page())
        self.page_stack.addWidget(self.build_history_page())
        self.page_stack.addWidget(self.build_models_page())

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
        self.vision_worker.voice_status_ready.connect(self.handle_auto_voice_status)
        self.vision_worker.error.connect(self.handle_worker_error)
        self.vision_worker.finished.connect(self.vision_thread.quit)
        self.wake_word_detected_signal.connect(self._handle_wake_word_detected)
        self.continued_voice_detected_signal.connect(self._handle_continued_voice_detected)
        self.continued_timeout_signal.connect(self._handle_continued_timeout)
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
            self.mic_button.setText("Wake-word: ON")
            self.mic_button.setEnabled(True)
            self.set_voice_input_state("listening for 'hey moss'")
            self.status_label.setText("Status: listening for 'hey moss'")
            self.update_audio_control_buttons()
            self.append_log("voice", "Wake-word listener started for 'hey moss'")
        except Exception as exc:
            print(f"Failed to start wake-word listener automatically: {exc}")
            self.set_voice_input_state("wake-word listener failed")
            self.append_log("error", f"Wake-word listener failed to start: {exc}")

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
            timeout_seconds=5.0,
        )
        self.switch_page("Dashboard")
        self.append_log("system", "DriveSense GUI ready")

    def apply_initial_window_geometry(self) -> None:
        app = QGuiApplication.instance()
        if app is None:
            self.resize(1500, 900)
            return

        # QGuiApplication.instance() returns QCoreApplication in type hints,
        # but at runtime it's always QGuiApplication when one exists.
        # Use cast to satisfy static type checkers.
        gui_app = cast(QGuiApplication, app)
        screen = gui_app.screenAt(QCursor.pos()) or gui_app.primaryScreen()
        if screen is None:
            self.resize(1500, 900)
            return

        available = screen.availableGeometry()
        width = min(1500, max(960, available.width() - 96))
        height = min(900, max(700, available.height() - 96))
        width = min(width, max(640, available.width()))
        height = min(height, max(480, available.height()))
        x = available.x() + max(0, (available.width() - width) // 2)
        y = available.y() + max(0, (available.height() - height) // 2)
        x = min(max(x, available.left()), max(available.left(), available.right() - width + 1))
        y = min(max(y, available.top()), max(available.top(), available.bottom() - height + 1))
        self.resize(width, height)
        self.move(x, y)

    def nav_button_style(self, active: bool) -> str:
        return (
            "QPushButton { background: transparent; border: none; padding: 22px 0 17px 0; "
            f"color: {'#ffffff' if active else '#c1c6d6'}; font-size: 15px; font-weight: 700; "
            + (f"border-bottom: 2px solid {'#0071e3' if active else 'transparent'};" if active else "")
            + "}"
            "QPushButton:hover { color: #ffffff; }"
        )

    def secondary_control_button_style(self, active: bool) -> str:
        bg = "#fee2e2" if active else "#eeeeee"
        hover = "#fecaca" if active else "#e1e1e1"
        fg = "#991b1b" if active else "#1a1c1c"
        return (
            f"QPushButton {{ background-color: {bg}; color: {fg}; border: none; "
            "border-radius: 10px; padding: 9px 12px; font-weight: 700; font-size: 13px; }"
            f"QPushButton:hover {{ background-color: {hover}; }}"
            "QPushButton:disabled { color: #9ca3af; background-color: #f3f4f6; }"
        )

    def switch_page(self, page_name: str) -> None:
        page_index = {"Dashboard": 0, "Logs": 1, "History": 2, "Models": 3}.get(page_name, 0)
        page_meta = {
            "Dashboard": ("Driver Monitoring", "System Active"),
            "Logs": ("Runtime Logs", "Voice, detection, and chatbot events"),
            "History": ("Attention History", "Recent driver attention curve"),
            "Models": ("Model Comparison", "LLM benchmark summary and current selection"),
        }
        self.page_stack.setCurrentIndex(page_index)
        title, _subtitle = page_meta.get(page_name, page_meta["Dashboard"])
        self.title_label.setText(title)
        for name, button in self.nav_buttons.items():
            button.setStyleSheet(self.nav_button_style(active=(name == page_name)))

    def build_card_frame(self) -> QFrame:
        card = QFrame()
        card.setStyleSheet(
            "QFrame { background-color: #ffffff; border: 1px solid #dadada; border-radius: 18px; }"
        )
        return card

    def build_emotion_alert_rules_card(self) -> QFrame:
        card = self.build_card_frame()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(10)

        title = QLabel("Emotion Alert Rules")
        title.setStyleSheet("color: #1a1c1c; font-size: 18px; font-weight: 800; border: none; background: transparent;")
        layout.addWidget(title)

        subtitle = QLabel("Only continuous driver emotions trigger alerts.")
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: #4b5563; font-size: 13px; border: none; background: transparent;")
        layout.addWidget(subtitle)

        def add_rule(emotions: list[str], action: str) -> None:
            row = QFrame()
            row.setStyleSheet("QFrame { border: none; border-top: 1px solid #eeeeee; background: transparent; }")
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 9, 0, 0)
            names = " / ".join(emotions)
            thresholds = sorted({int(EMOTION_TRIGGER_THRESHOLDS[name]) for name in emotions})
            threshold_text = f"{thresholds[0]}s" if len(thresholds) == 1 else "/".join(f"{value}s" for value in thresholds)
            name_label = QLabel(names)
            name_label.setStyleSheet("color: #1a1c1c; font-size: 14px; font-weight: 700; border: none; background: transparent;")
            detail_label = QLabel(f"{threshold_text}, {action}")
            detail_label.setAlignment(cast(Any, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter))
            detail_label.setStyleSheet("color: #4b5563; font-size: 13px; border: none; background: transparent;")
            row_layout.addWidget(name_label)
            row_layout.addWidget(detail_label, 1)
            layout.addWidget(row)

        full_by_threshold: dict[int, list[str]] = {}
        for emotion in sorted(EMOTION_FULL_DIALOGUE):
            threshold = int(EMOTION_TRIGGER_THRESHOLDS[emotion])
            full_by_threshold.setdefault(threshold, []).append(emotion)
        for emotions in full_by_threshold.values():
            add_rule(emotions, "beep + TTS + voice dialogue")

        tts_by_threshold: dict[int, list[str]] = {}
        for emotion in sorted(EMOTION_TTS_ONLY):
            threshold = int(EMOTION_TRIGGER_THRESHOLDS[emotion])
            tts_by_threshold.setdefault(threshold, []).append(emotion)
        for emotions in tts_by_threshold.values():
            add_rule(emotions, "beep + short TTS")

        inactive = QLabel("happy / neutral: no emotion alert")
        inactive.setStyleSheet(
            "color: #00a86b; font-size: 13px; font-weight: 700; border: none; "
            "background: transparent; padding-top: 6px;"
        )
        layout.addWidget(inactive)
        return card

    def build_logs_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(24)

        card = self.build_card_frame()
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(24, 22, 24, 22)
        card_layout.setSpacing(14)

        title = QLabel("Logs")
        title.setStyleSheet("font-size: 28px; font-weight: 800; color: #1a1c1c; border: none;")
        subtitle = QLabel("Key runtime events are collected here in time order.")
        subtitle.setStyleSheet("font-size: 15px; color: #4b5563; border: none;")
        self.logs_summary_label = QLabel("No events yet.")
        self.logs_summary_label.setStyleSheet("font-size: 14px; color: #6b7280; border: none;")
        self.logs_list = QListWidget()
        self.logs_list.setStyleSheet(
            "QListWidget { background: #f9f9f9; border: 1px solid #e4e4e4; border-radius: 14px; "
            "padding: 8px; font-size: 14px; color: #1f2937; }"
            "QListWidget::item { padding: 10px 8px; border-bottom: 1px solid #ececec; }"
            "QListWidget::item:selected { background: #e8f0ff; color: #1a1c1c; }"
        )

        card_layout.addWidget(title)
        card_layout.addWidget(subtitle)
        card_layout.addWidget(self.logs_summary_label)
        card_layout.addWidget(self.logs_list, 1)
        for entry in reversed(self.runtime_logs):
            self.logs_list.addItem(QListWidgetItem(entry))
        if self.runtime_logs:
            self.logs_summary_label.setText(f"Captured {len(self.runtime_logs)} recent events.")
        layout.addWidget(card, 1)
        return page

    def build_history_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(24)

        card = self.build_card_frame()
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(24, 22, 24, 22)
        card_layout.setSpacing(14)

        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        title = QLabel("History")
        title.setStyleSheet("font-size: 28px; font-weight: 800; color: #1a1c1c; border: none;")
        self.export_history_button = QPushButton("Export Chart")
        self.export_history_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.export_history_button.setStyleSheet(
            "QPushButton { background-color: #0059b5; color: white; border: none; border-radius: 8px; "
            "padding: 9px 16px; font-weight: 700; font-size: 13px; }"
            "QPushButton:hover { background-color: #0071e3; }"
            "QPushButton:pressed { background-color: #00458c; }"
        )
        self.export_history_button.clicked.connect(self.export_history_chart)
        header_row.addWidget(title)
        header_row.addStretch()
        header_row.addWidget(self.export_history_button)
        subtitle = QLabel("Attention and emotion confidence use the filtered driver state shown in the UI.")
        subtitle.setStyleSheet("font-size: 15px; color: #4b5563; border: none;")
        self.history_summary_label = QLabel("Waiting for driver-state samples.")
        self.history_summary_label.setStyleSheet("font-size: 14px; color: #6b7280; border: none;")
        attention_title = QLabel("Attention Score")
        attention_title.setStyleSheet("font-size: 16px; font-weight: 800; color: #1f2937; border: none;")
        self.attention_chart = AttentionHistoryChart()
        emotion_title = QLabel("Emotion Confidence")
        emotion_title.setStyleSheet("font-size: 16px; font-weight: 800; color: #1f2937; border: none;")
        self.emotion_chart = EmotionConfidenceHistoryChart()
        self.history_card = card

        card_layout.addLayout(header_row)
        card_layout.addWidget(subtitle)
        card_layout.addWidget(self.history_summary_label)
        card_layout.addWidget(attention_title)
        card_layout.addWidget(self.attention_chart, 1)
        card_layout.addWidget(emotion_title)
        card_layout.addWidget(self.emotion_chart, 1)
        layout.addWidget(card, 1)
        return page

    def build_models_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(24)

        card = self.build_card_frame()
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(24, 22, 24, 22)
        card_layout.setSpacing(14)

        title = QLabel("LLM Comparison")
        title.setStyleSheet("font-size: 28px; font-weight: 800; color: #1a1c1c; border: none;")
        subtitle = QLabel("Results are loaded from benchmark CSV files under benchmark_results/llm_benchmark.")
        subtitle.setStyleSheet("font-size: 15px; color: #4b5563; border: none;")
        self.models_summary_label = QLabel(f"Current selected model: {self.args.default_llm_model}")
        self.models_summary_label.setStyleSheet("font-size: 14px; color: #1f2937; border: none;")
        self.models_table = QTableWidget(0, 4)
        self.models_table.setHorizontalHeaderLabels(
            ["Model", "Avg Latency (ms)", "Avg Manual Score", "Current Status"]
        )
        self.models_table.setStyleSheet(
            "QTableWidget { background: #f9f9f9; border: 1px solid #e4e4e4; border-radius: 14px; "
            "gridline-color: #ececec; font-size: 14px; color: #1f2937; }"
            "QHeaderView::section { background: #eeeeee; border: none; padding: 10px; font-weight: 700; }"
        )
        cast(QHeaderView, self.models_table.verticalHeader()).setVisible(False)
        cast(QHeaderView, self.models_table.horizontalHeader()).setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.models_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.models_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.refresh_models_table()

        card_layout.addWidget(title)
        card_layout.addWidget(subtitle)
        card_layout.addWidget(self.models_summary_label)
        card_layout.addWidget(self.models_table, 1)
        layout.addWidget(card, 1)
        return page

    def load_llm_benchmark_rows(self) -> list[dict[str, str]]:
        summary_path = DEFAULT_LLM_BENCHMARK_DIR / "model_summary.csv"
        scores_path = DEFAULT_LLM_BENCHMARK_DIR / "manual_scores_filled.csv"
        rows: dict[str, dict[str, str]] = {}

        if summary_path.exists():
            with summary_path.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    model = str(row.get("model", "")).strip()
                    if model:
                        rows[model] = {
                            "model": model,
                            "avg_latency_ms": str(row.get("avg_latency_ms", "")).strip(),
                            "average_score": "",
                        }

        if scores_path.exists():
            grouped_scores: dict[str, list[float]] = {}
            with scores_path.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    model = str(row.get("model", "")).strip()
                    score_text = str(row.get("average_score", "")).strip()
                    if not model or not score_text:
                        continue
                    try:
                        grouped_scores.setdefault(model, []).append(float(score_text))
                    except ValueError:
                        continue
            for model, scores in grouped_scores.items():
                record = rows.setdefault(
                    model,
                    {"model": model, "avg_latency_ms": "", "average_score": ""},
                )
                record["average_score"] = f"{sum(scores) / len(scores):.2f}"

        return [rows[key] for key in SUPPORTED_LLM_MODELS if key in rows] or [
            {"model": model, "avg_latency_ms": "", "average_score": ""}
            for model in SUPPORTED_LLM_MODELS
        ]

    def refresh_models_table(self) -> None:
        self.models_table.setRowCount(len(self.llm_benchmark_rows))
        selected_model = self.model_combo.currentText() if hasattr(self, "model_combo") else self.args.default_llm_model
        for row_index, row in enumerate(self.llm_benchmark_rows):
            model = row.get("model", "")
            status = []
            if model == selected_model:
                status.append("selected")
            if model == self.last_used_model:
                status.append("last used")
            display_values = [
                model,
                row.get("avg_latency_ms", "") or "-",
                row.get("average_score", "") or "-",
                ", ".join(status) or "available",
            ]
            for col_index, value in enumerate(display_values):
                item = QTableWidgetItem(value)
                if model == selected_model:
                    item.setBackground(QColor("#e8f0ff"))
                self.models_table.setItem(row_index, col_index, item)
        if hasattr(self, "models_summary_label"):
            self.models_summary_label.setText(
                f"Current selected model: {selected_model} | Last reply model: {self.last_used_model}"
                + (" (fallback)" if self.last_fallback_used else "")
            )

    def append_log(self, category: str, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        entry = f"[{timestamp}] {category.upper()}: {message}"
        self.runtime_logs.append(entry)
        self.runtime_logs = self.runtime_logs[-200:]
        if hasattr(self, "logs_list"):
            self.logs_list.insertItem(0, QListWidgetItem(entry))
            while self.logs_list.count() > 200:
                self.logs_list.takeItem(self.logs_list.count() - 1)
            self.logs_summary_label.setText(f"Captured {len(self.runtime_logs)} recent events.")

    def set_voice_input_state(self, text: str) -> None:
        self.voice_input_state = text
        if hasattr(self, "voice_input_label"):
            self.voice_input_label.setText(f"Voice input: {text}")

    def set_voice_output_state(self, text: str) -> None:
        self.voice_output_state = text
        if hasattr(self, "voice_output_label"):
            self.voice_output_label.setText(f"Voice output: {text}")

    def set_voice_dialogue_state(self, text: str) -> None:
        self.voice_dialogue_state = text
        if hasattr(self, "voice_dialogue_label"):
            self.voice_dialogue_label.setText(f"Auto dialogue: {text}")

    def refresh_voice_status_labels(self) -> None:
        if not hasattr(self, "tts_status_label"):
            return
        self.tts_status_label.setText(
            "TTS: muted" if self.tts_muted else "TTS: unmuted"
        )
        if not self.voice_input_enabled:
            wake_word_state = "off"
        elif self._in_continued_mode:
            wake_word_state = "follow-up window"
        elif self.wake_word_listening:
            wake_word_state = "listening for 'hey moss'"
        else:
            wake_word_state = "paused"
        self.wake_word_status_label.setText(f"Wake word: {wake_word_state}")

    def record_history_sample(self, driver_state: dict[str, Any]) -> None:
        now = time.perf_counter()
        if now - self._last_history_point_at < 0.5:
            return
        self._last_history_point_at = now
        elapsed = now - self._history_started_at
        score = float(compute_attention_score(driver_state))
        self.attention_history.append((elapsed, score))
        self.attention_history = self.attention_history[-180:]

        emotion_confidences = {emotion: 0.0 for emotion in EMOTION_HISTORY_LABELS}
        driver_detected = bool(driver_state.get("driver_detected", False))
        emotion = str(driver_state.get("emotion", "neutral")).lower()
        if driver_detected and emotion in emotion_confidences:
            emotion_confidences[emotion] = max(
                0.0,
                min(1.0, float(driver_state.get("emotion_confidence", 0.0))),
            )
        self.emotion_history.append((elapsed, emotion_confidences))
        self.emotion_history = self.emotion_history[-180:]

        if hasattr(self, "attention_chart"):
            self.attention_chart.set_points(self.attention_history)
        if hasattr(self, "emotion_chart"):
            self.emotion_chart.set_points(self.emotion_history)
        if hasattr(self, "history_summary_label") and self.attention_history:
            emotion_label = emotion if driver_detected else "none"
            emotion_confidence = emotion_confidences.get(emotion, 0.0) if driver_detected else 0.0
            self.history_summary_label.setText(
                f"Attention {score:.0f}/100 | Emotion {normalize_label(emotion_label)} {emotion_confidence:.2f}"
            )

    def export_history_chart(self) -> None:
        if not self.attention_history:
            QMessageBox.information(self, "Export History Chart", "No history samples to export yet.")
            return

        export_dir = PROJECT_ROOT / "debug_exports" / "history_charts"
        export_dir.mkdir(parents=True, exist_ok=True)
        default_path = export_dir / f"history_{time.strftime('%Y%m%d_%H%M%S')}.png"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export History Chart",
            str(default_path),
            "PNG images (*.png)",
        )
        if not file_path:
            return
        if not file_path.lower().endswith(".png"):
            file_path = f"{file_path}.png"

        selected_path = Path(file_path)
        attention_path = selected_path.with_name(f"{selected_path.stem}_attention.png")
        emotion_path = selected_path.with_name(f"{selected_path.stem}_emotion.png")

        attention_saved = self.attention_chart.grab().save(str(attention_path), "PNG")
        emotion_saved = self.emotion_chart.grab().save(str(emotion_path), "PNG")
        if not attention_saved or not emotion_saved:
            QMessageBox.warning(self, "Export History Chart", "Failed to save one or more chart images.")
            self.append_log(
                "error",
                f"Failed to export history charts to {attention_path} and {emotion_path}",
            )
            return

        self.status_label.setText(f"Status: exported attention and emotion history charts")
        self.append_log("history", f"Exported attention chart to {attention_path}")
        self.append_log("history", f"Exported emotion chart to {emotion_path}")

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
        self.eye_monitoring_enabled = bool(
            state.get("eye_monitoring_enabled", self.eye_monitoring_enabled)
        )
        self.record_history_sample(state)
        snapshot = (driver_detected, self.current_emotion, self.current_risk, focus_level)
        if snapshot != self._last_logged_snapshot:
            self._last_logged_snapshot = snapshot
            self.append_log(
                "vision",
                f"driver={'yes' if driver_detected else 'no'}, emotion={self.current_emotion}, risk={self.current_risk}, focus_level={focus_level}",
            )
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
            eye_text = f"warming up ({eye_warmup_remaining:.1f}s)"
        else:
            eye_text = f"{normalize_label(state['eye_label'])} ({state['eye_confidence']:.2f})"
        if not self.eye_monitoring_enabled:
            eye_text = f"{eye_text} - monitor off"
        self.eye_label.setText(eye_text)
        self.eye_label.setStyleSheet(
            "color: #ff9500; font-size: 17px; font-weight: 700; border: none; background: transparent;"
            if eye_warmup_active
            else "color: #0059b5; font-size: 17px; font-weight: 700; border: none; background: transparent;"
        )
        self.risk_label.setText(self.current_risk)
        self.risk_label.setStyleSheet(
            f"color: {risk_color_hex(self.current_risk)}; font-size: 17px; font-weight: 700; border: none; background: transparent;"
        )
        emotion_warning = bool(state.get("emotion_warning_active", False))
        if state["focus_alert"] and emotion_warning:
            streak_lbl = str(state.get("emotion_streak_label", "")).capitalize()
            self.focus_label.setText(f"{streak_lbl} alert active")
        elif state["focus_alert"]:
            self.focus_label.setText("Please stay focused")
        elif focus_level == 1:
            self.focus_label.setText("Stay attentive")
        else:
            self.focus_label.setText("OK")
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
            self.reason_label.setText(trigger_reason)
        else:
            self.reason_label.setText("none")
        # Emotion streak display
        emotion_streak_lbl = str(state.get("emotion_streak_label", "neutral"))
        emotion_streak_dur = float(state.get("emotion_streak_duration", 0.0))
        if emotion_streak_lbl not in ("neutral", "happy") and emotion_streak_dur > 1.0:
            self.emotion_streak_label.setText(
                f"{emotion_streak_lbl} ({emotion_streak_dur:.1f}s)"
            )
            self.emotion_streak_label.setStyleSheet(
                f"color: {emotion_color_hex(emotion_streak_lbl)}; font-size: 17px; font-weight: 700; border: none; background: transparent;"
            )
        else:
            self.emotion_streak_label.setText("none")
            self.emotion_streak_label.setStyleSheet(
                "color: #00a86b; font-size: 17px; font-weight: 700; border: none; background: transparent;"
            )
        emotion_timer_threshold = EMOTION_TRIGGER_THRESHOLDS.get(emotion_streak_lbl)
        if emotion_timer_threshold is not None:
            remaining = max(0.0, emotion_timer_threshold - emotion_streak_dur)
            self.emotion_timer_label.setText(
                f"{emotion_streak_lbl} {emotion_streak_dur:.1f}s / {emotion_timer_threshold:.0f}s"
                + (f" ({remaining:.1f}s left)" if remaining > 0 else " (triggered)")
            )
            timer_color = "#e53935" if emotion_warning else emotion_color_hex(emotion_streak_lbl)
            self.emotion_timer_label.setStyleSheet(
                f"color: {timer_color}; font-size: 17px; font-weight: 700; border: none; background: transparent;"
            )
        else:
            self.emotion_timer_label.setText("not active")
            self.emotion_timer_label.setStyleSheet(
                "color: #00a86b; font-size: 17px; font-weight: 700; border: none; background: transparent;"
            )
        # Emotion alert counter
        emotion_trigger_count = int(state.get("emotion_trigger_count", 0))
        emotion_breakdown = state.get("emotion_trigger_breakdown", {})
        if emotion_trigger_count > 0 and emotion_breakdown:
            parts = [f"{k} x{v}" for k, v in emotion_breakdown.items()]
            self.emotion_alerts_label.setText(
                f"{emotion_trigger_count} ({', '.join(parts)})"
            )
            self.emotion_alerts_label.setStyleSheet(
                "color: #e53935; font-size: 17px; font-weight: 700; border: none; background: transparent;"
            )
        else:
            self.emotion_alerts_label.setText("0")
            self.emotion_alerts_label.setStyleSheet(
                "color: #00a86b; font-size: 17px; font-weight: 700; border: none; background: transparent;"
            )
        self.refresh_voice_status_labels()
        self.context_label.setText(build_context_summary(state))

    def handle_worker_error(self, message: str) -> None:
        self.status_label.setText(f"Status: error - {message}")
        self.append_log("error", f"Vision worker: {message}")

    def handle_model_selection_change(self, model_name: str) -> None:
        self.last_selected_model = model_name
        message = f"Selected chat model: {model_name}"
        self.status_label.setText(f"Status: {message}")
        print(message)
        self.refresh_models_table()
        self.append_log("models", message)

    def push_chat_settings_to_worker(self, *args: object) -> None:
        self.vision_worker.update_chat_settings(
            self.model_combo.currentText(),
            float(self.temperature_spin.value()),
        )

    def send_text_message(self) -> None:
        text = self.input_edit.text().strip()
        if not text or self.chatbot is None or self.chat_worker is not None:
            return
        if not is_supported_zh_en_text(text):
            self.status_label.setText("Status: only Chinese and English text input is supported")
            self.append_log("error", "Rejected unsupported text input")
            return

        self.input_edit.clear()
        self.add_message(text, is_user=True)
        history_snapshot = list(self.conversation_history)
        self.conversation_history.append({"role": "user", "content": text})
        self.append_log("chat", f"Manual text input: {text[:80]}")
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
        self.append_log(
            "chat",
            f"Sending {'auto' if auto_trigger else 'manual'} request via {selected_model}",
        )
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
        print(
            f"OpenRouter reply received from {payload['model']} "
            f"in {payload['latency_ms']:.0f} ms"
        )
        self.refresh_models_table()
        self.append_log(
            "chat",
            f"Reply via {payload['model']} in {payload['latency_ms']:.0f} ms"
            + (" (fallback)" if self.last_fallback_used else ""),
        )
        self.status_label.setText("Status: assistant reply queued for speech output")

    def handle_chat_error(self, message: str) -> None:
        self.status_label.setText(f"Status: chatbot error - {message}")
        print(f"Chat error: {message}", flush=True)
        self.append_log("error", f"Chatbot: {message}")

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
        self.refresh_models_table()
        self.set_voice_input_state("follow-up speech received")
        self.set_voice_output_state("reply queued")
        self.set_voice_dialogue_state("completed")
        latency_ms = payload.get("latency_ms")
        if isinstance(latency_ms, (int, float)):
            self.status_label.setText("Status: voice dialogue completed; reply queued")
            self.append_log(
                "voice",
                f"Voice dialogue reply via {model} in {float(latency_ms):.0f} ms"
                + (" (fallback)" if self.last_fallback_used else ""),
            )
        else:
            self.status_label.setText("Status: voice dialogue completed")
            self.append_log("voice", f"Voice dialogue completed via {model}")

    def handle_voice_dialogue_error(self, message: str) -> None:
        if message.strip() == "No speech detected.":
            self.set_voice_input_state("no follow-up speech detected")
            self.set_voice_output_state("idle")
            self.status_label.setText("Status: no speech detected")
            print("Voice dialogue: no speech detected", flush=True)
            self.append_log("voice", "No follow-up speech detected")
            return
        self.set_voice_dialogue_state("error")
        self.set_voice_output_state("idle")
        self.status_label.setText(f"Status: voice dialogue error - {message}")
        print(f"Voice dialogue error: {message}", flush=True)
        self.append_log("error", f"Voice dialogue: {message}")

    @pyqtSlot(str)
    def handle_auto_voice_status(self, message: str) -> None:
        lower = message.lower()
        if "tts" in lower or "beep" in lower:
            self.set_voice_output_state(message)
        if "waiting" in lower or "speech" in lower or "record" in lower:
            self.set_voice_input_state(message)
        if "dialogue" in lower:
            self.set_voice_dialogue_state(message)
        self.status_label.setText(f"Status: {message}")
        self.refresh_voice_status_labels()

    def clear_chat_worker(self) -> None:
        self.send_button.setEnabled(True)
        self.mic_button.setEnabled(self.voice_input_enabled)
        self.chat_worker = None

    def update_audio_control_buttons(self) -> None:
        self.mute_button.setText("Unmute TTS" if self.tts_muted else "Mute TTS")
        self.mute_button.setStyleSheet(self.secondary_control_button_style(active=self.tts_muted))
        self.stop_listening_button.setText(
            "Start Listening" if not self.voice_input_enabled else "Stop Listening"
        )
        self.stop_listening_button.setStyleSheet(
            self.secondary_control_button_style(active=not self.voice_input_enabled)
        )
        self.eye_monitor_button.setText(
            "Disable Eye Monitor" if self.eye_monitoring_enabled else "Enable Eye Monitor"
        )
        self.eye_monitor_button.setStyleSheet(
            self.secondary_control_button_style(active=not self.eye_monitoring_enabled)
        )
        self.ptt_button.setEnabled(self.voice_input_enabled)
        self.mic_button.setEnabled(self.voice_input_enabled)
        if not self.voice_input_enabled:
            self.mic_button.setText("Wake-word: OFF")
        elif self.wake_word_listening:
            self.mic_button.setText("Wake-word: ON")
        self.refresh_voice_status_labels()

    def _pause_voice_listeners_for_tts(self) -> None:
        if self.continued_listener is not None:
            self.continued_listener.stop(wait=False)
        if self.wake_word_listener is not None and self.wake_word_listening:
            self.wake_word_listener.stop(wait=False)
            self.wake_word_listening = False
        self.update_audio_control_buttons()

    def interrupt_tts_for_manual_recording(self) -> None:
        TextToSpeech.interrupt_all()
        if self.continued_listener is not None:
            self.continued_listener.stop(wait=False)
        if self.wake_word_listener is not None and self.wake_word_listening:
            self.wake_word_listener.stop(wait=False)
            self.wake_word_listening = False
        self._in_continued_mode = False
        self.set_voice_output_state("interrupted by Hold to Talk")
        self.set_voice_input_state("manual recording starting")
        self.append_log("voice", "Hold to Talk interrupted TTS and pending speech")
        self.update_audio_control_buttons()

    def speak_reply_async(self, text: str, emotion: str | None = None) -> None:
        if not text.strip():
            self._on_tts_finished()
            return

        if self.tts_muted or TTSQueue.instance().is_muted():
            self.set_voice_output_state("muted; reply skipped")
            self.status_label.setText("Status: assistant reply audio muted")
            self.append_log("voice", "Assistant reply TTS skipped because mute is enabled")
            return

        self._pause_voice_listeners_for_tts()
        self.set_voice_input_state("paused while TTS is speaking")
        self.set_voice_output_state("speaking assistant reply")
        self.status_label.setText("Status: speaking assistant reply...")
        self.append_log("voice", f"Speaking assistant reply: {text[:80]}")

        def on_done() -> None:
            QMetaObject.invokeMethod(
                self,
                "_on_tts_finished",
                Qt.ConnectionType.QueuedConnection,
            )

        def on_error(exc: BaseException) -> None:
            print(f"TTS error: {exc}")
            QMetaObject.invokeMethod(
                self,
                "_on_tts_failed",
                Qt.ConnectionType.QueuedConnection,
            )

        self.reply_tts.speak(
            text,
            emotion=emotion,
            wait=False,
            on_done=on_done,
            on_error=on_error,
            priority=TTS_PRIORITY_CHAT,
        )

    @pyqtSlot()
    def _on_tts_finished(self) -> None:
        if self.speech_worker is not None:
            self.set_voice_output_state("idle")
            self.append_log("voice", "TTS callback ignored during manual recording")
            self.update_audio_control_buttons()
            return
        if not self.voice_input_enabled:
            self._in_continued_mode = False
            self.set_voice_output_state("idle")
            self.set_voice_input_state("listening stopped")
            self.status_label.setText("Status: speech finished; listening is stopped")
            self.append_log("voice", "TTS finished; follow-up listening skipped because listening is stopped")
            self.update_audio_control_buttons()
            return
        self._in_continued_mode = True
        if self.continued_listener is not None:
            self.continued_listener.start()
            self.set_voice_output_state("idle")
            self.set_voice_input_state("listening for follow-up (5 s)")
            self.status_label.setText("Status: listening for follow-up (5 s)...")
            self.append_log("voice", "Entered 5-second follow-up listening window")

    @pyqtSlot()
    def _on_tts_failed(self) -> None:
        self.set_voice_output_state("failed")
        self.status_label.setText("Status: assistant reply audio failed")
        self.append_log("error", "Assistant reply TTS failed")
        self._on_tts_finished()

    def start_recording(self, push_to_talk: bool = False) -> None:
        """Start recording.

        If `push_to_talk` is True, disable VAD and record until user releases button.
        Otherwise use VAD with configured silence threshold.
        """
        if push_to_talk:
            self.interrupt_tts_for_manual_recording()
        if not self.voice_input_enabled:
            self.set_voice_input_state("listening stopped")
            self.status_label.setText("Status: microphone listening is stopped")
            self.append_log("voice", "Recording request ignored because listening is stopped")
            return
        if not push_to_talk and VoiceIOGate.is_tts_active():
            self.set_voice_input_state("blocked while TTS is speaking")
            self.status_label.setText("Status: TTS is speaking; voice input ignored")
            self.append_log("voice", "Recording request ignored while TTS is active")
            return
        if self.speech_worker is not None:
            return
        if push_to_talk:
            self.set_voice_input_state("recording push-to-talk")
            self.status_label.setText("Status: recording (push-to-talk)...")
            self.mic_button.setText("Recording...")
            self.append_log("voice", "Push-to-talk recording started")
            # Push-to-talk: disable VAD, stop on explicit release.
            self.speech_worker = SpeechWorker(
                self.args.whisper_model_size,
                duration_seconds=self.args.max_recording_duration,
                vad_enabled=False,
            )
        else:
            self.set_voice_input_state("recording with auto-stop")
            self.status_label.setText("Status: recording audio (auto-stop on silence)...")
            self.mic_button.setText("Recording...")
            self.append_log("voice", "Voice recording started with VAD")
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
            self.set_voice_input_state("recording stop requested")
            self.append_log("voice", "Recording stop requested")

    def handle_transcription(self, text: str) -> None:
        if VoiceIOGate.is_tts_active():
            self.input_edit.clear()
            self.set_voice_input_state("transcription ignored during TTS")
            self.status_label.setText("Status: TTS is speaking; transcription ignored")
            self.append_log("voice", "Transcription ignored while TTS is active")
            return
        self.input_edit.setText(text)
        if text.strip():
            detected_language = detect_text_language(text)
            if detected_language is None or not is_supported_zh_en_text(text):
                self.status_label.setText("Status: only Chinese and English voice input is supported")
                self.append_log("error", "Rejected unsupported voice transcription")
                self.input_edit.clear()
                return
            self.set_voice_input_state("transcription ready")
            self.status_label.setText("Status: transcription ready")
            self.append_log("voice", f"Transcription ready: {text[:80]}")
            self.send_text_message()
        else:
            # 空转录：如果还在 continued 窗口，重新等；否则恢复 wake-word
            if self._in_continued_mode:
                if self.voice_input_enabled and self.continued_listener is not None:
                    self.continued_listener.start()
                    self.set_voice_input_state("listening for follow-up (5 s)")
                    self.status_label.setText("Status: listening for follow-up (5 s)...")
            else:
                if (
                    self.voice_input_enabled
                    and self.wake_word_listener is not None
                    and not self.wake_word_listening
                ):
                    self.wake_word_listener.start()
                    self.wake_word_listening = True
                self.status_label.setText(
                    "Status: listening for 'hey moss'"
                    if self.voice_input_enabled
                    else "Status: microphone listening stopped"
                )
                self.set_voice_input_state(
                    "listening for 'hey moss'"
                    if self.voice_input_enabled
                    else "listening stopped"
                )
            self.update_audio_control_buttons()
            self.append_log("voice", "Empty transcription result")

    def handle_speech_error(self, message: str) -> None:
        if not self.voice_input_enabled and (
            "No audio was captured" in message or "microphone" in message.lower()
        ):
            self.set_voice_input_state("listening stopped")
            self.status_label.setText("Status: microphone listening stopped")
            self.append_log("voice", "Speech worker stopped after listening was disabled")
            return
        if message.strip() == "Only Chinese and English voice input is supported.":
            self.set_voice_input_state("unsupported speech language")
            self.status_label.setText("Status: only Chinese and English voice input is supported")
            self.append_log("error", "Speech: unsupported language")
            return
        self.set_voice_input_state("speech error")
        self.status_label.setText(f"Status: speech error - {message}")
        self.append_log("error", f"Speech: {message}")

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
            if (
                self.voice_input_enabled
                and self.wake_word_listener is not None
                and not self.wake_word_listening
            ):
                self.wake_word_listener.start()
                self.wake_word_listening = True
            self.status_label.setText(
                "Status: listening for 'hey moss'"
                if self.voice_input_enabled
                else "Status: microphone listening stopped"
            )
            self.set_voice_input_state(
                "listening for 'hey moss'"
                if self.voice_input_enabled
                else "listening stopped"
            )
        self.update_audio_control_buttons()
        self.append_log("voice", "Speech worker cleared")
    
    def toggle_tts_mute(self) -> None:
        self.tts_muted = not self.tts_muted
        TTSQueue.instance().set_muted(self.tts_muted)
        if self.tts_muted:
            self.set_voice_output_state("muted")
            self.status_label.setText("Status: TTS muted")
            self.append_log("voice", "TTS muted; pending speech jobs cleared")
        else:
            self.set_voice_output_state("idle")
            self.status_label.setText("Status: TTS unmuted")
            self.append_log("voice", "TTS unmuted")
        self.update_audio_control_buttons()

    def toggle_eye_monitoring(self) -> None:
        self.eye_monitoring_enabled = not self.eye_monitoring_enabled
        self.vision_worker.update_eye_monitoring_enabled(self.eye_monitoring_enabled)
        if self.eye_monitoring_enabled:
            self.status_label.setText("Status: eye monitoring enabled")
            self.append_log("system", "Eye monitoring enabled")
        else:
            self.status_label.setText("Status: eye monitoring disabled")
            self.append_log("system", "Eye monitoring disabled; closed-eye alerts suppressed")
        self.update_audio_control_buttons()

    def stop_all_listening(self) -> None:
        if self.continued_listener is not None:
            self.continued_listener.stop(wait=False)
        if self.wake_word_listener is not None and self.wake_word_listening:
            self.wake_word_listener.stop(wait=False)
        self.wake_word_listening = False
        self._in_continued_mode = False
        if self.speech_worker is not None:
            self.speech_worker.stop_recording()

    def start_background_listening(self) -> None:
        if self.wake_word_listener is not None and not self.wake_word_listening:
            self.wake_word_listener.start()
            self.wake_word_listening = True

    def toggle_voice_input(self) -> None:
        self.voice_input_enabled = not self.voice_input_enabled
        self.vision_worker.update_voice_dialogue_enabled(
            self.voice_input_enabled and bool(self.args.enable_voice_dialogue)
        )
        if self.voice_input_enabled:
            self.start_background_listening()
            self.set_voice_input_state("listening for 'hey moss'")
            self.set_voice_dialogue_state(
                "enabled" if bool(self.args.enable_voice_dialogue) else "disabled"
            )
            self.status_label.setText("Status: listening for 'hey moss'")
            self.append_log("voice", "Microphone listening enabled")
        else:
            self.stop_all_listening()
            self.set_voice_input_state("listening stopped")
            self.set_voice_dialogue_state("disabled while listening is stopped")
            self.status_label.setText("Status: microphone listening stopped")
            self.append_log("voice", "Microphone listening stopped")
        self.update_audio_control_buttons()

    def toggle_wake_word_listening(self) -> None:
        """Toggle wake-word listening on/off."""
        if self.wake_word_listener is None:
            return
        self.toggle_voice_input()

    def on_wake_word_detected(self) -> None:
        """Background-thread callback from the wake-word listener."""
        self.wake_word_detected_signal.emit()

    def on_continued_voice_detected(self) -> None:
        """Background-thread callback from the continued listener."""
        self.continued_voice_detected_signal.emit()
        
    def on_continued_timeout(self) -> None:
        """Background-thread timeout callback from the continued listener."""
        self.continued_timeout_signal.emit()

    @pyqtSlot()
    def _handle_wake_word_detected(self) -> None:
        if not self.voice_input_enabled:
            return
        if VoiceIOGate.is_tts_active():
            self.set_voice_input_state("wake-word ignored during TTS")
            self.append_log("voice", "Wake-word ignored while TTS is active")
            return
        if self.speech_worker is not None:
            return
        self.set_voice_input_state("wake-word detected; recording")
        self.status_label.setText("Status: wake-word detected! Recording for 5 seconds...")
        print("[Wake-word] Triggered speech recording.")
        self.append_log("voice", "Wake-word detected")
        self.start_recording()

    @pyqtSlot()
    def _handle_continued_voice_detected(self) -> None:
        if not self.voice_input_enabled:
            return
        if VoiceIOGate.is_tts_active():
            self.set_voice_input_state("follow-up ignored during TTS")
            self.append_log("voice", "Follow-up voice ignored while TTS is active")
            return
        if self.speech_worker is not None:
            return
        self.set_voice_input_state("follow-up detected; recording")
        self.status_label.setText("Status: follow-up voice detected, recording...")
        self.append_log("voice", "Follow-up speech detected")
        self.start_recording(push_to_talk=False)

    @pyqtSlot()
    def _handle_continued_timeout(self) -> None:
        self._in_continued_mode = False
        if self.voice_input_enabled and self.wake_word_listener is not None:
            self.wake_word_listener.start()
            self.wake_word_listening = True
        self.set_voice_input_state(
            "listening for 'hey moss'"
            if self.voice_input_enabled
            else "listening stopped"
        )
        self.status_label.setText(
            "Status: listening for 'hey moss'"
            if self.voice_input_enabled
            else "Status: microphone listening stopped"
        )
        self.update_audio_control_buttons()
        self.append_log("voice", "Follow-up timeout; returned to wake-word listening")

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
