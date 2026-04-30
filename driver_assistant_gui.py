from __future__ import annotations

import argparse
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import torch
from PyQt5.QtCore import QObject, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
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

from chatbot import DEFAULT_MODEL, DriverAssistantChatbot, SUPPORTED_LLM_MODELS
from realtime_emotion_webcam import (
    classify_crops,
    choose_driver_face,
    draw_tag,
    ensure_face_model,
    estimate_eye_boxes,
    expand_box,
    load_timm_classifier,
    normalize_checkpoint_path,
    normalize_label,
    resolve_devices,
    resolve_latest_timm_model,
)
from speech_to_text import WhisperTranscriber, record_microphone_audio


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
        default=Path(r"G:\731\runs_timm\efficientnet_b0\best_model.pth"),
        help="Path to a timm emotion checkpoint file or run directory.",
    )
    parser.add_argument(
        "--eye-model",
        type=Path,
        default=Path(r"G:\731\runs_timm\eye_efficientnet_b0\best_model.pth"),
        help="Path to a timm eye-state checkpoint file or run directory.",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path(r"G:\731\runs_timm"),
        help="Directory containing timm training runs.",
    )
    parser.add_argument(
        "--face-model",
        type=Path,
        default=Path(r"G:\731\weights\yolov8n-face-lindevs.pt"),
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
        "--focus-seconds",
        type=float,
        default=2.0,
        help="Closed-eye duration before the focus warning appears.",
    )
    parser.add_argument(
        "--driver-side",
        type=str,
        choices=["left", "center", "right", "largest"],
        default="right",
        help="Heuristic used to choose the driver's face when multiple faces are visible.",
    )
    parser.add_argument("--default-llm-model", type=str, default=DEFAULT_MODEL, help="Default OpenRouter model.")
    parser.add_argument("--default-temperature", type=float, default=1.0, help="Default chatbot temperature.")
    parser.add_argument("--whisper-model-size", type=str, default="base", help="faster-whisper model size.")
    return parser.parse_args()


def emotion_color_bgr(emotion: str) -> tuple[int, int, int]:
    return EMOTION_COLORS_BGR.get(emotion, EMOTION_COLORS_BGR["neutral"])


def emotion_color_hex(emotion: str) -> str:
    return EMOTION_COLORS_HEX.get(emotion, EMOTION_COLORS_HEX["neutral"])


class ChatBubble(QFrame):
    def __init__(self, text: str, is_user: bool) -> None:
        super().__init__()
        bubble = QLabel(text)
        bubble.setWordWrap(True)
        bubble.setTextInteractionFlags(Qt.TextSelectableByMouse)
        bubble.setStyleSheet(
            (
                "background-color: #c62828; color: white; border-radius: 14px; "
                "padding: 10px 12px; font-size: 14px;"
            )
            if is_user
            else (
                "background-color: #0d47a1; color: white; border-radius: 14px; "
                "padding: 10px 12px; font-size: 14px;"
            )
        )
        bubble.setMaximumWidth(340)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        if is_user:
            layout.addStretch()
            layout.addWidget(bubble)
        else:
            layout.addWidget(bubble)
            layout.addStretch()


class VisionWorker(QObject):
    frame_ready = pyqtSignal(QImage)
    state_ready = pyqtSignal(object)
    error = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.args = args
        self._running = True

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

            cap = cv2.VideoCapture(self.args.camera_index)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.args.capture_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.args.capture_height)
            if not cap.isOpened():
                raise RuntimeError(f"Unable to open webcam index {self.args.camera_index}.")

            last_state = {
                "emotion": "neutral",
                "emotion_confidence": 0.0,
                "eye_label": "open_eye",
                "eye_confidence": 0.0,
                "risk": "OK",
                "focus_alert": False,
                "emotion_model_path": str(emotion_model_path),
                "eye_model_path": str(eye_model_path),
            }
            previous_time = time.perf_counter()
            closed_eye_start: float | None = None
            previous_driver_center_x: float | None = None

            while self._running:
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("Failed to read a frame from the webcam.")

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
                crops_rgb = []
                if face_result.boxes is not None:
                    raw_boxes = face_result.boxes.xyxy.cpu().numpy().astype(int)
                    raw_confidences = face_result.boxes.conf.cpu().numpy().tolist()
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
                        crops_rgb.append(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))

                emotion_predictions = classify_crops(emotion_classifier, crops_rgb, torch_device)
                eye_predictions = classify_crops(eye_classifier, crops_rgb, torch_device)
                primary_face = None
                if detections and emotion_predictions and eye_predictions:
                    faces = []
                    for detection, crop_rgb, emotion_prediction, eye_prediction in zip(
                        detections,
                        crops_rgb,
                        emotion_predictions,
                        eye_predictions,
                    ):
                        x1, y1, x2, y2, detection_conf = detection
                        emotion_label, emotion_confidence = emotion_prediction
                        eye_label, eye_confidence = eye_prediction
                        faces.append(
                            {
                                "bbox": (x1, y1, x2, y2),
                                "area": (x2 - x1) * (y2 - y1),
                                "emotion": emotion_label,
                                "emotion_confidence": emotion_confidence,
                                "eye_label": eye_label,
                                "eye_confidence": eye_confidence,
                                "eye_boxes": estimate_eye_boxes(crop_rgb.shape),
                                "detection_confidence": detection_conf,
                            }
                        )

                    primary_face = choose_driver_face(
                        faces,
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
                        color = emotion_color_bgr(emotion)
                        is_driver_face = face is primary_face
                        cv2.rectangle(
                            frame,
                            (x1, y1),
                            (x2, y2),
                            color,
                            4 if is_driver_face else 3,
                        )
                        draw_tag(
                            frame,
                            f"{normalize_label(emotion)} {face['emotion_confidence']:.2f}",
                            (x1, y1),
                            color,
                        )
                        for eye_box in face["eye_boxes"]:
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
                            f"{normalize_label(face['eye_label'])} {face['eye_confidence']:.2f}",
                            (x1, min(frame_h - 5, y2 + 28)),
                            (255, 0, 0),
                        )
                        if is_driver_face:
                            draw_tag(frame, "driver", (x1, y2 + 56), (255, 255, 255))

                if primary_face is not None:
                    current_emotion = primary_face["emotion"]
                    emotion_confidence = float(primary_face["emotion_confidence"])
                    current_eye_label = primary_face["eye_label"]
                    eye_confidence = float(primary_face["eye_confidence"])
                else:
                    current_emotion = "neutral"
                    emotion_confidence = 0.0
                    current_eye_label = "open_eye"
                    eye_confidence = 0.0

                if (
                    primary_face is not None
                    and current_eye_label == "closed_eye"
                    and eye_confidence >= self.args.classification_confidence
                ):
                    if closed_eye_start is None:
                        closed_eye_start = time.perf_counter()
                else:
                    closed_eye_start = None

                risk = EMOTION_TO_RISK.get(current_emotion, "OK")
                focus_alert = (
                    closed_eye_start is not None
                    and time.perf_counter() - closed_eye_start >= self.args.focus_seconds
                )
                if focus_alert:
                    risk = "HIGH"
                last_state = {
                    "emotion": current_emotion,
                    "emotion_confidence": emotion_confidence,
                    "eye_label": current_eye_label,
                    "eye_confidence": eye_confidence,
                    "risk": risk,
                    "focus_alert": focus_alert,
                    "driver_side": self.args.driver_side,
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
                if focus_alert:
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
    ) -> None:
        super().__init__()
        self.chatbot = chatbot
        self.emotion = emotion
        self.user_message = user_message
        self.conversation_history = conversation_history
        self.model = model
        self.temperature = temperature
        self.auto_trigger = auto_trigger

    def run(self) -> None:
        try:
            response = self.chatbot.generate_reply(
                emotion=self.emotion,
                user_message=self.user_message,
                model=self.model,
                temperature=self.temperature,
                conversation_history=self.conversation_history,
                auto_trigger=self.auto_trigger,
            )
            self.result_ready.emit(asdict(response))
        except Exception as exc:
            self.error.emit(str(exc))


class SpeechWorker(QThread):
    transcription_ready = pyqtSignal(str)
    error = pyqtSignal(str)

    _cache_lock = threading.Lock()
    _transcriber_cache: dict[tuple[str, str], WhisperTranscriber] = {}

    def __init__(self, model_size: str, duration_seconds: float = 5.0) -> None:
        super().__init__()
        self.model_size = model_size
        self.duration_seconds = duration_seconds
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
        self.setWindowTitle("Driver Assistant")
        self.resize(1500, 900)

        self.current_emotion = "neutral"
        self.current_risk = "OK"
        self.conversation_history: list[dict[str, str]] = []
        self.chat_worker: ChatWorker | None = None
        self.speech_worker: SpeechWorker | None = None

        try:
            self.chatbot = DriverAssistantChatbot(app_title="Driver Assistant GUI")
        except Exception as exc:
            self.chatbot = None
            QMessageBox.warning(self, "OpenRouter", str(exc))

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(16, 16, 16, 16)
        root_layout.setSpacing(18)

        left_panel = QVBoxLayout()
        right_panel = QVBoxLayout()
        root_layout.addLayout(left_panel, 3)
        root_layout.addLayout(right_panel, 2)

        self.video_label = QLabel("Camera starting...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(860, 560)
        self.video_label.setStyleSheet("background-color: #101418; color: white; border-radius: 12px;")
        left_panel.addWidget(self.video_label)

        info_card = QFrame()
        info_card.setStyleSheet("background-color: #111827; color: white; border-radius: 12px;")
        info_layout = QVBoxLayout(info_card)
        self.emotion_label = QLabel("Emotion: neutral")
        self.eye_label = QLabel("Eyes: open eye")
        self.risk_label = QLabel("Risk: OK")
        self.focus_label = QLabel("Focus: OK")
        self.driver_label = QLabel("Driver heuristic: right")
        self.model_path_label = QLabel("Emotion model: loading...")
        self.eye_model_path_label = QLabel("Eye model: loading...")
        self.status_label = QLabel("Status: ready")
        for label in [
            self.emotion_label,
            self.eye_label,
            self.risk_label,
            self.focus_label,
            self.driver_label,
            self.model_path_label,
            self.eye_model_path_label,
            self.status_label,
        ]:
            label.setWordWrap(True)
            info_layout.addWidget(label)
        left_panel.addWidget(info_card)

        controls_row = QHBoxLayout()
        self.model_combo = QComboBox()
        self.model_combo.addItems(SUPPORTED_LLM_MODELS)
        default_index = self.model_combo.findText(self.args.default_llm_model)
        if default_index >= 0:
            self.model_combo.setCurrentIndex(default_index)
        self.model_combo.currentTextChanged.connect(self.handle_model_selection_change)
        self.temperature_spin = QDoubleSpinBox()
        self.temperature_spin.setRange(0.0, 2.0)
        self.temperature_spin.setSingleStep(0.1)
        self.temperature_spin.setValue(self.args.default_temperature)
        controls_row.addWidget(QLabel("LLM Model"))
        controls_row.addWidget(self.model_combo, 1)
        controls_row.addWidget(QLabel("Temperature"))
        controls_row.addWidget(self.temperature_spin)
        right_panel.addLayout(controls_row)

        self.chat_scroll = QScrollArea()
        self.chat_scroll.setWidgetResizable(True)
        self.chat_scroll.setStyleSheet("background-color: #f3f4f6; border: none;")
        self.chat_container = QWidget()
        self.chat_layout = QVBoxLayout(self.chat_container)
        self.chat_layout.setContentsMargins(8, 8, 8, 8)
        self.chat_layout.setSpacing(6)
        self.chat_layout.addStretch()
        self.chat_scroll.setWidget(self.chat_container)
        right_panel.addWidget(self.chat_scroll, 1)

        input_row = QHBoxLayout()
        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("Type a short message to the assistant...")
        self.input_edit.returnPressed.connect(self.send_text_message)
        self.mic_button = QPushButton("Hold to Talk")
        self.mic_button.pressed.connect(self.start_recording)
        self.mic_button.released.connect(self.stop_recording)
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.send_text_message)
        input_row.addWidget(self.input_edit, 1)
        input_row.addWidget(self.mic_button)
        input_row.addWidget(self.send_button)
        right_panel.addLayout(input_row)

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
        self.vision_worker.error.connect(self.handle_worker_error)
        self.vision_worker.finished.connect(self.vision_thread.quit)
        self.vision_thread.start()

    def add_message(self, text: str, is_user: bool) -> None:
        bubble = ChatBubble(text, is_user=is_user)
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, bubble)
        scrollbar = self.chat_scroll.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def update_frame(self, image: QImage) -> None:
        pixmap = QPixmap.fromImage(image).scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.video_label.setPixmap(pixmap)

    def update_emotion_state(self, state: dict) -> None:
        self.current_emotion = state["emotion"]
        self.current_risk = state["risk"]
        color = emotion_color_hex(self.current_emotion)
        self.emotion_label.setText(
            f"Emotion: {normalize_label(self.current_emotion)} ({state['emotion_confidence']:.2f})"
        )
        self.emotion_label.setStyleSheet(f"color: {color}; font-size: 18px; font-weight: 600;")
        self.eye_label.setText(
            f"Eyes: {normalize_label(state['eye_label'])} ({state['eye_confidence']:.2f})"
        )
        self.eye_label.setStyleSheet("color: #60a5fa; font-size: 18px; font-weight: 600;")
        self.risk_label.setText(f"Risk: {self.current_risk}")
        self.risk_label.setStyleSheet(
            f"color: {color}; font-size: 18px; font-weight: 600;"
        )
        self.focus_label.setText(
            "Focus: Please stay focused" if state["focus_alert"] else "Focus: OK"
        )
        self.focus_label.setStyleSheet(
            "color: #ef4444; font-size: 18px; font-weight: 600;"
            if state["focus_alert"]
            else "color: #34d399; font-size: 18px; font-weight: 600;"
        )
        self.driver_label.setText(f"Driver heuristic: {state['driver_side']}")
        self.model_path_label.setText(f"Emotion model: {state['emotion_model_path']}")
        self.eye_model_path_label.setText(f"Eye model: {state['eye_model_path']}")

    def handle_worker_error(self, message: str) -> None:
        self.status_label.setText(f"Status: error - {message}")

    def handle_model_selection_change(self, model_name: str) -> None:
        message = f"Selected chat model: {model_name}"
        self.status_label.setText(f"Status: {message}")
        print(message)

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
        )
        self.chat_worker.result_ready.connect(self.handle_chat_response)
        self.chat_worker.error.connect(self.handle_chat_error)
        self.chat_worker.finished.connect(self.clear_chat_worker)
        self.chat_worker.start()

    def handle_chat_response(self, payload: dict) -> None:
        self.add_message(payload["text"], is_user=False)
        self.conversation_history.append({"role": "assistant", "content": payload["text"]})
        print(
            f"OpenRouter reply received from {payload['model']} "
            f"in {payload['latency_ms']:.0f} ms"
        )
        self.status_label.setText(
            f"Status: OpenRouter reply in {payload['latency_ms']:.0f} ms via {payload['model']}"
        )

    def handle_chat_error(self, message: str) -> None:
        self.status_label.setText(f"Status: chatbot error - {message}")

    def clear_chat_worker(self) -> None:
        self.send_button.setEnabled(True)
        self.mic_button.setEnabled(True)
        self.chat_worker = None

    def start_recording(self) -> None:
        if self.speech_worker is not None:
            return
        self.status_label.setText("Status: recording audio...")
        self.mic_button.setText("Recording...")
        self.speech_worker = SpeechWorker(self.args.whisper_model_size, duration_seconds=5.0)
        self.speech_worker.transcription_ready.connect(self.handle_transcription)
        self.speech_worker.error.connect(self.handle_speech_error)
        self.speech_worker.finished.connect(self.clear_speech_worker)
        self.speech_worker.start()

    def stop_recording(self) -> None:
        if self.speech_worker is not None:
            self.speech_worker.stop_recording()

    def handle_transcription(self, text: str) -> None:
        self.input_edit.setText(text)
        self.status_label.setText("Status: transcription ready")
        if text.strip():
            self.send_text_message()

    def handle_speech_error(self, message: str) -> None:
        self.status_label.setText(f"Status: speech error - {message}")

    def clear_speech_worker(self) -> None:
        self.mic_button.setText("Hold to Talk")
        self.speech_worker = None

    def closeEvent(self, event) -> None:
        if self.speech_worker is not None:
            self.speech_worker.stop_recording()
            self.speech_worker.wait(2000)
        if self.chat_worker is not None:
            self.chat_worker.wait(2000)
        self.vision_worker.stop()
        self.vision_thread.quit()
        self.vision_thread.wait(3000)
        super().closeEvent(event)


def main() -> None:
    args = parse_args()
    app = QApplication(sys.argv)
    window = DriverAssistantWindow(args)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
