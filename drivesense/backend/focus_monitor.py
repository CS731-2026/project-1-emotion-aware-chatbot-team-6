"""Drowsiness focus monitor.

Tracks how long the driver has had their eyes closed and triggers a multi-modal
intervention when the duration exceeds a threshold (default 2 seconds):

    1. Audible beep alert (Windows winsound, fallback to console bell).
    2. Spoken check-in question via pyttsx3.
    3. Full voice dialogue loop (record -> transcribe -> LLM -> speak) using
       the existing VoiceChatPipeline.

The monitor is designed to be polled once per processed frame from the main
vision loop. All side effects (audio, network calls) run on a background
thread so the camera loop never blocks.
"""

from __future__ import annotations

import logging
import platform
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from drivesense.backend.voice_chat import NoSpeechDetectedError

logger = logging.getLogger(__name__)


@dataclass
class FocusMonitorConfig:
    """Tunable thresholds for the focus monitor."""

    closed_eye_seconds: float = 2.0
    """Continuous closed-eye duration that triggers a high-risk event."""

    cooldown_seconds: float = 30.0
    """Minimum gap between two high-risk triggers."""

    beep_frequency_hz: int = 1000
    """Frequency of the alert beep."""

    beep_duration_ms: int = 400
    """Duration of the alert beep."""

    check_in_question: str = "Hey, are you feeling tired? How are you doing?"
    """The TTS question played after the beep."""

    record_seconds: float = 5.0
    """How long to listen for the driver's reply."""

    chat_model: str = "openai/gpt-4o-mini"
    """OpenRouter model id used for the LLM reply step."""


@dataclass
class _State:
    """Internal mutable state of the monitor.

    Kept in its own dataclass so the public class stays readable.
    """

    closed_eye_started_at: Optional[float] = None
    last_trigger_at: Optional[float] = None
    is_handling_event: bool = False
    last_warning_active: bool = False
    chat_model: str = "openai/gpt-4o-mini"
    temperature: float = 1.0
    driver_state: dict[str, Any] | None = None
    closed_eye_duration: float = 0.0
    current_level: int = 0
    trigger_reason: str = ""
    repeat_count: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


def _play_beep(frequency_hz: int, duration_ms: int) -> None:
    """Play a short alert beep.

    Uses winsound on Windows; falls back to printing the BEL character so
    something still happens on Mac/Linux during testing.
    """
    if platform.system() == "Windows":
        try:
            import winsound
            winsound.Beep(frequency_hz, duration_ms)
            return
        except Exception as exc:  # pragma: no cover - hardware dependent
            logger.warning("winsound beep failed: %s", exc)

    print("\a", end="", flush=True)


def _build_check_in_question(
    default_question: str,
    emotion: str,
    driver_state: dict[str, Any] | None,
) -> str:
    risk = str((driver_state or {}).get("risk", "OK")).upper()
    normalized_emotion = (emotion or "neutral").strip().lower()

    if risk == "HIGH":
        return "Please stay focused. Take a breath."
    if normalized_emotion in {"anger", "disgust"}:
        return "Please stay focused. Take a slow breath."
    if normalized_emotion in {"fear", "surprise"}:
        return "Please stay focused. Stay calm."
    if normalized_emotion == "sad":
        return "Please stay focused. Are you okay?"
    if normalized_emotion == "happy":
        return "Please stay focused for a moment."
    if normalized_emotion == "neutral":
        return "Please stay focused. Eyes on the road."
    return default_question


class FocusMonitor:
    """Watch the driver's eye state and run a voice intervention when drowsy.

    Typical usage from a vision loop::

        monitor = FocusMonitor(
            config=FocusMonitorConfig(closed_eye_seconds=2.0),
            tts=TextToSpeech(),
            voice_pipeline=VoiceChatPipeline(chatbot=DriverAssistantChatbot()),
        )

        for frame in camera_stream():
            both_eyes_closed = run_eye_classifier(frame)
            current_emotion = run_emotion_classifier(frame)
            warning_active = monitor.update(
                eyes_closed=both_eyes_closed,
                emotion=current_emotion,
            )
            if warning_active:
                draw_warning(frame)

    The `update` method is non-blocking; the dialogue loop runs on a daemon
    thread.
    """

    def __init__(
        self,
        config: Optional[FocusMonitorConfig] = None,
        tts: Optional[object] = None,
        voice_pipeline: Optional[object] = None,
        on_voice_result: Optional[Callable[[Any], None]] = None,
        on_voice_error: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.config = config or FocusMonitorConfig()
        self._tts = tts
        self._voice_pipeline = voice_pipeline
        self._on_voice_result = on_voice_result
        self._on_voice_error = on_voice_error
        self._state = _State()
        self._state.chat_model = self.config.chat_model

    def get_state_snapshot(self) -> dict[str, Any]:
        with self._state.lock:
            return {
                "closed_eye_duration": self._state.closed_eye_duration,
                "focus_level": self._state.current_level,
                "trigger_reason": self._state.trigger_reason,
                "repeat_count": self._state.repeat_count,
                "focus_alert": self._state.last_warning_active,
            }

    def set_runtime_context(
        self,
        chat_model: str | None = None,
        temperature: float | None = None,
        driver_state: dict[str, Any] | None = None,
    ) -> None:
        with self._state.lock:
            if chat_model:
                self._state.chat_model = chat_model
            if temperature is not None:
                self._state.temperature = temperature
            if driver_state is not None:
                self._state.driver_state = dict(driver_state)

    def update(
        self,
        eyes_closed: bool,
        emotion: str = "neutral",
        driver_state: dict[str, Any] | None = None,
        now: Optional[float] = None,
    ) -> bool:
        """Feed one frame's eye state into the monitor.

        Args:
            eyes_closed: True when both eyes are currently classified closed.
            emotion: The driver's current detected emotion. Passed through to
                the TTS engine and the LLM prompt.
            driver_state: Per-frame driver state snapshot. When provided, the
                monitor stores a synchronized copy with the computed
                focus-alert/risk values before any intervention is launched.
            now: Override the current timestamp (mostly for unit tests).

        Returns:
            True if the on-screen warning should currently be drawn.
        """
        if now is None:
            now = time.perf_counter()

        preview_threshold = max(self.config.closed_eye_seconds * 0.6, 0.5)

        with self._state.lock:
            if eyes_closed:
                if self._state.closed_eye_started_at is None:
                    self._state.closed_eye_started_at = now
                duration = now - self._state.closed_eye_started_at
            else:
                self._state.closed_eye_started_at = None
                duration = 0.0

            warning_active = duration >= self.config.closed_eye_seconds
            preview_active = duration >= preview_threshold and not warning_active
            should_trigger = (
                warning_active
                and not self._state.is_handling_event
                and not self._is_in_cooldown(now)
            )

            if should_trigger:
                self._state.is_handling_event = True
                self._state.last_trigger_at = now
                self._state.repeat_count += 1

            self._state.closed_eye_duration = duration
            self._state.last_warning_active = warning_active
            if warning_active:
                self._state.current_level = 3 if self._state.repeat_count >= 2 else 2
                self._state.trigger_reason = f"Eyes closed for {duration:.1f}s"
            elif preview_active:
                self._state.current_level = 1
                self._state.trigger_reason = f"Eyes closed for {duration:.1f}s"
            else:
                self._state.current_level = 0
                self._state.trigger_reason = ""
            if driver_state is not None:
                synced_driver_state = dict(driver_state)
                synced_driver_state["focus_alert"] = warning_active
                synced_driver_state["closed_eye_duration"] = duration
                synced_driver_state["focus_level"] = self._state.current_level
                synced_driver_state["trigger_reason"] = self._state.trigger_reason
                if warning_active:
                    synced_driver_state["risk"] = "HIGH"
                self._state.driver_state = synced_driver_state

        if should_trigger:
            logger.info(
                "High-risk drowsiness event triggered after %.2fs of closed eyes",
                duration,
            )
            self._launch_intervention(emotion=emotion)

        return warning_active

    def _is_in_cooldown(self, now: float) -> bool:
        if self._state.last_trigger_at is None:
            return False
        return (now - self._state.last_trigger_at) < self.config.cooldown_seconds

    def _launch_intervention(self, emotion: str) -> None:
        """Run the beep -> TTS -> dialogue loop on a background thread."""
        with self._state.lock:
            chat_model = self._state.chat_model
            temperature = self._state.temperature
            driver_state = dict(self._state.driver_state) if self._state.driver_state else None
        thread = threading.Thread(
            target=self._run_intervention,
            kwargs={
                "emotion": emotion,
                "chat_model": chat_model,
                "temperature": temperature,
                "driver_state": driver_state,
            },
            daemon=True,
            name="FocusMonitor-Intervention",
        )
        thread.start()

    def _run_intervention(
        self,
        emotion: str,
        chat_model: str,
        temperature: float,
        driver_state: dict[str, Any] | None,
    ) -> None:
        try:
            _play_beep(
                self.config.beep_frequency_hz,
                self.config.beep_duration_ms,
            )

            if self._tts is not None:
                try:
                    check_in_question = _build_check_in_question(
                        self.config.check_in_question,
                        emotion,
                        driver_state,
                    )
                    self._tts.speak(
                        check_in_question,
                        emotion=emotion,
                        wait=True,
                    )
                except Exception as exc:
                    logger.exception("TTS speak failed: %s", exc)

            if self._voice_pipeline is not None:
                try:
                    result = self._voice_pipeline.process_voice_input(
                        duration_seconds=self.config.record_seconds,
                        emotion=emotion,
                        auto_trigger=False,
                        model=chat_model,
                        temperature=temperature,
                        driver_state=driver_state,
                    )
                    logger.info(
                        "Driver said: %r | Assistant: %r",
                        result.user_input,
                        result.bot_reply,
                    )
                    if self._on_voice_result is not None:
                        self._on_voice_result(result)
                except NoSpeechDetectedError as exc:
                    logger.info("Voice pipeline skipped: %s", exc)
                    if self._on_voice_error is not None:
                        self._on_voice_error(str(exc))
                except Exception as exc:
                    logger.exception("Voice pipeline failed: %s", exc)
                    if self._on_voice_error is not None:
                        self._on_voice_error(str(exc))

        finally:
            with self._state.lock:
                self._state.is_handling_event = False

    def reset(self) -> None:
        """Clear all state. Useful when the camera selection changes."""
        with self._state.lock:
            self._state.closed_eye_started_at = None
            self._state.last_trigger_at = None
            self._state.is_handling_event = False
            self._state.last_warning_active = False
            self._state.closed_eye_duration = 0.0
            self._state.current_level = 0
            self._state.trigger_reason = ""
            self._state.repeat_count = 0
