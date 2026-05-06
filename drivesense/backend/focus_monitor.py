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
from typing import Optional

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
    ) -> None:
        self.config = config or FocusMonitorConfig()
        self._tts = tts
        self._voice_pipeline = voice_pipeline
        self._state = _State()

    def update(
        self,
        eyes_closed: bool,
        emotion: str = "neutral",
        now: Optional[float] = None,
    ) -> bool:
        """Feed one frame's eye state into the monitor.

        Args:
            eyes_closed: True when both eyes are currently classified closed.
            emotion: The driver's current detected emotion. Passed through to
                the TTS engine and the LLM prompt.
            now: Override the current timestamp (mostly for unit tests).

        Returns:
            True if the on-screen warning should currently be drawn.
        """
        if now is None:
            now = time.perf_counter()

        with self._state.lock:
            if eyes_closed:
                if self._state.closed_eye_started_at is None:
                    self._state.closed_eye_started_at = now
                duration = now - self._state.closed_eye_started_at
            else:
                self._state.closed_eye_started_at = None
                duration = 0.0

            warning_active = duration >= self.config.closed_eye_seconds
            should_trigger = (
                warning_active
                and not self._state.is_handling_event
                and not self._is_in_cooldown(now)
            )

            if should_trigger:
                self._state.is_handling_event = True
                self._state.last_trigger_at = now

            self._state.last_warning_active = warning_active

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
        thread = threading.Thread(
            target=self._run_intervention,
            kwargs={"emotion": emotion},
            daemon=True,
            name="FocusMonitor-Intervention",
        )
        thread.start()

    def _run_intervention(self, emotion: str) -> None:
        try:
            _play_beep(
                self.config.beep_frequency_hz,
                self.config.beep_duration_ms,
            )

            if self._tts is not None:
                try:
                    self._tts.speak(
                        self.config.check_in_question,
                        emotion=emotion,
                    )
                except Exception as exc:
                    logger.exception("TTS speak failed: %s", exc)

            if self._voice_pipeline is not None:
                try:
                    result = self._voice_pipeline.process_voice_input(
                        duration_seconds=self.config.record_seconds,
                        emotion=emotion,
                        auto_trigger=False,
                        model=self.config.chat_model,
                    )
                    logger.info(
                        "Driver said: %r | Assistant: %r",
                        result.user_input,
                        result.bot_reply,
                    )
                except Exception as exc:
                    logger.exception("Voice pipeline failed: %s", exc)

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
