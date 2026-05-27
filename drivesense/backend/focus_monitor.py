"""Drowsiness and emotion focus monitor.

Tracks two trigger paths for multi-modal intervention:

  PATH A - Drowsiness:
    Driver eyes closed >= threshold (default 2 s) -> beep + LLM TTS.

  PATH B - Emotion:
    Dangerous emotion sustained >= threshold -> beep + LLM TTS/dialogue.
    Thresholds per emotion:
      fear          -> 3 s  (HIGH risk, full dialogue)
      sad           -> 3 s  (MED risk, full dialogue)
      disgust       -> 3 s  (LOW risk, full dialogue)
      anger         -> 3 s  (HIGH risk, TTS only, no dialogue)
      surprise      -> 3 s  (MED risk, TTS only, no dialogue)
      happy/neutral -> never triggered

Both paths share a single cooldown timer so they never fire simultaneously.
The monitor is polled once per processed frame from the vision loop. Audio and
network side effects run on background threads so the camera loop never blocks.
"""

from __future__ import annotations

import logging
import platform
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from drivesense.backend.chatbot import DEFAULT_MODEL, ChatbotResponse, DriverAssistantChatbot
from drivesense.backend.speech import TTS_PRIORITY_ALERT, TextToSpeech
from drivesense.backend.voice_chat import NoSpeechDetectedError, VoiceChatPipeline

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Emotion trigger configuration
# ---------------------------------------------------------------------------

EMOTION_TRIGGER_THRESHOLDS: dict[str, float] = {
    "anger": 3.0,
    "fear": 3.0,
    "sad": 3.0,
    "disgust": 3.0,
    "surprise": 3.0,
}
"""Seconds of sustained emotion before the system intervenes."""

EMOTION_FULL_DIALOGUE: set[str] = {"disgust", "fear", "sad"}
"""Emotions that get the complete beep -> LLM TTS -> record -> dialogue loop."""

EMOTION_TTS_ONLY: set[str] = {"anger", "surprise"}
"""Emotions that get beep -> TTS only (no microphone recording / dialogue)."""

EMOTION_CHECK_IN_SENTENCES: dict[str, str] = {
    "anger": "I noticed you seem tense. Take a slow breath - the road needs your calm focus.",
    "fear": "You seem a bit uneasy. It's okay - stay steady, you've got this.",
    "sad": "You seem a little down. Would you like to take a short break when it's safe?",
    "disgust": "Try to stay relaxed and keep your attention on the road.",
    "surprise": "Stay calm and keep your eyes steady on the road ahead.",
}


@dataclass
class FocusMonitorConfig:
    """Tunable thresholds for the focus monitor."""

    closed_eye_seconds: float = 2.0
    """Continuous closed-eye duration that triggers a high-risk event."""

    cooldown_seconds: float = 10.0
    """Minimum gap between two high-risk triggers (shared by eye & emotion)."""

    beep_frequency_hz: int = 1000
    """Frequency of the alert beep."""

    beep_duration_ms: int = 400
    """Duration of the alert beep."""

    check_in_question: str = "Hey, are you feeling tired? How are you doing?"
    """The TTS question played after the beep (eye-closure path)."""

    record_seconds: float = 5.0
    """How long to listen for the driver's reply."""

    chat_model: str = DEFAULT_MODEL
    """OpenRouter model id used for the LLM reply step."""

    alert_max_output_tokens: int = 40
    """Maximum tokens for the short LLM-generated focus alert."""

    enable_emotion_trigger: bool = True
    """Enable emotion-based intervention (Path B)."""


@dataclass
class _State:
    """Internal mutable state of the monitor."""

    # --- Eye tracking (Path A) ---
    closed_eye_started_at: Optional[float] = None
    closed_eye_duration: float = 0.0

    # --- Emotion tracking (Path B) ---
    emotion_streak_label: str = "neutral"
    emotion_streak_started_at: Optional[float] = None
    emotion_streak_duration: float = 0.0

    # --- Shared trigger state ---
    last_trigger_at: Optional[float] = None
    is_handling_event: bool = False
    last_warning_active: bool = False
    last_emotion_warning_active: bool = False
    chat_model: str = DEFAULT_MODEL
    temperature: float = 1.0
    driver_state: dict[str, Any] | None = None
    current_level: int = 0
    trigger_reason: str = ""
    repeat_count: int = 0
    emotion_trigger_count: int = 0
    emotion_trigger_breakdown: dict[str, int] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)


def _play_beep(frequency_hz: int, duration_ms: int) -> None:
    """Play a short alert beep."""
    if platform.system() == "Windows":
        try:
            import winsound
            winsound.Beep(frequency_hz, duration_ms)
            return
        except Exception as exc:
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


def _build_alert_prompt(driver_state: dict[str, Any] | None) -> str:
    """Build LLM prompt for eye-closure (drowsiness) alerts."""
    closed_duration = float((driver_state or {}).get("closed_eye_duration", 0.0))
    risk = str((driver_state or {}).get("risk", "HIGH")).upper()
    return (
        "The system detected that the driver's eyes were closed long enough to trigger "
        f"a focus alert. Risk={risk}. ClosedEyeDuration={closed_duration:.1f}s. "
        "Say exactly one short, calm sentence to help the driver refocus. "
        "Do not mention being an AI. Do not ask a long question."
    )


def _build_emotion_alert_prompt(
    emotion: str,
    duration: float,
    driver_state: dict[str, Any] | None,
) -> str:
    """Build LLM prompt for emotion-based alerts."""
    risk = str((driver_state or {}).get("risk", "OK")).upper()
    return (
        f"The driver has been showing '{emotion}' emotion for {duration:.1f} seconds. "
        f"Risk={risk}. "
        "Say exactly one short, calm, supportive sentence to help the driver feel better "
        "and refocus on driving safely. "
        "Do not mention being an AI. Do not ask a long question. "
        "Match your tone to the emotion - be soothing for anger/fear, gentle for sadness."
    )


class FocusMonitor:
    """Watch the driver's eye state AND emotion, run voice interventions.

    Two trigger paths:
      A) Closed eyes >= threshold -> full intervention
      B) Dangerous emotion sustained >= threshold -> intervention
    """

    def __init__(
        self,
        config: Optional[FocusMonitorConfig] = None,
        tts: Optional[TextToSpeech] = None,
        voice_pipeline: Optional[VoiceChatPipeline] = None,
        alert_chatbot: Optional[DriverAssistantChatbot] = None,
        on_voice_result: Optional[Callable[[Any], None]] = None,
        on_voice_error: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.config = config or FocusMonitorConfig()
        self._tts = tts
        self._voice_pipeline = voice_pipeline
        self._alert_chatbot = alert_chatbot or getattr(voice_pipeline, "chatbot", None)
        self._on_voice_result = on_voice_result
        self._on_voice_error = on_voice_error
        self._state = _State()
        self._state.chat_model = self.config.chat_model
        self._voice_dialogue_enabled = True

    def set_voice_dialogue_enabled(self, enabled: bool) -> None:
        self._voice_dialogue_enabled = enabled

    def get_state_snapshot(self) -> dict[str, Any]:
        with self._state.lock:
            return {
                "closed_eye_duration": self._state.closed_eye_duration,
                "focus_level": self._state.current_level,
                "trigger_reason": self._state.trigger_reason,
                "repeat_count": self._state.repeat_count,
                "focus_alert": self._state.last_warning_active,
                # --- Emotion fields ---
                "emotion_streak_label": self._state.emotion_streak_label,
                "emotion_streak_duration": self._state.emotion_streak_duration,
                "emotion_warning_active": self._state.last_emotion_warning_active,
                "emotion_trigger_count": self._state.emotion_trigger_count,
                "emotion_trigger_breakdown": dict(self._state.emotion_trigger_breakdown),
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
        """Feed one frame's eye state and emotion into the monitor.

        Returns:
            True if the on-screen warning should currently be drawn
            (either eye-closure OR emotion warning).
        """
        if now is None:
            now = time.perf_counter()

        preview_threshold = max(self.config.closed_eye_seconds * 0.6, 0.5)
        normalized_emotion = (emotion or "neutral").strip().lower()

        with self._state.lock:
            # ---------------------------------------------------------------
            # PATH A: Eye-closure tracking
            # ---------------------------------------------------------------
            if eyes_closed:
                if self._state.closed_eye_started_at is None:
                    self._state.closed_eye_started_at = now
                eye_duration = now - self._state.closed_eye_started_at
            else:
                self._state.closed_eye_started_at = None
                eye_duration = 0.0

            eye_warning_active = eye_duration >= self.config.closed_eye_seconds
            preview_active = eye_duration >= preview_threshold and not eye_warning_active
            should_trigger_eye = (
                eye_warning_active
                and not self._state.is_handling_event
                and not self._is_in_cooldown(now)
            )

            if should_trigger_eye:
                self._state.is_handling_event = True
                self._state.last_trigger_at = now
                self._state.repeat_count += 1

            self._state.closed_eye_duration = eye_duration
            self._state.last_warning_active = eye_warning_active

            # ---------------------------------------------------------------
            # PATH B: Emotion streak tracking (NEW)
            # ---------------------------------------------------------------
            should_trigger_emotion = False
            emotion_warning_active = False

            if self.config.enable_emotion_trigger:
                if normalized_emotion == self._state.emotion_streak_label:
                    if self._state.emotion_streak_started_at is not None:
                        self._state.emotion_streak_duration = now - self._state.emotion_streak_started_at
                else:
                    self._state.emotion_streak_label = normalized_emotion
                    self._state.emotion_streak_started_at = now
                    self._state.emotion_streak_duration = 0.0

                threshold = EMOTION_TRIGGER_THRESHOLDS.get(normalized_emotion)
                if threshold is not None and self._state.emotion_streak_duration >= threshold:
                    emotion_warning_active = True
                    should_trigger_emotion = (
                        not self._state.is_handling_event
                        and not self._is_in_cooldown(now)
                        and not should_trigger_eye
                    )
                    if should_trigger_emotion:
                        self._state.is_handling_event = True
                        self._state.last_trigger_at = now
                        self._state.emotion_trigger_count += 1
                        self._state.emotion_trigger_breakdown[normalized_emotion] = (
                            self._state.emotion_trigger_breakdown.get(normalized_emotion, 0) + 1
                        )

                self._state.last_emotion_warning_active = emotion_warning_active

            # ---------------------------------------------------------------
            # Compute combined focus level & trigger reason
            # ---------------------------------------------------------------
            if eye_warning_active:
                self._state.current_level = 3 if self._state.repeat_count >= 2 else 2
                self._state.trigger_reason = f"Eyes closed for {eye_duration:.1f}s"
            elif emotion_warning_active:
                self._state.current_level = 2
                self._state.trigger_reason = (
                    f"{normalized_emotion.capitalize()} detected for "
                    f"{self._state.emotion_streak_duration:.1f}s"
                )
            elif preview_active:
                self._state.current_level = 1
                self._state.trigger_reason = f"Eyes closed for {eye_duration:.1f}s"
            else:
                emotion_preview = False
                if self.config.enable_emotion_trigger:
                    et = EMOTION_TRIGGER_THRESHOLDS.get(normalized_emotion)
                    if (
                        et is not None
                        and self._state.emotion_streak_duration >= et * 0.6
                        and self._state.emotion_streak_duration < et
                    ):
                        emotion_preview = True
                        self._state.current_level = 1
                        self._state.trigger_reason = (
                            f"{normalized_emotion.capitalize()} for "
                            f"{self._state.emotion_streak_duration:.1f}s"
                        )
                if not emotion_preview:
                    self._state.current_level = 0
                    self._state.trigger_reason = ""

            if driver_state is not None:
                synced_driver_state = dict(driver_state)
                synced_driver_state["focus_alert"] = eye_warning_active or emotion_warning_active
                synced_driver_state["closed_eye_duration"] = eye_duration
                synced_driver_state["focus_level"] = self._state.current_level
                synced_driver_state["trigger_reason"] = self._state.trigger_reason
                synced_driver_state["emotion_warning_active"] = emotion_warning_active
                synced_driver_state["emotion_streak_label"] = self._state.emotion_streak_label
                synced_driver_state["emotion_streak_duration"] = self._state.emotion_streak_duration
                if eye_warning_active or emotion_warning_active:
                    synced_driver_state["risk"] = "HIGH"
                self._state.driver_state = synced_driver_state

        # ---------------------------------------------------------------
        # Launch interventions (outside the lock)
        # ---------------------------------------------------------------
        if should_trigger_eye:
            logger.info(
                "High-risk drowsiness event triggered after %.2fs of closed eyes",
                eye_duration,
            )
            self._launch_intervention(emotion=emotion, trigger_type="eye")

        if should_trigger_emotion:
            logger.info(
                "Emotion event triggered: %s sustained for %.2fs",
                normalized_emotion,
                self._state.emotion_streak_duration,
            )
            self._launch_intervention(emotion=emotion, trigger_type="emotion")

        return eye_warning_active or emotion_warning_active

    def _is_in_cooldown(self, now: float) -> bool:
        if self._state.last_trigger_at is None:
            return False
        return (now - self._state.last_trigger_at) < self.config.cooldown_seconds

    def _launch_intervention(self, emotion: str, trigger_type: str = "eye") -> None:
        """Run the beep -> TTS -> dialogue loop on a background thread."""
        with self._state.lock:
            chat_model = self._state.chat_model
            temperature = self._state.temperature
            driver_state = dict(self._state.driver_state) if self._state.driver_state else None
            emotion_duration = self._state.emotion_streak_duration
        thread = threading.Thread(
            target=self._run_intervention,
            kwargs={
                "emotion": emotion,
                "chat_model": chat_model,
                "temperature": temperature,
                "driver_state": driver_state,
                "trigger_type": trigger_type,
                "emotion_duration": emotion_duration,
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
        trigger_type: str = "eye",
        emotion_duration: float = 0.0,
    ) -> None:
        try:
            _play_beep(
                self.config.beep_frequency_hz,
                self.config.beep_duration_ms,
            )

            normalized_emotion = (emotion or "neutral").strip().lower()
            use_dialogue = (
                trigger_type == "emotion"
                and normalized_emotion in EMOTION_FULL_DIALOGUE
            )

            # TTS alert sentence
            if self._tts is not None:
                try:
                    if trigger_type == "emotion":
                        alert_sentence = self._build_emotion_alert_sentence(
                            normalized_emotion,
                            emotion_duration,
                            chat_model,
                            temperature,
                            driver_state,
                        )
                    else:
                        alert_sentence = self._build_alert_sentence(
                            self.config.check_in_question,
                            emotion,
                            chat_model,
                            temperature,
                            driver_state,
                        )
                    self._tts.speak(
                        alert_sentence,
                        emotion=emotion,
                        wait=use_dialogue,
                        priority=TTS_PRIORITY_ALERT,
                        drop_pending_below_priority=TTS_PRIORITY_ALERT,
                    )
                except Exception as exc:
                    logger.exception("TTS speak failed: %s", exc)

            if (
                use_dialogue
                and self._voice_dialogue_enabled
                and self._voice_pipeline is not None
            ):
                try:
                    result = self._voice_pipeline.process_voice_input(
                        duration_seconds=self.config.record_seconds,
                        emotion=emotion,
                        auto_trigger=(trigger_type == "emotion"),
                        model=chat_model,
                        temperature=temperature,
                        driver_state=driver_state,
                    )
                    logger.info(
                        "[%s trigger] Driver said: %r | Assistant: %r",
                        trigger_type,
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
                if trigger_type == "emotion":
                    self._state.emotion_streak_started_at = None
                    self._state.emotion_streak_duration = 0.0

    def _build_alert_sentence(
        self,
        default_question: str,
        emotion: str,
        chat_model: str,
        temperature: float,
        driver_state: dict[str, Any] | None,
    ) -> str:
        """Build TTS sentence for eye-closure (drowsiness) alerts."""
        fallback_sentence = _build_check_in_question(
            default_question, emotion, driver_state,
        )
        if self._alert_chatbot is None:
            return fallback_sentence
        try:
            logger.info(
                "Generating dynamic focus alert via LLM | model=%s emotion=%s",
                chat_model, emotion,
            )
            response = self._alert_chatbot.generate_reply(
                emotion=emotion,
                user_message=_build_alert_prompt(driver_state),
                model=chat_model,
                temperature=temperature,
                conversation_history=[],
                max_output_tokens=self.config.alert_max_output_tokens,
                auto_trigger=True,
                driver_state=driver_state,
            )
            text = response.text.strip()
            if text:
                logger.info(
                    "Dynamic focus alert generated | model=%s fallback=%s text=%r",
                    response.model, response.fallback_used, text,
                )
                return text
        except Exception as exc:
            logger.exception("Dynamic focus alert failed, using fallback: %s", exc)
        return fallback_sentence

    def _build_emotion_alert_sentence(
        self,
        emotion: str,
        duration: float,
        chat_model: str,
        temperature: float,
        driver_state: dict[str, Any] | None,
    ) -> str:
        """Build TTS sentence for emotion-based alerts."""
        fallback_sentence = EMOTION_CHECK_IN_SENTENCES.get(
            emotion, "Please stay calm and focused on the road.",
        )
        if self._alert_chatbot is None:
            return fallback_sentence
        try:
            logger.info(
                "Generating emotion alert via LLM | model=%s emotion=%s duration=%.1fs",
                chat_model, emotion, duration,
            )
            prompt = _build_emotion_alert_prompt(emotion, duration, driver_state)
            response = self._alert_chatbot.generate_reply(
                emotion=emotion,
                user_message=prompt,
                model=chat_model,
                temperature=temperature,
                conversation_history=[],
                max_output_tokens=self.config.alert_max_output_tokens,
                auto_trigger=True,
                driver_state=driver_state,
            )
            text = response.text.strip()
            if text:
                logger.info(
                    "Emotion alert generated | model=%s emotion=%s text=%r",
                    response.model, emotion, text,
                )
                return text
        except Exception as exc:
            logger.exception("Emotion alert LLM failed, using fallback: %s", exc)
        return fallback_sentence

    def reset(self) -> None:
        """Clear all state. Useful when the camera selection changes."""
        with self._state.lock:
            self._state.closed_eye_started_at = None
            self._state.last_trigger_at = None
            self._state.is_handling_event = False
            self._state.last_warning_active = False
            self._state.last_emotion_warning_active = False
            self._state.closed_eye_duration = 0.0
            self._state.current_level = 0
            self._state.trigger_reason = ""
            self._state.repeat_count = 0
            self._state.emotion_streak_label = "neutral"
            self._state.emotion_streak_started_at = None
            self._state.emotion_streak_duration = 0.0
            self._state.emotion_trigger_count = 0
            self._state.emotion_trigger_breakdown.clear()
