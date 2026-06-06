"""Integrated voice chat pipeline: record -> transcribe -> LLM reply -> speak."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Callable

from drivesense.backend.chatbot import (
    DEFAULT_MODEL,
    ChatbotResponse,
    DriverAssistantChatbot,
)
from drivesense.backend.speech import (
    TextToSpeech,
    TTS_PRIORITY_VOICE_REPLY,
    VoiceIOGate,
    WhisperTranscriber,
    is_supported_zh_en_text,
    record_microphone_audio,
)


class NoSpeechDetectedError(ValueError):
    """Raised when speech recording completes but no useful text is transcribed."""


@dataclass
class VoiceChatResult:
    """Result of a voice chat interaction."""

    user_input: str
    bot_reply: str
    emotion: str
    model: str
    selected_model: str
    latency_ms: float
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    fallback_used: bool


class VoiceChatPipeline:
    """Run a serialized record -> transcribe -> chat -> speak dialogue."""

    _pipeline_lock = threading.Lock()

    def __init__(
        self,
        chatbot: DriverAssistantChatbot,
        whisper_model_size: str = "base",
        tts_rate: int = 150,
        tts_volume: float = 1.0,
    ) -> None:
        self.chatbot = chatbot
        self.transcriber = WhisperTranscriber(
            model_size=whisper_model_size,
            device="cpu",
            compute_type="int8",
        )
        self.tts = TextToSpeech(rate=tts_rate, volume=tts_volume)

    def process_voice_input(
        self,
        duration_seconds: float = 5.0,
        follow_up_seconds: float = 0.0,
        emotion: str = "neutral",
        temperature: float = 1.0,
        conversation_history: list[dict[str, str]] | None = None,
        auto_trigger: bool = False,
        model: str | None = None,
        driver_state: dict[str, Any] | None = None,
        on_user_input: Callable[[str], None] | None = None,
        on_turn: Callable[[VoiceChatResult], None] | None = None,
        on_status: Callable[[str], None] | None = None,
        wait_for_tts_idle: bool = True,
    ) -> VoiceChatResult:
        """
        Record voice, transcribe, get LLM reply, and speak it aloud.

        Returns:
            VoiceChatResult with transcribed user input, bot reply, and metrics.
        """
        # The pipeline lock rejects duplicate calls to this object; the shared
        # session lock also blocks competing voice flows elsewhere in the app.
        if not type(self)._pipeline_lock.acquire(blocking=False):
            raise RuntimeError("Another voice interaction is already running.")
        if not VoiceIOGate.acquire_session(blocking=False):
            type(self)._pipeline_lock.release()
            raise RuntimeError("Another voice session is already active.")

        try:
            turn_results: list[VoiceChatResult] = []
            current_duration = duration_seconds
            current_history = list(conversation_history or [])

            while True:
                if wait_for_tts_idle and not VoiceIOGate.wait_until_tts_idle(timeout=3.0):
                    raise NoSpeechDetectedError("Voice input skipped while TTS is active.")
                if on_status is not None:
                    on_status(f"Recording for {current_duration:.0f} s...")
                print(f"Recording for {current_duration} seconds...")
                audio = record_microphone_audio(
                    duration_seconds=current_duration,
                    vad_enabled=not auto_trigger,
                )

                if audio.size == 0:
                    if turn_results:
                        if on_status is not None:
                            on_status("No follow-up speech detected")
                        break
                    raise ValueError("No audio was captured. Please try again.")
                if VoiceIOGate.is_tts_active():
                    if turn_results:
                        if on_status is not None:
                            on_status("No follow-up speech detected")
                        break
                    raise NoSpeechDetectedError("No speech detected.")

                result = self.transcriber.transcribe_audio(audio)
                user_input = result.text.strip()
                print(f"Transcribed: {user_input}")
                if not user_input:
                    if turn_results:
                        if on_status is not None:
                            on_status("No follow-up speech detected")
                        break
                    raise NoSpeechDetectedError("No speech detected.")
                if not is_supported_zh_en_text(user_input):
                    raise ValueError("Only Chinese and English voice input is supported.")
                if on_user_input is not None:
                    # Publish transcription before the remote LLM call so the
                    # driver sees their input immediately.
                    on_user_input(user_input)

                print("Generating response...")
                bot_response: ChatbotResponse = self.chatbot.generate_reply(
                    emotion=emotion,
                    user_message=user_input,
                    model=model or DEFAULT_MODEL,
                    temperature=temperature,
                    conversation_history=current_history,
                    auto_trigger=auto_trigger,
                    driver_state=driver_state,
                )

                current_history.append({"role": "user", "content": user_input})
                current_history.append({"role": "assistant", "content": bot_response.text})

                turn_result = VoiceChatResult(
                    user_input=user_input,
                    bot_reply=bot_response.text,
                    emotion=bot_response.emotion,
                    model=bot_response.model,
                    selected_model=bot_response.selected_model,
                    latency_ms=bot_response.latency_ms,
                    prompt_tokens=bot_response.prompt_tokens,
                    completion_tokens=bot_response.completion_tokens,
                    total_tokens=bot_response.total_tokens,
                    fallback_used=bot_response.fallback_used,
                )
                turn_results.append(turn_result)
                if on_turn is not None:
                    on_turn(turn_result)

                print("Queueing reply TTS...")
                self.tts.speak(
                    bot_response.text,
                    emotion=emotion,
                    # Automatic dialogue must finish speaking before opening
                    # the next follow-up microphone window.
                    wait=auto_trigger,
                    priority=TTS_PRIORITY_VOICE_REPLY,
                )
                if auto_trigger:
                    VoiceIOGate.clear_tts_active()

                if not auto_trigger or follow_up_seconds <= 0:
                    break
                # Every successful automatic turn opens another short window;
                # the loop ends only when that window contains no speech.
                current_duration = follow_up_seconds
                wait_for_tts_idle = False
                if on_status is not None:
                    on_status(f"Listening for follow-up ({follow_up_seconds:.0f} s)...")

            if not turn_results:
                raise NoSpeechDetectedError("No speech detected.")
            return turn_results[-1]
        finally:
            VoiceIOGate.release_session()
            type(self)._pipeline_lock.release()

    def process_voice_input_async(
        self,
        on_success: Callable[[VoiceChatResult], None],
        on_error: Callable[[str], None],
        **kwargs,
    ) -> None:
        """Process voice input in a background thread."""

        def worker() -> None:
            try:
                result = self.process_voice_input(**kwargs)
                on_success(result)
            except Exception as exc:
                on_error(str(exc))

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
