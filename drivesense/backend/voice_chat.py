"""Integrated voice chat pipeline: record → transcribe → LLM reply → speak."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Callable

from drivesense.backend.chatbot import ChatbotResponse, DriverAssistantChatbot
from drivesense.backend.speech import TextToSpeech, WhisperTranscriber, record_microphone_audio


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
    """Complete voice chat: record → transcribe → chat → speak."""
    
    def __init__(
        self,
        chatbot: DriverAssistantChatbot,
        whisper_model_size: str = "base",
        tts_rate: int = 150,
        tts_volume: float = 1.0,
    ) -> None:
        self.chatbot = chatbot
        self.transcriber = WhisperTranscriber(model_size=whisper_model_size)
        self.tts = TextToSpeech(rate=tts_rate, volume=tts_volume)
    
    def process_voice_input(
        self,
        duration_seconds: float = 5.0,
        emotion: str = "neutral",
        temperature: float = 1.0,
        conversation_history: list[dict[str, str]] | None = None,
        auto_trigger: bool = False,
        model: str | None = None,
        driver_state: dict[str, Any] | None = None,
    ) -> VoiceChatResult:
        """
        Record voice, transcribe, get LLM reply, and speak it aloud.
        
        Returns:
            VoiceChatResult with transcribed user input, bot reply, and metrics.
        """
        # Step 1: Record and transcribe
        print(f"Recording for {duration_seconds} seconds...")
        audio = record_microphone_audio(duration_seconds=duration_seconds)
        
        if audio.size == 0:
            raise ValueError("No audio was captured. Please try again.")
        
        result = self.transcriber.transcribe_audio(audio)
        user_input = result.text
        print(f"Transcribed: {user_input}")
        if not user_input.strip():
            raise NoSpeechDetectedError("No speech detected.")
        
        # Step 2: Get LLM reply
        print("Generating response...")
        bot_response: ChatbotResponse = self.chatbot.generate_reply(
            emotion=emotion,
            user_message=user_input,
            model=model,
            temperature=temperature,
            conversation_history=conversation_history,
            auto_trigger=auto_trigger,
            driver_state=driver_state,
        )
        
        # Step 3: Queue the reply TTS without blocking the result path.
        print("Queueing reply TTS...")
        self.tts.speak(bot_response.text, emotion=emotion, wait=False)
        
        return VoiceChatResult(
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
    
    def process_voice_input_async(
        self,
        on_success: Callable[[VoiceChatResult], None],
        on_error: Callable[[str], None],
        **kwargs,
    ) -> None:
        """Process voice input in a background thread."""
        def worker():
            try:
                result = self.process_voice_input(**kwargs)
                on_success(result)
            except Exception as e:
                on_error(str(e))
        
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
