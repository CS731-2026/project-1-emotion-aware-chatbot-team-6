"""Wake-word detection for hands-free voice input during driving.

Continuously listens to microphone, detects "hey moss" or similar utterances,
and triggers a 5-second voice recording session for LLM interaction.

Also provides ContinuedConversationListener for post-reply listening without
requiring wake-word re-trigger.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import sounddevice as sd
import torch

from drivesense.backend.speech import WhisperTranscriber, record_microphone_audio

logger = logging.getLogger(__name__)


@dataclass
class WakeWordConfig:
    """Configuration for wake-word detection."""

    keywords: Optional[list[str]] = None
    """Keywords to listen for, e.g. ['hey moss', 'hey', 'moss']."""

    chunk_duration_seconds: float = 1.0
    """Duration of audio to transcribe per detection chunk."""

    confidence_threshold: float = 0.6
    """Minimum confidence for keyword match (similarity score)."""

    whisper_model_size: str = "tiny"
    """Whisper model size for lightweight transcription."""

    def __post_init__(self):
        if self.keywords is None:
            self.keywords = ["hey moss", "hey", "moss"]


class SimpleStringMatcher:
    """Simple substring/fuzzy matcher for wake words."""

    @staticmethod
    def match(text: str, keyword: str, threshold: float = 0.6) -> bool:
        """Check if keyword appears in text (simple substring + basic similarity)."""
        text_lower = text.lower().strip()
        keyword_lower = keyword.lower().strip()

        # Exact substring match (highest confidence).
        if keyword_lower in text_lower:
            return True

        # Fuzzy: check if most characters of keyword appear in order.
        text_idx = 0
        matched_chars = 0
        for char in keyword_lower:
            while text_idx < len(text_lower) and text_lower[text_idx] != char:
                text_idx += 1
            if text_idx < len(text_lower):
                matched_chars += 1
                text_idx += 1
        fuzzy_score = matched_chars / len(keyword_lower) if keyword_lower else 0
        return fuzzy_score >= threshold


class WakeWordListener:
    """Listen continuously and detect wake words to trigger voice recording."""

    def __init__(
        self,
        config: Optional[WakeWordConfig] = None,
        on_detected: Optional[Callable[[], None]] = None,
    ) -> None:
        self.config = config or WakeWordConfig()
        self.on_detected = on_detected
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Lazy-load transcriber on first use.
        self._transcriber: Optional[WhisperTranscriber] = None

    def get_transcriber(self) -> WhisperTranscriber:
        """Get or create the Whisper transcriber (lazy load)."""
        if self._transcriber is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._transcriber = WhisperTranscriber(
                model_size=self.config.whisper_model_size,
                device=device,
            )
        return self._transcriber

    def start(self) -> None:
        """Start the background wake-word listener thread."""
        with self._lock:
            if self._running:
                return
            self._running = True

        self._thread = threading.Thread(target=self._listen_loop, daemon=True, name="WakeWordListener")
        self._thread.start()
        logger.info(f"Wake-word listener started. Listening for: {self.config.keywords}")

    def stop(self) -> None:
        """Stop the background listener thread."""
        with self._lock:
            if not self._running:
                return
            self._running = False

        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.info("Wake-word listener stopped.")

    def _listen_loop(self) -> None:
        """Main listening loop: continuously record and transcribe short chunks.
        Will recover from transient errors by sleeping and retrying instead of exiting."""
        matcher = SimpleStringMatcher()

        # Keep the outer loop running while the listener is marked as running.
        while self._running:
            try:
                transcriber = self.get_transcriber()

                while self._running:
                    try:
                        # Record a short chunk (e.g., 1 second) to transcribe.
                        # VAD disabled: short fixed chunks, not user speech input.
                        audio = record_microphone_audio(
                            duration_seconds=self.config.chunk_duration_seconds,
                            sample_rate=16000,
                            channels=1,
                            vad_enabled=False,
                        )

                        if audio.size == 0:
                            # No audio captured; continue.
                            continue

                        # Transcribe the chunk with Whisper.
                        result = transcriber.transcribe_audio(audio, sample_rate=16000)
                        transcribed_text = result.text.strip()

                        if transcribed_text:
                            logger.debug(f"[Wake-word] Transcribed chunk: '{transcribed_text}'")

                            # Check if any keyword matches.
                            for keyword in self.config.keywords:
                                if matcher.match(
                                    transcribed_text,
                                    keyword,
                                    threshold=self.config.confidence_threshold,
                                ):
                                    logger.info(f"[Wake-word] Detected: '{keyword}' in '{transcribed_text}'")
                                    if self.on_detected:
                                        try:
                                            self.on_detected()
                                        except Exception:
                                            logger.exception("[Wake-word] Exception in on_detected handler")
                                    break

                    except Exception as exc:
                        logger.exception(f"[Wake-word] Error during chunk transcription: {exc}")
                        # Short sleep to avoid tight error loops on repeated failures.
                        time.sleep(0.1)

            except Exception as exc:
                # Log unexpected errors (e.g., transcriber load failure) and retry after a pause.
                logger.exception(f"[Wake-word] Listener encountered fatal error, will retry in 1s: {exc}")
                time.sleep(1.0)

        logger.debug("[Wake-word] Listener loop exited.")


class ContinuedConversationListener:
    """Listen for follow-up user input after LLM reply without requiring wake-word.
    
    Auto-triggers 5-second recording if user speaks within timeout window.
    Falls back to wake-word listening when timeout expires.
    """

    def __init__(
        self,
        config: Optional[WakeWordConfig] = None,
        on_voice_detected: Optional[Callable[[], None]] = None,
        on_timeout: Optional[Callable[[], None]] = None,
        timeout_seconds: float = 10.0,
    ) -> None:
        self.config = config or WakeWordConfig()
        self.on_voice_detected = on_voice_detected
        self.on_timeout = on_timeout
        self.timeout_seconds = timeout_seconds
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._transcriber: Optional[WhisperTranscriber] = None

    def get_transcriber(self) -> WhisperTranscriber:
        """Get or create the Whisper transcriber (lazy load)."""
        if self._transcriber is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._transcriber = WhisperTranscriber(
                model_size=self.config.whisper_model_size,
                device=device,
            )
        return self._transcriber

    def start(self) -> None:
        """Start the continued conversation listener thread."""
        with self._lock:
            if self._running:
                return
            self._running = True

        self._thread = threading.Thread(
            target=self._listen_loop, daemon=True, name="ContinuedConversationListener"
        )
        self._thread.start()
        logger.info(f"[Continued] Listening for follow-up input (timeout: {self.timeout_seconds}s)")

    def stop(self) -> None:
        """Stop the continued conversation listener."""
        with self._lock:
            if not self._running:
                return
            self._running = False

        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.info("[Continued] Listener stopped.")

    def _listen_loop(self) -> None:
        """仅用 RMS 能量检测用户是否开口，不跑 Whisper。"""
        VAD_RMS_THRESHOLD = 0.018   # 根据环境噪音可调
        CHUNK_SECONDS = 0.4          # 每 400ms 采一次，响应快

        while self._running:
            try:
                start_time = time.time()
                timed_out = False

                while self._running:
                    if time.time() - start_time > self.timeout_seconds:
                        timed_out = True
                        self._running = False
                        break

                    try:
                        audio = record_microphone_audio(
                            duration_seconds=CHUNK_SECONDS,
                            sample_rate=16000,
                            channels=1,
                            vad_enabled=False,
                        )
                        if audio.size == 0:
                            continue

                        rms = float(np.sqrt(np.mean(audio ** 2)))
                        logger.debug(f"[Continued] RMS={rms:.4f}")

                        if rms > VAD_RMS_THRESHOLD:
                            logger.info(f"[Continued] Voice detected (rms={rms:.4f}), triggering recording")
                            self._running = False
                            if self.on_voice_detected:
                                try:
                                    self.on_voice_detected()
                                except Exception:
                                    logger.exception("[Continued] on_voice_detected error")
                            break

                    except Exception as exc:
                        logger.exception(f"[Continued] chunk error: {exc}")
                        time.sleep(0.1)

                if timed_out and self.on_timeout:
                    try:
                        self.on_timeout()
                    except Exception:
                        logger.exception("[Continued] on_timeout error")

            except Exception as exc:
                logger.exception(f"[Continued] fatal error, retry 1s: {exc}")
                time.sleep(1.0)

        logger.debug("[Continued] loop exited")