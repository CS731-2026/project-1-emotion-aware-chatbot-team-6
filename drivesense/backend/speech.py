from __future__ import annotations

import argparse
import base64
import platform
import subprocess
import tempfile
import threading
import time
import unicodedata
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import sounddevice as sd
import torch
from faster_whisper import WhisperModel

from drivesense.backend.tts_queue import TTSQueue

TTS_PRIORITY_CHAT = 10
TTS_PRIORITY_VOICE_REPLY = 50
TTS_PRIORITY_ALERT = 100
TTS_CAPTURE_GUARD_SECONDS = 0.8


@dataclass
class TranscriptionResult:
    text: str
    language: str | None
    duration_seconds: float
    audio_path: str


class VoiceIOGate:
    """Global guard for microphone access and full voice sessions."""

    _mic_lock = threading.Lock()
    _session_lock = threading.Lock()
    _tts_state_lock = threading.Lock()
    _tts_active_until = 0.0

    @classmethod
    def acquire_microphone(
        cls,
        blocking: bool = True,
        timeout: float | None = None,
    ) -> bool:
        if not blocking:
            return cls._mic_lock.acquire(blocking=False)
        if timeout is None:
            return cls._mic_lock.acquire(blocking=blocking)
        return cls._mic_lock.acquire(blocking=blocking, timeout=timeout)

    @classmethod
    def release_microphone(cls) -> None:
        if cls._mic_lock.locked():
            cls._mic_lock.release()

    @classmethod
    def acquire_session(
        cls,
        blocking: bool = True,
        timeout: float | None = None,
    ) -> bool:
        if not blocking:
            return cls._session_lock.acquire(blocking=False)
        if timeout is None:
            return cls._session_lock.acquire(blocking=blocking)
        return cls._session_lock.acquire(blocking=blocking, timeout=timeout)

    @classmethod
    def release_session(cls) -> None:
        if cls._session_lock.locked():
            cls._session_lock.release()

    @classmethod
    def mark_tts_started(cls) -> None:
        with cls._tts_state_lock:
            cls._tts_active_until = float("inf")

    @classmethod
    def mark_tts_finished(cls, guard_seconds: float = TTS_CAPTURE_GUARD_SECONDS) -> None:
        with cls._tts_state_lock:
            cls._tts_active_until = time.monotonic() + guard_seconds

    @classmethod
    def is_tts_active(cls) -> bool:
        with cls._tts_state_lock:
            return time.monotonic() < cls._tts_active_until

    @classmethod
    def wait_until_tts_idle(
        cls,
        timeout: float | None = None,
        stop_event: threading.Event | None = None,
    ) -> bool:
        started_at = time.monotonic()
        while cls.is_tts_active():
            if stop_event is not None and stop_event.is_set():
                return False
            if timeout is not None and time.monotonic() - started_at >= timeout:
                return False
            time.sleep(0.05)
        return True


def detect_text_language(text: str) -> str | None:
    stripped = (text or "").strip()
    if not stripped:
        return None
    cjk_count = sum(1 for char in stripped if "\u4e00" <= char <= "\u9fff")
    latin_count = sum(1 for char in stripped if char.isascii() and char.isalpha())
    if cjk_count > 0:
        return "zh"
    if latin_count > 0:
        return "en"
    return None


def is_supported_zh_en_text(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return False
    if detect_text_language(stripped) is None:
        return False

    for char in stripped:
        if char.isspace():
            continue
        if "\u4e00" <= char <= "\u9fff":
            continue
        if char.isascii():
            continue
        category = unicodedata.category(char)
        if category.startswith("P") or category.startswith("S"):
            continue
        return False
    return True


def record_microphone_audio(
    duration_seconds: float = 5.0,
    sample_rate: int = 16000,
    channels: int = 1,
    stop_event: threading.Event | None = None,
    vad_enabled: bool = True,
    min_duration_seconds: float = 0.5,
    silence_duration_seconds: float = 1.0,
    wait_for_lock: bool = True,
    lock_timeout_seconds: float | None = None,
) -> np.ndarray:
    """Record audio from microphone with optional voice activity detection (VAD).
    
    Args:
        duration_seconds: Maximum recording duration in seconds.
        sample_rate: Audio sample rate (Hz).
        channels: Number of audio channels.
        stop_event: Threading event to signal early stop.
        vad_enabled: Enable intelligent stop on prolonged silence.
        min_duration_seconds: Minimum recording duration before VAD can trigger.
        silence_duration_seconds: Duration of silence (seconds) before auto-stop.
    
    Returns:
        Recorded audio as numpy array.
    """
    if wait_for_lock:
        if not VoiceIOGate.wait_until_tts_idle(
            timeout=lock_timeout_seconds,
            stop_event=stop_event,
        ):
            return np.empty((0,), dtype=np.float32)
    elif VoiceIOGate.is_tts_active():
        return np.empty((0,), dtype=np.float32)

    if not VoiceIOGate.acquire_microphone(
        blocking=wait_for_lock,
        timeout=lock_timeout_seconds,
    ):
        return np.empty((0,), dtype=np.float32)

    chunks: list[np.ndarray] = []

    def callback(indata, frames, time_info, status) -> None:
        if status:
            print(f"Audio status: {status}")
        chunks.append(indata.copy())

    try:
        with sd.InputStream(
            samplerate=sample_rate,
            channels=channels,
            dtype="float32",
            callback=callback,
        ):
            started_at = time.perf_counter()
            last_voice_time = started_at  # Track when we last detected voice
            vad_threshold = 0.02  # Energy threshold for voice detection (RMS)
            chunk_duration = 0.05  # Process every 50ms for responsiveness

            while time.perf_counter() - started_at < duration_seconds:
                if stop_event and stop_event.is_set():
                    break

                elapsed = time.perf_counter() - started_at

                # Check for prolonged silence if VAD is enabled.
                if vad_enabled and elapsed >= min_duration_seconds:
                    if chunks:
                        recent_chunk = chunks[-1]
                        rms_energy = np.sqrt(np.mean(recent_chunk**2))

                        if rms_energy > vad_threshold:
                            last_voice_time = time.perf_counter()
                        else:
                            silence_duration = time.perf_counter() - last_voice_time
                            if silence_duration >= silence_duration_seconds:
                                print(
                                    f"[VAD] Detected {silence_duration:.1f}s of silence; "
                                    f"stopping early at {elapsed:.1f}s"
                                )
                                break

                sd.sleep(int(chunk_duration * 1000))

        if not chunks:
            return np.empty((0,), dtype=np.float32)

        audio = np.concatenate(chunks, axis=0).squeeze()
        return audio.astype(np.float32)
    finally:
        VoiceIOGate.release_microphone()


def save_wav(audio: np.ndarray, sample_rate: int, output_path: Path) -> None:
    audio_int16 = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio_int16 * 32767).astype(np.int16)
    with wave.open(str(output_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(audio_int16.tobytes())


class WhisperTranscriber:
    def __init__(
        self,
        model_size: str = "base",
        device: str | None = None,
        compute_type: str | None = None,
    ) -> None:
        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if compute_type is None:
            compute_type = "float16" if resolved_device == "cuda" else "int8"

        self.model = WhisperModel(
            model_size,
            device=resolved_device,
            compute_type=compute_type,
        )

    def transcribe_audio(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: str | None = None,
    ) -> TranscriptionResult:
        if audio.size == 0:
            raise ValueError("No audio was captured.")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            temp_path = Path(tmp_file.name)
        save_wav(audio, sample_rate, temp_path)

        segments, info = self.model.transcribe(
            str(temp_path),
            language=language,
            vad_filter=True,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        return TranscriptionResult(
            text=text,
            language=getattr(info, "language", None),
            duration_seconds=audio.shape[0] / sample_rate,
            audio_path=str(temp_path),
        )


class TextToSpeech:
    """Convert text to speech using a local engine."""

    _engine_lock = threading.RLock()
    _shared_engine: Any = None
    _shared_voices: list[Any] | None = None

    EMOTION_RATE_OFFSETS = {
        "anger": -6,
        "fear": -10,
        "sad": -12,
        "happy": 8,
        "surprise": 6,
        "disgust": -4,
        "neutral": 0,
    }

    EMOTION_VOLUME = {
        "anger": 0.9,
        "fear": 0.88,
        "sad": 0.85,
        "happy": 1.0,
        "surprise": 0.98,
        "disgust": 0.9,
        "neutral": 0.95,
    }

    def __init__(
        self,
        rate: int = 150,
        volume: float = 1.0,
    ) -> None:
        self.rate = rate
        self.volume = volume
        self._use_windows_sapi = platform.system() == "Windows"
        self._pyttsx3 = None
        if not self._use_windows_sapi:
            try:
                import pyttsx3
            except ImportError as exc:  # pragma: no cover - environment specific
                raise RuntimeError(
                    "pyttsx3 is required for voice output. Install it with pip."
                ) from exc
            self._pyttsx3 = pyttsx3

    def _get_engine(self) -> Any:
        cls = type(self)
        with cls._engine_lock:
            if cls._shared_engine is None:
                if self._pyttsx3 is None:
                    raise RuntimeError("pyttsx3 is unavailable in this environment.")
                cls._shared_engine = self._pyttsx3.init()
            return cls._shared_engine

    @classmethod
    def _reset_shared_engine(cls) -> None:
        with cls._engine_lock:
            engine = cls._shared_engine
            cls._shared_engine = None
            cls._shared_voices = None
            if engine is not None:
                try:
                    engine.stop()
                except Exception:
                    pass

    def _get_voices(self) -> list[Any]:
        cls = type(self)
        with cls._engine_lock:
            if cls._shared_voices is None:
                engine = self._get_engine()
                cls._shared_voices = list(engine.getProperty("voices") or [])
            return cls._shared_voices

    @staticmethod
    def prepare_spoken_text(text: str) -> str:
        """Keep the LLM reply unchanged except for normalizing whitespace."""
        return " ".join((text or "").split())

    def select_voice(self, emotion: str | None = None) -> str | None:
        """Prefer a softer or livelier voice depending on the driver emotion."""
        if self._use_windows_sapi:
            return None
        try:
            voices = self._get_voices()
        except Exception:
            return None

        if not voices:
            return None

        emotion_key = (emotion or "neutral").lower()
        preference_keywords = {
            "anger": ["female", "zira", "susan", "eva", "hazel"],
            "fear": ["female", "zira", "susan", "eva", "hazel"],
            "sad": ["female", "zira", "susan", "eva", "hazel"],
            "happy": ["male", "david", "mark", "alex", "richard"],
            "surprise": ["male", "david", "mark", "alex", "richard"],
            "disgust": ["female", "zira", "susan", "eva", "hazel"],
            "neutral": ["female", "male"],
        }

        keywords = preference_keywords.get(emotion_key, ["female", "male"])
        fallback_voice = voices[0]
        for keyword in keywords:
            for voice in voices:
                voice_text = f"{getattr(voice, 'name', '')} {getattr(voice, 'id', '')}".lower()
                if keyword in voice_text:
                    return getattr(voice, "id", None)

        return getattr(fallback_voice, "id", None)

    def _speak_with_windows_sapi(self, text: str, emotion: str | None = None) -> None:
        emotion_key = (emotion or "neutral").lower()
        emotion_rate_offset = self.EMOTION_RATE_OFFSETS.get(emotion_key, 0)
        base_rate = int(round((self.rate - 150) / 15)) + 2
        sapi_rate = max(-10, min(10, base_rate + int(round(emotion_rate_offset / 4))))
        volume_percent = max(
            0,
            min(100, int(round(self.EMOTION_VOLUME.get(emotion_key, self.volume) * 100))),
        )
        encoded_text = base64.b64encode(text.encode("utf-8")).decode("ascii")
        script = "\n".join(
            [
                "Add-Type -AssemblyName System.Speech",
                "$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer",
                f"$synth.Rate = {sapi_rate}",
                f"$synth.Volume = {volume_percent}",
                f"$bytes = [System.Convert]::FromBase64String('{encoded_text}')",
                "$text = [System.Text.Encoding]::UTF8.GetString($bytes)",
                "$synth.Speak($text)",
                "$synth.Dispose()",
            ]
        )
        subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                script,
            ],
            check=True,
            timeout=max(15, min(60, len(text) // 8 + 8)),
            capture_output=True,
            text=True,
        )

    def _speak_now(self, text: str, emotion: str | None = None) -> None:
        spoken_text = self.prepare_spoken_text(text)
        if not spoken_text:
            return

        VoiceIOGate.mark_tts_started()
        try:
            if self._use_windows_sapi:
                self._speak_with_windows_sapi(spoken_text, emotion=emotion)
                return

            engine = self._get_engine()
            emotion_key = (emotion or "neutral").lower()
            emotion_rate = self.EMOTION_RATE_OFFSETS.get(emotion_key, 0)
            emotion_volume = self.EMOTION_VOLUME.get(emotion_key, self.volume)
            voice_id = self.select_voice(emotion)

            try:
                try:
                    engine.stop()
                except Exception:
                    pass

                if voice_id:
                    engine.setProperty("voice", voice_id)
                engine.setProperty("rate", self.rate + emotion_rate)
                engine.setProperty("volume", emotion_volume)
                engine.say(spoken_text)
                engine.runAndWait()
            except Exception:
                type(self)._reset_shared_engine()
                raise
            finally:
                try:
                    engine.stop()
                except Exception:
                    pass
        finally:
            VoiceIOGate.mark_tts_finished()

    def speak(
        self,
        text: str,
        emotion: str | None = None,
        wait: bool = False,
        on_done: Callable[[], None] | None = None,
        on_error: Callable[[BaseException], None] | None = None,
        priority: int = TTS_PRIORITY_CHAT,
        drop_pending_below_priority: int | None = None,
    ) -> None:
        """Queue speech on the shared TTS worker thread."""
        spoken_text = self.prepare_spoken_text(text)
        if not spoken_text:
            return

        TTSQueue.instance().submit(
            lambda: self._speak_now(spoken_text, emotion=emotion),
            wait=wait,
            on_done=on_done,
            on_error=on_error,
            priority=priority,
            drop_pending_below_priority=drop_pending_below_priority,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record microphone audio locally and transcribe it with faster-whisper."
    )
    parser.add_argument("--duration", type=float, default=5.0, help="Recording length.")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Audio sample rate.")
    parser.add_argument("--model-size", type=str, default="base", help="Whisper model size.")
    parser.add_argument("--language", type=str, default=None, help="Optional language hint.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("Recording...")
    audio = record_microphone_audio(
        duration_seconds=args.duration,
        sample_rate=args.sample_rate,
    )
    transcriber = WhisperTranscriber(model_size=args.model_size)
    result = transcriber.transcribe_audio(
        audio,
        sample_rate=args.sample_rate,
        language=args.language,
    )
    print(f"Transcript: {result.text}")
    print(f"Language: {result.language}")
    print(f"Saved WAV: {result.audio_path}")


if __name__ == "__main__":
    main()
