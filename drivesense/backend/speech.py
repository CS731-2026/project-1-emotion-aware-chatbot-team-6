from __future__ import annotations

import argparse
import tempfile
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sounddevice as sd
import torch
from faster_whisper import WhisperModel


@dataclass
class TranscriptionResult:
    text: str
    language: str | None
    duration_seconds: float
    audio_path: str


def record_microphone_audio(
    duration_seconds: float = 5.0,
    sample_rate: int = 16000,
    channels: int = 1,
    stop_event: threading.Event | None = None,
) -> np.ndarray:
    chunks: list[np.ndarray] = []

    def callback(indata, frames, time_info, status) -> None:
        if status:
            print(f"Audio status: {status}")
        chunks.append(indata.copy())

    with sd.InputStream(
        samplerate=sample_rate,
        channels=channels,
        dtype="float32",
        callback=callback,
    ):
        started_at = time.perf_counter()
        while time.perf_counter() - started_at < duration_seconds:
            if stop_event and stop_event.is_set():
                break
            sd.sleep(50)

    if not chunks:
        return np.empty((0,), dtype=np.float32)

    audio = np.concatenate(chunks, axis=0).squeeze()
    return audio.astype(np.float32)


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
