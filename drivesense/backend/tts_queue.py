from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Callable


@dataclass
class _SpeakJob:
    task: Callable[[], None]
    done: threading.Event
    error: list[BaseException]


class TTSQueue:
    """Single-consumer queue for all pyttsx3 speech work."""

    _instance: "TTSQueue | None" = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._jobs: queue.Queue[_SpeakJob] = queue.Queue()
        self._worker = threading.Thread(
            target=self._run,
            daemon=True,
            name="DriveSense-TTSQueue",
        )
        self._worker.start()

    @classmethod
    def instance(cls) -> "TTSQueue":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def submit(self, task: Callable[[], None], wait: bool = False) -> None:
        job = _SpeakJob(task=task, done=threading.Event(), error=[])
        self._jobs.put(job)
        if not wait:
            return

        job.done.wait()
        if job.error:
            raise RuntimeError("TTS job failed.") from job.error[0]

    def _run(self) -> None:
        while True:
            job = self._jobs.get()
            try:
                job.task()
            except BaseException as exc:
                job.error.append(exc)
            finally:
                job.done.set()
                self._jobs.task_done()
