from __future__ import annotations

import heapq
import threading
import time
from dataclasses import dataclass
from typing import Callable


@dataclass
class _SpeakJob:
    task: Callable[[], None]
    done: threading.Event
    error: list[BaseException]
    on_done: Callable[[], None] | None = None
    on_error: Callable[[BaseException], None] | None = None
    priority: int = 0
    seq: int = 0
    cancelled: bool = False


class TTSQueue:
    """Single-consumer priority queue for all pyttsx3 speech work."""

    _instance: "TTSQueue | None" = None
    _instance_lock = threading.Lock()
    DISPATCH_GRACE_SECONDS = 0.05

    def __init__(self) -> None:
        self._jobs: list[tuple[int, int, _SpeakJob]] = []
        self._seq = 0
        self._muted = False
        self._cv = threading.Condition()
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

    def is_muted(self) -> bool:
        with self._cv:
            return self._muted

    def set_muted(self, muted: bool, clear_pending: bool = True) -> None:
        removed_jobs: list[_SpeakJob] = []
        with self._cv:
            self._muted = muted
            if muted and clear_pending and self._jobs:
                removed_jobs = [job for _, _, job in self._jobs]
                self._jobs.clear()
                heapq.heapify(self._jobs)
            self._cv.notify_all()

        for removed_job in removed_jobs:
            removed_job.cancelled = True
            removed_job.done.set()

    def submit(
        self,
        task: Callable[[], None],
        wait: bool = False,
        on_done: Callable[[], None] | None = None,
        on_error: Callable[[BaseException], None] | None = None,
        priority: int = 0,
        drop_pending_below_priority: int | None = None,
    ) -> None:
        removed_jobs: list[_SpeakJob] = []
        job = _SpeakJob(
            task=task,
            done=threading.Event(),
            error=[],
            on_done=on_done,
            on_error=on_error,
            priority=priority,
        )
        with self._cv:
            if self._muted:
                job.done.set()
                return

            job.seq = self._seq
            self._seq += 1

            if drop_pending_below_priority is not None and self._jobs:
                kept: list[tuple[int, int, _SpeakJob]] = []
                for sort_priority, seq, pending_job in self._jobs:
                    pending_priority = -sort_priority
                    if pending_priority < drop_pending_below_priority:
                        removed_jobs.append(pending_job)
                    else:
                        kept.append((sort_priority, seq, pending_job))
                self._jobs = kept
                heapq.heapify(self._jobs)

            heapq.heappush(self._jobs, (-priority, job.seq, job))
            self._cv.notify()

        for removed_job in removed_jobs:
            self._cancel_job(
                removed_job,
                RuntimeError("TTS job was dropped by a higher-priority request."),
            )

        if not wait:
            return

        job.done.wait()
        if job.error:
            raise RuntimeError("TTS job failed.") from job.error[0]

    def _cancel_job(self, job: _SpeakJob, exc: BaseException) -> None:
        job.cancelled = True
        job.error.append(exc)
        if job.on_error is not None:
            try:
                job.on_error(exc)
            except BaseException:
                pass
        job.done.set()

    def _run(self) -> None:
        while True:
            with self._cv:
                while not self._jobs:
                    self._cv.wait()

                deadline = time.monotonic() + self.DISPATCH_GRACE_SECONDS
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self._cv.wait(timeout=remaining)

                _, _, job = heapq.heappop(self._jobs)

            if job.cancelled:
                continue

            with self._cv:
                muted = self._muted
            if muted:
                job.done.set()
                continue

            try:
                job.task()
            except BaseException as exc:
                job.error.append(exc)
                if job.on_error is not None:
                    try:
                        job.on_error(exc)
                    except BaseException:
                        pass
            finally:
                if not job.error and job.on_done is not None:
                    try:
                        job.on_done()
                    except BaseException:
                        pass
                job.done.set()
