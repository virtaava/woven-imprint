"""Per-character background worker — serialized bookkeeping off the hot path.

One daemon thread per Character. Tasks run in submission order, so
DB writes stay ordered per character. Failures are counted per label
and logged at WARNING (never silently dropped).
"""

from __future__ import annotations

import queue
import threading
import time

from .log import logger

_STOP = object()


class BackgroundWorker:
    def __init__(self, name: str = ""):
        self._queue: queue.Queue = queue.Queue()
        self.success_counts: dict[str, int] = {}
        self.failure_counts: dict[str, int] = {}
        self._closed = False
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"woven-bg-{name}")
        self._thread.start()

    def submit(self, label: str, fn, *args, **kwargs) -> None:
        if self._closed:
            raise RuntimeError("BackgroundWorker is closed")
        self._queue.put((label, fn, args, kwargs))

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is _STOP:
                self._queue.task_done()
                return
            label, fn, args, kwargs = item
            try:
                fn(*args, **kwargs)
                self.success_counts[label] = self.success_counts.get(label, 0) + 1
            except Exception as e:
                logger.warning("Background task '%s' failed: %s", label, e)
                self.failure_counts[label] = self.failure_counts.get(label, 0) + 1
            finally:
                self._queue.task_done()

    def flush(self, timeout: float | None = None) -> bool:
        """Block until all submitted tasks finished. Returns False on timeout."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while self._queue.unfinished_tasks:  # noqa: SLF001 — documented Queue attr
            if deadline is not None and time.monotonic() > deadline:
                return False
            time.sleep(0.01)
        return True

    @property
    def is_alive(self) -> bool:
        """Whether the worker thread is still running.

        True if a previous close()/join() timed out while a task was still
        in flight — callers use this to decide whether it's safe to treat
        the worker as fully stopped.
        """
        return self._thread.is_alive()

    def close(self, timeout: float | None = 5.0) -> None:
        # Idempotent-safe: even on a repeated call (e.g. retrying after a
        # prior close() timed out), still join with the given timeout —
        # only the STOP sentinel is sent at most once.
        if not self._closed:
            self._closed = True
            self._queue.put(_STOP)
        self._thread.join(timeout=timeout)
