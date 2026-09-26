"""Platform-safe recurrent thread selection for Unitree control loops.

Unitree's released ``RecurrentThread`` is backed by Linux ``timerfd`` and its
module imports the process C library with ``ctypes.CDLL(None)``.  Importing it
on Windows therefore fails before DDS can create a receive-only LowState
subscriber.  Keep that implementation on POSIX and provide the same small
lifecycle API with ``threading`` on Windows.
"""

from __future__ import annotations

import math
import os
import threading
import time
from collections.abc import Callable
from typing import Any


class WindowsRecurrentThread:
    """Windows equivalent of Unitree's ``RecurrentThread`` lifecycle API."""

    def __init__(
        self,
        interval: float = 1.0,
        target: Callable[..., Any] | None = None,
        name: str | None = None,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> None:
        interval = float(interval)
        if not math.isfinite(interval) or interval <= 0.0:
            raise ValueError("recurrent thread interval must be positive and finite")
        if not callable(target):
            raise TypeError("recurrent thread target must be callable")

        self._interval = interval
        self._target = target
        self._args = args
        self._kwargs = {} if kwargs is None else dict(kwargs)
        self._stop_event = threading.Event()
        self._start_lock = threading.Lock()
        self._started = False
        self._thread = threading.Thread(
            target=self._run,
            name=name,
            daemon=True,
        )

    def Start(self) -> None:
        """Start once, matching the public Unitree SDK method name."""

        with self._start_lock:
            if self._started:
                raise RuntimeError("recurrent thread can only be started once")
            self._started = True
            self._thread.start()

    def Stop(self) -> None:
        """Request termination without blocking."""

        self._stop_event.set()

    def Wait(self, timeout: float | None = None) -> bool:
        """Request termination and return whether the worker has stopped."""

        self.Stop()
        if not self._started:
            return True
        if threading.current_thread() is self._thread:
            return False
        self._thread.join(timeout)
        return not self._thread.is_alive()

    def GetId(self) -> int | None:
        return self._thread.ident

    def GetNativeId(self) -> int | None:
        return self._thread.native_id

    def _run(self) -> None:
        # Unitree runs the callback immediately and then waits on its periodic
        # timerfd.  Use monotonic deadlines here so callback duration does not
        # accumulate drift, while Event.wait keeps shutdown prompt on Windows.
        next_deadline = time.perf_counter()
        while not self._stop_event.is_set():
            try:
                self._target(*self._args, **self._kwargs)
            except BaseException as exc:
                print(
                    "[WindowsRecurrentThread] target raised "
                    f"{type(exc).__name__}: {exc}"
                )

            now = time.perf_counter()
            next_deadline += self._interval
            if next_deadline <= now:
                missed = math.floor((now - next_deadline) / self._interval) + 1
                next_deadline += missed * self._interval
            self._stop_event.wait(max(0.0, next_deadline - now))


def get_recurrent_thread_class():
    """Return a Windows-safe class without importing Linux-only SDK helpers."""

    if os.name == "nt":
        return WindowsRecurrentThread

    from unitree_sdk2py.utils.thread import RecurrentThread

    return RecurrentThread
