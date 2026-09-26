#!/usr/bin/env python3
"""Focused regression tests for the Windows Unitree recurrent-thread shim."""

from __future__ import annotations

from pathlib import Path
import os
import sys
import threading
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.adapters.unitree_thread import (  # noqa: E402
    WindowsRecurrentThread,
    get_recurrent_thread_class,
)


def test_windows_factory_avoids_linux_timerfd() -> None:
    if os.name == "nt":
        assert get_recurrent_thread_class() is WindowsRecurrentThread


def test_runs_and_stops() -> None:
    calls = 0
    first_call = threading.Event()
    lock = threading.Lock()

    def callback() -> None:
        nonlocal calls
        with lock:
            calls += 1
        first_call.set()

    worker = WindowsRecurrentThread(interval=0.01, target=callback, name="test")
    worker.Start()
    assert first_call.wait(0.5), "callback never ran"
    time.sleep(0.04)
    assert worker.Wait(0.5), "worker did not stop"

    with lock:
        stopped_count = calls
    assert stopped_count >= 2, stopped_count
    time.sleep(0.03)
    with lock:
        assert calls == stopped_count, "callback ran after Wait confirmed stop"


def test_bounded_wait() -> None:
    entered = threading.Event()
    release = threading.Event()

    def blocking_callback() -> None:
        entered.set()
        release.wait(1.0)

    worker = WindowsRecurrentThread(interval=0.01, target=blocking_callback)
    worker.Start()
    assert entered.wait(0.5), "blocking callback never started"
    assert worker.Wait(0.01) is False, "bounded Wait reported a false join"
    release.set()
    assert worker.Wait(0.5), "worker did not stop after callback was released"


def test_rejects_unsafe_configuration() -> None:
    for interval in (0.0, -0.01, float("nan"), float("inf")):
        try:
            WindowsRecurrentThread(interval=interval, target=lambda: None)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted unsafe interval {interval!r}")


def main() -> int:
    test_windows_factory_avoids_linux_timerfd()
    test_runs_and_stops()
    test_bounded_wait()
    test_rejects_unsafe_configuration()
    print("UNITREE WINDOWS RECURRENT THREAD TEST PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
