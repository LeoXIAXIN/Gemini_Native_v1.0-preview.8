#!/usr/bin/env python3
"""Deterministic checks for native CHINGMU upstream-stall diagnostics."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.adapters.chingmu import FrameFreshnessWatchdog  # noqa: E402


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def main() -> int:
    clock = FakeClock()
    watchdog = FrameFreshnessWatchdog(
        clock=clock,
        warning_seconds=0.75,
        repeat_seconds=5.0,
    )
    clock.now = 0.74
    assert watchdog.poll() is None
    clock.now = 0.75
    assert watchdog.poll() == 0.75
    clock.now = 1.0
    assert watchdog.poll() is None
    assert watchdog.frame_received() == 1.0
    clock.now = 1.74
    assert watchdog.poll() is None
    clock.now = 1.75
    assert watchdog.poll() == 0.75
    clock.now = 6.75
    assert watchdog.poll() == 5.75

    print("CHINGMU FRAME FRESHNESS WATCHDOG TEST PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
