#!/usr/bin/env python3
"""Read-only Windows proof that G1 is publishing Develop-mode LowState.

The field firmware may stop ``MotionSwitcher`` RPC service in the same state
that enables low-level telemetry, so ``CheckMode`` is not a reliable Develop
probe.  This executable subscribes only to ``rt/lowstate``.  It never
constructs a publisher, sends a command, selects a mode, or releases an owner.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile


if str(Path(__file__).resolve().parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.adapters import unitree_lowstate_probe as common
from src.adapters.unitree_lowstate_probe_windows import (
    load_windows_bindings,
    validate_interface_address,
)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify live G1 Develop LowState without LowCmd."
    )
    parser.add_argument("--interface-address", required=True)
    parser.add_argument("--g1-version", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--duration", type=float, default=2.0)
    return parser


def probe_preentered_develop(
    interface_address: str,
    g1_version: str | None = None,
    *,
    duration_seconds: float = 2.0,
) -> tuple[str, dict[str, object]]:
    """Return only after a bounded receive-only LowState probe passes."""

    configured = validate_interface_address(interface_address)
    trace_path = (
        Path(os.environ.get("LOCALAPPDATA", tempfile.gettempdir()))
        / "ChingmuGeminiNative"
        / "logs"
        / "cyclonedds-develop-preflight.log"
    )
    config = common.ProbeConfig(
        network_interface=configured,
        report_path=Path(tempfile.gettempdir()) / "unused-develop-probe.json",
        g1_version=g1_version,  # Deprecated compatibility argument; ignored.
        duration_seconds=float(duration_seconds),
        min_rate_hz=20.0,
        domain_id=0,
        queue_depth=10,
        verify_interface=False,
    )
    report = common.run_probe(
        config,
        bindings_loader=lambda: load_windows_bindings(trace_path),
        write_report=False,
    )
    if report.get("status") != "pass":
        summary = report.get("summary", {})
        received = int(summary.get("received_samples", 0))
        rate = float(summary.get("receive_rate_hz", 0.0))
        raise RuntimeError(
            "no valid Develop LowState stream was received "
            f"(samples={received}, rate={rate:.1f} Hz). Keep the real task "
            "stopped, enter damping/zero-torque with the physical remote, "
            "release all buttons, press L2+R2, then run preflight again"
        )
    return configured, report


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        configured, report = probe_preentered_develop(
            args.interface_address,
            args.g1_version,
            duration_seconds=args.duration,
        )
    except Exception as exc:
        print(f"DEVELOP_PREFLIGHT_FAIL: {type(exc).__name__}: {exc}", flush=True)
        return 2
    summary = report["summary"]
    print(
        "DEVELOP_PREFLIGHT_PASS: fresh LowState on "
        f"{configured}, samples={int(summary['received_samples'])}, "
        f"rate={float(summary['receive_rate_hz']):.1f} Hz; "
        "official SDK subscriber only; no command writer was opened",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
