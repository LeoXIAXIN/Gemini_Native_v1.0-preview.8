#!/usr/bin/env python3
"""Native-Windows, receive-only probe using the official Unitree SDK2 Python.

The official ChannelSubscriber, HG LowState_ IDL and CRC implementation are
shared with the control path. No publisher is created and no command is sent.
"""

from __future__ import annotations

import argparse
import ipaddress
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence

# Direct-script bootstrap: make the src package importable when this file is
# executed directly by the acceptance command.
if str(Path(__file__).resolve().parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.adapters import unitree_lowstate_probe as common


def validate_interface_address(value: str) -> str:
    try:
        address = ipaddress.ip_address(value.strip())
    except ValueError as exc:
        raise ValueError("interface address must be a valid IPv4 address") from exc
    if not isinstance(address, ipaddress.IPv4Address):
        raise ValueError("Unitree Windows DDS probe currently requires IPv4")
    if (
        address.is_loopback
        or address.is_unspecified
        or address.is_multicast
        or address.is_link_local
    ):
        raise ValueError(
            "select the robot-facing, non-loopback Windows Ethernet IPv4 address"
        )
    return str(address)


def load_windows_bindings(trace_path: Path) -> common.SdkBindings:
    """Configure the Windows adapter before initializing the official factory."""

    from src.adapters.unitree import configure_windows_unitree_interface

    bindings = common.load_sdk_bindings()

    def initialize(domain_id: int, interface_address: str) -> None:
        configured = configure_windows_unitree_interface(
            validate_interface_address(interface_address), trace_path=trace_path
        )
        bindings.initialize(domain_id, configured)

    return common.SdkBindings(
        initialize=initialize,
        subscriber_type=bindings.subscriber_type,
        state_type=bindings.state_type,
        calculate_crc=bindings.calculate_crc,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Native-Windows receive-only Unitree G1 DDS probe. It subscribes "
            "to rt/lowstate and has no command transmission path."
        )
    )
    parser.add_argument(
        "--interface-address",
        required=True,
        help=(
            "IPv4 address of the Windows Ethernet adapter connected to G1, "
            "for example 192.168.123.99"
        ),
    )
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument(
        "--g1-version",
        default=None,
        help=argparse.SUPPRESS,  # Deprecated and ignored.
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=common.DEFAULT_DURATION_SECONDS,
    )
    parser.add_argument(
        "--min-rate-hz",
        type=float,
        default=common.DEFAULT_MIN_RATE_HZ,
    )
    parser.add_argument("--queue-depth", type=int, default=10)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path.cwd() / "unitree_lowstate_windows_report.json",
    )
    parser.add_argument(
        "--trace",
        type=Path,
        default=(
            Path(os.environ.get("LOCALAPPDATA", tempfile.gettempdir()))
            / "CHINGMU_GMR_Control"
            / "logs"
            / "cyclonedds_windows.log"
        ),
    )
    return parser


def persist_probe_result(
    acceptance_path: Path,
    report: Mapping[str, Any],
) -> tuple[Path, bool]:
    """Write every attempt but replace the acceptance report only on PASS."""

    acceptance_path = Path(acceptance_path).expanduser().resolve()
    attempt_path = acceptance_path.with_name(
        f"{acceptance_path.stem}.last_attempt{acceptance_path.suffix}"
    )
    common.write_json_atomic(attempt_path, report)
    accepted = report.get("status") == "pass"
    if accepted:
        common.write_json_atomic(acceptance_path, report)
    return attempt_path, accepted


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    if os.name != "nt":
        parser.error(
            "This probe requires native Windows. Use "
            "installer/unitree_lowstate_probe.py under Linux/WSL."
        )
    try:
        address = validate_interface_address(args.interface_address)
        acceptance_path = args.report.expanduser().resolve()
        config = common.ProbeConfig(
            network_interface=address,
            report_path=acceptance_path,
            g1_version=args.g1_version,
            duration_seconds=args.duration,
            min_rate_hz=args.min_rate_hz,
            domain_id=args.domain_id,
            queue_depth=args.queue_depth,
            verify_interface=False,
        )
    except ValueError as exc:
        print(f"Configuration error: {exc}", flush=True)
        return 2

    print("Unitree G1 native-Windows receive-only DDS probe", flush=True)
    print("  This process uses the official SDK subscriber and creates no writer.", flush=True)
    print(
        f"  adapter IPv4={address}, topic={common.TOPIC}, "
        f"duration={config.duration_seconds:.1f}s",
        flush=True,
    )
    report = common.run_probe(
        config,
        bindings_loader=lambda: load_windows_bindings(args.trace),
        progress=_progress_line,
        write_report=False,
    )
    report["runtime"] = {
        "platform": "windows_native",
        "interface_address": address,
        "cyclonedds_trace": str(args.trace.expanduser().resolve()),
    }
    attempt_path, accepted = persist_probe_result(acceptance_path, report)
    print(f"Result: {report['status'].upper()}", flush=True)
    print(f"Attempt report: {attempt_path}", flush=True)
    print(f"CycloneDDS trace: {args.trace.expanduser().resolve()}", flush=True)
    if report["status"] != "pass":
        if acceptance_path.is_file():
            print(
                f"Previous acceptance report preserved: {acceptance_path}",
                flush=True,
            )
        for check in report["checks"]:
            if not check["passed"]:
                print(f"  FAIL {check['id']}: {check['detail']}", flush=True)
        if int(report.get("summary", {}).get("received_samples", 0)) == 0:
            print(
                "  NEXT: keep G1 on a loaded support rig, enter damping/"
                "zero-torque mode, press L2+R2 for Develop mode, and rerun "
                "this receive-only probe.",
                flush=True,
            )
            print(
                "  Verify that --interface-address is this PC's G1 Ethernet "
                "IPv4, not the mocap server or robot IPv4.",
                flush=True,
            )
        return 1
    if accepted:
        print(f"Acceptance report updated: {acceptance_path}", flush=True)
    return 0


def _progress_line(snapshot: Mapping[str, Any], elapsed: float) -> None:
    received = int(snapshot.get("received_samples", 0))
    rate = received / elapsed if elapsed > 0 else 0.0
    print(
        f"  listening: {elapsed:5.1f}s, samples={received}, "
        f"average={rate:6.1f} Hz",
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
