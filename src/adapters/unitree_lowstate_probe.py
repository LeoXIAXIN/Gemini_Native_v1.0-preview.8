#!/usr/bin/env python3
"""Strictly receive-only Unitree G1 LowState acceptance probe.

The process opens one DDS subscription to ``rt/lowstate`` and validates the
incoming telemetry.  It deliberately has no command-message or transmission
API.  This makes it suitable for the first network/DDS check before enabling
any real-robot control process.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import importlib
import json
import math
import os
from pathlib import Path
import socket
import tempfile
import threading
import time
from typing import Any, Callable, Mapping, Optional, Sequence


TOPIC = "rt/lowstate"
EXPECTED_MOTOR_COUNT = 29
REPORT_SCHEMA_VERSION = 2
DEFAULT_DURATION_SECONDS = 60.0
DEFAULT_MIN_RATE_HZ = 20.0
UINT32_MODULUS = 1 << 32
UINT32_HALF_RANGE = 1 << 31
SDK_PROVENANCE = {
    "provider": "unitree_sdk2_python",
    "channel_api": "unitree_sdk2py.core.channel.ChannelSubscriber",
    "state_type": "unitree_hg.msg.dds_.LowState_",
    "crc_api": "unitree_sdk2py.utils.crc.CRC.Crc",
}


@dataclass(frozen=True)
class ProbeConfig:
    network_interface: str
    report_path: Path
    # Deprecated compatibility input. Hardware labels do not select the DDS mode.
    g1_version: str | None = None
    duration_seconds: float = DEFAULT_DURATION_SECONDS
    min_rate_hz: float = DEFAULT_MIN_RATE_HZ
    domain_id: int = 0
    queue_depth: int = 10
    verify_interface: bool = True

    def __post_init__(self) -> None:
        interface = self.network_interface.strip()
        if not interface:
            raise ValueError("network interface must not be empty")
        if interface.casefold() in {"lo", "lo0", "localhost"}:
            raise ValueError("loopback is forbidden; select the robot-facing interface")
        if not math.isfinite(self.duration_seconds) or self.duration_seconds <= 0:
            raise ValueError("duration must be a positive finite number")
        if not math.isfinite(self.min_rate_hz) or self.min_rate_hz <= 0:
            raise ValueError("minimum receive rate must be a positive finite number")
        if not 0 <= self.domain_id <= 232:
            raise ValueError("DDS domain id must be in the range 0..232")
        if self.queue_depth <= 0:
            raise ValueError("queue depth must be positive")
        object.__setattr__(self, "network_interface", interface)
        object.__setattr__(self, "report_path", Path(self.report_path))

@dataclass(frozen=True)
class SdkBindings:
    initialize: Callable[[int, str], None]
    subscriber_type: Callable[[str, Any], Any]
    state_type: Any
    calculate_crc: Callable[[Any], int]


class TelemetryValidationError(ValueError):
    def __init__(self, category: str, message: str) -> None:
        super().__init__(message)
        self.category = category


def load_sdk_bindings() -> SdkBindings:
    """Use the official channel, HG IDL and CRC implementation."""

    channel = importlib.import_module("unitree_sdk2py.core.channel")
    messages = importlib.import_module("unitree_sdk2py.idl.unitree_hg.msg.dds_")
    crc = importlib.import_module("unitree_sdk2py.utils.crc").CRC()
    return SdkBindings(
        initialize=getattr(channel, "ChannelFactoryInitialize"),
        subscriber_type=getattr(channel, "ChannelSubscriber"),
        state_type=getattr(messages, "LowState_"),
        calculate_crc=crc.Crc,
    )


def _get_attr(value: Any, names: Sequence[str]) -> Any:
    for name in names:
        if hasattr(value, name):
            return getattr(value, name)
    raise AttributeError(f"missing field; expected one of {', '.join(names)}")


def _finite_values(value: Any, count: int, label: str) -> tuple[float, ...]:
    try:
        items = tuple(float(value[index]) for index in range(count))
    except (IndexError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain at least {count} numeric values") from exc
    if not all(math.isfinite(item) for item in items):
        raise ValueError(f"{label} contains a non-finite value")
    return items


def _extract_telemetry(
    message: Any,
) -> tuple[int, int, int, tuple[float, ...], tuple[float, ...]]:
    try:
        tick = int(_get_attr(message, ("tick",)))
        if not 0 <= tick < UINT32_MODULUS:
            raise ValueError("tick is outside the expected uint32 range")
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise TelemetryValidationError("tick", str(exc)) from exc

    try:
        mode_machine = int(
            _get_attr(message, ("mode_machine", "modeMachine"))
        )
        if not 0 <= mode_machine <= 255:
            raise ValueError("mode_machine is outside the expected uint8 range")
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise TelemetryValidationError("mode_machine", str(exc)) from exc

    try:
        imu = _get_attr(message, ("imu_state", "imuState", "imu"))
        quaternion = _finite_values(
            _get_attr(imu, ("quaternion", "quat")), 4, "IMU quaternion"
        )
    except (AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
        raise TelemetryValidationError("imu", str(exc)) from exc

    try:
        motors = _get_attr(message, ("motor_state", "motorState", "motors"))
    except AttributeError as exc:
        raise TelemetryValidationError("motor", str(exc)) from exc
    try:
        motor_count = len(motors)
    except TypeError as exc:
        raise TelemetryValidationError(
            "motor", "motor state collection has no length"
        ) from exc
    if motor_count < EXPECTED_MOTOR_COUNT:
        raise TelemetryValidationError(
            "motor",
            f"expected at least {EXPECTED_MOTOR_COUNT} motor states, got {motor_count}",
        )

    positions: list[float] = []
    for index in range(EXPECTED_MOTOR_COUNT):
        motor = motors[index]
        try:
            values = (
                float(_get_attr(motor, ("q", "position"))),
                float(_get_attr(motor, ("dq", "qd", "velocity"))),
                float(_get_attr(motor, ("tau_est", "tauEst", "torque"))),
            )
        except (AttributeError, TypeError, ValueError, OverflowError) as exc:
            raise TelemetryValidationError(
                "motor", f"motor state {index}: {exc}"
            ) from exc
        if not all(math.isfinite(item) for item in values):
            raise TelemetryValidationError(
                "motor", f"motor state {index} contains a non-finite value"
            )
        positions.append(values[0])
    return tick, mode_machine, motor_count, quaternion, tuple(positions)


class TelemetryCollector:
    """Thread-safe callback accumulator and validator."""

    def __init__(self, clock: Callable[[], float]) -> None:
        self._clock = clock
        self.calculate_crc: Callable[[Any], int] | None = None
        self._lock = threading.Lock()
        self.received_samples = 0
        self.valid_samples = 0
        self.invalid_samples = 0
        self.local_time_regressions = 0
        self.tick_regressions = 0
        self.duplicate_ticks = 0
        self.advancing_ticks = 0
        self.missing_tick_samples = 0
        self.invalid_mode_machine_samples = 0
        self.mode_machine_changes = 0
        self.crc_valid_samples = 0
        self.crc_invalid_samples = 0
        self.imu_invalid_samples = 0
        self.motor_invalid_samples = 0
        self.first_tick: Optional[int] = None
        self.last_tick: Optional[int] = None
        self.last_mode_machine: Optional[int] = None
        self.last_motor_state_count: Optional[int] = None
        self.observed_mode_machine_counts: dict[int, int] = {}
        self.observed_motor_state_counts: dict[int, int] = {}
        self._last_receive_time: Optional[float] = None
        self.last_quaternion: Optional[tuple[float, ...]] = None
        self.last_motor_positions: Optional[tuple[float, ...]] = None
        self.errors: list[str] = []

    def _remember_error(self, sample_number: int, message: str) -> None:
        if len(self.errors) < 20:
            self.errors.append(f"sample {sample_number}: {message}")

    def observe(self, message: Any) -> None:
        received_at = float(self._clock())
        with self._lock:
            self.received_samples += 1
            sample_number = self.received_samples
            if self._last_receive_time is not None and received_at < self._last_receive_time:
                self.local_time_regressions += 1
            self._last_receive_time = received_at

            try:
                tick, mode_machine, motor_count, quaternion, positions = (
                    _extract_telemetry(message)
                )
                try:
                    expected_crc = int(_get_attr(message, ("crc",)))
                    if self.calculate_crc is None:
                        raise ValueError("official SDK CRC binding is unavailable")
                    if not 0 <= expected_crc < UINT32_MODULUS:
                        raise ValueError("crc is outside the expected uint32 range")
                    if self.calculate_crc(message) != expected_crc:
                        raise ValueError("LowState CRC mismatch")
                except Exception as exc:
                    raise TelemetryValidationError("crc", str(exc)) from exc
            except TelemetryValidationError as exc:
                text = str(exc)
                self.invalid_samples += 1
                if exc.category == "tick":
                    self.missing_tick_samples += 1
                elif exc.category == "mode_machine":
                    self.invalid_mode_machine_samples += 1
                elif exc.category == "imu":
                    self.imu_invalid_samples += 1
                elif exc.category == "motor":
                    self.motor_invalid_samples += 1
                elif exc.category == "crc":
                    self.crc_invalid_samples += 1
                self._remember_error(sample_number, text)
                return
            except Exception as exc:
                self.invalid_samples += 1
                self._remember_error(sample_number, f"{type(exc).__name__}: {exc}")
                return

            if self.first_tick is None:
                self.first_tick = tick
            if self.last_tick is not None:
                delta = (tick - self.last_tick) & (UINT32_MODULUS - 1)
                if delta == 0:
                    self.duplicate_ticks += 1
                elif delta >= UINT32_HALF_RANGE:
                    self.tick_regressions += 1
                else:
                    self.advancing_ticks += 1
            self.last_tick = tick
            if self.last_mode_machine is not None and mode_machine != self.last_mode_machine:
                self.mode_machine_changes += 1
            self.last_mode_machine = mode_machine
            self.last_motor_state_count = motor_count
            self.observed_mode_machine_counts[mode_machine] = (
                self.observed_mode_machine_counts.get(mode_machine, 0) + 1
            )
            self.observed_motor_state_counts[motor_count] = (
                self.observed_motor_state_counts.get(motor_count, 0) + 1
            )
            self.last_quaternion = quaternion
            self.last_motor_positions = positions
            self.valid_samples += 1
            self.crc_valid_samples += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "received_samples": self.received_samples,
                "valid_samples": self.valid_samples,
                "invalid_samples": self.invalid_samples,
                "local_time_regressions": self.local_time_regressions,
                "tick_regressions": self.tick_regressions,
                "duplicate_ticks": self.duplicate_ticks,
                "advancing_ticks": self.advancing_ticks,
                "missing_tick_samples": self.missing_tick_samples,
                "invalid_mode_machine_samples": self.invalid_mode_machine_samples,
                "mode_machine_changes": self.mode_machine_changes,
                "crc_valid_samples": self.crc_valid_samples,
                "crc_invalid_samples": self.crc_invalid_samples,
                "imu_invalid_samples": self.imu_invalid_samples,
                "motor_invalid_samples": self.motor_invalid_samples,
                "first_tick": self.first_tick,
                "last_tick": self.last_tick,
                "last_mode_machine": self.last_mode_machine,
                "last_motor_state_count": self.last_motor_state_count,
                "observed_mode_machine_counts": {
                    str(key): value
                    for key, value in sorted(self.observed_mode_machine_counts.items())
                },
                "observed_motor_state_counts": {
                    str(key): value
                    for key, value in sorted(self.observed_motor_state_counts.items())
                },
                "last_quaternion": self.last_quaternion,
                "last_motor_positions": self.last_motor_positions,
                "validation_errors": list(self.errors),
            }


def evaluate_checks(
    summary: Mapping[str, Any], config: ProbeConfig
) -> list[dict[str, Any]]:
    """Return the deterministic acceptance checks for a completed run."""

    received = int(summary.get("received_samples", 0))
    rate = float(summary.get("receive_rate_hz", 0.0))
    checks = [
        {
            "id": "telemetry_received",
            "passed": received >= 2,
            "detail": f"received {received} samples",
        },
        {
            "id": "receive_rate",
            "passed": rate >= config.min_rate_hz,
            "detail": f"{rate:.2f} Hz (minimum {config.min_rate_hz:.2f} Hz)",
        },
        {
            "id": "observation_duration",
            "passed": float(summary.get("elapsed_seconds", 0.0)) >= config.duration_seconds - 0.01,
            "detail": f"observed {float(summary.get('elapsed_seconds', 0.0)):.2f}s",
        },
        {
            "id": "callback_time_monotonic",
            "passed": int(summary.get("local_time_regressions", 0)) == 0,
            "detail": f"regressions={int(summary.get('local_time_regressions', 0))}",
        },
        {
            "id": "device_tick_monotonic",
            "passed": (
                received >= 2
                and int(summary.get("missing_tick_samples", 0)) == 0
                and int(summary.get("tick_regressions", 0)) == 0
                and int(summary.get("advancing_ticks", 0)) > 0
            ),
            "detail": (
                f"regressions={int(summary.get('tick_regressions', 0))}, "
                f"duplicates={int(summary.get('duplicate_ticks', 0))}, "
                f"advances={int(summary.get('advancing_ticks', 0))}, "
                f"missing={int(summary.get('missing_tick_samples', 0))}"
            ),
        },
        {
            "id": "mode_machine_stable",
            "passed": (
                received > 0
                and int(summary.get("invalid_mode_machine_samples", 0)) == 0
                and int(summary.get("mode_machine_changes", 0)) == 0
                and len(summary.get("observed_mode_machine_counts", {})) == 1
            ),
            "detail": (
                f"last={summary.get('last_mode_machine')}, "
                f"observed={summary.get('observed_mode_machine_counts', {})}, "
                f"changes={int(summary.get('mode_machine_changes', 0))}"
            ),
        },
        {
            "id": "lowstate_crc_valid",
            "passed": (
                received > 0
                and int(summary.get("crc_valid_samples", 0)) == received
                and int(summary.get("crc_invalid_samples", 0)) == 0
            ),
            "detail": (
                f"valid={int(summary.get('crc_valid_samples', 0))}, "
                f"invalid={int(summary.get('crc_invalid_samples', 0))}"
            ),
        },
        {
            "id": "imu_quaternion_finite",
            "passed": (
                received > 0
                and int(summary.get("imu_invalid_samples", 0)) == 0
                and int(summary.get("invalid_samples", 0)) == 0
            ),
            "detail": f"invalid={int(summary.get('imu_invalid_samples', 0))}",
        },
        {
            "id": "motor_states_29_finite",
            "passed": (
                received > 0
                and int(summary.get("motor_invalid_samples", 0)) == 0
                and int(summary.get("invalid_samples", 0)) == 0
                and len(summary.get("observed_motor_state_counts", {})) == 1
            ),
            "detail": (
                f"invalid motor samples={int(summary.get('motor_invalid_samples', 0))}, "
                f"all invalid samples={int(summary.get('invalid_samples', 0))}"
            ),
        },
    ]
    return checks


def write_json_atomic(path: Path, report: Mapping[str, Any]) -> None:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=str(path.parent),
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            json.dump(report, temporary, ensure_ascii=False, indent=2, sort_keys=True)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def _available_interfaces() -> set[str]:
    return {name for _, name in socket.if_nameindex()}


def _close_subscription(subscription: Any) -> None:
    for method_name in ("Close", "close"):
        method = getattr(subscription, method_name, None)
        if callable(method):
            method()
            return


def run_probe(
    config: ProbeConfig,
    *,
    bindings_loader: Callable[[], SdkBindings] = load_sdk_bindings,
    clock: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
    utc_now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    progress: Optional[Callable[[Mapping[str, Any], float], None]] = None,
    write_report: bool = True,
) -> dict[str, Any]:
    """Run the bounded probe and optionally atomically write its report."""

    started_utc = utc_now()
    process_started = float(clock())
    acquisition_started = process_started
    collector = TelemetryCollector(clock)
    subscription: Any = None
    fatal_errors: list[str] = []
    interrupted = False
    next_progress = process_started + 5.0

    try:
        if config.verify_interface:
            interfaces = _available_interfaces()
            if config.network_interface not in interfaces:
                available = ", ".join(sorted(interfaces)) or "none"
                raise RuntimeError(
                    f"network interface {config.network_interface!r} was not found; "
                    f"available: {available}"
                )

        bindings = bindings_loader()
        collector.calculate_crc = bindings.calculate_crc
        bindings.initialize(config.domain_id, config.network_interface)
        subscription = bindings.subscriber_type(TOPIC, bindings.state_type)
        subscription.Init(collector.observe, config.queue_depth)

        # DDS setup time is not part of the requested observation window.
        acquisition_started = float(clock())
        next_progress = acquisition_started + 5.0
        deadline = acquisition_started + config.duration_seconds
        while True:
            now = float(clock())
            if now >= deadline:
                break
            sleeper(min(0.1, deadline - now))
            now = float(clock())
            if progress is not None and now >= next_progress:
                progress(collector.snapshot(), max(0.0, now - acquisition_started))
                next_progress = now + 5.0
    except KeyboardInterrupt:
        interrupted = True
        fatal_errors.append("probe interrupted by operator")
    except Exception as exc:  # setup/runtime failures belong in the report
        fatal_errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        if subscription is not None:
            try:
                _close_subscription(subscription)
            except Exception as exc:
                fatal_errors.append(f"subscription close failed: {type(exc).__name__}: {exc}")

    finished = float(clock())
    elapsed = max(0.0, finished - acquisition_started)
    process_elapsed = max(0.0, finished - process_started)
    summary = collector.snapshot()
    summary["elapsed_seconds"] = elapsed
    summary["process_elapsed_seconds"] = process_elapsed
    summary["receive_rate_hz"] = (
        float(summary["received_samples"]) / elapsed if elapsed > 0 else 0.0
    )
    checks = evaluate_checks(summary, config)
    if fatal_errors:
        checks.append(
            {
                "id": "runtime_completed",
                "passed": False,
                "detail": "; ".join(fatal_errors),
            }
        )
    passed = bool(checks) and all(bool(check["passed"]) for check in checks)

    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "probe": "unitree_g1_lowstate_receive_only",
        "sdk": dict(SDK_PROVENANCE),
        "status": "pass" if passed else "fail",
        "started_at_utc": started_utc.isoformat(),
        "finished_at_utc": utc_now().isoformat(),
        "configuration": {
            "topic": TOPIC,
            "network_interface": config.network_interface,
            "mode_machine_source": "LowState.mode_machine",
            "domain_id": config.domain_id,
            "duration_seconds": config.duration_seconds,
            "minimum_receive_rate_hz": config.min_rate_hz,
            "queue_depth": config.queue_depth,
        },
        "safety_properties": {
            "receive_only": True,
            "command_transmission_capability_present": False,
        },
        "summary": summary,
        "checks": checks,
        "fatal_errors": fatal_errors,
        "interrupted": interrupted,
    }
    if write_report:
        write_json_atomic(config.report_path, report)
    return report


def _progress_line(snapshot: Mapping[str, Any], elapsed: float) -> None:
    received = int(snapshot.get("received_samples", 0))
    rate = received / elapsed if elapsed > 0 else 0.0
    print(
        f"  listening: {elapsed:5.1f}s, samples={received}, "
        f"average={rate:6.1f} Hz",
        flush=True,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Receive-only Unitree G1 DDS acceptance probe. It subscribes to "
            "rt/lowstate and cannot transmit robot commands."
        )
    )
    parser.add_argument(
        "--interface",
        required=True,
        dest="network_interface",
        help="non-loopback robot-facing interface, for example eth0",
    )
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument(
        "--g1-version",
        default=None,
        help=argparse.SUPPRESS,  # Deprecated and ignored; retained for old callers.
    )
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION_SECONDS)
    parser.add_argument("--min-rate-hz", type=float, default=DEFAULT_MIN_RATE_HZ)
    parser.add_argument("--queue-depth", type=int, default=10)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path.cwd() / "unitree_lowstate_probe_report.json",
        help="JSON result path (written atomically)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        config = ProbeConfig(
            network_interface=args.network_interface,
            report_path=args.report,
            g1_version=args.g1_version,
            duration_seconds=args.duration,
            min_rate_hz=args.min_rate_hz,
            domain_id=args.domain_id,
            queue_depth=args.queue_depth,
        )
    except ValueError as exc:
        print(f"Configuration error: {exc}", flush=True)
        return 2

    print("Unitree G1 receive-only DDS acceptance probe", flush=True)
    print("  This process has no command transmission path.", flush=True)
    print(
        f"  interface={config.network_interface}, topic={TOPIC}, "
        f"duration={config.duration_seconds:.1f}s",
        flush=True,
    )
    report = run_probe(config, progress=_progress_line)
    print(f"Result: {report['status'].upper()}", flush=True)
    print(f"Report: {config.report_path.expanduser().resolve()}", flush=True)
    if report["status"] != "pass":
        for check in report["checks"]:
            if not check["passed"]:
                print(f"  FAIL {check['id']}: {check['detail']}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
