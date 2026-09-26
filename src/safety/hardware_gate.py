"""HardwareGate: LowState acceptance report verification and gate state.

Enforcement of the three physical gates (LowState acceptance / ARM G1 /
physical remote ownership) lives in the G1 core state machine
(``src/safety/g1_core.py``), executed inside the process-isolated hgpt real
runner.  This module re-exposes the report verification and records gate
state for the UI/application layer — it is a record/verify layer, not an
enforcement gate.
"""

from __future__ import annotations

import datetime as _datetime
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any


def verify_unitree_lowstate_probe_report(
    path: Path,
    *,
    interface: str,
    g1_version: str | None = None,
    now_utc: _datetime.datetime | None = None,
    maximum_age_seconds: float = 86400.0,
) -> tuple[bool, str]:
    """Verify a complete official-SDK report; legacy motor labels are ignored.

    Reports are local diagnostic records, not a cryptographic attestation. A
    previous custom-IDL or schema-1 report must be regenerated with this probe.
    """

    from src.adapters.unitree_lowstate_probe import (
        DEFAULT_DURATION_SECONDS,
        DEFAULT_MIN_RATE_HZ,
        ProbeConfig,
        REPORT_SCHEMA_VERSION,
        SDK_PROVENANCE,
        TOPIC,
        evaluate_checks,
    )

    del g1_version  # Compatibility only; never maps hardware labels to DDS values.
    try:
        report = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(report, dict):
            raise ValueError("report must be an object")
        config = report["configuration"]
        safety = report["safety_properties"]
        summary = report["summary"]
        if not all(isinstance(value, dict) for value in (config, safety, summary)):
            raise ValueError("configuration, safety_properties and summary must be objects")
        finished_text = str(report["finished_at_utc"])
        finished = _datetime.datetime.fromisoformat(finished_text.replace("Z", "+00:00"))
        if finished.tzinfo is None:
            raise ValueError("report timestamp must include its timezone")
        now = now_utc or _datetime.datetime.now(_datetime.timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=_datetime.timezone.utc)
        age = (now.astimezone(_datetime.timezone.utc) - finished.astimezone(_datetime.timezone.utc)).total_seconds()
        checks = report.get("checks")
        reasons: list[str] = []
        if report.get("schema_version") != REPORT_SCHEMA_VERSION:
            reasons.append("unsupported report schema; rerun the official SDK LowState probe")
        if report.get("sdk") != SDK_PROVENANCE:
            reasons.append("report was not produced with the required official SDK bindings")
        if report.get("probe") != "unitree_g1_lowstate_receive_only":
            reasons.append("wrong probe type")
        if report.get("status") != "pass" or report.get("interrupted") is not False or report.get("fatal_errors") != []:
            reasons.append("probe did not complete successfully")
        if safety.get("receive_only") is not True or safety.get(
            "command_transmission_capability_present"
        ) is not False:
            reasons.append("receive-only safety properties are invalid")
        if config.get("network_interface") != interface:
            reasons.append(
                f"report interface {config.get('network_interface')!r} does not match {interface!r}"
            )
        if config.get("topic") != TOPIC or config.get("mode_machine_source") != "LowState.mode_machine":
            reasons.append("report topic or mode_machine source is invalid")
        probe_config = ProbeConfig(
            network_interface=interface,
            report_path=Path(path),
            duration_seconds=float(config["duration_seconds"]),
            min_rate_hz=float(config["minimum_receive_rate_hz"]),
            domain_id=config["domain_id"],
            queue_depth=config["queue_depth"],
            verify_interface=False,
        )
        if probe_config.duration_seconds < DEFAULT_DURATION_SECONDS:
            reasons.append("probe duration was shorter than 60 seconds")
        if probe_config.min_rate_hz < DEFAULT_MIN_RATE_HZ:
            reasons.append("report receive-rate threshold was below the acceptance minimum")
        if probe_config.domain_id != 0:
            reasons.append("report DDS domain does not match the control domain 0")
        counters = (
            "received_samples", "valid_samples", "invalid_samples", "local_time_regressions",
            "tick_regressions", "duplicate_ticks", "advancing_ticks", "missing_tick_samples",
            "invalid_mode_machine_samples", "mode_machine_changes", "crc_valid_samples",
            "crc_invalid_samples", "imu_invalid_samples", "motor_invalid_samples",
        )
        for name in counters:
            if type(summary.get(name)) is not int or summary[name] < 0:
                raise ValueError(f"missing or invalid summary counter: {name}")
        received = summary["received_samples"]
        if received < 2 or summary["valid_samples"] != received or summary["invalid_samples"] != 0:
            reasons.append("received and valid sample counts are inconsistent")
        if summary["advancing_ticks"] + summary["duplicate_ticks"] + summary["tick_regressions"] != received - 1:
            reasons.append("device tick counters are inconsistent")
        for name in ("first_tick", "last_tick"):
            if type(summary.get(name)) is not int or not 0 <= summary[name] < (1 << 32):
                raise ValueError(f"missing or invalid {name}")
        mode = summary.get("last_mode_machine")
        if type(mode) is not int or not 0 <= mode <= 255 or summary.get("observed_mode_machine_counts") != {str(mode): received}:
            reasons.append("mode_machine must be a single valid value for every frame")
        motors = summary.get("last_motor_state_count")
        if type(motors) is not int or motors < 29 or summary.get("observed_motor_state_counts") != {str(motors): received}:
            reasons.append("motor state structure is missing or changed during the probe")
        for name, minimum_length in (("last_quaternion", 4), ("last_motor_positions", 29)):
            values = summary.get(name)
            if not isinstance(values, list) or len(values) < minimum_length or not all(
                isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
                for value in values
            ):
                raise ValueError(f"missing or non-finite telemetry: {name}")
        elapsed = float(summary["elapsed_seconds"])
        rate = float(summary["receive_rate_hz"])
        if not math.isfinite(elapsed) or elapsed <= 0 or not math.isfinite(rate) or not math.isclose(rate, received / elapsed, rel_tol=1e-6):
            reasons.append("observation duration or receive rate is invalid")
        computed_checks = evaluate_checks(summary, probe_config)
        required = {item["id"] for item in computed_checks}
        if not isinstance(checks, list) or not checks or any(
            not isinstance(item, dict) or not isinstance(item.get("id"), str) or item.get("passed") is not True
            for item in checks
        ):
            reasons.append("one or more LowState acceptance checks are missing or failed")
        else:
            recorded = [item["id"] for item in checks]
            if len(recorded) != len(set(recorded)) or not required.issubset(recorded):
                reasons.append("required LowState acceptance checks are missing or duplicated")
        failed = [item["id"] for item in computed_checks if item["passed"] is not True]
        if failed:
            reasons.append("telemetry failed recomputed checks: " + ", ".join(failed))
        if not math.isfinite(maximum_age_seconds) or maximum_age_seconds <= 0:
            raise ValueError("maximum report age must be a positive finite number")
        if age < -300.0 or age > maximum_age_seconds:
            reasons.append(f"report age {age / 3600.0:.1f} h is outside the allowed window")
        if reasons:
            return False, "; ".join(reasons)
        return True, (
            f"Official SDK receive-only LowState report passed for {interface}, "
            f"mode_machine={mode} ({age / 60.0:.0f} minutes old)"
        )
    except (OSError, KeyError, TypeError, ValueError, OverflowError, AttributeError) as exc:
        return False, f"LowState report is missing or invalid: {exc}"


@dataclass(frozen=True)
class HardwareGateState:
    """Recorded (not enforced) gate status for one real-output session."""

    lowstate_report_ok: bool = False
    lowstate_report_message: str = ""
    arm_g1_requested: bool = False
    physical_remote_owned: bool = False
    debug_receive_only: bool = True

    @property
    def real_output_possible(self) -> bool:
        """Mirror the frozen checklist; a recorded boolean, not a capability.

        Enforcement remains with the G1 core state machine
        (``src/safety/g1_core.py``) inside the hgpt real-runner process.
        """
        return (
            self.lowstate_report_ok
            and self.arm_g1_requested
            and self.physical_remote_owned
            and not self.debug_receive_only
        )


class HardwareGate:
    """Verifies the frozen LowState acceptance report (module-local logic)."""

    def verify_lowstate_report(
        self,
        path: Path,
        *,
        interface: str,
        g1_version: str | None = None,
        now_utc: _datetime.datetime | None = None,
        maximum_age_seconds: float = 86400.0,
    ) -> tuple[bool, str]:
        return verify_unitree_lowstate_probe_report(
            Path(path),
            interface=interface,
            g1_version=g1_version,
            now_utc=now_utc,
            maximum_age_seconds=maximum_age_seconds,
        )
