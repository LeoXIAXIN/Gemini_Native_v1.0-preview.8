"""RESP client helpers for the G1 safety keys.

This is intentionally a tiny standard-library client.  The browser and web
supervisor are not a hardware safety chain; they only feed the same safety
state machine used by the simulation/real backend.
"""

from __future__ import annotations

import json
import math
import socket
import threading
import time
from typing import Any, Callable

SAFETY_CONTROL_KEY = "g1_safety_control"
SAFETY_STATUS_KEY = "g1_safety_status"
SAFETY_REDIS_ADDRESS = ("127.0.0.1", 6379)
SAFETY_REDIS_TIMEOUT_SECONDS = 0.75

SAFETY_STATES = frozenset(
    {
        "DISARMED",
        "ARMING",
        "BLEND_IN",
        "STANDBY",
        "TELEOP",
        "INPUT_LOSS",
        "RECOVER_STAND",
        "E_STOP",
        "FAULT",
        "LEGACY_DIRECT",
    }
)
REFERENCE_HEALTH_STATES = frozenset(
    {"LIVE", "SOFT_STALE", "HARD_STALE", "WAIT_FRESH", "STANDBY", "INVALID"}
)

_safety_sequence_lock = threading.Lock()
_last_safety_sequence = 0


def _next_safety_sequence() -> int:
    """Return a process-local monotonic sequence based on Unix nanoseconds.

    ``time.time_ns()`` keeps the sequence increasing across ordinary service
    restarts.  The process-local maximum also protects simultaneous HTTP
    requests from receiving the same sequence on coarse clocks.
    """

    global _last_safety_sequence
    with _safety_sequence_lock:
        _last_safety_sequence = max(time.time_ns(), _last_safety_sequence + 1)
        return _last_safety_sequence


def _redis_command(*parts: str | bytes) -> bytes:
    encoded = [part if isinstance(part, bytes) else part.encode("utf-8") for part in parts]
    chunks = [f"*{len(encoded)}\r\n".encode("ascii")]
    for part in encoded:
        chunks.extend((f"${len(part)}\r\n".encode("ascii"), part, b"\r\n"))
    return b"".join(chunks)


def write_safety_control_command(
    action: str,
    *,
    operator_confirmed: bool = False,
    physical_estop_released: bool = False,
    connection_factory: Callable[[tuple[str, int], float], Any] | None = None,
) -> dict[str, Any]:
    """Write one safety command directly to local Redis using RESP."""

    normalized = str(action).strip().lower()
    if normalized not in {"estop", "reset"}:
        raise ValueError("Unsupported safety action")
    if normalized == "reset" and not (
        operator_confirmed and physical_estop_released
    ):
        raise ValueError("Safety reset requires both explicit confirmations")

    sequence = _next_safety_sequence()
    command: dict[str, Any] = {
        "sequence": sequence,
        "action": normalized,
    }
    if normalized == "reset":
        command.update(
            {
                "operator_confirmed": True,
                "physical_estop_released": True,
            }
        )
    value = json.dumps(command, ensure_ascii=False, separators=(",", ":"))
    connect = connection_factory or socket.create_connection
    client = connect(SAFETY_REDIS_ADDRESS, SAFETY_REDIS_TIMEOUT_SECONDS)
    try:
        client.sendall(_redis_command("SET", SAFETY_CONTROL_KEY, value))
        response = b""
        while b"\r\n" not in response and len(response) < 1024:
            chunk = client.recv(1024 - len(response))
            if not chunk:
                break
            response += chunk
    finally:
        client.close()

    if response != b"+OK\r\n":
        detail = response.decode("utf-8", errors="replace").strip() or "no response"
        raise RuntimeError(f"Redis rejected safety command: {detail}")
    return command


def _redis_get(
    key: str,
    *,
    connection_factory: Callable[[tuple[str, int], float], Any] | None = None,
) -> bytes | None:
    """Read one local Redis bulk string using a bounded RESP parser.

    ``None`` is Redis' ordinary response for an absent or expired key.  Other
    reply types are rejected so a protocol error can never be presented as
    valid safety telemetry.
    """

    connect = connection_factory or socket.create_connection
    client = connect(SAFETY_REDIS_ADDRESS, SAFETY_REDIS_TIMEOUT_SECONDS)
    try:
        client.sendall(_redis_command("GET", key))
        response = b""
        expected_size: int | None = None
        body_offset: int | None = None
        while len(response) < 65536 + 128:
            chunk = client.recv(min(4096, 65536 + 128 - len(response)))
            if not chunk:
                break
            response += chunk
            line_end = response.find(b"\r\n")
            if line_end < 0:
                continue
            header = response[:line_end]
            if not header.startswith(b"$"):
                detail = header.decode("utf-8", errors="replace") or "empty reply"
                raise RuntimeError(f"Unexpected Redis GET reply: {detail}")
            try:
                length = int(header[1:])
            except ValueError as exc:
                raise RuntimeError("Invalid Redis bulk-string length") from exc
            if length == -1:
                return None
            if length < 0 or length > 65536:
                raise RuntimeError("Redis safety status is too large")
            body_offset = line_end + 2
            expected_size = body_offset + length + 2
            if len(response) >= expected_size:
                if response[expected_size - 2 : expected_size] != b"\r\n":
                    raise RuntimeError("Malformed Redis bulk-string terminator")
                return response[body_offset : body_offset + length]
        raise RuntimeError("Incomplete Redis GET response")
    finally:
        client.close()


def _unavailable_safety_status(reason: str) -> dict[str, Any]:
    return {
        "available": False,
        "healthy": False,
        "state": "UNAVAILABLE",
        "reason": reason,
        "estop_latched": None,
        "fault_latched": None,
    }


def read_safety_status(
    *,
    connection_factory: Callable[[tuple[str, int], float], Any] | None = None,
) -> dict[str, Any]:
    """Return fail-closed, display-safe safety telemetry from local Redis.

    This function is observational only.  Redis being stopped must not break
    the web supervisor, while missing, expired, or malformed telemetry must
    also never look like a healthy safety state.
    """

    try:
        raw = _redis_get(SAFETY_STATUS_KEY, connection_factory=connection_factory)
        if raw is None:
            return _unavailable_safety_status("Safety telemetry is absent or expired")
        decoded = json.loads(raw.decode("utf-8"))
        if not isinstance(decoded, dict):
            raise ValueError("status must be a JSON object")

        state = decoded.get("state")
        reason = decoded.get("reason")
        estop_latched = decoded.get("estop_latched")
        fault_latched = decoded.get("fault_latched")
        reference_health = decoded.get("reference_health")
        state_source = decoded.get("state_source")
        real_hardware_allowed = decoded.get("real_hardware_allowed")
        teleop_weight = decoded.get("teleop_weight")
        reference_age_ms = decoded.get("reference_age_ms")
        schema_version = decoded.get("schema_version")
        strategy_enabled = decoded.get("strategy_enabled")
        debug_mode = decoded.get("debug_mode")
        bypass_active = decoded.get("bypass_active")
        hard_guards_enabled = decoded.get("hard_guards_enabled")
        limited = decoded.get("limited")

        if schema_version != 1:
            raise ValueError("unsupported safety status schema")
        if state not in SAFETY_STATES:
            raise ValueError("unknown safety state")
        if not isinstance(reason, str) or len(reason) > 1000:
            raise ValueError("invalid safety reason")
        if not isinstance(estop_latched, bool) or not isinstance(fault_latched, bool):
            raise ValueError("invalid safety latch flags")
        if reference_health not in REFERENCE_HEALTH_STATES:
            raise ValueError("unknown reference health")
        if state_source not in {"mujoco", "unitree_lowstate"}:
            raise ValueError("unknown robot state source")
        if not isinstance(real_hardware_allowed, bool):
            raise ValueError("invalid hardware authorization flag")
        telemetry_flags = {
            "strategy_enabled": strategy_enabled,
            "debug_mode": debug_mode,
            "bypass_active": bypass_active,
            "hard_guards_enabled": hard_guards_enabled,
            "limited": limited,
        }
        if any(not isinstance(value, bool) for value in telemetry_flags.values()):
            raise ValueError("invalid or missing safety telemetry flag")
        is_legacy_direct = state == "LEGACY_DIRECT"
        if is_legacy_direct:
            if not (
                not strategy_enabled
                and bypass_active
                and not hard_guards_enabled
                and state_source == "mujoco"
                and not real_hardware_allowed
            ):
                raise ValueError("inconsistent legacy-direct telemetry")
        elif not strategy_enabled or bypass_active or not hard_guards_enabled:
            raise ValueError("inconsistent enabled safety strategy telemetry")
        if isinstance(teleop_weight, bool) or not isinstance(teleop_weight, (int, float)):
            raise ValueError("invalid teleop weight")
        if not math.isfinite(float(teleop_weight)) or not 0.0 <= float(teleop_weight) <= 1.0:
            raise ValueError("invalid teleop weight")
        if isinstance(reference_age_ms, bool) or not isinstance(
            reference_age_ms, (int, float)
        ):
            raise ValueError("invalid reference age")
        if not math.isfinite(float(reference_age_ms)) or float(reference_age_ms) < 0.0:
            raise ValueError("invalid reference age")

        result = dict(decoded)
        result.update(
            {
                "available": True,
                "healthy": (
                    state not in {"E_STOP", "FAULT", "LEGACY_DIRECT"}
                    and not estop_latched
                    and not fault_latched
                ),
                "teleop_weight": float(teleop_weight),
                "reference_age_ms": float(reference_age_ms),
            }
        )
        return result
    except (OSError, RuntimeError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        return _unavailable_safety_status(f"Safety telemetry unavailable: {exc}")
