#!/usr/bin/env python3
"""Local web supervisor for CHINGMU -> GMR -> G1 simulation/real pipelines.

The service deliberately uses only the Python standard library.  It binds to
loopback, launches one of a small set of fixed Bash launchers without a shell,
and only signals the process group that it created.
"""

from __future__ import annotations

import argparse
import collections
import datetime as _datetime
import ipaddress
import json
import math
import mimetypes
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
import urllib.parse
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

# Direct-script bootstrap: make the src package importable when this file is
# executed as a script (sys.path[0] is src/controller).  It must run BEFORE
# every src import so that `python src/controller/motion_control_app.py` and
# `python -m src.controller.motion_control_app` behave identically.
if str(Path(__file__).resolve().parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.application.gemini_license import LicenseManager, LicenseStatus

# M10b: this is the canonical controller implementation; config primitives
# and the migrated helpers come from src directly (no legacy imports).
from src.domain.g1_versions import DEFAULT_MODEL_PROFILE, model_profile_from_config
from src.infrastructure import config as _config_impl
from src.infrastructure.logging import (
    classify_component as _classify_component_impl,
    classify_level as _classify_level_impl,
)
from src.safety import channel as _channel
from src.safety.hardware_gate import verify_unitree_lowstate_probe_report


SERVICE_NAME = "chingmu-gemini"
BASE_DIR = Path(__file__).resolve().parents[2]
STATIC_DIR = BASE_DIR / "src" / "controller" / "static"
CONFIG_PATH = BASE_DIR / "config" / "gemini_native_config.json"
LOG_DIR = BASE_DIR / "storage" / "logs"
LICENSE_PUBLIC_KEY_PATH = BASE_DIR / "resources" / "license" / "public_key.json"
LICENSE_FILE_PATH = BASE_DIR / "resources" / "license" / "ChingmuGemini.license"
SAFETY_CONTROL_KEY = "g1_safety_control"
SAFETY_STATUS_KEY = "g1_safety_status"
SAFETY_REDIS_ADDRESS = ("127.0.0.1", 6379)
SAFETY_REDIS_TIMEOUT_SECONDS = 0.75
# Per-request authorization for enabling a real LowCmd publisher.  This is
# deliberately not persisted in the configuration: every transition from the
# web service's stopped state into non-debug real output needs a fresh token.
REAL_OUTPUT_AUTHORIZATION = "ARM G1"

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
_license_status_lock = threading.Lock()
_license_manager = LicenseManager(
    public_key_path=LICENSE_PUBLIC_KEY_PATH,
    license_path=LICENSE_FILE_PATH,
)
_last_license_status = LicenseStatus(
    allowed=False,
    code="not_checked",
    message="尚未检查产品许可证。",
)


def check_product_license() -> LicenseStatus:
    """Verify permission to start one new task without affecting a live task."""

    global _last_license_status
    status = _license_manager.authorize_new_session()
    with _license_status_lock:
        _last_license_status = status
    return status


def product_license_payload() -> dict[str, Any]:
    """Return only the operator-safe cached license status."""

    with _license_status_lock:
        return _last_license_status.to_public_dict()

SIMULATION_BACKEND_SCRIPTS = {
    "gmr_preview": BASE_DIR / "src" / "orchestration" / "preview_supervisor.py",
    "twist": BASE_DIR / "src" / "orchestration" / "sim_supervisor.py",
    "humanoid_gpt": BASE_DIR / "src" / "orchestration" / "sim_supervisor.py",
}
REAL_BACKEND_SCRIPT = BASE_DIR / "src" / "orchestration" / "real_supervisor.py"
# Backwards-compatible alias used by a few older tests and helpers.
BACKEND_SCRIPTS = SIMULATION_BACKEND_SCRIPTS


def launcher_for_config(config: dict[str, Any]) -> Path:
    """Return the fixed launcher selected by backend and execution mode."""

    if config["execution_mode"] == "simulation_only":
        return SIMULATION_BACKEND_SCRIPTS[config["backend"]]
    if config["execution_mode"] in {"real_only", "parallel"}:
        if config["backend"] not in {"twist", "humanoid_gpt"}:
            raise ConfigError("real output requires backend=twist or humanoid_gpt")
        return REAL_BACKEND_SCRIPT
    raise ConfigError("shadow mode is not connected yet")

STALE_PROJECT_EXECUTABLES = frozenset(
    str(path.resolve())
    for path in (
        *BACKEND_SCRIPTS.values(),
        REAL_BACKEND_SCRIPT,
        BASE_DIR / "src" / "adapters" / "chingmu.py",
        BASE_DIR / "src" / "motion" / "gmr_bridge.py",
        BASE_DIR / "src" / "motion" / "gmr_preview.py",
        BASE_DIR / "src" / "motion" / "live_sim.py",
        BASE_DIR / "src" / "motion" / "play_track.py",
    )
)

DEFAULT_CONFIG: dict[str, Any] = {
    "backend": "twist",
    "execution_mode": "simulation_only",
    # Real mode starts receive-only.  This is an output authorization gate,
    # not the optional experimental safety strategy below.
    "debug_mode": True,
    "safety_strategy_enabled": False,
    "server_ip": "192.168.2.100",
    "skeleton_id": 0,
    "human_height": 1.75,
    "publish_fps": 50.0,
    "transition_seconds": 1.5,
    "device": "auto",
    "model_profile": DEFAULT_MODEL_PROFILE,
    "unitree_network_interface": "eth0",
    "unitree_robot_ip": "192.168.123.164",
}

CONFIG_KEYS = frozenset(DEFAULT_CONFIG)
BUSY_PHASES = frozenset(
    {
        "preflight",
        "starting",
        "waiting_mocap",
        "starting_backend",
        "running",
        "stopping",
    }
)


# M10b: same class object as src/infrastructure/config.ConfigError.
ConfigError = _config_impl.ConfigError


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
    """Write one safety command directly to local Redis using RESP.

    This is intentionally a tiny standard-library client.  The browser and
    web supervisor are not a hardware safety chain; they only feed the same
    simulation safety state machine used by ``humanoid_gpt_live_sim.py``.
    """

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


def _as_finite_float(value: Any, name: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool):
        raise ConfigError(f"{name} must be a number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be a number") from exc
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise ConfigError(f"{name} must be between {minimum:g} and {maximum:g}")
    return number


def _as_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{name} must be a JSON boolean")
    return value


def validate_config(candidate: Any, base: dict[str, Any] | None = None) -> dict[str, Any]:
    # Implementation lives in src/infrastructure/config.py; ``defaults``
    # preserves the legacy seeding semantics (the native entry mutates
    # DEFAULT_CONFIG before any validation call).
    return _config_impl.validate_config(
        candidate, base, defaults=DEFAULT_CONFIG
    )


def verify_hgpt_trt_report(path: Path) -> tuple[bool, str]:
    """Verify the install-time, two-policy TensorRT execution report."""

    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        models = report.get("models")
        reasons: list[str] = []
        if report.get("ok") is not True:
            reasons.append("runtime report is not ok")
        if not isinstance(models, list) or len(models) != 2:
            reasons.append("both released ONNX policies were not tested")
            models = []
        expected_names = {
            "pns_wo_priv216.onnx",
            "07140632_G1-Walk_v2.0.0_baseline.onnx",
        }
        observed_names: set[str] = set()
        for model in models:
            observed_names.add(Path(str(model.get("path", ""))).name)
            providers = model.get("providers")
            if not isinstance(providers, list) or not providers or providers[0] != "TensorrtExecutionProvider":
                reasons.append("TensorRT is not the first provider for every policy")
            p99 = float(model.get("inference_p99_ms", float("inf")))
            if not math.isfinite(p99) or p99 > 20.0:
                reasons.append(f"policy inference p99 is too high: {p99:g} ms")
        if observed_names != expected_names:
            reasons.append("runtime report does not cover the pinned tracking and walking policies")
        if reasons:
            return False, "; ".join(dict.fromkeys(reasons))
        return True, "Both released Humanoid-GPT policies passed TensorRT execution"
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return False, f"Humanoid-GPT TensorRT report is missing or invalid: {exc}"


def load_config() -> tuple[dict[str, Any], str | None]:
    # Implementation lives in src/infrastructure/config.py.
    return _config_impl.load_config(CONFIG_PATH, DEFAULT_CONFIG)


def persist_config(config: dict[str, Any]) -> None:
    # Implementation lives in src/infrastructure/config.py.
    _config_impl.persist_config(config, CONFIG_PATH)


def build_launch_environment(
    config: dict[str, Any], base_environment: dict[str, str] | None = None
) -> dict[str, str]:
    """Build the fixed launcher environment from a validated configuration."""

    return _config_impl.build_launch_environment(config, base_environment)


def _iso_now() -> str:
    return _datetime.datetime.now().astimezone().isoformat(timespec="milliseconds")


def _iso_from_epoch(timestamp: float | None) -> str | None:
    if timestamp is None:
        return None
    return _datetime.datetime.fromtimestamp(timestamp).astimezone().isoformat(
        timespec="seconds"
    )


def _empty_metrics() -> dict[str, Any]:
    return {
        "sender_fps": None,
        "source_fps": None,
        "local_rx_fps": None,
        "ik_fps": None,
        "local_drops": 0,
        "bridge_fps": None,
        "backend_fps": None,
        "frame_id": None,
        "root_z": None,
        "packet_age_ms": None,
        "input_age_ms": None,
        "rejected": 0,
        "device": None,
        "model_loaded": False,
    }


def _proc_start_time(pid: int) -> str | None:
    """Return Linux /proc start ticks, which protects cleanup from PID reuse."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        after_name = stat[stat.rfind(")") + 2 :].split()
        # after_name starts at proc field 3; starttime is field 22.
        return after_name[19]
    except (OSError, IndexError):
        return None


def _proc_command_paths(pid: int) -> set[str]:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
        arguments = [part.decode("utf-8", "replace") for part in raw.split(b"\0") if part]
        cwd = Path(os.readlink(f"/proc/{pid}/cwd"))
    except OSError:
        return set()
    paths: set[str] = set()
    for index, argument in enumerate(arguments):
        # ``python -m deploy.play_track`` has no .py argument.  Resolve that
        # one exact, fixed project module so cleanup can find an orphaned real
        # Humanoid-GPT process without broad process-name matching.
        if (
            argument == "-m"
            and index + 1 < len(arguments)
            and arguments[index + 1] == "deploy.play_track"
        ):
            module_path = cwd / "deploy" / "play_track.py"
            try:
                paths.add(str(module_path.resolve(strict=False)))
            except OSError:
                pass
        if argument.startswith("-") or not argument.endswith((".py", ".sh")):
            continue
        candidate = Path(argument)
        if not candidate.is_absolute():
            candidate = cwd / candidate
        try:
            paths.add(str(candidate.resolve(strict=False)))
        except OSError:
            continue
    return paths


def find_stale_project_processes() -> list[dict[str, Any]]:
    """Find only processes whose argv resolves to a fixed project executable."""
    if os.name != "posix" or not Path("/proc").is_dir():
        return []
    records: list[dict[str, Any]] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == os.getpid():
            continue
        matched = _proc_command_paths(pid) & STALE_PROJECT_EXECUTABLES
        start_time = _proc_start_time(pid)
        if matched and start_time is not None:
            records.append(
                {
                    "pid": pid,
                    "start_time": start_time,
                    "paths": sorted(matched),
                }
            )
    return records


def stale_record_is_alive(record: dict[str, Any]) -> bool:
    pid = int(record["pid"])
    if _proc_start_time(pid) != record["start_time"]:
        return False
    return bool(_proc_command_paths(pid) & set(record["paths"]))


SENDER_RE = re.compile(
    r"Sender FPS:\s*(?P<fps>[0-9.]+),\s*frame:\s*(?P<frame>-?[0-9]+)"
    r"(?:,\s*local drops:\s*(?P<drops>[0-9]+))?",
    re.I,
)
BRIDGE_RE = re.compile(
    r"Bridge\s+(?P<bridge>[0-9.]+)\s*Hz\s*\|\s*source\s+"
    r"(?P<source>[0-9.]+)\s*Hz\s*\|\s*frame\s+(?P<frame>-?[0-9]+)\s*\|\s*"
    r"root z\s+(?P<root>-?[0-9.]+)\s*m\s*\|\s*(?:IK\s+)?rejected\s+"
    r"(?P<rejected>[0-9]+)",
    re.I,
)
BRIDGE_LOCAL_RE = re.compile(
    r"Bridge\s+(?P<bridge>[0-9.]+)\s*Hz\s*\|\s*local RX\s+"
    r"(?P<local_rx>[0-9.]+)\s*Hz\s*\|\s*IK\s+(?P<ik>[0-9.]+)\s*Hz\s*\|\s*"
    r"frame\s+(?P<frame>-?[0-9]+)\s*\|\s*root z\s+"
    r"(?P<root>-?[0-9.]+)\s*m\s*\|\s*rejected\s+(?P<rejected>[0-9]+)",
    re.I,
)
HGPT_RE = re.compile(
    r"Humanoid-GPT\s+(?P<fps>[0-9.]+)\s*Hz\s*\|\s*frame\s+"
    r"(?P<frame>-?[0-9]+)\s*\|\s*qpos z\s+(?P<root>-?[0-9.]+)\s*m\s*\|\s*"
    r"packet age\s+(?P<age>[0-9.]+)\s*ms",
    re.I,
)
TWIST_RUNTIME_RE = re.compile(
    r"TWIST MuJoCo\s+(?P<fps>[0-9.]+)\s*Hz\s*\|\s*policy\s+"
    r"(?P<policy>[0-9.]+)\s*Hz\s*\|\s*root z\s+"
    r"(?P<root>-?[0-9.]+)\s*m\s*\|\s*mimic delta\s+"
    r"(?P<delta>[0-9.]+)",
    re.I,
)
GMR_PREVIEW_RE = re.compile(
    r"GMR Preview\s+(?P<fps>[0-9.]+)\s*Hz\s*\|\s*frame\s+"
    r"(?P<frame>-?[0-9]+)\s*\|\s*root z\s+(?P<root>-?[0-9.]+)\s*m\s*\|\s*"
    r"rejected\s+(?P<rejected>[0-9]+)",
    re.I,
)


def classify_component(line: str, backend: str | None) -> str:
    lowered = line.lower()
    if "sender fps" in lowered or "[sdk " in lowered or "vrpn" in lowered:
        return "chingmu"
    if "bridge" in lowered or "ik frame" in lowered or "use ik config" in lowered:
        return "gmr"
    if "humanoid-gpt" in lowered or "[onnx]" in lowered:
        return "humanoid_gpt"
    if "gmr preview" in lowered or "gmr exact preview" in lowered:
        return "gmr"
    if (
        "policy device" in lowered
        or "policy loaded" in lowered
        or "motor id" in lowered
        or "degrees of freedom" in lowered
    ):
        return "twist"
    return backend or "launcher"


def classify_level(line: str) -> str:
    lowered = line.lower()
    if any(
        marker in lowered
        for marker in (
            "traceback",
            "segmentation fault",
            "fatal python error",
            "address already in use",
            "cannot find",
            "did not start",
            "no valid live",
            "invalid_skeleton",
        )
    ):
        return "error"
    if "error" in lowered or "exception" in lowered:
        return "error"
    if "warning" in lowered or "rejected" in lowered or "waiting" in lowered:
        return "warning"
    return "info"


class PipelineManager:
    """Own one launcher process group and expose a thread-safe status snapshot."""

    def __init__(self) -> None:
        config, warning = load_config()
        if warning is None and not CONFIG_PATH.exists():
            try:
                persist_config(config)
            except OSError as exc:
                warning = f"Could not create the configuration file: {exc}"
        self.lock = threading.RLock()
        self.config = config
        self.phase = "stopped"
        self.message = "Ready"
        self.error: str | None = warning
        self.backend: str | None = None
        self.process: subprocess.Popen[str] | None = None
        self.started_epoch: float | None = None
        self.started_monotonic: float | None = None
        self.exit_code: int | None = None
        self.metrics = _empty_metrics()
        self.last_input_monotonic: float | None = None
        self.session_error_hint: str | None = None
        self.logs: collections.deque[dict[str, Any]] = collections.deque(maxlen=2000)
        self.log_counter = 0
        self.session_log_path: Path | None = None
        self.log_handle: Any = None
        self.monitor_thread: threading.Thread | None = None
        self.stop_thread: threading.Thread | None = None
        self.safety_command_lock = threading.Lock()
        if warning:
            self._append_log(warning, component="service", level="warning")

    def config_payload(self) -> dict[str, Any]:
        with self.lock:
            config = dict(self.config)
            config.pop("model_profile", None)
        return {
            "ok": True,
            "config": config,
            "options": {
                "backends": ["gmr_preview", "twist", "humanoid_gpt"],
                "execution_modes": [
                    "simulation_only",
                    "real_only",
                    "parallel",
                    "shadow",
                ],
                "debug_mode": [False, True],
                "safety_strategy_enabled": [True, False],
                "devices": ["auto", "cpu", "cuda"],
            },
        }

    def update_config(self, update: Any) -> dict[str, Any]:
        if isinstance(update, dict) and {"model_profile", "g1_version"} & update.keys():
            raise ConfigError("model_profile is internal deployment configuration")
        with self.lock:
            if (
                self.phase in BUSY_PHASES
                or (self.process is not None and self.process.poll() is None)
            ):
                raise RuntimeError("Stop the current simulation before changing settings")
            validated = validate_config(update, self.config)
            persist_config(validated)
            self.config = validated
            self.error = None
            self.message = "Configuration saved"
        self._append_log("Configuration saved", component="service")
        return self.config_payload()

    def request_emergency_stop(self) -> tuple[int, dict[str, Any]]:
        """Latch the software E-stop without stopping the managed process."""

        with self.lock:
            backend = self.config["backend"]
            execution_mode = self.config["execution_mode"]
            strategy_enabled = bool(
                self.config.get("safety_strategy_enabled", False)
            )
        if backend != "humanoid_gpt" or execution_mode != "simulation_only":
            return HTTPStatus.CONFLICT, {
                "ok": False,
                "error": (
                    "The software E-stop is currently wired only to the "
                    "Humanoid-GPT simulation safety gateway; it must not be "
                    "presented as effective for TWIST or real hardware"
                ),
            }
        if not strategy_enabled:
            return HTTPStatus.CONFLICT, {
                "ok": False,
                "error": (
                    "Software E-stop is unavailable while the safety strategy is "
                    "disabled in experimental direct mode"
                ),
            }
        try:
            with self.safety_command_lock:
                command = write_safety_control_command("estop")
        except (OSError, RuntimeError) as exc:
            self._append_log(
                f"Could not publish software E-stop: {exc}",
                component="safety",
                level="error",
            )
            return HTTPStatus.SERVICE_UNAVAILABLE, {
                "ok": False,
                "error": f"Could not reach local Redis safety channel: {exc}",
            }

        self._append_log(
            f"Software E-stop latch requested (sequence={command['sequence']}); "
            "the pipeline was deliberately left running",
            component="safety",
            level="warning",
        )
        return HTTPStatus.ACCEPTED, {
            "ok": True,
            "latched": True,
            "sequence": command["sequence"],
            "message": "Software E-stop latch command published; process remains active",
        }

    def request_safety_reset(self, payload: Any) -> tuple[int, dict[str, Any]]:
        """Reset the software latch only for the Humanoid-GPT simulator."""

        with self.lock:
            backend = self.config["backend"]
            execution_mode = self.config["execution_mode"]
            strategy_enabled = bool(
                self.config.get("safety_strategy_enabled", False)
            )
        if execution_mode != "simulation_only" or backend != "humanoid_gpt":
            return HTTPStatus.FORBIDDEN, {
                "ok": False,
                "error": (
                    "Web safety reset is only available for backend=humanoid_gpt "
                    "with execution_mode=simulation_only; real robot modes can never "
                    "be reset from this page"
                ),
            }
        if not strategy_enabled:
            return HTTPStatus.CONFLICT, {
                "ok": False,
                "error": (
                    "Safety reset is unavailable while the safety strategy is "
                    "disabled in experimental direct mode"
                ),
            }
        if not isinstance(payload, dict) or payload.get("operator_confirmed") is not True:
            return HTTPStatus.BAD_REQUEST, {
                "ok": False,
                "error": "Safety reset requires explicit operator confirmation",
            }
        if payload.get("physical_estop_released") is not True:
            return HTTPStatus.BAD_REQUEST, {
                "ok": False,
                "error": "Safety reset requires confirmation that the simulated E-stop is released",
            }

        try:
            with self.safety_command_lock:
                command = write_safety_control_command(
                    "reset",
                    operator_confirmed=True,
                    physical_estop_released=True,
                )
        except (OSError, RuntimeError) as exc:
            self._append_log(
                f"Could not publish simulation safety reset: {exc}",
                component="safety",
                level="error",
            )
            return HTTPStatus.SERVICE_UNAVAILABLE, {
                "ok": False,
                "error": f"Could not reach local Redis safety channel: {exc}",
            }

        self._append_log(
            f"Simulation safety latch reset requested (sequence={command['sequence']})",
            component="safety",
            level="warning",
        )
        return HTTPStatus.ACCEPTED, {
            "ok": True,
            "latched": False,
            "sequence": command["sequence"],
            "message": "Simulation safety reset command published",
        }

    def status(self, log_limit: int = 200) -> dict[str, Any]:
        with self.lock:
            process_alive = self.process is not None and self.process.poll() is None
            bounded_log_limit = max(0, min(log_limit, 500))
            recent_logs = (
                list(self.logs)[-bounded_log_limit:] if bounded_log_limit else []
            )
            uptime = (
                max(0.0, time.monotonic() - self.started_monotonic)
                if self.started_monotonic is not None and process_alive
                else 0.0
            )
            metrics = dict(self.metrics)
            if self.last_input_monotonic is not None:
                metrics["input_age_ms"] = round(
                    max(0.0, time.monotonic() - self.last_input_monotonic) * 1000.0,
                    1,
                )
            configured_backend = self.config.get("backend", DEFAULT_CONFIG["backend"])
            strategy_enabled = bool(
                self.config.get("safety_strategy_enabled", False)
            )
            configured_debug_mode = bool(self.config.get("debug_mode", False))
            payload = {
                "ok": True,
                "phase": self.phase,
                "running": self.phase == "running" and process_alive,
                "active": process_alive,
                "backend": self.backend,
                "pid": self.process.pid if process_alive and self.process else None,
                "started_at": _iso_from_epoch(self.started_epoch),
                "uptime_seconds": round(uptime, 1),
                "exit_code": self.exit_code,
                "error": self.error,
                "message": self.message,
                "debug_mode": bool(self.config.get("debug_mode", False)),
                "safety_strategy_enabled": bool(
                    self.config.get("safety_strategy_enabled", False)
                ),
                "metrics": metrics,
                "logs": recent_logs,
                "session_log": (
                    str(self.session_log_path) if self.session_log_path else None
                ),
                "product_license": product_license_payload(),
            }
        # Safety telemetry is intentionally read outside the manager lock: a
        # stopped Redis server may consume the short socket timeout, but it
        # must not block start/stop/configuration operations.
        safety = read_safety_status()
        if configured_backend != "humanoid_gpt":
            safety = _unavailable_safety_status(
                "Safety strategy telemetry is not integrated with the selected backend"
            )
        elif safety.get("available"):
            telemetry_matches = (
                safety.get("strategy_enabled") is strategy_enabled
                and safety.get("debug_mode") is configured_debug_mode
                and (
                    (not strategy_enabled and safety.get("state") == "LEGACY_DIRECT")
                    or (strategy_enabled and safety.get("state") != "LEGACY_DIRECT")
                )
            )
            if not telemetry_matches:
                safety = _unavailable_safety_status(
                    "Safety telemetry does not match the saved debug/strategy configuration"
                )
        if safety.get("available"):
            safety["bypassed"] = bool(safety["bypass_active"])
        else:
            # This describes only the saved selection, never a verified live
            # state.  ``available=false`` keeps the UI fail-closed.
            safety["strategy_enabled"] = strategy_enabled
            safety["bypassed"] = not strategy_enabled
        payload["safety"] = safety
        return payload

    def _append_log(
        self,
        message: str,
        component: str | None = None,
        level: str | None = None,
    ) -> None:
        message = message.rstrip("\r\n")
        if not message:
            return
        with self.lock:
            component = component or classify_component(message, self.backend)
            level = level or classify_level(message)
            self.log_counter += 1
            entry = {
                "id": self.log_counter,
                "timestamp": _iso_now(),
                "component": component,
                "level": level,
                "message": message,
            }
            self.logs.append(entry)
            if level == "error" and component not in {"http", "service"}:
                self.session_error_hint = message
            if self.log_handle is not None:
                try:
                    self.log_handle.write(
                        f"{entry['timestamp']} [{component}] [{level.upper()}] {message}\n"
                    )
                    self.log_handle.flush()
                except Exception:
                    pass
            self._parse_line_locked(message)

    def _parse_line_locked(self, line: str) -> None:
        sender = SENDER_RE.search(line)
        if sender:
            self.last_input_monotonic = time.monotonic()
            self.metrics["sender_fps"] = float(sender.group("fps"))
            self.metrics["frame_id"] = int(sender.group("frame"))
            if sender.group("drops") is not None:
                self.metrics["local_drops"] = int(sender.group("drops"))

        local_bridge = BRIDGE_LOCAL_RE.search(line)
        bridge = BRIDGE_RE.search(line) if local_bridge is None else None
        if local_bridge:
            self.last_input_monotonic = time.monotonic()
            self.metrics.update(
                {
                    "bridge_fps": float(local_bridge.group("bridge")),
                    "source_fps": float(local_bridge.group("local_rx")),
                    "local_rx_fps": float(local_bridge.group("local_rx")),
                    "ik_fps": float(local_bridge.group("ik")),
                    "frame_id": int(local_bridge.group("frame")),
                    "root_z": float(local_bridge.group("root")),
                    "rejected": int(local_bridge.group("rejected")),
                }
            )
        elif bridge:
            self.last_input_monotonic = time.monotonic()
            self.metrics.update(
                {
                    "bridge_fps": float(bridge.group("bridge")),
                    "source_fps": float(bridge.group("source")),
                    "frame_id": int(bridge.group("frame")),
                    "root_z": float(bridge.group("root")),
                    "rejected": int(bridge.group("rejected")),
                }
            )

        hgpt = HGPT_RE.search(line)
        if hgpt:
            self.last_input_monotonic = time.monotonic()
            self.metrics.update(
                {
                    "backend_fps": float(hgpt.group("fps")),
                    "frame_id": int(hgpt.group("frame")),
                    "root_z": float(hgpt.group("root")),
                    "packet_age_ms": float(hgpt.group("age")),
                }
            )

        twist_runtime = TWIST_RUNTIME_RE.search(line)
        if twist_runtime:
            self.metrics.update(
                {
                    "backend_fps": float(twist_runtime.group("policy")),
                    "root_z": float(twist_runtime.group("root")),
                }
            )

        preview = GMR_PREVIEW_RE.search(line)
        if preview:
            self.last_input_monotonic = time.monotonic()
            self.metrics.update(
                {
                    "bridge_fps": float(preview.group("fps")),
                    "backend_fps": float(preview.group("fps")),
                    "frame_id": int(preview.group("frame")),
                    "root_z": float(preview.group("root")),
                    "rejected": int(preview.group("rejected")),
                }
            )

        device_match = re.search(
            r"(?:Policy device:\s*|device=)([A-Za-z0-9_-]+)", line, re.I
        )
        if device_match:
            self.metrics["device"] = device_match.group(1).lower()
        if "policy loaded from" in line.lower() or "[onnx] loaded" in line.lower():
            self.metrics["model_loaded"] = True

        if self.phase == "stopping":
            return
        lowered = line.lower()
        if (
            "waiting_for_udp_frame" in lowered
            or "waiting_for_local_frame" in lowered
            or "gmr receiver ready on unix:" in lowered
            or "waiting on 127.0.0.1" in lowered
            or "waiting up to" in lowered
            or "waiting for the first" in lowered
        ):
            self.phase = "waiting_mocap"
            self.message = "Waiting for live CHINGMU frames"
        if "local_frame_received" in lowered or "received first local frame packet" in lowered:
            self.phase = "waiting_mocap"
            self.message = "Live CHINGMU frame received; decoding skeleton"
        if "retargeting_first_frame" in lowered or "retargeting first valid frame" in lowered:
            self.phase = "waiting_mocap"
            self.message = "Live skeleton received; retargeting the first frame"
        if "invalid_skeleton" in lowered:
            self.phase = "waiting_mocap"
            self.message = (
                "Live CHINGMU packets arrived, but the selected skeleton is incomplete; "
                "check Skeleton ID and core segment solving"
            )
        if (
            "first live target received" in lowered
            or "starting humanoid-gpt released policy" in lowered
            or "starting twist policy" in lowered
            or "starting official-state-feedback real controller" in lowered
        ):
            self.phase = "starting_backend"
            self.message = "Live motion received; starting selected policy backend"
        if "successfully connected to the robot" in lowered:
            self.phase = "starting_backend"
            self.message = "Unitree LowState connected; waiting for upstream remote handover"
        twist_ready = (
            self.backend == "twist"
            and self.phase == "starting_backend"
            and "motor id 24:" in lowered
        )
        if (
            "humanoid-gpt live mujoco backend ready" in lowered
            or twist_ready
            or preview is not None
            or "begin main policy loop" in lowered
            or "<mode: locomotion> starting control loop" in lowered
            or "experimental_direct_real read-only ready" in lowered
            or "lowstate received. starting read-only policy inference" in lowered
            or "a received. twist direct tracking is live" in lowered
        ):
            self.phase = "running"
            self.message = (
                "Selected simulation/real pipeline is running"
                if self.config.get("execution_mode") == "parallel"
                else "Selected control pipeline is running"
            )

    def _new_session_log(self, backend: str, config: dict[str, Any]) -> None:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        stamp = _datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = LOG_DIR / f"session_{stamp}_{backend}.log"
        handle = path.open("a", encoding="utf-8", buffering=1)
        handle.write(
            f"{_iso_now()} [service] [INFO] Session config: "
            f"{json.dumps(config, ensure_ascii=False, sort_keys=True)}\n"
        )
        with self.lock:
            if self.log_handle is not None:
                try:
                    self.log_handle.close()
                except Exception:
                    pass
            self.session_log_path = path
            self.log_handle = handle

    def preflight(self, config: dict[str, Any] | None = None) -> dict[str, Any]:
        with self.lock:
            checked_config = dict(self.config if config is None else config)
            process_alive = self.process is not None and self.process.poll() is None

        checks: list[dict[str, Any]] = []

        def add(name: str, ok: bool, message: str, required: bool = True) -> None:
            checks.append(
                {"name": name, "ok": bool(ok), "message": message, "required": required}
            )

        license_status = check_product_license()
        add(
            "product_license",
            license_status.allowed,
            license_status.message,
        )

        execution_mode = checked_config["execution_mode"]
        if execution_mode == "simulation_only":
            add(
                "输出方式",
                True,
                "仅运行 MuJoCo；不会创建 Unitree 真机发送器",
            )
        elif execution_mode in {"real_only", "parallel"}:
            add(
                "实验性真机直通",
                True,
                (
                    "使用上游官方 LowState→策略→LowCmd 闭环；当前为调试模式，"
                    "只读取 LowState，不发布 LowCmd"
                    if checked_config["debug_mode"]
                    else (
                        "启动前必须用实体手柄进入 Develop；程序确认 LowState 新鲜、"
                        "版本匹配和 START 授权后才发布 LowCmd；网页停止不是急停"
                    )
                ),
            )
        else:
            add(
                "安全影子模式",
                False,
                "影子模式尚未接入；LowState 只读验证请用真机模式并勾选调试模式",
            )

        add(
            "platform",
            os.name == "posix",
            "WSL/Linux process-group control is available"
            if os.name == "posix"
            else "Run this service inside WSL/Linux",
        )
        add("session", not process_alive, "No managed pipeline is active")
        add("bash", shutil.which("bash") is not None, "Bash launcher is available")
        add("flock", shutil.which("flock") is not None, "flock is available")
        if checked_config["backend"] in {"twist", "humanoid_gpt"}:
            redis_ok = (
                shutil.which("redis-server") is not None
                and shutil.which("redis-cli") is not None
            )
            add(
                "redis",
                redis_ok,
                "Redis server and client are available"
                if redis_ok
                else "Redis server/client not found",
            )

        try:
            launcher = launcher_for_config(checked_config)
        except ConfigError as exc:
            launcher = None
            add("launcher", False, str(exc))
        else:
            add("launcher", launcher.is_file(), f"Launcher: {launcher}")
        home = Path.home()
        add(
            "sdk",
            (home / "ChingMuPythonSDKs_Linux" / "ChingmuDLL" / "libCMVrpn.so").is_file(),
            "CHINGMU Linux SDK library found",
        )
        add(
            "sdk_cwd",
            (home / "GMR").is_dir(),
            "Safe Linux SDK working directory ~/GMR found",
        )
        add(
            "gmr_python",
            (home / "miniforge3" / "envs" / "gmr" / "bin" / "python").is_file(),
            "GMR Python environment found",
        )

        is_real = execution_mode in {"real_only", "parallel"}
        if is_real:
            interface = checked_config["unitree_network_interface"]
            robot_ip = checked_config["unitree_robot_ip"]
            interface_path = Path("/sys/class/net") / interface
            interface_ok = interface != "lo" and interface_path.is_dir()
            add(
                "unitree_interface",
                interface_ok,
                (
                    f"WSL/Linux interface {interface} is visible"
                    if interface_ok
                    else f"Interface {interface} is not visible inside WSL/Linux"
                ),
            )

            ping_binary = shutil.which("ping")
            ping_ok = False
            if ping_binary and interface_ok:
                try:
                    ping_result = subprocess.run(
                        [ping_binary, "-c", "1", "-W", "1", "-I", interface, robot_ip],
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=3.0,
                        check=False,
                    )
                    ping_ok = ping_result.returncode == 0
                except (OSError, subprocess.SubprocessError):
                    ping_ok = False
            add(
                "unitree_ping",
                ping_ok,
                (
                    f"Unitree G1 {robot_ip} responds through {interface}"
                    if ping_ok
                    else (
                        "ping is not installed inside WSL/Linux"
                        if not ping_binary
                        else f"Cannot reach Unitree G1 {robot_ip} through {interface}"
                    )
                ),
                required=False,
            )

            real_state_dir = home / ".local" / "state" / "chingmu-robot-control"
            runtime_env_path = real_state_dir / "real_runtime.env"
            runtime_values: dict[str, str] = {}
            try:
                for line in runtime_env_path.read_text(encoding="utf-8").splitlines():
                    if line and not line.lstrip().startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        runtime_values[key.strip()] = value.strip()
            except OSError:
                pass
            cyclone_home = Path(runtime_values.get("CYCLONEDDS_HOME", ""))
            runtime_ok = runtime_env_path.is_file() and (
                cyclone_home / "lib" / "libddsc.so"
            ).is_file()
            add(
                "unitree_real_runtime",
                runtime_ok,
                (
                    f"Pinned real runtime found: {cyclone_home}"
                    if runtime_ok
                    else "Pinned CycloneDDS runtime is missing; rerun 01_安装环境.bat"
                ),
            )

            lowstate_ok, lowstate_message = verify_unitree_lowstate_probe_report(
                real_state_dir / "unitree_lowstate_probe_report.json",
                interface=interface,
            )
            add("unitree_lowstate_receive_only", lowstate_ok, lowstate_message)

            real_python = (
                home
                / "miniforge3"
                / "envs"
                / ("gmr" if checked_config["backend"] == "twist" else "h-gpt-real")
                / "bin"
                / "python"
            )
            sdk_import_ok = False
            sdk_message = f"Unitree SDK2 import failed in {real_python}"
            if real_python.is_file():
                try:
                    dependency_probe = (
                        "import unitree_sdk2py, redis, torch, yaml, termcolor"
                        if checked_config["backend"] == "twist"
                        else "import unitree_sdk2py, redis"
                    )
                    sdk_result = subprocess.run(
                        [str(real_python), "-c", dependency_probe],
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        timeout=8.0,
                        check=False,
                    )
                    sdk_import_ok = sdk_result.returncode == 0
                    if not sdk_import_ok and sdk_result.stdout:
                        sdk_message += f": {sdk_result.stdout.strip()[-300:]}"
                except (OSError, subprocess.SubprocessError) as exc:
                    sdk_message += f": {exc}"
            if sdk_import_ok:
                sdk_message = f"Unitree SDK2 and real-runner dependencies are importable in {real_python}"
            add("unitree_sdk2", sdk_import_ok, sdk_message)

            if checked_config["backend"] == "humanoid_gpt":
                trt_report_ok, trt_report_message = verify_hgpt_trt_report(
                    real_state_dir / "hgpt_real_runtime_report.json"
                )
                add("humanoid_gpt_trt_report", trt_report_ok, trt_report_message)

                trt_ok = False
                if real_python.is_file():
                    try:
                        probe_env = dict(os.environ)
                        nvidia_path = runtime_values.get("CHINGMU_NVIDIA_LIBRARY_PATH", "")
                        runtime_library_paths = [
                            value
                            for value in (
                                str(cyclone_home / "lib") if runtime_ok else "",
                                nvidia_path,
                                probe_env.get("LD_LIBRARY_PATH", ""),
                            )
                            if value
                        ]
                        if runtime_library_paths:
                            probe_env["LD_LIBRARY_PATH"] = ":".join(runtime_library_paths)
                        trt_result = subprocess.run(
                            [
                                str(real_python),
                                "-c",
                                (
                                    "import onnxruntime as o; "
                                    "raise SystemExit(0 if 'TensorrtExecutionProvider' "
                                    "in o.get_available_providers() else 3)"
                                ),
                            ],
                            stdin=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            timeout=8.0,
                            check=False,
                            env=probe_env,
                        )
                        trt_ok = trt_result.returncode == 0
                    except (OSError, subprocess.SubprocessError):
                        trt_ok = False
                add(
                    "tensorrt",
                    trt_ok,
                    (
                        "TensorRT provider is available"
                        if trt_ok
                        else "Humanoid-GPT official real mode requires TensorrtExecutionProvider"
                    ),
                )

        if checked_config["backend"] == "twist":
            add(
                "twist_policy",
                (BASE_DIR / "TWIST" / "assets" / "twist_general_motion_tracker.pt").is_file(),
                "TWIST released policy found",
            )
        elif checked_config["backend"] == "humanoid_gpt":
            if execution_mode in {"simulation_only", "parallel"}:
                simulation_python = home / "miniforge3" / "envs" / "h-gpt" / "bin" / "python"
                add(
                    "humanoid_gpt_sim_python",
                    simulation_python.is_file(),
                    "Humanoid-GPT simulation Python environment found",
                )
            if is_real:
                real_hgpt_python = home / "miniforge3" / "envs" / "h-gpt-real" / "bin" / "python"
                add(
                    "humanoid_gpt_real_python",
                    real_hgpt_python.is_file(),
                    "Humanoid-GPT TensorRT/Unitree real Python environment found",
                )
            add(
                "humanoid_gpt_policy",
                (
                    BASE_DIR
                    / "storage"
                    / "ckpts"
                    / "pns_wo_priv216.onnx"
                ).is_file(),
                "Humanoid-GPT released ONNX policy found",
            )
            asset = (
                BASE_DIR
                / "storage"
                / "assets"
                / model_profile_from_config(checked_config)
            )
            add("g1_asset", asset.is_dir(), f"G1 asset: {asset.name}")

        display_ok = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        add(
            "display",
            display_ok,
            "WSLg display is available" if display_ok else "No DISPLAY/WAYLAND_DISPLAY found",
            required=(
                execution_mode in {"simulation_only", "parallel"}
                or checked_config["backend"] == "humanoid_gpt"
            ),
        )

        frame_transport_path = BASE_DIR / "src" / "adapters" / "frame_transport.py"
        add(
            "frame_transport",
            frame_transport_path.is_file(),
            (
                "Unix local frame transport is available"
                if frame_transport_path.is_file()
                else "Local frame transport helper is missing"
            ),
        )

        if checked_config["device"] == "cuda":
            cuda_ok = shutil.which("nvidia-smi") is not None
            add(
                "cuda",
                cuda_ok,
                "NVIDIA runtime detected" if cuda_ok else "nvidia-smi was not found",
            )

        errors = [item["message"] for item in checks if item["required"] and not item["ok"]]
        return {"ok": not errors, "checks": checks, "errors": errors}

    def start(self, authorization: str | None = None) -> tuple[int, dict[str, Any]]:
        with self.lock:
            if (
                self.phase in BUSY_PHASES
                or (self.process is not None and self.process.poll() is None)
            ):
                return HTTPStatus.CONFLICT, self._status_with_result(
                    False, "A pipeline is already active"
                )
            config = dict(self.config)
            if (
                config["execution_mode"] in {"real_only", "parallel"}
                and not config["debug_mode"]
                and authorization != REAL_OUTPUT_AUTHORIZATION
            ):
                message = (
                    "Real LowCmd output requires a fresh ARM G1 authorization "
                    "for this start request"
                )
                return HTTPStatus.PRECONDITION_REQUIRED, self._status_with_result(
                    False, message
                )
            if config["execution_mode"] == "shadow":
                blocked_message = (
                    "影子模式尚未接入；LowState 只读验证请使用真机模式并勾选调试模式"
                )
                self.phase = "error"
                self.message = blocked_message
                self.error = blocked_message
                self._append_log(blocked_message, component="service", level="error")
                return HTTPStatus.PRECONDITION_FAILED, self._status_with_result(
                    False, blocked_message
                )
            self.phase = "preflight"
            self.message = "Checking environment"
            self.error = None

        preflight = self.preflight(config)
        if not preflight["ok"]:
            with self.lock:
                if self.phase != "preflight":
                    return HTTPStatus.CONFLICT, self._status_with_result(
                        False, "Start was cancelled"
                    )
                self.phase = "error"
                self.error = "; ".join(preflight["errors"])
                self.message = "Environment check failed"
            self._append_log(self.error, component="service", level="error")
            payload = self._status_with_result(False, self.message)
            payload["preflight"] = preflight
            return HTTPStatus.PRECONDITION_FAILED, payload

        launcher = launcher_for_config(config)
        bash = shutil.which("bash") or "/usr/bin/bash"
        env = build_launch_environment(config)

        # Keep the reservation lock through Popen so a concurrent stop/config
        # request can never observe "starting" without an owned process.
        with self.lock:
            if self.phase != "preflight":
                return HTTPStatus.CONFLICT, self._status_with_result(
                    False, "Start was cancelled"
                )
            self.phase = "starting"
            self.message = "Starting pipeline"
            self._new_session_log(config["backend"], config)
            try:
                process = subprocess.Popen(
                    [bash, str(launcher)],
                    cwd=str(BASE_DIR),
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    # On Linux this performs setsid(), giving the service one exact
                    # process group to stop without using pkill or name matching.
                    start_new_session=(os.name == "posix"),
                )
            except Exception as exc:
                self.phase = "error"
                self.error = f"Could not start launcher: {exc}"
                self.message = "Launch failed"
                self._append_log(self.error, component="service", level="error")
                self._close_log()
                return HTTPStatus.INTERNAL_SERVER_ERROR, self._status_with_result(
                    False, self.message
                )
            self.process = process
            self.backend = config["backend"]
            self.started_epoch = time.time()
            self.started_monotonic = time.monotonic()
            self.exit_code = None
            self.metrics = _empty_metrics()
            self.last_input_monotonic = None
            self.session_error_hint = None
            self.monitor_thread = threading.Thread(
                target=self._monitor_process,
                args=(process,),
                name="pipeline-log-monitor",
                daemon=True,
            )
            monitor = self.monitor_thread
        self._append_log(
            f"Started {config['backend']} launcher as process group {process.pid}",
            component="service",
        )
        monitor.start()
        return HTTPStatus.ACCEPTED, self._status_with_result(True, "Start accepted")

    def _monitor_process(self, process: subprocess.Popen[str]) -> None:
        try:
            if process.stdout is not None:
                for line in iter(process.stdout.readline, ""):
                    self._append_log(line)
                process.stdout.close()
            return_code = process.wait()
        except Exception as exc:
            self._append_log(
                f"Log monitor failed: {exc}", component="service", level="error"
            )
            try:
                return_code = process.poll()
            except Exception:
                return_code = None

        with self.lock:
            if self.process is not process:
                return
            was_stopping = self.phase == "stopping"
            runtime_error = self.session_error_hint
            self.exit_code = return_code
            self.process = None
            self.started_monotonic = None
            if was_stopping or (return_code == 0 and runtime_error is None):
                self.phase = "stopped"
                self.message = (
                    "Pipeline stopped" if was_stopping else "Control pipeline closed"
                )
                self.error = None
            else:
                self.phase = "error"
                self.message = "Pipeline exited unexpectedly"
                self.error = runtime_error or f"Launcher exited with code {return_code}"
        self._append_log(
            f"Launcher exited with code {return_code}",
            component="service",
            level=(
                "info"
                if was_stopping or (return_code == 0 and runtime_error is None)
                else "error"
            ),
        )
        self._close_log()

    def _status_with_result(self, accepted: bool, result_message: str) -> dict[str, Any]:
        payload = self.status()
        payload["accepted"] = accepted
        payload["result_message"] = result_message
        return payload

    @staticmethod
    def _signal_process_group(process: subprocess.Popen[str], signum: int) -> None:
        if process.poll() is not None:
            return
        try:
            if os.name == "posix":
                os.killpg(process.pid, signum)
            else:
                process.send_signal(signum)
        except ProcessLookupError:
            pass

    @staticmethod
    def _signal_launcher(process: subprocess.Popen[str], signum: int) -> None:
        """Signal Bash first so its trap can stop children in a safe order."""
        if process.poll() is not None:
            return
        try:
            process.send_signal(signum)
        except ProcessLookupError:
            pass

    @staticmethod
    def _wait_for_exit(process: subprocess.Popen[str], timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if process.poll() is not None:
                return True
            time.sleep(0.1)
        return process.poll() is not None

    def request_stop(
        self, force: bool = False, *, source: str = "internal request"
    ) -> tuple[int, dict[str, Any]]:
        with self.lock:
            process = self.process
            if process is None or process.poll() is not None:
                self.phase = "stopped"
                self.message = "No managed pipeline is running"
                return HTTPStatus.OK, self._status_with_result(True, self.message)
            if self.stop_thread is not None and self.stop_thread.is_alive():
                return HTTPStatus.ACCEPTED, self._status_with_result(
                    True, "Stop already in progress"
                )
            self.phase = "stopping"
            action = "Force stopping pipeline" if force else "Stopping pipeline"
            self.message = f"{action} (source: {source})"
            self.stop_thread = threading.Thread(
                target=self._stop_worker,
                args=(process, force),
                name="pipeline-stop",
                daemon=True,
            )
            thread = self.stop_thread
        self._append_log(self.message, component="service", level="warning")
        thread.start()
        return HTTPStatus.ACCEPTED, self._status_with_result(True, self.message)

    def _stop_worker(self, process: subprocess.Popen[str], force: bool) -> None:
        int_timeout = 0.75 if force else 12.0
        term_timeout = 0.75 if force else 4.0
        # Let the launcher trap coordinate its child shutdown order.  Only
        # escalate to the whole group if that orderly shutdown gets stuck.
        self._signal_launcher(process, signal.SIGINT)
        if not self._wait_for_exit(process, int_timeout):
            self._append_log(
                "Pipeline did not stop after SIGINT; sending SIGTERM",
                component="service",
                level="warning",
            )
            self._signal_process_group(process, signal.SIGTERM)
        if not self._wait_for_exit(process, term_timeout):
            self._append_log(
                "Pipeline did not stop after SIGTERM; sending SIGKILL",
                component="service",
                level="warning",
            )
            self._signal_process_group(process, signal.SIGKILL)
            self._wait_for_exit(process, 2.0)

    def cleanup(self) -> tuple[int, dict[str, Any]]:
        """Remove exact-path project orphans without broad name matching."""
        with self.lock:
            process = self.process
            if process is not None and process.poll() is None:
                return HTTPStatus.CONFLICT, self._status_with_result(
                    False,
                    "A managed session is active; use Stop or Force stop first",
                )
            self.phase = "stopping"
            self.message = "Checking for project process remnants"

        if os.name != "posix" or not Path("/proc").is_dir():
            with self.lock:
                self.phase = "error"
                self.error = "Residual cleanup is only available inside WSL/Linux"
                self.message = self.error
            return HTTPStatus.NOT_IMPLEMENTED, self._status_with_result(
                False, self.message
            )

        records = find_stale_project_processes()
        matched_count = len(records)
        if records:
            summary = ", ".join(
                f"{record['pid']}:{Path(record['paths'][0]).name}" for record in records
            )
            self._append_log(
                f"Cleaning exact-path project remnants: {summary}",
                component="service",
                level="warning",
            )
            records = self._signal_stale_records(records, signal.SIGINT, 3.0)
            if records:
                records = self._signal_stale_records(records, signal.SIGTERM, 2.0)
            if records:
                records = self._signal_stale_records(records, signal.SIGKILL, 1.0)

        remaining = find_stale_project_processes()
        if remaining:
            detail = ", ".join(str(record["pid"]) for record in remaining)
            with self.lock:
                self.phase = "error"
                self.error = f"Could not stop project process(es): {detail}"
                self.message = "Residual cleanup incomplete"
            return HTTPStatus.CONFLICT, self._status_with_result(False, self.error)
        with self.lock:
            self.phase = "stopped"
            self.error = None
            self.message = (
                f"Cleanup complete; removed {matched_count} project remnant(s); "
                "the managed local frame socket can be recreated safely"
            )
        payload = self._status_with_result(True, self.message)
        payload["cleanup"] = {
            "matched": matched_count,
            "remaining": 0,
            "frame_transport_ready": True,
        }
        return HTTPStatus.OK, payload

    @staticmethod
    def _signal_stale_records(
        records: list[dict[str, Any]], signum: int, timeout: float
    ) -> list[dict[str, Any]]:
        for record in records:
            if not stale_record_is_alive(record):
                continue
            try:
                os.kill(int(record["pid"]), signum)
            except (ProcessLookupError, PermissionError):
                pass
        deadline = time.monotonic() + timeout
        alive = records
        while time.monotonic() < deadline:
            alive = [record for record in records if stale_record_is_alive(record)]
            if not alive:
                return []
            time.sleep(0.1)
        return [record for record in records if stale_record_is_alive(record)]

    def shutdown(self) -> None:
        """Gracefully stop the managed pipeline before the web service exits."""
        self.request_stop(force=False, source="control service shutdown")
        with self.lock:
            thread = self.stop_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=18.0)
        with self.lock:
            process = self.process
        if process is not None and process.poll() is None:
            self._signal_process_group(process, signal.SIGKILL)
            self._wait_for_exit(process, 2.0)
        self._close_log()

    def _close_log(self) -> None:
        with self.lock:
            handle = self.log_handle
            self.log_handle = None
        if handle is not None:
            try:
                handle.flush()
                handle.close()
            except Exception:
                pass

    def downloadable_log(self) -> Path | None:
        with self.lock:
            if self.log_handle is not None:
                try:
                    self.log_handle.flush()
                except Exception:
                    pass
            path = self.session_log_path
        return path if path is not None and path.is_file() else None


# A browser can cancel a request or close a keep-alive connection at any time.
# Windows reports an aborted connection as ConnectionAbortedError (10053).
# Keep this list narrow: other I/O failures still need an error log/response.
_HTTP_CLIENT_DISCONNECT_ERRORS = (
    BrokenPipeError,
    ConnectionResetError,
    ConnectionAbortedError,
)


class LocalControlHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[BaseHTTPRequestHandler],
        manager: PipelineManager,
    ) -> None:
        super().__init__(server_address, handler_class)
        self.manager = manager

    def handle_error(self, request: Any, client_address: Any) -> None:
        error = sys.exc_info()[1]
        if isinstance(error, _HTTP_CLIENT_DISCONNECT_ERRORS):
            return
        self.manager._append_log(
            f"HTTP client {client_address} failed: {error}",
            component="http",
            level="error",
        )


class ControlRequestHandler(BaseHTTPRequestHandler):
    server_version = "ChingmuGemini/1.0"
    protocol_version = "HTTP/1.1"

    @property
    def manager(self) -> PipelineManager:
        return self.server.manager  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:
        # Browser polling is intentionally quiet.  Pipeline logs remain visible.
        if args and str(args[1]).startswith("5"):
            self.manager._append_log(fmt % args, component="http", level="error")

    def _common_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; style-src 'self' 'unsafe-inline'; "
            "script-src 'self'; img-src 'self' data:; connect-src 'self'",
        )

    def _send_json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._common_headers()
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, status: int, message: str) -> None:
        self._send_json(status, {"ok": False, "error": message})

    def _read_json(self) -> Any:
        raw_length = self.headers.get("Content-Length", "0")
        try:
            length = int(raw_length)
        except ValueError as exc:
            self.close_connection = True
            raise ConfigError("Invalid Content-Length") from exc
        if length < 0 or length > 65536:
            self.close_connection = True
            raise ConfigError("Request body is too large")
        body = self.rfile.read(length) if length else b""
        content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip()
        if content_type != "application/json":
            raise ConfigError("State-changing requests require application/json")
        if length == 0:
            return {}
        try:
            return json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ConfigError("Request body is not valid UTF-8 JSON") from exc

    def _discard_request_body(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.close_connection = True
            return
        if 0 < length <= 65536:
            self.rfile.read(length)
        elif length > 65536:
            self.close_connection = True

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        parsed = urllib.parse.urlsplit(self.path)
        path = parsed.path
        if path == "/api/health":
            self._send_json(
                HTTPStatus.OK,
                {
                    "service": SERVICE_NAME,
                    "ok": True,
                    "runtime": (
                        "wsl_linux"
                        if os.name == "posix" and Path("/proc").is_dir()
                        else "unsupported"
                    ),
                    "phase": self.manager.status(log_limit=0)["phase"],
                    "product_license": product_license_payload(),
                },
            )
            return
        if path == "/api/license":
            query = urllib.parse.parse_qs(parsed.query)
            status = (
                check_product_license()
                if query.get("refresh", ["0"])[0] == "1"
                else None
            )
            self._send_json(
                HTTPStatus.OK,
                (status.to_public_dict() if status else product_license_payload()),
            )
            return
        if path == "/api/config":
            self._send_json(HTTPStatus.OK, self.manager.config_payload())
            return
        if path == "/api/status":
            self._send_json(HTTPStatus.OK, self.manager.status())
            return
        if path == "/api/logs/download":
            self._download_log()
            return
        self._serve_static(path)

    def do_PUT(self) -> None:  # noqa: N802
        if not self._origin_is_local():
            self._discard_request_body()
            self._send_error_json(HTTPStatus.FORBIDDEN, "Cross-origin request denied")
            return
        parsed = urllib.parse.urlsplit(self.path)
        if parsed.path != "/api/config":
            self._send_error_json(HTTPStatus.NOT_FOUND, "Unknown API endpoint")
            return
        self._handle_config_update()

    def do_POST(self) -> None:  # noqa: N802
        if not self._origin_is_local():
            self._discard_request_body()
            self._send_error_json(HTTPStatus.FORBIDDEN, "Cross-origin request denied")
            return
        path = urllib.parse.urlsplit(self.path).path
        if path == "/api/config":
            self._handle_config_update()
            return
        try:
            payload = self._read_json()
        except ConfigError as exc:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
            return

        if path == "/api/safety/estop":
            status, result = self.manager.request_emergency_stop()
            self._send_json(status, result)
            return

        if path == "/api/safety/reset":
            status, result = self.manager.request_safety_reset(payload)
            self._send_json(status, result)
            return

        if path == "/api/preflight":
            config = None
            if payload:
                try:
                    proposed = payload.get("config", payload)
                    config = validate_config(proposed, self.manager.config)
                except (AttributeError, ConfigError) as exc:
                    self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
                    return
            result = self.manager.preflight(config)
            # Preflight is an inspection API: failed checks are data for the UI,
            # not an HTTP transport failure.  /api/start still refuses them.
            self._send_json(HTTPStatus.OK, result)
            return

        if path == "/api/start":
            authorization = (
                payload.get("authorization") if isinstance(payload, dict) else None
            )
            if payload:
                try:
                    if not isinstance(payload, dict):
                        raise ConfigError("Start request must be a JSON object")
                    if "config" in payload:
                        proposed = payload["config"]
                    else:
                        proposed = {
                            key: value
                            for key, value in payload.items()
                            if key != "authorization"
                        }
                    if proposed:
                        self.manager.update_config(proposed)
                except ConfigError as exc:
                    self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
                    return
                except RuntimeError as exc:
                    self._send_error_json(HTTPStatus.CONFLICT, str(exc))
                    return
            status, result = self.manager.start(authorization=authorization)
            self._send_json(status, result)
            return

        if path == "/api/stop":
            status, result = self.manager.request_stop(
                force=False, source="operator API /api/stop"
            )
            self._send_json(status, result)
            return
        if path == "/api/force-stop":
            status, result = self.manager.request_stop(
                force=True, source="operator API /api/force-stop"
            )
            self._send_json(status, result)
            return
        if path == "/api/cleanup":
            status, result = self.manager.cleanup()
            self._send_json(status, result)
            return
        if path == "/api/shutdown":
            self._send_json(
                HTTPStatus.ACCEPTED,
                {"ok": True, "message": "Service shutdown requested"},
            )
            threading.Thread(
                target=self._shutdown_server,
                name="http-shutdown",
                daemon=True,
            ).start()
            return
        self._send_error_json(HTTPStatus.NOT_FOUND, "Unknown API endpoint")

    def _origin_is_local(self) -> bool:
        origin = self.headers.get("Origin")
        if not origin:
            # Native launchers, curl, and readiness probes do not send Origin.
            return True
        try:
            parsed = urllib.parse.urlsplit(origin)
            expected_port = self.server.server_address[1]
            origin_port = parsed.port or (443 if parsed.scheme == "https" else 80)
            return (
                parsed.scheme == "http"
                and parsed.hostname in {"127.0.0.1", "localhost"}
                and origin_port == expected_port
            )
        except ValueError:
            return False

    def _handle_config_update(self) -> None:
        try:
            payload = self._read_json()
            proposed = payload.get("config", payload) if isinstance(payload, dict) else payload
            result = self.manager.update_config(proposed)
        except ConfigError as exc:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
            return
        except RuntimeError as exc:
            self._send_error_json(HTTPStatus.CONFLICT, str(exc))
            return
        except _HTTP_CLIENT_DISCONNECT_ERRORS:
            # A cancelled request body is a transport disconnect, not a failed
            # configuration write. Let the server close this request quietly.
            raise
        except OSError as exc:
            self._send_error_json(
                HTTPStatus.INTERNAL_SERVER_ERROR, f"Could not save configuration: {exc}"
            )
            return
        self._send_json(HTTPStatus.OK, result)

    def _shutdown_server(self) -> None:
        self.manager.shutdown()
        self.server.shutdown()

    def _download_log(self) -> None:
        path = self.manager.downloadable_log()
        if path is None:
            self._send_error_json(HTTPStatus.NOT_FOUND, "No session log is available")
            return
        try:
            size = path.stat().st_size
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(size))
            self.send_header(
                "Content-Disposition", f'attachment; filename="{path.name}"'
            )
            self._common_headers()
            self.end_headers()
            with path.open("rb") as handle:
                shutil.copyfileobj(handle, self.wfile)
        except _HTTP_CLIENT_DISCONNECT_ERRORS:
            # Cancelling a download must use the same connection cleanup path.
            raise
        except OSError as exc:
            if not self.wfile.closed:
                self.manager._append_log(
                    f"Log download failed: {exc}", component="http", level="error"
                )

    def _serve_static(self, request_path: str) -> None:
        relative = "index.html" if request_path in {"", "/"} else request_path.lstrip("/")
        try:
            target = (STATIC_DIR / urllib.parse.unquote(relative)).resolve()
            static_root = STATIC_DIR.resolve()
            target.relative_to(static_root)
        except (ValueError, OSError):
            self._send_error_json(HTTPStatus.NOT_FOUND, "Static file not found")
            return
        if not target.is_file():
            # A client-side route receives the app shell; missing assets remain 404.
            if "." not in Path(relative).name and (STATIC_DIR / "index.html").is_file():
                target = STATIC_DIR / "index.html"
            else:
                self._send_error_json(HTTPStatus.NOT_FOUND, "Static file not found")
                return
        try:
            body = target.read_bytes()
        except OSError as exc:
            self._send_error_json(
                HTTPStatus.INTERNAL_SERVER_ERROR, f"Could not read static file: {exc}"
            )
            return
        mime_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        if mime_type.startswith("text/") or mime_type in {
            "application/javascript",
            "application/json",
        }:
            mime_type += "; charset=utf-8"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime_type)
        self.send_header("Content-Length", str(len(body)))
        self._common_headers()
        self.end_headers()
        self.wfile.write(body)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CHINGMU G1 local motion control UI")
    parser.add_argument(
        "--host",
        choices=["127.0.0.1", "localhost"],
        default="127.0.0.1",
        help="Loopback address only (default: 127.0.0.1)",
    )
    parser.add_argument("--port", type=int, default=8765)
    browser = parser.add_mutually_exclusive_group()
    browser.add_argument("--open-browser", dest="open_browser", action="store_true")
    browser.add_argument("--no-browser", dest="open_browser", action="store_false")
    parser.set_defaults(open_browser=True)
    args = parser.parse_args(argv)
    if not 1024 <= args.port <= 65535:
        parser.error("--port must be between 1024 and 65535")
    if args.host == "localhost":
        args.host = "127.0.0.1"
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manager = PipelineManager()
    server = LocalControlHTTPServer(
        (args.host, args.port), ControlRequestHandler, manager
    )
    url = f"http://{args.host}:{args.port}/"
    stopping = threading.Event()

    def request_server_shutdown(signum: int, _frame: Any) -> None:
        if stopping.is_set():
            return
        stopping.set()
        manager._append_log(
            f"Service received signal {signum}", component="service", level="warning"
        )
        threading.Thread(target=server.shutdown, daemon=True).start()

    handled_signals = [signal.SIGINT, signal.SIGTERM]
    if hasattr(signal, "SIGHUP"):
        handled_signals.append(signal.SIGHUP)
    for handled_signal in handled_signals:
        signal.signal(handled_signal, request_server_shutdown)

    if args.open_browser:
        threading.Timer(0.7, lambda: webbrowser.open(url)).start()

    print(f"{SERVICE_NAME} listening on {url}", flush=True)
    print(
        "No Unitree DDS or motor command interface is loaded until an operator "
        "explicitly starts an experimental real/parallel mode.",
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        manager.shutdown()
        print(f"{SERVICE_NAME} stopped.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise


# ===================================================================== #
# M10b: migrated helpers — these aliases keep the module surface intact. #
# ===================================================================== #
SAFETY_CONTROL_KEY = _channel.SAFETY_CONTROL_KEY
SAFETY_STATUS_KEY = _channel.SAFETY_STATUS_KEY
SAFETY_REDIS_ADDRESS = _channel.SAFETY_REDIS_ADDRESS
SAFETY_REDIS_TIMEOUT_SECONDS = _channel.SAFETY_REDIS_TIMEOUT_SECONDS
SAFETY_STATES = _channel.SAFETY_STATES
REFERENCE_HEALTH_STATES = _channel.REFERENCE_HEALTH_STATES
_redis_command = _channel._redis_command
write_safety_control_command = _channel.write_safety_control_command
_redis_get = _channel._redis_get
_unavailable_safety_status = _channel._unavailable_safety_status
read_safety_status = _channel.read_safety_status
classify_component = _classify_component_impl
classify_level = _classify_level_impl
build_launch_environment = _config_impl.build_launch_environment
