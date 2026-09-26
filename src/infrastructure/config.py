"""Configuration schema, validation and pure native normalization."""

from __future__ import annotations

import ipaddress
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Mapping

from src.domain.errors import ConfigurationError
from src.domain.g1_versions import (
    DEFAULT_MODEL_PROFILE,
    model_profile_from_config,
    require_model_profile,
)

# Frozen native defaults (architecture/PHASE0_INTERFACE_FREEZE.md §5).
FROZEN_DEFAULTS: dict[str, Any] = {
    "backend": "humanoid_gpt",
    "execution_mode": "simulation_only",
    "debug_mode": True,
    "safety_strategy_enabled": False,
    "server_ip": "127.0.0.1",
    "skeleton_id": 0,
    "human_height": 1.75,
    "publish_fps": 50.0,
    "transition_seconds": 1.5,
    "device": "cpu",
    "model_profile": DEFAULT_MODEL_PROFILE,
    "unitree_network_interface": "192.168.123.100",
    "unitree_robot_ip": "192.168.123.164",
}

CONFIG_FIELDS: dict[str, str] = {
    "backend": "enum {humanoid_gpt, gmr_preview}",
    "execution_mode": "enum {simulation_only, real_only, parallel}",
    "debug_mode": "bool (default true)",
    "safety_strategy_enabled": "bool (default false; humanoid_gpt sim/parallel only)",
    "server_ip": "IPv4 (MCAvatar host)",
    "skeleton_id": "int 0..255",
    "human_height": "float m, 1.2..2.2",
    "publish_fps": "float Hz, 20..60 (policy requires 50)",
    "transition_seconds": "float s, 0.5..5.0",
    "device": "forced to cpu on Windows",
    "model_profile": "internal packaged model asset directory; independent of SDK mode_machine",
    "unitree_network_interface": "local adapter IPv4 (non-loopback) or legacy names",
    "unitree_robot_ip": "IPv4 (robot address)",
}


def validate_unitree_windows_address(value: Any) -> str:
    """Validate the non-loopback Windows adapter IPv4 address."""
    try:
        address = ipaddress.ip_address(str(value).strip())
    except ValueError as exc:
        raise ConfigError(
            "unitree_network_interface must be the Windows adapter IPv4 address"
        ) from exc
    if not isinstance(address, ipaddress.IPv4Address):
        raise ConfigError("Unitree Windows DDS currently requires IPv4")
    if (
        address.is_loopback
        or address.is_unspecified
        or address.is_multicast
        or address.is_link_local
    ):
        raise ConfigError(
            "select the non-loopback Windows Ethernet IPv4 connected to G1"
        )
    return str(address)


class ConfigurationService:
    """Read-only configuration contract + pure native normalization mirror."""

    @property
    def frozen_defaults(self) -> dict[str, Any]:
        return dict(FROZEN_DEFAULTS)

    def load_snapshot(self, path: Path) -> dict[str, Any]:
        """Read one config file as JSON (never writes, never normalizes)."""
        try:
            loaded = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigurationError(f"cannot read configuration: {exc}") from exc
        if not isinstance(loaded, dict):
            raise ConfigurationError("configuration must be a JSON object")
        return loaded

    def validate_base(
        self, candidate: Any, current: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        """Validate with the module-local ``validate_config`` (pure, no disk)."""
        try:
            return dict(
                validate_config(candidate, dict(current) if current else None)
            )
        except ConfigError as exc:
            raise ConfigurationError(str(exc)) from exc

    def _validate_unitree_address(self, value: Any) -> str:
        try:
            return validate_unitree_windows_address(value)
        except ConfigError as exc:
            raise ConfigurationError(str(exc)) from exc

    def normalize_native(
        self, candidate: Any, current: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        """Pure mirror of the legacy native ``update_config`` narrowing.

        Same rules, same errors, no ``persist_config`` call.
        """
        validated = self.validate_base(candidate, current)
        if validated["backend"] not in {"humanoid_gpt", "gmr_preview"}:
            raise ConfigurationError(
                "Gemini Native supports real-time generative teleoperation "
                "or GMR preview"
            )
        if (
            validated["backend"] == "gmr_preview"
            and validated["execution_mode"] != "simulation_only"
        ):
            raise ConfigurationError("GMR preview supports simulation only")
        if validated["execution_mode"] in {"real_only", "parallel"}:
            validated["unitree_network_interface"] = self._validate_unitree_address(
                validated["unitree_network_interface"]
            )
        validated["device"] = "cpu"
        validated["model_profile"] = require_model_profile(validated["model_profile"])
        return validated

    def migration_preview(self, current: Mapping[str, Any]) -> dict[str, Any]:
        """Return the normalized configuration a startup would persist."""
        migrated = dict(current)
        if migrated.get("backend") not in {"humanoid_gpt", "gmr_preview"}:
            migrated["backend"] = "humanoid_gpt"
        if migrated.get("execution_mode") not in {
            "simulation_only",
            "real_only",
            "parallel",
        }:
            migrated["execution_mode"] = "simulation_only"
        if (
            migrated["backend"] == "gmr_preview"
            and migrated["execution_mode"] != "simulation_only"
        ):
            migrated["execution_mode"] = "simulation_only"
        migrated["device"] = "cpu"
        migrated["model_profile"] = model_profile_from_config(migrated)
        migrated.pop("g1_version", None)
        if migrated.get("unitree_network_interface") in {
            None,
            "",
            "eth0",
            "lo",
        }:
            migrated["unitree_network_interface"] = "192.168.123.100"
        return migrated


# Shared defaults before native-Windows normalization.
BASE_DEFAULT_CONFIG: dict[str, Any] = {
    "backend": "twist",
    "execution_mode": "simulation_only",
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

CONFIG_KEYS = frozenset(BASE_DEFAULT_CONFIG)
_LEGACY_BACKENDS = frozenset({"gmr_preview", "twist", "humanoid_gpt"})


class ConfigError(ValueError):
    """Raised when a configuration supplied by the browser is unsafe."""


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


def validate_config(
    candidate: Any,
    base: dict[str, Any] | None = None,
    *,
    defaults: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(candidate, dict):
        raise ConfigError("configuration must be a JSON object")
    unknown = sorted(set(candidate) - CONFIG_KEYS - {"g1_version"})
    if unknown:
        raise ConfigError(f"unknown configuration field(s): {', '.join(unknown)}")

    # Always seed from defaults so saved files and callers from older versions
    # remain compatible when a new optional setting is introduced.  The legacy
    # caller passes its (possibly native-mutated) DEFAULT_CONFIG as ``defaults``.
    result: dict[str, Any] = {}
    # Migrate each layer before merging so an older saved model cannot be
    # hidden by the new default. Never retain a hardware selector in output.
    for layer in (BASE_DEFAULT_CONFIG if defaults is None else defaults, base, candidate):
        if layer is None:
            continue
        migrated_layer = dict(layer)
        if "model_profile" in migrated_layer or "g1_version" in migrated_layer:
            try:
                migrated_layer["model_profile"] = model_profile_from_config(migrated_layer)
            except ValueError as exc:
                raise ConfigError(str(exc)) from exc
        migrated_layer.pop("g1_version", None)
        result.update(migrated_layer)

    backend = str(result["backend"]).strip()
    if backend not in _LEGACY_BACKENDS:
        raise ConfigError("backend must be gmr_preview, twist, or humanoid_gpt")
    result["backend"] = backend

    execution_mode = str(result["execution_mode"]).strip().lower()
    if execution_mode not in {
        "simulation_only",
        "real_only",
        "parallel",
        "shadow",
    }:
        raise ConfigError(
            "execution_mode must be simulation_only, real_only, parallel, or shadow"
        )
    result["execution_mode"] = execution_mode

    result["debug_mode"] = _as_bool(result["debug_mode"], "debug_mode")
    result["safety_strategy_enabled"] = _as_bool(
        result["safety_strategy_enabled"], "safety_strategy_enabled"
    )
    if result["safety_strategy_enabled"] and not (
        backend == "humanoid_gpt"
        and execution_mode in {"simulation_only", "parallel"}
    ):
        raise ConfigError(
            "the experimental safety strategy is currently implemented only for "
            "the Humanoid-GPT MuJoCo branch; keep it disabled for TWIST, "
            "real-only, preview, and shadow modes"
        )
    if backend == "gmr_preview" and execution_mode != "simulation_only":
        raise ConfigError("GMR preview supports simulation_only mode only")

    server_ip = str(result["server_ip"]).strip()
    try:
        parsed_ip = ipaddress.ip_address(server_ip)
    except ValueError as exc:
        raise ConfigError("server_ip must be a valid IPv4 address") from exc
    if parsed_ip.version != 4:
        raise ConfigError("server_ip must be an IPv4 address")
    result["server_ip"] = str(parsed_ip)

    skeleton_id = result["skeleton_id"]
    if isinstance(skeleton_id, bool):
        raise ConfigError("skeleton_id must be an integer")
    try:
        skeleton_id = int(skeleton_id)
    except (TypeError, ValueError) as exc:
        raise ConfigError("skeleton_id must be an integer") from exc
    if skeleton_id < 0 or skeleton_id > 255:
        raise ConfigError("skeleton_id must be between 0 and 255")
    result["skeleton_id"] = skeleton_id

    result["human_height"] = _as_finite_float(
        result["human_height"], "human_height", 1.2, 2.2
    )
    result["publish_fps"] = _as_finite_float(
        result["publish_fps"], "publish_fps", 20.0, 60.0
    )
    result["transition_seconds"] = _as_finite_float(
        result["transition_seconds"], "transition_seconds", 0.5, 5.0
    )

    device = str(result["device"]).strip().lower()
    if device not in {"auto", "cpu", "cuda"}:
        raise ConfigError("device must be auto, cpu, or cuda")
    result["device"] = device

    try:
        result["model_profile"] = require_model_profile(result["model_profile"])
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc

    network_interface = str(result["unitree_network_interface"]).strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}", network_interface):
        raise ConfigError(
            "unitree_network_interface must be a Linux interface name such as eth0 or enx..."
        )
    result["unitree_network_interface"] = network_interface

    robot_ip = str(result["unitree_robot_ip"]).strip()
    try:
        parsed_robot_ip = ipaddress.ip_address(robot_ip)
    except ValueError as exc:
        raise ConfigError("unitree_robot_ip must be a valid IPv4 address") from exc
    if parsed_robot_ip.version != 4:
        raise ConfigError("unitree_robot_ip must be an IPv4 address")
    result["unitree_robot_ip"] = str(parsed_robot_ip)
    return result


def persist_config(config: dict[str, Any], path: Path) -> None:
    """Atomic persist; identical to the legacy ``persist_config`` behavior."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_config(path: Path, defaults: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    """Load settings and atomically migrate the retired hardware selector."""
    path = Path(path)
    if not path.exists():
        return dict(defaults), None
    try:
        with path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        validated = validate_config(loaded, defaults=defaults)
    except Exception as exc:
        return dict(defaults), f"Invalid saved configuration; defaults loaded: {exc}"
    if "g1_version" in loaded:
        try:
            persist_config(validated, path)
        except OSError as exc:
            return validated, f"Saved settings loaded; model profile migration could not be persisted: {exc}"
    return validated, None


def build_launch_environment(
    config: dict[str, Any], base_environment: dict[str, str] | None = None
) -> dict[str, str]:
    """Build the launcher environment from a validated configuration."""

    env = dict(os.environ if base_environment is None else base_environment)
    env.pop("G1_VERSION", None)
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "CHINGMU_SERVER_IP": str(config["server_ip"]),
            "CHINGMU_HUMAN_ID": str(config["skeleton_id"]),
            "CHINGMU_HUMAN_HEIGHT": f"{config['human_height']:.6g}",
            "MOTION_PUBLISH_FPS": f"{config['publish_fps']:.6g}",
            "CHINGMU_TRANSITION_SECONDS": f"{config['transition_seconds']:.6g}",
            "TWIST_DEVICE": str(config["device"]),
            "HGPT_DEVICE": (
                "cpu" if config["device"] == "auto" else str(config["device"])
            ),
            "G1_MODEL_PROFILE": model_profile_from_config(config),
            "HGPT_SAFETY_STRATEGY_ENABLED": (
                "1" if config["safety_strategy_enabled"] else "0"
            ),
            "HGPT_DEBUG_MODE": "1" if config["debug_mode"] else "0",
            "CONTROL_BACKEND": str(config["backend"]),
            "CONTROL_EXECUTION_MODE": str(config["execution_mode"]),
            "RUN_MUJOCO": "1" if config["execution_mode"] == "parallel" else "0",
            "UNITREE_REAL_DEBUG": "1" if config["debug_mode"] else "0",
            "UNITREE_NETWORK_INTERFACE": str(config["unitree_network_interface"]),
            "UNITREE_ROBOT_IP": str(config["unitree_robot_ip"]),
        }
    )
    return env
