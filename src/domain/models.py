"""Domain data models (Phase 1).

Every class documents: type, dimensions, units, coordinate frame, time base,
source/target, sequence/schema version and freshness requirement.  Arrays are
stored as ``numpy.ndarray`` (read/write copies are the caller's job); models
are ``frozen`` (dataclass) and validate shapes in ``__post_init__``.

No model imports device SDKs.  Wire-format conversion lives in
``src.domain.legacy_codec``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Sequence

import numpy as np

from src.domain.enums import (
    Backend,
    ExecutionMode,
    ReferenceHealth,
    RobotStateSource,
    SafetyState,
    TaskPhase,
)
from src.domain.errors import ValidationError

# GMR Unitree G1 model: root position (3) + root quaternion wxyz (4) + 29 DoF.
G1_QPOS_SIZE = 36
G1_DOF = 29
# TWIST mimic target consumed by the released tracking policy.
MIMIC_TARGET_SIZE = 33
# CHINGMU live capture: up to 150 segments; standard skeleton uses first 23.
CHINGMU_MAX_SEGMENTS = 150
CHINGMU_STANDARD_SEGMENTS = 23


def _require_float_array(
    value: Any, shape: tuple[int, ...], name: str
) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != shape:
        raise ValidationError(
            f"{name} must have shape {shape}, got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise ValidationError(f"{name} contains NaN or infinity")
    return array


@dataclass(frozen=True, eq=False)
class MocapFrame:
    """One decoded CHINGMU skeleton frame (transport-agnostic).

    Dimensions/units:
      * positions: (N, 3) float64, millimeters, CHINGMU live Z-up frame;
      * rotations: (N, 4) float64, local (segment-parent) quaternions xyzw;
      * detected:  (N,) bool, per-segment SDK detection flag.
    Time base: ``frame_id`` is the SDK timecode (monotonic integer);
    ``received_monotonic`` is the consumer's ``time.monotonic()``.
    Freshness: consumers must reject frames whose frame_id stopped advancing
    or whose received age exceeds the bridge timeout (see G1Reference).
    """

    frame_id: int
    positions: np.ndarray
    rotations: np.ndarray
    detected: np.ndarray
    session_id: str = ""
    source: str = "chingmu_windows_udp"
    received_monotonic: float | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.frame_id, int):
            raise ValidationError("frame_id must be an integer")
        positions = _require_float_array(
            self.positions, (len(self.positions), 3), "positions"
        )
        rotations = _require_float_array(
            self.rotations, (len(self.rotations), 4), "rotations"
        )
        if positions.shape[0] != rotations.shape[0]:
            raise ValidationError("positions and rotations segment counts differ")
        detected = np.asarray(self.detected)
        if detected.ndim != 1 or detected.shape[0] != positions.shape[0]:
            raise ValidationError("detected must be a 1-D flag per segment")
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "rotations", rotations)
        object.__setattr__(self, "detected", detected.astype(bool))
        if self.schema_version != 1:
            raise ValidationError("unsupported MocapFrame schema version")

    @property
    def segment_count(self) -> int:
        return int(self.positions.shape[0])

    @property
    def standard_segments_detected(self) -> int:
        limit = min(CHINGMU_STANDARD_SEGMENTS, self.segment_count)
        return int(np.count_nonzero(self.detected[:limit]))


@dataclass(frozen=True, eq=False)
class G1Reference:
    """GMR reference packet as published on ``action_qpos_g1_packet``.

    Dimensions/units:
      * qpos: (36,) float64 G1 generalized position
              [root_pos(3, m), root_quat_wxyz(4, unit), dof(29, rad)];
      * mimic: (33,) float32 TWIST mimic target, optional in this packet.
    Time base: ``generated_monotonic`` / ``valid_until_monotonic`` are
    ``time.monotonic()`` of the producing process; comparable across local
    processes on the same machine only.
    Freshness: consumers must verify ``sequence`` increased AND the current
    monotonic clock is before ``valid_until_monotonic``; a repeatedly-read
    identical Redis value is NOT a new frame.
    """

    session_id: str
    sequence: int
    frame_id: int
    generated_monotonic: float
    valid_until_monotonic: float
    robot_model: str
    dof_order_hash: str
    qpos: np.ndarray | None
    mimic: np.ndarray | None = None
    source: str = "g1_gmr_bridge"
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValidationError("unsupported G1Reference schema version")
        if not isinstance(self.sequence, int) or self.sequence < 1:
            raise ValidationError("sequence must be a positive integer")
        if self.qpos is None and self.mimic is None:
            raise ValidationError("qpos packet and mimic packet may not both be absent")
        if self.qpos is not None:
            object.__setattr__(
                self, "qpos", _require_float_array(self.qpos, (G1_QPOS_SIZE,), "qpos")
            )
        if self.mimic is not None:
            mimic = _require_float_array(self.mimic, (MIMIC_TARGET_SIZE,), "mimic")
            object.__setattr__(self, "mimic", mimic.astype(np.float32))
        if not math.isfinite(self.generated_monotonic):
            raise ValidationError("generated_monotonic must be finite")
        if not math.isfinite(self.valid_until_monotonic):
            raise ValidationError("valid_until_monotonic must be finite")

    def age_seconds(self, now_monotonic: float) -> float:
        return max(0.0, float(now_monotonic) - float(self.generated_monotonic))

    def is_fresh(self, now_monotonic: float) -> bool:
        return float(now_monotonic) <= float(self.valid_until_monotonic)

    @property
    def root_position(self) -> np.ndarray:
        if self.qpos is None:
            raise ValidationError("packet carries no qpos")
        return self.qpos[:3].copy()

    @property
    def dof_position(self) -> np.ndarray:
        if self.qpos is None:
            raise ValidationError("packet carries no qpos")
        return self.qpos[7:].copy()


@dataclass(frozen=True, eq=False)
class RobotState:
    """Generalized robot state snapshot from one backend.

    Dimensions: qpos (nq,) / qvel (nv, optional); motor_state (35,) for
    Unitree LowState with per-motor id/state arrays (kept as raw ndarray).
    Time base: ``observed_monotonic`` is the local reading time.
    """

    source: RobotStateSource
    qpos: np.ndarray
    qvel: np.ndarray | None = None
    observed_monotonic: float | None = None
    imu: np.ndarray | None = None
    motor_state: np.ndarray | None = None
    lowstate_received_monotonic: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source, RobotStateSource):
            raise ValidationError("source must be a RobotStateSource")
        qpos = np.asarray(self.qpos, dtype=float)
        if qpos.ndim != 1 or qpos.shape[0] == 0:
            raise ValidationError("qpos must be a non-empty 1-D array")
        object.__setattr__(self, "qpos", qpos)
        if self.qvel is not None:
            qvel = np.asarray(self.qvel, dtype=float)
            # nv may differ from nq (e.g. MuJoCo quaternion joints: nq=36, nv=35).
            if qvel.ndim != 1 or qvel.shape[0] == 0:
                raise ValidationError("qvel must be a non-empty 1-D array")
            object.__setattr__(self, "qvel", qvel)


@dataclass(frozen=True, eq=False)
class PolicyObservation:
    """Observation vector consumed by the ONNX policy.

    Dimensions: values (obs_dim,) float32 in the policy's pinned layout;
    policy constants are owned by the legacy Humanoid-GPT tracking code and
    must not be duplicated here beyond the schema tag.
    """

    values: np.ndarray
    schema: str
    timestamp_monotonic: float | None = None

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float32)
        if values.ndim != 1:
            raise ValidationError("observation must be a 1-D vector")
        object.__setattr__(self, "values", values)


@dataclass(frozen=True, eq=False)
class MotorCommand:
    """One 29-DoF motor target (post safety limiting).

    Dimensions: target_qpos (29,) float32 radians, joint order pinned by
    ``GMR_29_DOF_NAMES`` / policy constants (order hash checked upstream).
    Freshness: single-use; a stale MotorCommand must never be applied twice.
    """

    target_qpos: np.ndarray
    timestamp_monotonic: float
    source: str = "policy"

    def __post_init__(self) -> None:
        target = _require_float_array(self.target_qpos, (G1_DOF,), "target_qpos")
        object.__setattr__(self, "target_qpos", target.astype(np.float32))
        if not math.isfinite(self.timestamp_monotonic):
            raise ValidationError("timestamp_monotonic must be finite")


@dataclass(frozen=True, eq=False)
class SafeCommand:
    """Command on the ``g1_safety_control`` channel.

    RESET additionally requires both operator confirmations (the legacy
    writer refuses to publish otherwise).  Latching behavior (estop/fault) is
    owned by the safety state machine and must never be bypassed here.
    """

    sequence: int
    action: str
    operator_confirmed: bool = False
    physical_estop_released: bool = False

    def __post_init__(self) -> None:
        if self.action not in {"estop", "reset"}:
            raise ValidationError("safety action must be estop or reset")
        if self.action == "reset" and not (
            self.operator_confirmed and self.physical_estop_released
        ):
            raise ValidationError(
                "safety reset requires both operator confirmations"
            )


@dataclass(frozen=True, eq=False)
class SafetyStatus:
    """Fail-closed safety telemetry snapshot (schema_version 1).

    ``healthy`` is True only when no latch is active and the state is not a
    terminal/legacy state — matching the legacy reader semantics.
    """

    state: SafetyState
    reason: str
    estop_latched: bool
    fault_latched: bool
    reference_health: ReferenceHealth
    state_source: RobotStateSource
    real_hardware_allowed: bool
    teleop_weight: float
    reference_age_ms: float
    strategy_enabled: bool
    debug_mode: bool
    bypass_active: bool
    hard_guards_enabled: bool
    limited: bool
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValidationError("unsupported safety status schema")
        if not isinstance(self.state, SafetyState):
            raise ValidationError("state must be a SafetyState")
        if not isinstance(self.reference_health, ReferenceHealth):
            raise ValidationError("reference_health must be a ReferenceHealth")
        if not isinstance(self.state_source, RobotStateSource):
            raise ValidationError("state_source must be a RobotStateSource")
        if not 0.0 <= self.teleop_weight <= 1.0:
            raise ValidationError("teleop_weight must be within [0, 1]")
        if self.reference_age_ms < 0.0:
            raise ValidationError("reference_age_ms must be non-negative")

    @property
    def healthy(self) -> bool:
        return (
            self.state not in {SafetyState.E_STOP, SafetyState.FAULT,
                               SafetyState.LEGACY_DIRECT}
            and not self.estop_latched
            and not self.fault_latched
        )


@dataclass(frozen=True, eq=False)
class CheckItem:
    """One preflight check entry (name/ok/message/required)."""

    name: str
    ok: bool
    message: str
    required: bool = True


@dataclass(frozen=True, eq=False)
class PreflightResult:
    """Preflight outcome; failed optional checks do not flip ``ok``."""

    ok: bool
    checks: tuple[CheckItem, ...]
    errors: tuple[str, ...] = ()

    @property
    def failed_required(self) -> tuple[CheckItem, ...]:
        return tuple(item for item in self.checks if item.required and not item.ok)


@dataclass(frozen=True, eq=False)
class PipelineStatus:
    """Task status snapshot compatible with the legacy ``/api/status`` payload."""

    phase: TaskPhase
    running: bool
    active: bool
    backend: Backend | None = None
    pid: int | None = None
    started_at: str | None = None
    uptime_seconds: float = 0.0
    exit_code: int | None = None
    error: str | None = None
    message: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    session_log: str | None = None


@dataclass(frozen=True, eq=False)
class AuthorizationContext:
    """Session-bound binding for the one-shot real-output chain.

    Every field is part of the capability binding hash; any change invalidates
    a previously issued capability (legacy ``binding_from_control_config``).
    """

    backend: str
    execution_mode: str
    debug_mode: bool
    startup_handover_enabled: bool
    server_ip: str
    skeleton_id: int
    model_profile: str
    unitree_network_interface: str
    unitree_robot_ip: str
    mocap_type: str = "chingmu_redis"
    redis_host: str = "127.0.0.1"
    redis_port: int = 6379
    redis_key: str = "action_qpos_g1_packet"
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValidationError("unsupported authorization schema")
        if self.backend not in {member.value for member in Backend}:
            raise ValidationError("unsupported authorization backend")
        if self.execution_mode not in {member.value for member in ExecutionMode}:
            raise ValidationError("unsupported authorization execution_mode")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "backend": self.backend,
            "execution_mode": self.execution_mode,
            "debug_mode": self.debug_mode,
            "startup_handover_enabled": self.startup_handover_enabled,
            "server_ip": self.server_ip,
            "skeleton_id": self.skeleton_id,
            "model_profile": self.model_profile,
            "unitree_network_interface": self.unitree_network_interface,
            "unitree_robot_ip": self.unitree_robot_ip,
            "mocap_type": self.mocap_type,
            "redis_host": self.redis_host,
            "redis_port": self.redis_port,
            "redis_key": self.redis_key,
        }

    @classmethod
    def from_control_config(cls, config: dict[str, Any]) -> "AuthorizationContext":
        """Map a legacy control config to the binding (mirrors the old broker).

        ``startup_handover_enabled`` is forced True for real output by the
        legacy controller; callers outside that flow must pass it explicitly.
        """
        return cls(
            backend=str(config["backend"]),
            execution_mode=str(config["execution_mode"]),
            debug_mode=bool(config["debug_mode"]),
            startup_handover_enabled=bool(config.get("startup_handover_enabled", True)),
            server_ip=str(config["server_ip"]),
            skeleton_id=int(config["skeleton_id"]),
            model_profile=str(config["model_profile"]),
            unitree_network_interface=str(config["unitree_network_interface"]),
            unitree_robot_ip=str(config["unitree_robot_ip"]),
        )
