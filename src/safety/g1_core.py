"""Transport-independent safety state machine for a Unitree G1 control pipeline.

This package deliberately contains no DDS, Unitree SDK, or MuJoCo imports.  The
default profile is simulation-only; a hardware transport must provide its own
reviewed profile and an independent physical emergency stop.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import math
from typing import Deque, Iterable, Optional, Sequence, Tuple

import numpy as np


G1_29DOF_JOINT_NAMES: Tuple[str, ...] = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
    "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)

# Source of these simulation limits: TWIST/assets/g1/g1_29dof_rev_1_0.xml.
# They are model metadata, not authorization to command physical hardware.
G1_29DOF_Q_MIN = np.array([
    -2.5307, -0.5236, -2.7576, -0.087267, -0.87267, -0.2618,
    -2.5307, -2.9671, -2.7576, -0.087267, -0.87267, -0.2618,
    -2.618, -0.52, -0.52,
    -3.0892, -1.5882, -2.618, -1.0472, -1.97222, -1.61443, -1.61443,
    -3.0892, -2.2515, -2.618, -1.0472, -1.97222, -1.61443, -1.61443,
], dtype=np.float64)
G1_29DOF_Q_MAX = np.array([
    2.8798, 2.9671, 2.7576, 2.8798, 0.5236, 0.2618,
    2.8798, 0.5236, 2.7576, 2.8798, 0.5236, 0.2618,
    2.618, 0.52, 0.52,
    2.6704, 2.2515, 2.618, 2.0944, 1.97222, 1.61443, 1.61443,
    2.6704, 1.5882, 2.618, 2.0944, 1.97222, 1.61443, 1.61443,
], dtype=np.float64)
G1_29DOF_TORQUE = np.array([
    88, 139, 88, 139, 50, 50,
    88, 139, 88, 139, 50, 50,
    88, 50, 50,
    25, 25, 25, 25, 25, 5, 5,
    25, 25, 25, 25, 25, 5, 5,
], dtype=np.float64)
G1_29DOF_STAND = np.array([
    0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0,
    0, 0, 0,
    0.2, 0.2, 0, 1.28, 0, 0, 0,
    0.2, -0.2, 0, 1.28, 0, 0, 0,
], dtype=np.float64)


def compute_dof_hash(names: Iterable[str]) -> str:
    canonical = "\n".join(str(name).strip() for name in names)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def minimum_jerk(progress: float) -> float:
    u = float(np.clip(progress, 0.0, 1.0))
    return u * u * u * (10.0 + u * (-15.0 + 6.0 * u))


class SafetyState(str, Enum):
    DISARMED = "DISARMED"
    ARMING = "ARMING"
    BLEND_IN = "BLEND_IN"
    STANDBY = "STANDBY"
    TELEOP = "TELEOP"
    INPUT_LOSS = "INPUT_LOSS"
    RECOVER_STAND = "RECOVER_STAND"
    E_STOP = "E_STOP"
    FAULT = "FAULT"


class ValidationError(ValueError):
    pass


def _vector(value: Sequence[float], size: int, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1).copy()
    if array.shape != (size,):
        raise ValidationError(f"{label} shape {array.shape}, expected ({size},)")
    if not np.all(np.isfinite(array)):
        raise ValidationError(f"{label} contains NaN or infinity")
    return array


@dataclass
class PolicyCommand:
    seq: int
    timestamp: float
    q: Sequence[float]
    dq: Sequence[float]
    tau_ff: Sequence[float]
    kp: Sequence[float]
    kd: Sequence[float]
    model: str
    dof_hash: str
    deadman: bool = True
    source_valid: bool = True
    active_mask: Optional[Sequence[bool]] = None

    def arrays(self, dof: int) -> Tuple[np.ndarray, ...]:
        return tuple(
            _vector(value, dof, name)
            for value, name in (
                (self.q, "policy.q"), (self.dq, "policy.dq"),
                (self.tau_ff, "policy.tau_ff"), (self.kp, "policy.kp"),
                (self.kd, "policy.kd"),
            )
        )

    def active(self, dof: int) -> np.ndarray:
        if self.active_mask is None:
            return np.ones(dof, dtype=bool)
        values = list(self.active_mask)
        if len(values) != dof or any(
            not isinstance(value, (bool, np.bool_)) for value in values
        ):
            raise ValidationError(
                f"policy.active_mask must contain exactly {dof} booleans"
            )
        return np.asarray(values, dtype=bool)


@dataclass
class RobotState:
    timestamp: float
    tick: int
    q: Sequence[float]
    dq: Sequence[float]
    imu_rpy: Sequence[float]
    imu_gyro: Sequence[float]
    model: str
    dof_hash: str
    valid: bool = True

    def arrays(self, dof: int) -> Tuple[np.ndarray, ...]:
        return (
            _vector(self.q, dof, "robot.q"),
            _vector(self.dq, dof, "robot.dq"),
            _vector(self.imu_rpy, 3, "robot.imu_rpy"),
            _vector(self.imu_gyro, 3, "robot.imu_gyro"),
        )


@dataclass
class SafeCommand:
    timestamp: float
    seq: int
    state: SafetyState
    q: np.ndarray
    dq: np.ndarray
    tau_ff: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    teleop_weight: float
    reason: str
    limited: bool
    estop_latched: bool
    fault_latched: bool
    command_active: bool
    real_hardware_allowed: bool = False
    predicted_torque: np.ndarray = field(default_factory=lambda: np.zeros(0))


@dataclass
class SafetyProfile:
    dof: int
    model: str
    dof_hash: str
    q_min: Sequence[float]
    q_max: Sequence[float]
    standby_q: Sequence[float]
    max_velocity: Sequence[float]
    max_acceleration: Sequence[float]
    max_predicted_torque: Sequence[float]
    position_margin: Sequence[float] | float = 0.05
    # The released Humanoid-GPT zero-velocity walk/balance policy uses
    # Kp=300 on the three waist joints.  This is a simulation validation
    # ceiling, not permission to use the same gain on physical hardware.
    max_kp: float = 300.0
    max_kd: float = 20.0
    input_soft_stale_s: float = 0.080
    input_hard_stale_s: float = 0.250
    state_stale_s: float = 0.050
    policy_output_stale_s: float = 0.050
    future_tolerance_s: float = 0.020
    fresh_dwell_s: float = 0.500
    arming_dwell_s: float = 0.100
    blend_in_s: float = 0.750
    recovery_s: float = 1.000
    control_period_s: float = 0.002
    max_slew_dt_s: float = 0.020
    imu_soft_tilt_rad: float = math.radians(25.0)
    imu_hard_tilt_rad: float = math.radians(45.0)
    imu_soft_gyro_rad_s: float = 3.0
    imu_hard_gyro_rad_s: float = 6.0
    clip_event_limit: int = 4
    clip_window_s: float = 0.500
    standby_kp: float = 20.0
    standby_kd: float = 1.5
    estop_kd: float = 8.0
    simulation_only: bool = True
    simulation_joint_interpolation_test: bool = False
    profile_name: str = "g1-29dof-simulation-only"

    def __post_init__(self) -> None:
        if self.dof <= 0:
            raise ValidationError("dof must be positive")
        for name in (
            "q_min", "q_max", "standby_q", "max_velocity",
            "max_acceleration", "max_predicted_torque",
        ):
            setattr(self, name, _vector(getattr(self, name), self.dof, name))
        if np.any(self.q_min >= self.q_max):
            raise ValidationError("every q_min must be less than q_max")
        if np.any(self.max_velocity <= 0) or np.any(self.max_acceleration <= 0):
            raise ValidationError("velocity and acceleration limits must be positive")
        if np.any(self.max_predicted_torque <= 0):
            raise ValidationError("torque limits must be positive")
        if np.isscalar(self.position_margin):
            self.position_margin = np.full(self.dof, float(self.position_margin))
        else:
            self.position_margin = _vector(self.position_margin, self.dof, "position_margin")
        if np.any(self.position_margin < 0) or np.any(
            self.q_min + self.position_margin >= self.q_max - self.position_margin
        ):
            raise ValidationError("position margin leaves no usable joint range")
        if not (0 < self.input_soft_stale_s < self.input_hard_stale_s):
            raise ValidationError("input stale thresholds must satisfy 0 < soft < hard")
        if self.fresh_dwell_s < 0 or self.blend_in_s <= 0 or self.recovery_s <= 0:
            raise ValidationError("dwell/blend/recovery durations are invalid")
        if not self.simulation_only and self.simulation_joint_interpolation_test:
            raise ValidationError(
                "static standby interpolation is a simulation test fixture and is forbidden for hardware profiles"
            )

    @classmethod
    def simulation_default(cls, dof: int = 29) -> "SafetyProfile":
        names = G1_29DOF_JOINT_NAMES if dof == 29 else tuple(f"joint_{i}" for i in range(dof))
        if dof == 29:
            q_min = G1_29DOF_Q_MIN.copy()
            q_max = G1_29DOF_Q_MAX.copy()
            standby = G1_29DOF_STAND.copy()
            torque = G1_29DOF_TORQUE.copy()
            margin = np.maximum(0.05, 0.05 * (q_max - q_min))
        else:
            q_min = np.full(dof, -math.pi)
            q_max = np.full(dof, math.pi)
            standby = np.zeros(dof)
            torque = np.full(dof, 40.0)
            margin = np.maximum(0.05, 0.05 * (q_max - q_min))
        return cls(
            dof=dof,
            model=f"unitree_g1_{dof}dof",
            dof_hash=compute_dof_hash(names),
            q_min=q_min,
            q_max=q_max,
            standby_q=standby,
            max_velocity=np.full(dof, 4.0),
            max_acceleration=np.full(dof, 20.0),
            max_predicted_torque=torque,
            position_margin=margin,
            simulation_only=True,
        )

    def assert_transport_allowed(self, transport: str) -> None:
        if self.simulation_only and transport.lower() not in {"simulation", "mujoco", "test", "null"}:
            raise PermissionError(
                f"profile {self.profile_name!r} is simulation-only; refusing transport {transport!r}"
            )


_ALLOWED = {
    SafetyState.DISARMED: {SafetyState.ARMING, SafetyState.E_STOP, SafetyState.FAULT},
    SafetyState.ARMING: {SafetyState.STANDBY, SafetyState.DISARMED, SafetyState.E_STOP, SafetyState.FAULT},
    SafetyState.STANDBY: {SafetyState.BLEND_IN, SafetyState.RECOVER_STAND, SafetyState.DISARMED, SafetyState.E_STOP, SafetyState.FAULT},
    SafetyState.BLEND_IN: {SafetyState.TELEOP, SafetyState.INPUT_LOSS, SafetyState.RECOVER_STAND, SafetyState.STANDBY, SafetyState.DISARMED, SafetyState.E_STOP, SafetyState.FAULT},
    SafetyState.TELEOP: {SafetyState.INPUT_LOSS, SafetyState.RECOVER_STAND, SafetyState.DISARMED, SafetyState.E_STOP, SafetyState.FAULT},
    SafetyState.INPUT_LOSS: {SafetyState.BLEND_IN, SafetyState.RECOVER_STAND, SafetyState.STANDBY, SafetyState.DISARMED, SafetyState.E_STOP, SafetyState.FAULT},
    SafetyState.RECOVER_STAND: {SafetyState.STANDBY, SafetyState.DISARMED, SafetyState.E_STOP, SafetyState.FAULT},
    SafetyState.E_STOP: {SafetyState.DISARMED},
    SafetyState.FAULT: {SafetyState.DISARMED, SafetyState.E_STOP},
}


class G1SafetyGateway:
    """Deterministic command guard; it is not itself a balance controller."""

    def __init__(self, profile: Optional[SafetyProfile] = None) -> None:
        self.profile = profile or SafetyProfile.simulation_default()
        self.state = SafetyState.DISARMED
        self.reason = "initially disarmed"
        self.estop_latched = False
        self.fault_latched = False
        self.operator_deadman = False
        self.resume_required = True
        self._resume_requested = False
        self._state_enter = 0.0
        self._last_step_time: Optional[float] = None
        self._last_state_time: Optional[float] = None
        self._last_state_tick: Optional[int] = None
        self._last_policy_seq: Optional[int] = None
        self._last_policy_time: Optional[float] = None
        self._last_policy: Optional[PolicyCommand] = None
        self._last_standby_seq: Optional[int] = None
        self._last_standby_time: Optional[float] = None
        self._last_standby: Optional[PolicyCommand] = None
        self._standby_missing_since: Optional[float] = None
        self._fresh_since: Optional[float] = None
        self._loss_started: Optional[float] = None
        self._loss_hard = False
        self._blend_start_q: Optional[np.ndarray] = None
        self._blend_start_dq: Optional[np.ndarray] = None
        self._blend_start_tau: Optional[np.ndarray] = None
        self._blend_start_kp: Optional[np.ndarray] = None
        self._blend_start_kd: Optional[np.ndarray] = None
        self._standby_start_q: Optional[np.ndarray] = None
        self._standby_engaging = False
        self._recovery_start_q: Optional[np.ndarray] = None
        self._recovery_start_dq: Optional[np.ndarray] = None
        self._recovery_start_tau: Optional[np.ndarray] = None
        self._recovery_start_kp: Optional[np.ndarray] = None
        self._recovery_start_kd: Optional[np.ndarray] = None
        self._last_q: Optional[np.ndarray] = None
        self._last_q_rate: Optional[np.ndarray] = None
        self._last_dq: Optional[np.ndarray] = None
        self._last_tau: Optional[np.ndarray] = None
        self._last_kp: Optional[np.ndarray] = None
        self._last_kd: Optional[np.ndarray] = None
        self._output_seq = 0
        self._clip_events: Deque[float] = deque()

    def _clear_runtime_history(self) -> None:
        """Clear stale controller history before a new explicit arm cycle."""

        self._last_step_time = None
        self._last_state_time = None
        self._last_state_tick = None
        self._last_policy_seq = None
        self._last_policy_time = None
        self._last_policy = None
        self._last_standby_seq = None
        self._last_standby_time = None
        self._last_standby = None
        self._standby_missing_since = None
        self._fresh_since = None
        self._loss_started = None
        self._loss_hard = False
        self._blend_start_q = None
        self._blend_start_dq = None
        self._blend_start_tau = None
        self._blend_start_kp = None
        self._blend_start_kd = None
        self._standby_start_q = None
        self._standby_engaging = False
        self._recovery_start_q = None
        self._recovery_start_dq = None
        self._recovery_start_tau = None
        self._recovery_start_kp = None
        self._recovery_start_kd = None
        self._last_q = None
        self._last_q_rate = None
        self._last_dq = None
        self._last_tau = None
        self._last_kp = None
        self._last_kd = None
        self._clip_events.clear()

    def _transition(self, target: SafetyState, now: float, reason: str) -> None:
        if target == self.state:
            self.reason = reason
            return
        if target not in _ALLOWED[self.state]:
            raise RuntimeError(f"illegal safety transition {self.state.value} -> {target.value}")
        previous = self.state
        if target == SafetyState.STANDBY and previous == SafetyState.ARMING:
            self._standby_start_q = None if self._last_q is None else self._last_q.copy()
            self._standby_engaging = True
        elif target == SafetyState.STANDBY:
            self._standby_start_q = None
            self._standby_engaging = False
        self.state = target
        self._state_enter = now
        self.reason = reason

    def _prepare_blend_start(self, q_robot: np.ndarray) -> None:
        self._blend_start_q = self._last_q.copy() if self._last_q is not None else q_robot.copy()
        self._blend_start_dq = self._last_dq.copy() if self._last_dq is not None else np.zeros(self.profile.dof)
        self._blend_start_tau = self._last_tau.copy() if self._last_tau is not None else np.zeros(self.profile.dof)
        self._blend_start_kp = self._last_kp.copy() if self._last_kp is not None else np.zeros(self.profile.dof)
        self._blend_start_kd = self._last_kd.copy() if self._last_kd is not None else np.full(self.profile.dof, self.profile.estop_kd)

    def set_deadman(self, held: bool, now: float) -> None:
        self.operator_deadman = bool(held)
        if not held:
            self._resume_requested = False
            self.resume_required = True
            if self.state in {SafetyState.TELEOP, SafetyState.BLEND_IN}:
                self._enter_input_loss(now, "operator deadman released", hard=True)

    def arm(self, now: float) -> bool:
        if self.estop_latched or self.fault_latched or self.state != SafetyState.DISARMED:
            return False
        if not self.operator_deadman:
            return False
        self._transition(SafetyState.ARMING, now, "operator arm requested")
        self.resume_required = True
        return True

    def resume(self, now: float) -> bool:
        if self.estop_latched or self.fault_latched or not self.operator_deadman:
            return False
        if self.state != SafetyState.STANDBY:
            return False
        self._resume_requested = True
        self.reason = "operator resume requested; waiting for fresh dwell"
        return True

    def disarm(self, now: float, reason: str = "operator disarm") -> None:
        if self.state not in {SafetyState.E_STOP, SafetyState.FAULT, SafetyState.DISARMED}:
            self._transition(SafetyState.DISARMED, now, reason)
        elif self.state == SafetyState.DISARMED:
            self.reason = reason
        self._resume_requested = False
        self.operator_deadman = False
        self.resume_required = True
        self._clear_runtime_history()

    def emergency_stop(self, now: float, reason: str = "emergency stop requested") -> None:
        self.estop_latched = True
        if self.state != SafetyState.E_STOP:
            self._transition(SafetyState.E_STOP, now, reason)
        else:
            self.reason = reason
        self._resume_requested = False

    def _fault(self, now: float, reason: str) -> None:
        self.fault_latched = True
        if self.state != SafetyState.E_STOP and self.state != SafetyState.FAULT:
            self._transition(SafetyState.FAULT, now, reason)
        self.reason = reason
        self._resume_requested = False

    def reset(
        self,
        now: float,
        robot_state: RobotState,
        *,
        operator_confirmed: bool,
        physical_estop_released: bool,
    ) -> bool:
        if self.state not in {SafetyState.E_STOP, SafetyState.FAULT}:
            return False
        if not operator_confirmed or not physical_estop_released:
            return False
        try:
            _, _, rpy, gyro = self._validate_robot(robot_state, now, update_monotonic=False)
        except ValidationError:
            return False
        if self._imu_level(rpy, gyro) != "normal":
            return False
        self.estop_latched = False
        self.fault_latched = False
        self.operator_deadman = False
        self.resume_required = True
        self._resume_requested = False
        self._clear_runtime_history()
        self._transition(SafetyState.DISARMED, now, "latched stop reset; explicit arm required")
        return True

    def _validate_robot(self, robot: RobotState, now: float, *, update_monotonic: bool = True) -> Tuple[np.ndarray, ...]:
        if not robot.valid:
            raise ValidationError("LowState marked invalid")
        if (
            isinstance(robot.tick, (bool, np.bool_))
            or not isinstance(robot.tick, (int, np.integer))
            or int(robot.tick) < 0
        ):
            raise ValidationError("LowState tick must be a non-negative integer")
        if robot.model != self.profile.model or robot.dof_hash != self.profile.dof_hash:
            raise ValidationError("LowState model/DoF hash mismatch")
        if not math.isfinite(robot.timestamp) or robot.timestamp > now + self.profile.future_tolerance_s:
            raise ValidationError("LowState timestamp is invalid or in the future")
        if now - robot.timestamp > self.profile.state_stale_s:
            raise ValidationError("LowState is stale")
        arrays = robot.arrays(self.profile.dof)
        if update_monotonic and self._last_state_time is not None:
            if robot.timestamp < self._last_state_time:
                raise ValidationError("LowState timestamp moved backwards")
            if robot.timestamp > self._last_state_time and robot.tick <= int(self._last_state_tick):
                raise ValidationError("LowState tick is not strictly monotonic")
        if update_monotonic and (self._last_state_time is None or robot.timestamp > self._last_state_time):
            self._last_state_time = robot.timestamp
            self._last_state_tick = int(robot.tick)
        return arrays

    def _validate_policy(self, command: PolicyCommand, now: float) -> Tuple[np.ndarray, ...]:
        if not isinstance(command.source_valid, (bool, np.bool_)) or not isinstance(
            command.deadman, (bool, np.bool_)
        ):
            raise ValidationError("policy validity/deadman must be boolean")
        if not command.source_valid:
            raise ValidationError("policy source marked invalid")
        if command.model != self.profile.model or command.dof_hash != self.profile.dof_hash:
            raise ValidationError("policy model/DoF hash mismatch")
        if not isinstance(command.seq, (int, np.integer)) or command.seq < 0:
            raise ValidationError("policy sequence is invalid")
        if not math.isfinite(command.timestamp) or command.timestamp > now + self.profile.future_tolerance_s:
            raise ValidationError("policy timestamp is invalid or in the future")
        arrays = command.arrays(self.profile.dof)
        command.active(self.profile.dof)
        if np.any(arrays[3] < 0) or np.any(arrays[4] < 0):
            raise ValidationError("negative policy gain")
        if np.any(arrays[3] > self.profile.max_kp) or np.any(arrays[4] > self.profile.max_kd):
            raise ValidationError("policy gain exceeds configured maximum")
        if self._last_policy_seq is not None:
            if command.seq < self._last_policy_seq:
                raise ValidationError("policy sequence moved backwards")
            if command.seq == self._last_policy_seq and command.timestamp != self._last_policy_time:
                raise ValidationError("same policy sequence has a different timestamp")
            if command.seq == self._last_policy_seq and self._last_policy is not None:
                previous = self._last_policy.arrays(self.profile.dof)
                if any(not np.array_equal(new, old) for new, old in zip(arrays, previous)):
                    raise ValidationError("same policy sequence has a different payload")
                if command.deadman != self._last_policy.deadman:
                    raise ValidationError("same policy sequence changed deadman")
                if not np.array_equal(
                    command.active(self.profile.dof),
                    self._last_policy.active(self.profile.dof),
                ):
                    raise ValidationError("same policy sequence changed active mask")
            if command.seq > self._last_policy_seq and command.timestamp <= float(self._last_policy_time):
                raise ValidationError("new policy sequence has a non-increasing timestamp")
        if self._last_policy_seq is None or command.seq > self._last_policy_seq:
            self._last_policy_seq = int(command.seq)
            self._last_policy_time = float(command.timestamp)
            self._last_policy = command
        return arrays

    def _validate_standby_policy(self, command: PolicyCommand, now: float) -> Tuple[np.ndarray, ...]:
        """Validate the independent, continuously-running balance policy."""
        if not isinstance(command.source_valid, (bool, np.bool_)) or not isinstance(
            command.deadman, (bool, np.bool_)
        ):
            raise ValidationError("standby validity/deadman must be boolean")
        if not command.source_valid:
            raise ValidationError("standby balance policy marked invalid")
        if command.model != self.profile.model or command.dof_hash != self.profile.dof_hash:
            raise ValidationError("standby balance policy model/DoF hash mismatch")
        if not isinstance(command.seq, (int, np.integer)) or command.seq < 0:
            raise ValidationError("standby balance policy sequence is invalid")
        if not math.isfinite(command.timestamp) or command.timestamp > now + self.profile.future_tolerance_s:
            raise ValidationError("standby balance policy timestamp is invalid")
        arrays = command.arrays(self.profile.dof)
        if not np.all(command.active(self.profile.dof)):
            raise ValidationError("standby balance policy must own every joint")
        if np.any(arrays[3] < 0) or np.any(arrays[4] < 0):
            raise ValidationError("negative standby balance policy gain")
        if np.any(arrays[3] > self.profile.max_kp) or np.any(arrays[4] > self.profile.max_kd):
            raise ValidationError("standby balance policy gain exceeds configured maximum")
        if self._last_standby_seq is not None:
            if command.seq < self._last_standby_seq:
                raise ValidationError("standby balance policy sequence moved backwards")
            if command.seq == self._last_standby_seq and command.timestamp != self._last_standby_time:
                raise ValidationError("same standby sequence has a different timestamp")
            if command.seq == self._last_standby_seq and self._last_standby is not None:
                previous = self._last_standby.arrays(self.profile.dof)
                if any(not np.array_equal(new, old) for new, old in zip(arrays, previous)):
                    raise ValidationError("same standby sequence has a different payload")
                if command.deadman != self._last_standby.deadman:
                    raise ValidationError("same standby sequence changed deadman")
            if command.seq > self._last_standby_seq and command.timestamp <= float(self._last_standby_time):
                raise ValidationError("new standby sequence has a non-increasing timestamp")
        if self._last_standby_seq is None or command.seq > self._last_standby_seq:
            self._last_standby_seq = int(command.seq)
            self._last_standby_time = float(command.timestamp)
            self._last_standby = command
        return arrays

    def _imu_level(self, rpy: np.ndarray, gyro: np.ndarray) -> str:
        tilt = float(np.max(np.abs(rpy[:2])))
        speed = float(np.linalg.norm(gyro))
        if tilt >= self.profile.imu_hard_tilt_rad or speed >= self.profile.imu_hard_gyro_rad_s:
            return "hard"
        if tilt >= self.profile.imu_soft_tilt_rad or speed >= self.profile.imu_soft_gyro_rad_s:
            return "soft"
        return "normal"

    def _enter_input_loss(self, now: float, reason: str, hard: bool = False) -> None:
        if self.state in {SafetyState.TELEOP, SafetyState.BLEND_IN}:
            self._transition(SafetyState.RECOVER_STAND if hard else SafetyState.INPUT_LOSS, now, reason)
            self._loss_started = now
            self._recovery_start_q = None if self._last_q is None else self._last_q.copy()
            self._recovery_start_dq = None if self._last_dq is None else self._last_dq.copy()
            self._recovery_start_tau = None if self._last_tau is None else self._last_tau.copy()
            self._recovery_start_kp = None if self._last_kp is None else self._last_kp.copy()
            self._recovery_start_kd = None if self._last_kd is None else self._last_kd.copy()
        elif hard and self.state == SafetyState.INPUT_LOSS:
            self._transition(SafetyState.RECOVER_STAND, now, reason)
        self._loss_hard = self._loss_hard or hard
        if hard:
            self.resume_required = True
        self._fresh_since = None
        self._resume_requested = False

    def _fresh(
        self,
        command: Optional[PolicyCommand],
        now: float,
        teleop_source_age_s: float,
        teleop_source_valid: bool,
    ) -> bool:
        return bool(
            command is not None
            and command.source_valid
            and command.deadman
            and self.operator_deadman
            and now - command.timestamp <= self.profile.policy_output_stale_s
            and teleop_source_valid
            and teleop_source_age_s <= self.profile.input_soft_stale_s
        )

    def _update_fresh_dwell(self, fresh: bool, now: float) -> bool:
        if not fresh:
            self._fresh_since = None
            return False
        if self._fresh_since is None:
            self._fresh_since = now
        return now - self._fresh_since >= self.profile.fresh_dwell_s

    def _base_target(
        self,
        q_robot: np.ndarray,
        standby_arrays: Optional[Tuple[np.ndarray, ...]],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        p = self.profile
        q = p.standby_q.copy()
        dq = np.zeros(p.dof)
        tau = np.zeros(p.dof)
        kp = np.full(p.dof, p.standby_kp)
        kd = np.full(p.dof, p.standby_kd)
        weight = 0.0
        if standby_arrays is not None:
            q, dq, tau, kp, kd = (value.copy() for value in standby_arrays)
        if self.state == SafetyState.STANDBY and self._standby_engaging:
            engage = minimum_jerk((self._last_step_time - self._state_enter) / p.blend_in_s)
            start = self._standby_start_q if self._standby_start_q is not None else q_robot
            q = (1.0 - engage) * start + engage * q
            dq = engage * dq
            tau = engage * tau
            kp = engage * kp
            kd = (1.0 - engage) * np.full(p.dof, p.estop_kd) + engage * kd
            if engage >= 1.0:
                self._standby_engaging = False
        elif self.state == SafetyState.DISARMED:
            q = q_robot.copy()
            dq.fill(0.0)
            tau.fill(0.0)
            kp.fill(0.0)
            kd.fill(0.0)
        elif self.state == SafetyState.ARMING:
            # Arming is a verification dwell, not a zero-torque interval.
            # Keep the controller in the same damping-safe form used by the
            # official low-level examples until the balance policy is accepted.
            q = q_robot.copy()
            dq.fill(0.0)
            tau.fill(0.0)
            kp.fill(0.0)
            kd.fill(p.estop_kd)
        elif self.state in {SafetyState.E_STOP, SafetyState.FAULT}:
            q = q_robot.copy()
            dq = np.zeros(p.dof)
            tau.fill(0.0)
            kp.fill(0.0)
            kd.fill(p.estop_kd)
        return q, dq, tau, kp, kd, weight

    def _apply_safety_limits(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        tau: np.ndarray,
        kp: np.ndarray,
        kd: np.ndarray,
        q_robot: np.ndarray,
        dq_robot: np.ndarray,
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool, bool]:
        p = self.profile
        limited = False
        hard_limited = False
        q_bounded = np.clip(q, p.q_min + p.position_margin, p.q_max - p.position_margin)
        q_clamped = not np.allclose(q_bounded, q, rtol=0, atol=1e-12)
        limited |= q_clamped
        hard_limited |= q_clamped
        dq_bounded = np.clip(dq, -p.max_velocity, p.max_velocity)
        dq_clamped = not np.allclose(dq_bounded, dq, rtol=0, atol=1e-12)
        limited |= dq_clamped
        hard_limited |= dq_clamped
        if self._last_q is None:
            self._last_q = q_robot.copy()
        if self._last_q_rate is None:
            self._last_q_rate = np.zeros(p.dof)
        if self._last_dq is None:
            self._last_dq = dq_robot.copy()
        desired_q_velocity = np.clip((q_bounded - self._last_q) / dt, -p.max_velocity, p.max_velocity)
        q_velocity = np.clip(
            desired_q_velocity,
            self._last_q_rate - p.max_acceleration * dt,
            self._last_q_rate + p.max_acceleration * dt,
        )
        q_out = self._last_q + q_velocity * dt
        limited |= not np.allclose(q_out, q_bounded, rtol=0, atol=1e-12)
        dq_out = np.clip(
            dq_bounded,
            self._last_dq - p.max_acceleration * dt,
            self._last_dq + p.max_acceleration * dt,
        )
        limited |= not np.allclose(dq_out, dq_bounded, rtol=0, atol=1e-12)
        tau_out = np.clip(tau, -p.max_predicted_torque, p.max_predicted_torque)
        tau_clamped = not np.allclose(tau_out, tau, rtol=0, atol=1e-12)
        limited |= tau_clamped
        hard_limited |= tau_clamped
        predicted = tau_out + kp * (q_out - q_robot) + kd * (dq_out - dq_robot)
        torque_bad = np.abs(predicted) > p.max_predicted_torque
        if np.any(torque_bad):
            # A violating joint is replaced with a zero-error PD target.  Do not
            # hide a dangerous request by relying on feed-forward cancellation.
            q_out[torque_bad] = q_robot[torque_bad]
            dq_out[torque_bad] = dq_robot[torque_bad]
            tau_out[torque_bad] = 0.0
            q_velocity[torque_bad] = 0.0
            predicted = tau_out + kp * (q_out - q_robot) + kd * (dq_out - dq_robot)
            limited = True
            hard_limited = True
        self._last_q = q_out.copy()
        self._last_q_rate = q_velocity.copy()
        self._last_dq = dq_out.copy()
        return q_out, dq_out, tau_out, predicted, limited, hard_limited

    def step(
        self,
        policy: Optional[PolicyCommand],
        robot: RobotState,
        now: float,
        *,
        standby_policy: Optional[PolicyCommand] = None,
        teleop_source_age_s: Optional[float] = None,
        teleop_source_valid: bool = True,
        dt: Optional[float] = None,
    ) -> SafeCommand:
        if not math.isfinite(now):
            raise ValueError("now must be finite monotonic time")
        if dt is None:
            raw_dt = self.profile.control_period_s if self._last_step_time is None else now - self._last_step_time
        else:
            raw_dt = dt
        if raw_dt <= 0 or not math.isfinite(raw_dt):
            self._fault(now, "control time moved backwards or stopped")
            raw_dt = self.profile.control_period_s
        dt_eff = float(np.clip(raw_dt, 1e-6, self.profile.max_slew_dt_s))
        self._last_step_time = now

        try:
            q_robot, dq_robot, rpy, gyro = self._validate_robot(robot, now)
        except ValidationError as exc:
            self._fault(now, str(exc))
            q_robot = self.profile.standby_q.copy()
            dq_robot = np.zeros(self.profile.dof)
            rpy = gyro = np.zeros(3)

        policy_arrays: Optional[Tuple[np.ndarray, ...]] = None
        if policy is not None and not self.fault_latched:
            try:
                policy_arrays = self._validate_policy(policy, now)
            except ValidationError as exc:
                self._fault(now, str(exc))

        standby_arrays: Optional[Tuple[np.ndarray, ...]] = None
        if standby_policy is not None and not self.fault_latched:
            try:
                standby_arrays = self._validate_standby_policy(standby_policy, now)
                self._standby_missing_since = None
            except ValidationError as exc:
                self._fault(now, str(exc))
        elif self._last_standby is not None and not self.fault_latched:
            try:
                standby_arrays = self._validate_standby_policy(self._last_standby, now)
            except ValidationError as exc:
                self._fault(now, str(exc))

        needs_balance_policy = self.state not in {
            SafetyState.DISARMED, SafetyState.ARMING, SafetyState.E_STOP, SafetyState.FAULT
        }
        if needs_balance_policy and not self.profile.simulation_joint_interpolation_test:
            standby_age = math.inf if self._last_standby_time is None else now - self._last_standby_time
            if standby_arrays is None or standby_age > self.profile.input_hard_stale_s:
                if self._standby_missing_since is None:
                    self._standby_missing_since = now
                if (
                    self._last_standby_time is not None
                    or now - self._standby_missing_since >= self.profile.input_hard_stale_s
                ):
                    self._fault(now, "standby balance policy missing or hard stale")

        # Teleop policy freshness is distinct from upstream mocap/reference
        # freshness and from the independent standby balance policy. A stalled
        # teleop policy is a hard input loss, but the safety process can still
        # recover through a fresh standby policy. Only loss of that fallback (or
        # LowState) is a latched fault.
        policy_output_age = math.inf if policy is None else now - policy.timestamp

        imu_level = self._imu_level(rpy, gyro)
        if imu_level == "hard":
            self.emergency_stop(now, "hard IMU attitude/rate threshold exceeded")
        elif imu_level == "soft" and self.state in {SafetyState.TELEOP, SafetyState.BLEND_IN, SafetyState.INPUT_LOSS}:
            self._enter_input_loss(now, "soft IMU threshold exceeded", hard=True)

        command_age = math.inf if policy is None else now - policy.timestamp
        source_age = command_age if teleop_source_age_s is None else float(teleop_source_age_s)
        if not math.isfinite(source_age) or source_age < 0:
            teleop_source_valid = False
            source_age = math.inf
        fresh = policy_arrays is not None and self._fresh(
            policy, now, source_age, bool(teleop_source_valid)
        )
        fresh_dwell = self._update_fresh_dwell(fresh, now)

        if not self.estop_latched and not self.fault_latched:
            if self.state == SafetyState.ARMING and now - self._state_enter >= self.profile.arming_dwell_s:
                self._transition(SafetyState.STANDBY, now, "armed in standby; explicit resume required")
            elif self.state in {SafetyState.TELEOP, SafetyState.BLEND_IN}:
                if not self.operator_deadman or policy is None or not policy.deadman:
                    self._enter_input_loss(now, "deadman or policy command missing", hard=True)
                elif policy_output_age > self.profile.policy_output_stale_s:
                    self._enter_input_loss(now, "teleop policy output stale", hard=True)
                elif not teleop_source_valid or source_age > self.profile.input_hard_stale_s:
                    self._enter_input_loss(now, "teleop reference hard stale or invalid", hard=True)
                elif source_age > self.profile.input_soft_stale_s:
                    self._enter_input_loss(now, "teleop reference soft stale", hard=False)
                elif self.state == SafetyState.BLEND_IN and now - self._state_enter >= self.profile.blend_in_s:
                    self._transition(SafetyState.TELEOP, now, "teleoperation blend complete")
            elif self.state == SafetyState.INPUT_LOSS:
                if not teleop_source_valid or source_age > self.profile.input_hard_stale_s:
                    self._enter_input_loss(now, "teleop reference reached hard stale", hard=True)
                elif fresh_dwell and not self._loss_hard:
                    self._prepare_blend_start(q_robot)
                    self._transition(SafetyState.BLEND_IN, now, "short input loss recovered after fresh dwell")
            elif self.state == SafetyState.RECOVER_STAND:
                if now - self._state_enter >= self.profile.recovery_s:
                    self._transition(SafetyState.STANDBY, now, "recovery complete; manual resume required")
            elif self.state == SafetyState.STANDBY and self._resume_requested and fresh_dwell:
                self._prepare_blend_start(q_robot)
                self._resume_requested = False
                self.resume_required = False
                self._loss_hard = False
                self._transition(SafetyState.BLEND_IN, now, "fresh input and operator resume accepted")
        q, dq, tau, kp, kd, weight = self._base_target(q_robot, standby_arrays)
        if self.state in {SafetyState.TELEOP, SafetyState.BLEND_IN} and policy_arrays is not None:
            pq, pdq, ptau, pkp, pkd = policy_arrays
            active = policy.active(self.profile.dof)
            # Inactive policy joints are owned by the independent standby
            # controller.  This is required for TWIST's four uncommanded wrist
            # pitch/yaw joints; never fill them with guessed zeros.
            pq = np.where(active, pq, q)
            pdq = np.where(active, pdq, dq)
            ptau = np.where(active, ptau, tau)
            pkp = np.where(active, pkp, kp)
            pkd = np.where(active, pkd, kd)
            if self.state == SafetyState.TELEOP:
                weight = 1.0
            else:
                weight = minimum_jerk((now - self._state_enter) / self.profile.blend_in_s)
            start = self._blend_start_q if self._blend_start_q is not None else q_robot
            start_dq = self._blend_start_dq if self._blend_start_dq is not None else np.zeros(self.profile.dof)
            start_tau = self._blend_start_tau if self._blend_start_tau is not None else np.zeros(self.profile.dof)
            start_kp = self._blend_start_kp if self._blend_start_kp is not None else np.zeros(self.profile.dof)
            start_kd = self._blend_start_kd if self._blend_start_kd is not None else np.full(self.profile.dof, self.profile.estop_kd)
            q = (1.0 - weight) * start + weight * pq
            dq = (1.0 - weight) * start_dq + weight * pdq
            tau = (1.0 - weight) * start_tau + weight * ptau
            kp = (1.0 - weight) * start_kp + weight * pkp
            kd = (1.0 - weight) * start_kd + weight * pkd
        elif self.state in {SafetyState.INPUT_LOSS, SafetyState.RECOVER_STAND}:
            start = self._recovery_start_q if self._recovery_start_q is not None else q_robot
            elapsed = 0.0 if self._loss_started is None else now - self._loss_started
            recovery_weight = minimum_jerk(elapsed / self.profile.recovery_s)
            # The destination comes from a continuously-running balance policy.
            # Static standby_q is only reachable through the explicitly enabled
            # simulation_joint_interpolation_test fixture.
            balance_q = q.copy()
            q = (1.0 - recovery_weight) * start + recovery_weight * balance_q
            start_dq = self._recovery_start_dq if self._recovery_start_dq is not None else np.zeros(self.profile.dof)
            start_tau = self._recovery_start_tau if self._recovery_start_tau is not None else np.zeros(self.profile.dof)
            start_kp = self._recovery_start_kp if self._recovery_start_kp is not None else np.zeros(self.profile.dof)
            start_kd = self._recovery_start_kd if self._recovery_start_kd is not None else np.full(self.profile.dof, self.profile.estop_kd)
            dq = (1.0 - recovery_weight) * start_dq + recovery_weight * dq
            tau = (1.0 - recovery_weight) * start_tau + recovery_weight * tau
            kp = (1.0 - recovery_weight) * start_kp + recovery_weight * kp
            kd = (1.0 - recovery_weight) * start_kd + recovery_weight * kd
            weight = 1.0 - recovery_weight

        q, dq, tau, predicted, limited, hard_limited = self._apply_safety_limits(
            q, dq, tau, kp, kd, q_robot, dq_robot, dt_eff
        )
        # Slew limiting is the expected way a smooth controller approaches a
        # moving target.  Only actual range/torque clamps count toward the
        # repeated-command-violation watchdog.
        if hard_limited and self.state in {SafetyState.TELEOP, SafetyState.BLEND_IN}:
            self._clip_events.append(now)
            cutoff = now - self.profile.clip_window_s
            while self._clip_events and self._clip_events[0] < cutoff:
                self._clip_events.popleft()
            if len(self._clip_events) >= self.profile.clip_event_limit:
                self._enter_input_loss(now, "frequent command limiting; leaving teleoperation", hard=True)
        else:
            cutoff = now - self.profile.clip_window_s
            while self._clip_events and self._clip_events[0] < cutoff:
                self._clip_events.popleft()

        self._output_seq += 1
        self._last_tau = tau.copy()
        self._last_kp = kp.copy()
        self._last_kd = kd.copy()
        active = self.state not in {SafetyState.DISARMED}
        return SafeCommand(
            timestamp=now,
            seq=self._output_seq,
            state=self.state,
            q=q,
            dq=dq,
            tau_ff=tau,
            kp=kp,
            kd=kd,
            teleop_weight=float(weight),
            reason=self.reason,
            limited=limited,
            estop_latched=self.estop_latched,
            fault_latched=self.fault_latched,
            command_active=active,
            real_hardware_allowed=not self.profile.simulation_only,
            predicted_torque=predicted,
        )
