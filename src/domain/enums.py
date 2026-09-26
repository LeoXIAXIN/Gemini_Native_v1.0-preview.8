"""Domain enums mirroring the frozen Phase-0 contract.

Values are frozen by architecture/PHASE0_INTERFACE_FREEZE.md; adding a new
member is allowed, renaming or renumbering an existing one is not.
"""

from __future__ import annotations

from enum import Enum


class Backend(str, Enum):
    """Selectable pipeline backends (Native Windows scope)."""

    HUMANOID_GPT = "humanoid_gpt"
    GMR_PREVIEW = "gmr_preview"
    # Base-layer legacy value; not selectable through the Native config.
    TWIST = "twist"


class ExecutionMode(str, Enum):
    """Execution modes; gmr_preview supports simulation_only only."""

    SIMULATION_ONLY = "simulation_only"
    REAL_ONLY = "real_only"
    PARALLEL = "parallel"


class TaskPhase(str, Enum):
    """Controller phase names used by the legacy PipelineManager."""

    STOPPED = "stopped"
    PREFLIGHT = "preflight"
    STARTING = "starting"
    WAITING_MOCAP = "waiting_mocap"
    STARTING_BACKEND = "starting_backend"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


class BridgeStatus(str, Enum):
    """Values (or prefixes) of the ``chingmu_gmr_bridge_status`` state key."""

    STARTING_GMR_RECEIVER = "starting_gmr_receiver"
    WAITING_FOR_LOCAL_FRAME = "waiting_for_local_frame"
    LOCAL_FRAME_RECEIVED = "local_frame_received"
    RETARGETING_FIRST_FRAME = "retargeting_first_frame"
    LIVE = "live"  # prefix: live:<frame_id>:<value>
    STOPPED = "stopped"
    INVALID_SKELETON = "invalid_skeleton"  # prefix: invalid_skeleton:<detail>
    ERROR = "error"  # prefix: error:<detail>


class SafetyState(str, Enum):
    """g1_safety state machine states (never auto-recover from latches)."""

    DISARMED = "DISARMED"
    ARMING = "ARMING"
    BLEND_IN = "BLEND_IN"
    STANDBY = "STANDBY"
    TELEOP = "TELEOP"
    INPUT_LOSS = "INPUT_LOSS"
    RECOVER_STAND = "RECOVER_STAND"
    E_STOP = "E_STOP"
    FAULT = "FAULT"
    LEGACY_DIRECT = "LEGACY_DIRECT"


class ReferenceHealth(str, Enum):
    """Reference freshness health reported by the safety telemetry."""

    LIVE = "LIVE"
    SOFT_STALE = "SOFT_STALE"
    HARD_STALE = "HARD_STALE"
    WAIT_FRESH = "WAIT_FRESH"
    STANDBY = "STANDBY"
    INVALID = "INVALID"


class RobotStateSource(str, Enum):
    """Where a RobotState snapshot originated."""

    MUJOCO = "mujoco"
    UNITREE_LOWSTATE = "unitree_lowstate"


class SafetyCommandAction(str, Enum):
    """Actions accepted by the ``g1_safety_control`` channel."""

    ESTOP = "estop"
    RESET = "reset"
