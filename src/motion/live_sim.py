#!/usr/bin/env python3
"""Track live CHINGMU/GMR qpos with Humanoid-GPT's released MuJoCo policy.

Simulation only.  This module does not import Unitree SDK2 and cannot publish
real-robot DDS or motor commands.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Sequence

import numpy as np

if str(Path(__file__).resolve().parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.safety.g1_core import (
    G1_29DOF_JOINT_NAMES,
    G1SafetyGateway,
    PolicyCommand,
    RobotState,
    SafeCommand,
    SafetyProfile,
    SafetyState,
)
from src.safety.g1_protocol import PolicyTargetEnvelope, encode_policy_target
from src.safety.g1_reference import (
    ReferenceBlend,
    ReferenceHealth,
    ReferenceSample,
    ReferenceWatchdog,
)


def _yaw_from_wxyz(quaternion: np.ndarray) -> float:
    w, x, y, z = quaternion
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _quat_mul_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = left
    w2, x2, y2, z2 = right
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float32,
    )


def _rpy_from_wxyz(quaternion: np.ndarray) -> np.ndarray:
    """Return roll/pitch/yaw without importing SciPy in the safety loop."""

    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    norm = max(float(np.linalg.norm([w, x, y, z])), 1e-12)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.asarray([roll, pitch, yaw], dtype=np.float64)


def _validate_mujoco_kinematic_state(
    qpos: Sequence[float], qvel: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Validate the complete floating-base state before any actuator step.

    Checking only the 29 joint values is insufficient: a NaN in root XYZ or
    base velocity can still poison policy observations and MuJoCo integration.
    The returned arrays are the canonical 29 joint position/velocity slices.
    """

    full_qpos = np.asarray(qpos, dtype=np.float64).reshape(-1)
    full_qvel = np.asarray(qvel, dtype=np.float64).reshape(-1)
    if full_qpos.shape != (36,):
        raise ValueError(f"MuJoCo qpos shape {full_qpos.shape}, expected (36,)")
    if full_qvel.shape != (35,):
        raise ValueError(f"MuJoCo qvel shape {full_qvel.shape}, expected (35,)")
    if not np.all(np.isfinite(full_qpos)):
        raise ValueError("MuJoCo qpos contains NaN or infinity")
    if not np.all(np.isfinite(full_qvel)):
        raise ValueError("MuJoCo qvel contains NaN or infinity")
    quaternion_norm = float(np.linalg.norm(full_qpos[3:7]))
    if not np.isfinite(quaternion_norm) or quaternion_norm < 1e-8:
        raise ValueError("MuJoCo root quaternion has near-zero norm")
    return full_qpos[7:].copy(), full_qvel[6:].copy()


class RootFrameAnchor:
    """Align the first human XY/yaw with the simulated robot XY/yaw."""

    def __init__(self) -> None:
        self.source_xy: np.ndarray | None = None
        self.source_yaw: float | None = None
        self.target_xy: np.ndarray | None = None
        self.target_yaw: float | None = None

    def reset(self) -> None:
        """Force the next live frame to anchor at the robot's current root."""

        self.source_xy = None
        self.source_yaw = None
        self.target_xy = None
        self.target_yaw = None

    def apply(self, source_qpos: np.ndarray, robot_qpos: np.ndarray) -> np.ndarray:
        qpos = source_qpos.copy()
        if self.source_xy is None:
            self.source_xy = qpos[:2].copy()
            self.source_yaw = _yaw_from_wxyz(qpos[3:7])
            self.target_xy = robot_qpos[:2].copy()
            self.target_yaw = _yaw_from_wxyz(robot_qpos[3:7])
            print(
                "Anchored first reference root: "
                f"human xy={self.source_xy.round(3)}, yaw={self.source_yaw:.3f} -> "
                f"robot xy={self.target_xy.round(3)}, yaw={self.target_yaw:.3f}",
                flush=True,
            )

        yaw_offset = float(self.target_yaw - self.source_yaw)
        c, s = np.cos(yaw_offset), np.sin(yaw_offset)
        delta_xy = qpos[:2] - self.source_xy
        qpos[:2] = self.target_xy + np.array(
            [c * delta_xy[0] - s * delta_xy[1], s * delta_xy[0] + c * delta_xy[1]],
            dtype=np.float32,
        )
        yaw_half = yaw_offset * 0.5
        yaw_rotation = np.array(
            [np.cos(yaw_half), 0.0, 0.0, np.sin(yaw_half)], dtype=np.float32
        )
        qpos[3:7] = _quat_mul_wxyz(yaw_rotation, qpos[3:7])
        qpos[3:7] /= max(float(np.linalg.norm(qpos[3:7])), 1e-8)
        return qpos


def _stationary_stand_reference(
    default_qpos: Sequence[float], robot_qpos: Sequence[float]
) -> np.ndarray:
    """Place the released upright stand at the robot's current XY and yaw."""

    stand = np.asarray(default_qpos, dtype=np.float32).reshape(-1).copy()
    robot = np.asarray(robot_qpos, dtype=np.float32).reshape(-1)
    if stand.shape != (36,) or robot.shape != (36,):
        raise ValueError("stand and robot qpos must each contain 36 values")
    if not np.all(np.isfinite(stand)) or not np.all(np.isfinite(robot)):
        raise ValueError("stand and robot qpos must be finite")
    stand[:2] = robot[:2]
    yaw = _yaw_from_wxyz(robot[3:7])
    stand[3:7] = np.asarray(
        [np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)],
        dtype=np.float32,
    )
    return stand


def _synchronize_latched_command(
    command: SafeCommand,
    *,
    profile: SafetyProfile,
    robot_q: Sequence[float],
    estop_latched: bool,
    fault_latched: bool,
    reason: str,
) -> SafeCommand:
    """Make telemetry/output reflect a latch raised at the actuator boundary.

    The final finite/torque check runs after the normal command was selected.
    If it raises a latch, publishing that earlier tracking state for one more
    cycle would be misleading.  Replace it with the same damping-safe command
    that will be used on the next cycle before publishing Redis telemetry.
    """

    if not estop_latched and not fault_latched:
        raise ValueError("a latched command requires E-stop or fault")
    measured_q = np.asarray(robot_q, dtype=np.float64).reshape(-1)
    if measured_q.shape != (profile.dof,):
        measured_q = np.zeros(profile.dof, dtype=np.float64)
    else:
        measured_q = np.nan_to_num(
            measured_q, nan=0.0, posinf=0.0, neginf=0.0
        )

    command.state = SafetyState.E_STOP if estop_latched else SafetyState.FAULT
    command.q = measured_q
    command.dq = np.zeros(profile.dof)
    command.tau_ff = np.zeros(profile.dof)
    command.kp = np.zeros(profile.dof)
    command.kd = np.full(profile.dof, profile.estop_kd)
    command.teleop_weight = 0.0
    command.reason = reason
    command.limited = True
    command.estop_latched = bool(estop_latched)
    command.fault_latched = bool(fault_latched)
    command.command_active = True
    command.real_hardware_allowed = False
    return command


def _state_value(value: SafetyState | str) -> str:
    return value.value if isinstance(value, SafetyState) else str(value)


def _legacy_direct_step(
    mj_sim,
    state,
    motor_targets: np.ndarray,
):
    """Delegate targets to the original released MuJoCo tracking step."""

    return mj_sim.step(state, motor_targets)


def _consume_simulation_remote_command(
    redis_client, key: str, last_sequence: int
) -> tuple[int, str] | None:
    """Consume one strictly newer simulation-only START/A/B command.

    The single one-shot key keeps the transport compatible with the alpha.5
    simulated START button.  New callers may send ``action`` values ``a`` and
    ``b`` on the same key; malformed, stale, and unknown events are deleted and
    rejected before they can retrigger.
    """

    raw = redis_client.get(key)
    if raw is None:
        return None
    # Delete before parsing so malformed/stale values cannot retrigger forever.
    redis_client.delete(key)
    try:
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("command must be a JSON object")
        if int(payload.get("schema_version", -1)) != 1:
            raise ValueError("unsupported schema_version")
        action = str(payload.get("action", "")).strip().lower()
        if action not in {"start", "a", "b"}:
            raise ValueError("action must be start, a, or b")
        sequence = int(payload["sequence"])
        if sequence <= int(last_sequence):
            raise ValueError("sequence is stale or replayed")
        return sequence, action
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"Rejected simulated remote command: {exc}", flush=True)
        return None


def _consume_simulation_start_command(
    redis_client, key: str, last_sequence: int
) -> int | None:
    """Backward-compatible START-only wrapper used by alpha.5 callers/tests."""

    command = _consume_simulation_remote_command(redis_client, key, last_sequence)
    if command is None:
        return None
    sequence, action = command
    if action != "start":
        print(f"Rejected simulated START command: action was {action}", flush=True)
        return None
    return sequence


def _environment_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be 0/1 or true/false")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", default="storage/ckpts/pns_wo_priv216.onnx"
    )
    parser.add_argument("--policy-type", choices=["mlp", "transformer"], default="mlp")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--redis-host", default="127.0.0.1")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="run MuJoCo without a viewer (intended for automated simulation checks)",
    )
    parser.add_argument(
        "--frequency",
        type=float,
        default=50.0,
        help="fixed policy/control rate; the released policies require 50 Hz",
    )
    parser.add_argument("--transition-seconds", type=float, default=1.5)
    parser.add_argument(
        "--startup-handover",
        action=argparse.BooleanOptionalAction,
        default=_environment_flag("HGPT_STARTUP_HANDOVER", False),
        help=(
            "simulation-only QA: start already standing, wait for simulated "
            "START, stabilize the released stand policy, then use A/B to blend "
            "between live CHINGMU and the stationary stand reference"
        ),
    )
    parser.add_argument(
        "--official-standup-trace",
        type=Path,
        default=None,
        help=(
            "deprecated alpha.5 compatibility option; accepted but ignored by "
            "the remote-only simulation workflow"
        ),
    )
    parser.add_argument(
        "--simulation-start-key",
        default="g1_sim_start_command",
        help="Redis one-shot key carrying simulation-only start/a/b events",
    )
    parser.add_argument(
        "--startup-stand-seconds",
        type=float,
        default=3.0,
        help="deprecated compatibility alias; stand starts already upright",
    )
    parser.add_argument("--startup-settle-seconds", type=float, default=0.5)
    parser.add_argument("--startup-blend-seconds", type=float, default=1.5)
    parser.add_argument(
        "--walk-checkpoint",
        default="storage/ckpts/G1-Walk/07140632_G1-Walk_v2.0.0_baseline.onnx",
        help="independent zero-velocity balance policy used during source loss",
    )
    parser.add_argument(
        "--safety-auto-resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="simulation convenience only; real hardware must use a physical deadman/resume",
    )
    parser.add_argument(
        "--debug-mode",
        action=argparse.BooleanOptionalAction,
        default=_environment_flag("HGPT_DEBUG_MODE", False),
        help="enable simulation debugging options; never authorizes hardware output",
    )
    parser.add_argument(
        "--safety-strategy-enabled",
        action=argparse.BooleanOptionalAction,
        default=_environment_flag("HGPT_SAFETY_STRATEGY_ENABLED", False),
        help="experimental simulation safety gateway; disabled unless explicitly enabled",
    )
    parser.add_argument("--safety-control-key", default="g1_safety_control")
    args = parser.parse_args()
    if abs(args.frequency - 50.0) > 1e-9:
        parser.error("Humanoid-GPT tracking and walk policies require --frequency 50")
    for duration_name in (
        "startup_stand_seconds",
        "startup_settle_seconds",
        "startup_blend_seconds",
    ):
        duration = float(getattr(args, duration_name))
        if not np.isfinite(duration) or duration < 0.0:
            parser.error(f"--{duration_name.replace('_', '-')} must be non-negative")
    repo = args.repo.resolve()
    os.chdir(repo)
    sys.path.insert(0, str(repo))

    import redis
    import mujoco
    from loop_rate_limiters import RateLimiter

    from src.motion.play_track import LiveRefConverter
    from src.motion.deploy_constants import KDs_walking, KPs_walking
    from src.motion.startup_handover import (
        RemoteHandoverCommand,
        RemoteHandoverPhase,
        RemoteOnlyHandover,
        blend_floating_base_qpos,
        smootherstep01,
    )
    from src.motion.walk_policy import WalkPolicy
    from src.motion import tracking_constants as consts
    from src.motion.infer_utils import (
        G1TrackInferFn,
        G1TrackMjSim,
        g1_infer_env_config,
    )
    from src.motion.tracking_policy import Args as PolicyArgs, get_policy_onnx
    from src.motion.sim_mj import get_sensor_data as mj_sensor

    checkpoint = str((repo / args.checkpoint).resolve())
    redis_client = redis.Redis(
        host=args.redis_host,
        port=args.redis_port,
        db=0,
        socket_timeout=0.015,
        socket_connect_timeout=0.050,
        protocol=2,
    )
    redis_client.ping()
    # A START click belongs to one simulator process only.  Clear a command
    # left by a crashed/previous process before announcing readiness.
    redis_client.delete(args.simulation_start_key)

    if args.startup_handover and args.official_standup_trace is not None:
        print(
            "Remote-only START/A/B workflow: --official-standup-trace is "
            "deprecated and ignored; MuJoCo starts from the released upright "
            "default pose and does not emulate a firmware crouch/stand action.",
            flush=True,
        )

    env_cfg = g1_infer_env_config(ctrl_dt=1.0 / args.frequency)
    policy_args = PolicyArgs(
        load_path=checkpoint,
        policy_type=args.policy_type,
        device=args.device,
    )
    policy = get_policy_onnx(policy_args)
    # The legacy-direct debug path intentionally recreates the original
    # Humanoid-GPT -> G1TrackMjSim.step control loop.  Do not even load the
    # independent safety fallback policy when that path is selected.  START
    # warm-up uses the released tracking policy itself with a stationary stand
    # reference, so its balance correction is already at full authority when
    # the simulation-only support rig is released.
    walk_policy = (
        WalkPolicy(str((repo / args.walk_checkpoint).resolve()))
        if args.safety_strategy_enabled else None
    )

    default_qpos = consts.DEFAULT_QPOS.astype(np.float32).copy()
    init_qpos = default_qpos.copy()
    mj_sim = G1TrackMjSim(
        init_qpos=init_qpos,
        headless=args.headless,
        ctrl_dt=1.0 / args.frequency,
    )
    if args.headless:
        # The upstream MJSim creates ``viewer`` only in graphical mode.
        mj_sim.viewer = None
    infer_class = G1TrackInferFn
    if args.policy_type == "transformer":
        from projects.tracking_transformer.infer_utils import (
            G1TrackTransformerInferFn,
        )
        infer_class = G1TrackTransformerInferFn
    infer_fn = infer_class(env_cfg, mj_sim.mj_model, policy, privileged=False)
    try:
        state = mj_sim.init_state()
    except (ImportError, OSError, RuntimeError) as exc:
        if args.headless:
            raise
        # Windows Application Control can reject MuJoCo's optional unsigned
        # _simulate extension while the signed/core physics runtime remains
        # usable.  Preserve the simulation pipeline in that environment; only
        # the local native viewer is disabled.
        print(
            "[Simulation] MuJoCo viewer is unavailable "
            f"({type(exc).__name__}: {exc}); continuing headless. Physics, "
            "policy inference, Redis state, and simulated controls remain active.",
            flush=True,
        )
        args.headless = True
        mj_sim.headless = True
        mj_sim.viewer = None
        state = mj_sim.init_state()
    state = mj_sim.reset(state)
    live_converter = LiveRefConverter(mj_sim.mj_model, 1.0 / args.frequency)

    if mj_sim.viewer is not None:
        camera = mj_sim.viewer.cam
        camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        camera.trackbodyid = 0
        camera.azimuth = 90.0
        camera.elevation = -20.0
        camera.distance = 2.0

    print("Humanoid-GPT live MuJoCo backend ready", flush=True)
    print(
        f"checkpoint={args.checkpoint}, policy_type={args.policy_type}, "
        f"device={args.device}",
        flush=True,
    )
    print("Simulation only: no Unitree SDK2, DDS, or motor commands.", flush=True)

    rate = RateLimiter(frequency=args.frequency, warn=False)
    previous_ref = None
    first_live_time = None
    last_frame_id = None
    steps = 0
    report_start = time.monotonic()
    stale_reported = False
    root_anchor = RootFrameAnchor()
    profile = SafetyProfile.simulation_default(29)
    if walk_policy is not None:
        profile.standby_q = walk_policy.default_qpos.astype(np.float64)
    profile.blend_in_s = max(0.75, float(args.transition_seconds))
    profile.recovery_s = max(1.0, float(args.transition_seconds))
    safety = G1SafetyGateway(profile) if args.safety_strategy_enabled else None
    # Legacy-direct must not even instantiate the new reference watchdog or
    # blend state machine.  They exist only in the safety-enabled branch.
    reference_watchdog = (
        ReferenceWatchdog(
            36,
            soft_stale_s=profile.input_soft_stale_s,
            hard_stale_s=profile.input_hard_stale_s,
            fresh_dwell_s=profile.fresh_dwell_s,
            manual_resume=not args.safety_auto_resume,
        )
        if args.safety_strategy_enabled
        else None
    )
    reference_blend = (
        ReferenceBlend(
            default_qpos,
            blend_in_s=profile.blend_in_s,
            blend_out_s=profile.recovery_s,
            quaternion_slice=slice(3, 7),
        )
        if args.safety_strategy_enabled
        else None
    )
    safety_session = uuid.uuid4().hex
    if args.safety_strategy_enabled:
        assert safety is not None
        safety.set_deadman(True, report_start)
        if not safety.arm(report_start):
            raise RuntimeError("simulation safety gateway refused initial arm")
    policy_seq = 0
    standby_seq = 0
    sim_tick = 0
    latest_sample: ReferenceSample | None = None
    latest_aligned_sample: ReferenceSample | None = None
    previous_reference_health: ReferenceHealth | None = None
    last_control_seq = -1
    last_safety_state: str | None = None
    source_session_id: str | None = None
    seen_source_sessions: set[str] = set()
    expected_source_dof_hash = hashlib.sha256(
        "\n".join(G1_29DOF_JOINT_NAMES).encode("utf-8")
    ).hexdigest()
    track_failures = 0
    standby_failures = 0
    last_control_time = report_start
    remote_handover = (
        RemoteOnlyHandover(
            stabilize_seconds=args.startup_settle_seconds,
            blend_seconds=args.startup_blend_seconds,
        )
        if args.startup_handover
        else None
    )
    remote_last_phase = None
    remote_last_advance_at = report_start
    remote_stand_reference = init_qpos.copy()
    remote_transition_anchor = init_qpos.copy()
    remote_effective_reference = init_qpos.copy()
    remote_latest_live_reference: np.ndarray | None = None
    remote_reference_stream_failed = False
    remote_reference_sequence = 0
    simulation_start_last_sequence = 0
    if args.startup_handover:
        state.mj_data.qpos[:] = init_qpos
        state.mj_data.qvel[:] = 0.0
        mujoco.mj_forward(mj_sim.mj_model, state.mj_data)
        print(
            "Remote handover waiting: MuJoCo is already upright at the released "
            "default stand. Press simulated START once to enable closed-loop stand "
            "stabilization; then A enters live CHINGMU and B returns to stand. "
            "No crouch pose, official motion trace, robot firmware, or DDS command "
            "is used by this simulator.",
            flush=True,
        )
    if args.safety_strategy_enabled:
        print(
            "G1 safety gateway enabled (simulation-only): independent balance policy, "
            "latched E-stop, source/policy/LowState watchdogs",
            flush=True,
        )
    else:
        print(
            "LEGACY DIRECT mode: G1 safety gateway, software E-stop, hard guards, "
            "and independent balance fallback are completely disabled. "
            "Humanoid-GPT targets go through the original G1TrackMjSim.step PD path. "
            "Simulation only; hardware output remains impossible.",
            flush=True,
        )

    try:
        while mj_sim.viewer is None or mj_sim.viewer.is_running():
            cycle_started = time.monotonic()
            now = cycle_started
            try:
                raw_packet = redis_client.get("action_qpos_g1_packet")
            except Exception as error:
                raw_packet = None
                if not stale_reported:
                    print(
                        f"Live reference store unavailable; entering watchdog recovery: {error}",
                        flush=True,
                    )
            # Freshness is evaluated after I/O; a slow Redis call must consume
            # watchdog budget instead of being hidden by cycle-start time.
            now = time.monotonic()
            candidate_sample: ReferenceSample | None = None
            frame_id = -1 if last_frame_id is None else last_frame_id
            if raw_packet is not None:
                try:
                    packet = json.loads(raw_packet)
                    if int(packet.get("schema_version", -1)) != 1:
                        raise ValueError("unsupported action_qpos_g1_packet schema")
                    packet_session = str(packet["session_id"])
                    if not (8 <= len(packet_session) <= 128 and packet_session.isascii()):
                        raise ValueError("invalid source session_id")
                    if packet.get("robot_model") != profile.model:
                        raise ValueError("source robot model mismatch")
                    if packet.get("dof_order_hash") != expected_source_dof_hash:
                        raise ValueError("source 29-DoF order hash mismatch")
                    frame_id = int(packet["frame_id"])
                    qpos_raw = np.asarray(packet["qpos"], dtype=np.float32)
                    if qpos_raw.shape != (36,) or not np.all(np.isfinite(qpos_raw)):
                        raise ValueError(f"qpos must be 36 finite values, got {qpos_raw.shape}")
                    quat_norm = float(np.linalg.norm(qpos_raw[3:7]))
                    if quat_norm < 1e-8:
                        raise ValueError("root quaternion has near-zero norm")
                    qpos_raw[3:7] /= quat_norm
                    packet_time = float(
                        packet.get("generated_monotonic", packet.get("monotonic"))
                    )
                    valid_until = float(packet["valid_until_monotonic"])
                    if not np.isfinite(packet_time) or packet_time > now + 0.020:
                        raise ValueError("source timestamp is invalid or in the future")
                    if not np.isfinite(valid_until) or valid_until < packet_time:
                        raise ValueError("source validity deadline is invalid")
                    if args.safety_strategy_enabled:
                        if now > valid_until:
                            raise ValueError("source packet validity deadline expired")
                    elif now - packet_time > 0.75:
                        # Match the pre-safety live simulator: repeat the last
                        # fresh target briefly, then freeze MuJoCo after 0.75 s.
                        raise ValueError("legacy-direct GMR packet is older than 0.75 s")
                    packet_seq = int(packet.get("sequence", frame_id))
                    if source_session_id != packet_session:
                        if packet_session in seen_source_sessions:
                            raise ValueError("previous source session replayed")
                        source_restarted = source_session_id is not None
                        if args.safety_strategy_enabled:
                            assert reference_watchdog is not None
                            reference_watchdog.reset()
                        if source_restarted and args.safety_strategy_enabled:
                            assert reference_watchdog is not None
                            # A producer restart can reset sequence and jump
                            # its reference.  Require the same continuous-fresh
                            # dwell as a hard disconnect before using it.
                            reference_watchdog.force_standby()
                        source_session_id = packet_session
                        seen_source_sessions.add(packet_session)
                        latest_sample = None
                        latest_aligned_sample = None
                        previous_reference_health = None
                        print(
                            f"Accepted new GMR source session {packet_session[:12]}...",
                            flush=True,
                        )
                    candidate_sample = ReferenceSample(
                        packet_seq, packet_time, qpos_raw, np.zeros(36), valid=True
                    )
                    last_frame_id = frame_id
                except Exception as error:
                    if not stale_reported:
                        print(f"Rejected live reference packet: {error}", flush=True)
                    stale_reported = True

            remote_reference_override: np.ndarray | None = None
            remote_sample = None
            if remote_handover is not None:
                command = _consume_simulation_remote_command(
                    redis_client,
                    args.simulation_start_key,
                    simulation_start_last_sequence,
                )
                if command is not None:
                    command_sequence, action = command
                    simulation_start_last_sequence = command_sequence
                    accepted = False
                    if action == "start":
                        accepted = remote_handover.command(action)
                        if accepted:
                            remote_stand_reference = _stationary_stand_reference(
                                default_qpos, state.mj_data.qpos
                            )
                            remote_transition_anchor = remote_stand_reference.copy()
                            remote_effective_reference = remote_stand_reference.copy()
                            live_converter.reset()
                            previous_ref = None
                            reset_history = getattr(infer_fn, "reset_history", None)
                            if callable(reset_history):
                                reset_history()
                            infer_fn.info["last_action"].fill(0.0)
                            infer_fn.info["nn_action"].fill(0.0)
                            infer_fn.info["motor_targets"][:] = state.mj_data.qpos[7:]
                    elif action == "a":
                        if candidate_sample is None:
                            print(
                                "Remote A ignored: no fresh CHINGMU reference is "
                                "available; stand policy remains active.",
                                flush=True,
                            )
                        else:
                            root_anchor.reset()
                            remote_latest_live_reference = root_anchor.apply(
                                np.asarray(candidate_sample.position, dtype=np.float32),
                                state.mj_data.qpos,
                            )
                            remote_transition_anchor = remote_effective_reference.copy()
                            accepted = remote_handover.command(action)
                    elif action == "b":
                        remote_transition_anchor = remote_effective_reference.copy()
                        remote_stand_reference = _stationary_stand_reference(
                            default_qpos, state.mj_data.qpos
                        )
                        accepted = remote_handover.command(action)

                    if accepted:
                        print(
                            f"Remote command accepted: {action.upper()} | "
                            f"sequence {command_sequence}",
                            flush=True,
                        )
                    elif not (action == "a" and candidate_sample is None):
                        print(
                            f"Remote command ignored in phase "
                            f"{remote_handover.phase.value}: {action.upper()}",
                            flush=True,
                        )

                # Keep the MuJoCo rehearsal behavior equal to the real
                # remote-only path: once a live source becomes stale, freeze
                # the last effective pose and blend back to a stationary
                # feedback-policy stand.  A recovered source stays hot but is
                # not re-entered until the operator presses A again.
                live_phase = remote_handover.phase in {
                    RemoteHandoverPhase.BLEND_TO_LIVE,
                    RemoteHandoverPhase.LIVE_TRACKING,
                }
                if live_phase and candidate_sample is None:
                    remote_transition_anchor = remote_effective_reference.copy()
                    remote_stand_reference = _stationary_stand_reference(
                        default_qpos, state.mj_data.qpos
                    )
                    remote_handover.command(RemoteHandoverCommand.STAND)
                    if not remote_reference_stream_failed:
                        print(
                            "Live CHINGMU reference became unavailable; "
                            "blending back to policy stand. A is required "
                            "after the source recovers.",
                            flush=True,
                        )
                    remote_reference_stream_failed = True
                elif candidate_sample is not None and remote_reference_stream_failed:
                    print(
                        "Live CHINGMU reference recovered; remaining in policy "
                        "stand until simulated A is pressed again.",
                        flush=True,
                    )
                    remote_reference_stream_failed = False

                remote_dt = max(0.0, now - remote_last_advance_at)
                remote_last_advance_at = now
                remote_sample = remote_handover.advance(remote_dt)
                if remote_sample.phase != remote_last_phase:
                    phase_messages = {
                        RemoteHandoverPhase.WAIT_FOR_START: (
                            "REMOTE phase: upright and waiting for START"
                        ),
                        RemoteHandoverPhase.STABILIZING_STAND: (
                            "REMOTE phase: released tracking policy stabilizing the "
                            "stationary upright stand reference"
                        ),
                        RemoteHandoverPhase.WAIT_FOR_LIVE: (
                            "REMOTE phase: stand policy stable; waiting for A"
                        ),
                        RemoteHandoverPhase.BLEND_TO_LIVE: (
                            "REMOTE phase: A accepted; blending stand reference to "
                            "live CHINGMU"
                        ),
                        RemoteHandoverPhase.LIVE_TRACKING: (
                            "REMOTE phase: live CHINGMU tracking engaged; B returns "
                            "to stand"
                        ),
                        RemoteHandoverPhase.BLEND_TO_STAND: (
                            "REMOTE phase: B accepted; blending live reference back "
                            "to stationary stand"
                        ),
                        RemoteHandoverPhase.STAND_HOLD: (
                            "REMOTE phase: stationary stand restored; A may re-enter live"
                        ),
                    }
                    print(
                        f"{phase_messages[remote_sample.phase]} | "
                        f"root z {float(state.mj_data.qpos[2]):.3f} m",
                        flush=True,
                    )
                    remote_last_phase = remote_sample.phase

                if remote_sample.phase == RemoteHandoverPhase.WAIT_FOR_START:
                    state.mj_data.qpos[:] = remote_stand_reference
                    state.mj_data.qvel[:] = 0.0
                    mujoco.mj_forward(mj_sim.mj_model, state.mj_data)
                    mj_sim.view(state)
                    rate.sleep()
                    continue

                if (
                    candidate_sample is not None
                    and remote_sample.phase
                    in {
                        RemoteHandoverPhase.BLEND_TO_LIVE,
                        RemoteHandoverPhase.LIVE_TRACKING,
                    }
                ):
                    latest_sample = candidate_sample
                    remote_latest_live_reference = root_anchor.apply(
                        np.asarray(candidate_sample.position, dtype=np.float32),
                        state.mj_data.qpos,
                    )

                if remote_sample.phase == RemoteHandoverPhase.BLEND_TO_LIVE:
                    if remote_latest_live_reference is None:
                        remote_reference_override = remote_transition_anchor.copy()
                    else:
                        remote_reference_override = blend_floating_base_qpos(
                            remote_transition_anchor,
                            remote_latest_live_reference,
                            smootherstep01(remote_sample.phase_progress),
                        )
                elif remote_sample.phase == RemoteHandoverPhase.LIVE_TRACKING:
                    remote_reference_override = (
                        remote_stand_reference.copy()
                        if remote_latest_live_reference is None
                        else remote_latest_live_reference.copy()
                    )
                elif remote_sample.phase == RemoteHandoverPhase.BLEND_TO_STAND:
                    remote_reference_override = blend_floating_base_qpos(
                        remote_transition_anchor,
                        remote_stand_reference,
                        smootherstep01(remote_sample.phase_progress),
                    )
                else:
                    remote_reference_override = remote_stand_reference.copy()

                remote_effective_reference = remote_reference_override.copy()
                remote_reference_sequence += 1
                candidate_sample = ReferenceSample(
                    remote_reference_sequence,
                    now,
                    remote_reference_override,
                    np.zeros(36),
                    valid=True,
                )

            if not args.safety_strategy_enabled:
                # Reproduce the pre-safety online simulator before any
                # ReferenceWatchdog/ReferenceBlend/G1SafetyGateway work.  A
                # fresh GMR frame drives the released tracking policy and the
                # repository's original MuJoCo PD step directly.  A stale or
                # invalid frame freezes simulation time, exactly as the older
                # live path did; it does not synthesize a standby target.
                motor_targets: np.ndarray | None = None
                legacy_reference_health = "WAIT_FRESH"
                if candidate_sample is not None:
                    latest_sample = candidate_sample
                    source_age = max(0.0, now - float(candidate_sample.timestamp))
                    legacy_reference_health = "LIVE"
                    try:
                        aligned = (
                            np.asarray(candidate_sample.position, dtype=np.float32)
                            if remote_reference_override is not None
                            else root_anchor.apply(
                                np.asarray(candidate_sample.position, dtype=np.float32),
                                state.mj_data.qpos,
                            )
                        )
                        ref_new = live_converter.convert(aligned)
                        ref_curr = previous_ref if previous_ref is not None else ref_new
                        candidate_targets = np.asarray(
                            infer_fn.infer_onnx(
                                state, {"ref_curr": ref_curr, "ref_next": ref_new}
                            ),
                            dtype=np.float64,
                        ).reshape(-1)
                        if candidate_targets.shape != (29,) or not np.all(
                            np.isfinite(candidate_targets)
                        ):
                            raise ValueError(
                                "legacy tracking policy did not return 29 finite targets"
                            )
                        motor_targets = candidate_targets
                        # START blends only the stationary stand reference into
                        # the CHINGMU reference.  The tracking policy output is
                        # never attenuated: its balance correction must remain
                        # at full authority from the first unsupported step.
                        previous_ref = ref_new
                        state = _legacy_direct_step(mj_sim, state, motor_targets)
                        policy_seq += 1
                        steps += 1
                        track_failures = 0
                    except Exception as error:
                        # The pre-safety direct runner was fail-fast.  Do not
                        # disguise a policy/conversion failure as stale input
                        # while the UI continues to report LEGACY_DIRECT.
                        raise RuntimeError(
                            "legacy-direct tracking failed; simulation stopped: "
                            f"{type(error).__name__}: {error}"
                        ) from error
                else:
                    source_age = (
                        1e6
                        if latest_sample is None
                        else max(0.0, now - float(latest_sample.timestamp))
                    )
                    legacy_reference_health = (
                        "WAIT_FRESH" if latest_sample is None else "HARD_STALE"
                    )

                control_now = time.monotonic()
                legacy_reason = (
                    "safety lock OFF: original Humanoid-GPT direct MuJoCo PD path"
                    if motor_targets is not None
                    else "safety lock OFF: no fresh policy target; MuJoCo time frozen"
                )
                mj_sim.view(state)
                try:
                    if motor_targets is None:
                        redis_client.delete("g1_raw_policy_target_packet")
                        redis_client.delete("g1_legacy_direct_sim_output")
                    else:
                        redis_client.set(
                            "g1_raw_policy_target_packet",
                            encode_policy_target(
                                PolicyTargetEnvelope(
                                    session_id=safety_session,
                                    seq=policy_seq,
                                    sender_monotonic=control_now,
                                    valid_for_s=profile.policy_output_stale_s,
                                    backend="humanoid_gpt_legacy_direct",
                                    state_source="mujoco",
                                    model=profile.model,
                                    dof_hash=profile.dof_hash,
                                    q=motor_targets,
                                    dq=np.zeros(29),
                                    tau_ff=np.zeros(29),
                                    kp=consts.KPs,
                                    kd=consts.KDs,
                                    active_mask=np.ones(29, dtype=bool),
                                    deadman=False,
                                    teleop_source_age_s=source_age,
                                    teleop_source_valid=True,
                                )
                            ),
                            px=100,
                        )
                        redis_client.set(
                            "g1_legacy_direct_sim_output",
                            json.dumps(
                                {
                                    "schema_version": 1,
                                    "sequence": policy_seq,
                                    "generated_monotonic": control_now,
                                    "q": motor_targets.tolist(),
                                    "kp": np.asarray(consts.KPs, dtype=float).tolist(),
                                    "kd": np.asarray(consts.KDs, dtype=float).tolist(),
                                    "state_source": "mujoco",
                                    "real_hardware_allowed": False,
                                },
                                allow_nan=False,
                            ),
                            px=100,
                        )
                    redis_client.delete("g1_raw_standby_target_packet")
                    redis_client.delete("g1_safe_sim_output")
                    redis_client.set(
                        "g1_safety_status",
                        json.dumps(
                            {
                                "schema_version": 1,
                                "reported_monotonic": control_now,
                                "state": "LEGACY_DIRECT",
                                "reason": legacy_reason,
                                "estop_latched": False,
                                "fault_latched": False,
                                "limited": False,
                                "teleop_weight": (
                                    1.0 if motor_targets is not None else 0.0
                                ),
                                "reference_health": legacy_reference_health,
                                "reference_age_ms": source_age * 1000.0,
                                "strategy_enabled": False,
                                "debug_mode": bool(args.debug_mode),
                                "bypass_active": True,
                                "hard_guards_enabled": False,
                                "state_source": "mujoco",
                                "real_hardware_allowed": False,
                            },
                            allow_nan=False,
                        ),
                        px=1000,
                    )
                except Exception as error:
                    if not stale_reported:
                        print(f"Legacy-direct telemetry publish warning: {error}", flush=True)

                if last_safety_state != "LEGACY_DIRECT":
                    print(f"Control state: LEGACY_DIRECT | {legacy_reason}", flush=True)
                    last_safety_state = "LEGACY_DIRECT"
                elapsed = control_now - report_start
                if elapsed >= 2.0:
                    print(
                        f"Humanoid-GPT {steps / elapsed:.1f} Hz | frame {frame_id} | "
                        f"qpos z {float(state.mj_data.qpos[2]):.3f} m | "
                        f"packet age {source_age*1000:.1f} ms | safety OFF",
                        flush=True,
                    )
                    steps = 0
                    report_start = control_now
                stale_reported = candidate_sample is None
                rate.sleep()
                continue

            assert reference_watchdog is not None and reference_blend is not None
            decision = reference_watchdog.update(candidate_sample, now)
            if candidate_sample is not None and decision.health != ReferenceHealth.INVALID:
                latest_sample = candidate_sample
            if decision.health == ReferenceHealth.STANDBY and args.safety_auto_resume:
                # This is simulation convenience only.  The real gateway never
                # substitutes a web/software decision for its physical deadman.
                reference_watchdog.request_resume()
                decision = reference_watchdog.update(candidate_sample, now)

            source_live = decision.use_live and latest_sample is not None
            source_age = decision.age_s if np.isfinite(decision.age_s) else 1e6
            if source_live:
                if previous_reference_health != ReferenceHealth.LIVE:
                    # Do not chase the human's absolute position accumulated
                    # while disconnected.  Re-anchor at the robot's current root
                    # and warm-start the tracking policy history.
                    root_anchor.reset()
                    remote_policy_is_warm = (
                        remote_reference_override is not None
                        and previous_ref is not None
                    )
                    if not remote_policy_is_warm:
                        live_converter.reset()
                        previous_ref = None
                        reset_history = getattr(infer_fn, "reset_history", None)
                        if callable(reset_history):
                            reset_history()
                        infer_fn.info["last_action"].fill(0.0)
                        infer_fn.info["nn_action"].fill(0.0)
                        infer_fn.info["motor_targets"][:] = state.mj_data.qpos[7:]
                aligned = (
                    np.asarray(latest_sample.position, dtype=np.float32)
                    if remote_reference_override is not None
                    else root_anchor.apply(
                        np.asarray(latest_sample.position, dtype=np.float32),
                        state.mj_data.qpos,
                    )
                )
                latest_aligned_sample = ReferenceSample(
                    latest_sample.seq,
                    latest_sample.timestamp,
                    aligned,
                    np.zeros(36),
                    valid=True,
                )
                if first_live_time is None:
                    first_live_time = now
            elif previous_reference_health == ReferenceHealth.LIVE:
                # Stand at the current XY/yaw instead of pulling the reference
                # toward the MuJoCo world origin during a dropout.
                robot_root = state.mj_data.qpos.copy()
                standby_reference = default_qpos.copy()
                standby_reference[:2] = robot_root[:2]
                yaw = _yaw_from_wxyz(robot_root[3:7])
                standby_reference[3:7] = np.array(
                    [np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)],
                    dtype=np.float32,
                )
                reference_blend.standby[:] = standby_reference

            motor_targets: np.ndarray | None = None
            reference_output = None
            try:
                reference_output = reference_blend.step(
                    decision, latest_aligned_sample, now
                )
                qpos_reference = reference_output.position.astype(np.float32)
                ref_new = live_converter.convert(qpos_reference)
                ref_curr = previous_ref if previous_ref is not None else ref_new
                candidate_targets = np.asarray(
                    infer_fn.infer_onnx(
                        state, {"ref_curr": ref_curr, "ref_next": ref_new}
                    ),
                    dtype=np.float64,
                ).reshape(-1)
                if candidate_targets.shape != (29,) or not np.all(
                    np.isfinite(candidate_targets)
                ):
                    raise ValueError(
                        f"tracking policy returned {candidate_targets.shape} non-finite/invalid targets"
                    )
                motor_targets = candidate_targets
                previous_ref = ref_new
                track_failures = 0
            except Exception as error:
                track_failures += 1
                if track_failures == 1 or track_failures % 25 == 0:
                    print(
                        f"Tracking policy failure #{track_failures}; safety recovery remains active: "
                        f"{type(error).__name__}: {error}",
                        flush=True,
                    )

            assert safety is not None and walk_policy is not None
            sensor_valid = True
            try:
                gyro = np.asarray(
                    mj_sensor(mj_sim.mj_model, state.mj_data, "gyro_pelvis"),
                    dtype=np.float64,
                ).copy()
                if gyro.shape != (3,) or not np.all(np.isfinite(gyro)):
                    raise ValueError(f"invalid pelvis gyro {gyro}")
            except Exception as error:
                sensor_valid = False
                gyro = np.zeros(3, dtype=np.float64)
                print(f"MuJoCo state sensor invalid: {error}", flush=True)

            try:
                measured_joint_q, measured_joint_dq = (
                    _validate_mujoco_kinematic_state(
                        state.mj_data.qpos, state.mj_data.qvel
                    )
                )
            except ValueError as error:
                sensor_valid = False
                measured_joint_q = np.asarray(
                    state.mj_data.qpos[7:], dtype=np.float64
                ).reshape(-1)
                measured_joint_dq = np.asarray(
                    state.mj_data.qvel[6:], dtype=np.float64
                ).reshape(-1)
                if not stale_reported:
                    print(
                        f"Invalid MuJoCo kinematic state; actuator stepping will be frozen: {error}",
                        flush=True,
                    )

            standby_targets: np.ndarray | None = None
            if sensor_valid:
                try:
                    candidate_standby = np.asarray(
                        walk_policy.infer(
                            state.mj_data.qpos[3:7],
                            gyro,
                            state.mj_data.qpos[7:],
                            state.mj_data.qvel[6:],
                            np.zeros(3, dtype=np.float32),
                        ),
                        dtype=np.float64,
                    ).reshape(-1)
                    if candidate_standby.shape != (29,) or not np.all(
                        np.isfinite(candidate_standby)
                    ):
                        raise ValueError(
                            f"standby policy returned {candidate_standby.shape} non-finite/invalid targets"
                        )
                    standby_targets = candidate_standby
                    standby_failures = 0
                except Exception as error:
                    standby_failures += 1
                    if standby_failures == 1 or standby_failures % 25 == 0:
                        print(
                            f"Standby balance policy failure #{standby_failures}: "
                            f"{type(error).__name__}: {error}",
                            flush=True,
                        )

            sim_tick += 1
            policy_seq += 1
            standby_seq += 1
            control_now = time.monotonic()
            actual_dt = max(1e-6, control_now - last_control_time)
            source_age = (
                1e6
                if latest_sample is None
                else max(0.0, control_now - float(latest_sample.timestamp))
            )
            teleop_source_valid = decision.health in {
                ReferenceHealth.LIVE,
                ReferenceHealth.SOFT_STALE,
                ReferenceHealth.HARD_STALE,
            }
            robot_state = RobotState(
                timestamp=control_now,
                tick=sim_tick,
                q=state.mj_data.qpos[7:].copy(),
                dq=state.mj_data.qvel[6:].copy(),
                imu_rpy=_rpy_from_wxyz(state.mj_data.qpos[3:7]),
                imu_gyro=gyro,
                model=profile.model,
                dof_hash=profile.dof_hash,
                valid=sensor_valid,
            )
            track_command = None if motor_targets is None else PolicyCommand(
                seq=policy_seq,
                timestamp=control_now,
                q=motor_targets,
                dq=np.zeros(29),
                tau_ff=np.zeros(29),
                kp=consts.KPs,
                kd=consts.KDs,
                model=profile.model,
                dof_hash=profile.dof_hash,
                deadman=True,
                source_valid=True,
            )
            standby_command = None if standby_targets is None else PolicyCommand(
                seq=standby_seq,
                timestamp=control_now,
                q=standby_targets,
                dq=np.zeros(29),
                tau_ff=np.zeros(29),
                kp=KPs_walking,
                kd=KDs_walking,
                model=profile.model,
                dof_hash=profile.dof_hash,
                deadman=True,
                source_valid=True,
            )

            try:
                raw_control = redis_client.get(args.safety_control_key)
            except Exception as error:
                raw_control = None
                if not stale_reported:
                    print(f"Safety control store unavailable: {error}", flush=True)
            if raw_control is not None:
                try:
                    control = json.loads(raw_control)
                    control_seq = int(control.get("sequence", -1))
                    if control_seq > last_control_seq:
                        last_control_seq = control_seq
                        action = str(control.get("action", "")).lower()
                        if action == "estop":
                            safety.emergency_stop(control_now, "software E-stop requested")
                        elif action == "disarm":
                            safety.disarm(control_now)
                        elif action == "reset":
                            operator_confirmed = control.get("operator_confirmed")
                            estop_released = control.get("physical_estop_released")
                            if not isinstance(operator_confirmed, bool) or not isinstance(
                                estop_released, bool
                            ):
                                raise ValueError("reset confirmations must be JSON booleans")
                            reset_ok = safety.reset(
                                control_now,
                                robot_state,
                                operator_confirmed=operator_confirmed,
                                physical_estop_released=estop_released,
                            )
                            if reset_ok and args.safety_auto_resume:
                                # Simulation-only convenience.  Real hardware
                                # requires a separate physical arm action.
                                safety.set_deadman(True, control_now)
                                safety.arm(control_now)
                        elif action == "arm":
                            safety.set_deadman(True, control_now)
                            safety.arm(control_now)
                        elif action == "resume":
                            reference_watchdog.request_resume()
                            safety.set_deadman(True, control_now)
                            safety.resume(control_now)
                        elif action == "deadman":
                            held = control.get("held")
                            if not isinstance(held, bool):
                                raise ValueError("deadman held must be a JSON boolean")
                            safety.set_deadman(held, control_now)
                except Exception as error:
                    print(f"Ignored invalid safety control request: {error}", flush=True)

            if (
                args.safety_auto_resume
                and safety.state == SafetyState.STANDBY
                and decision.use_live
            ):
                safety.resume(control_now)
            safe = safety.step(
                track_command,
                robot_state,
                control_now,
                standby_policy=standby_command,
                teleop_source_age_s=source_age,
                teleop_source_valid=teleop_source_valid,
                dt=actual_dt,
            )
            last_control_time = control_now

            # Both normal and debug-bypass paths pass through this single
            # MuJoCo actuator boundary; hardware output is impossible here.
            command_finite = all(
                np.all(np.isfinite(values))
                for values in (safe.q, safe.dq, safe.tau_ff, safe.kp, safe.kd)
            )
            actuator_latch_reason: str | None = None
            if not command_finite:
                actuator_latch_reason = "non-finite actuator command blocked"
                safety.emergency_stop(control_now, actuator_latch_reason)
                state.mj_data.ctrl[:] = 0.0
                print("Blocked non-finite actuator command; simulation step frozen", flush=True)
            elif not sensor_valid:
                state.mj_data.ctrl[:] = 0.0
            else:
                for _ in range(mj_sim.num_sim_substeps):
                    torque = (
                        safe.tau_ff
                        + safe.kp * (safe.q - state.mj_data.qpos[7:])
                        + safe.kd * (safe.dq - state.mj_data.qvel[6:])
                    )
                    if not np.all(np.isfinite(torque)):
                        actuator_latch_reason = "non-finite torque blocked"
                        state.mj_data.ctrl[:] = 0.0
                        safety.emergency_stop(control_now, actuator_latch_reason)
                        break
                    bounded_torque = np.clip(
                        torque,
                        -profile.max_predicted_torque,
                        profile.max_predicted_torque,
                    )
                    if not np.allclose(
                        bounded_torque, torque, rtol=0.0, atol=1e-12
                    ):
                        safe.limited = True
                    state.mj_data.ctrl[:] = bounded_torque
                    mujoco.mj_step(mj_sim.mj_model, state.mj_data)
            if actuator_latch_reason is not None:
                safe = _synchronize_latched_command(
                    safe,
                    profile=profile,
                    robot_q=measured_joint_q,
                    estop_latched=safety.estop_latched,
                    fault_latched=safety.fault_latched,
                    reason=actuator_latch_reason,
                )
            mj_sim.view(state)
            steps += 1
            safe_state_value = _state_value(safe.state)

            try:
                active_mask = np.ones(29, dtype=bool)
                envelope_common = dict(
                    session_id=safety_session,
                    sender_monotonic=control_now,
                    valid_for_s=profile.policy_output_stale_s,
                    state_source="mujoco",
                    model=profile.model,
                    dof_hash=profile.dof_hash,
                    dq=np.zeros(29),
                    tau_ff=np.zeros(29),
                    active_mask=active_mask,
                    deadman=True,
                    teleop_source_age_s=source_age,
                    teleop_source_valid=teleop_source_valid,
                )
                if motor_targets is None:
                    redis_client.delete("g1_raw_policy_target_packet")
                else:
                    redis_client.set(
                        "g1_raw_policy_target_packet",
                        encode_policy_target(
                            PolicyTargetEnvelope(
                                seq=policy_seq,
                                backend="humanoid_gpt",
                                q=motor_targets,
                                kp=consts.KPs,
                                kd=consts.KDs,
                                **envelope_common,
                            )
                        ),
                        px=100,
                    )
                if standby_targets is None:
                    redis_client.delete("g1_raw_standby_target_packet")
                else:
                    redis_client.set(
                        "g1_raw_standby_target_packet",
                        encode_policy_target(
                            PolicyTargetEnvelope(
                                seq=standby_seq,
                                backend="g1_walk_standby",
                                q=standby_targets,
                                kp=KPs_walking,
                                kd=KDs_walking,
                                **envelope_common,
                            )
                        ),
                        px=100,
                    )
                redis_client.set(
                    "g1_safe_sim_output",
                    json.dumps(
                        {
                            "schema_version": 1,
                            "sequence": int(policy_seq),
                            "generated_monotonic": control_now,
                            "state": safe_state_value,
                            "q": np.asarray(safe.q, dtype=float).tolist(),
                            "dq": np.asarray(safe.dq, dtype=float).tolist(),
                            "tau_ff": np.asarray(safe.tau_ff, dtype=float).tolist(),
                            "kp": np.asarray(safe.kp, dtype=float).tolist(),
                            "kd": np.asarray(safe.kd, dtype=float).tolist(),
                            "state_source": "mujoco",
                            "real_hardware_allowed": False,
                        },
                        allow_nan=False,
                    ),
                    px=100,
                )
                redis_client.set(
                    "g1_safety_status",
                    json.dumps(
                        {
                            "schema_version": 1,
                            "reported_monotonic": control_now,
                            "state": safe_state_value,
                            "reason": safe.reason,
                            "estop_latched": safe.estop_latched,
                            "fault_latched": safe.fault_latched,
                            "limited": safe.limited,
                            "teleop_weight": safe.teleop_weight,
                            "reference_health": decision.health.value,
                            "reference_age_ms": source_age * 1000.0,
                            "strategy_enabled": True,
                            "debug_mode": args.debug_mode,
                            "bypass_active": False,
                            "hard_guards_enabled": True,
                            "state_source": "mujoco",
                            "real_hardware_allowed": False,
                        }
                    ),
                    px=1000,
                )
            except Exception as error:
                if not stale_reported:
                    print(f"Safety telemetry publish warning: {error}", flush=True)

            if safe_state_value != last_safety_state:
                print(
                    f"Safety state: {safe_state_value} | {safe.reason}", flush=True
                )
                last_safety_state = safe_state_value
            stale_reported = decision.health != ReferenceHealth.LIVE
            previous_reference_health = decision.health
            elapsed = control_now - report_start
            if elapsed >= 2.0:
                print(
                    f"Humanoid-GPT {steps / elapsed:.1f} Hz | frame {frame_id} | "
                    f"source {decision.health.value}/{source_age*1000:.1f} ms | "
                    f"safety {safe_state_value} w={safe.teleop_weight:.2f}",
                    flush=True,
                )
                steps = 0
                report_start = control_now
            rate.sleep()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            redis_client.delete(args.simulation_start_key)
        except Exception:
            pass
        if getattr(mj_sim, "viewer", None) is not None:
            mj_sim.viewer.close()
        print("Humanoid-GPT live simulation stopped.", flush=True)


if __name__ == "__main__":
    main()
