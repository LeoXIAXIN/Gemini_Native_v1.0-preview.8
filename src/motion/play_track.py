"""Main entry point for tracking deployment (simulation and real robot).

Modified in July 2026 to add the CHINGMU/GMR Redis reference source and local
deployment integration while preserving the released Humanoid-GPT policy path.

Usage:
    # Simulation (offline tracking)
    python -m deploy.play_track --track_dir storage/data/exps

    # Simulation (online retarget, Noitom PNLink)
    python -m deploy.play_track --track_dir storage/data/exps --mocap_type pnlink

    # Simulation (online retarget, Xsens MVN over TCP on port 9763)
    python -m deploy.play_track --track_dir storage/data/exps \
        --mocap_type xsens --xsens_protocol tcp --xsens_port 9763

    # Real robot
    python -m deploy.play_track --real --net enx6c1ff7579fdf

    # Real robot with an existing CHINGMU/GMR Redis reference
    python -m deploy.play_track --real --net enx6c1ff7579fdf \
        --mocap_type chingmu_redis

Modes (keyboard number keys):
    0 = Walk policy
    1 = Online retarget (mocap)
    2+ = Offline tracking (reference trajectories from track_dir)
"""

from __future__ import annotations

import os
import time
import signal
import sys
import threading
import mujoco
from pathlib import Path
from dataclasses import dataclass

import tyro
import numpy as np
from jax import tree_util as jtu
from loop_rate_limiters import RateLimiter

from src.motion import tracking_constants as consts
from src.motion.tracking_constants import KPT_NAMES
from src.motion.convert_qpos2kpt import qpos2kpt
from src.motion.transforms_np import base2navi, quat2mat
from src.motion.sim_mj import get_sensor_data as mj_sensor
from src.motion.tracking_policy import Args as PolicyArgs, get_policy_onnx
from src.motion.infer_utils import G1TrackMjSim, G1TrackInferFn, g1_infer_env_config, apply_ema_qpos

from src.motion.walk_policy import WalkPolicy
from src.motion.keyboard_cmd import DeployKeyboardCMD
from src.adapters.chingmu_redis import ChingMuRedisMocapBuffer
from src.motion.deploy_constants import DEFAULT_QPOS as DEFAULT_QPOS_JOINT, KPs_walking, KDs_walking
from src.motion.startup_handover import (
    HandoverPhase,
    RemoteHandoverCommand,
    RemoteHandoverPhase,
    RemoteOnlyHandover,
    StartupHandover,
    blend_floating_base_qpos,
    smootherstep01,
)
from src.motion.remote_handover import (
    MotionOwnerMonitor,
    OFFICIAL_MOTION_OWNER,
    ReleasedChordGate,
    RemoteHandoverError,
    SafeRemoteButtonEdges,
    StableStandDwell,
    checked_motion_owner,
    create_motion_switcher_client,
    stable_stand_reason,
    validate_policy_stand_target,
    validate_shadow_target_match,
)
from src.motion.tracking_metrics import (
    calculate_kpt_mae_error,
    calculate_joint_tracking_error,
    calculate_root_tracking_error,
    calculate_trajectory_length,
    calculate_max_errors,
)

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
import pygame  # noqa: E402  – needed for DeployKeyboardCMD fork workaround


# ---------------------------------------------------------------------------
# Mocap buffer with caching and linear extrapolation
# ---------------------------------------------------------------------------

class MocapBuffer:
    """Read the latest mocap data from the shared buffer."""

    def __init__(self, buf, ts):
        self._buf = buf
        self._ts = ts

    def read(self) -> tuple[np.ndarray, float]:
        from src.motion.retarget import read_mocap_buffer
        qpos_full, ts = read_mocap_buffer(self._buf, self._ts)
        return qpos_full, ts


def _stop_and_wait_recurrent_thread(
    thread,
    timeout_seconds: float = 1.0,
) -> None:
    """Stop a Unitree recurrent thread and wait until it cannot publish again.

    The released ``unitree_sdk2py`` implementation uses ``Wait()`` itself to
    request termination.  Some SDK forks also expose ``Stop()``, so call it
    when present before the mandatory bounded ``Wait(timeout)`` join.  A thread
    which cannot be positively joined must never be followed by another motor
    write (including shutdown damping).
    """
    if thread is None:
        return
    stop = getattr(thread, "Stop", None)
    if callable(stop):
        stop()
    wait = getattr(thread, "Wait", None)
    if not callable(wait):
        raise RuntimeError("RecurrentThread has no Wait() method")
    timeout = float(timeout_seconds)
    if not np.isfinite(timeout) or timeout <= 0.0:
        raise ValueError("thread join timeout must be positive and finite")
    try:
        joined = wait(timeout)
    except TypeError as exc:
        raise RuntimeError(
            "RecurrentThread.Wait() does not support a bounded timeout"
        ) from exc
    if joined is True:
        return
    if joined is False:
        raise TimeoutError(
            f"policy thread did not stop within {timeout:.3f} seconds"
        )

    # The released Unitree Python SDK's RecurrentThread.Wait(timeout) discards
    # the boolean returned by its Future base class and therefore returns None
    # even after a successful join.  Confirm the terminal Future state without
    # blocking; code 0 is ready and code 2 is failed-but-terminated.  Either
    # state proves that this thread can no longer publish.
    get_result = getattr(thread, "GetResult", None)
    if callable(get_result):
        result = get_result(0.0)
        if getattr(result, "code", None) in (0, 2):
            return
    raise TimeoutError(
        f"policy thread did not stop within {timeout:.3f} seconds"
    )


def _send_shutdown_damping(
    low_ctrl,
    frame_count: int,
    interval: float,
    owner_monitor=None,
) -> None:
    """Publish finite damping only while the empty owner remains fresh.

    Ownership is checked immediately before *every* write.  A conflict,
    unknown result, or stale monitor sample raises before that frame is sent;
    the caller then closes the writer without attempting more damping.
    """

    for _ in range(max(1, int(frame_count))):
        if owner_monitor is not None:
            owner_monitor.require_fresh_empty_owner()
        low_ctrl.set_motor_damping()
        time.sleep(max(0.0, float(interval)))


def _require_policy_write_freshness(
    low_ctrl,
    sensor_captured_at: float,
    deadline_seconds: float,
) -> None:
    """Fail before a policy write if its inference frame missed the deadline."""

    now = time.monotonic()
    age = now - float(sensor_captured_at)
    deadline = float(deadline_seconds)
    if not np.isfinite(age) or not np.isfinite(deadline) or deadline <= 0.0:
        raise RuntimeError("invalid real-time policy deadline state")
    if age > deadline:
        raise TimeoutError(
            f"policy output missed its LowState control deadline: "
            f"{age * 1000.0:.1f} ms > {deadline * 1000.0:.1f} ms"
        )
    # This second read is intentionally after inference and immediately before
    # Write.  It validates that the DDS LowState stream itself is still fresh;
    # the returned newer frame is not silently substituted into the inference.
    low_ctrl.get_sensor_state()


class _AsyncMocapReferenceCache:
    """Poll a potentially blocking reference source outside the control loop."""

    def __init__(self, source, *, poll_interval: float = 0.005) -> None:
        self._source = source
        self._poll_interval = max(0.001, float(poll_interval))
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._latest = None
        self._timestamp = None
        self._error: BaseException | None = RuntimeError(
            "CHINGMU reference cache has not received its first packet"
        )
        self._thread = threading.Thread(
            target=self._run,
            name="chingmu-reference-cache",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                qpos, timestamp = self._source.read()
                qpos = np.asarray(qpos, dtype=np.float32)
                if qpos.shape != (36,) or not np.all(np.isfinite(qpos)):
                    raise ValueError(
                        "CHINGMU reference cache requires one finite 36-value qpos"
                    )
                with self._lock:
                    self._latest = qpos.copy()
                    self._timestamp = float(timestamp)
                    self._error = None
            except BaseException as exc:
                with self._lock:
                    self._error = exc
            self._stop.wait(self._poll_interval)

    def read(self) -> tuple[np.ndarray, float]:
        """Return a copy without performing Redis/network I/O."""

        with self._lock:
            error = self._error
            latest = None if self._latest is None else self._latest.copy()
            timestamp = self._timestamp
        if error is not None:
            raise RuntimeError(
                f"CHINGMU asynchronous reference unavailable: "
                f"{type(error).__name__}: {error}"
            ) from error
        if latest is None or timestamp is None:
            raise RuntimeError("CHINGMU asynchronous reference is not ready")
        return latest, float(timestamp)

    def stop_and_join(self, timeout_seconds: float = 0.5) -> None:
        self._stop.set()
        self._thread.join(max(0.0, float(timeout_seconds)))
        if self._thread.is_alive():
            raise TimeoutError("CHINGMU reference cache thread did not stop")


# ---------------------------------------------------------------------------
# Live reference converter: qpos_full -> ref_state dict for G1TrackInferFn
# ---------------------------------------------------------------------------

def _batch_rot_log_so3(R: np.ndarray) -> np.ndarray:
    """Batch SO(3) log for K rotation matrices. R: (K,3,3) -> (K,3)."""
    tr = np.trace(R, axis1=1, axis2=2)  # (K,)
    cos_theta = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
    theta = np.arccos(cos_theta)  # (K,)
    skew = np.stack([
        R[:, 2, 1] - R[:, 1, 2],
        R[:, 0, 2] - R[:, 2, 0],
        R[:, 1, 0] - R[:, 0, 1],
    ], axis=1)  # (K, 3)
    small = theta < 1e-6
    safe_theta = np.where(small, 1.0, theta)
    k = np.where(small, 0.5, theta / (2.0 * np.sin(safe_theta)))
    return k[:, None] * skew


def _batch_pose_delta_to_twist(T_prev: np.ndarray, T_curr: np.ndarray,
                               dt: float) -> np.ndarray:
    """Batch compute 6D twists (ang, lin) in world. T: (K,4,4) -> (K,6)."""
    R0, p0 = T_prev[:, :3, :3], T_prev[:, :3, 3]
    R1, p1 = T_curr[:, :3, :3], T_curr[:, :3, 3]
    v_w = (p1 - p0) / dt
    R_delta = np.einsum("kij,kjl->kil", R0.transpose(0, 2, 1), R1)
    rotvec_body = _batch_rot_log_so3(R_delta)
    w_w = np.einsum("kij,kj->ki", R0, rotvec_body / dt)
    return np.concatenate([w_w, v_w], axis=1).astype(np.float32)


def _quat_to_yaw(q: np.ndarray) -> float:
    """Extract yaw (z-rotation) from a wxyz quaternion."""
    w, x, y, z = q
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two wxyz quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=np.float32)


def _wrap_to_pi(a: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return float(np.arctan2(np.sin(a), np.cos(a)))


class LiveRefConverter:
    """Convert a raw qpos_full (from mocap) to the ref_state dict expected
    by G1TrackInferFn, computing keypoint poses via MuJoCo FK.

    Keypoint velocities are computed via finite differences between
    consecutive frames, matching the pre-computed kpt_cvel_in_gv used
    in training reference data.

    For real-robot deployment, call ``set_robot_initial_pose()`` when
    entering tracking mode so that the reference root pose is rebiased
    to align with the robot's initial frame (IMU yaw + phantom [0,0,z]).
    """

    def __init__(self, mj_model: mujoco.MjModel, ctrl_dt: float):
        self.mj_model = mj_model
        self.mj_data = mujoco.MjData(mj_model)
        self.ctrl_dt = ctrl_dt
        self.kpt_body_ids = np.array([mj_model.body(n).id for n in KPT_NAMES])
        self._prev_kpt2wrd_pose = None
        self._prev_gv2wrd_pose = None
        self._prev_qpos = None
        # Initial-pose calibration (P3): set via set_robot_initial_pose()
        self._ref_init_xy = None
        self._ref_init_yaw = None
        self._robot_init_xy = None
        self._robot_init_yaw = None

    def reset(self):
        self._prev_kpt2wrd_pose = None
        self._prev_gv2wrd_pose = None
        self._prev_qpos = None
        self._ref_init_xy = None
        self._ref_init_yaw = None
        self._robot_init_xy = None
        self._robot_init_yaw = None

    def set_robot_initial_pose(self, robot_quat: np.ndarray, robot_xy: np.ndarray):
        """Record the robot's pose at the moment tracking begins.
        Called once per tracking session so that reference poses can be
        rebiased into the robot's coordinate frame."""
        self._robot_init_yaw = _quat_to_yaw(robot_quat)
        self._robot_init_xy = robot_xy[:2].copy()

    def _rebias_qpos(self, qpos_full: np.ndarray) -> np.ndarray:
        """Rebias reference root (xy, yaw) so that the first frame aligns
        with the robot's initial pose recorded by set_robot_initial_pose().
        This makes update_coord_cmd() compute correct relative differences
        on the real robot where the phantom state starts at [0,0,0.78]."""
        if self._robot_init_yaw is None:
            return qpos_full

        qpos = qpos_full.copy()

        # Capture reference initial pose on first call
        if self._ref_init_xy is None:
            self._ref_init_yaw = _quat_to_yaw(qpos[3:7])
            self._ref_init_xy = qpos[:2].copy()

        # Compute delta from reference initial
        ref_yaw = _quat_to_yaw(qpos[3:7])
        d_yaw = _wrap_to_pi(ref_yaw - self._ref_init_yaw)
        d_xy = qpos[:2] - self._ref_init_xy

        # Rotate d_xy by the yaw offset between robot and reference initial
        yaw_offset = _wrap_to_pi(self._robot_init_yaw - self._ref_init_yaw)
        c, s = np.cos(yaw_offset), np.sin(yaw_offset)
        rotated_dxy = np.array([c * d_xy[0] - s * d_xy[1],
                                s * d_xy[0] + c * d_xy[1]], dtype=np.float32)

        # Apply to robot initial pose
        qpos[:2] = self._robot_init_xy + rotated_dxy
        new_yaw = self._robot_init_yaw + d_yaw

        # Apply yaw correction in WORLD frame: q_new = q_dz * q_ref.
        # Right-multiplication applies a BODY-frame yaw and distorts roll/pitch
        # when the torso is not upright (e.g. bending/squatting).
        d_yaw_apply = _wrap_to_pi(new_yaw - ref_yaw)
        c2, s2 = np.cos(d_yaw_apply / 2), np.sin(d_yaw_apply / 2)
        q_dz = np.array([c2, 0.0, 0.0, s2], dtype=np.float32)
        q_new = _quat_mul_wxyz(q_dz, qpos[3:7])
        qpos[3:7] = q_new / np.clip(np.linalg.norm(q_new), 1e-8, None)

        return qpos

    def convert(self, qpos_full: np.ndarray) -> dict:
        """Convert a single qpos_full (36,) to ref_state dict (batch dim = 1)."""
        qpos_full = self._rebias_qpos(qpos_full)
        self.mj_data.qpos[:] = qpos_full
        self.mj_data.qvel[:] = 0.0
        mujoco.mj_forward(self.mj_model, self.mj_data)

        # Gravity-view frame
        base2wrd_rot = quat2mat(qpos_full[3:7])
        gvi2wrd_rot = base2navi(base2wrd_rot)
        gvi2wrd_pose = np.eye(4, dtype=np.float32)
        gvi2wrd_pose[:2, 3] = qpos_full[:2]
        gvi2wrd_pose[:3, :3] = gvi2wrd_rot

        # Keypoint poses in world (vectorised)
        num_kpt = len(self.kpt_body_ids)
        kpt2wrd_pose = np.tile(np.eye(4, dtype=np.float32), (num_kpt, 1, 1))
        kpt2wrd_pose[:, :3, 3] = self.mj_data.xpos[self.kpt_body_ids]
        kpt2wrd_pose[:, :3, :3] = self.mj_data.xmat[self.kpt_body_ids].reshape(-1, 3, 3)

        # Transform to gravity-view
        kpt2gv_pose = np.linalg.inv(gvi2wrd_pose) @ kpt2wrd_pose  # (K,4,4)

        # Keypoint velocities via finite differences (vectorised, no scipy)
        kpt_cvel_in_wrd = np.zeros((num_kpt, 6), dtype=np.float32)
        if self._prev_kpt2wrd_pose is not None:
            kpt_cvel_in_wrd = _batch_pose_delta_to_twist(
                self._prev_kpt2wrd_pose, kpt2wrd_pose, self.ctrl_dt
            )
        self._prev_kpt2wrd_pose = kpt2wrd_pose.copy()

        # Rotate world-frame velocities to gravity-view frame
        R_wrd2gv = gvi2wrd_pose[:3, :3].T
        kpt_cvel_in_gv = np.zeros_like(kpt_cvel_in_wrd)
        kpt_cvel_in_gv[:, :3] = kpt_cvel_in_wrd[:, :3] @ R_wrd2gv.T
        kpt_cvel_in_gv[:, 3:] = kpt_cvel_in_wrd[:, 3:] @ R_wrd2gv.T

        # Gravity-view velocity (root frame linear + yaw angular)
        gv_vel = np.zeros(3, dtype=np.float32)
        if self._prev_gv2wrd_pose is not None:
            curr2prev = np.linalg.inv(self._prev_gv2wrd_pose) @ gvi2wrd_pose
            gv_vel[:2] = curr2prev[:2, 3] / self.ctrl_dt
            gv_vel[2] = np.arctan2(curr2prev[1, 0], curr2prev[0, 0]) / self.ctrl_dt
        self._prev_gv2wrd_pose = gvi2wrd_pose.copy()

        # Joint velocity from finite differences
        qvel = np.zeros(35, dtype=np.float32)
        if self._prev_qpos is not None:
            qvel[6:] = (qpos_full[7:] - self._prev_qpos[7:]) / self.ctrl_dt
        self._prev_qpos = qpos_full.copy()

        # Build ref_state dict with batch dimension
        return {
            "qpos": qpos_full[None].astype(np.float32),
            "qvel": qvel[None].astype(np.float32),
            "kpt2gv_pose": kpt2gv_pose[None].astype(np.float32),
            "kpt_cvel_in_gv": kpt_cvel_in_gv[None].astype(np.float32),
            "gv2wrd_pose": gvi2wrd_pose[None].astype(np.float32),
            "gv_vel": gv_vel[None].astype(np.float32),
        }


# ---------------------------------------------------------------------------
# Reference motion loading
# ---------------------------------------------------------------------------

def load_offline_motions(track_dir: str, mj_model: mujoco.MjModel, freq: int = 50) -> list[dict]:
    """Load .npz reference trajectories, apply EMA and convert to kpt format.

    Returns list of dicts. Each dict has numpy-array fields (safe for
    jtu.tree_map) plus a ``_filename`` key that is excluded before tree_map.
    """
    folder = Path(track_dir)
    if folder.is_file():
        files = [folder]
    else:
        files = sorted(folder.glob("*.npz"))

    motions = []
    for f in files:
        data = dict(np.load(f, allow_pickle=True))
        if "qpos" not in data and {"root_pos", "root_rot", "dof_pos"} <= data.keys():
            data["qpos"] = np.concatenate(
                [data["root_pos"], data["root_rot"], data["dof_pos"]], axis=1
            )
        if "qpos" not in data:
            print(f"[WARN] Skipping {f.name}: no qpos field")
            continue

        data["qpos"] = apply_ema_qpos(data["qpos"])
        freq_src = float(data.get("frequency", 50))
        kpt_data = qpos2kpt(
            mj_model, np.float32(data["qpos"]),
            freq_src=freq_src, freq_tgt=freq,
            interp_sec=0.5, end_default_sec=0.5,
            debug=False, foot_contact_est=False,
            height_clip_mode=None, video_path=None,
        )
        motions.append({"data": kpt_data, "filename": f.name})
        print(f"  Mode {len(motions)+1}: {f.name} ({len(kpt_data['qpos'])} frames)")

    return motions


# ---------------------------------------------------------------------------
# Offline tracking metrics (same format as inference.py)
# ---------------------------------------------------------------------------

def _print_offline_metrics(traj_metrics: dict, filename: str, ref_traj: dict, mj_model) -> None:
    """Print tracking error metrics in the same format as scripts/inference.py."""
    actual_traj_len = len(ref_traj["qpos"])
    traj_length_ratio, termination_step = calculate_trajectory_length(
        traj_metrics["state_history"], ref_traj, mj_model
    )
    avg_kpt_pos = np.mean(traj_metrics["kpt_pos_errors"])
    avg_kpt_rot = np.mean(traj_metrics["kpt_rot_errors"])
    avg_joint_pos = np.mean(traj_metrics["joint_pos_errors"])
    avg_joint_vel = np.mean(traj_metrics["joint_vel_errors"])
    avg_root_pos = np.mean(traj_metrics["root_pos_errors"])
    avg_root_vel = np.mean(traj_metrics["root_vel_errors"])
    avg_root_yaw = np.mean(traj_metrics["root_yaw_errors"])
    max_errors = calculate_max_errors(traj_metrics)

    print(f"\n  [Offline Track] {filename} completed:")
    print(f"    Completion: {traj_length_ratio:.4f} ({termination_step}/{actual_traj_len} steps)")
    print(f"    KPT Position MAE: {avg_kpt_pos:.6f} m (Max: {max_errors['max_kpt_pos_error']:.6f} m)")
    print(f"    KPT Rotation MAE: {avg_kpt_rot:.6f} rad (Max: {max_errors['max_kpt_rot_error']:.6f} rad)")
    print(f"    Joint Position MAE: {avg_joint_pos:.6f} rad (Max: {max_errors['max_joint_pos_error']:.6f} rad)")
    print(f"    Joint Velocity MAE: {avg_joint_vel:.6f} rad/s (Max: {max_errors['max_joint_vel_error']:.6f} rad/s)")
    print(f"    Root Pos Error: {avg_root_pos:.3f} mm (Max: {max_errors['max_root_pos_error']:.3f} mm)")
    print(f"    Root Vel Error: {avg_root_vel:.3f} mm/s (Max: {max_errors['max_root_vel_error']:.3f} mm/s)")
    print(f"    Root Yaw Error: {avg_root_yaw:.6f} rad (Max: {max_errors['max_root_yaw_error']:.6f} rad)\n")


# ---------------------------------------------------------------------------
# Simulation loop
# ---------------------------------------------------------------------------

def run_sim(args: "DeployArgs"):
    freq = args.freq
    ctrl_dt = 1.0 / freq
    env_cfg = g1_infer_env_config(ctrl_dt=ctrl_dt)

    # Load ONNX tracking policy
    policy_args = PolicyArgs(
        load_path=args.onnx_track,
        policy_type=args.policy_type,
    )
    track_policy = get_policy_onnx(policy_args)

    # Load walk policy
    walk_policy = WalkPolicy(args.onnx_walk)

    # Load offline reference motions
    convert_model = mujoco.MjModel.from_xml_path(args.convert_xml_path)
    print("Loading offline reference motions...")
    print("  Mode 0: Walk")
    print("  Mode 1: Online retarget")
    ref_motions = load_offline_motions(args.track_dir, convert_model, freq)

    # Keyboard
    keyboard = DeployKeyboardCMD(num_track_ref=len(ref_motions))

    # MuJoCo sim (correct sim_dt=0.001 from Humanoid-GPT)
    init_qpos = consts.DEFAULT_QPOS.copy()
    mj_sim = G1TrackMjSim(init_qpos=init_qpos, headless=False, ctrl_dt=ctrl_dt)
    infer_fn = G1TrackInferFn(env_cfg, mj_sim.mj_model, track_policy, privileged=False)
    state = mj_sim.init_state()
    state = mj_sim.reset(state)

    # Camera view: match inference.py
    if not mj_sim.headless and mj_sim.viewer is not None:
        viewer_cam = mj_sim.viewer.cam
        viewer_cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer_cam.trackbodyid = 0
        viewer_cam.azimuth = 90.0
        viewer_cam.elevation = -20.0
        viewer_cam.distance = 2.0

    # Live retarget converter
    live_converter = LiveRefConverter(mj_sim.mj_model, ctrl_dt)

    # Online retarget (optional)
    mocap_buffer = None
    buf_hand = None
    if not args.no_mocap:
        try:
            from src.motion.retarget import start_realtime_retarget
            buf_mocap, ts_mocap, buf_hand = start_realtime_retarget(
                robot="unitree_g1",
                dof_full=7 + 29,
                actual_human_height=args.human_height,
                visualize_retarget=args.visualize_retarget,
                mocap_type=args.mocap_type,
                buffer_ms=args.buffer_ms,
                xsens_host=args.xsens_host,
                xsens_port=args.xsens_port,
                xsens_protocol=args.xsens_protocol,
            )
            mocap_buffer = MocapBuffer(buf_mocap, ts_mocap)
            print("[Mocap] Retarget subprocess started.")
        except Exception as e:
            print(f"[Mocap] Failed to start retarget: {e}. Online mode disabled.")

    rate = RateLimiter(frequency=freq, warn=False)
    last_mode = 0
    track_step = 0
    ref_traj = None
    traj_metrics = None  # For offline tracking: kpt/joint/root errors, state_history
    traj_filename = None
    prev_online_ref = None

    print("\n=== Simulation ready. Press number keys to switch modes. ===\n")

    try:
        while True:
            if keyboard.check_reset_request():
                state = mj_sim.reset(state)
                infer_fn.info["last_action"][:] = 0
                infer_fn.info["step"] = 0
                live_converter.reset()
                print("[Reset] Simulation reset.")

            cmd = keyboard.step_command()
            mode = cmd.mode

            # Mode transitions
            entering_track = (last_mode == 0) and (mode >= 1)
            leaving_track = (last_mode >= 1) and (mode == 0)

            if entering_track:
                infer_fn.info["last_action"][:] = 0
                live_converter.reset()
                prev_online_ref = None
                if mode >= 2:
                    traj_idx = mode - 2
                    if traj_idx < len(ref_motions):
                        ref_traj = ref_motions[traj_idx]["data"]
                        track_step = 0
                        traj_filename = ref_motions[traj_idx]["filename"]
                        traj_metrics = {
                            "kpt_pos_errors": [],
                            "kpt_rot_errors": [],
                            "joint_pos_errors": [],
                            "joint_vel_errors": [],
                            "root_pos_errors": [],
                            "root_vel_errors": [],
                            "root_yaw_errors": [],
                            "state_history": [],
                        }
                        print(f"[Track] Start offline: {traj_filename}")
                    else:
                        print(f"[Track] Invalid trajectory index {traj_idx}")
                        mode = 0
                elif mode == 1:
                    print("[Track] Start online retarget")

            if leaving_track:
                if last_mode >= 2 and traj_metrics is not None and len(traj_metrics["state_history"]) > 0:
                    _print_offline_metrics(traj_metrics, traj_filename, ref_traj, mj_sim.mj_model)
                traj_metrics = None
                traj_filename = None
                live_converter.reset()
                print("[Track] Back to walk mode")

            if mode == 0:
                # Walk policy
                cmd_vel = np.array([cmd.vel_lin_x, cmd.vel_lin_y, cmd.vel_ang_yaw], dtype=np.float32)
                gyro = mj_sensor(mj_sim.mj_model, state.mj_data, "gyro_pelvis")
                motor_targets = walk_policy.infer(
                    state.mj_data.qpos[3:7], gyro,
                    state.mj_data.qpos[7:], state.mj_data.qvel[6:],
                    cmd_vel,
                )
                # Walk uses different PD gains than tracking
                for _ in range(mj_sim.num_sim_substeps):
                    torques = KPs_walking * (motor_targets - state.mj_data.qpos[7:]) + KDs_walking * (-state.mj_data.qvel[6:])
                    torques = np.clip(torques, -consts.TORQUE_LIMIT, consts.TORQUE_LIMIT)
                    state.mj_data.ctrl[:] = torques
                    mujoco.mj_step(mj_sim.mj_model, state.mj_data)

            elif mode == 1:
                # Online retarget
                if mocap_buffer is not None:
                    qpos_full, _ = mocap_buffer.read()
                    ref_new = live_converter.convert(qpos_full)
                    if prev_online_ref is None:
                        ref_curr = ref_new
                    else:
                        ref_curr = prev_online_ref
                    ref_next = ref_new
                    prev_online_ref = ref_new
                    motor_targets = infer_fn.infer_onnx(
                        state, {"ref_curr": ref_curr, "ref_next": ref_next}
                    )
                    state = mj_sim.step(state, motor_targets)

            else:
                # Offline tracking (mode >= 2)
                if ref_traj is not None:
                    traj_len = len(ref_traj["qpos"])
                    ref_curr = jtu.tree_map(lambda x: x[track_step][None], ref_traj)
                    next_step = min(track_step + 1, traj_len - 1)
                    ref_next = jtu.tree_map(lambda x: x[next_step][None], ref_traj)
                    motor_targets = infer_fn.infer_onnx(
                        state, {"ref_curr": ref_curr, "ref_next": ref_next}
                    )
                    state = mj_sim.step(state, motor_targets)

                    # Collect metrics (same as inference.py)
                    if traj_metrics is not None:
                        kpt_pos_mae, kpt_rot_mae = calculate_kpt_mae_error(
                            state, ref_curr, ref_next, mj_sim.mj_model
                        )
                        joint_pos_mae, joint_vel_mae = calculate_joint_tracking_error(
                            state, ref_curr
                        )
                        root_pos_err_mm, root_vel_err_mms, root_yaw_err = calculate_root_tracking_error(
                            state, ref_curr
                        )
                        traj_metrics["kpt_pos_errors"].append(kpt_pos_mae)
                        traj_metrics["kpt_rot_errors"].append(kpt_rot_mae)
                        traj_metrics["joint_pos_errors"].append(joint_pos_mae)
                        traj_metrics["joint_vel_errors"].append(joint_vel_mae)
                        traj_metrics["root_pos_errors"].append(root_pos_err_mm)
                        traj_metrics["root_vel_errors"].append(root_vel_err_mms)
                        traj_metrics["root_yaw_errors"].append(root_yaw_err)
                        traj_metrics["state_history"].append({
                            "qpos": state.mj_data.qpos.copy(),
                            "qvel": state.mj_data.qvel.copy(),
                            "xpos": state.mj_data.xpos.copy(),
                            "xmat": state.mj_data.xmat.copy(),
                        })

                    track_step = track_step + 1

                    # Print metrics when trajectory completes (after processing last frame)
                    if track_step >= traj_len and traj_metrics is not None and len(traj_metrics["state_history"]) > 0:
                        _print_offline_metrics(traj_metrics, traj_filename, ref_traj, mj_sim.mj_model)
                        traj_metrics = None
                        traj_filename = None
                        ref_traj = None  # Avoid index out of bounds on next iteration

            last_mode = mode
            mj_sim.view(state)
            rate.sleep()

            if cmd.kill:
                break

    except KeyboardInterrupt:
        pass
    finally:
        keyboard.close()
        print("[Sim] Exited.")


# ---------------------------------------------------------------------------
# Real-robot loop
# ---------------------------------------------------------------------------

def _save_measured_official_standup_trace(
    samples,
    *,
    completion,
    robot_variant: str,
    mode_machine: int,
    firmware_id: str,
    output_path: str,
) -> Path:
    """Persist the LowState side channel from one acknowledged FSM 706 run.

    This helper never sends a command.  It is called only after
    ``run_official_squat_to_stand`` has observed motion and a stable endpoint,
    and before the high-level ``ai`` mode is released.  Consequently a
    disk/schema failure leaves LowCmd gated and the firmware controller in
    charge.
    """

    from src.motion.official_standup_trace import OfficialStandupTrace

    if not samples:
        raise ValueError("official stand-up trace contains no LowState samples")
    timestamps = np.asarray(
        [sample.monotonic_time for sample in samples], dtype=np.float64
    )
    timestamps -= timestamps[0]
    joint_qpos = np.stack([sample.joint_qpos for sample in samples]).astype(
        np.float32
    )
    joint_qvel = np.stack([sample.joint_qvel for sample in samples]).astype(
        np.float32
    )
    root_quat = np.stack([sample.root_quat_wxyz for sample in samples]).astype(
        np.float32
    )
    root_gyro = np.stack([sample.root_gyro for sample in samples]).astype(
        np.float32
    )

    # The public high-level API does not attach an FSM value to each LowState
    # packet.  Mark the samples before measured motion as Damp (1) and the
    # remainder with the FSM ID that the completion detector actually polled.
    displacement = np.max(np.abs(joint_qpos - joint_qpos[0]), axis=1)
    speed = np.max(np.abs(joint_qvel), axis=1)
    moving = np.flatnonzero((displacement >= 0.05) | (speed >= 0.10))
    if moving.size == 0:
        raise ValueError("official stand-up trace contains no measured motion")
    motion_index = int(moving[0])
    observed_fsm = int(completion.observed_fsm_id)
    fsm_id = np.full(timestamps.shape, 1, dtype=np.int32)
    fsm_id[motion_index:] = observed_fsm

    normalized_firmware_id = str(firmware_id).strip()
    if not normalized_firmware_id:
        normalized_firmware_id = f"unreported-mode-machine-{int(mode_machine)}"
    trace = OfficialStandupTrace(
        robot_variant=str(robot_variant),
        mode_machine=int(mode_machine),
        firmware_id=normalized_firmware_id,
        timestamps=timestamps,
        joint_qpos=joint_qpos,
        joint_qvel=joint_qvel,
        root_quat_wxyz=root_quat,
        root_gyro=root_gyro,
        fsm_id=fsm_id,
        metadata={
            "capture_source": "Unitree rt/lowstate",
            "official_action": "Damp -> SetFsmId(706)",
            "fsm_per_sample": "damp_until_measured_motion_then_polled_completion_fsm",
            "completion_elapsed_seconds": float(completion.elapsed_seconds),
            "capture_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "joint_order": list(consts.MotorName.FULL),
        },
    )
    destination = Path(output_path).expanduser()
    trace.save(destination)
    return destination

def run_real(args: "DeployArgs"):
    """Install cancellation for the complete real-robot lifecycle."""
    if not args.debug:
        if args.mocap_type.strip().lower() != "chingmu_redis":
            raise RuntimeError(
                "Non-debug real output is restricted to the owner-aware "
                "CHINGMU Redis path; use --mocap-type chingmu_redis"
            )
        if not args.startup_handover:
            raise RuntimeError(
                "Non-debug real output requires the remote-only owner handover"
            )
        project_root = Path(__file__).resolve().parents[2]
        project_root_text = str(project_root)
        if project_root_text not in sys.path:
            sys.path.insert(0, project_root_text)
        from src.application.authorization import require_runner_authorization

        # Consume the final one-shot capability before any Unitree SDK2 import,
        # ChannelFactoryInitialize call, or LowCmd publisher construction.
        require_runner_authorization(
            backend="humanoid_gpt",
            network_interface=args.net,
            model_profile=args.model_profile,
            redis_host=args.redis_host,
            redis_port=args.redis_port,
            redis_key=args.redis_key,
        )
    stop_event = threading.Event()
    interrupt_event = threading.Event()

    def _sigint(*_):
        interrupt_event.set()
        stop_event.set()

    previous_sigint = signal.signal(signal.SIGINT, _sigint)
    try:
        return _run_real_with_cancel(args, stop_event, interrupt_event)
    finally:
        signal.signal(signal.SIGINT, previous_sigint)


def _run_real_with_cancel(
    args: "DeployArgs",
    stop_event: threading.Event,
    interrupt_event: threading.Event | None = None,
):
    if os.name == "nt":
        from src.adapters.unitree import configure_windows_unitree_interface

        args.net = configure_windows_unitree_interface(args.net)
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from src.adapters.unitree_thread import get_recurrent_thread_class
    from src.motion.real_robot import LowLevelControlG1, KeyMap
    from src.motion.hand_control import Dex3Controller, update_hand_from_mocap, HAND_POSES
    from src.motion.retarget import (
        start_realtime_retarget,
        read_hand_buffer,
    )

    RecurrentThread = get_recurrent_thread_class()

    freq = args.freq
    ctrl_dt = 1.0 / freq
    env_cfg = g1_infer_env_config(ctrl_dt = ctrl_dt)
    use_chingmu_redis = args.mocap_type.strip().lower() == "chingmu_redis"
    # CHINGMU real output always uses the ownership-safe remote handover.  The
    # historical CLI flag is retained only for package compatibility; allowing
    # ``--no-startup-handover`` here would create a LowCmd writer while the
    # official ``ai`` controller may still own the robot.
    use_remote_handover = use_chingmu_redis and not args.debug
    if use_remote_handover and not args.startup_handover:
        print(
            "[Real] --no-startup-handover is ignored for CHINGMU real output; "
            "the receive-only LowState and physical START gates are mandatory."
        )

    # Select the robot-facing DDS interface at the real business entry.  The
    # actual Develop credential is a bounded wait for fresh LowState below;
    # this firmware may stop its MotionSwitcher RPC service in Develop mode.
    ChannelFactoryInitialize(0, args.net)
    motion_switcher = None

    # Linux/Jetson real deployment requires TensorRT.  Gemini Native uses the
    # packaged Windows ONNX Runtime CPU provider; the same released policy and
    # feedback loop remain in use.
    policy_args = PolicyArgs(
        load_path=args.onnx_track,
        policy_type=args.policy_type,
        device="cpu" if os.name == "nt" else "cuda:0",
    )
    track_policy = get_policy_onnx(
        policy_args,
        use_trt=os.name != "nt",
        strict_trt=os.name != "nt",
    )
    walk_policy = WalkPolicy(args.onnx_walk)

    # Construct the complete released-policy runtime before touching Unitree
    # DDS.  With no robot attached the reference cache below therefore keeps
    # consuming GMR output while the initialized policy waits for feedback; no
    # closed-loop inference or LowCmd is possible without a fresh LowState.
    xml_path = str(consts.ROOT_PATH / "scene_mjx_track.xml")
    phantom_model = mujoco.MjModel.from_xml_path(xml_path)
    phantom_model.opt.timestep = 0.001
    infer_fn = G1TrackInferFn(
        env_cfg, phantom_model, track_policy, privileged=False
    )
    live_converter = LiveRefConverter(phantom_model, ctrl_dt)

    # Offline motions
    convert_model = mujoco.MjModel.from_xml_path(args.convert_xml_path)
    print("Loading offline reference motions...")
    print("  Mode 0: Walk")
    print("  Mode 1: Online retarget")
    ref_motions = load_offline_motions(args.track_dir, convert_model, freq)

    # Pre-init pygame then quit video so the fork in KeyboardCmdPad
    # does not inherit an X11 fd (avoids "X connection broken").
    pygame.init()
    pygame.display.quit()

    keyboard = DeployKeyboardCMD(num_track_ref=len(ref_motions))
    mocap_buffer = None
    mocap_reference_cache = None
    buf_hand = None
    if use_chingmu_redis:
        # CHINGMU already supplies the online G1 reference, so mode 1 is the
        # useful default.  The physical A/B buttons own live/stand selection in
        # the remote-only handover; the keyboard remains a diagnostic fallback.
        keyboard.mode = 1
        # Attach to the already-running GMR reference before waiting for a
        # robot.  The launcher owns the CHINGMU -> GMR producer processes, so
        # this consumer and the policy can remain warm with no G1 connected.
        mocap_buffer = ChingMuRedisMocapBuffer(
            host=args.redis_host,
            port=args.redis_port,
            key=args.redis_key,
            require_integrity=True,
        )
        mocap_reference_cache = _AsyncMocapReferenceCache(
            mocap_buffer,
            poll_interval=min(ctrl_dt, 0.005),
        )
        mocap_reference_cache.start()
        mocap_buffer = mocap_reference_cache
        print(
            f"[Mocap] CHINGMU Redis reference attached before LowState: "
            f"{args.redis_host}:{args.redis_port}/{args.redis_key}"
        )

    # Robot + phantom sim for reference FK only (no robot-side FK needed)
    if use_remote_handover:
        print(
            "<Mode: Reference hot> The CHINGMU -> GMR -> Redis producer and "
            "released policy are loaded before Unitree LowState. They continue "
            "while the robot is absent; no LowCmd writer exists in this wait."
        )
    try:
        low_ctrl = LowLevelControlG1(
            ctrl_dt=ctrl_dt,
            debug=args.debug,
            defer_publisher=use_remote_handover,
            lowstate_timeout_seconds=args.lowstate_timeout_seconds,
            should_cancel=stop_event.is_set,
        )
    except InterruptedError:
        if mocap_reference_cache is not None:
            mocap_reference_cache.stop_and_join()
        keyboard.close()
        print("[Real] Cancelled while waiting for LowState.")
        return
    except Exception:
        if mocap_reference_cache is not None:
            mocap_reference_cache.stop_and_join()
        keyboard.close()
        raise

    if use_chingmu_redis:
        print(
            "<Mode: Develop detected> Fresh, version-matched LowState is "
            "arriving. LowCmd remains closed."
        )
    if use_remote_handover:
        # MotionSwitcher is an optional secondary observation.  Some G1
        # firmware disables this RPC endpoint in physical Develop mode even
        # though rt/lowstate is healthy.  Never reject that valid state solely
        # because CheckMode is unavailable; if it does answer, a non-empty
        # owner remains a hard conflict.
        try:
            candidate_switcher = create_motion_switcher_client()
            optional_owner = checked_motion_owner(
                candidate_switcher,
                context="after fresh Develop LowState was validated",
            )
        except Exception as exc:
            print(
                "[Real] MotionSwitcher CheckMode is unavailable in Develop "
                f"({type(exc).__name__}: {exc}); using fresh LowState as the "
                "runtime Develop lease. LowCmd remains closed."
            )
        else:
            if optional_owner != "":
                raise RemoteHandoverError(
                    f"fresh LowState arrived but MotionSwitcher reports active "
                    f"owner {optional_owner!r}; refusing LowCmd"
                )
            motion_switcher = candidate_switcher
            print(
                "[Real] Optional MotionSwitcher check also confirmed an empty "
                "owner."
            )

    # Online reference.  CHINGMU/GMR has already performed the retargeting and
    # publishes a 36-value G1 qpos, so it replaces only the mocap buffer.  The
    # official real LowState -> infer_onnx_real -> LowCmd path below is shared.
    if not use_chingmu_redis:
        buf_mocap, ts_mocap, buf_hand = start_realtime_retarget(
            robot="unitree_g1", dof_full=7 + 29,
            actual_human_height=args.human_height,
            visualize_retarget=args.visualize_retarget,
            mocap_type=args.mocap_type,
            buffer_ms=args.buffer_ms,
            xsens_host=args.xsens_host,
            xsens_port=args.xsens_port,
            xsens_protocol=args.xsens_protocol,
        )
        mocap_buffer = MocapBuffer(buf_mocap, ts_mocap)

    # Hand controller
    hand_ctrl = None
    if args.enable_hand and use_chingmu_redis:
        print("[Hand] Disabled for CHINGMU Redis reference mode.")
    elif args.enable_hand:
        try:
            hand_ctrl = Dex3Controller(net=args.net, re_init=False)
        except Exception as e:
            print(f"[Hand] Failed to init: {e}")

    last_mode = 0
    track_step = 0
    ref_traj = None
    last_left_hand = None
    last_right_hand = None
    prev_online_ref = None
    remote_handover = None
    remote_reference_anchor = None
    remote_last_live_reference = None
    remote_reference_stream_failed = False
    remote_last_output_reference = None
    remote_transition_from_reference = None
    remote_expected_first_target = None
    remote_first_policy_target_pending = False
    owner_monitor = None
    remote_button_edges = SafeRemoteButtonEdges(
        (KeyMap.A, KeyMap.B, KeyMap.select),
        (KeyMap.L1, KeyMap.R1, KeyMap.L2, KeyMap.R2),
        failsafe_chord=(KeyMap.L2, KeyMap.B),
    )
    remote_exit_requested = False
    remote_policy_stand_ready = False
    remote_policy_dwell = StableStandDwell()
    remote_last_phase = None

    def cancel_requested() -> bool:
        # Before LowCmd takeover, only the web/terminal cancellation path is
        # active.  Official firmware may consume remote buttons while it owns
        # the robot, so interpreting SELECT here could create a false edge.
        return stop_event.is_set()

    def locomotion_step(
        startup_live_weight: float = 1.0,
        startup_reference_anchor: np.ndarray | None = None,
        publish_motor_target: bool = True,
    ):
        nonlocal last_mode, track_step, ref_traj, last_left_hand, last_right_hand
        nonlocal prev_online_ref, remote_exit_requested
        nonlocal remote_policy_stand_ready, remote_last_phase
        nonlocal remote_reference_anchor, remote_last_live_reference
        nonlocal remote_reference_stream_failed
        nonlocal remote_last_output_reference, remote_transition_from_reference
        nonlocal remote_expected_first_target, remote_first_policy_target_pending

        if stop_event.is_set():
            return

        if owner_monitor is not None:
            # This is a lock/timestamp read only.  When the optional
            # MotionSwitcher RPC is available, its potentially blocking call
            # runs in MotionOwnerMonitor's independent thread.
            owner_monitor.require_fresh_empty_owner()

        (
            root_quat,
            root_gyro,
            jnt_qpos,
            jnt_qvel,
            sensor_captured_at,
            remote_buttons,
        ) = low_ctrl.get_sensor_state_with_timestamp_and_buttons()
        control_deadline_seconds = max(0.050, 2.5 * ctrl_dt)
        cmd = keyboard.step_command()
        mode = cmd.mode

        if use_remote_handover and remote_handover is not None:
            # In the remote-only path one feedback tracking policy owns the
            # whole handover.  A/B change only its reference weight, so there
            # is no discontinuous switch to a different motor policy.
            mode = 1
            button_events = remote_button_edges.update(remote_buttons)
            if button_events.failsafe_stop:
                remote_exit_requested = True
                print(
                    "<Mode: LowCmd exit requested> A fresh released-then-pressed "
                    "L2+B chord was observed during LowCmd control. Stopping the "
                    "policy and entering the bounded writer-close cleanup. This "
                    "software action is not a physical emergency stop."
                )
                stop_event.set()
                return
            events = button_events.actions
            if KeyMap.A in events:
                if not remote_policy_stand_ready:
                    print(
                        "[Real] A ignored: the feedback policy has not completed "
                        "its stable standing dwell."
                    )
                else:
                    # A must never advance the handover on the strength of a
                    # previously cached pose.  Verify the asynchronous cache
                    # first; it rejects packets older than 120 ms and preserves
                    # the policy-stand phase when CHINGMU is interrupted.
                    try:
                        fresh_reference, _ = mocap_buffer.read()
                    except Exception as exc:
                        remote_reference_stream_failed = True
                        print(
                            "[Real] A ignored: no fresh CHINGMU reference is "
                            f"available ({type(exc).__name__}: {exc}); the "
                            "feedback-policy stand remains active."
                        )
                    else:
                        remote_last_live_reference = np.asarray(
                            fresh_reference, dtype=np.float32
                        ).copy()
                        remote_reference_stream_failed = False
                        if remote_handover.command(RemoteHandoverCommand.LIVE):
                            remote_transition_from_reference = np.asarray(
                                remote_last_output_reference
                                if remote_last_output_reference is not None
                                else remote_reference_anchor,
                                dtype=np.float32,
                            ).copy()
                            print(
                                "<Mode: CHINGMU teleoperation> A accepted after "
                                "a fresh reference check; blending the frozen "
                                "policy stand into the live CHINGMU reference."
                            )
            if KeyMap.B in events:
                if remote_handover.command(RemoteHandoverCommand.STAND):
                    live_template = (
                        remote_last_live_reference
                        if remote_last_live_reference is not None
                        else remote_reference_anchor
                    )
                    current_output = np.asarray(
                        remote_last_output_reference
                        if remote_last_output_reference is not None
                        else live_template,
                        dtype=np.float32,
                    ).copy()
                    remote_transition_from_reference = current_output
                    remote_reference_anchor = build_measured_stand_anchor(
                        live_template, root_quat, jnt_qpos
                    )
                    remote_policy_stand_ready = False
                    remote_policy_dwell.reset()
                    print(
                        "<Mode: Policy stand> B accepted; freezing a standing "
                        "reference at the current location and blending back to it."
                    )
            if KeyMap.select in events:
                remote_exit_requested = True
                remote_policy_stand_ready = False
                remote_policy_dwell.reset()
                if remote_handover.command(RemoteHandoverCommand.STAND):
                    live_template = (
                        remote_last_live_reference
                        if remote_last_live_reference is not None
                        else remote_reference_anchor
                    )
                    current_output = np.asarray(
                        remote_last_output_reference
                        if remote_last_output_reference is not None
                        else live_template,
                        dtype=np.float32,
                    ).copy()
                    remote_transition_from_reference = current_output
                    remote_reference_anchor = build_measured_stand_anchor(
                        live_template, root_quat, jnt_qpos
                    )
                print(
                    "<Mode: Return stand> SELECT accepted; returning to the "
                    "feedback-policy stand before closing LowCmd."
                )

            remote_sample = remote_handover.advance(ctrl_dt)
            if remote_sample.phase != remote_last_phase:
                remote_last_phase = remote_sample.phase
                if remote_sample.phase in {
                    RemoteHandoverPhase.WAIT_FOR_LIVE,
                    RemoteHandoverPhase.STAND_HOLD,
                }:
                    print(
                        "<Mode: Policy stand> Frozen standing reference active; "
                        "waiting for a released-then-pressed A."
                    )

        entering_track = (last_mode == 0) and (mode >= 1)
        leaving_track = (last_mode >= 1) and (mode == 0)

        if entering_track:
            if not use_remote_handover:
                infer_fn.info["last_action"][:] = 0
                live_converter.reset()
                prev_online_ref = None
                # Calibrate reference frame to robot's current pose
                robot_xy = np.array([0.0, 0.0], dtype=np.float32)
                live_converter.set_robot_initial_pose(root_quat, robot_xy)
            if mode >= 2:
                traj_idx = mode - 2
                if traj_idx < len(ref_motions):
                    ref_traj = ref_motions[traj_idx]["data"]
                    track_step = 0

        if leaving_track:
            live_converter.reset()

        if mode == 0:
            cmd_vel = np.array([cmd.vel_lin_x, cmd.vel_lin_y, cmd.vel_ang_yaw], dtype=np.float32)
            motor_targets = walk_policy.infer(root_quat, root_gyro, jnt_qpos, jnt_qvel, cmd_vel)
            if stop_event.is_set():
                return
            if publish_motor_target:
                _require_policy_write_freshness(
                    low_ctrl, sensor_captured_at, control_deadline_seconds
                )
                if stop_event.is_set():
                    return
                low_ctrl.step(motor_targets, KPs_walking, KDs_walking)
        else:
            if mode == 1:
                if (
                    use_remote_handover
                    and remote_handover is not None
                    and remote_handover.sample().phase in {
                        RemoteHandoverPhase.STABILIZING_STAND,
                        RemoteHandoverPhase.WAIT_FOR_LIVE,
                        RemoteHandoverPhase.BLEND_TO_STAND,
                        RemoteHandoverPhase.STAND_HOLD,
                    }
                ):
                    # Policy stand is fully closed-loop on LowState and the
                    # frozen anchor.  A Redis outage cannot terminate balance.
                    qpos_full = np.asarray(
                        remote_reference_anchor, dtype=np.float32
                    ).copy()
                else:
                    try:
                        qpos_full, _ = mocap_buffer.read()
                        if use_remote_handover and remote_handover is not None:
                            remote_last_live_reference = np.asarray(
                                qpos_full, dtype=np.float32
                            ).copy()
                            if remote_reference_stream_failed:
                                print(
                                    "[Mocap] CHINGMU reference recovered; remain "
                                    "in policy stand until A is pressed again."
                                )
                            remote_reference_stream_failed = False
                    except BaseException as exc:
                        if not use_remote_handover or remote_handover is None:
                            raise
                        live_template = (
                            remote_last_live_reference
                            if remote_last_live_reference is not None
                            else remote_reference_anchor
                        )
                        current_output = np.asarray(
                            remote_last_output_reference
                            if remote_last_output_reference is not None
                            else live_template,
                            dtype=np.float32,
                        ).copy()
                        remote_transition_from_reference = current_output
                        remote_reference_anchor = build_measured_stand_anchor(
                            live_template, root_quat, jnt_qpos
                        )
                        remote_handover.command(RemoteHandoverCommand.STAND)
                        remote_policy_stand_ready = False
                        remote_policy_dwell.reset()
                        qpos_full = current_output.copy()
                        if not remote_reference_stream_failed:
                            print(
                                "[Mocap] Live CHINGMU reference unavailable "
                                f"({type(exc).__name__}: {exc}); freezing the last "
                                "valid reference and blending back to policy stand."
                            )
                        remote_reference_stream_failed = True
                if use_remote_handover and remote_handover is not None:
                    # Every direction change is rebased from the last reference
                    # actually delivered to the converter.  This prevents B,
                    # SELECT, or a source dropout halfway through a blend from
                    # jumping to a newly constructed stand endpoint.
                    sample_now = remote_handover.sample()
                    if sample_now.phase in {
                        RemoteHandoverPhase.BLEND_TO_LIVE,
                        RemoteHandoverPhase.BLEND_TO_STAND,
                    }:
                        transition_from = np.asarray(
                            remote_transition_from_reference
                            if remote_transition_from_reference is not None
                            else remote_reference_anchor,
                            dtype=np.float32,
                        )
                        transition_target = (
                            qpos_full
                            if sample_now.phase == RemoteHandoverPhase.BLEND_TO_LIVE
                            else remote_reference_anchor
                        )
                        qpos_full = blend_floating_base_qpos(
                            transition_from,
                            transition_target,
                            smootherstep01(sample_now.phase_progress),
                        )
                    elif sample_now.phase in {
                        RemoteHandoverPhase.STABILIZING_STAND,
                        RemoteHandoverPhase.WAIT_FOR_LIVE,
                        RemoteHandoverPhase.STAND_HOLD,
                    }:
                        qpos_full = np.asarray(
                            remote_reference_anchor, dtype=np.float32
                        ).copy()
                        remote_transition_from_reference = None
                    else:
                        remote_transition_from_reference = None
                    remote_last_output_reference = np.asarray(
                        qpos_full, dtype=np.float32
                    ).copy()
                elif startup_reference_anchor is not None:
                    qpos_full = blend_floating_base_qpos(
                        startup_reference_anchor,
                        qpos_full,
                        startup_live_weight,
                    )
                ref_new = live_converter.convert(qpos_full)
                if prev_online_ref is None:
                    ref_curr = ref_new
                else:
                    ref_curr = prev_online_ref
                ref_next = ref_new
                prev_online_ref = ref_new
            else:
                if ref_traj is not None:
                    traj_len = len(ref_traj["qpos"])
                    ref_curr = jtu.tree_map(lambda x: x[track_step][None], ref_traj)
                    next_step = min(track_step + 1, traj_len - 1)
                    ref_next = jtu.tree_map(lambda x: x[next_step][None], ref_traj)
                    track_step = min(track_step + 1, traj_len - 1)
                else:
                    last_mode = mode
                    return

            motor_targets = np.asarray(infer_fn.infer_onnx_real(
                root_quat, root_gyro, jnt_qpos, jnt_qvel,
                {"ref_curr": ref_curr, "ref_next": ref_next},
            )).flatten()
            if stop_event.is_set():
                return
            if publish_motor_target:
                # Startup blends only the stationary robot reference into the
                # live CHINGMU reference.  The released tracking policy retains
                # full balance authority; its motor output is never attenuated.
                if owner_monitor is not None:
                    owner_monitor.require_fresh_empty_owner()
                    if remote_first_policy_target_pending:
                        validate_policy_stand_target(jnt_qpos, motor_targets)
                        validate_shadow_target_match(
                            remote_expected_first_target, motor_targets
                        )
                        remote_first_policy_target_pending = False
                _require_policy_write_freshness(
                    low_ctrl, sensor_captured_at, control_deadline_seconds
                )
                if stop_event.is_set():
                    return
                if owner_monitor is not None:
                    # Keep the nonblocking owner read as the final gate closest
                    # to DDS Write; never put MotionSwitcher RPC in this thread.
                    owner_monitor.require_fresh_empty_owner()
                if stop_event.is_set():
                    return
                low_ctrl.step(motor_targets, consts.KPs, consts.KDs)

                if use_remote_handover and remote_handover is not None:
                    sample = remote_handover.sample()
                    if (
                        sample.live_weight == 0.0
                        and sample.phase in {
                            RemoteHandoverPhase.WAIT_FOR_LIVE,
                            RemoteHandoverPhase.STAND_HOLD,
                        }
                    ):
                        remote_policy_stand_ready = remote_policy_dwell.update(
                            time.monotonic(),
                            root_quat,
                            root_gyro,
                            jnt_qpos,
                            jnt_qvel,
                        )
                        if remote_policy_stand_ready and remote_exit_requested:
                            print(
                                "<Mode: Policy stand stable> Feedback-policy "
                                "stand is stable; stopping the control thread."
                            )
                            stop_event.set()
                    else:
                        remote_policy_stand_ready = False
                        remote_policy_dwell.reset()

            if publish_motor_target and hand_ctrl is not None:
                hand_cmd = read_hand_buffer(buf_hand)
                last_left_hand, last_right_hand = update_hand_from_mocap(
                    hand_ctrl, hand_cmd, last_left_hand, last_right_hand,
                )

        last_mode = mode

    thread_errors = []

    def guarded_locomotion_step():
        try:
            locomotion_step()
        except BaseException as exc:
            thread_errors.append(exc)
            stop_event.set()

    # Startup, locomotion, and shutdown share one cancellation event.  The
    # control thread is joined before the final damping burst, preventing a
    # late policy frame from overwriting the shutdown LowCmd.
    loco_thread = None
    damping_frames = max(1, int(round(0.2 / ctrl_dt)))
    clean_remote_handback = False

    def build_measured_stand_anchor(
        reference_template,
        root_quat,
        joint_qpos,
    ) -> np.ndarray:
        """Freeze a stand at the current reference XY/yaw and robot posture."""

        anchor = np.asarray(reference_template, dtype=np.float32).copy()
        if anchor.shape != (36,) or not np.all(np.isfinite(anchor)):
            raise RemoteHandoverError(
                f"CHINGMU standing anchor must be one finite 36-value qpos, "
                f"received {anchor.shape}"
            )
        measured_quat = np.asarray(root_quat, dtype=np.float32).copy()
        measured_quat /= np.clip(np.linalg.norm(measured_quat), 1e-8, None)

        # Keep CHINGMU XY/yaw as the converter's relative-motion zero point.
        # Replacing it with world zero would make the first live frame look like
        # a large translation/yaw jump.  Only the robot's measured roll/pitch is
        # transplanted into that reference yaw; rebiasing then maps it back to
        # the measured robot orientation exactly.
        reference_yaw = _quat_to_yaw(anchor[3:7])
        measured_yaw = _quat_to_yaw(measured_quat)
        yaw_delta = _wrap_to_pi(reference_yaw - measured_yaw)
        half = yaw_delta * 0.5
        yaw_quat = np.asarray(
            [np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float32
        )
        anchor[2] = float(consts.DEFAULT_QPOS[2])
        anchor[3:7] = _quat_mul_wxyz(yaw_quat, measured_quat)
        anchor[3:7] /= np.clip(np.linalg.norm(anchor[3:7]), 1e-8, None)
        anchor[7:] = np.asarray(joint_qpos, dtype=np.float32)
        return anchor

    def seed_policy_history_from_measured(joint_qpos) -> np.ndarray:
        """Invert ``nn2motor_action`` so the observation starts at measured q."""

        measured = np.asarray(joint_qpos, dtype=np.float32)
        if measured.shape != (29,) or not np.all(np.isfinite(measured)):
            raise RemoteHandoverError(
                "cannot seed policy history from an invalid measured joint pose"
            )
        action_ids = np.asarray(infer_fn.ctrl_id_act, dtype=np.intp)
        denominator = (
            float(infer_fn.env_config.action_scale)
            * np.asarray(infer_fn.act_scale, dtype=np.float32)[action_ids]
        )
        if np.any(np.abs(denominator) < 1e-8):
            raise RemoteHandoverError("policy action scale contains zero")
        nominal = np.asarray(
            infer_fn._nom_jnt_qpos[0, action_ids], dtype=np.float32
        )
        seed = (measured[action_ids] - nominal) / denominator
        expected_shape = infer_fn.info["last_action"].shape
        seed = np.asarray(seed, dtype=np.float32).reshape(expected_shape)
        infer_fn.info["last_action"][...] = seed
        infer_fn.info["nn_action"][...] = seed
        infer_fn.info["motor_targets"][...] = measured
        return seed.copy()

    def prepare_frozen_policy_stand(
        root_quat,
        root_gyro,
        joint_qpos,
        joint_qvel,
        reference_template=None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build and dry-run a stationary reference without retaining history."""

        nonlocal prev_online_ref, remote_last_live_reference
        if reference_template is None:
            reference_template, _ = mocap_buffer.read()
            remote_last_live_reference = np.asarray(
                reference_template, dtype=np.float32
            ).copy()
        else:
            reference_template = np.asarray(
                reference_template, dtype=np.float32
            ).copy()
        anchor = build_measured_stand_anchor(
            reference_template, root_quat, joint_qpos
        )
        measured_seed = seed_policy_history_from_measured(joint_qpos)
        live_converter.reset()
        live_converter.set_robot_initial_pose(
            np.asarray(root_quat, dtype=np.float32),
            np.zeros(2, dtype=np.float32),
        )
        frozen = live_converter.convert(anchor)
        target = np.asarray(
            infer_fn.infer_onnx_real(
                root_quat,
                root_gyro,
                joint_qpos,
                joint_qvel,
                {"ref_curr": frozen, "ref_next": frozen},
            )
        ).reshape(-1)
        target = validate_policy_stand_target(joint_qpos, target)

        # Shadow inference was never sent.  Preserve the inverse action of the
        # measured hold (not zero and not the unsent action) as policy history.
        infer_fn.info["last_action"][...] = measured_seed
        infer_fn.info["nn_action"][...] = measured_seed
        infer_fn.info["motor_targets"][...] = np.asarray(
            joint_qpos, dtype=np.float32
        )
        infer_fn.info["step"] = 0
        live_converter.reset()
        prev_online_ref = None
        return anchor, target

    def require_optional_ownerless(context: str) -> None:
        """Fail on a reported owner; tolerate an unavailable optional RPC."""

        if motion_switcher is None:
            return
        owner = checked_motion_owner(motion_switcher, context=context)
        if owner != "":
            raise RemoteHandoverError(
                f"MotionSwitcher owner changed to {owner!r} {context}; "
                "refusing LowCmd"
            )

    try:
        if use_remote_handover:
            print(
                "<Mode: Develop handover> G1 must already be in Develop mode, "
                "standing still on the loaded support rig with the remote sticks "
                "centred. LowCmd is closed."
            )
            require_optional_ownerless("while starting the Develop handover")
            develop_dwell = StableStandDwell()
            develop_stand_ready = False
            next_owner_poll = 0.0
            last_ready_target = None
            last_shadow_error_log = 0.0

            while True:
                if cancel_requested():
                    return
                root_quat, root_gyro, joint_qpos, joint_qvel = (
                    low_ctrl.get_sensor_state()
                )
                now = time.monotonic()
                if motion_switcher is not None and now >= next_owner_poll:
                    require_optional_ownerless(
                        "while validating the pre-entered Develop stand"
                    )
                    next_owner_poll = now + 0.05

                stable_now = develop_dwell.update(
                    now, root_quat, root_gyro, joint_qpos, joint_qvel
                )
                if stable_now:
                    # Keep the complete reference/policy path hot while the
                    # publisher is still absent.  This is a shadow inference
                    # only: its action is validated and discarded.  Resetting
                    # policy history prevents unsent actions from becoming a
                    # false part of the first real observation after takeover.
                    try:
                        _, last_ready_target = prepare_frozen_policy_stand(
                            root_quat,
                            root_gyro,
                            joint_qpos,
                            joint_qvel,
                        )
                    except Exception as exc:
                        # Data/policy stay hot, but a missing or invalid
                        # reference simply keeps LowCmd closed.  It is safe to
                        # retry because no motor publisher exists.
                        develop_stand_ready = False
                        last_ready_target = None
                        if now - last_shadow_error_log >= 1.0:
                            print(
                                "[Real] Shadow policy is not ready "
                                f"({type(exc).__name__}: {exc}); keeping "
                                "LowCmd closed and retrying."
                            )
                            last_shadow_error_log = now
                        time.sleep(ctrl_dt)
                        continue
                    develop_stand_ready = True
                    print(
                        "<Mode: Develop ready for START> Fresh LowState, "
                        "standing dwell, and policy continuity checks "
                        "passed. Release every remote button, then press and hold "
                        "only START for two new LowState frames. Keep the support "
                        "rig loaded; LowCmd remains closed."
                    )
                    break
                develop_stand_ready = False
                last_ready_target = None
                time.sleep(ctrl_dt)

            if not develop_stand_ready or last_ready_target is None:
                raise RemoteHandoverError(
                    "stable standing/shadow gates did not remain ready; LowCmd "
                    "stays closed"
                )

            start_timeout = float(args.remote_start_timeout_seconds)
            if not np.isfinite(start_timeout) or start_timeout <= 0.0:
                raise ValueError(
                    "remote_start_timeout_seconds must be positive and finite"
                )
            tick_before_start = low_ctrl.low_state_tick
            if tick_before_start is None:
                raise RemoteHandoverError(
                    "no valid LowState tick exists before START authorization; "
                    "LowCmd stays closed"
                )
            start_deadline = time.monotonic() + start_timeout
            start_gate = ReleasedChordGate((KeyMap.start,))
            previous_remote_tick = tick_before_start
            next_owner_poll = 0.0
            start_tick = None
            while start_tick is None:
                if cancel_requested():
                    return
                now = time.monotonic()
                remaining = start_deadline - now
                if remaining <= 0.0:
                    raise RemoteHandoverError(
                        "timed out waiting for released-then-pressed START in "
                        "pre-entered Develop mode; LowCmd stays closed"
                    )
                if motion_switcher is not None and now >= next_owner_poll:
                    require_optional_ownerless(
                        "while waiting for physical START authorization"
                    )
                    next_owner_poll = now + 0.05
                remote_tick, remote_buttons, remote_state = (
                    low_ctrl.wait_for_remote_gate_state_after(
                        previous_remote_tick,
                        timeout_seconds=min(0.10, remaining),
                        should_cancel=cancel_requested,
                    )
                )
                previous_remote_tick = remote_tick
                reason = stable_stand_reason(*remote_state)
                if reason is not None:
                    raise RemoteHandoverError(
                        "G1 moved before START authorization: " + reason
                    )
                was_released = start_gate.released
                if start_gate.update(remote_buttons):
                    start_tick = remote_tick
                    print(
                        "<Mode: START authorization accepted> Fresh LowState "
                        "observed an exclusive two-frame START hold after every "
                        "remote button was released. LowCmd is still closed "
                        "pending one newer stable LowState."
                    )
                    break
                if not was_released and start_gate.released:
                    print(
                        "<Mode: START gate armed> A fresh stable LowState "
                        "observed all 16 remote buttons released. Press and hold "
                        "only START for at least two new LowState frames."
                    )
                time.sleep(ctrl_dt)

            require_optional_ownerless(
                "immediately after physical START authorization"
            )

            root_quat, root_gyro, joint_qpos, joint_qvel = (
                low_ctrl.wait_for_sensor_state_after(
                    start_tick,
                    timeout_seconds=0.25,
                    should_cancel=cancel_requested,
                )
            )
            reason = stable_stand_reason(
                root_quat, root_gyro, joint_qpos, joint_qvel
            )
            if reason is not None:
                raise RemoteHandoverError(
                    "G1 moved after START authorization: " + reason
                )
            remote_reference_anchor, _ = prepare_frozen_policy_stand(
                root_quat,
                root_gyro,
                joint_qpos,
                joint_qvel,
            )
            live_converter.reset()
            live_converter.set_robot_initial_pose(
                np.asarray(root_quat, dtype=np.float32),
                np.zeros(2, dtype=np.float32),
            )
            prev_online_ref = None
            seed_policy_history_from_measured(joint_qpos)
            infer_fn.info["step"] = 0

            remote_handover = RemoteOnlyHandover(
                stabilize_seconds=args.startup_settle_seconds,
                blend_seconds=args.startup_blend_seconds,
            )
            remote_handover.command(RemoteHandoverCommand.START)
            require_optional_ownerless(
                "immediately before opening the LowCmd writer"
            )
            if cancel_requested():
                return
            low_ctrl.enable_lowcmd_publisher()
            if cancel_requested():
                return

            # Publisher.Init may itself take hundreds of milliseconds.  Never
            # use pre-Init evidence as the first command's Develop lease.
            require_optional_ownerless(
                "after LowCmd writer Init and before any motor command"
            )
            tick_after_init = low_ctrl.low_state_tick
            root_quat, root_gyro, joint_qpos, joint_qvel = (
                low_ctrl.wait_for_sensor_state_after(
                    tick_after_init,
                    timeout_seconds=0.25,
                    should_cancel=cancel_requested,
                )
            )
            reason = stable_stand_reason(
                root_quat, root_gyro, joint_qpos, joint_qvel
            )
            if reason is not None:
                raise RemoteHandoverError(
                    "G1 moved while the LowCmd writer initialized: " + reason
                )
            remote_reference_anchor, remote_expected_first_target = (
                prepare_frozen_policy_stand(
                    root_quat,
                    root_gyro,
                    joint_qpos,
                    joint_qvel,
                    reference_template=remote_last_live_reference,
                )
            )
            live_converter.reset()
            live_converter.set_robot_initial_pose(
                np.asarray(root_quat, dtype=np.float32),
                np.zeros(2, dtype=np.float32),
            )
            prev_online_ref = None
            seed_policy_history_from_measured(joint_qpos)
            infer_fn.info["step"] = 0
            remote_first_policy_target_pending = True

            if cancel_requested():
                return
            require_optional_ownerless(
                "immediately before the first measured hold"
            )
            # Require one newer LowState and use it immediately for the first
            # measured hold.  If MotionSwitcher RPC is available, keep its
            # optional nonblocking monitor as an additional conflict detector.
            tick_after_final_owner_check = low_ctrl.low_state_tick
            if motion_switcher is not None:
                owner_monitor = MotionOwnerMonitor(motion_switcher)
                owner_monitor.start_after_validated_check()
            root_quat, root_gyro, joint_qpos, joint_qvel = (
                low_ctrl.wait_for_sensor_state_after(
                    tick_after_final_owner_check,
                    timeout_seconds=0.25,
                    should_cancel=cancel_requested,
                )
            )
            reason = stable_stand_reason(
                root_quat, root_gyro, joint_qpos, joint_qvel
            )
            if reason is not None:
                raise RemoteHandoverError(
                    "G1 moved after the final Develop check: " + reason
                )
            if owner_monitor is not None:
                owner_monitor.require_fresh_empty_owner()
            low_ctrl.step(
                np.asarray(joint_qpos, dtype=np.float32).copy(),
                consts.KPs,
                consts.KDs,
            )
            print(
                "<Mode: Measured hold> First LowCmd exactly holds the newest "
                "post-START measured standing pose; feedback policy starts on "
                "the next control cycle."
            )

            # The measured hold is already active, so it is now safe to rebuild
            # the stationary policy reference and shadow target from the exact
            # post-check LowState.  It reuses the already validated reference
            # template and therefore has no Redis/cache availability dependency.
            remote_reference_anchor, remote_expected_first_target = (
                prepare_frozen_policy_stand(
                    root_quat,
                    root_gyro,
                    joint_qpos,
                    joint_qvel,
                    reference_template=remote_last_live_reference,
                )
            )
            remote_last_output_reference = np.asarray(
                remote_reference_anchor, dtype=np.float32
            ).copy()
            live_converter.reset()
            live_converter.set_robot_initial_pose(
                np.asarray(root_quat, dtype=np.float32),
                np.zeros(2, dtype=np.float32),
            )
            prev_online_ref = None
            seed_policy_history_from_measured(joint_qpos)
            infer_fn.info["step"] = 0
            remote_first_policy_target_pending = True
            print("<Mode: Locomotion> Starting feedback-policy stand loop...")
            loco_thread = RecurrentThread(
                interval=ctrl_dt,
                target=guarded_locomotion_step,
                name="loco",
            )
            loco_thread.Start()

            while not cancel_requested():
                time.sleep(0.05)
            clean_remote_handback = remote_exit_requested and not thread_errors
            if thread_errors:
                raise RuntimeError(
                    f"real control loop stopped after "
                    f"{type(thread_errors[0]).__name__}: {thread_errors[0]}"
                ) from thread_errors[0]
        else:
            print("<Mode: Damping> Waiting for a released-then-pressed <START>...")
            start_was_released = False
            while True:
                if cancel_requested():
                    return
                low_ctrl.get_sensor_state()
                start_pressed = low_ctrl.remote_buttons[KeyMap.start] == 1
                if not start_pressed:
                    start_was_released = True
                elif start_was_released:
                    break
                low_ctrl.set_motor_damping()
                time.sleep(ctrl_dt)
            reached_default = low_ctrl.move_to_default_pos(
                duration=2.0,
                dt=ctrl_dt,
                should_cancel=cancel_requested,
            )
            if not reached_default or cancel_requested():
                stop_event.set()
                return
            print("<Mode: Default> Waiting for <A> on remote...")
            while low_ctrl.remote_buttons[KeyMap.A] != 1:
                if cancel_requested():
                    return
                low_ctrl.step(DEFAULT_QPOS_JOINT, consts.KPs, consts.KDs)
                time.sleep(ctrl_dt)
            if cancel_requested():
                return

            print("<Mode: Locomotion> Starting control loop...")
            loco_thread = RecurrentThread(
                interval=ctrl_dt, target=guarded_locomotion_step, name="loco"
            )
            loco_thread.Start()
            while not cancel_requested():
                time.sleep(0.05)
            if thread_errors:
                raise RuntimeError(
                    f"real control loop stopped after "
                    f"{type(thread_errors[0]).__name__}: {thread_errors[0]}"
                ) from thread_errors[0]
    finally:
        stop_event.set()
        cleanup_errors: list[BaseException] = []
        loco_joined = True
        try:
            _stop_and_wait_recurrent_thread(loco_thread, timeout_seconds=1.0)
        except BaseException as exc:
            loco_joined = False
            cleanup_errors.append(exc)
            print(
                "[Real] Policy thread join was not confirmed; skipping damping "
                "and closing the LowCmd writer immediately."
            )

        writer_was_enabled = low_ctrl.has_publisher
        remote_owner_safe = True
        if use_remote_handover and writer_was_enabled:
            if owner_monitor is not None:
                remote_owner_safe = owner_monitor.safe_for_damping()
                detail = owner_monitor.fault_detail
            else:
                try:
                    low_ctrl.get_sensor_state()
                except BaseException as exc:
                    remote_owner_safe = False
                    detail = f"LowState Develop lease failed: {exc}"
                else:
                    detail = "fresh LowState Develop lease"
            if not remote_owner_safe:
                print(
                    "[Real] Ownership/Develop evidence is conflicting or stale; no "
                    f"damping frames will be sent ({detail})."
                )

        if (writer_was_enabled or low_ctrl.debug) and loco_joined and remote_owner_safe:
            try:
                _send_shutdown_damping(
                    low_ctrl,
                    damping_frames,
                    ctrl_dt,
                    owner_monitor=owner_monitor if use_remote_handover else None,
                )
            except BaseException as exc:
                cleanup_errors.append(exc)
                print(
                    "[Real] Shutdown damping stopped before the next frame "
                    "because ownership/freshness or DDS validation failed; "
                    "closing the LowCmd writer."
                )
        elif not writer_was_enabled:
            print(
                "[Real] LowCmd was never enabled; no low-level shutdown frames "
                "were transmitted."
            )

        if writer_was_enabled:
            try:
                # No damping is attempted after a join/owner fault.  Closing the
                # writer is the only permitted action in those paths.
                low_ctrl.disable_lowcmd_publisher()
                print(
                    "<Mode: LowCmd closed> DDS writer closure confirmed. "
                    "The project is no longer sending motor commands."
                )
            except BaseException as exc:
                cleanup_errors.append(exc)

        owner_monitor_joined = True
        if owner_monitor is not None:
            try:
                owner_monitor.stop_and_join()
            except BaseException as exc:
                owner_monitor_joined = False
                cleanup_errors.append(exc)
                print(
                    "[Real] MotionSwitcher monitor did not join cleanly; "
                    "automatic handback observation is disabled."
                )

        if (
            clean_remote_handback
            and motion_switcher is not None
            and owner_monitor_joined
            and not cleanup_errors
        ):
            print(
                "<Mode: Restore official control> Now use the physical remote "
                "to restore G1 regular motion control; the program will only "
                "observe owner 'ai'."
            )
            deadline = time.monotonic() + float(
                args.remote_handback_timeout_seconds
            )
            restored = False
            handback_cancelled = False
            while time.monotonic() < deadline:
                if interrupt_event is not None and interrupt_event.is_set():
                    handback_cancelled = True
                    print(
                        "[Real] Physical-remote handback observation was "
                        "cancelled after LowCmd closed; exiting without waiting "
                        "for owner 'ai'."
                    )
                    break
                owner = checked_motion_owner(
                    motion_switcher,
                    context="while observing physical-remote handback",
                )
                if owner == OFFICIAL_MOTION_OWNER:
                    restored = True
                    print(
                        "<Mode: Official control restored> MotionSwitcher owner "
                        "is 'ai'; remote-only handover is complete."
                    )
                    break
                if owner:
                    cleanup_errors.append(
                        RemoteHandoverError(
                            f"unexpected owner {owner!r} during physical-remote "
                            "handback"
                        )
                    )
                    break
                time.sleep(0.10)
            if not restored and not handback_cancelled and not cleanup_errors:
                cleanup_errors.append(
                    RemoteHandoverError(
                        "LowCmd closed, but official owner 'ai' was not observed "
                        "before the handback timeout; keep the robot on its "
                        "support rig"
                    )
                )

        if mocap_reference_cache is not None:
            try:
                mocap_reference_cache.stop_and_join(timeout_seconds=0.5)
            except BaseException as exc:
                cleanup_errors.append(exc)
                print(
                    "[Real] CHINGMU reference cache did not stop cleanly; "
                    "the daemon worker will not be reused."
                )

        keyboard.close()
        print("[Real] Exited.")
        if cleanup_errors:
            first = cleanup_errors[0]
            raise RuntimeError(
                f"real-controller cleanup was not fully confirmed: "
                f"{type(first).__name__}: {first}"
            ) from first


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@dataclass
class DeployArgs:
    onnx_walk: str = "storage/ckpts/G1-Walk/07140632_G1-Walk_v2.0.0_baseline.onnx"
    track_dir: str = "storage/test"
    onnx_track: str = "storage/ckpts/pns_wo_priv216.onnx"
    policy_type: str = "mlp"
    convert_xml_path: str = str(consts.TRACK_XML)
    real: bool = False
    debug: bool = False
    freq: int = 50
    lowstate_timeout_seconds: float | None = 10.0
    model_profile: str = consts.G1_MODEL_PROFILE
    startup_handover: bool = True
    startup_stand_seconds: float = 3.0
    startup_settle_seconds: float = 0.5
    startup_blend_seconds: float = 1.5
    # Field firmware supplies LowState only after the operator has already
    # entered Develop mode.  START is then observed from fresh LowState as a
    # physical authorization; it is never sent or synthesized by this program.
    remote_start_timeout_seconds: float = 60.0
    remote_handback_timeout_seconds: float = 60.0
    official_standup_trace_output: str = ""
    official_standup_firmware_id: str = "unreported"

    # Mocap
    no_mocap: bool = False
    mocap_type: str = "pnlink"  # one of: pnlink | xsens | chingmu_redis
    human_height: float = 1.7
    visualize_retarget: bool = True
    buffer_ms: float = 30.0

    # Existing CHINGMU/GMR Redis reference (mocap_type=chingmu_redis)
    redis_host: str = "127.0.0.1"
    redis_port: int = 6379
    redis_key: str = "action_qpos_g1_packet"

    # Xsens MVN streamer (only used when --mocap_type xsens).
    xsens_host: str = "0.0.0.0"      # local bind address for the receiver
    xsens_port: int = 9763           # default MVN Network Streamer port
    xsens_protocol: str = "tcp"      # "tcp" or "udp" - match MVN Studio setting

    # Real robot
    net: str = "enx6c1ff76e8ef5"
    enable_hand: bool = False


def main(args: DeployArgs):
    if args.model_profile != consts.G1_MODEL_PROFILE:
        raise ValueError("model profile differs from G1_MODEL_PROFILE; launch through the managed supervisor")
    if args.real:
        run_real(args)
    else:
        run_sim(args)


if __name__ == "__main__":
    main(tyro.cli(DeployArgs))
