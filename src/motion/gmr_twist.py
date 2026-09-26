"""Stream exact CHINGMU -> GMR targets to the TWIST RESP interface.

This is the high-level half of the pipeline.  It never talks to a robot and
never sends motor commands.  The output is TWIST's 33-value ``action_mimic_g1``
reference consumed by ``server_low_level_g1_sim.py``.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import redis
from scipy.spatial.transform import Rotation

from src.adapters.gmr_headless import load_general_motion_retargeting
from src.motion.gmr_exact import BODY_NAMES, MAX_SEGMENTS, make_exact_gmr_frame


# GMR Unitree G1 is 29 DoF.  TWIST's released policy uses the same first 19
# joints, wrist roll, then the right arm; wrist pitch/yaw are not policy DoFs.
GMR_29_DOF_NAMES = [
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
]
TWIST_25_DOF_NAMES = [
    *GMR_29_DOF_NAMES[:20],
    *GMR_29_DOF_NAMES[22:27],
]
GMR_TO_TWIST = np.array(
    [GMR_29_DOF_NAMES.index(name) for name in TWIST_25_DOF_NAMES], dtype=int
)

DEFAULT_MIMIC_OBS = np.concatenate(
    [
        np.array([0.793]),                 # pelvis height
        np.zeros(3),                       # roll, pitch, yaw
        np.zeros(3),                       # root velocity in root frame
        np.zeros(1),                       # root yaw velocity
        np.array(
            [
                -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,
                -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,
                0.0, 0.0, 0.0,
                0.0, 0.2, 0.0, 1.2, 0.0,
                0.0, -0.2, 0.0, 1.2, 0.0,
            ]
        ),
    ]
).astype(np.float32)


class RootVelocityEstimator:
    """Finite-difference root velocity with a short causal moving average."""

    def __init__(self, window: int = 5) -> None:
        self.previous_time: float | None = None
        self.previous_position: np.ndarray | None = None
        self.previous_rotation: Rotation | None = None
        self.linear_history: deque[np.ndarray] = deque(maxlen=window)
        self.angular_history: deque[np.ndarray] = deque(maxlen=window)

    def update(
        self, timestamp: float, position: np.ndarray, quaternion_wxyz: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        rotation = Rotation.from_quat(quaternion_wxyz[[1, 2, 3, 0]])
        local_linear = np.zeros(3)
        local_angular = np.zeros(3)
        if (
            self.previous_time is not None
            and self.previous_position is not None
            and self.previous_rotation is not None
        ):
            dt = timestamp - self.previous_time
            if 0.005 <= dt <= 0.2:
                world_linear = (position - self.previous_position) / dt
                world_delta = rotation * self.previous_rotation.inv()
                world_angular = world_delta.as_rotvec() / dt
                local_linear = rotation.inv().apply(world_linear)
                local_angular = rotation.inv().apply(world_angular)
                self.linear_history.append(local_linear)
                self.angular_history.append(local_angular)
            else:
                self.linear_history.clear()
                self.angular_history.clear()

        self.previous_time = timestamp
        self.previous_position = position.copy()
        self.previous_rotation = rotation
        if self.linear_history:
            local_linear = np.mean(np.stack(self.linear_history), axis=0)
            local_angular = np.mean(np.stack(self.angular_history), axis=0)
        return local_linear, local_angular


def gmr_qpos_to_twist_mimic(
    qpos: np.ndarray,
    timestamp: float,
    velocity_estimator: RootVelocityEstimator,
) -> np.ndarray:
    """Convert GMR's 36-value G1 qpos into TWIST's 33-value target."""
    qpos = np.asarray(qpos, dtype=float)
    if qpos.shape != (36,):
        raise ValueError(f"Expected GMR qpos shape (36,), got {qpos.shape}")
    if not np.all(np.isfinite(qpos)):
        raise ValueError("GMR qpos contains NaN or infinity")

    root_position = qpos[:3]
    root_quaternion_wxyz = qpos[3:7]
    quaternion_norm = np.linalg.norm(root_quaternion_wxyz)
    if quaternion_norm < 1e-8:
        raise ValueError("GMR root quaternion has zero length")
    root_quaternion_wxyz = root_quaternion_wxyz / quaternion_norm
    rotation = Rotation.from_quat(root_quaternion_wxyz[[1, 2, 3, 0]])
    roll_pitch_yaw = rotation.as_euler("xyz")
    local_linear, local_angular = velocity_estimator.update(
        timestamp, root_position, root_quaternion_wxyz
    )
    twist_dof = qpos[7:][GMR_TO_TWIST]
    mimic = np.concatenate(
        [
            root_position[2:3],
            roll_pitch_yaw,
            local_linear,
            local_angular[2:3],
            twist_dof,
        ]
    ).astype(np.float32)
    if mimic.shape != (33,):
        raise AssertionError(f"TWIST mimic target must be 33 values, got {mimic.shape}")
    return mimic


def publish(redis_client: redis.Redis, mimic: np.ndarray) -> None:
    redis_client.set("action_mimic_g1", json.dumps(mimic.tolist()))
    redis_client.set("action_hand_g1", json.dumps(np.zeros(14).tolist()))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exact CHINGMU -> GMR -> TWIST high-level bridge (simulation-safe)"
    )
    parser.add_argument("--server-ip", default="192.168.2.100")
    parser.add_argument(
        "--sdk-root", type=Path, default=Path.home() / "ChingMuPythonSDKs_Linux"
    )
    parser.add_argument("--human-id", type=int, default=0)
    parser.add_argument("--human-height", type=float, default=1.75)
    parser.add_argument("--publish-fps", type=float, default=50.0)
    parser.add_argument("--redis-host", default="127.0.0.1")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--transition-seconds", type=float, default=1.5)
    args = parser.parse_args()

    redis_client = redis.Redis(
        host=args.redis_host,
        port=args.redis_port,
        db=0,
        socket_timeout=1.0,
        protocol=2,
    )
    redis_client.ping()
    publish(redis_client, DEFAULT_MIMIC_OBS)
    redis_client.set("chingmu_gmr_bridge_status", "starting")

    retargeter = load_general_motion_retargeting()(
        src_human="bvh_nokov",
        tgt_robot="unitree_g1",
        actual_human_height=args.human_height,
    )
    library = ctypes.CDLL(str(args.sdk_root / "ChingmuDLL" / "libCMVrpn.so"))
    library.CMVrpnStartExtern.argtypes = []
    library.CMVrpnStartExtern.restype = None
    library.CMVrpnQuitExtern.argtypes = []
    library.CMVrpnQuitExtern.restype = None
    library.CMRetargetHumanExternTC.argtypes = [
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int),
    ]
    library.CMRetargetHumanExternTC.restype = ctypes.c_int

    host = f"MCAvatar@{args.server_ip}".encode("gbk")
    timecode = (ctypes.c_int * 1)()
    positions = (ctypes.c_double * (MAX_SEGMENTS * 3))()
    quaternions = (ctypes.c_double * (MAX_SEGMENTS * 4))()
    detected = (ctypes.c_int * MAX_SEGMENTS)()
    velocity_estimator = RootVelocityEstimator(window=5)
    last_qpos = retargeter.configuration.data.qpos.copy()
    last_timecode: int | None = None
    last_output = DEFAULT_MIMIC_OBS.copy()
    first_live_time: float | None = None
    next_publish = time.monotonic()
    publish_period = 1.0 / args.publish_fps
    call_index = 0
    accepted = 0
    rejected = 0
    published = 0
    report_start = time.monotonic()

    print("CHINGMU -> exact GMR -> TWIST Redis bridge", flush=True)
    print(
        "Simulation target only: no DDS and no motor commands. Ctrl+C to stop.",
        flush=True,
    )
    print(
        f"Redis action_mimic_g1 initialized with {len(DEFAULT_MIMIC_OBS)} values.",
        flush=True,
    )
    print(f"Starting CHINGMU SDK for {host.decode('gbk')}...", flush=True)
    library.CMVrpnStartExtern()
    redis_client.set("chingmu_gmr_bridge_status", "waiting_for_first_frame")
    print("CHINGMU SDK started; waiting for the first mocap frame...", flush=True)
    try:
        while True:
            ok = library.CMRetargetHumanExternTC(
                host,
                args.human_id,
                call_index,
                timecode,
                positions,
                quaternions,
                detected,
            )
            if not ok:
                time.sleep(0.002)
                continue
            frame_id = int(timecode[0])
            if frame_id == last_timecode:
                time.sleep(0.001)
                continue
            last_timecode = frame_id
            call_index += 1

            position_array = np.ctypeslib.as_array(positions).reshape(MAX_SEGMENTS, 3).copy()
            quaternion_array = (
                np.ctypeslib.as_array(quaternions).reshape(MAX_SEGMENTS, 4).copy()
            )
            norms = np.linalg.norm(quaternion_array[: len(BODY_NAMES)], axis=1)
            if np.any(norms < 1e-8):
                rejected += 1
                continue
            quaternion_array[: len(BODY_NAMES)] /= norms[:, None]
            human_frame = make_exact_gmr_frame(position_array, quaternion_array)
            try:
                qpos = retargeter.retarget(human_frame).copy()
            except Exception as error:
                retargeter.configuration.update(last_qpos)
                rejected += 1
                if rejected == 1 or rejected % 25 == 0:
                    print(f"Rejected IK frame {frame_id}: {type(error).__name__}: {error}")
                continue
            last_qpos = qpos
            accepted += 1

            now = time.monotonic()
            if now < next_publish:
                continue
            while next_publish <= now:
                next_publish += publish_period
            try:
                live_target = gmr_qpos_to_twist_mimic(qpos, now, velocity_estimator)
            except ValueError as error:
                rejected += 1
                print(f"Rejected TWIST target at frame {frame_id}: {error}")
                continue

            # Smoothly enter the live target so the simulated controller does not
            # receive an instantaneous jump from its default standing reference.
            if first_live_time is None:
                first_live_time = now
            alpha = min(1.0, (now - first_live_time) / args.transition_seconds)
            output = DEFAULT_MIMIC_OBS + alpha * (live_target - DEFAULT_MIMIC_OBS)
            publish(redis_client, output)
            redis_client.set(
                "chingmu_gmr_bridge_status", f"live:{frame_id}:{live_target[0]:.4f}"
            )
            last_output = output
            published += 1

            elapsed = now - report_start
            if elapsed >= 2.0:
                print(
                    f"Bridge {published / elapsed:.1f} Hz | frame {frame_id} | "
                    f"root z {live_target[0]:.3f} m | IK rejected {rejected}"
                )
                published = 0
                report_start = now
    except KeyboardInterrupt:
        print("Returning TWIST target to its default stand...")
    finally:
        steps = max(1, int(args.transition_seconds * args.publish_fps))
        for step in range(1, steps + 1):
            output = last_output + (DEFAULT_MIMIC_OBS - last_output) * (step / steps)
            try:
                publish(redis_client, output)
            except Exception:
                break
            time.sleep(publish_period)
        library.CMVrpnQuitExtern()
        try:
            redis_client.set("chingmu_gmr_bridge_status", "stopped")
        except Exception:
            pass
        print(f"Stopped cleanly. Accepted IK frames: {accepted}; rejected: {rejected}")


if __name__ == "__main__":
    raise SystemExit(main())
