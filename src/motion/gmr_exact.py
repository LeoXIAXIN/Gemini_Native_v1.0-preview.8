"""Exact, reference-free CHINGMU retarget stream to GMR/MuJoCo."""

from __future__ import annotations

import argparse
import ctypes
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from src.adapters.gmr_headless import load_general_motion_retargeting


MAX_SEGMENTS = 150
BODY_NAMES = [
    "Hips", "Spine", "Spine1", "Spine2", "Spine3", "Neck", "Head",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase",
    "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase",
]
BODY_PARENTS = [-1, 0, 1, 2, 3, 4, 5, 4, 7, 8, 9, 4, 11, 12, 13, 0, 15, 16, 17, 0, 19, 20, 21]

# CHINGMU live (Z-up, mm) -> FBX/BVH (Y-up, cm): (-X, Z, Y).
LIVE_TO_BVH = Rotation.from_matrix(
    np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
)
# GMR's Nokov loader premultiplies FBX/BVH global orientations by +90deg X.
BVH_TO_GMR = Rotation.from_matrix(
    np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
)
LIVE_TO_GMR_POSITION = np.diag([-1.0, -1.0, 1.0])


def make_exact_gmr_frame(
    positions_mm: np.ndarray, local_xyzw: np.ndarray
) -> dict[str, list[np.ndarray]]:
    """Convert the first 23 standard CHINGMU segments exactly as FBX/BVH export."""
    positions = positions_mm[: len(BODY_NAMES)] @ LIVE_TO_GMR_POSITION.T * 0.001
    live_local = Rotation.from_quat(local_xyzw[: len(BODY_NAMES)])
    bvh_local = LIVE_TO_BVH * live_local * LIVE_TO_BVH.inv()
    bvh_global: list[Rotation] = []
    for joint, parent in enumerate(BODY_PARENTS):
        if parent < 0:
            bvh_global.append(bvh_local[joint])
        else:
            bvh_global.append(bvh_global[parent] * bvh_local[joint])

    result: dict[str, list[np.ndarray]] = {}
    for joint, name in enumerate(BODY_NAMES):
        gmr_xyzw = (BVH_TO_GMR * bvh_global[joint]).as_quat()
        result[name] = [positions[joint], gmr_xyzw[[3, 0, 1, 2]]]
    result["LeftFootMod"] = [result["LeftFoot"][0], result["LeftToeBase"][1]]
    result["RightFootMod"] = [result["RightFoot"][0], result["RightToeBase"][1]]
    return result


def main() -> None:
    from general_motion_retargeting import RobotMotionViewer

    parser = argparse.ArgumentParser()
    parser.add_argument("--server-ip", default="192.168.2.100")
    parser.add_argument(
        "--sdk-root", type=Path, default=Path.home() / "ChingMuPythonSDKs_Linux"
    )
    parser.add_argument("--human-id", type=int, default=0)
    parser.add_argument("--human-height", type=float, default=1.75)
    args = parser.parse_args()

    retargeter = load_general_motion_retargeting()(
        src_human="bvh_nokov",
        tgt_robot="unitree_g1",
        actual_human_height=args.human_height,
    )
    viewer = RobotMotionViewer(
        robot_type="unitree_g1",
        motion_fps=120,
        transparent_robot=0,
        record_video=False,
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
    call_index = 0
    last_timecode = None
    last_qpos = retargeter.configuration.data.qpos.copy()
    rendered = 0
    rejected = 0
    started = time.monotonic()

    print("Exact CHINGMU retarget -> FBX/BVH -> GMR (Ctrl+C to stop)")
    print("No reference BVH and no learned calibration are used.")
    library.CMVrpnStartExtern()
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
                time.sleep(0.003)
                continue
            frame_id = int(timecode[0])
            if frame_id == last_timecode:
                time.sleep(0.001)
                continue
            last_timecode = frame_id
            call_index += 1

            position_array = np.ctypeslib.as_array(positions).reshape(MAX_SEGMENTS, 3).copy()
            quaternion_array = np.ctypeslib.as_array(quaternions).reshape(MAX_SEGMENTS, 4).copy()
            norms = np.linalg.norm(quaternion_array[: len(BODY_NAMES)], axis=1)
            if np.any(norms < 1e-8):
                continue
            quaternion_array[: len(BODY_NAMES)] /= norms[:, None]
            human_frame = make_exact_gmr_frame(position_array, quaternion_array)
            try:
                qpos = retargeter.retarget(human_frame)
            except Exception as error:
                retargeter.configuration.update(last_qpos)
                rejected += 1
                if rejected == 1 or rejected % 25 == 0:
                    print(f"Rejected frame {frame_id}: {type(error).__name__}: {error}")
                continue
            last_qpos = qpos.copy()
            viewer.step(
                root_pos=qpos[:3],
                root_rot=qpos[3:7],
                dof_pos=qpos[7:],
                human_motion_data=retargeter.scaled_human_data,
                rate_limit=False,
                follow_camera=True,
            )
            rendered += 1
            elapsed = time.monotonic() - started
            if elapsed >= 2.0:
                print(
                    f"Live FPS: {rendered / elapsed:.1f}, frame: {frame_id}, "
                    f"rejected IK: {rejected}"
                )
                rendered = 0
                started = time.monotonic()
    except KeyboardInterrupt:
        print("Stopping exact live pipeline...")
    finally:
        library.CMVrpnQuitExtern()
        viewer.close()
        print("CHINGMU and GMR stopped.")


if __name__ == "__main__":
    raise SystemExit(main())
