"""Exact CHINGMU local frames to GMR kinematic MuJoCo preview."""

from __future__ import annotations

import argparse
import pickle
import socket
import time

# Direct-script bootstrap: the preview supervisor launches this file as a
# script (M17 src-only deployment); make the src package importable first.
import sys as _sys
from pathlib import Path as _Path

if str(_Path(__file__).resolve().parents[2]) not in _sys.path:
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import numpy as np

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer

from src.adapters.frame_transport import (
    add_frame_transport_arguments,
    open_frame_receiver,
)
from src.motion.gmr_exact import BODY_NAMES, make_exact_gmr_frame


def main() -> None:
    parser = argparse.ArgumentParser()
    add_frame_transport_arguments(parser)
    parser.add_argument("--human-height", type=float, default=1.75)
    parser.add_argument("--viewer-fps", type=float, default=50.0)
    args = parser.parse_args()

    retargeter = GMR(
        src_human="bvh_nokov",
        tgt_robot="unitree_g1",
        actual_human_height=args.human_height,
    )
    viewer = RobotMotionViewer(
        robot_type="unitree_g1",
        motion_fps=int(args.viewer_fps),
        transparent_robot=0,
        record_video=False,
    )
    frame_receiver = open_frame_receiver(
        args.frame_transport,
        args.frame_socket,
        args.udp_host,
        args.udp_port,
    )
    frame_receiver.settimeout(0.5)

    print(f"GMR exact preview ready on {frame_receiver.endpoint}", flush=True)
    last_qpos = retargeter.configuration.data.qpos.copy()
    last_render = 0.0
    render_period = 1.0 / args.viewer_fps
    rendered = 0
    rejected = 0
    report_start = time.monotonic()
    try:
        while True:
            try:
                packet, _address = frame_receiver.recvfrom(65535)
            except socket.timeout:
                continue
            frame_id, positions, quaternions, _detected = pickle.loads(packet)
            positions = np.asarray(positions, dtype=float)
            quaternions = np.asarray(quaternions, dtype=float)
            if positions.shape[0] < len(BODY_NAMES) or quaternions.shape[0] < len(BODY_NAMES):
                rejected += 1
                continue
            norms = np.linalg.norm(quaternions[: len(BODY_NAMES)], axis=1)
            if np.any(norms < 1e-8):
                rejected += 1
                continue
            quaternions[: len(BODY_NAMES)] /= norms[:, None]
            frame = make_exact_gmr_frame(positions, quaternions)
            try:
                qpos = retargeter.retarget(frame).copy()
            except Exception as error:
                retargeter.configuration.update(last_qpos)
                rejected += 1
                if rejected == 1 or rejected % 25 == 0:
                    print(
                        f"Rejected IK frame {frame_id}: {type(error).__name__}: {error}",
                        flush=True,
                    )
                continue
            last_qpos = qpos
            now = time.monotonic()
            if now - last_render < render_period:
                continue
            last_render = now
            viewer.step(
                root_pos=qpos[:3],
                root_rot=qpos[3:7],
                dof_pos=qpos[7:],
                human_motion_data=retargeter.scaled_human_data,
                rate_limit=False,
                follow_camera=True,
            )
            rendered += 1
            elapsed = now - report_start
            if elapsed >= 2.0:
                print(
                    f"GMR Preview {rendered / elapsed:.1f} Hz | frame {frame_id} | "
                    f"root z {qpos[2]:.3f} m | rejected {rejected}",
                    flush=True,
                )
                rendered = 0
                report_start = now
    except KeyboardInterrupt:
        pass
    finally:
        frame_receiver.close()
        viewer.close()
        print("GMR exact preview stopped.", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
