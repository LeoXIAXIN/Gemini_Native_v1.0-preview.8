"""Receive isolated CHINGMU frames and publish robot-policy targets."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import socket
import time
import uuid

import numpy as np
import redis

from src.adapters.frame_transport import (
    add_frame_transport_arguments,
    open_frame_receiver,
)
from src.adapters.gmr_headless import load_general_motion_retargeting
from src.motion.gmr_exact import BODY_NAMES, make_exact_gmr_frame
from src.motion.gmr_twist import (
    DEFAULT_MIMIC_OBS,
    GMR_29_DOF_NAMES,
    RootVelocityEstimator,
    gmr_qpos_to_twist_mimic,
    publish,
)


def _compact_joint_names(names: list[str], limit: int = 8) -> str:
    """Keep Redis/console diagnostics readable for an incomplete skeleton."""

    if not names:
        return "none"
    if len(names) <= limit:
        return ",".join(names)
    return f"{','.join(names[:limit])},+{len(names) - limit}_more"


def _validate_core_skeleton(
    positions: np.ndarray,
    quaternions: np.ndarray,
    detected: object,
) -> tuple[str | None, str | None]:
    """Describe frames that cannot represent the standard 23-part skeleton.

    The old receiver silently discarded these frames and left its status at
    ``waiting_for_frame``.  That made a wrong Skeleton ID look like GMR had
    never started even though UDP packets were arriving normally.
    """

    required = len(BODY_NAMES)
    if (
        positions.ndim != 2
        or positions.shape[0] < required
        or positions.shape[1] < 3
        or quaternions.ndim != 2
        or quaternions.shape[0] < required
        or quaternions.shape[1] < 4
    ):
        status = f"shape:p{tuple(positions.shape)}:q{tuple(quaternions.shape)}"
        detail = (
            f"packet shape is positions={positions.shape}, quaternions={quaternions.shape}; "
            f"expected at least ({required}, 3) and ({required}, 4)"
        )
        return status, detail

    core_positions = positions[:required, :3]
    core_quaternions = quaternions[:required, :4]
    invalid_positions = ~np.all(np.isfinite(core_positions), axis=1)
    norms = np.linalg.norm(core_quaternions, axis=1)
    invalid_quaternions = ~np.isfinite(norms) | (norms < 1e-8)
    if not np.any(invalid_positions) and not np.any(invalid_quaternions):
        return None, None

    bad_position_names = [
        BODY_NAMES[index] for index in np.flatnonzero(invalid_positions)
    ]
    bad_quaternion_names = [
        BODY_NAMES[index] for index in np.flatnonzero(invalid_quaternions)
    ]
    detected_array = np.asarray(detected).reshape(-1)
    missing_detected_names = (
        [
            BODY_NAMES[index]
            for index in range(required)
            if index >= detected_array.size or not bool(detected_array[index])
        ]
        if detected_array.size
        else list(BODY_NAMES)
    )
    detected_count = required - len(missing_detected_names)

    status_parts = [f"detected={detected_count}/{required}"]
    detail_parts = [f"SDK detected {detected_count}/{required} core segments"]
    if bad_position_names:
        compact = _compact_joint_names(bad_position_names)
        status_parts.append(f"bad_pos={compact}")
        detail_parts.append(f"invalid positions: {compact}")
    if bad_quaternion_names:
        compact = _compact_joint_names(bad_quaternion_names)
        status_parts.append(f"bad_quat={compact}")
        detail_parts.append(f"zero/invalid quaternions: {compact}")
    if missing_detected_names:
        compact = _compact_joint_names(missing_detected_names)
        status_parts.append(f"missing={compact}")
        detail_parts.append(f"not detected: {compact}")
    return ";".join(status_parts), "; ".join(detail_parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_frame_transport_arguments(parser)
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
    redis_client.set("chingmu_gmr_bridge_status", "starting_gmr_receiver")

    retargeter = load_general_motion_retargeting()(
        src_human="bvh_nokov",
        tgt_robot="unitree_g1",
        actual_human_height=args.human_height,
    )
    try:
        frame_receiver = open_frame_receiver(
            args.frame_transport,
            args.frame_socket,
            args.udp_host,
            args.udp_port,
        )
    except OSError as error:
        redis_client.set(
            "chingmu_gmr_bridge_status",
            f"error:frame_bind:{args.frame_transport}:{error.errno}:{error.strerror}",
        )
        raise
    except Exception as error:
        redis_client.set(
            "chingmu_gmr_bridge_status",
            f"error:frame_bind:{args.frame_transport}:{type(error).__name__}",
        )
        raise
    frame_receiver.settimeout(0.5)
    redis_client.set("chingmu_gmr_bridge_status", "waiting_for_local_frame")
    print(f"GMR receiver ready on {frame_receiver.endpoint}", flush=True)

    estimator = RootVelocityEstimator(window=5)
    last_qpos = retargeter.configuration.data.qpos.copy()
    last_output = DEFAULT_MIMIC_OBS.copy()
    first_live_time: float | None = None
    next_publish = time.monotonic()
    publish_period = 1.0 / args.publish_fps
    raw_received = 0
    raw_received_total = 0
    received = 0
    accepted_total = 0
    rejected = 0
    published = 0
    sequence = 0
    report_start = time.monotonic()
    last_invalid_report = 0.0
    first_ik_started = False
    session_id = uuid.uuid4().hex
    dof_order_hash = hashlib.sha256(
        "\n".join(GMR_29_DOF_NAMES).encode("utf-8")
    ).hexdigest()
    try:
        while True:
            try:
                packet, _address = frame_receiver.recvfrom(65535)
            except socket.timeout:
                continue
            raw_received += 1
            raw_received_total += 1
            if raw_received_total == 1:
                redis_client.set("chingmu_gmr_bridge_status", "local_frame_received")
                print(
                    f"GMR received first local frame packet ({len(packet)} bytes); "
                    "decoding skeleton.",
                    flush=True,
                )
            frame_id, positions, quaternions, detected = pickle.loads(packet)
            positions = np.asarray(positions, dtype=float)
            quaternions = np.asarray(quaternions, dtype=float)
            invalid_status, invalid_detail = _validate_core_skeleton(
                positions, quaternions, detected
            )
            if invalid_status is not None:
                rejected += 1
                now = time.monotonic()
                if last_invalid_report == 0.0 or now - last_invalid_report >= 5.0:
                    status = f"invalid_skeleton:{invalid_status}"
                    redis_client.set("chingmu_gmr_bridge_status", status)
                    print(
                        f"GMR bridge invalid_skeleton frame {frame_id}: {invalid_detail}. "
                        "Check MCAvatar Skeleton ID and core skeleton solving.",
                        flush=True,
                    )
                    last_invalid_report = now
                continue
            norms = np.linalg.norm(quaternions[: len(BODY_NAMES), :4], axis=1)
            positions = positions[:, :3]
            quaternions = quaternions[:, :4]
            quaternions[: len(BODY_NAMES)] /= norms[:, None]
            human_frame = make_exact_gmr_frame(positions, quaternions)
            if not first_ik_started:
                first_ik_started = True
                redis_client.set("chingmu_gmr_bridge_status", "retargeting_first_frame")
                print(f"GMR retargeting first valid frame {frame_id}...", flush=True)
            try:
                qpos = retargeter.retarget(human_frame).copy()
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
            received += 1
            accepted_total += 1
            if accepted_total == 1:
                print(f"GMR first frame {frame_id} retargeted successfully.", flush=True)

            now = time.monotonic()
            if now < next_publish:
                continue
            while next_publish <= now:
                next_publish += publish_period
            target = gmr_qpos_to_twist_mimic(qpos, now, estimator)
            if first_live_time is None:
                first_live_time = now
            alpha = min(1.0, (now - first_live_time) / args.transition_seconds)
            output = DEFAULT_MIMIC_OBS + alpha * (target - DEFAULT_MIMIC_OBS)
            publish(redis_client, output)
            sequence += 1
            packet_meta = {
                "schema_version": 1,
                "session_id": session_id,
                "sequence": int(sequence),
                "frame_id": int(frame_id),
                "generated_monotonic": float(now),
                "valid_until_monotonic": float(now + 0.08),
                "robot_model": "unitree_g1_29dof",
                "dof_order_hash": dof_order_hash,
            }
            # Unlike the legacy ``action_mimic_g1`` Redis value, this packet
            # carries a session, sequence and freshness deadline.  Safety
            # consumers must never treat a repeatedly-read Redis value as a
            # newly received frame.
            redis_client.set(
                "action_mimic_g1_packet",
                json.dumps({**packet_meta, "mimic": output.tolist()}),
            )
            # Full 29-DoF GMR reference for alternative tracking backends such
            # as Humanoid-GPT.  The monotonic timestamp is comparable across
            # local WSL processes and allows a consumer-side stale-data guard.
            redis_client.set(
                "action_qpos_g1_packet",
                json.dumps(
                    {
                        **packet_meta,
                        # Kept for old consumers while the safety gateway uses
                        # ``generated_monotonic`` and ``sequence`` above.
                        "monotonic": float(now),
                        "qpos": qpos.tolist(),
                    }
                ),
            )
            redis_client.set(
                "chingmu_gmr_bridge_status", f"live:{frame_id}:{target[0]:.4f}"
            )
            last_output = output
            published += 1

            elapsed = now - report_start
            if elapsed >= 2.0:
                print(
                    f"Bridge {published / elapsed:.1f} Hz | local RX "
                    f"{raw_received / elapsed:.1f} Hz | IK {received / elapsed:.1f} Hz "
                    f"| frame {frame_id} | root z {target[0]:.3f} m | rejected {rejected}",
                    flush=True,
                )
                raw_received = 0
                received = 0
                published = 0
                report_start = now
    except KeyboardInterrupt:
        print("Returning TWIST target to default stand...", flush=True)
    finally:
        steps = max(1, int(args.transition_seconds * args.publish_fps))
        for step in range(1, steps + 1):
            output = last_output + (DEFAULT_MIMIC_OBS - last_output) * (step / steps)
            try:
                publish(redis_client, output)
            except Exception:
                break
            time.sleep(publish_period)
        try:
            redis_client.set("chingmu_gmr_bridge_status", "stopped")
        except Exception:
            pass
        frame_receiver.close()
        print(
            f"GMR receiver stopped; local packets: {raw_received_total}, "
            f"IK accepted: {accepted_total}, rejected: {rejected}",
            flush=True,
        )


if __name__ == "__main__":
    raise SystemExit(main())
