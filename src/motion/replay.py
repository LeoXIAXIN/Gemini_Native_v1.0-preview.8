"""Replay an approved CHINGMU recording through the UDP frame contract."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import pickle
import socket
import time

import numpy as np

from src.adapters.runtime import PipelinePaths
from src.domain.process_spec import build_replay_sender_command


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay CHINGMU NPZ frames to the Windows-native GMR bridge."
    )
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--udp-host", default="127.0.0.1")
    parser.add_argument("--udp-port", type=int, default=15150)
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--loop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Stop after this many seconds; zero runs until Ctrl+C.",
    )
    return parser


def load_recording(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "source_timecodes",
            "source_positions",
            "global_t_local_r_quaternions",
        }
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"recording is missing arrays: {sorted(missing)}")
        frame_ids = np.asarray(archive["source_timecodes"], dtype=np.int64)
        positions = np.asarray(archive["source_positions"], dtype=np.float64)
        quaternions = np.asarray(
            archive["global_t_local_r_quaternions"], dtype=np.float64
        )
    if frame_ids.ndim != 1 or positions.ndim != 3 or quaternions.ndim != 3:
        raise ValueError("recording arrays have invalid dimensions")
    if not (len(frame_ids) == len(positions) == len(quaternions)):
        raise ValueError("recording arrays have different frame counts")
    if positions.shape[1:] != (63, 3) or quaternions.shape[1:] != (63, 4):
        raise ValueError(
            "recording must contain CHINGMU 63x3 positions and 63x4 quaternions"
        )
    if len(frame_ids) == 0:
        raise ValueError("recording contains no frames")
    if not np.isfinite(positions).all() or not np.isfinite(quaternions).all():
        raise ValueError("recording contains non-finite skeleton values")
    return frame_ids, positions, quaternions


def encode_frame(
    frame_id: int, positions: np.ndarray, quaternions: np.ndarray
) -> bytes:
    detected = [1] * int(positions.shape[0])
    return pickle.dumps(
        (
            int(frame_id),
            positions.tolist(),
            quaternions.tolist(),
            detected,
        ),
        protocol=5,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    recording = args.recording.expanduser().resolve()
    if not recording.is_file():
        raise FileNotFoundError(f"recording does not exist: {recording}")
    if not 1 <= args.udp_port <= 65535:
        raise SystemExit("--udp-port must be between 1 and 65535")
    if args.fps <= 0:
        raise SystemExit("--fps must be positive")
    if args.duration < 0:
        raise SystemExit("--duration must be zero or positive")

    frame_ids, positions, quaternions = load_recording(recording)
    period = 1.0 / args.fps
    started = time.monotonic()
    deadline = started + args.duration if args.duration > 0 else None
    next_frame = started
    sent = 0
    print(
        f"Recorded CHINGMU replay ready: {len(frame_ids)} frames -> "
        f"udp:{args.udp_host}:{args.udp_port} at {args.fps:.1f} Hz",
        flush=True,
    )
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
            while True:
                for index, frame_id in enumerate(frame_ids):
                    now = time.monotonic()
                    if deadline is not None and now >= deadline:
                        print(f"Recorded CHINGMU replay completed: {sent} frames", flush=True)
                        return 0
                    delay = next_frame - now
                    if delay > 0:
                        time.sleep(delay)
                    sender.sendto(
                        encode_frame(frame_id, positions[index], quaternions[index]),
                        (args.udp_host, args.udp_port),
                    )
                    sent += 1
                    next_frame += period
                    if sent == 1:
                        print("Recorded CHINGMU replay sent first frame.", flush=True)
                if not args.loop:
                    break
    except KeyboardInterrupt:
        pass
    print(f"Recorded CHINGMU replay stopped: {sent} frames", flush=True)
    return 0


class ReplayService:
    """Builds the frozen replay command and exposes the moved module."""

    def build_command(
        self,
        paths: PipelinePaths,
        recording: Path,
        *,
        udp_host: str = "127.0.0.1",
        udp_port: int = 15150,
        fps: float = 50.0,
    ):
        return build_replay_sender_command(
            paths,
            recording,
            udp_host=udp_host,
            udp_port=udp_port,
            fps=fps,
        )

    @property
    def legacy_module(self):
        """Back-compat alias: this module IS the implementation now."""
        import sys
        return sys.modules[__name__]


if __name__ == "__main__":
    raise SystemExit(main())
