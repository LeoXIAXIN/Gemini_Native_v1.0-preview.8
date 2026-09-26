"""Windows-native CHINGMU receiver that forwards raw frames to local GMR.

This is the Windows counterpart of ``chingmu_vrpn_udp_sender.py``.  It uses
CHINGMU's official x64 ``CMVrpn.dll`` and deliberately stays standard-library
only so the vendor DLL remains isolated from GMR, ONNX Runtime and Torch native
runtimes.
"""

from __future__ import annotations

import argparse
import ctypes
import faulthandler
import os
import pickle
import struct
import time
from pathlib import Path
from typing import Any, Callable, Sequence

# Direct-script bootstrap: the supervisors launch this file as a script
# (M17 src-only deployment); make the src package importable first.
import sys as _sys

if str(Path(__file__).resolve().parents[2]) not in _sys.path:
    _sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.adapters.frame_transport import (
    add_frame_transport_arguments,
    open_frame_sender,
)

MAX_SEGMENTS = 150
SEGMENT_COUNT = 63
EXPECTED_PE_MACHINE_X64 = 0x8664
EXPECTED_PE32_PLUS_MAGIC = 0x020B
DEFAULT_SDK_ROOT = (
    Path(__file__).resolve().parent / "official_chingmu_python_windows"
)
FRAME_STALE_WARNING_SECONDS = 0.75
FRAME_STALE_REPEAT_SECONDS = 5.0


class FrameFreshnessWatchdog:
    """Report an upstream SDK stall without confusing it with local UDP loss."""

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        warning_seconds: float = FRAME_STALE_WARNING_SECONDS,
        repeat_seconds: float = FRAME_STALE_REPEAT_SECONDS,
    ) -> None:
        if warning_seconds <= 0.0 or repeat_seconds <= 0.0:
            raise ValueError("frame watchdog intervals must be positive")
        self._clock = clock
        self._warning_seconds = float(warning_seconds)
        self._repeat_seconds = float(repeat_seconds)
        self._last_frame_at = float(clock())
        self._next_warning_at = self._last_frame_at + self._warning_seconds
        self._warning_active = False

    def poll(self) -> float | None:
        """Return the current stall age only when a warning should be logged."""

        now = float(self._clock())
        if now < self._next_warning_at:
            return None
        self._warning_active = True
        self._next_warning_at = now + self._repeat_seconds
        return max(0.0, now - self._last_frame_at)

    def frame_received(self) -> float | None:
        """Record a new frame and return the recovered stall duration, if any."""

        now = float(self._clock())
        recovered_age = (
            max(0.0, now - self._last_frame_at)
            if self._warning_active
            else None
        )
        self._last_frame_at = now
        self._next_warning_at = now + self._warning_seconds
        self._warning_active = False
        return recovered_age


def inspect_pe_architecture(path: Path) -> tuple[int, int]:
    """Return the PE machine and optional-header magic without extra packages."""

    with path.open("rb") as handle:
        header = handle.read(4096)
    if len(header) < 64 or header[:2] != b"MZ":
        raise RuntimeError(f"CHINGMU runtime is not a Windows PE file: {path}")
    pe_offset = struct.unpack_from("<I", header, 0x3C)[0]
    required = pe_offset + 26
    if required > len(header):
        with path.open("rb") as handle:
            header = handle.read(required)
    if len(header) < required or header[pe_offset : pe_offset + 4] != b"PE\0\0":
        raise RuntimeError(f"CHINGMU runtime has an invalid PE header: {path}")
    machine = struct.unpack_from("<H", header, pe_offset + 4)[0]
    optional_magic = struct.unpack_from("<H", header, pe_offset + 24)[0]
    return machine, optional_magic


def validate_windows_dll(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"CHINGMU Windows SDK is missing: {path}. "
            "Place the official ChingmuDLL/CMVrpn.dll at that path or pass "
            "--dll-path."
        )
    machine, optional_magic = inspect_pe_architecture(path)
    if (
        machine != EXPECTED_PE_MACHINE_X64
        or optional_magic != EXPECTED_PE32_PLUS_MAGIC
    ):
        raise RuntimeError(
            "CMVrpn.dll must be the official Windows x64 build "
            f"(machine=0x{machine:04X}, format=0x{optional_magic:04X})"
        )
    return path


def bind_sdk(library: Any) -> Any:
    """Bind the lifecycle and supported human-frame entry points."""

    library.CMVrpnStartExtern.argtypes = []
    library.CMVrpnStartExtern.restype = None
    library.CMVrpnQuitExtern.argtypes = []
    library.CMVrpnQuitExtern.restype = None
    library.CMHumanGlobalTLocalRTC.argtypes = [
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int),
    ]
    library.CMHumanGlobalTLocalRTC.restype = ctypes.c_int
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
    return library


def load_sdk(
    path: Path,
    *,
    loader: Callable[[str], Any] = ctypes.CDLL,
) -> Any:
    """Load and bind the official Windows SDK from an explicit safe path."""

    path = validate_windows_dll(path)
    add_directory = getattr(os, "add_dll_directory", None)
    if add_directory is None:
        return bind_sdk(loader(str(path)))
    with add_directory(str(path.parent)):
        return bind_sdk(loader(str(path)))


def build_frame_packet(
    frame_id: int,
    positions: Sequence[float],
    rotations: Sequence[float],
    detected: Sequence[int],
    *,
    segment_count: int = SEGMENT_COUNT,
) -> bytes:
    """Build the byte-for-byte-compatible tuple consumed by the GMR bridge."""

    pos = [
        list(positions[index * 3 : index * 3 + 3])
        for index in range(segment_count)
    ]
    quat = [
        list(rotations[index * 4 : index * 4 + 4])
        for index in range(segment_count)
    ]
    seen = list(detected[:segment_count])
    return pickle.dumps((int(frame_id), pos, quat, seen), protocol=5)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Receive CHINGMU skeleton frames through the Windows SDK."
    )
    parser.add_argument("--server-ip", default="127.0.0.1")
    parser.add_argument("--human-id", type=int, default=0)
    parser.add_argument(
        "--human-api",
        choices=("global", "retarget"),
        default="global",
        help=(
            "CHINGMU human-frame API. Native Windows defaults to "
            "CMHumanGlobalTLocalRTC because live CMRetargetHumanExternTC "
            "can report detected segments while returning zero rotations."
        ),
    )
    parser.add_argument("--sdk-root", type=Path, default=DEFAULT_SDK_ROOT)
    parser.add_argument(
        "--dll-path",
        type=Path,
        help="Explicit path to the official x64 CMVrpn.dll.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate and load the DLL without starting its receive thread.",
    )
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=0.0,
        help="Optional bounded capture duration; zero runs until Ctrl+C.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional frame limit; zero runs until Ctrl+C.",
    )
    add_frame_transport_arguments(parser, default_transport="udp")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    faulthandler.enable(all_threads=True)
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    if os.name != "nt":
        parser.error(
            "This sender requires native Windows. Use "
            "chingmu_vrpn_udp_sender.py under Linux/WSL."
        )
    if args.frame_transport != "udp":
        parser.error("The Windows sender currently supports local UDP only.")
    if args.duration_seconds < 0:
        parser.error("--duration-seconds must be zero or positive.")
    if args.max_frames < 0:
        parser.error("--max-frames must be zero or positive.")

    dll_path = args.dll_path or args.sdk_root / "ChingmuDLL" / "CMVrpn.dll"
    dll_path = validate_windows_dll(dll_path)
    machine, optional_magic = inspect_pe_architecture(dll_path)
    print(f"[SDK 1/3] Windows x64 runtime: {dll_path}", flush=True)
    print(
        f"[SDK 2/3] PE machine=0x{machine:04X}, format=0x{optional_magic:04X}",
        flush=True,
    )
    library = load_sdk(dll_path)
    print("[SDK 3/3] Required entry points resolved", flush=True)
    if args.check_only:
        print("CHINGMU Windows SDK load check: PASS", flush=True)
        return 0

    host = f"MCAvatar@{args.server_ip}".encode("gbk")
    positions = (ctypes.c_double * (MAX_SEGMENTS * 3))()
    rotations = (ctypes.c_double * (MAX_SEGMENTS * 4))()
    detected = (ctypes.c_int * MAX_SEGMENTS)()
    timecode = (ctypes.c_int * 1)()
    frame_sender = open_frame_sender(
        args.frame_transport,
        args.frame_socket,
        args.udp_host,
        args.udp_port,
    )

    interval_started = time.monotonic()
    capture_started = interval_started
    deadline = (
        capture_started + args.duration_seconds
        if args.duration_seconds > 0
        else None
    )
    sent_interval = 0
    sent_total = 0
    dropped = 0
    last_timecode: int | None = None
    call_index = 0
    started = False
    freshness = FrameFreshnessWatchdog()
    print(f"Starting official CHINGMU Windows SDK for {host!r}...", flush=True)
    print(
        "CHINGMU human API: "
        + (
            "CMHumanGlobalTLocalRTC"
            if args.human_api == "global"
            else "CMRetargetHumanExternTC"
        ),
        flush=True,
    )
    try:
        library.CMVrpnStartExtern()
        started = True
        print(
            f"CHINGMU -> {frame_sender.endpoint}; Ctrl+C stops capture.",
            flush=True,
        )
        while True:
            if deadline is not None and time.monotonic() >= deadline:
                break
            if args.human_api == "global":
                ok = library.CMHumanGlobalTLocalRTC(
                    host,
                    args.human_id,
                    timecode,
                    positions,
                    rotations,
                    detected,
                )
            else:
                ok = library.CMRetargetHumanExternTC(
                    host,
                    args.human_id,
                    call_index,
                    timecode,
                    positions,
                    rotations,
                    detected,
                )
            if not ok:
                stale_age = freshness.poll()
                if stale_age is not None:
                    print(
                        "WARNING: CHINGMU SDK produced no valid frame for "
                        f"{stale_age:.2f}s; this is upstream of local UDP. "
                        "Check MCAvatar solving and the selected Skeleton ID.",
                        flush=True,
                    )
                time.sleep(0.005)
                continue
            frame_id = int(timecode[0])
            if frame_id == last_timecode:
                stale_age = freshness.poll()
                if stale_age is not None:
                    print(
                        "WARNING: CHINGMU SDK timecode has not advanced for "
                        f"{stale_age:.2f}s; this is upstream of local UDP. "
                        "Check MCAvatar solving and the selected Skeleton ID.",
                        flush=True,
                    )
                time.sleep(0.001)
                continue
            recovered_age = freshness.frame_received()
            if recovered_age is not None:
                print(
                    "CHINGMU SDK frame stream recovered after "
                    f"{recovered_age:.2f}s without a new frame.",
                    flush=True,
                )
            last_timecode = frame_id
            call_index += 1
            packet = build_frame_packet(
                frame_id, positions, rotations, detected
            )
            if frame_sender.send(packet):
                sent_interval += 1
                sent_total += 1
            else:
                dropped += 1
            if args.max_frames > 0 and sent_total >= args.max_frames:
                break
            now = time.monotonic()
            elapsed = now - interval_started
            if elapsed >= 0.5:
                print(
                    f"Sender FPS: {sent_interval / elapsed:.1f}, "
                    f"frame: {frame_id}, local drops: {dropped}",
                    flush=True,
                )
                sent_interval = 0
                dropped = 0
                interval_started = now
    except KeyboardInterrupt:
        pass
    finally:
        if started:
            library.CMVrpnQuitExtern()
        frame_sender.close()
        print("CHINGMU Windows sender stopped.", flush=True)
    return 0


class ChingmuAdapter:
    """Transport wrapper for the official CHINGMU Windows SDK (same module)."""

    MAX_SEGMENTS = MAX_SEGMENTS
    SEGMENT_COUNT = SEGMENT_COUNT

    def inspect_pe_architecture(self, path: Path) -> tuple[int, int]:
        return inspect_pe_architecture(path)

    def validate_windows_dll(self, path: Path) -> Path:
        return validate_windows_dll(path)

    def load_sdk(
        self, path: Path, *, loader: Callable[[str], Any] | None = None
    ) -> Any:
        if loader is None:
            return load_sdk(path)
        return load_sdk(path, loader=loader)

    def build_frame_packet(
        self,
        frame_id: int,
        positions: Sequence[float],
        rotations: Sequence[float],
        detected: Sequence[int],
        *,
        segment_count: int = SEGMENT_COUNT,
    ) -> bytes:
        return build_frame_packet(
            frame_id,
            positions,
            rotations,
            detected,
            segment_count=segment_count,
        )


if __name__ == "__main__":
    raise SystemExit(main())
