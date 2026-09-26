"""Run CHINGMU -> GMR -> MuJoCo entirely in native Windows processes."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from typing import Sequence

# Direct-script bootstrap: the native controller launches this file directly
# (M17 src-only deployment); make the src package importable first.
import sys as _sys

if str(Path(__file__).resolve().parents[2]) not in _sys.path:
    _sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

PROJECT_ROOT = Path(__file__).resolve().parents[2]

READY_MARKER = "GMR exact preview ready on"


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Windows-native CHINGMU/GMR/MuJoCo preview supervisor."
    )
    parser.add_argument("--workspace-dir", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--dll-path", type=Path, required=True)
    parser.add_argument("--server-ip", default="127.0.0.1")
    parser.add_argument("--human-id", type=int, default=0)
    parser.add_argument("--human-height", type=float, default=1.75)
    parser.add_argument("--viewer-fps", type=float, default=50.0)
    parser.add_argument("--udp-host", default="127.0.0.1")
    parser.add_argument("--udp-port", type=int, default=15150)
    parser.add_argument("--ready-timeout", type=float, default=90.0)
    return parser


def _start_child(command: list[str], *, cwd: Path) -> subprocess.Popen[str]:
    return subprocess.Popen(
        command,
        cwd=str(cwd),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )


def _pump_output(
    process: subprocess.Popen[str],
    label: str,
    ready: threading.Event | None = None,
) -> None:
    if process.stdout is None:
        return
    try:
        for line in iter(process.stdout.readline, ""):
            clean = line.rstrip("\r\n")
            if clean:
                output = f"[{label}] {clean}"
                console_encoding = sys.stdout.encoding or "utf-8"
                console_safe = output.encode(
                    console_encoding, errors="replace"
                ).decode(console_encoding, errors="replace")
                print(console_safe, flush=True)
            if ready is not None and READY_MARKER.lower() in clean.lower():
                ready.set()
    finally:
        process.stdout.close()


def _stop_child(process: subprocess.Popen[str] | None, label: str) -> None:
    if process is None or process.poll() is not None:
        return
    print(f"Stopping {label}...", flush=True)
    process.terminate()
    try:
        process.wait(timeout=4.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2.0)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    if os.name != "nt":
        parser.error("This supervisor must run in native Windows Python.")
    if args.human_id < 0:
        parser.error("--human-id must be zero or positive.")
    if not 1 <= args.udp_port <= 65535:
        parser.error("--udp-port must be between 1 and 65535.")
    if args.human_height <= 0 or args.viewer_fps <= 0:
        parser.error("--human-height and --viewer-fps must be positive.")

    workspace_dir = args.workspace_dir.expanduser().resolve()
    from src.adapters.runtime import PipelinePaths

    paths = PipelinePaths(
        workspace_dir=workspace_dir,
        gmr_python=Path(sys.executable).resolve(),
        hgpt_python=Path(sys.executable).resolve(),
        hgpt_repo=workspace_dir,
        dll_path=args.dll_path.expanduser().resolve(),
    )
    preview_script = paths.gmr_preview_exact_script
    sender_script = paths.chingmu_sender_script
    dll_path = args.dll_path.expanduser().resolve()
    for required in (preview_script, sender_script, dll_path):
        if not required.is_file():
            raise FileNotFoundError(f"Required Windows runtime file is missing: {required}")

    stop_requested = threading.Event()

    def request_stop(_signum: int, _frame: object) -> None:
        stop_requested.set()

    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        signum = getattr(signal, name, None)
        if signum is not None:
            signal.signal(signum, request_stop)

    preview: subprocess.Popen[str] | None = None
    sender: subprocess.Popen[str] | None = None
    ready = threading.Event()
    exit_code = 0
    try:
        print(
            "Starting Windows-native GMR/MuJoCo receiver on "
            f"udp:{args.udp_host}:{args.udp_port}",
            flush=True,
        )
        preview = _start_child(
            [
                sys.executable,
                "-u",
                str(preview_script),
                "--frame-transport",
                "udp",
                "--udp-host",
                args.udp_host,
                "--udp-port",
                str(args.udp_port),
                "--human-height",
                str(args.human_height),
                "--viewer-fps",
                str(args.viewer_fps),
            ],
            cwd=workspace_dir,
        )
        threading.Thread(
            target=_pump_output,
            args=(preview, "gmr"),
            kwargs={"ready": ready},
            daemon=True,
        ).start()

        deadline = time.monotonic() + max(1.0, args.ready_timeout)
        while not ready.is_set() and not stop_requested.is_set():
            preview_code = preview.poll()
            if preview_code is not None:
                raise RuntimeError(
                    f"GMR/MuJoCo preview exited before readiness (code {preview_code})"
                )
            if time.monotonic() >= deadline:
                raise TimeoutError("GMR/MuJoCo preview did not become ready in time")
            stop_requested.wait(0.1)

        if stop_requested.is_set():
            return 0

        print("GMR receiver ready; starting official CHINGMU Windows SDK.", flush=True)
        sender = _start_child(
            [
                sys.executable,
                "-u",
                str(sender_script),
                "--dll-path",
                str(dll_path),
                "--server-ip",
                args.server_ip,
                "--human-id",
                str(args.human_id),
                "--human-api",
                "global",
                "--frame-transport",
                "udp",
                "--udp-host",
                args.udp_host,
                "--udp-port",
                str(args.udp_port),
            ],
            cwd=workspace_dir,
        )
        threading.Thread(
            target=_pump_output,
            args=(sender, "capture"),
            daemon=True,
        ).start()

        while not stop_requested.is_set():
            preview_code = preview.poll()
            sender_code = sender.poll()
            if preview_code is not None:
                print(
                    f"GMR/MuJoCo preview exited with code {preview_code}.",
                    flush=True,
                )
                exit_code = preview_code or 3
                break
            if sender_code is not None:
                print(
                    f"CHINGMU Windows sender exited with code {sender_code}.",
                    flush=True,
                )
                exit_code = sender_code or 4
                break
            stop_requested.wait(0.2)
    except (OSError, RuntimeError, TimeoutError) as exc:
        print(f"Windows-native preview failed: {type(exc).__name__}: {exc}", flush=True)
        exit_code = 2
    finally:
        _stop_child(sender, "CHINGMU Windows sender")
        _stop_child(preview, "GMR/MuJoCo preview")
        print("Windows-native CHINGMU/GMR/MuJoCo preview stopped.", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
