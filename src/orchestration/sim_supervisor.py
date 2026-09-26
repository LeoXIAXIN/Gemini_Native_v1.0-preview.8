"""Supervise CHINGMU -> GMR -> Humanoid-GPT -> MuJoCo on native Windows."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time

# Direct-script bootstrap: the native controller launches this file directly
# (M17 src-only deployment); make the src package importable first.
import sys as _sys
from pathlib import Path as _Path

if str(_Path(__file__).resolve().parents[2]) not in _sys.path:
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))


PROJECT_ROOT = _Path(__file__).resolve().parents[2]

from src.adapters.runtime import PipelinePaths
from src.domain.g1_versions import DEFAULT_MODEL_PROFILE, SUPPORTED_MODEL_PROFILES


STATE_READY_MARKER = "Gemini Native local state store ready"
GMR_READY_MARKER = "GMR receiver ready on"
HGPT_READY_MARKER = "Humanoid-GPT live MuJoCo backend ready"
SCRIPT_BOOTSTRAP = (
    "import runpy,sys;"
    "sys.path.insert(0,sys.argv[1]);"
    "sys.argv=sys.argv[2:];"
    "runpy.run_path(sys.argv[0],run_name='__main__')"
)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Windows-native Gemini Humanoid-GPT simulation supervisor."
    )
    parser.add_argument("--workspace-dir", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--gmr-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--hgpt-python", type=Path, required=True)
    parser.add_argument("--hgpt-repo", type=Path, required=True)
    capture = parser.add_mutually_exclusive_group(required=True)
    capture.add_argument("--dll-path", type=Path)
    capture.add_argument("--replay-recording", type=Path)
    parser.add_argument("--server-ip", default="127.0.0.1")
    parser.add_argument("--human-id", type=int, default=0)
    parser.add_argument("--human-height", type=float, default=1.75)
    parser.add_argument("--publish-fps", type=float, default=50.0)
    parser.add_argument("--transition-seconds", type=float, default=1.5)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--model-profile", choices=SUPPORTED_MODEL_PROFILES, default=DEFAULT_MODEL_PROFILE)
    parser.add_argument("--udp-host", default="127.0.0.1")
    parser.add_argument("--udp-port", type=int, default=15150)
    parser.add_argument("--state-host", default="127.0.0.1")
    parser.add_argument("--state-port", type=int, default=6379)
    parser.add_argument(
        "--startup-handover",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--safety-strategy-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--ready-timeout", type=float, default=120.0)
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Automated-check duration; zero runs until stopped.",
    )
    return parser


def _start_child(
    command: list[str], *, cwd: Path, environment: dict[str, str]
) -> subprocess.Popen[str]:
    return subprocess.Popen(
        command,
        cwd=str(cwd),
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
    )


def _python_script_command(
    python: Path, workspace_dir: Path, script: Path, *arguments: str
) -> list[str]:
    """Run a src script from an isolated embedded-Python ``._pth``."""

    return [
        str(python),
        "-u",
        "-c",
        SCRIPT_BOOTSTRAP,
        str(PROJECT_ROOT),
        str(script),
        *arguments,
    ]


def _pump_output(
    process: subprocess.Popen[str],
    label: str,
    markers: dict[str, threading.Event],
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
            lowered = clean.lower()
            for marker, event in markers.items():
                if marker.lower() in lowered:
                    event.set()
    finally:
        process.stdout.close()


def _stop_child(process: subprocess.Popen[str] | None, label: str) -> None:
    if process is None or process.poll() is not None:
        return
    print(f"Stopping {label}...", flush=True)
    ctrl_break = getattr(signal, "CTRL_BREAK_EVENT", None)
    if ctrl_break is not None:
        try:
            process.send_signal(ctrl_break)
        except OSError:
            pass
    try:
        process.wait(timeout=5.0)
        return
    except subprocess.TimeoutExpired:
        pass
    process.terminate()
    try:
        process.wait(timeout=4.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2.0)


def _resp_command(host: str, port: int, *items: str) -> bytes | None:
    chunks = [f"*{len(items)}\r\n".encode("ascii")]
    for item in items:
        encoded = item.encode("utf-8")
        chunks.extend(
            [f"${len(encoded)}\r\n".encode("ascii"), encoded, b"\r\n"]
        )
    with socket.create_connection((host, port), timeout=0.35) as client:
        client.settimeout(0.35)
        client.sendall(b"".join(chunks))
        stream = client.makefile("rb")
        first = stream.readline()
        if first.startswith(b"+"):
            return first[1:-2]
        if first == b"$-1\r\n":
            return None
        if first.startswith(b"$"):
            size = int(first[1:-2])
            payload = stream.read(size)
            if stream.read(2) != b"\r\n":
                raise RuntimeError("truncated local state response")
            return payload
        if first.startswith(b"-"):
            raise RuntimeError(first[1:-2].decode("utf-8", errors="replace"))
        raise RuntimeError("invalid local state response")


def _wait_for_event(
    event: threading.Event,
    process: subprocess.Popen[str],
    *,
    label: str,
    timeout: float,
    stop_requested: threading.Event,
) -> None:
    deadline = time.monotonic() + timeout
    while not event.is_set():
        if stop_requested.is_set():
            raise KeyboardInterrupt
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(f"{label} exited before readiness (code {return_code})")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"{label} did not become ready in time")
        event.wait(0.1)


def _wait_for_state_prefix(
    host: str,
    port: int,
    key: str,
    prefix: str,
    *,
    process: subprocess.Popen[str],
    timeout: float,
    stop_requested: threading.Event,
) -> str:
    deadline = time.monotonic() + timeout
    last_value = ""
    while True:
        if stop_requested.is_set():
            raise KeyboardInterrupt
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(
                f"process exited while waiting for {key} (code {return_code})"
            )
        try:
            value = _resp_command(host, port, "GET", key)
            last_value = value.decode("utf-8", errors="replace") if value else ""
        except OSError:
            last_value = ""
        if last_value.startswith(prefix):
            return last_value
        if last_value.startswith("error:"):
            raise RuntimeError(f"{key} reported {last_value}")
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"timed out waiting for {key}={prefix}*; last value={last_value!r}"
            )
        stop_requested.wait(0.1)


def _validate_args(args: argparse.Namespace) -> PipelinePaths:
    if os.name != "nt":
        raise RuntimeError("This supervisor must run in native Windows Python.")
    if args.human_id < 0:
        raise ValueError("--human-id must be zero or positive")
    if args.human_height <= 0 or args.publish_fps <= 0:
        raise ValueError("--human-height and --publish-fps must be positive")
    if args.duration < 0:
        raise ValueError("--duration must be zero or positive")
    for port in (args.udp_port, args.state_port):
        if not 1 <= port <= 65535:
            raise ValueError("ports must be between 1 and 65535")

    workspace_dir = args.workspace_dir.expanduser().resolve()
    gmr_python = args.gmr_python.expanduser().resolve()
    hgpt_python = args.hgpt_python.expanduser().resolve()
    hgpt_repo = args.hgpt_repo.expanduser().resolve()
    capture_path = (
        args.dll_path.expanduser().resolve()
        if args.dll_path is not None
        else args.replay_recording.expanduser().resolve()
    )
    paths = PipelinePaths(
        workspace_dir=workspace_dir,
        gmr_python=gmr_python,
        hgpt_python=hgpt_python,
        hgpt_repo=hgpt_repo,
        dll_path=capture_path,
    )
    required = [
        gmr_python,
        hgpt_python,
        hgpt_repo / "storage" / "ckpts" / "pns_wo_priv216.onnx",
        paths.state_store_script,
        paths.gmr_bridge_script,
        paths.hgpt_sim_script,
    ]
    required.append(capture_path)
    missing = [path for path in required if not path.is_file()]
    if not hgpt_repo.is_dir():
        missing.append(hgpt_repo)
    if missing:
        raise FileNotFoundError(f"Required native runtime paths are missing: {missing}")
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        paths = _validate_args(args)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Gemini Native preflight failed: {type(exc).__name__}: {exc}", flush=True)
        return 2

    workspace_dir = paths.workspace_dir
    gmr_python = paths.gmr_python
    hgpt_python = paths.hgpt_python

    stop_requested = threading.Event()

    def request_stop(_signum: int, _frame: object) -> None:
        stop_requested.set()

    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        signum = getattr(signal, name, None)
        if signum is not None:
            signal.signal(signum, request_stop)

    environment = dict(os.environ)
    environment.pop("G1_VERSION", None)
    environment.update(
        {
            "PYTHONUTF8": "1",
            "PYTHONUNBUFFERED": "1",
            "G1_MODEL_PROFILE": args.model_profile,
        }
    )
    children: list[tuple[str, subprocess.Popen[str]]] = []
    state_ready = threading.Event()
    gmr_ready = threading.Event()
    hgpt_ready = threading.Event()
    exit_code = 0
    started = time.monotonic()
    try:
        state_store = _start_child(
            [
                str(gmr_python),
                "-u",
                str(paths.state_store_script),
                "--host",
                args.state_host,
                "--port",
                str(args.state_port),
            ],
            cwd=workspace_dir,
            environment=environment,
        )
        children.append(("local state store", state_store))
        threading.Thread(
            target=_pump_output,
            args=(state_store, "state", {STATE_READY_MARKER: state_ready}),
            daemon=True,
        ).start()
        _wait_for_event(
            state_ready,
            state_store,
            label="local state store",
            timeout=args.ready_timeout,
            stop_requested=stop_requested,
        )
        if _resp_command(args.state_host, args.state_port, "PING") != b"PONG":
            raise RuntimeError("local state store PING failed")

        gmr = _start_child(
            _python_script_command(
                gmr_python,
                workspace_dir,
                paths.gmr_bridge_script,
                "--frame-transport",
                "udp",
                "--udp-host",
                args.udp_host,
                "--udp-port",
                str(args.udp_port),
                "--human-height",
                str(args.human_height),
                "--publish-fps",
                str(args.publish_fps),
                "--redis-host",
                args.state_host,
                "--redis-port",
                str(args.state_port),
                "--transition-seconds",
                str(args.transition_seconds),
            ),
            cwd=workspace_dir,
            environment=environment,
        )
        children.append(("GMR bridge", gmr))
        threading.Thread(
            target=_pump_output,
            args=(gmr, "gmr", {GMR_READY_MARKER: gmr_ready}),
            daemon=True,
        ).start()
        _wait_for_event(
            gmr_ready,
            gmr,
            label="GMR bridge",
            timeout=args.ready_timeout,
            stop_requested=stop_requested,
        )
        _wait_for_state_prefix(
            args.state_host,
            args.state_port,
            "chingmu_gmr_bridge_status",
            "waiting_for_local_frame",
            process=gmr,
            timeout=args.ready_timeout,
            stop_requested=stop_requested,
        )
        print("waiting_for_local_frame", flush=True)

        if args.replay_recording is not None:
            capture_command = _python_script_command(
                gmr_python,
                workspace_dir,
                (
                    paths.replay_sender_script
                ),
                "--recording",
                str(args.replay_recording.expanduser().resolve()),
                "--udp-host",
                args.udp_host,
                "--udp-port",
                str(args.udp_port),
                "--fps",
                str(args.publish_fps),
            )
            capture_label = "recorded CHINGMU replay"
        else:
            capture_command = _python_script_command(
                gmr_python,
                workspace_dir,
                paths.chingmu_sender_script,
                "--dll-path",
                str(args.dll_path.expanduser().resolve()),
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
            )
            capture_label = "CHINGMU Windows sender"
        capture = _start_child(
            capture_command, cwd=workspace_dir, environment=environment
        )
        children.append((capture_label, capture))
        threading.Thread(
            target=_pump_output,
            args=(capture, "capture", {}),
            daemon=True,
        ).start()

        live_status = _wait_for_state_prefix(
            args.state_host,
            args.state_port,
            "chingmu_gmr_bridge_status",
            "live:",
            process=gmr,
            timeout=args.ready_timeout,
            stop_requested=stop_requested,
        )
        print(f"First live target received ({live_status}).", flush=True)
        print("Starting Humanoid-GPT released policy...", flush=True)

        hgpt_command = _python_script_command(
            hgpt_python,
            workspace_dir,
            paths.hgpt_sim_script,
            "--repo",
            str(args.hgpt_repo.expanduser().resolve()),
            "--redis-host",
            args.state_host,
            "--redis-port",
            str(args.state_port),
            "--frequency",
            "50",
            "--transition-seconds",
            str(args.transition_seconds),
            "--device",
            "cpu" if args.device == "auto" else args.device,
            (
                "--startup-handover"
                if args.startup_handover
                else "--no-startup-handover"
            ),
            (
                "--safety-strategy-enabled"
                if args.safety_strategy_enabled
                else "--no-safety-strategy-enabled"
            ),
            "--headless" if args.headless else "--no-headless",
        )
        hgpt = _start_child(
            hgpt_command,
            cwd=args.hgpt_repo.expanduser().resolve(),
            environment=environment,
        )
        children.append(("Humanoid-GPT MuJoCo", hgpt))
        threading.Thread(
            target=_pump_output,
            args=(hgpt, "hgpt", {HGPT_READY_MARKER: hgpt_ready}),
            daemon=True,
        ).start()
        _wait_for_event(
            hgpt_ready,
            hgpt,
            label="Humanoid-GPT MuJoCo",
            timeout=args.ready_timeout,
            stop_requested=stop_requested,
        )

        while not stop_requested.is_set():
            if args.duration > 0 and time.monotonic() - started >= args.duration:
                break
            for label, process in children:
                return_code = process.poll()
                if return_code is not None:
                    raise RuntimeError(f"{label} exited with code {return_code}")
            stop_requested.wait(0.2)
    except KeyboardInterrupt:
        pass
    except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
        print(
            f"Gemini Native Humanoid-GPT pipeline failed: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )
        exit_code = 3
    finally:
        for label, process in reversed(children):
            _stop_child(process, label)
        print(
            "Chingmu Gemini Native Humanoid-GPT simulation stopped.",
            flush=True,
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
