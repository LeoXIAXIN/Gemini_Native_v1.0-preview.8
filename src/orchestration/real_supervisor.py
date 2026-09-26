"""Supervise native-Windows CHINGMU -> GMR -> Humanoid-GPT -> Unitree G1.

Debug mode subscribes to LowState and never constructs a LowCmd publisher.
Non-debug mode preserves the Alpha5 three-stage web/launcher/runner
authorization chain before the Humanoid-GPT real controller can import and
initialize Unitree DDS.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import os
from pathlib import Path
import signal
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

from src.orchestration.sim_supervisor import (
    GMR_READY_MARKER,
    HGPT_READY_MARKER,
    STATE_READY_MARKER,
    _pump_output,
    _python_script_command,
    _resp_command,
    _start_child,
    _stop_child,
    _wait_for_event,
    _wait_for_state_prefix,
)
from src.adapters.runtime import PipelinePaths
from src.domain.g1_versions import DEFAULT_MODEL_PROFILE, SUPPORTED_MODEL_PROFILES


PROJECT_ROOT = Path(__file__).resolve().parents[2]


MODULE_BOOTSTRAP = (
    "import runpy,sys;"
    "sys.path[:0]=[sys.argv[1],sys.argv[2]];"
    "module=sys.argv[3];"
    "sys.argv=[module,*sys.argv[4:]];"
    "runpy.run_module(module,run_name='__main__')"
)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gemini Native Humanoid-GPT Unitree G1 supervisor."
    )
    parser.add_argument("--workspace-dir", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--gmr-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--hgpt-python", type=Path, required=True)
    parser.add_argument("--hgpt-repo", type=Path, required=True)
    parser.add_argument("--dll-path", type=Path, required=True)
    parser.add_argument("--server-ip", default="127.0.0.1")
    parser.add_argument("--human-id", type=int, default=0)
    parser.add_argument("--human-height", type=float, default=1.75)
    parser.add_argument("--publish-fps", type=float, default=50.0)
    parser.add_argument("--transition-seconds", type=float, default=1.5)
    parser.add_argument("--model-profile", choices=SUPPORTED_MODEL_PROFILES, default=DEFAULT_MODEL_PROFILE)
    parser.add_argument("--unitree-interface-address", required=True)
    parser.add_argument(
        "--execution-mode",
        choices=["real_only", "parallel"],
        default="real_only",
    )
    parser.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Receive-only LowState mode; never creates a LowCmd publisher.",
    )
    parser.add_argument("--udp-host", default="127.0.0.1")
    parser.add_argument("--udp-port", type=int, default=15150)
    parser.add_argument("--state-host", default="127.0.0.1")
    parser.add_argument("--state-port", type=int, default=6379)
    parser.add_argument("--ready-timeout", type=float, default=120.0)
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Automated-check duration; zero runs until stopped.",
    )
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser


def _module_command(
    python: Path,
    workspace_dir: Path,
    repo: Path,
    module: str,
    *arguments: str,
) -> list[str]:
    return [
        str(python),
        "-u",
        "-c",
        MODULE_BOOTSTRAP,
        str(PROJECT_ROOT),
        str(repo),
        module,
        *arguments,
    ]


def _validate_args(
    args: argparse.Namespace,
) -> PipelinePaths:
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
    dll_path = args.dll_path.expanduser().resolve()
    paths = PipelinePaths(
        workspace_dir=workspace_dir,
        gmr_python=gmr_python,
        hgpt_python=hgpt_python,
        hgpt_repo=hgpt_repo,
        dll_path=dll_path,
    )
    required = [
        gmr_python,
        hgpt_python,
        dll_path,
        paths.real_authorization_script,
        paths.state_store_script,
        paths.gmr_bridge_script,
        paths.chingmu_sender_script,
        paths.play_track_script,
        hgpt_repo / "storage" / "ckpts" / "pns_wo_priv216.onnx",
        hgpt_repo
        / "storage"
        / "ckpts"
        / "G1-Walk"
        / "07140632_G1-Walk_v2.0.0_baseline.onnx",
    ]
    missing = [path for path in required if not path.is_file()]
    if not hgpt_repo.is_dir():
        missing.append(hgpt_repo)
    if missing:
        raise FileNotFoundError(
            f"Required Gemini Native real-runtime paths are missing: {missing}"
        )
    return paths


def _consume_launcher_authorization(
    args: argparse.Namespace,
    environment: dict[str, str],
) -> None:
    from src.application.authorization import (
        ENV_TOKEN,
        consume_authorization_from_environment,
    )

    next_token = consume_authorization_from_environment(
        "launcher",
        expected={
            "backend": "humanoid_gpt",
            "execution_mode": args.execution_mode,
            "debug_mode": False,
            "unitree_network_interface": args.unitree_interface_address,
            "model_profile": args.model_profile,
        },
        environment=environment,
    )
    if not next_token:
        raise RuntimeError("launcher authorization did not produce a session token")
    environment[ENV_TOKEN] = next_token
    print(
        "Launcher ARM G1 authorization consumed; LowCmd remains closed.",
        flush=True,
    )


def _prepare_runner_authorization(environment: dict[str, str]) -> None:
    from src.application.authorization import (
        ENV_TOKEN,
        consume_authorization_from_environment,
    )

    next_token = consume_authorization_from_environment(
        "prepare_runner",
        expected={"backend": "humanoid_gpt", "debug_mode": False},
        environment=environment,
    )
    if not next_token:
        raise RuntimeError("runner authorization did not produce a final token")
    environment[ENV_TOKEN] = next_token


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        paths = _validate_args(args)
    except (OSError, RuntimeError, ValueError) as exc:
        print(
            f"Gemini Native real preflight failed: {type(exc).__name__}: {exc}",
            flush=True,
        )
        return 2

    workspace_dir = paths.workspace_dir
    gmr_python = paths.gmr_python
    hgpt_python = paths.hgpt_python
    hgpt_repo = paths.hgpt_repo

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
    sim_ready = threading.Event()
    exit_code = 0
    started = time.monotonic()

    try:
        if not args.debug:
            _consume_launcher_authorization(args, environment)

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
        children.append(("GMR reference bridge", gmr))
        threading.Thread(
            target=_pump_output,
            args=(gmr, "gmr", {GMR_READY_MARKER: gmr_ready}),
            daemon=True,
        ).start()
        _wait_for_event(
            gmr_ready,
            gmr,
            label="GMR reference bridge",
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

        capture = _start_child(
            _python_script_command(
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
            ),
            cwd=workspace_dir,
            environment=environment,
        )
        children.append(("CHINGMU Windows sender", capture))
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

        if args.execution_mode == "parallel":
            simulation = _start_child(
                _python_script_command(
                    hgpt_python,
                    workspace_dir,
                    paths.hgpt_sim_script,
                    "--repo",
                    str(hgpt_repo),
                    "--redis-host",
                    args.state_host,
                    "--redis-port",
                    str(args.state_port),
                    "--frequency",
                    "50",
                    "--transition-seconds",
                    str(args.transition_seconds),
                    "--device",
                    "cpu",
                    "--no-startup-handover",
                    "--no-safety-strategy-enabled",
                    "--headless" if args.headless else "--no-headless",
                ),
                cwd=hgpt_repo,
                environment=environment,
            )
            children.append(("parallel Humanoid-GPT MuJoCo", simulation))
            threading.Thread(
                target=_pump_output,
                args=(simulation, "sim", {HGPT_READY_MARKER: sim_ready}),
                daemon=True,
            ).start()
            _wait_for_event(
                sim_ready,
                simulation,
                label="parallel Humanoid-GPT MuJoCo",
                timeout=args.ready_timeout,
                stop_requested=stop_requested,
            )

        if not args.debug:
            _prepare_runner_authorization(environment)

        real_args = [
            "--real",
            "--net",
            args.unitree_interface_address,
            "--mocap-type",
            "chingmu_redis",
            "--redis-host",
            args.state_host,
            "--redis-port",
            str(args.state_port),
            "--redis-key",
            "action_qpos_g1_packet",
            "--human-height",
            str(args.human_height),
            "--model-profile",
            args.model_profile,
            "--no-visualize-retarget",
            "--startup-handover",
        ]
        if args.debug:
            real_args.append("--debug")
        real = _start_child(
            _module_command(
                hgpt_python,
                workspace_dir,
                hgpt_repo,
                paths.play_track_module,
                *real_args,
            ),
            cwd=hgpt_repo,
            environment=environment,
        )
        # Keep this child last: reversed cleanup stops the real controller
        # first, allowing its finite damping/close sequence to complete before
        # simulation, capture, GMR, or the local store are stopped.
        children.append(("Unitree G1 real controller", real))
        threading.Thread(
            target=_pump_output,
            args=(real, "unitree", {}),
            daemon=True,
        ).start()
        print(
            "waiting_for_unitree_lowstate "
            + (
                "(read-only; LowCmd disabled)"
                if args.debug
                else (
                    "(ARM G1 accepted; pre-entered Develop and physical START "
                    "gates still required)"
                )
            ),
            flush=True,
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
            f"Gemini Native Unitree pipeline failed: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )
        exit_code = 3
    finally:
        for label, process in reversed(children):
            _stop_child(process, label)
        print("Chingmu Gemini Native Unitree pipeline stopped.", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
