#!/usr/bin/env python3
"""Chingmu Gemini Native v1.0 Preview Windows control service."""

from __future__ import annotations

import importlib.util
from http import HTTPStatus
import ipaddress
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
import urllib.parse
import webbrowser
from typing import Any

# Direct-script bootstrap: make the src package importable when this file is
# executed as a script (sys.path[0] is src/controller).
if str(Path(__file__).resolve().parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.application.authorization import (
    ENV_CAPABILITY as REAL_AUTH_CAPABILITY_ENV,
    RealOutputAuthorizationBroker,
    binding_from_control_config,
)
from src.adapters.runtime import HGPT_POLICY_NAME, default_pipeline_paths
from src.domain.g1_versions import (
    DEFAULT_MODEL_PROFILE,
    model_profile_from_config,
)
from src.domain.process_spec import (
    build_preview_pipeline_command,
    build_real_pipeline_command,
    build_sim_pipeline_command,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
GEMINI_NATIVE_UI_DIR = PACKAGE_ROOT / "src" / "controller" / "static"


def _load_gemini_base() -> Any:
    # M10b: the base controller is the migrated src implementation; the
    # module-level attribute overrides below keep the same semantics as the
    # legacy dynamic overlay load.
    from src.controller import motion_control_app as base_module

    return base_module


base = _load_gemini_base()
base.SERVICE_NAME = "chingmu-gemini-native"
base.STATIC_DIR = GEMINI_NATIVE_UI_DIR
base.CONFIG_PATH = PACKAGE_ROOT / "config" / "gemini_native_config.json"
base.LOG_DIR = PACKAGE_ROOT / "storage" / "logs"

NATIVE_DLL_PATH = Path(sys.executable).resolve().parent / "ChingmuDLL" / "CMVrpn.dll"
NATIVE_HGPT_REPO = PACKAGE_ROOT
NATIVE_LOWSTATE_REPORT = (
    PACKAGE_ROOT / "storage" / "reports" / "unitree_lowstate_windows_report.json"
)
NATIVE_DEVELOP_PROBE = (
    PACKAGE_ROOT / "src" / "adapters" / "unitree_develop_probe_windows.py"
)
NATIVE_HGPT_PYTHON = Path(
    os.environ.get(
        "CHINGMU_GEMINI_NATIVE_HGPT_PYTHON",
        str(PACKAGE_ROOT / "runtime-hgpt" / "python.exe"),
    )
).resolve()
if not NATIVE_HGPT_PYTHON.is_file():
    development_runtime = PACKAGE_ROOT / "tmp" / "windows_hgpt_runtime_312" / "python.exe"
    if development_runtime.is_file():
        NATIVE_HGPT_PYTHON = development_runtime


def _can_bind_loopback_tcp(port: int) -> tuple[bool, str]:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind(("127.0.0.1", port))
    except OSError as exc:
        return False, f"127.0.0.1:{port} is already in use: {exc}"
    finally:
        probe.close()
    return True, f"127.0.0.1:{port} is available"


def _can_bind_loopback_udp(port: int) -> tuple[bool, str]:
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.bind(("127.0.0.1", port))
    except OSError as exc:
        return False, f"UDP 127.0.0.1:{port} is already in use: {exc}"
    finally:
        probe.close()
    return True, f"UDP 127.0.0.1:{port} is available"


def _validate_unitree_windows_address(value: Any) -> str:
    try:
        address = ipaddress.ip_address(str(value).strip())
    except ValueError as exc:
        raise base.ConfigError(
            "unitree_network_interface must be the Windows adapter IPv4 address"
        ) from exc
    if not isinstance(address, ipaddress.IPv4Address):
        raise base.ConfigError("Unitree Windows DDS currently requires IPv4")
    if (
        address.is_loopback
        or address.is_unspecified
        or address.is_multicast
        or address.is_link_local
    ):
        raise base.ConfigError(
            "select the non-loopback Windows Ethernet IPv4 connected to G1"
        )
    return str(address)


class GeminiNativePipelineManager(base.PipelineManager):
    """Gemini v1 UI contract backed by native Windows simulation processes."""

    def __init__(self) -> None:
        base.DEFAULT_CONFIG.update(
            {
                "backend": "humanoid_gpt",
                "execution_mode": "simulation_only",
                "debug_mode": True,
                "server_ip": "127.0.0.1",
                "device": "cpu",
                "model_profile": DEFAULT_MODEL_PROFILE,
                "unitree_network_interface": "192.168.123.100",
            }
        )
        super().__init__()
        migrated = dict(self.config)
        if migrated.get("backend") not in {"humanoid_gpt", "gmr_preview"}:
            migrated["backend"] = "humanoid_gpt"
        if migrated.get("execution_mode") not in {
            "simulation_only",
            "real_only",
            "parallel",
        }:
            migrated["execution_mode"] = "simulation_only"
        if (
            migrated["backend"] == "gmr_preview"
            and migrated["execution_mode"] != "simulation_only"
        ):
            migrated["execution_mode"] = "simulation_only"
        migrated["device"] = "cpu"
        migrated["model_profile"] = model_profile_from_config(migrated)
        migrated.pop("g1_version", None)
        if migrated.get("unitree_network_interface") in {
            None,
            "",
            "eth0",
            "lo",
        }:
            migrated["unitree_network_interface"] = "192.168.123.100"
        if migrated != self.config:
            base.persist_config(migrated)
            self.config = migrated
            self._append_log(
                "Migrated saved settings to the Gemini Native Windows scope",
                component="service",
            )
        self.real_authorization_broker = RealOutputAuthorizationBroker(
            runtime_directory=PACKAGE_ROOT / "storage" / "auth"
        )
        self.real_authorization_capability: str | None = None

    def config_payload(self) -> dict[str, Any]:
        payload = super().config_payload()
        payload["runtime"] = "windows_native"
        payload["product_name"] = "Chingmu Gemini Native v1.0 Preview"
        payload["options"].update(
            {
                "backends": ["humanoid_gpt", "gmr_preview"],
                "execution_modes": [
                    "simulation_only",
                    "real_only",
                    "parallel",
                ],
                "devices": ["cpu"],
            }
        )
        payload["capabilities"] = {
            "windows_web_ui": True,
            "chingmu_windows_capture": True,
            "gmr": True,
            "gmr_preview": True,
            "humanoid_gpt_simulation": True,
            "mujoco_windows": True,
            "wsl_required": False,
            "unitree_lowstate": True,
            "unitree_lowcmd": True,
            "unitree_lowcmd_experimental": True,
        }
        return payload

    def update_config(self, update: Any) -> dict[str, Any]:
        if isinstance(update, dict) and {"model_profile", "g1_version"} & update.keys():
            raise base.ConfigError("model_profile is internal deployment configuration")
        with self.lock:
            if (
                self.phase in base.BUSY_PHASES
                or (self.process is not None and self.process.poll() is None)
            ):
                raise RuntimeError(
                    "Stop the current Gemini Native task before changing settings"
                )
            validated = base.validate_config(update, self.config)
            if validated["backend"] not in {"humanoid_gpt", "gmr_preview"}:
                raise base.ConfigError(
                    "Gemini Native supports real-time generative teleoperation "
                    "or GMR preview"
                )
            if (
                validated["backend"] == "gmr_preview"
                and validated["execution_mode"] != "simulation_only"
            ):
                raise base.ConfigError("GMR preview supports simulation only")
            if validated["execution_mode"] in {"real_only", "parallel"}:
                validated["unitree_network_interface"] = (
                    _validate_unitree_windows_address(
                        validated["unitree_network_interface"]
                    )
                )
            validated["device"] = "cpu"
            base.persist_config(validated)
            self.config = validated
            self.error = None
            self.message = "Configuration saved"
        self._append_log("Configuration saved", component="service")
        return self.config_payload()

    def preflight(self, config: dict[str, Any] | None = None) -> dict[str, Any]:
        with self.lock:
            checked = dict(self.config if config is None else config)
            process_alive = self.process is not None and self.process.poll() is None

        checks: list[dict[str, Any]] = []

        def add(name: str, ok: bool, message: str, required: bool = True) -> None:
            checks.append(
                {"name": name, "ok": bool(ok), "message": message, "required": required}
            )

        license_status = base.check_product_license()
        add("product_license", license_status.allowed, license_status.message)
        add(
            "platform",
            os.name == "nt",
            "Native Windows runtime detected"
            if os.name == "nt"
            else "Gemini Native requires Windows",
        )
        add("session", not process_alive, "No managed Native pipeline is active")
        backend = str(checked.get("backend", ""))
        execution_mode = str(checked.get("execution_mode", ""))
        valid_selection = (
            backend == "gmr_preview" and execution_mode == "simulation_only"
        ) or (
            backend == "humanoid_gpt"
            and execution_mode in {"simulation_only", "real_only", "parallel"}
        )
        add(
            "windows_path",
            valid_selection,
            (
                f"Gemini Native path selected: {backend}/{execution_mode}"
                if valid_selection
                else "Select GMR preview simulation or the generative "
                "teleoperation simulation/real path"
            ),
        )
        if backend == "humanoid_gpt":
            add(
                "cpu_policy",
                checked.get("device") in {"auto", "cpu"},
                "Humanoid-GPT ONNX policy will run on CPU"
                if checked.get("device") in {"auto", "cpu"}
                else "Gemini Native currently packages the CPU policy runtime",
            )

        paths = default_pipeline_paths()
        required_paths = {
            "native_ui": GEMINI_NATIVE_UI_DIR / "index.html",
            "capture": paths.chingmu_sender_script,
        }
        if backend == "gmr_preview":
            required_paths.update(
                {
                    "preview_pipeline": paths.preview_pipeline_script,
                    "gmr_preview": paths.gmr_preview_exact_script,
                }
            )
        elif backend == "humanoid_gpt":
            required_paths.update(
                {
                    "gmr_bridge": paths.gmr_bridge_script,
                    "local_state": paths.state_store_script,
                    "hgpt_python": paths.hgpt_python,
                    "hgpt_policy": (
                        paths.hgpt_repo
                        / "storage"
                        / "ckpts"
                        / HGPT_POLICY_NAME
                    ),
                    "g1_asset": (
                        paths.hgpt_repo
                        / "storage"
                        / "assets"
                        / model_profile_from_config(checked)
                        / "scene_mjx_track.xml"
                    ),
                }
            )
            if execution_mode == "simulation_only":
                required_paths.update(
                    {
                        "native_sim_pipeline": paths.sim_pipeline_script,
                        "hgpt_sim_runner": paths.hgpt_sim_script,
                    }
                )
            else:
                required_paths.update(
                    {
                        "native_real_pipeline": paths.real_pipeline_script,
                        "real_authorization": paths.real_authorization_script,
                        "hgpt_real_runner": paths.play_track_script,
                        "windows_unitree_config": paths.windows_unitree_script,
                        "unitree_develop_probe": NATIVE_DEVELOP_PROBE,
                    }
                )
        for name, path in required_paths.items():
            add(name, path.is_file(), f"{name}: {path}")

        if backend == "gmr_preview":
            udp_ok, udp_message = _can_bind_loopback_udp(15150)
            add("preview_udp_port", udp_ok, udp_message)
        else:
            state_port_ok, state_port_message = _can_bind_loopback_tcp(6379)
            add("local_state_port", state_port_ok, state_port_message)

        add(
            "chingmu_dll",
            NATIVE_DLL_PATH.is_file(),
            (
                f"Official CHINGMU Windows DLL found: {NATIVE_DLL_PATH}"
                if NATIVE_DLL_PATH.is_file()
                else f"CHINGMU Windows DLL is missing: {NATIVE_DLL_PATH}"
            ),
        )
        if NATIVE_DLL_PATH.is_file() and os.name == "nt":
            try:
                from src.adapters.chingmu import load_sdk

                load_sdk(NATIVE_DLL_PATH)
                sdk_ok = True
                sdk_message = "CHINGMU Windows SDK symbols resolved"
            except Exception as exc:
                sdk_ok = False
                sdk_message = f"CHINGMU Windows SDK load failed: {exc}"
            add("chingmu_sdk_load", sdk_ok, sdk_message)

        try:
            import mujoco
            import qpsolvers
            from src.adapters.gmr_headless import (
                load_general_motion_retargeting,
            )

            GeneralMotionRetargeting = load_general_motion_retargeting()
            retargeter = GeneralMotionRetargeting(
                src_human="bvh_nokov",
                tgt_robot="unitree_g1",
                actual_human_height=float(checked["human_height"]),
                verbose=False,
            )
            gmr_ok = (
                retargeter.model.nq == 36 and "daqp" in qpsolvers.available_solvers
            )
            gmr_message = (
                f"Windows headless GMR + MuJoCo {mujoco.__version__} + DAQP "
                "are ready (viewer is loaded only when simulation starts)"
            )
        except Exception as exc:
            gmr_ok = False
            gmr_message = f"Windows GMR runtime failed: {type(exc).__name__}: {exc}"
        add("gmr_runtime", gmr_ok, gmr_message)

        if backend == "humanoid_gpt":
            hgpt_ok = False
            hgpt_message = (
                f"Humanoid-GPT runtime is missing: {NATIVE_HGPT_PYTHON}"
            )
            if NATIVE_HGPT_PYTHON.is_file() and NATIVE_HGPT_REPO.is_dir():
                probe = (
                    "import sys;"
                    f"sys.path.insert(0, {str(Path(__file__).resolve().parents[2])!r});"
                    f"sys.path.insert(0, {str(NATIVE_HGPT_REPO)!r});"
                    "import jax,flax,mujoco,onnxruntime,pygame,redis,tyro;"
                    "from loop_rate_limiters import RateLimiter;"
                    "from src.motion.infer_utils import G1TrackMjSim;"
                    "print(sys.version.split()[0],mujoco.__version__,"
                    "onnxruntime.__version__)"
                )
                try:
                    result = subprocess.run(
                        [str(NATIVE_HGPT_PYTHON), "-c", probe],
                        cwd=str(NATIVE_HGPT_REPO),
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        timeout=45.0,
                        check=False,
                        env=base.build_launch_environment(checked),
                    )
                    hgpt_ok = result.returncode == 0
                    hgpt_message = (
                        "Humanoid-GPT Windows runtime ready: "
                        f"{result.stdout.strip()}"
                        if hgpt_ok
                        else "Humanoid-GPT import probe failed: "
                        f"{result.stdout.strip()[-600:]}"
                    )
                except (OSError, subprocess.SubprocessError) as exc:
                    hgpt_message = f"Humanoid-GPT import probe failed: {exc}"
            add("humanoid_gpt_runtime", hgpt_ok, hgpt_message)

        if backend == "humanoid_gpt" and execution_mode in {
            "real_only",
            "parallel",
        }:
            try:
                interface_address = _validate_unitree_windows_address(
                    checked.get("unitree_network_interface")
                )
                address_ok = True
                address_message = (
                    "Unitree Windows adapter IPv4 is valid: "
                    f"{interface_address}"
                )
            except base.ConfigError as exc:
                interface_address = ""
                address_ok = False
                address_message = str(exc)
            add("unitree_adapter_address", address_ok, address_message)

            adapter_present = False
            if address_ok:
                try:
                    local_ipv4 = {
                        item[4][0]
                        for item in socket.getaddrinfo(
                            socket.gethostname(),
                            None,
                            socket.AF_INET,
                        )
                    }
                    adapter_present = interface_address in local_ipv4
                except Exception as exc:
                    address_message = f"Could not enumerate Windows adapters: {exc}"
            add(
                "unitree_adapter_present",
                adapter_present,
                (
                    f"Windows adapter IPv4 is assigned: {interface_address}"
                    if adapter_present
                    else address_message
                    if not address_ok or address_message.startswith("Could not")
                    else "The configured Unitree IPv4 is not assigned to an "
                    "active Windows adapter"
                ),
            )

            unitree_probe = (
                "import sys;"
                f"sys.path.insert(0,{str(PACKAGE_ROOT)!r});"
                "from unitree_sdk2py.core.channel import "
                "ChannelPublisher,ChannelSubscriber;"
                "from unitree_sdk2py.idl.default import "
                "unitree_hg_msg_dds__LowCmd_,unitree_hg_msg_dds__LowState_;"
                "from unitree_sdk2py.idl.unitree_hg.msg.dds_ import "
                "LowCmd_,LowState_;"
                "from unitree_sdk2py.utils.crc import CRC;"
                "c=unitree_hg_msg_dds__LowCmd_();"
                "s=unitree_hg_msg_dds__LowState_();"
                "assert isinstance(c,LowCmd_) and isinstance(s,LowState_);"
                "assert len(c.motor_cmd)==35 and len(s.motor_state)==35;"
                "print(CRC().Crc(c),LowCmd_.__idl_typename__,"
                "LowState_.__idl_typename__)"
            )
            try:
                result = subprocess.run(
                    [str(NATIVE_HGPT_PYTHON), "-c", unitree_probe],
                    cwd=str(NATIVE_HGPT_REPO),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=30.0,
                    check=False,
                )
                unitree_ok = result.returncode == 0
                unitree_message = (
                    "Official Unitree SDK2 HG LowState/LowCmd types "
                    "and CRC are ready: "
                    f"{result.stdout.strip()}"
                    if unitree_ok
                    else "Windows Unitree runtime import failed: "
                    f"{result.stdout.strip()[-600:]}"
                )
            except (OSError, subprocess.SubprocessError) as exc:
                unitree_ok = False
                unitree_message = f"Windows Unitree runtime import failed: {exc}"
            add("unitree_windows_runtime", unitree_ok, unitree_message)

            develop_ok = False
            if unitree_ok and adapter_present and NATIVE_DEVELOP_PROBE.is_file():
                try:
                    result = subprocess.run(
                        [
                            str(NATIVE_HGPT_PYTHON),
                            "-u",
                            str(NATIVE_DEVELOP_PROBE),
                            "--interface-address",
                            interface_address,
                        ],
                        cwd=str(PACKAGE_ROOT),
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        timeout=10.0,
                        check=False,
                    )
                    develop_ok = result.returncode == 0
                    develop_message = result.stdout.strip()[-900:]
                except (OSError, subprocess.SubprocessError) as exc:
                    develop_message = f"Develop preflight failed: {exc}"
            elif not unitree_ok:
                develop_message = "Unitree runtime must pass before Develop check"
            elif not adapter_present:
                develop_message = "Unitree adapter must be active before Develop check"
            else:
                develop_message = f"Develop probe is missing: {NATIVE_DEVELOP_PROBE}"
            add("unitree_develop_mode", develop_ok, develop_message)

            if checked.get("debug_mode", True):
                add(
                    "lowstate_acceptance",
                    True,
                    "Read-only debug mode will validate live LowState and "
                    "cannot create a LowCmd publisher",
                )
            elif address_ok:
                report_ok, report_message = (
                    base.verify_unitree_lowstate_probe_report(
                        NATIVE_LOWSTATE_REPORT,
                        interface=interface_address,
                    )
                )
                add(
                    "lowstate_acceptance",
                    report_ok,
                    report_message,
                )

        errors = [
            item["message"]
            for item in checks
            if item["required"] and not item["ok"]
        ]
        return {"ok": not errors, "checks": checks, "errors": errors}

    def start(self, authorization: str | None = None) -> tuple[int, dict[str, Any]]:
        with self.lock:
            if (
                self.phase in base.BUSY_PHASES
                or (self.process is not None and self.process.poll() is None)
            ):
                return HTTPStatus.CONFLICT, self._status_with_result(
                    False, "A Gemini Native task is already active"
                )
            config = dict(self.config)
            if (
                config["execution_mode"] in {"real_only", "parallel"}
                and not config["debug_mode"]
                and authorization != base.REAL_OUTPUT_AUTHORIZATION
            ):
                return HTTPStatus.PRECONDITION_REQUIRED, self._status_with_result(
                    False,
                    "Real LowCmd output requires a fresh ARM G1 "
                    "authorization for this start request",
                )
            self.phase = "preflight"
            self.message = "Checking Chingmu Gemini Native runtime"
            self.error = None

        preflight = self.preflight(config)
        if not preflight["ok"]:
            with self.lock:
                self.phase = "error"
                self.error = "; ".join(preflight["errors"])
                self.message = "Gemini Native environment check failed"
            self._append_log(self.error, component="service", level="error")
            payload = self._status_with_result(False, self.message)
            payload["preflight"] = preflight
            return HTTPStatus.PRECONDITION_FAILED, payload

        paths = default_pipeline_paths()
        if config["backend"] == "gmr_preview":
            command = list(build_preview_pipeline_command(paths, config).command)
            launch_message = "Starting Gemini Native GMR preview"
        elif config["execution_mode"] == "simulation_only":
            command = list(build_sim_pipeline_command(paths, config).command)
            launch_message = (
                "Starting Gemini Native real-time generative simulation"
            )
        else:
            command = list(build_real_pipeline_command(paths, config).command)
            launch_message = (
                "Starting Gemini Native Unitree read-only check"
                if config["debug_mode"]
                else "Starting authorized Gemini Native Unitree control"
            )
        environment = dict(os.environ)
        environment.update({"PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"})
        issued_capability: str | None = None
        real_write_authorized = (
            config["execution_mode"] in {"real_only", "parallel"}
            and not config["debug_mode"]
        )
        if real_write_authorized:
            try:
                authorization_config = dict(config)
                authorization_config["startup_handover_enabled"] = True
                authorization_environment = self.real_authorization_broker.issue(
                    binding_from_control_config(authorization_config)
                )
                environment.update(authorization_environment)
                issued_capability = authorization_environment[
                    REAL_AUTH_CAPABILITY_ENV
                ]
            except Exception as exc:
                with self.lock:
                    self.phase = "error"
                    self.error = (
                        "Could not create the one-shot Windows real-output "
                        f"authorization: {exc}"
                    )
                    self.message = "Real-output authorization failed"
                self._append_log(
                    self.error, component="service", level="error"
                )
                return (
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    self._status_with_result(False, self.message),
                )

        with self.lock:
            if self.phase != "preflight":
                self.real_authorization_broker.revoke(issued_capability)
                return HTTPStatus.CONFLICT, self._status_with_result(
                    False, "Start was cancelled"
                )
            self.phase = "starting"
            self.message = launch_message
            self._new_session_log(config["backend"], config)
            try:
                process = subprocess.Popen(
                    command,
                    cwd=str(PACKAGE_ROOT),
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    creationflags=getattr(
                        subprocess, "CREATE_NEW_PROCESS_GROUP", 0
                    ),
                )
            except Exception as exc:
                self.real_authorization_broker.revoke(issued_capability)
                self.phase = "error"
                self.error = f"Could not start Gemini Native pipeline: {exc}"
                self.message = "Launch failed"
                self._append_log(self.error, component="service", level="error")
                self._close_log()
                return HTTPStatus.INTERNAL_SERVER_ERROR, self._status_with_result(
                    False, self.message
                )
            self.process = process
            self.real_authorization_capability = issued_capability
            self.backend = config["backend"]
            self.started_epoch = time.time()
            self.started_monotonic = time.monotonic()
            self.exit_code = None
            self.metrics = base._empty_metrics()
            self.last_input_monotonic = None
            self.session_error_hint = None
            self.monitor_thread = threading.Thread(
                target=self._monitor_process,
                args=(process,),
                name="gemini-native-pipeline-monitor",
                daemon=True,
            )
            monitor = self.monitor_thread
        self._append_log(
            f"Started Chingmu Gemini Native pipeline {process.pid}",
            component="service",
        )
        monitor.start()
        return HTTPStatus.ACCEPTED, self._status_with_result(True, "Start accepted")

    def _monitor_process(self, process: subprocess.Popen[str]) -> None:
        try:
            super()._monitor_process(process)
        finally:
            with self.lock:
                capability = self.real_authorization_capability
                self.real_authorization_capability = None
            self.real_authorization_broker.revoke(capability)

    def _stop_worker(self, process: subprocess.Popen[str], force: bool) -> None:
        if process.poll() is not None:
            return
        graceful_timeout = 1.0 if force else 10.0
        ctrl_break = getattr(signal, "CTRL_BREAK_EVENT", None)
        if ctrl_break is not None:
            try:
                process.send_signal(ctrl_break)
            except (OSError, ProcessLookupError):
                pass
        if self._wait_for_exit(process, graceful_timeout):
            return
        self._append_log(
            "Native supervisor did not stop after CTRL+BREAK; terminating it",
            component="service",
            level="warning",
        )
        try:
            process.terminate()
        except (OSError, ProcessLookupError):
            pass
        if not self._wait_for_exit(process, 3.0):
            try:
                process.kill()
            except (OSError, ProcessLookupError):
                pass
            self._wait_for_exit(process, 2.0)

    def cleanup(self) -> tuple[int, dict[str, Any]]:
        with self.lock:
            process = self.process
            if process is not None and process.poll() is None:
                return HTTPStatus.CONFLICT, self._status_with_result(
                    False, "A managed Native session is active; stop it first"
                )
            self.phase = "stopped"
            self.error = None
            self.message = "No managed Gemini Native task is active"
        payload = self._status_with_result(True, self.message)
        payload["cleanup"] = {
            "matched": 0,
            "remaining": 0,
            "frame_transport_ready": True,
        }
        return HTTPStatus.OK, payload

    def shutdown(self) -> None:
        self.request_stop(force=False, source="Gemini Native service shutdown")
        with self.lock:
            thread = self.stop_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=12.0)
        with self.lock:
            process = self.process
        if process is not None and process.poll() is None:
            try:
                process.kill()
            except (OSError, ProcessLookupError):
                pass
            self._wait_for_exit(process, 2.0)
        self.real_authorization_broker.close()
        self._close_log()


class GeminiNativeRequestHandler(base.ControlRequestHandler):
    server_version = "ChingmuGeminiNative/1.0-Preview"

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = urllib.parse.urlsplit(self.path).path
        if path == "/api/health":
            self._send_json(
                HTTPStatus.OK,
                {
                    "service": "chingmu-gemini-native",
                    "product": "Chingmu Gemini Native v1.0 Preview",
                    "ok": True,
                    "runtime": "windows_native",
                    "wsl_required": False,
                    "phase": self.manager.status(log_limit=0)["phase"],
                    "product_license": base.product_license_payload(),
                },
            )
            return
        super().do_GET()


def main(argv: list[str] | None = None) -> int:
    if os.name != "nt":
        raise RuntimeError("Chingmu Gemini Native requires native Windows")
    if not GEMINI_NATIVE_UI_DIR.is_dir():
        raise FileNotFoundError(f"Gemini Native UI is missing: {GEMINI_NATIVE_UI_DIR}")

    args = base.parse_args(argv)
    base.DEFAULT_CONFIG.update(
        {
            "backend": "humanoid_gpt",
            "execution_mode": "simulation_only",
            "debug_mode": True,
            "safety_strategy_enabled": False,
            "server_ip": "127.0.0.1",
            "device": "cpu",
            "model_profile": DEFAULT_MODEL_PROFILE,
            "unitree_network_interface": "192.168.123.100",
        }
    )
    manager = GeminiNativePipelineManager()
    server = base.LocalControlHTTPServer(
        (args.host, args.port), GeminiNativeRequestHandler, manager
    )
    url = f"http://{args.host}:{args.port}/"
    stopping = threading.Event()

    def request_server_shutdown(signum: int, _frame: Any) -> None:
        if stopping.is_set():
            return
        stopping.set()
        manager._append_log(
            f"Gemini Native UI received signal {signum}",
            component="service",
            level="warning",
        )
        threading.Thread(target=server.shutdown, daemon=True).start()

    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        signum = getattr(signal, name, None)
        if signum is not None:
            signal.signal(signum, request_server_shutdown)

    if args.open_browser:
        threading.Timer(0.7, lambda: webbrowser.open(url)).start()

    print(f"chingmu-gemini-native listening on {url}", flush=True)
    print(
        "Chingmu Gemini Native v1.0 Preview: no WSL; GMR preview and "
        "Humanoid-GPT simulation are available. Unitree real output remains "
        "behind LowState acceptance, ARM G1, and physical-remote gates.",
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        manager.shutdown()
        print("chingmu-gemini-native stopped.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
