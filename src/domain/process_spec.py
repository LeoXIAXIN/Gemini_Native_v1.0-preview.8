"""Frozen process launch contract: ProcessSpec and the command builders.

M-decoupling: this pure contract moved here from
``src/orchestration/process_spec.py`` so application services can depend on
domain instead of orchestration (no application -> orchestration edge).
``src/orchestration/process_spec.py`` re-exports every name below for
compatibility.

Layering rule (wrap-up audit): domain defines ONLY the process contract —
``ProcessSpec``, the frozen markers/bootstraps/stop orders and the pure
command builders parameterized by a ``PipelineLayout``.  It contains NO
concrete deployment layout (no pipeline-script names, repository folders or
runtime discovery).  The concrete layout lives in
``src.adapters.runtime.PipelinePaths`` and is resolved by
``src.adapters.runtime.default_pipeline_paths``.

These builders are pure functions: they never spawn a process or read/write
configuration.  They define the single launch contract consumed by the src
controller and supervisors.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

# Frozen ready markers (match the legacy supervisors verbatim).
STATE_READY_MARKER = "Gemini Native local state store ready"
GMR_READY_MARKER = "GMR receiver ready on"
HGPT_READY_MARKER = "Humanoid-GPT live MuJoCo backend ready"
PREVIEW_READY_MARKER = "GMR exact preview ready on"

# Frozen runpy bootstraps (match the legacy supervisors verbatim).
SCRIPT_BOOTSTRAP = (
    "import runpy,sys;"
    "sys.path.insert(0,sys.argv[1]);"
    "sys.argv=sys.argv[2:];"
    "runpy.run_path(sys.argv[0],run_name='__main__')"
)
MODULE_BOOTSTRAP = (
    "import runpy,sys;"
    "sys.path[:0]=[sys.argv[1],sys.argv[2]];"
    "module=sys.argv[3];"
    "sys.argv=[module,*sys.argv[4:]];"
    "runpy.run_module(module,run_name='__main__')"
)


class PipelineLayout(Protocol):
    """The resolved launch layout a command builder consumes.

    Domain only declares the NAMES it needs; the concrete deployment layout
    (where each script lives, which module name and model files the legacy
    chain uses) is resolved by the adapters layer
    (``src.adapters.runtime.PipelinePaths``).
    """

    workspace_dir: Path
    gmr_python: Path
    hgpt_python: Path
    hgpt_repo: Path
    dll_path: Path
    package_root: Path
    sim_pipeline_script: Path
    real_pipeline_script: Path
    preview_pipeline_script: Path
    state_store_script: Path
    gmr_bridge_script: Path
    chingmu_sender_script: Path
    replay_sender_script: Path
    hgpt_sim_script: Path
    play_track_module: str


@dataclass(frozen=True)
class ProcessSpec:
    """One launchable command plus the readiness/stop contract.

    ``command`` is the exact argv list the legacy code builds; ``cwd`` and
    ``environment`` mirror the legacy Popen call; ``ready_markers`` are the
    case-insensitive substrings the legacy supervisor waits for.
    """

    name: str
    command: tuple[str, ...]
    cwd: Path
    environment: Mapping[str, str]
    ready_markers: tuple[str, ...] = ()


def _environment_update(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """Mirror the fixed environment the legacy supervisors add."""
    import os

    environment = dict(os.environ)
    environment.update({"PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"})
    if extra:
        environment.update({str(k): str(v) for k, v in extra.items()})
    return environment


# --------------------------------------------------------------------- #
# Child commands (mirror the legacy supervisors exactly)                #
# --------------------------------------------------------------------- #

def build_state_store_command(
    paths: PipelineLayout, *, host: str = "127.0.0.1", port: int = 6379
) -> ProcessSpec:
    return ProcessSpec(
        name="local state store",
        command=(
            str(paths.gmr_python),
            "-u",
            str(paths.state_store_script),
            "--host",
            str(host),
            "--port",
            str(port),
        ),
        cwd=paths.workspace_dir,
        environment=_environment_update(),
        ready_markers=(STATE_READY_MARKER,),
    )


def build_gmr_bridge_command(
    paths: PipelineLayout,
    *,
    udp_host: str = "127.0.0.1",
    udp_port: int = 15150,
    human_height: float = 1.75,
    publish_fps: float = 50.0,
    state_host: str = "127.0.0.1",
    state_port: int = 6379,
    transition_seconds: float = 1.5,
) -> ProcessSpec:
    return ProcessSpec(
        name="GMR bridge",
        command=(
            str(paths.gmr_python),
            "-u",
            "-c",
            SCRIPT_BOOTSTRAP,
            str(paths.package_root),
            str(paths.gmr_bridge_script),
            "--frame-transport",
            "udp",
            "--udp-host",
            str(udp_host),
            "--udp-port",
            str(udp_port),
            "--human-height",
            str(human_height),
            "--publish-fps",
            str(publish_fps),
            "--redis-host",
            str(state_host),
            "--redis-port",
            str(state_port),
            "--transition-seconds",
            str(transition_seconds),
        ),
        cwd=paths.workspace_dir,
        environment=_environment_update(),
        ready_markers=(GMR_READY_MARKER,),
    )


def build_chingmu_sender_command(
    paths: PipelineLayout,
    *,
    server_ip: str = "127.0.0.1",
    human_id: int = 0,
    udp_host: str = "127.0.0.1",
    udp_port: int = 15150,
) -> ProcessSpec:
    return ProcessSpec(
        name="CHINGMU Windows sender",
        command=(
            str(paths.gmr_python),
            "-u",
            "-c",
            SCRIPT_BOOTSTRAP,
            str(paths.package_root),
            str(paths.chingmu_sender_script),
            "--dll-path",
            str(paths.dll_path),
            "--server-ip",
            str(server_ip),
            "--human-id",
            str(human_id),
            "--human-api",
            "global",
            "--frame-transport",
            "udp",
            "--udp-host",
            str(udp_host),
            "--udp-port",
            str(udp_port),
        ),
        cwd=paths.workspace_dir,
        environment=_environment_update(),
    )


def build_replay_sender_command(
    paths: PipelineLayout,
    recording: Path,
    *,
    udp_host: str = "127.0.0.1",
    udp_port: int = 15150,
    fps: float = 50.0,
) -> ProcessSpec:
    return ProcessSpec(
        name="recorded CHINGMU replay",
        command=(
            str(paths.gmr_python),
            "-u",
            "-c",
            SCRIPT_BOOTSTRAP,
            str(paths.package_root),
            str(paths.replay_sender_script),
            "--recording",
            str(recording.expanduser().resolve()),
            "--udp-host",
            str(udp_host),
            "--udp-port",
            str(udp_port),
            "--fps",
            str(fps),
        ),
        cwd=paths.workspace_dir,
        environment=_environment_update(),
    )


def build_hgpt_sim_command(
    paths: PipelineLayout,
    *,
    state_host: str = "127.0.0.1",
    state_port: int = 6379,
    frequency: int = 50,
    transition_seconds: float = 1.5,
    device: str = "cpu",
    startup_handover: bool = True,
    safety_strategy_enabled: bool = False,
    headless: bool = False,
) -> ProcessSpec:
    return ProcessSpec(
        name="Humanoid-GPT MuJoCo",
        command=(
            str(paths.hgpt_python),
            "-u",
            "-c",
            SCRIPT_BOOTSTRAP,
            str(paths.package_root),
            str(paths.hgpt_sim_script),
            "--repo",
            str(paths.hgpt_repo),
            "--redis-host",
            str(state_host),
            "--redis-port",
            str(state_port),
            "--frequency",
            str(frequency),
            "--transition-seconds",
            str(transition_seconds),
            "--device",
            "cpu" if device == "auto" else str(device),
            "--startup-handover" if startup_handover else "--no-startup-handover",
            (
                "--safety-strategy-enabled"
                if safety_strategy_enabled
                else "--no-safety-strategy-enabled"
            ),
            "--headless" if headless else "--no-headless",
        ),
        cwd=paths.hgpt_repo,
        environment=_environment_update(),
        ready_markers=(HGPT_READY_MARKER,),
    )


def build_play_track_command(
    paths: PipelineLayout,
    *,
    unitree_interface_address: str,
    human_height: float = 1.75,
    model_profile: str = "unitree_g1_5010",
    debug: bool = True,
    state_host: str = "127.0.0.1",
    state_port: int = 6379,
) -> ProcessSpec:
    real_args = [
        "--real",
        "--net",
        str(unitree_interface_address),
        "--mocap-type",
        "chingmu_redis",
        "--redis-host",
        str(state_host),
        "--redis-port",
        str(state_port),
        "--redis-key",
        "action_qpos_g1_packet",
        "--human-height",
        str(human_height),
        "--model-profile",
        str(model_profile),
        "--no-visualize-retarget",
        # The legacy supervisor passes --startup-handover unconditionally.
        "--startup-handover",
    ]
    if debug:
        real_args.append("--debug")
    return ProcessSpec(
        name="Unitree G1 real controller",
        command=(
            str(paths.hgpt_python),
            "-u",
            "-c",
            MODULE_BOOTSTRAP,
            str(paths.package_root),
            str(paths.hgpt_repo),
            paths.play_track_module,
            *real_args,
        ),
        cwd=paths.hgpt_repo,
        environment=_environment_update({"G1_MODEL_PROFILE": str(model_profile)}),
    )


# --------------------------------------------------------------------- #
# Supervisor commands (mirror the legacy controller's start() exactly)  #
# --------------------------------------------------------------------- #

def build_preview_pipeline_command(
    paths: PipelineLayout, config: Mapping[str, Any]
) -> ProcessSpec:
    return ProcessSpec(
        name="GMR preview supervisor",
        command=(
            str(paths.gmr_python),
            "-u",
            str(paths.preview_pipeline_script),
            "--workspace-dir",
            str(paths.workspace_dir),
            "--dll-path",
            str(paths.dll_path),
            "--server-ip",
            str(config["server_ip"]),
            "--human-id",
            str(config["skeleton_id"]),
            "--human-height",
            str(config["human_height"]),
            "--viewer-fps",
            str(config["publish_fps"]),
        ),
        cwd=paths.workspace_dir,
        environment=_environment_update(),
        ready_markers=(PREVIEW_READY_MARKER,),
    )


def build_sim_pipeline_command(
    paths: PipelineLayout, config: Mapping[str, Any]
) -> ProcessSpec:
    return ProcessSpec(
        name="Humanoid-GPT simulation supervisor",
        command=(
            str(paths.gmr_python),
            "-u",
            str(paths.sim_pipeline_script),
            "--workspace-dir",
            str(paths.workspace_dir),
            "--gmr-python",
            str(paths.gmr_python),
            "--hgpt-python",
            str(paths.hgpt_python),
            "--hgpt-repo",
            str(paths.hgpt_repo),
            "--dll-path",
            str(paths.dll_path),
            "--server-ip",
            str(config["server_ip"]),
            "--human-id",
            str(config["skeleton_id"]),
            "--human-height",
            str(config["human_height"]),
            "--publish-fps",
            str(config["publish_fps"]),
            "--transition-seconds",
            str(config["transition_seconds"]),
            "--device",
            "cpu",
            "--model-profile",
            str(config["model_profile"]),
            "--no-startup-handover",
            (
                "--safety-strategy-enabled"
                if config.get("safety_strategy_enabled", False)
                else "--no-safety-strategy-enabled"
            ),
        ),
        cwd=paths.workspace_dir,
        environment=_environment_update(),
    )


def build_real_pipeline_command(
    paths: PipelineLayout, config: Mapping[str, Any]
) -> ProcessSpec:
    return ProcessSpec(
        name="Unitree real supervisor",
        command=(
            str(paths.gmr_python),
            "-u",
            str(paths.real_pipeline_script),
            "--workspace-dir",
            str(paths.workspace_dir),
            "--gmr-python",
            str(paths.gmr_python),
            "--hgpt-python",
            str(paths.hgpt_python),
            "--hgpt-repo",
            str(paths.hgpt_repo),
            "--dll-path",
            str(paths.dll_path),
            "--server-ip",
            str(config["server_ip"]),
            "--human-id",
            str(config["skeleton_id"]),
            "--human-height",
            str(config["human_height"]),
            "--publish-fps",
            str(config["publish_fps"]),
            "--transition-seconds",
            str(config["transition_seconds"]),
            "--model-profile",
            str(config["model_profile"]),
            "--unitree-interface-address",
            str(config["unitree_network_interface"]),
            "--execution-mode",
            str(config["execution_mode"]),
            "--debug" if config["debug_mode"] else "--no-debug",
        ),
        cwd=paths.workspace_dir,
        environment=_environment_update(),
    )


# Frozen supervisor stop order (reversed children, frozen in Phase 0 §7.4).
SIM_PIPELINE_STOP_ORDER = (
    "Humanoid-GPT MuJoCo",
    "CHINGMU Windows sender",
    "recorded CHINGMU replay",
    "GMR bridge",
    "local state store",
)
REAL_PIPELINE_STOP_ORDER = (
    "Unitree G1 real controller",
    "parallel Humanoid-GPT MuJoCo",
    "CHINGMU Windows sender",
    "GMR bridge",
    "local state store",
)
PREVIEW_PIPELINE_STOP_ORDER = (
    "CHINGMU Windows sender",
    "GMR/MuJoCo preview",
)
