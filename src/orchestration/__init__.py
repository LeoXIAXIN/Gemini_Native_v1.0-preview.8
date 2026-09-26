"""Orchestration layer: process lifecycle specs and legacy adapter facades.

The frozen ProcessSpec contract lives in ``src/domain/process_spec.py`` (pure)
and is re-exported through ``src/orchestration.process_spec`` for the
historical import surface; runtime path resolution lives in
``src.adapters.runtime.default_pipeline_paths``.  Supervisors consume the
same frozen builders, so command assembly stays centralized in one place
(CODEX rule: interpreter, script path, module path, cwd, environment and
ready markers live in ProcessSpec — never scattered across new modules).
"""

from src.orchestration.process_manager import ProcessManagerFacade
from src.orchestration.process_spec import (
    GMR_READY_MARKER,
    HGPT_READY_MARKER,
    PREVIEW_READY_MARKER,
    STATE_READY_MARKER,
    MODULE_BOOTSTRAP,
    SCRIPT_BOOTSTRAP,
    PipelinePaths,
    ProcessSpec,
    build_chingmu_sender_command,
    build_gmr_bridge_command,
    build_hgpt_sim_command,
    build_play_track_command,
    build_preview_pipeline_command,
    build_real_pipeline_command,
    build_replay_sender_command,
    build_sim_pipeline_command,
    build_state_store_command,
    default_pipeline_paths,
)

__all__ = [
    "GMR_READY_MARKER",
    "HGPT_READY_MARKER",
    "MODULE_BOOTSTRAP",
    "PREVIEW_READY_MARKER",
    "SCRIPT_BOOTSTRAP",
    "STATE_READY_MARKER",
    "PipelinePaths",
    "ProcessManagerFacade",
    "ProcessSpec",
    "build_chingmu_sender_command",
    "build_gmr_bridge_command",
    "build_hgpt_sim_command",
    "build_play_track_command",
    "build_preview_pipeline_command",
    "build_real_pipeline_command",
    "build_replay_sender_command",
    "build_sim_pipeline_command",
    "build_state_store_command",
    "default_pipeline_paths",
]
