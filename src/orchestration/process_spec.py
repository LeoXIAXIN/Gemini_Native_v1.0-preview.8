"""Compatibility re-export of the frozen process launch contract.

M-decoupling: the canonical pure spec lives in ``src/domain/process_spec.py``
so application services can import it without depending on orchestration.
This module keeps the historical import surface identical (same names, same
objects); the concrete layout (``PipelinePaths``) and runtime resolution
(``default_pipeline_paths``) are defined in ``src.adapters.runtime`` and
re-exported here.
"""

from __future__ import annotations

from src.domain.process_spec import (
    GMR_READY_MARKER,
    HGPT_READY_MARKER,
    MODULE_BOOTSTRAP,
    PREVIEW_READY_MARKER,
    PREVIEW_PIPELINE_STOP_ORDER,
    REAL_PIPELINE_STOP_ORDER,
    SCRIPT_BOOTSTRAP,
    SIM_PIPELINE_STOP_ORDER,
    STATE_READY_MARKER,
    ProcessSpec,
    _environment_update,
    build_chingmu_sender_command,
    build_gmr_bridge_command,
    build_hgpt_sim_command,
    build_play_track_command,
    build_preview_pipeline_command,
    build_real_pipeline_command,
    build_replay_sender_command,
    build_sim_pipeline_command,
    build_state_store_command,
)
from src.adapters.runtime import (
    G1_WALK_NAME,
    HGPT_POLICY_NAME,
    PipelinePaths,
    default_pipeline_paths,
)

__all__ = [
    "G1_WALK_NAME",
    "GMR_READY_MARKER",
    "HGPT_POLICY_NAME",
    "HGPT_READY_MARKER",
    "MODULE_BOOTSTRAP",
    "PREVIEW_READY_MARKER",
    "PREVIEW_PIPELINE_STOP_ORDER",
    "REAL_PIPELINE_STOP_ORDER",
    "SCRIPT_BOOTSTRAP",
    "SIM_PIPELINE_STOP_ORDER",
    "STATE_READY_MARKER",
    "PipelinePaths",
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
