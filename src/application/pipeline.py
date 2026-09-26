"""M10 PipelineService: facade over the injected legacy pipeline manager.

The legacy ``PipelineManager`` subclass remains the single owner of process
lifecycle; this service delegates every call to it and centralizes command
assembly through the frozen domain contract (``src/domain/process_spec``).
No orchestration import is needed here: default path resolution is a lazy
adapter call (``src.adapters.runtime.default_pipeline_paths``).
"""

from __future__ import annotations

from typing import Any, Mapping

from src.adapters.runtime import PipelinePaths
from src.domain.process_spec import (
    ProcessSpec,
    build_preview_pipeline_command,
    build_real_pipeline_command,
    build_sim_pipeline_command,
)


def _default_pipeline_paths() -> PipelinePaths:
    """Resolve the fixed runtime paths lazily (adapter concern)."""
    from src.adapters.runtime import default_pipeline_paths

    return default_pipeline_paths()


class PipelineService:
    """Injected-manager facade; never instantiates or persists anything."""

    def __init__(self, manager: Any, paths: PipelinePaths | None = None) -> None:
        self._manager = manager
        self._paths = paths or _default_pipeline_paths()

    @property
    def paths(self) -> PipelinePaths:
        return self._paths

    def build_supervisor_command(self, config: Mapping[str, Any]) -> ProcessSpec:
        """Mirror the legacy controller's backend/mode dispatch exactly."""
        backend = config["backend"]
        mode = config["execution_mode"]
        if backend == "gmr_preview":
            return build_preview_pipeline_command(self._paths, config)
        if mode == "simulation_only":
            return build_sim_pipeline_command(self._paths, config)
        return build_real_pipeline_command(self._paths, config)

    # --- pure delegation to the legacy manager ---------------------------

    def preflight(self, config: dict[str, Any] | None = None) -> Any:
        return self._manager.preflight(config)

    def start(self, authorization: str | None = None) -> Any:
        return self._manager.start(authorization=authorization)

    def request_stop(self, force: bool = False, *, source: str = "internal request") -> Any:
        return self._manager.request_stop(force=force, source=source)

    def cleanup(self) -> Any:
        return self._manager.cleanup()

    def shutdown(self) -> None:
        self._manager.shutdown()

    def emergency_stop(self) -> Any:
        return self._manager.request_emergency_stop()

    def safety_reset(self, payload: Any) -> Any:
        return self._manager.request_safety_reset(payload)

    def status(self, log_limit: int = 200) -> Any:
        return self._manager.status(log_limit=log_limit)

    def config_payload(self) -> Any:
        return self._manager.config_payload()

    def update_config(self, update: Any) -> Any:
        return self._manager.update_config(update)

    def downloadable_log(self) -> Any:
        return self._manager.downloadable_log()
