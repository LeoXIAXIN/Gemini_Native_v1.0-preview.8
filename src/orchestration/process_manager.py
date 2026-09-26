"""ProcessManagerFacade: injected manager and deterministic stop orders."""

from __future__ import annotations

from typing import Any, Mapping

from src.orchestration.process_spec import (
    PREVIEW_PIPELINE_STOP_ORDER,
    REAL_PIPELINE_STOP_ORDER,
    SIM_PIPELINE_STOP_ORDER,
    PipelinePaths,
    ProcessSpec,
    build_preview_pipeline_command,
    build_real_pipeline_command,
    build_sim_pipeline_command,
    default_pipeline_paths,
)


class ProcessManagerFacade:
    """Lifecycle facade; process ownership stays with the injected manager."""

    def __init__(self, manager: Any, paths: PipelinePaths | None = None) -> None:
        self._manager = manager
        self._paths = paths or default_pipeline_paths()

    def launch_spec_for(self, config: Mapping[str, Any]) -> ProcessSpec:
        """Map the validated backend/mode pair to its launch specification."""
        backend = config["backend"]
        mode = config["execution_mode"]
        if backend == "gmr_preview":
            return build_preview_pipeline_command(self._paths, config)
        if mode == "simulation_only":
            return build_sim_pipeline_command(self._paths, config)
        return build_real_pipeline_command(self._paths, config)

    def stop_order_for(self, spec: ProcessSpec) -> tuple[str, ...]:
        if spec.name == "GMR preview supervisor":
            return PREVIEW_PIPELINE_STOP_ORDER
        if spec.name == "Unitree real supervisor":
            return REAL_PIPELINE_STOP_ORDER
        return SIM_PIPELINE_STOP_ORDER

    def status(self, log_limit: int = 200) -> Any:
        return self._manager.status(log_limit=log_limit)

    def request_stop(self, force: bool = False, *, source: str = "internal request") -> Any:
        return self._manager.request_stop(force=force, source=source)

    def shutdown(self) -> None:
        self._manager.shutdown()
